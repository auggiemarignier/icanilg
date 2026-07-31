"""Forward operations."""

import logging

import numpy as np
from raytracer import SphericalMesh
from tti.elastic.voigt import (
    gradient_C_wrt_A,
    gradient_C_wrt_C,
    gradient_C_wrt_F,
    gradient_C_wrt_L,
    gradient_C_wrt_N,
)
from tti.traveltimes.traveltimes import (
    calculate_path_direction_vector,
    calculate_relative_traveltime_voigt,
)

from .geometry import latlon_to_xyz

logger = logging.getLogger(__name__)


def construct_forward_map(
    ic_in: np.ndarray, ic_out: np.ndarray, mesh: SphericalMesh, normalisation: float
) -> np.ndarray:
    """Constructs the linear map from Love parameters to travel time.

    The input to the forward mapping is a stack of Love parameters, 3 (A,C,F) for each cell in the mesh.

    Combines various bits a pieces:
    1) Add 0 shear components (L, N) to input vector
    2) Mapping from a vector of elastic parameters to Voigt elastic tensor
        Making use of the derivative objects in `tti.elastic.voigt` a basis can be constructed
            basis = np.stack([dCdA, dCdC, dCdF, dCdL, dCdN], axis=0)
    3) With the `path_directions` compute the travel time in each cell as in `tti.traveltimes.traveltimes.calculate_relative_traveltime_voigt`
    4) With the `weights` calculated from `determine_weights` perform a weighted sum along each path as in `tti.traveltimes.traveltimes.TravelTimeCalculator._call_core.

    These four steps are to be combined into a single matrix that this function returns.

    The returned matrix has shape (n_paths, 3*n_segments)
    """

    path_directions = calculate_path_direction_vector(ic_in, ic_out)
    weights = _determine_weights(mesh, ic_in, path_directions)  # (n_segments, n_paths)
    n_segments, n_paths = weights.shape

    T = _compressional_to_love_transformation_matrix()  # (5, 3)
    n_params_per_seg = T.shape[1]

    dC = _love_vector_to_voigt_tensor_transformation()  # (5, 6, 6)
    dC_b = _expand_tensor_to_mesh(dC, mesh.n_cells)  # (n_segments, 5, 6, 6)
    dt_dC = calculate_relative_traveltime_voigt(
        path_directions, dC_b, normalisation=normalisation
    )  # (n_segments, 5, n_paths)
    dt_model = np.tensordot(dt_dC, T, axes=([1], [0]))  # (n_segments, n_paths, 3)
    dt_model = dt_model.transpose(0, 2, 1)  # (n_segments, 3, n_paths)
    weighted = weights[:, None, :] * dt_model  # (n_segments, 3, n_paths)
    M = weighted.reshape(
        n_segments * n_params_per_seg, n_paths
    ).T  # (n_paths, n_segments*3)
    return M


def _determine_weights(
    mesh: SphericalMesh, ic_in: np.ndarray, path_directions: np.ndarray
) -> np.ndarray:
    """Determine weights for each path based on the distance travelled in each region.

    Parameters
    ----------
    mesh : SphericalMesh
        The mesh defining the geometry and properties of the inner core.
    ic_in : ndarray, shape (num_paths, 3)
        Entry points of paths into the inner core (longitude (deg), latitude (deg), radius (km)).
    path_directions : ndarray, shape (num_paths, 3)
        Direction vectors for each path.

    Returns
    -------
    weights : ndarray, shape (num_segments, num_paths)
        Fractional distance of each path in each segment.
    """
    segment_distances = mesh.ray_distances_per_region(
        latlon_to_xyz(*ic_in[..., [1, 0, 2]].T), path_directions
    )
    total_distances = segment_distances.sum(axis=1)
    weights = segment_distances / total_distances[:, None]
    ws = weights.T
    logger.debug(
        "determine_weights: segment_distances.shape=%s total_distances.shape=%s",
        segment_distances.shape,
        total_distances.shape,
    )
    logger.debug(
        "determine_weights: computed weights for %d paths and %d segments",
        ic_in.shape[0],
        segment_distances.shape[1],
    )
    return ws


def _compressional_to_love_transformation_matrix() -> np.ndarray:
    """Map to the 5 Love basis (A,C,F,L,N) from the 3 model params (A,C,F)."""

    return np.array(
        [
            [1.0, 0.0, 0.0],  # A
            [0.0, 1.0, 0.0],  # C
            [0.0, 0.0, 1.0],  # F
            [0.0, 0.0, 0.0],  # L (not modelled)
            [0.0, 0.0, 0.0],  # N (not modelled)
        ],
        dtype=float,
    )


def _love_vector_to_voigt_tensor_transformation() -> np.ndarray:
    """Map to a Voigt tensor form an (A, C, F, L, N) vector.

    (A A-2N F 0 0 0)            (A)
    (A-2N A F 0 0 0)            (C)
    (F F C 0 0 0)               (F)
    (0 0 0 L 0 0)           = T (L)
    (0 0 0 0 L 0)               (N)
    (0 0 0 0 0 N)

    This function returns T of shape (5, 6, 6)
    """
    return np.stack(
        [
            gradient_C_wrt_A(),
            gradient_C_wrt_C(),
            gradient_C_wrt_F(),
            gradient_C_wrt_L(),
            gradient_C_wrt_N(),
        ],
        axis=0,
    )


def _expand_tensor_to_mesh(T: np.ndarray, n_segments: int) -> np.ndarray:
    # Broadcast T to (n_segments,5,6,6)
    return np.broadcast_to(T[None, ...], (n_segments, *T.shape))
