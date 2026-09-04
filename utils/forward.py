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
from tti.elastic.voigt_mapping import matrix_to_voigt
from tti.rotation import rotation_matrix_zy
from tti.traveltimes.traveltimes import (
    calculate_path_direction_vector,
    calculate_relative_traveltime_voigt,
)

from config.builders import make_builder
from config.components import register_builder

from .geometry import latlon_to_xyz, pairwise_angular_distance

logger = logging.getLogger(__name__)


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


def _isotropic_to_love_transformation_matrix() -> np.ndarray:
    """Map to the 5 Love basis (A,C,F,L,N) from the 1 isotropic parameter (lambda)."""

    # just for completeness I've written this out as [lambda] -> [lambda, mu] -> [A, C, F, L, N]
    T1 = np.array(
        [
            [1.0],  # lambda
            [0.0],  # mu (not modelled)
        ]
    )
    T2 = np.array(
        [
            [1.0, 2.0],  # A = lambda + 2mu
            [1.0, 2.0],  # C = lambda + 2mu
            [1.0, 0.0],  # F = lambda
            [0.0, 1.0],  # L = mu
            [0.0, 1.0],  # N = mu
        ],
        dtype=float,
    )
    return T2 @ T1


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


def construct_ssi_ak_filter(
    turning_point: np.ndarray, zeta: np.ndarray, radius: float = 15.0
) -> np.ndarray:
    """Use if IC turning point and angle with ERA are available.

    Following 10.1016/j.pepi.2020.106427, the SSI-AK path have 26<zeta<32.
    I've gone more conservative.
    Combined with turning points beneath the northern coast of South America, this filter is a decent option to find the correct paths.

    Returns:
        boolean array (n_paths x n_ssi_ak)
    """
    if turning_point.shape[0] != zeta.shape[0]:
        raise RuntimeError("Input data have incompatible shapes.")
    n_paths = zeta.shape[0]

    zeta_exclusion_range = np.array((20.0, 35.0))
    in_zeta_exclusion = (zeta_exclusion_range[0] < zeta) & (
        zeta < zeta_exclusion_range[1]
    )

    tp_xyz = latlon_to_xyz(*turning_point[:, [1, 0, 2]].T)
    tp_exclusion_zone_centre = np.array((-75.0, 7.0))  # lon, lat
    # pad with the radius so we get the 2D distance
    tp_exc = tp_exclusion_zone_centre.repeat(n_paths).reshape((n_paths, 2), order="F")
    tp_exc_r = np.column_stack([tp_exc, turning_point[:, -1]])
    tp_exc_xyz = latlon_to_xyz(*tp_exc_r[:, [1, 0, 2]].T)

    r = np.radians(radius)
    v1 = tp_xyz / np.linalg.norm(tp_xyz, axis=1)[:, None]
    v2 = tp_exc_xyz / np.linalg.norm(tp_exc_xyz, axis=1)[:, None]

    # a bit of excess computation here but oh well
    tp_distance_from_exclusion_centre = np.diag(pairwise_angular_distance(v1, v2))
    in_tp_exclusion = tp_distance_from_exclusion_centre < r

    ssi_ak_ind = np.argwhere(in_zeta_exclusion & in_tp_exclusion).squeeze()

    A = np.zeros((n_paths, 1), dtype=int)
    A[ssi_ak_ind] = 1

    return A


def count_ssi_ak_paths(
    turning_point: np.ndarray, zeta: np.ndarray, radius: float = 15.0
) -> int:
    """Helper function to return the number of paths in SSI-AK corridor."""
    A = construct_ssi_ak_filter(turning_point, zeta, radius)
    return A.sum()


def construct_ti_forward_map(
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
    n_paths = ic_in.shape[0]
    n_segments = mesh.n_cells

    T = _compressional_to_love_transformation_matrix()  # (5, n_params_per_seg)
    n_params_per_seg = T.shape[1]

    dC_base = _love_vector_to_voigt_tensor_transformation()  # (5, 6, 6)
    path_directions = calculate_path_direction_vector(ic_in, ic_out)
    dt_dC = calculate_relative_traveltime_voigt(
        path_directions, dC_base, normalisation=normalisation
    )  # (5, n_paths)

    dt_model = T.T @ dt_dC  # (n_params_per_seg, n_paths)

    weights = _determine_weights(mesh, ic_in, path_directions)  # (n_segments, n_paths)
    weighted = (
        weights[:, None, :] * dt_model[None, :, :]
    )  # (n_segments, n_params_per_seg, n_paths)
    M = weighted.reshape(n_segments * n_params_per_seg, n_paths).T
    return M


def construct_iso_forward_map(
    ic_in: np.ndarray, ic_out: np.ndarray, mesh: SphericalMesh, normalisation: float
) -> np.ndarray:
    """Constructs the linear map from isotopic parameters to travel time.

    The input to the forward mapping is a stack of isotropic parameters, 1 (lambda) for each cell in the mesh.

    Combines various bits a pieces:
    1) Convert to Love parameters
    2) Mapping from a vector of elastic parameters to Voigt elastic tensor
        Making use of the derivative objects in `tti.elastic.voigt` a basis can be constructed
            basis = np.stack([dCdA, dCdC, dCdF, dCdL, dCdN], axis=0)
    3) With the `path_directions` compute the travel time in each cell as in `tti.traveltimes.traveltimes.calculate_relative_traveltime_voigt`
    4) With the `weights` calculated from `determine_weights` perform a weighted sum along each path as in `tti.traveltimes.traveltimes.TravelTimeCalculator._call_core.

    These four steps are to be combined into a single matrix that this function returns.

    The returned matrix has shape (n_paths, 1*n_segments)
    """
    n_paths = ic_in.shape[0]
    n_segments = mesh.n_cells

    T = _isotropic_to_love_transformation_matrix()  # (5, n_params_per_seg)
    n_params_per_seg = T.shape[1]

    dC_base = _love_vector_to_voigt_tensor_transformation()  # (5, 6, 6)
    path_directions = calculate_path_direction_vector(ic_in, ic_out)
    dt_dC = calculate_relative_traveltime_voigt(
        path_directions, dC_base, normalisation=normalisation
    )  # (5, n_paths)

    dt_model = T.T @ dt_dC  # (n_params_per_seg, n_paths)

    weights = _determine_weights(mesh, ic_in, path_directions)  # (n_segments, n_paths)
    weighted = (
        weights[:, None, :] * dt_model[None, :, :]
    )  # (n_segments, n_params_per_seg, n_paths)
    M = weighted.reshape(n_segments * n_params_per_seg, n_paths).T
    return M


def construct_rti_forward_map(
    ic_in: np.ndarray, ic_out: np.ndarray, mesh: SphericalMesh, normalisation: float
) -> np.ndarray:
    """Constructs the radially transversely isotropic linear map from Love parameters to travel time.

    The input to the forward mapping is a stack of Love parameters, 3 (A,C,F) for each cell in the mesh.

    Combines various bits a pieces:
    1) Add 0 shear components (L, N) to input vector
    2) Mapping from a vector of elastic parameters to Voigt elastic tensor
        Making use of the derivative objects in `tti.elastic.voigt` a basis can be constructed
            basis = np.stack([dCdA, dCdC, dCdF, dCdL, dCdN], axis=0)
    3) Rotating the Voigt elastic tensor such that x3 is radial i.e. rotate to theta, phi
    4) With the `path_directions` compute the travel time in each cell as in `tti.traveltimes.traveltimes.calculate_relative_traveltime_voigt`
    5) With the `weights` calculated from `determine_weights` perform a weighted sum along each path as in `tti.traveltimes.traveltimes.TravelTimeCalculator._call_core.

    These five steps are to be combined into a single matrix that this function returns.

    The returned matrix has shape (n_paths, 3*n_segments)
    """
    n_paths = ic_in.shape[0]
    n_segments = mesh.n_cells

    T = _compressional_to_love_transformation_matrix()  # (5, n_params_per_seg)
    n_params_per_seg = T.shape[1]

    dC_base = _love_vector_to_voigt_tensor_transformation()  # (5, 6, 6)
    path_directions = calculate_path_direction_vector(ic_in, ic_out)

    per_cell_angles = np.tile(mesh.sampling.sampling_points(), (mesh.n_radial, 1))
    theta = per_cell_angles[:, 0]
    phi = per_cell_angles[:, 1]

    R_voigt = matrix_to_voigt(rotation_matrix_zy(phi, theta))[
        :, None, :, :
    ]  # (n_segments, 1, 6, 6)
    dC_rot = (R_voigt @ dC_base[None, ...]) @ R_voigt.swapaxes(
        -2, -1
    )  # (n_segments, 5, 6, 6)

    dt_dC = calculate_relative_traveltime_voigt(
        path_directions, dC_rot, normalisation=normalisation
    )  # (n_segments, 5, n_paths)

    dt_model = np.tensordot(
        dt_dC, T, axes=([1], [0])
    )  # (n_segments, n_paths, n_params_per_seg)
    dt_model = dt_model.transpose(0, 2, 1)  # (n_segments, n_params_per_seg, n_paths)

    weights = _determine_weights(mesh, ic_in, path_directions)  # (n_segments, n_paths)
    weighted = dt_model * weights[:, None, :]

    M = weighted.reshape(n_segments * n_params_per_seg, n_paths).T
    return M


register_builder("forward.build_forward", make_builder(construct_ti_forward_map))
register_builder("forward.build_rti_forward", make_builder(construct_rti_forward_map))
register_builder("forward.build_iso_forward", make_builder(construct_iso_forward_map))
register_builder("forward.ssi_ak_filter", make_builder(construct_ssi_ak_filter))
register_builder("forward.ssi_ak_bias_mean", make_builder(lambda: np.asarray([0])))
register_builder(
    "forward.ssi_ak_bias_cov", make_builder(lambda scale: np.asarray([[scale]]))
)
register_builder(
    "forward.eye", make_builder(lambda n_data, scale: scale * np.eye(n_data))
)
