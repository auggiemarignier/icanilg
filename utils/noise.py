"""Basic noise models."""

import numpy as np

from .geometry import latlon_to_unit_vectors, pairwise_angular_distance


def block_iid(ref_phase: list[str], ic_tt: np.ndarray) -> np.ndarray:
    """
    Taking the hierarchical noise levels reported by Brett et al, 2022.

    The noise levels for each reference phase are given in seconds, so we need to convert them to fractional traveltime perturbations by dividing by the inner core travel time.

    In principle this gives a different sigma for each observation.
    """

    noise_levels: dict[str, float] = {
        "ab": 0.95,
        "bc": 0.63,
        "cd": 0.29,
        "df": 0.95,
    }
    nl = np.array([noise_levels[phase] for phase in ref_phase])
    return np.diag(nl / ic_tt)


def compute_path_similarity(ic_in: np.ndarray, ic_out: np.ndarray) -> np.ndarray:
    """Compute symmetric path-similarity matrix using great-circle spherical distances."""
    E = latlon_to_unit_vectors(lat_deg=ic_in[:, 1], lon_deg=ic_in[:, 0])
    X = latlon_to_unit_vectors(lat_deg=ic_out[:, 1], lon_deg=ic_out[:, 0])

    d_EE = pairwise_angular_distance(E, E)  # Entry_i to Entry_j (diag = 0.0)
    d_XX = pairwise_angular_distance(X, X)  # Exit_i to Exit_j   (diag = 0.0)
    d_EX = pairwise_angular_distance(
        E, X
    )  # Entry_i to Exit_j  (diag ~ pi for deep rays)
    d_XE = d_EX.T

    d_parallel = np.sqrt(0.5 * (d_EE**2 + d_XX**2))
    d_anti = np.sqrt(0.5 * (d_EX**2 + d_XE**2))

    return np.minimum(d_parallel, d_anti)


def correlated_paths(
    ic_in: np.ndarray,
    ic_out: np.ndarray,
    corr_length: float,
    ref_phase: list[str],
    ic_tt: np.ndarray,
) -> np.ndarray:
    """Create a correlated covariance matrix, where the correlation is based on path similarity.

    The variance (diagonal) is given by `block_iid`.

    corr_length is given in degrees.
    """
    d = compute_path_similarity(ic_in, ic_out)
    L = np.radians(corr_length)
    diag = block_iid(ref_phase, ic_tt)
    C = diag @ np.exp(-d / L) @ diag
    nugget = np.diag(C) * 0.01  # helps conditioning
    return C + np.diag(nugget)
