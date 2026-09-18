"""Basic noise models."""

import numpy as np

from config.builders import make_builder
from config.components import register_builder

from .geometry import latlon_to_xyz, pairwise_angular_distance


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
    return np.diag(nl / ic_tt) ** 2


def compute_path_similarity(ic_in: np.ndarray, ic_out: np.ndarray) -> np.ndarray:
    """Compute symmetric path-similarity matrix using great-circle spherical distances."""
    E = latlon_to_xyz(lat_deg=ic_in[:, 1], lon_deg=ic_in[:, 0])
    X = latlon_to_xyz(lat_deg=ic_out[:, 1], lon_deg=ic_out[:, 0])

    d_EE = pairwise_angular_distance(E, E)
    d_XX = pairwise_angular_distance(X, X)

    return np.sqrt(0.5 * (d_EE**2 + d_XX**2))


def correlated_paths(
    ic_in: np.ndarray,
    ic_out: np.ndarray,
    corr_length: float,
    corr_scale: float,
) -> np.ndarray:
    """Create a correlated covariance matrix, where the correlation is based on path similarity.

    corr_length is given in degrees.
    corr_scale is a variance
    """
    d = compute_path_similarity(ic_in, ic_out)
    L = np.radians(corr_length)
    K = np.exp(-(d**2) / (2 * L**2))
    return corr_scale * K


register_builder("noise.block_iid", make_builder(block_iid))
register_builder("noise.correlated_paths", make_builder(correlated_paths))
register_builder("noise.zero_mean", make_builder(lambda n_data: np.zeros(n_data)))
