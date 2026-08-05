"""Various Gaussian prior functions on anisotropy parameters."""
import numpy as np
from raytracer import SphericalMesh

from .geometry import pairwise_angular_distance, thetaphi_to_unit_vectors


def iid(n_params: int, scale: float) -> np.ndarray:
    """Independent and Identically Distributed."""
    return np.eye(n_params) * scale


def _compute_lateral_covariance_kernel(
    theta: np.ndarray, phi: np.ndarray, phi0: float
) -> np.ndarray:
    r"""
    Compute the lateral covariance kernel.

    $$K_{lat} = \exp\left( -\frac{\psi(\theta_1, \phi_1; \theta_2, \phi_2)^2}{2 \psi_0^2} \right)$$
    """
    v = thetaphi_to_unit_vectors(theta, phi)
    psi = pairwise_angular_distance(v, v)

    return np.exp(-0.5 * (psi / phi0) ** 2)


def _compute_radial_covariance_kernel(r: np.ndarray, L_r: float) -> np.ndarray:
    # Pairwise radial distance matrix
    r1, r2 = np.meshgrid(r, r, indexing="ij")

    return np.exp(-0.5 * ((r1 - r2) / L_r) ** 2)


def spherically_correlated(
    mesh: SphericalMesh, sigma2: float, lat_corr_length: float, rad_corr_length: float
) -> np.ndarray:
    r"""
    Construct a correlated prior with a separable correlation kernel.

    $$C(\mathbf{r}_1, \mathbf{r}_2) = \sigma^2 \cdot \exp\left( -\frac{\psi(\theta_1, \phi_1; \theta_2, \phi_2)^2}{2 \psi_0^2} \right) \cdot \exp\left( -\frac{(r_1 - r_2)^2}{2 L_r^2} \right)$$

    Args:
        mesh: mesh on which the inversion is taking place
        sigma2: prior variance
        lat_corr_length: lateral correlation length in degrees
        rad_corr_length: radial correlation length in km
    """
    theta, phi = mesh.sampling.sampling_points().T
    phi = np.mod(phi + np.pi, 2 * np.pi) - np.pi  # ensure -pi < phi < pi
    rad = 0.5 * (mesh.radial_edges[:-1] + mesh.radial_edges[1:])  # get radial centres

    K_lat = _compute_lateral_covariance_kernel(
        theta, phi, lat_corr_length * np.pi / 180.0
    )
    K_r = _compute_radial_covariance_kernel(rad, rad_corr_length)

    # per-cell covariance: radial-major ⨂ lateral (so ordering matches SphericalMesh)
    return sigma2 * np.kron(K_r, K_lat)


def spherically_correlated_independent_anisotropy(
    mesh: SphericalMesh, sigma2: float, lat_corr_length: float, rad_corr_length: float
) -> np.ndarray:
    """
    Construct a spherically correlated prior where A, C and F are independent.

    i.e. p(A,C,F) = p(A)p(C)p(F)
    p(A) = p(A0,...,An)  which is correlated, similarly for p(C) and p(F)
    """
    C_cells = spherically_correlated(mesh, sigma2, lat_corr_length, rad_corr_length)

    # expand to interleaved parameters [A0,C0,F0,...]
    return np.kron(C_cells, np.eye(3))
