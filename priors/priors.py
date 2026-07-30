"""Various Gaussian prior functions on anisotropy parameters."""

import numpy as np
from raytracer import SphericalMesh


def iid(n_params: int, scale: float) -> np.ndarray:
    """Independent and Identically Distributed."""
    return np.eye(n_params) * scale


def angular_distance(
    theta1: np.ndarray, phi1: np.ndarray, theta2: np.ndarray, phi2: np.ndarray
) -> np.ndarray:
    r"""Numerically stable central angle (radians) between (theta1, phi1) and (theta2, phi2).

    theta = colatitude from North Pole [0, pi], phi = longitude [-pi, pi].
    Uses Vincenty/atan2 formulation to prevent precision loss for small arc distances.
    """
    dphi = phi2 - phi1

    sin_t1, cos_t1 = np.sin(theta1), np.cos(theta1)
    sin_t2, cos_t2 = np.sin(theta2), np.cos(theta2)
    cos_dphi = np.cos(dphi)

    # Numerator & Denominator for atan2
    term1 = sin_t2 * np.sin(dphi)
    term2 = sin_t2 * cos_t1 - cos_t2 * sin_t1 * cos_dphi

    num = np.sqrt(term1**2 + term2**2)
    den = cos_t1 * cos_t2 + sin_t1 * sin_t2 * cos_dphi

    return np.arctan2(num, den)


def _compute_lateral_covariance_kernel(
    theta: np.ndarray, phi: np.ndarray, phi0: float
) -> np.ndarray:
    r"""
    Compute the lateral covariance kernel.

    $$K_{lat} = \exp\left( -\frac{\psi(\theta_1, \phi_1; \theta_2, \phi_2)^2}{2 \psi_0^2} \right)$$
    """
    # Compute distances between all points on a spherical shell
    theta1, theta2 = np.meshgrid(theta, theta, indexing="ij")
    phi1, phi2 = np.meshgrid(phi, phi, indexing="ij")
    psi = angular_distance(theta1, phi1, theta2, phi2)

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
    phi = np.mod(phi + 0.5 * np.pi, np.pi) - 0.5 * np.pi  # ensure -pi/2 < phi < pi/2
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
