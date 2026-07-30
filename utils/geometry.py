"""Functions relating to spherical geometry."""

import numpy as np


def latlon_to_unit_vectors(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    """Convert latitude/longitude in DEGREES to unit 3D Cartesian vectors (N, 3)."""
    lat_deg = np.atleast_1d(lat_deg)
    lon_deg = np.atleast_1d(lon_deg)

    lon_wrapped = (lon_deg + 180.0) % 360.0 - 180.0
    lat_clipped = np.clip(lat_deg, -90.0, 90.0)

    lat_rad = np.radians(lat_clipped)
    lon_rad = np.radians(lon_wrapped)

    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)

    return np.column_stack([x, y, z])


def thetaphi_to_unit_vectors(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Convert colatitude (theta) and longitude (phi) in radians to unit 3D Cartesian vectors (N, 3)."""
    lat_deg = np.degrees(np.pi / 2 - theta)
    lon_deg = np.degrees(phi)
    return latlon_to_unit_vectors(lat_deg, lon_deg)


def pairwise_angular_distance(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Great-circle distance (radians) between unit vector sets v1 (N,3) and v2 (M,3)."""
    cos_theta = v1 @ v2.T

    # Enforce exact machine-level symmetry if computing self-distances
    if v1 is v2 or (v1.shape == v2.shape and np.array_equal(v1, v2)):
        cos_theta = 0.5 * (cos_theta + cos_theta.T)
        # Force the exact diagonal to 1.0 to prevent arccos(1.0 + eps) or small non-zeros
        np.fill_diagonal(cos_theta, 1.0)

    cos_theta_clipped = np.clip(cos_theta, -1.0, 1.0)
    return np.arccos(cos_theta_clipped)
