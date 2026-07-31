"""Functions relating to spherical geometry."""

import numpy as np


def latlon_to_xyz(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    radius: np.ndarray | float = 1.0,
) -> np.ndarray:
    """Convert latitude/longitude in degrees to Cartesian coordinates.

    Parameters
    ----------
    lat_deg : np.ndarray
        Latitude in degrees.
    lon_deg : np.ndarray
        Longitude in degrees.
    radius : np.ndarray | float, optional
        Radius of the sphere. Defaults to 1.0, which returns unit vectors.

    Returns
    -------
    xyz : np.ndarray
        Cartesian coordinates with shape (..., 3).
    """
    lat_deg = np.asarray(lat_deg)
    lon_deg = np.asarray(lon_deg)
    radius = np.asarray(radius)

    lon_wrapped = (lon_deg + 180.0) % 360.0 - 180.0
    lat_clipped = np.clip(lat_deg, -90.0, 90.0)

    lat_rad = np.radians(lat_clipped)
    lon_rad = np.radians(lon_wrapped)

    x = radius * np.cos(lat_rad) * np.cos(lon_rad)
    y = radius * np.cos(lat_rad) * np.sin(lon_rad)
    z = radius * np.sin(lat_rad)

    return np.stack([x, y, z], axis=-1)


def thetaphi_to_unit_vectors(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Convert colatitude (theta) and longitude (phi) in radians to unit 3D Cartesian vectors (N, 3)."""
    lat_deg = np.degrees(np.pi / 2 - theta)
    lon_deg = np.degrees(phi)
    return latlon_to_xyz(lat_deg, lon_deg)


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
