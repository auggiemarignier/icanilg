"""Testing the noise module."""

import numpy as np

from utils.noise import (
    compute_path_similarity,
    correlated_paths,
    pairwise_angular_distance,
    latlon_to_unit_vectors,
)


def test_path_similarity_same_paths() -> None:
    """The same paths should have a similarity of 0."""

    north_pole = [0.0, 90.0, 1.0]
    south_pole = [0.0, -90.0, 1.0]
    ic_in = np.array([north_pole, north_pole])
    ic_out = np.array([south_pole, south_pole])

    sim = compute_path_similarity(ic_in, ic_out)
    expected = np.zeros((2, 2))

    np.testing.assert_equal(np.diag(sim), np.zeros(2))
    np.testing.assert_allclose(sim, expected)


def test_path_similarity_antiparallel_paths() -> None:
    """Flipped entry and exit points should have a similarity of 0."""

    north_pole = [0.0, 90.0, 1.0]
    south_pole = [0.0, -90.0, 1.0]
    ic_in = np.array([north_pole, south_pole])
    ic_out = np.array([south_pole, north_pole])

    sim = compute_path_similarity(ic_in, ic_out)
    expected = np.zeros((2, 2))

    np.testing.assert_equal(np.diag(sim), np.zeros(2))
    np.testing.assert_allclose(sim, expected)


def test_pairwise_angular_distance_is_symmetric() -> None:
    rng = np.random.default_rng(42)
    n = 100

    in_lon = rng.uniform(-180, 180, n)
    in_lat = rng.uniform(-90, 90, n)
    in_rad = np.ones(100)
    ic_in = np.array([in_lon, in_lat, in_rad]).T

    d = pairwise_angular_distance(
        latlon_to_unit_vectors(ic_in[:, 1], ic_in[:, 0]),
        latlon_to_unit_vectors(ic_in[:, 1], ic_in[:, 0]),
    )
    np.testing.assert_allclose(np.diag(d), np.zeros(100), atol=1e-15)
    np.testing.assert_allclose(d, d.T, atol=1e-15)


def test_full_path_similarity_matrix_properties() -> None:
    """Tests that the overall path similarity kernel is symmetric with zero diagonal."""
    rng = np.random.default_rng(42)
    n = 100

    ic_in = np.column_stack(
        [
            rng.uniform(-180, 180, n),  # lon
            rng.uniform(-90, 90, n),  # lat
            np.ones(n),  # rad
        ]
    )

    ic_out = np.column_stack(
        [rng.uniform(-180, 180, n), rng.uniform(-90, 90, n), np.ones(n)]
    )

    D_path = compute_path_similarity(ic_in, ic_out)
    np.testing.assert_allclose(np.diag(D_path), np.zeros(n), atol=1e-15)
    np.testing.assert_allclose(D_path, D_path.T, atol=1e-15)


def test_correlated_paths_covariance_matrix_is_valid() -> None:
    rng = np.random.default_rng(42)
    n = 100
    ic_in = np.column_stack(
        [rng.uniform(-180, 180, n), rng.uniform(-90, 90, n), np.ones(n)]
    )
    ic_out = np.column_stack(
        [rng.uniform(-180, 180, n), rng.uniform(-90, 90, n), np.ones(n)]
    )
    corr_length = 1.0
    ref_phase = ["ab"] * n
    ic_tt = np.ones(100)

    Cd = correlated_paths(ic_in, ic_out, corr_length, ref_phase, ic_tt)
    np.linalg.cholesky(Cd)  # check for positive-definite
