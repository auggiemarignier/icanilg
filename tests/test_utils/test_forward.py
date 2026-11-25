import numpy as np
import pytest
from raytracer import SphericalMesh
from raytracer.sampling import MWSphericalSampling
from tti.traveltimes.traveltimes import calculate_path_direction_vector

from utils.forward import (
    _determine_weights,
    construct_iso_forward_map,
    construct_rti_forward_map,
    construct_ssi_ak_filter,
    construct_ti_forward_map,
)


def _random_ic_pairs(n_paths, seed=0):
    rng = np.random.RandomState(seed)
    lon = rng.uniform(-180.0, 180.0, size=n_paths)
    lat = rng.uniform(-90.0, 90.0, size=n_paths)
    r = np.ones(n_paths)
    ic_in = np.column_stack([lon, lat, r])

    # Make sure out points are different: rotate lon by random amount
    lon2 = (lon + rng.uniform(10.0, 180.0, size=n_paths)) % 360 - 180
    lat2 = rng.uniform(-90.0, 90.0, size=n_paths)
    ic_out = np.column_stack([lon2, lat2, r])
    return ic_in, ic_out


def test_weights_match_mesh_n_cells_on_typical_mesh() -> None:
    mesh = SphericalMesh(1.0, 3, MWSphericalSampling(4))
    ic_in, ic_out = _random_ic_pairs(20, seed=1)
    pd = calculate_path_direction_vector(ic_in, ic_out)

    w = _determine_weights(mesh, ic_in, pd)
    # In typical meshes the resolved number of segments should match mesh.n_cells
    assert w.shape[0] == mesh.n_cells
    assert w.shape[1] == ic_in.shape[0]


@pytest.mark.parametrize(
    "constructor,n_per_cell",
    [
        (construct_ti_forward_map, 3),
        (construct_rti_forward_map, 3),
        (construct_iso_forward_map, 1),
    ],
)
def test_forward_maps_dense_mesh_shapes_and_finite_values(
    constructor, n_per_cell
) -> None:
    mesh = SphericalMesh(1.0, 4, MWSphericalSampling(6))
    n_paths = 30
    ic_in, ic_out = _random_ic_pairs(n_paths, seed=2)

    G = constructor(ic_in, ic_out, mesh, normalisation=1.0)

    # Shape
    assert G.shape == (n_paths, n_per_cell * mesh.n_cells)

    # Applying random parameter vectors yields finite outputs
    m = np.random.RandomState(3).randn(n_per_cell * mesh.n_cells)
    out = G @ m

    assert out.shape == (n_paths,)
    assert np.all(np.isfinite(out))


def test__determine_weights() -> None:
    mesh = SphericalMesh(
        1.0, 1, MWSphericalSampling(1)
    )  # n_cells = 1 so all weights should be 1.0
    ic_in = np.array([[0.0, 90.0, 1.0]])
    ic_out = np.array([[0.0, -90.0, 1.0]])
    pd = calculate_path_direction_vector(ic_in, ic_out)
    n_paths = pd.shape[0]

    w = _determine_weights(mesh, ic_in, pd)
    assert w.shape == (mesh.n_cells, n_paths)
    np.testing.assert_allclose(w, np.ones_like(w))


def test_construct_ti_forward_map_applied() -> None:
    """Simple one-cell mesh with simple paths so we know what elements of the parameter vector it will pull out.

    First path is N-S so we expect first row to be [0, 1, 0] i.e. only C contributes.
    Second path is E-W so we expect second row to be [1, 0, 0] i.e. only A contributes
    """
    mesh = SphericalMesh(1.0, 1, MWSphericalSampling(1))
    ic_in = np.array([[0.0, 90.0, 1.0], [0.0, 0.0, 1.0]])
    ic_out = np.array([[0.0, -90.0, 1.0], [180.0, 0.0, 1.0]])
    n_paths = ic_out.shape[0]
    n_params = 3 * mesh.n_cells

    G = construct_ti_forward_map(ic_in, ic_out, mesh, normalisation=1.0)
    assert G.shape == (n_paths, n_params)
    np.testing.assert_allclose(G, np.array([[0, 1, 0], [1, 0, 0]]), atol=1e-15)

    m = np.arange(3)  # A = 0, C = 1, F = 2 => dt = 1
    np.testing.assert_allclose(G @ m, np.array([1, 0]))  # C, A


def test_construct_ssi_ak_filter() -> None:
    turning_points = np.array(
        [
            [0, 0, 1221.5],
            [0, 0, 1221.5],
            [-75.0, 7.0, 1000.0],
            [-75.0, 7.0, 1000.0],
            [-75.0, 7.0, 1000.0],
        ]
    )
    zeta = np.array([0.0, 30.0, 30, 0.0, 30.0])

    expected = np.array([[0], [0], [1], [0], [1]])  # 5x1
    actual = construct_ssi_ak_filter(turning_points, zeta)
    np.testing.assert_array_equal(actual, expected)


def test_construct_rti_matches_construct_forward_when_zero_tilt() -> None:
    mesh = SphericalMesh(1.0, 1, MWSphericalSampling(1))
    mesh.sampling.sampling_points = lambda: np.zeros((mesh.sampling.n_cells, 2))

    ic_in = np.array([[0.0, 90.0, 1.0], [0.0, 0.0, 1.0]])
    ic_out = np.array([[0.0, -90.0, 1.0], [180.0, 0.0, 1.0]])

    G = construct_ti_forward_map(ic_in, ic_out, mesh, normalisation=1.0)
    G_rti = construct_rti_forward_map(ic_in, ic_out, mesh, normalisation=1.0)

    np.testing.assert_allclose(G_rti, G)


def test_construct_rti_matches_construct_ti_when_polar_rotation() -> None:
    mesh = SphericalMesh(1.0, 1, MWSphericalSampling(1))
    mesh.sampling.sampling_points = lambda: (
        np.ones((mesh.sampling.n_cells, 2))
        * np.array([np.pi, 0.0])  # everything pointing south
    )

    ic_in = np.array([[0.0, 90.0, 1.0], [0.0, 0.0, 1.0]])
    ic_out = np.array([[0.0, -90.0, 1.0], [180.0, 0.0, 1.0]])

    G = construct_ti_forward_map(ic_in, ic_out, mesh, normalisation=1.0)
    G_rti = construct_rti_forward_map(ic_in, ic_out, mesh, normalisation=1.0)

    np.testing.assert_allclose(G_rti, G, atol=1e-14)


def test_construct_iso_matches_construct_ti() -> None:
    mesh = SphericalMesh(1.0, 1, MWSphericalSampling(1))
    mesh.sampling.sampling_points = lambda: (
        np.ones((mesh.sampling.n_cells, 2))
        * np.array([np.pi, 0.0])  # everything pointing south
    )

    ic_in = np.array([[0.0, 90.0, 1.0], [0.0, 0.0, 1.0]])
    ic_out = np.array([[0.0, -90.0, 1.0], [180.0, 0.0, 1.0]])

    G = construct_ti_forward_map(ic_in, ic_out, mesh, normalisation=1.0)
    G_iso = construct_iso_forward_map(ic_in, ic_out, mesh, normalisation=1.0)

    lmda = 2
    A = lmda
    C = lmda
    F = lmda

    iso_params = np.array([lmda] * mesh.n_cells)
    love_params = np.array([A, C, F] * mesh.n_cells)
    np.testing.assert_allclose(G_iso @ iso_params, G @ love_params, atol=1e-14)
