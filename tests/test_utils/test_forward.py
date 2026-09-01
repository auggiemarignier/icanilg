import numpy as np
from raytracer import SphericalMesh
from raytracer.sampling import MWSphericalSampling
from tti.elastic.voigt_mapping import matrix_to_voigt
from tti.rotation import rotation_matrix_zy
from tti.traveltimes.traveltimes import calculate_path_direction_vector

from utils.forward import (
    _determine_weights,
    _expand_tensor_to_mesh,
    _rotate_expanded_tensor_to_mesh,
    construct_rti_forward_map,
    construct_ssi_ak_filter,
    construct_ti_forward_map,
)


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

    expected = np.array([[0, 0], [0, 0], [1, 0], [0, 0], [0, 1]])  # 4x1
    actual = construct_ssi_ak_filter(turning_points, zeta)
    np.testing.assert_array_equal(actual, expected)


def test_expand_and_rotate_tensor_to_mesh_zero_tilt_returns_unrotated() -> None:
    mesh = SphericalMesh(1.0, 1, MWSphericalSampling(1))

    # override sampling points to be zero tilt for all lateral cells
    mesh.sampling.sampling_points = lambda: np.zeros((mesh.sampling.n_cells, 2))

    dC = np.arange(5 * 6 * 6, dtype=float).reshape((5, 6, 6))

    expanded = _expand_tensor_to_mesh(dC, mesh.n_cells)
    rotated = _rotate_expanded_tensor_to_mesh(expanded, mesh)

    expected = np.broadcast_to(dC[None, ...], (mesh.n_cells, *dC.shape))
    np.testing.assert_allclose(rotated, expected)


def test_expand_and_rotate_tensor_to_mesh_shape_and_rotation() -> None:
    mesh = SphericalMesh(1.0, 2, MWSphericalSampling(2))

    dC = np.arange(5 * 6 * 6, dtype=float).reshape((5, 6, 6))

    expanded = _expand_tensor_to_mesh(dC, mesh.n_cells)
    rotated = _rotate_expanded_tensor_to_mesh(expanded, mesh)

    assert rotated.shape == (mesh.n_cells, 5, 6, 6)

    # verify first cell equals explicit rotation computed from sampling point
    theta, phi = mesh.sampling.sampling_points()[0]
    R_voigt = matrix_to_voigt(rotation_matrix_zy(phi, theta))
    expected0 = R_voigt @ dC[0] @ R_voigt.swapaxes(-2, -1)

    np.testing.assert_allclose(rotated[0, 0], expected0)


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
