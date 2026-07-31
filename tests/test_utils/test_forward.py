import numpy as np
from raytracer import SphericalMesh
from raytracer.sampling import MWSphericalSampling
from tti.traveltimes.traveltimes import calculate_path_direction_vector

from utils import construct_forward_map
from utils.forward import _determine_weights


def test__determine_weights() -> None:
    mesh = SphericalMesh(1.0, 1, MWSphericalSampling(1))  # n_cells = 1 so all weights should be 1.0
    ic_in = np.array([[0.0, 90.0, 1.0]])
    ic_out = np.array([[0.0, -90.0, 1.0]])
    pd = calculate_path_direction_vector(ic_in, ic_out)
    n_paths = pd.shape[0]

    w = _determine_weights(mesh, ic_in, pd)
    assert w.shape == (mesh.n_cells, n_paths)
    np.testing.assert_allclose(w, np.ones_like(w))


def test_construct_forward_map_applied() -> None:
    """Simple one-cell mesh with simple paths so we know what elements of the parameter vector it will pull out.

    First path is N-S so we expect first row to be [0, 1, 0] i.e. only C contributes.
    Second path is E-W so we expect second row to be [1, 0, 0] i.e. only A contributes
    """
    mesh = SphericalMesh(1.0, 1, MWSphericalSampling(1))
    ic_in = np.array([[0.0, 90.0, 1.0], [0.0, 0.0, 1.0]])
    ic_out = np.array([[0.0, -90.0, 1.0], [180.0, 0.0, 1.0]])
    n_paths = ic_out.shape[0]
    n_params = 3 * mesh.n_cells

    G = construct_forward_map(ic_in, ic_out, mesh, normalisation=1.0)
    assert G.shape == (n_paths, n_params)
    np.testing.assert_allclose(G, np.array([[0, 1, 0], [1, 0, 0]]), atol=1e-15)

    m = np.arange(3) # A = 0, C = 1, F = 2 => dt = 1
    np.testing.assert_allclose(G@m, np.array([1, 0]))  # C, A
