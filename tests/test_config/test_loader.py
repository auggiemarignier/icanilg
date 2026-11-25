import json
from pathlib import Path

import numpy as np
from raytracer import SphericalMesh

from config.loader import load_config
from config.models import Config


def test_load_toml(tmp_path: Path):
    toml_cfg = """
[mesh]
radius = 1000.0
radial_resolution = 2
lateral_resolution = 3

[prior]
scale = 5.0
"""
    toml_file = tmp_path / "cfg.toml"
    # add required sections: data, components, output
    toml_cfg = (
        toml_cfg
        + """
[data]
file = "data/brett2024_ic_traveltimes.parquet"

[components]
inferred = []
nuisance = []

[output]
prefix = "R4L5"
"""
    )
    toml_file.write_bytes(toml_cfg.encode("utf8"))

    cfg = load_config(toml_file)
    assert isinstance(cfg, Config)
    assert cfg.mesh.radius == 1000.0


def test_load_json(tmp_path: Path):
    json_file = tmp_path / "cfg.json"
    json_file.write_text(
        json.dumps(
            {
                "data": {},
                "mesh": {"radius": 2000.0},
                "components": {"inferred": [], "nuisance": []},
                "output": {},
            }
        ),
        encoding="utf8",
    )
    cfg2 = load_config(json_file)
    assert cfg2.mesh.radius == 2000.0


def test_fallback_to_json(tmp_path: Path):
    # create a file without .toml or .json but containing JSON
    f = tmp_path / "cfg.conf"
    f.write_text(
        json.dumps(
            {
                "data": {},
                "mesh": {"radius": 1500.0},
                "components": {"inferred": [], "nuisance": []},
                "output": {},
            }
        ),
        encoding="utf8",
    )
    cfg = load_config(f)
    assert cfg.mesh.radius == 1500.0


def test_build_components_returns_inferred_and_nuisance_components():
    from linear_gaussian import GaussianComponent

    A_inf = np.zeros((5, 12))
    A_nuis = np.eye(5)

    cfg = Config.from_dict(
        {
            "data": {},
            "mesh": {},
            "components": {
                "inferred": [{"A": A_inf, "mu": np.zeros(12), "C": np.eye(12)}] * 2,
                "nuisance": [{"A": A_nuis, "mu": np.zeros(5), "C": np.eye(5)}],
            },
            "output": {},
        }
    )
    inferred, nuisance = cfg.components.to_gaussian_components({})

    assert len(inferred) == 2
    assert all(isinstance(comp, GaussianComponent) for comp in inferred)

    assert len(nuisance) == 1
    assert all(isinstance(comp, GaussianComponent) for comp in nuisance)


def test_build_mesh_from_cfg():
    cfg_obj = Config.from_dict(
        {
            "data": {},
            "mesh": {"radius": 1234.5, "radial_resolution": 2, "lateral_resolution": 3},
            "components": {"inferred": [], "nuisance": []},
            "output": {},
        }
    )
    mesh = cfg_obj.mesh.to_mesh()
    assert isinstance(mesh, SphericalMesh)
    # mesh should expose a positive number of cells
    n_cells = getattr(mesh, "n_cells", None)
    assert isinstance(n_cells, int) and n_cells > 0
