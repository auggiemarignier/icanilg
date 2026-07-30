from pathlib import Path

import numpy as np

from config.models import (
    ComponentsConfig,
    Config,
    DataConfig,
    MeshConfig,
    OutputConfig,
    config_to_json_dict,
)


def test_mesh_defaults():
    # Provide minimal valid config for strict validation
    cfg = Config.from_dict(
        {
            "data": {},
            "mesh": {},
            "components": {"inferred": [], "nuisance": []},
            "output": {},
        }
    )
    assert cfg.mesh.radius == 1221.5
    assert cfg.mesh.radial_resolution == 4
    assert cfg.mesh.lateral_resolution == 5
    assert cfg.mesh.sampling == "fib"


def test_components_resolution_basic():
    # simple numeric specs should yield identity/zeros as per components.resolve_field

    cfg = Config.from_dict(
        {
            "data": {},
            "mesh": {},
            "components": {
                "inferred": [{"A": 1, "C": 1, "mu": 1, "name": "c1"}],
                "nuisance": [{"A": 2, "C": 2, "mu": 2, "name": "n1"}],
            },
            "output": {},
        }
    )
    inferred, nuisance = cfg.components.to_gaussian_components()

    assert len(inferred) == 1
    comp = inferred[0]
    assert comp.A.shape == (1, 1)
    assert comp.mu.shape == (1,)
    assert comp.C.shape == (1, 1)

    assert len(nuisance) == 1
    comp2 = nuisance[0]
    assert comp2.A.shape == (2, 2)
    assert comp2.mu.shape == (2,)
    assert comp2.C.shape == (2, 2)


def test_data_config_default_and_override(tmp_path: Path):
    # default (resolves relative to project root)
    dc = DataConfig.from_dict({})
    assert isinstance(dc.file, Path)

    # explicit override
    dc2 = DataConfig.from_dict({"file": "some/path.parquet"})
    assert dc2.file == Path("some/path.parquet")


def test_mesh_validation():
    # valid defaults
    m = MeshConfig.from_dict({})
    assert m.radius > 0

    # invalid radius
    try:
        MeshConfig.from_dict({"radius": -1})
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for negative radius")

    # invalid resolutions
    try:
        MeshConfig.from_dict({"radial_resolution": 0})
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for zero radial_resolution")


def test_output_defaults():
    oc = OutputConfig.from_dict({})
    assert oc.prefix == "R4L5"


def test_components_normalisation():
    c = ComponentsConfig.from_dict(None)
    assert isinstance(c.inferred, list)
    assert isinstance(c.nuisance, list)


def test_config_from_file_toml(tmp_path: Path):
    content = """
[mesh]
radius = 3141.5
radial_resolution = 2
lateral_resolution = 3

[data]
file = "data/brett2024_ic_traveltimes.parquet"

[components]
inferred = []
nuisance = []

[output]
prefix = "R4L5"
"""
    p = tmp_path / "cfg.toml"
    p.write_text(content, encoding="utf8")

    cfg = Config.from_file(p)
    assert isinstance(cfg, Config)
    assert cfg.mesh.radius == 3141.5


def test_config_to_json_dict_serialisation():
    cfg = Config.from_dict(
        {
            "data": {"file": "x.parquet"},
            "mesh": {},
            "components": {"inferred": [], "nuisance": []},
            "output": {},
            "extras": {
                "arr": np.array([1, 2, 3]),
            },
        }
    )
    j = config_to_json_dict(cfg)
    assert isinstance(j["data"]["file"], str)
    assert isinstance(j["extras"]["arr"], list)
