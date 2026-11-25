"""Experiment configuration helpers.

Public API:
- `load_config(path: Path) -> Config` : returns a typed `Config` dataclass.
- `save_resolved_config(outdir: Path, cfg: Config|dict)` : write resolved config to disk.

Use the returned `Config` as follows:
- `cfg.mesh.to_mesh()` to obtain a `SphericalMesh`.
- `cfg.components.to_gaussian_components(context)` to obtain component lists.

This package provides lightweight, dependency-free config loading using
stdlib `tomllib` (TOML) and `json` fallbacks.
"""

from .loader import load_config, save_resolved_config

__all__ = ["load_config", "save_resolved_config"]
