"""Loading objects from configuration."""

import json
import tomllib
from pathlib import Path
from typing import Any

from .models import Config, config_to_json_dict


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as fh:
        return tomllib.load(fh)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf8") as fh:
        return json.load(fh)


def load_config(path: Path) -> Config:
    """Load a TOML or JSON configuration file and return a typed `Config`.

    Supports `.toml` and `.json`. Returns a `Config` dataclass instance.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    if path.suffix.lower() == ".toml":
        raw = _load_toml(path)
    elif path.suffix.lower() == ".json":
        raw = _load_json(path)
    else:
        try:
            raw = _load_toml(path)
        except tomllib.TOMLDecodeError:
            raw = _load_json(path)

    return Config.from_dict(raw)


# Note: `build_mesh` and `build_components` are intentionally removed. Use
# `Config.mesh.to_mesh()` and `Config.components.to_gaussian_components()`.


def save_resolved_config(outdir: Path, cfg: dict[str, Any]) -> None:
    """Save the resolved configuration into `outdir/config_resolved.json`.

    We use JSON for the dump to avoid adding toml writer dependencies.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    target = outdir / "config_resolved.json"
    # If user passed a dataclass `Config`, convert to a JSON-serialisable dict.
    if isinstance(cfg, Config):
        to_dump = config_to_json_dict(cfg)
    else:
        to_dump = cfg

    with target.open("w", encoding="utf8") as fh:
        json.dump(to_dump, fh, indent=2, ensure_ascii=False)
