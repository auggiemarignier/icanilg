"""Typed dataclasses for experiment configuration.

This module provides `Config.from_dict` / `from_file` helpers that convert the
plain dictionaries returned by `config.loader.load_config` into structured
dataclasses with sensible defaults and light validation. It intentionally
reuses the existing component resolution machinery in `config.components`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, NotRequired, TypedDict

import numpy as np
from raytracer import SphericalMesh

from .components import RawComponentSpec


class DataDict(TypedDict, total=False):
    """Mapping for the `data` section of a config.

    Keys:
        file: Optional path to the input data file (string).
    """

    file: str


class MeshDict(TypedDict, total=False):
    """Mapping for the `mesh` section.

    Keys:
        radius: Optional sphere radius (float).
        radial_resolution: Optional radial subdivision count (int).
        lateral_resolution: Optional lateral subdivision count (int).
    """

    radius: float
    radial_resolution: int
    lateral_resolution: int
    sampling: str


class ComponentsDict(TypedDict, total=False):
    """Mapping for the `components` section.

    Keys:
        inferred: Optional list of raw component builder specs for inferred
            components.
        nuisance: Optional list of raw builder specs for nuisance components.
    """

    inferred: list[RawComponentSpec]
    nuisance: list[RawComponentSpec]


class OutputDict(TypedDict, total=False):
    """Mapping for the `output` section.

    Keys:
        prefix: Optional filename prefix used for outputs.
    """

    prefix: str


class ConfigDict(TypedDict):
    """TypedDict describing a valid top-level configuration mapping.

    Required keys correspond to the main dataclass fields. `extras` may
    contain any other user-provided top-level items.
    """

    data: DataDict
    mesh: MeshDict
    components: ComponentsDict
    output: OutputDict
    extras: NotRequired[dict[str, Any]]


@dataclass
class DataConfig:
    """Configuration for the experiment input data file.

    Attributes:
        file: Path to the input data file.
    """

    file: Path

    @classmethod
    def from_dict(cls, d: DataDict, root: Path | None = None) -> DataConfig:
        """Create a `DataConfig` from a raw mapping.

        Args:
            d: Mapping containing optional `file` key.
            root: Optional project root to resolve relative paths.
        Returns:
            A `DataConfig` instance.
        """
        if d is None:
            d = {}
        if root is None:
            # project root assumed two levels up from this file
            root = Path(__file__).resolve().parent.parent
        file_path = Path(
            d.get("file", str(root / "data" / "brett2024_ic_traveltimes.parquet"))
        )
        return cls(file=file_path)


@dataclass
class MeshConfig:
    """Mesh construction parameters used to build a `SphericalMesh`.

    Attributes:
        radius: Sphere radius in same units as data.
        radial_resolution: Number of radial subdivisions.
        lateral_resolution: Number of lateral subdivisions.
    """

    radius: float = 1221.5
    radial_resolution: int = 4
    lateral_resolution: int = 5
    sampling: Literal["fib", "mw"] = "fib"

    @classmethod
    def from_dict(cls, d: MeshDict) -> MeshConfig:
        """Create a `MeshConfig` from a mapping, applying defaults.

        Raises `ValueError` for invalid numeric values.
        """

        if d is None:
            d = {}
        radius = float(d.get("radius", 1221.5))
        radial = int(d.get("radial_resolution", 4))
        lateral = int(d.get("lateral_resolution", 5))
        sampling = str(d.get("sampling", "fib"))
        # Basic validation
        if radius <= 0:
            raise ValueError("mesh.radius must be > 0")
        if radial < 1 or lateral < 1:
            raise ValueError("mesh resolutions must be >= 1")

        return cls(
            radius=radius,
            radial_resolution=radial,
            lateral_resolution=lateral,
            sampling=sampling,
        )

    def to_mesh(self) -> SphericalMesh:
        """Construct a `SphericalMesh` instance from this config.

        Returns:
            A `SphericalMesh` built with the configured parameters.
        """
        from raytracer import FibonacciSphericalSampling, MWSphericalSampling

        match self.sampling:
            case "fib":
                sampling = FibonacciSphericalSampling(self.lateral_resolution)
            case "mw":
                sampling = MWSphericalSampling(self.lateral_resolution)
            case _:
                raise ValueError(f"Invalid choice of sampling theorem {s}")

        return SphericalMesh(
            self.radius, self.radial_resolution, sampling
        )


@dataclass
class OutputConfig:
    """Configuration for output naming and prefixing.

    Attributes:
        prefix: Output filename prefix.
    """

    prefix: str = "R4L5"

    @classmethod
    def from_dict(cls, d: OutputDict) -> OutputConfig:
        """Create an `OutputConfig` from a mapping, using defaults when absent."""
        if d is None:
            d = {}
        return cls(prefix=str(d.get("prefix", "R4L5")))


@dataclass
class ComponentsConfig:
    """Container for component specifications.

    `inferred` and `nuisance` are lists of raw builder specs that will be
    resolved into runtime component objects.
    """

    inferred: list[RawComponentSpec]
    nuisance: list[RawComponentSpec]

    @classmethod
    def from_dict(cls, d: ComponentsDict) -> ComponentsConfig:
        """Normalize a mapping into a `ComponentsConfig`.

        Ensures the returned fields are lists (possibly empty).
        """
        if d is None:
            d = {}
        inferred = list(d.get("inferred", []) or [])
        nuisance = list(d.get("nuisance", []) or [])
        return cls(inferred=inferred, nuisance=nuisance)

    def to_gaussian_components(self, context: dict[str, Any] | None = None):
        """Resolve configured component specs to `GaussianComponent` objects.

        This delegates to `config.components.build_components` to preserve
        existing builder behaviour and registration.
        """
        from .components import build_components

        if context is None:
            context = {}
        inferred = build_components(self.inferred, context=context)
        nuisance = build_components(self.nuisance, context=context)
        return inferred, nuisance


@dataclass
class Config:
    """Typed, validated configuration object for experiments.

    Construct via `Config.from_dict` which enforces required top-level
    sections and applies defaults where appropriate.
    """

    data: DataConfig
    mesh: MeshConfig
    components: ComponentsConfig
    output: OutputConfig
    extras: dict[str, Any]

    @classmethod
    def from_dict(cls, d: ConfigDict) -> Config:
        """Build a `Config` from a raw mapping.

        Validates presence of required top-level sections and returns a
        populated `Config` dataclass.
        """
        cls._validate_incoming_dict(d)

        data = DataConfig.from_dict(d.get("data"))
        mesh = MeshConfig.from_dict(d.get("mesh"))
        components = ComponentsConfig.from_dict(d.get("components"))
        output = OutputConfig.from_dict(d.get("output"))
        extras = {
            k: v
            for k, v in d.items()
            if k not in ("data", "mesh", "components", "output")
        }
        return cls(
            data=data, mesh=mesh, components=components, output=output, extras=extras
        )

    @staticmethod
    def _validate_incoming_dict(d: ConfigDict) -> None:
        if not isinstance(d, dict):
            raise TypeError("Config.from_dict expects a dict")

        required = ("data", "mesh", "components", "output")
        missing = [k for k in required if k not in d or d.get(k) is None]
        if missing:
            raise ValueError(
                f"configuration missing required sections: {', '.join(missing)}"
            )

    @classmethod
    def from_file(cls, path: Path | str) -> Config:
        """Load a configuration file and return a validated `Config`.

        Delegates to `config.loader.load_config`, which parses TOML/JSON
        and constructs a `Config` via `from_dict`.
        """
        from .loader import load_config

        return load_config(Path(path))


def _make_json_compatible(obj: Any) -> Any:
    """Recursively convert dataclass-asdict output into JSON-serialisable types.

    - Path -> str
    - np.ndarray -> list
    """
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _make_json_compatible(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_compatible(v) for v in obj]
    if callable(obj):
        return obj.__name__
    return obj


def config_to_json_dict(cfg: Config) -> dict[str, Any]:
    """Return a JSON-serialisable dict for a `Config` dataclass."""
    return _make_json_compatible(asdict(cfg))
