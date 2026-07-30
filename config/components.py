"""Component builders and a small factory for declarative component configs.

This module provides helpers to construct `GaussianComponent` instances from
configuration dictionaries. It aims to be backwards-compatible with the
existing simple `from_A` / `identity` behaviour while allowing per-component
`A` and richer `C` specifications.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, NotRequired, TypedDict, cast

import numpy as np
from linear_gaussian import GaussianComponent


class _ArrayKinds(StrEnum):
    """The types of arrays in a GaussianComponent."""

    A = "A"
    C = "C"
    mu = "mu"


_array_default_builders: dict[_ArrayKinds, Callable[..., np.ndarray]] = {
    _ArrayKinds.A: np.eye,
    _ArrayKinds.C: np.eye,
    _ArrayKinds.mu: np.zeros,
}

BuilderFn = Callable[[dict[str, Any], dict[str, Any]], np.ndarray]


class _BuilderSpec(TypedDict):
    builder: str | BuilderFn
    kwargs: NotRequired[dict[str, Any]]


def _is_builderspec(obj: Any) -> bool:
    """Runtime check if an object matches the `_BuilderSpec` typed dictionary."""
    if not isinstance(obj, dict):
        return False
    if (builder := obj.get("builder")) is None:
        return False
    if not (callable(builder) or isinstance(builder, str)):
        return False
    if "kwargs" in obj and not isinstance(obj["kwargs"], dict):
        return False
    return True


class _NSpec(TypedDict):
    n: int


def _is_nspec(obj: Any) -> bool:
    """Runtime check if an object matches the `_NSpec` typed dictionary."""
    return isinstance(obj, dict) and isinstance(obj.get("n"), int)


FieldSpec = np.ndarray | _BuilderSpec | _NSpec | BuilderFn | int | str


class RawComponentSpec(TypedDict, total=False):
    """How a user defines the different bits of a GaussianComponent in dictionary form.

    Each of A, C, mu can be either:
        - a plain array
        - a builder function configuration (`_BuilderSpec`)
        - a single integer (`_NSpec`), normally `n_data` as this will return a default zero mean Gaussian with unit covariance and identity transformation.

    An optional name string can be given
    """

    A: FieldSpec
    C: FieldSpec
    mu: FieldSpec
    name: str


@dataclass
class ComponentSpec:
    """Structured GaussianComponent specification object."""

    A_spec: FieldSpec | None
    C_spec: FieldSpec | None
    mu_spec: FieldSpec | None
    name: str | None = None

    @classmethod
    def from_dict(cls, d: RawComponentSpec) -> ComponentSpec:
        """Normalise a dictionary into this data structure."""
        return cls(
            A_spec=d.get("A"),
            C_spec=d.get("C"),
            mu_spec=d.get("mu"),
            name=d.get("name"),
        )

    def to_gaussian_component(
        self, context: dict[str, Any] | None = None
    ) -> GaussianComponent:
        """Build the GaussianComponent object from this specification."""
        if context is None:
            context = {}

        A = resolve_field(self.A_spec, _ArrayKinds.A, context)
        C = resolve_field(self.C_spec, _ArrayKinds.C, context)
        mu = resolve_field(self.mu_spec, _ArrayKinds.mu, context)
        return GaussianComponent(A, mu, C)


def resolve_field(
    spec: FieldSpec | None,
    kind: _ArrayKinds,
    context: dict[str, Any],
) -> np.ndarray:
    """Normalise a field spec to an ndarray.

    Behaviour:
    - If spec is an ndarray/array-like -> return ndarray (preserve identity if ndarray).
    - If spec is builder form {'builder': <str|callable>, 'kwargs': {...}} -> call builder(kwargs, context).
    - If spec is {'n': int} or spec is None but context has 'n' -> identities/zeros rule.
    """
    if isinstance(spec, np.ndarray):
        return spec

    if _is_builderspec(spec):
        d = cast(_BuilderSpec, spec)
        builder = _resolve_builder(d["builder"])
        kwargs = d.get("kwargs", {})
        out = builder(kwargs, context)
        return np.asarray(out, dtype=float)
    if isinstance(spec, str) or callable(spec):
        d = cast(str | BuilderFn, spec)
        builder = _resolve_builder(d)
        out = builder({}, context)
        return np.asarray(out, dtype=float)

    if _is_nspec(spec):
        d = cast(_NSpec, spec)
        n = int(d["n"])
        return _array_default_builders[kind](n)
    if isinstance(spec, int):
        return _array_default_builders[kind](spec)

    if not spec:
        # treat missing as n-only fallback via context
        if (n := context.get("n")) is None:
            raise ValueError(f"Missing {kind} and no 'n' in context")
        return _array_default_builders[kind](n)

    raise TypeError(f"Cannot resolve {kind} from spec: {spec!r}")


_BUILDER_REGISTRY: dict[str, BuilderFn] = {}


def register_builder(name: str, fn: BuilderFn) -> None:
    """Register a builder callable under `name`.

    Builders should accept `(kwargs: dict, context: dict)` and return an
    array-like object convertible to `np.ndarray`.
    """
    _BUILDER_REGISTRY[name] = fn


def _resolve_builder(name_or_callable: str | BuilderFn) -> BuilderFn:
    """Resolve a builder by name, dotted path, or callable.

    Accepts a registered name (via `register_builder`), a dotted import path
    (e.g. ``module.sub:fn`` or ``module.sub.fn``), or a direct callable.

    """
    if callable(name_or_callable):
        return cast(BuilderFn, name_or_callable)

    name = str(name_or_callable)
    # Registered builder
    if name in _BUILDER_REGISTRY:
        return _BUILDER_REGISTRY[name]

    # Dotted import: module:attr or module.attr
    if ":" in name:
        mod_name, attr = name.split(":", 1)
    else:
        parts = name.rsplit(".", 1)
        if len(parts) == 2:
            mod_name, attr = parts
        else:
            raise ImportError(f"Unable to find a function called `{name}`.")

    mod = importlib.import_module(mod_name)
    try:
        return getattr(mod, attr)
    except AttributeError as exc:
        raise ImportError(f"Cannot import builder '{name}': {exc}") from exc


def build_components(
    components_cfg: Iterable[RawComponentSpec] | None,
    context: dict[str, Any] | None = None,
) -> list[GaussianComponent]:
    """Construct a list of `GaussianComponent` from a components config.

    - `components_cfg`: iterable of per-component specs (or None).
    - `context`: optional dict merged into the builder context.

    Returns an empty list when `components_cfg` is falsy.
    """
    components: list[GaussianComponent] = []
    if not components_cfg:
        return components

    for comp in components_cfg:
        spec = ComponentSpec.from_dict(comp)
        comp_obj = spec.to_gaussian_component(context)
        components.append(comp_obj)

    return components
