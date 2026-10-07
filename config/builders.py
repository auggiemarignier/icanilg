"""Generic builder adaptors and small helpers for config-defined builders.

Builders are callables with signature ``builder(kwargs, context)``.  The
``make_builder`` factory adapts an existing function (or import path) into that
signature by inspecting its parameters and pulling values from `kwargs` then
`context`.
"""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable
from functools import partial
from typing import Any


def _resolve_callable(obj: Any) -> Callable:
    if isinstance(obj, partial):
        return obj.func
    if callable(obj):
        return obj
    if isinstance(obj, str):
        if ":" in obj:
            mod, attr = obj.split(":", 1)
        else:
            mod, attr = obj.rsplit(".", 1)
        m = importlib.import_module(mod)
        return getattr(m, attr)
    raise TypeError("Builder must be a callable or import-path string")


def make_builder(target: Any) -> Callable[[dict, dict], Any]:
    """Wrap `target` so it can be called as ``builder(kwargs, context)``.

    Behaviour
    - Inspect the target's signature.
    - For each named parameter, use `kwargs[name]` if present, else `context[name]`.
    - If the parameter has a default, that default is used when neither supplies it.
    - If the target accepts ``**kwargs`` the merged mapping ``{**context, **kwargs}``
      is passed through.
    - Raises ``TypeError`` if a required parameter is missing.
    """
    fn = _resolve_callable(target)
    sig = inspect.signature(fn)
    params = sig.parameters
    accepts_var_kw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    )

    named_params = [
        name
        for name, p in params.items()
        if p.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    ]

    def builder(kwargs: dict, context: dict):
        kwargs = dict(kwargs or {})
        context = dict(context or {})

        if accepts_var_kw:
            merged = {**context, **kwargs}
            return fn(**merged)

        call_args = {}
        for name in named_params:
            if name in kwargs:
                call_args[name] = kwargs[name]
            elif name in context:
                call_args[name] = context[name]
            else:
                p = params[name]
                if p.default is not inspect._empty:
                    call_args[name] = p.default
                else:
                    raise TypeError(
                        f"Missing required argument '{name}' for {fn.__name__}"
                    )
        return fn(**call_args)

    builder.__name__ = fn.__name__
    return builder
