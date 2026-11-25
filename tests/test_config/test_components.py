import numpy as np
import pytest

from config.components import (
    _ArrayKinds,
    register_builder,
    resolve_field,
)


def dummy_builder(kwargs, context):
    n = int(kwargs.get("n") or context.get("n") or 1)
    return np.full((n, n), 7.0)


def test_register_and_resolve_builder_calls_kwargs_and_context():
    # register and resolve by name
    register_builder("test_dummy", dummy_builder)
    out = resolve_field(
        {"builder": "test_dummy", "kwargs": {"n": 3}}, _ArrayKinds.A, {}
    )
    assert out.shape == (3, 3)
    assert np.all(out == 7.0)


def test_dotted_import_resolves_local_builder():
    # dotted import (module:function) should resolve to our local test module
    spec = f"{__name__}:dummy_builder"
    out2 = resolve_field(spec, _ArrayKinds.A, {"n": 2})
    assert out2.shape == (2, 2)
    assert np.all(out2 == 7.0)


def test_callable_shorthand():
    out = resolve_field(dummy_builder, _ArrayKinds.A, {"n": 2})
    assert out.shape == (2, 2)
    assert np.all(out == 7.0)


def test_bare_int_mu_returns_zeros():
    out_mu = resolve_field(4, _ArrayKinds.mu, {})
    assert out_mu.shape == (4,)
    assert np.all(out_mu == 0.0)


def test_bare_int_A_returns_eye():
    out_A = resolve_field(3, _ArrayKinds.A, {})
    assert out_A.shape == (3, 3)
    assert np.allclose(out_A, np.eye(3))


def test_nspec_dict_returns_mu_zeros():
    out_mu2 = resolve_field({"n": 5}, _ArrayKinds.mu, {})
    assert out_mu2.shape == (5,)
    assert np.all(out_mu2 == 0.0)


def test_missing_n_raises():
    with pytest.raises(ValueError):
        resolve_field(None, _ArrayKinds.mu, {})


def test_invalid_builder_raises_importerror():
    with pytest.raises(ImportError):
        resolve_field("no.such.module:fn", _ArrayKinds.A, {})


def test_is_builderspec_predicate():
    from config.components import _is_builderspec

    assert _is_builderspec({"builder": "mod:fn"})
    assert _is_builderspec({"builder": dummy_builder, "kwargs": {}})
    assert not _is_builderspec({})
    assert not _is_builderspec({"builder": 3})
    assert not _is_builderspec({"builder": "x", "kwargs": "no"})


def test_is_nspec_predicate():
    from config.components import _is_nspec

    assert _is_nspec({"n": 3})
    assert not _is_nspec({"n": "3"})
    assert not _is_nspec(3)


def test_build_components_basic():
    from config.components import build_components

    cfg = [{"A": np.eye(2), "C": np.eye(2), "mu": np.zeros(2), "name": "c1"}]
    comps = build_components(cfg)
    assert len(comps) == 1
    comp = comps[0]
    assert np.allclose(comp.A, np.eye(2))
    assert np.allclose(comp.C, np.eye(2))
    assert np.allclose(comp.mu, np.zeros(2))


def test_build_components_defaults():
    from config.components import build_components

    # missing mu/C use context['n'] defaults
    cfg2 = [{"A": np.eye(3)}]
    comps2 = build_components(cfg2, context={"n": 3})
    assert len(comps2) == 1
    comp2 = comps2[0]
    assert np.allclose(comp2.A, np.eye(3))
    assert comp2.mu.shape == (3,)
