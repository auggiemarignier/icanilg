from functools import partial

import numpy as np

from config.builders import make_builder
from config.components import FieldSpec, RawComponentSpec
from utils.forward import (
    construct_iso_forward_map,
    construct_rti_forward_map,
    construct_ssi_ak_filter,
    construct_ti_forward_map,
)
from utils.noise import block_iid, correlated_paths
from utils.priors import mesh_iid, spherically_correlated_independent_dofs, zero_mean

NORMALISATION = 0.5
ti = make_builder(partial(construct_ti_forward_map, normalisation=NORMALISATION))
rti = make_builder(partial(construct_rti_forward_map, normalisation=NORMALISATION))
iso = make_builder(partial(construct_iso_forward_map, normalisation=NORMALISATION))

iid_prior = partial(mesh_iid, scale=0.01)
aniso_iid_prior = make_builder(partial(iid_prior, ndofs=3))
iso_iid_prior = make_builder(partial(iid_prior, ndofs=1))
corr_prior = partial(
    spherically_correlated_independent_dofs,
    sigma2=0.01,
    lat_corr_length=30,
    rad_corr_length=300,
)
aniso_corr_prior = make_builder(partial(corr_prior, ndofs=3))
iso_corr_prior = make_builder(partial(corr_prior, ndofs=1))

aniso_n_params_zeros = make_builder(partial(zero_mean, ndofs=3))
iso_n_params_zeros = make_builder(partial(zero_mean, ndofs=1))
n_data_zeros = make_builder(lambda n_data: np.zeros(n_data))
n_data_eye = make_builder(lambda n_data: np.eye(n_data))

SSI_AK_RAD = 15.0
ssi_ak_filter = make_builder(partial(construct_ssi_ak_filter, radius=SSI_AK_RAD))
ssi_ak_cov = make_builder(lambda: np.asarray([[0.02]]))
ssi_ak_mean = make_builder(lambda: np.asarray([0]))

block_iid = make_builder(block_iid)
correlated_paths = make_builder(
    partial(correlated_paths, corr_length=25, corr_scale=0.005**2)
)

def compose(A: FieldSpec, C: FieldSpec, mu: FieldSpec, name: str) -> RawComponentSpec:
    return {"A": A, "C": C, "mu": mu, "name": name}


def _import_string(varname: str) -> str:
    """Return a string that can import the variable varname from an external place."""
    return f"{__name__}.{varname}"


ssi_ak_spec = compose(
    _import_string("ssi_ak_filter"),
    _import_string("ssi_ak_cov"),
    _import_string("ssi_ak_mean"),
    "ssi_ak_bias",
)
ti_iid_spec = compose(
    _import_string("ti"),
    _import_string("aniso_iid_prior"),
    _import_string("aniso_n_params_zeros"),
    "ti_iid",
)
rti_iid_spec = compose(
    _import_string("rti"),
    _import_string("aniso_iid_prior"),
    _import_string("aniso_n_params_zeros"),
    "rti_iid",
)
iso_iid_spec = compose(
    _import_string("iso"),
    _import_string("iso_iid_prior"),
    _import_string("iso_n_params_zeros"),
    "iso_iid",
)
ti_corr_spec = compose(
    _import_string("ti"),
    _import_string("aniso_corr_prior"),
    _import_string("aniso_n_params_zeros"),
    "ti_corr",
)
rti_corr_spec = compose(
    _import_string("rti"),
    _import_string("aniso_corr_prior"),
    _import_string("aniso_n_params_zeros"),
    "rti_corr",
)
iso_corr_spec = compose(
    _import_string("iso"),
    _import_string("iso_corr_prior"),
    _import_string("iso_n_params_zeros"),
    "iso_corr",
)
INFERRED = [
    [ti_iid_spec],
    [rti_iid_spec],
    [iso_iid_spec],
    [ti_iid_spec, ssi_ak_spec],
    [rti_iid_spec, ssi_ak_spec],
    [iso_iid_spec, ssi_ak_spec],
    [ti_corr_spec],
    [rti_corr_spec],
    [iso_corr_spec],
    [ti_corr_spec, ssi_ak_spec],
    [rti_corr_spec, ssi_ak_spec],
    [iso_corr_spec, ssi_ak_spec],
]

block_iid_noise_spec = compose(
    _import_string("n_data_eye"),
    _import_string("block_iid"),
    _import_string("n_data_zeros"),
    "block_iid_noise",
)
correlated_paths_spec = compose(
    _import_string("n_data_eye"),
    _import_string("correlated_paths"),
    _import_string("n_data_zeros"),
    "correlated_paths",
)
NUISANCE = [
    [block_iid_noise_spec],
    [
        block_iid_noise_spec,
        correlated_paths_spec,
    ],
]
