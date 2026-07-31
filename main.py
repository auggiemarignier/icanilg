"""Solve the IC anisotropy problem modelled as a purely linear gaussian system."""

import argparse
import datetime
import logging
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from linear_gaussian import (
    calc_log_evidence,
    calc_posterior_cov,
    calc_posterior_mean,
)

from config import load_config, save_resolved_config
from config.builders import make_builder
from config.components import register_builder
from config.models import config_to_json_dict
from utils import (
    block_iid,
    construct_forward_map,
    correlated_paths,
    iid,
    spherically_correlated_independent_anisotropy,
)

# basic module logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


register_builder("forward.build_forward", make_builder(construct_forward_map))
register_builder("noise:block_iid", make_builder(block_iid))
register_builder("noise:correlated_paths", make_builder(correlated_paths))

register_builder("main:eye", make_builder(lambda n_data: np.eye(n_data)))
register_builder("prior.iid", make_builder(iid))
register_builder(
    "prior.spherically_correlated",
    make_builder(spherically_correlated_independent_anisotropy),
)
register_builder("main:prior_mean", make_builder(lambda n_params: np.zeros(n_params)))
register_builder("main:noise_mean", make_builder(lambda n_data: np.zeros(n_data)))


ROOT = Path(__file__).parent.resolve()


def save(
    mean: np.ndarray, cov: np.ndarray, ev: float, outdir: Path | str | None = None
) -> Path:
    """Save the mean and covariance in numpy files, and the evidence in a txt files.

    outdir is an optional root output directory.  Output files will be saved in a timestamped subdirectory.

    Returns the full output directory path.
    """
    if outdir is None:
        outdir = ROOT / "outputs"
    outdir = Path(outdir)

    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    hex = uuid.uuid4().hex[:8]
    outdir /= f"{now}_{hex}"
    outdir.mkdir(parents=True, exist_ok=False)
    np.save(outdir / "mean", mean)
    np.save(outdir / "cov", cov)
    with open(outdir / "evidence.txt", "w") as f:
        f.write(str(ev))
    return outdir


def main():
    """Run an inversion based on a given config file."""
    parser = argparse.ArgumentParser(
        description="Run IC anisotropy inference from a config file"
    )
    parser.add_argument(
        "--config", "-c", default=str(ROOT / "experiments" / "default.toml")
    )
    args = parser.parse_args()

    cfg = load_config(Path(args.config))

    data_file = Path(cfg.data.file)
    logger.info("Reading data from %s", data_file)
    df = pd.read_parquet(data_file)
    data = (df.delta_t / df.inner_core_travel_time).astype(float).to_numpy()
    ic_in = np.stack(df.in_location.tolist())
    ic_out = np.stack(df.out_location.tolist())

    logger.debug(
        "Loaded data: n_obs=%d ic_in.shape=%s ic_out.shape=%s",
        data.shape[0],
        ic_in.shape,
        ic_out.shape,
    )

    mesh = cfg.mesh.to_mesh()

    context = {  # all the arguments to constructors only known at runtime
        "n_params": mesh.n_cells * 3,
        "n_data": data.size,
        "ic_in": ic_in,
        "ic_out": ic_out,
        "ref_phase": df.reference_phase.to_list(),
        "ic_tt": df.inner_core_travel_time.astype(float).to_numpy(),
        "mesh": mesh,
    }
    inferred, nuisance = cfg.components.to_gaussian_components(context=context)

    logger.info(
        "Running inference: prior cov shape=%s noise cov shape=%s",
        inferred[0].C.shape,
        nuisance[0].C.shape,
    )

    Cp = calc_posterior_cov(inferred, nuisance)
    logger.info("Posterior covariance shape: %s", Cp.shape)
    mp = calc_posterior_mean(data, inferred, nuisance)
    logger.info("Posterior mean shape: %s", mp.shape)
    Zp = calc_log_evidence(data, inferred, nuisance)
    logger.info("Log-evidence: %s", Zp)

    outdir = save(mp, Cp, Zp, cfg.output.prefix)

    # save resolved configuration for provenance
    resolved = {
        "config": config_to_json_dict(cfg),
        "derived": {
            "n_data": int(context["n_data"]),
            "n_params": int(context["n_params"]),
            "mesh_n_cells": getattr(mesh, "n_cells", None),
        },
    }
    save_resolved_config(outdir, resolved)
    logger.info("Saved outputs and resolved config to %s", outdir)


if __name__ == "__main__":
    main()
