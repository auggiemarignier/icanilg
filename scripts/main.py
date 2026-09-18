"""Solve the IC anisotropy problem modelled as a purely linear gaussian system."""

import argparse
import datetime
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from linear_gaussian import (
    GaussianComponent,
    calc_log_evidence,
    calc_posterior_cov,
    calc_posterior_mean,
    calc_posterior_predictive_cov,
    calc_posterior_predictive_mean,
)

from analysis import AnalysisPipeline, save_analysis
from analysis.pipeline import Step
from analysis.ssi_ak_ppd import (
    ppd_mahalanobis_ssi_ak,
    ppd_mahalanobis_ssi_ak_complement,
    ppd_mahalanobis_total,
)
from config import load_config, save_resolved_config
from config.models import Config, config_to_json_dict
from utils.distributions import Posterior, PosteriorPredictive
from utils.forward import construct_ssi_ak_filter

logger = logging.getLogger(__name__)


ROOT = Path(__file__).parent.parent.resolve()


def save_full_outputs(
    posterior: Posterior,
    ppd: PosteriorPredictive,
    outdir: Path,
) -> Path:
    """Save the posterior and posterior predictive.

    outdir is an optional root output directory.
    Output files will be saved in a subdirectory <outdir>/<run_id>.

    Returns the full output directory path.
    """
    from joblib import dump

    outdir.mkdir(parents=True, exist_ok=False)
    dump(posterior, outdir / "posterior.joblib")
    dump(ppd, outdir / "ppd.joblib")
    return outdir


def load_data(file: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """From a data file return the traveltimes and some context."""
    df = pd.read_parquet(file)
    data = (
        (df.delta_t / df.inner_core_travel_time).astype(float).to_numpy().astype(float)
    )
    ic_in = np.stack(df.in_location.tolist())
    ic_out = np.stack(df.out_location.tolist())
    turning_point = np.stack(df.turning_point.tolist())

    context = {  # all the arguments to constructors only known at runtime
        "n_data": data.size,
        "ic_in": ic_in,
        "ic_out": ic_out,
        "turning_point": turning_point,
        "ref_phase": df.reference_phase.to_list(),
        "ic_tt": df.inner_core_travel_time.astype(float).to_numpy(),
        "zeta": df.zeta.astype(float).to_numpy(),
    }
    return data, context


def _default_ssi_ak_filter(
    cfg: Config, turning_point: np.ndarray, zeta: np.ndarray
) -> np.ndarray:
    try:
        radius = float(cfg.components.inferred[1]["A"]["kwargs"]["radius"])
    except IndexError or KeyError:
        radius = 15.0
    return construct_ssi_ak_filter(turning_point, zeta, radius).sum(axis=1).astype(bool)


def _run_id() -> str:
    from uuid import uuid4

    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    hex = uuid4().hex[:8]
    return f"{now}_{hex}"


def infer_posterior(
    d: np.ndarray, inferred: list[GaussianComponent], nuisance: list[GaussianComponent]
) -> Posterior:
    """Infer the posterior distribution."""
    Cp = calc_posterior_cov(inferred, nuisance)
    mp = calc_posterior_mean(d, inferred, nuisance)
    Zp = calc_log_evidence(d, inferred, nuisance)
    return Posterior(mp, Cp, d, Zp)


def infer_ppd(
    d: np.ndarray, inferred: list[GaussianComponent], nuisance: list[GaussianComponent]
) -> PosteriorPredictive:
    """Infer the posterior predictive distribution."""
    C_pred = calc_posterior_predictive_cov(inferred, nuisance)
    mu_pred = calc_posterior_predictive_mean(d, inferred, nuisance)
    return PosteriorPredictive(mu_pred, C_pred, d)


analysis_steps: tuple[Step, ...] = (
    ("ln_evidence", lambda posterior, ppd, context: {"ln_evidence": posterior.ln_Z}),
    ("mahalanobis_total", ppd_mahalanobis_total),
    ("mahalanobis_ssi_ak", ppd_mahalanobis_ssi_ak),
    ("mahalanobis_ssi_ak_comp", ppd_mahalanobis_ssi_ak_complement),
)


def run(
    data: np.ndarray,
    context: dict[str, Any],
    cfg: Config,
) -> tuple[Posterior, PosteriorPredictive, dict[str, Any], dict[str, Any]]:
    """Run an experiment given a config."""

    logger.info("Creating mesh")
    mesh = cfg.mesh.to_mesh()
    local_ctx = {"mesh": mesh}

    logger.info("Configuring Gaussian Components")
    inferred, nuisance = cfg.components.to_gaussian_components(
        context={**context, **local_ctx}
    )

    logger.info("Running inference: n_params=%d", local_ctx["n_params"])
    posterior = infer_posterior(data, inferred, nuisance)
    logger.info("Posterior covariance shape: %s", posterior.cov.shape)
    logger.info("Posterior mean shape: %s", posterior.mean.shape)
    logger.info("Log-evidence: %s", posterior.ln_Z)

    ppd = infer_ppd(data, inferred, nuisance)
    logger.info("Posterior predicted covariance shape: %s", ppd.cov.shape)
    logger.info("Posterior predicted mean shape: %s", ppd.mean.shape)

    logger.info("Running Posterior Analysis")
    summary = AnalysisPipeline(
        analysis_steps,
        defaults={
            "filt": _default_ssi_ak_filter(
                cfg, context["turning_point"], context["zeta"]
            )
        },
    )(posterior, ppd, {**context, **local_ctx})
    logger.info("Analysis summary: %s", summary["scalars"])

    resolved = {
        "config": config_to_json_dict(cfg),
        "derived": {
            "n_data": int(context["n_data"]),
            "n_params": int(local_ctx["n_params"]),
            "mesh_n_cells": getattr(mesh, "n_cells", None),
        },
    }
    return posterior, ppd, summary, resolved


def main():
    """Run an inversion based on a given config file."""
    parser = argparse.ArgumentParser(
        description="Run IC anisotropy inference from a config file"
    )
    parser.add_argument(
        "--config", "-c", default=str(ROOT / "experiments" / "default.toml")
    )
    parser.add_argument("--run-id", default=_run_id())
    parser.add_argument(
        "--save-dists",
        action="store_true",
        help="Save the full Posterior and Posterior Predictive",
    )
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    logger.info("Reading data from %s", cfg.data.file)
    data, context = load_data(cfg.data.file)
    logger.debug("Loaded data: n_obs=%d", data.shape[0])
    posterior, ppd, summary, resolved = run(data, context, cfg)

    outdir = ROOT / "outputs" / cfg.output.prefix / args.run_id
    outdir.mkdir(parents=True, exist_ok=False)
    save_resolved_config(outdir, resolved)
    if args.save_dists:
        save_full_outputs(posterior, ppd, outdir)
    save_analysis(summary, outdir)
    logger.info("Saved outputs and resolved config to %s", outdir)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    main()
