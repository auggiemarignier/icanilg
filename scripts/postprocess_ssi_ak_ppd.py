"""Template to startup postprocessing scripts."""

import argparse
import json
import logging
import traceback
from collections.abc import Iterable
from json import dump
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import pandas as pd
from joblib import load

from config.models import Config
from utils.distributions import Posterior, PosteriorPredictive
from utils.forward import construct_ssi_ak_filter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent.resolve()
OUTDIR = ROOT / "outputs"


def _mahalanobis(d: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
    """Compute the squared Mahalanobis distance.

    Yes, I know scipy has the mahalanobis distance in it.
    """
    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        raise ValueError("Matrix is singular or not numerically positive definite")
    r = d - mean
    y = np.linalg.solve(L, r)
    return float(y @ y)


def load_results(results_dir: Path) -> tuple[Posterior, PosteriorPredictive, Config]:
    """Load the posterior and posterior predictive distributions, and the original config."""
    if not results_dir.exists():
        raise FileNotFoundError

    posterior = cast(Posterior, load(results_dir / "posterior.joblib"))
    ppd = cast(PosteriorPredictive, load(results_dir / "ppd.joblib"))
    with open(results_dir / "config_resolved.json", encoding="utf-8") as f:
        cfg = Config.from_dict(json.load(f)["config"])

    return posterior, ppd, cfg


dict_s = dict[str, Any]


class AnalysisFn(Protocol):
    """Contract for analysis functions."""

    def __call__(
        self, posterior: Posterior, ppd: PosteriorPredictive, context: dict_s
    ) -> dict_s | None:
        """Analysis functions do whatever they want to posterior and ppd."""


def ppd_mahalanobis_total(
    posterior: Posterior, ppd: PosteriorPredictive, context: dict_s
) -> dict_s:
    """AnalysisFn to compute the total Mahalanobis distance of the observed data from the ppd mean."""

    return {"mahalanobis_total": _mahalanobis(ppd.d, ppd.mean, ppd.cov)}


def ppd_mahalanobis_ssi_ak(
    posterior: Posterior, ppd: PosteriorPredictive, context: dict_s
) -> dict_s:
    """AnalysisFn to compute the total Mahalanobis distance of the observed data from the ppd mean for the SSI-AK subset."""
    filt = context["filt"]
    d = ppd.d[np.ix_(filt)]
    m = ppd.mean[np.ix_(filt)]
    cov = ppd.cov[np.ix_(filt, filt)]

    return {"mahalanobis_ssi_ak": _mahalanobis(d, m, cov) / _mahalanobis(ppd.d, ppd.mean, ppd.cov)}


def ppd_mahalanobis_ssi_ak_complement(
    posterior: Posterior, ppd: PosteriorPredictive, context: dict_s
) -> dict_s:
    """AnalysisFn to compute the total Mahalanobis distance of the observed data from the ppd mean for complement of the SSI-AK subset."""
    filt = np.logical_not(context["filt"])
    d = ppd.d[np.ix_(filt)]
    m = ppd.mean[np.ix_(filt)]
    cov = ppd.cov[np.ix_(filt, filt)]

    return {"mahalanobis_ssi_ak_comp": _mahalanobis(d, m, cov) / _mahalanobis(ppd.d, ppd.mean, ppd.cov)}


Step = tuple[str, AnalysisFn]  # (name, callable)

_pipeline: tuple[Step, ...] = (
    ("mahalanobis_total", ppd_mahalanobis_total),
    ("mahalanobis_ssi_ak", ppd_mahalanobis_ssi_ak),
    ("mahalanobis_ssi_ak_comp", ppd_mahalanobis_ssi_ak_complement),
    ("ln_evidence", lambda posterior, ppd, context: {"ln_evidence": posterior.ln_Z}),
)


def _normalise_output(out: dict_s | None) -> tuple[dict_s, dict_s, dict_s]:
    """Convert Step output into (scalars, artifacts, meta)."""
    if out is None:
        return {}, {}, {}
    if not isinstance(out, dict):
        raise TypeError("Step must return None or dict")

    if any(k in out for k in ("scalars", "artifacts", "meta")):
        # if caller used canonical keys, use them
        scalars = dict(out.get("scalars", {}))
        artifacts = dict(out.get("artifacts", {}))
        meta = dict(out.get("meta", {}))
    else:
        # treat out as scalars dictionary
        scalars, artifacts, meta = dict(out), {}, {}

    def _convert(v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, (np.generic, np.floating)):
            return float(v)
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v

    scalars = {str(k): _convert(v) for k, v in scalars.items()}
    artifacts = {str(k): _convert(v) for k, v in artifacts.items()}
    meta = {str(k): _convert(v) for k, v in meta.items()}

    return scalars, artifacts, meta


class AnalysisPipeline:
    """Runner for all the analysis functions."""

    def __init__(
        self,
        steps: Iterable[Step],
        fail_fast: bool = False,
        defaults: dict_s | None = None,
    ) -> None:
        _names = [step[0] for step in steps]
        _unique_names = set(_names)
        if len(_unique_names) != len(_names):
            raise ValueError("Duplicate analysis step names found.")

        self.steps = tuple(steps)
        self.fail_fast = fail_fast
        self.defaults: dict_s = dict(defaults or {})

    def __call__(
        self, posterior: Posterior, ppd: PosteriorPredictive, context: dict_s
    ) -> dict_s:
        """Run the pipeline."""
        ctx: dict_s = {**self.defaults, **context}
        summary: dict[str, dict_s] = {"scalars": {}, "artifacts": {}, "meta": {}}

        for name, fn in self.steps:
            logger.info("Starting step %s", name)
            try:
                out = fn(posterior, ppd, ctx)
            except Exception as exc:
                if self.fail_fast:
                    raise exc
                summary["meta"][name] = {"error": traceback.format_exc()}
                continue

            scalars, artifacts, meta = _normalise_output(out)
            if scalars:
                if len(scalars) == 1:
                    # only a single value so unpack to avoid having {name: {name: value}}
                    k, v = next(iter(scalars.items()))
                    summary["scalars"][name] = v if k == name else scalars
                else:
                    summary["scalars"][name] = scalars
            if artifacts:
                summary["artifacts"][name] = artifacts
            if meta:
                summary["meta"] = meta

        return summary


def load_context(cfg: Config) -> dict_s:
    """Load a lot of context from the config."""
    mesh = cfg.mesh.to_mesh()

    data_file = ROOT / cfg.data.file
    df = pd.read_parquet(data_file)
    data = (df.delta_t / df.inner_core_travel_time).astype(float).to_numpy()
    ic_in = np.stack(df.in_location.tolist())
    ic_out = np.stack(df.out_location.tolist())
    turning_point = np.stack(df.turning_point.tolist())
    return {  # all the arguments to constructors only known at runtime
        "n_params": mesh.n_cells * 3,
        "n_data": data.size,
        "ic_in": ic_in,
        "ic_out": ic_out,
        "turning_point": turning_point,
        "ref_phase": df.reference_phase.to_list(),
        "ic_tt": df.inner_core_travel_time.astype(float).to_numpy(),
        "zeta": df.zeta.astype(float).to_numpy(),
        "mesh": mesh,
    }


def main():
    """Analyse the posterior and posterior predictive distributions."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    results_dir = Path(args.run_id)
    posterior, ppd, cfg = load_results(results_dir)
    context = load_context(cfg)
    outdir = results_dir / "analysis"
    outdir.mkdir(parents=True, exist_ok=True)

    context["outdir"] = outdir
    # try to find the ssi-ak filter radius from the config
    try:
        radius = float(cfg.components.inferred[1]["A"]["kwargs"]["radius"])
    except IndexError or KeyError:
        radius = 15.0
    context["filt"] = (
        construct_ssi_ak_filter(context["turning_point"], context["zeta"], radius)
        .sum(axis=1)
        .astype(bool)
    )

    summary = AnalysisPipeline(_pipeline)(posterior, ppd, context)
    with open(outdir / "analysis.json", "w") as f:
        dump(summary, f, indent=2)
    logger.info("Analysis summary: %s", summary["scalars"])


if __name__ == "__main__":
    main()
