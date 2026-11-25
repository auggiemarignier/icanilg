"""Posterior and PPD Analysis."""

import logging
import traceback
from collections.abc import Iterable
from json import dump
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from utils.distributions import Posterior, PosteriorPredictive

from ._types import dict_s

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class AnalysisFn(Protocol):
    """Contract for analysis functions."""

    def __call__(
        self, posterior: Posterior, ppd: PosteriorPredictive, context: dict_s
    ) -> dict_s | None:
        """Analysis functions do whatever they want to posterior and ppd."""


Step = tuple[str, AnalysisFn]  # (name, callable)


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


def save_analysis(summary: dict_s, outdir: Path) -> None:
    """Save the analysis dictionary in a subdirectory of the output directory."""

    outdir /= "analysis"
    outdir.mkdir(parents=True, exist_ok=False)
    with open(outdir / "analysis.json", "w") as f:
        dump(summary, f, indent=2)
