"""Rerun an experiment using the resolved config saved in the runs database."""

import argparse
import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

from config.loader import save_resolved_config
from config.models import Config
from utils import mem

from .main import ROOT, load_data, run, save_analysis, save_full_outputs

logger = logging.getLogger(__name__)
DEFAULT_DB_PATH = ROOT / "outputs" / "giant_experiment" / "runs.db"


def load_run_config(db_path: Path, run_id: str) -> tuple[Config, dict[str, Any]]:
    """Read and reconstruct the config from the run's resolved result."""
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT data FROM results WHERE run_id = ? AND name = ? "
            "ORDER BY id DESC LIMIT 1",
            (run_id, "resolved"),
        ).fetchone()

    if row is None:
        raise ValueError(f"No resolved result found for run_id {run_id!r} in {db_path}")

    resolved = json.loads(row[0])
    try:
        config_data = resolved["config"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"Resolved result for run_id {run_id!r} does not contain a config"
        ) from exc
    return Config.from_dict(config_data), resolved


def rerun(
    run_id: str,
    db_path: Path = DEFAULT_DB_PATH,
    save_dists: bool = False,
) -> Path:
    """Rerun inference and analysis for ``run_id`` and save results."""
    mem.clear(warn=False)
    cfg, _ = load_run_config(db_path, run_id)
    logger.info("Reading data from %s", cfg.data.file)
    data, context = load_data(cfg.data.file)
    posterior, ppd, summary, resolved = run(data, context, cfg)

    outdir = ROOT / cfg.output.prefix / f"{run_id}"
    outdir.mkdir(parents=True, exist_ok=False)
    save_resolved_config(outdir, resolved)
    if save_dists:
        save_full_outputs(posterior, ppd, outdir)
    save_analysis(summary, outdir)
    logger.info("Saved rerun outputs and resolved config to %s", outdir)
    return outdir


def main() -> None:
    """Parse command-line options and rerun the selected experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id", help="Run identifier stored in the database")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument(
        "--save-dists",
        action="store_true",
        help="Save the full Posterior and Posterior Predictive",
    )
    args = parser.parse_args()
    rerun(args.run_id, args.db, args.save_dists)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    main()
