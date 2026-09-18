"""Script to loop through a bunch of configurations."""

import json
import logging
import os
import socket
import sqlite3
from collections.abc import Generator
from contextlib import closing, contextmanager
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from typing import Any

from joblib import Parallel, delayed
from linear_gaussian import clear_cache

from config.models import (
    ComponentsConfig,
    ComponentsDict,
    Config,
    DataConfig,
    MeshConfig,
    OutputConfig,
)

from .main import load_data, run

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

SHARED_DATA = None
SHARED_CONTEXT = None


def _init_worker(data_path: Path) -> None:
    global SHARED_DATA, SHARED_CONTEXT

    SHARED_DATA, SHARED_CONTEXT = load_data(data_path)


DATA_CFG = DataConfig(Path("data/brett2024_ic_traveltimes.parquet"))
DB_PATH = Path("outputs/giant_experiment/ge/runs.db")


@contextmanager
def db_connection(db_path: Path) -> Generator[sqlite3.Connection]:
    """Open a configured SQLite connection and transaction."""
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with closing(sqlite3.connect(db_path, timeout=30)) as conn:
        conn.execute("PRAGMA foreign_keys = ON")

        with conn:
            yield conn


def ensure_db(db_path: Path) -> None:
    """Check the database exists, otherwise build it."""
    with db_connection(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY,
                run_id TEXT UNIQUE,
                status TEXT,
                started_at TEXT,
                finished_at TEXT,
                options TEXT,
                pid INTEGER,
                host TEXT,
                details TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY,
                run_id TEXT,
                name TEXT,
                data TEXT,
                created_at TEXT,
                FOREIGN KEY (run_id) REFERENCES runs (run_id) ON DELETE CASCADE
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_results_run_id ON results (run_id)"
        )


def _now_iso() -> str:
    return datetime.now(UTC).isoformat() + "Z"


def try_mark_running(db_path: Path, run_id: str, options: Any) -> bool:
    """Attempt to insert a run row with status 'running'. Returns True if inserted.

    If a row with the same `run_id` already exists, return False.
    """
    try:
        with db_connection(db_path) as conn:
            conn.execute(
                """
                INSERT INTO runs (run_id, status, started_at, options, pid, host)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    "running",
                    _now_iso(),
                    json.dumps(list(options)),
                    os.getpid(),
                    socket.gethostname(),
                ),
            )
    except sqlite3.IntegrityError:
        return False
    return True


def mark_finished(db_path: Path, run_id: str, details: str | None = None) -> None:
    """Mark run_id as finished in the database."""
    with db_connection(db_path) as conn:
        conn.execute(
            "UPDATE runs SET status = ?, finished_at = ?, details = ? WHERE run_id = ?",
            ("completed", _now_iso(), details, run_id),
        )


def mark_failed(db_path: Path, run_id: str, details: str | None = None) -> None:
    """Mark run_id as failed in the database."""
    with db_connection(db_path) as conn:
        conn.execute(
            "UPDATE runs SET status = ?, finished_at = ?, details = ? WHERE run_id = ?",
            ("failed", _now_iso(), details, run_id),
        )


def insert_result(db_path: Path, run_id: str, name: str, data: Any) -> None:
    """Insert `data` with a `name` into the `run_id` row in the `results` table."""
    with db_connection(db_path) as conn:
        conn.execute(
            "INSERT INTO results (run_id, name, data, created_at) VALUES (?, ?, ?, ?)",
            (run_id, name, json.dumps(data), _now_iso()),
        )


noise_covar_kwargs = {
    "noise.block_iid": {},
    "noise.correlated_paths": {"corr_length": 5},
}
prior_covar_kwargs = {
    "prior.iid": {"scale": 0.1},
    "prior.spherically_correlated": {
        "sigma2": 0.1,
        "lat_corr_length": 30,
        "rad_corr_length": 300,
    },
}


def process(options: tuple[str, str, str, float, int, int]) -> None:
    """Configure and run."""
    run_id = "-".join(str(o) for o in options)
    fwd, noise_c, prior_c, ssiak, radres, latres = options
    if not try_mark_running(DB_PATH, run_id, options):
        logger.info("Skipping already-run job %s", run_id)
        return
    mesh_cfg = MeshConfig(radial_resolution=radres, lateral_resolution=latres)
    output_cfg = OutputConfig("giant_experiment/ge")
    components: ComponentsDict = {
        "inferred": [
            {
                "A": {"builder": fwd, "kwargs": {"normalisation": 0.5}},
                "C": {"builder": prior_c, "kwargs": prior_covar_kwargs[prior_c]},
                "mu": {"builder": "prior.zero_mean", "kwargs": {}},
            }
        ],
        "nuisance": [
            {
                "A": {"builder": "forward.eye", "kwargs": {"scale": 1.0}},
                "C": {"builder": noise_c, "kwargs": noise_covar_kwargs[noise_c]},
                "mu": {"builder": "noise.zero_mean", "kwargs": {}},
            }
        ],
    }
    if ssiak != 0.0:
        components["inferred"].append(
            {
                "A": {"builder": "forward.ssi_ak_filter", "kwargs": {"radius": ssiak}},
                "C": {
                    "builder": "forward.ssi_ak_bias_cov",
                    "kwargs": {"scale": 0.05, "radius": ssiak},
                },
                "mu": {
                    "builder": "forward.ssi_ak_bias_mean",
                    "kwargs": {"radius": ssiak},
                },
            }
        )
    components_cfg = ComponentsConfig.from_dict(components)

    cfg = Config(DATA_CFG, mesh_cfg, components_cfg, output_cfg, {})
    try:
        logger.info("Running job %s", run_id)
        _, _, summary, resolved = run(SHARED_DATA, SHARED_CONTEXT, cfg)
        insert_result(DB_PATH, run_id, "summary", summary)
        insert_result(DB_PATH, run_id, "resolved", resolved)
        mark_finished(DB_PATH, run_id)
    except Exception as e:  # noqa: BLE001 - surface failure in DB
        mark_failed(DB_PATH, run_id, str(e))
        logger.exception("Run failed: %s", run_id)
        raise
    finally:
        clear_cache()


if __name__ == "__main__":
    rad_res = list(range(1, 11))
    lat_res = [1] + list(range(10, 70, 10))
    forwards = [
        "forward.build_forward",
        "forward.build_rti_forward",
        "forward.build_iso_forward",
    ]
    noise_covars = ["noise.block_iid", "noise.correlated_paths"]
    prior_covars = ["prior.iid", "prior.spherically_correlated"]
    ssiak_rads = [0.0, 15.0]

    ensure_db(DB_PATH)

    prod = product(forwards, noise_covars, prior_covars, ssiak_rads, rad_res, lat_res)
    Parallel(
        n_jobs=4,
        backend="multiprocessing",
        initializer=_init_worker,
        initargs=(str(DATA_CFG.file),),
    )(delayed(process)(options) for options in prod)
