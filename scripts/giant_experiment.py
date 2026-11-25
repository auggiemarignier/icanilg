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

import numpy as np
from joblib import Parallel, delayed
from linear_gaussian import clear_cache
from tqdm import tqdm

from config.components import RawComponentSpec
from config.models import (
    ComponentsConfig,
    Config,
    DataConfig,
    MeshConfig,
    OutputConfig,
)
from utils import mem

from ._model_specs import INFERRED, NUISANCE
from .main import load_data, run

logger = logging.getLogger(__name__)

DATA_CFG = DataConfig(Path("data/brett2024_ic_traveltimes.parquet"))
DB_PATH = Path("outputs/giant_experiment/runs.db")


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


def try_mark_running(db_path: Path, run_id: str) -> bool:
    """Attempt to insert a run row with status 'running'. Returns True if inserted.

    If a row with the same `run_id` already exists, return False.
    """
    try:
        with db_connection(db_path) as conn:
            conn.execute(
                """
                INSERT INTO runs (run_id, status, started_at, pid, host)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    "running",
                    _now_iso(),
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


def record_failure(run_id: str, exc: Exception) -> None:
    """Mark in the database when something fails.

    If marking in the database fails, just log it.
    """
    logger.exception("Run failed: %s", run_id)
    try:
        mark_failed(DB_PATH, run_id, str(exc))
    except Exception:
        logger.exception("Could not record failure: %s", run_id)


def run_job(
    data: np.ndarray,
    context: dict[str, np.ndarray],
    options: tuple[list[RawComponentSpec], list[RawComponentSpec], int, int],
    run_id: str,
) -> None:
    """Main analysis job."""
    inferred, nuisance, radres, latres = options
    mesh_cfg = MeshConfig(radial_resolution=radres, lateral_resolution=latres)
    output_cfg = OutputConfig(str(DB_PATH.parent))
    components_cfg = ComponentsConfig(inferred, nuisance)

    cfg = Config(DATA_CFG, mesh_cfg, components_cfg, output_cfg, {})
    logger.info("Running job %s", run_id)
    _, _, summary, resolved = run(data, context, cfg)
    insert_result(DB_PATH, run_id, "summary", summary)
    insert_result(DB_PATH, run_id, "resolved", resolved)
    mark_finished(DB_PATH, run_id)


def _build_run_id(
    options: tuple[list[RawComponentSpec], list[RawComponentSpec], int, int],
) -> str:
    strs = []
    for o in options[0] + options[1]:
        strs.append(o.get("name", ""))
    for o in options[2:]:
        strs.append(str(o))
    return "-".join(strs)


def process(
    data: np.ndarray,
    context: dict[str, np.ndarray],
    options: tuple[list[RawComponentSpec], list[RawComponentSpec], int, int],
) -> None:
    """Determine if a job needs to be run.  If so, run it."""
    run_id = _build_run_id(options)
    try:
        if not try_mark_running(DB_PATH, run_id):
            logger.info("Skipping already-run job %s", run_id)
            return
    except Exception:
        logger.exception("Failed to mark as running %s", run_id)
        return

    try:
        run_job(data, context, options, run_id)
    except Exception as exc:
        record_failure(run_id, exc)
    finally:
        clear_cache()


def _disable_worker_logging() -> None:
    logging.disable(logging.INFO)


if __name__ == "__main__":
    logging.disable(logging.INFO)
    mem.clear(warn=False)

    rad_res = list(range(1, 11)) + [20, 50]
    lat_res = [1] + list(range(10, 70, 10)) + [100, 150, 200]

    ensure_db(DB_PATH)

    data, context = load_data(DATA_CFG.file)

    prod = list(product(INFERRED, NUISANCE, rad_res, lat_res))
    p = Parallel(
        n_jobs=os.cpu_count() - 2,
        return_as="generator_unordered",
        batch_size=1,
        initializer=_disable_worker_logging,
    )

    with tqdm(total=len(prod), desc="Jobs") as progress:
        for _ in p(delayed(process)(data, context, options) for options in prod):
            progress.update(1)
