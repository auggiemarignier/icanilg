"""Unpack summary results from the experiment database into a SQL table."""

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

DEFAULT_DB_PATH = Path("outputs/giant_experiment/runs.db")
TABLE_NAME = "summary_results"


def _quote_identifier(identifier: str) -> str:
    """Quote an SQLite identifier safely."""
    return '"' + identifier.replace('"', '""') + '"'


def unpack_summaries(db_path: Path = DEFAULT_DB_PATH) -> int:
    """Write summary payloads to ``summary_results`` and return their count.

    Scalar entries become columns. The ``artifacts`` and ``meta`` dictionaries
    are retained as JSON text because they may contain nested or large values.
    Re-running the script replaces the derived table contents.
    """
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT run_id, data FROM results WHERE name = ? ORDER BY id",
            ("summary",),
        ).fetchall()

        parsed_rows: list[tuple[str, dict[str, Any]]] = []
        scalar_names: set[str] = set()
        for run_id, raw_data in rows:
            payload = json.loads(raw_data)
            if not isinstance(payload, dict):
                raise TypeError(f"Summary for {run_id!r} is not a dictionary")
            scalars = payload.get("scalars", {})
            if not isinstance(scalars, dict):
                raise TypeError(f"Summary scalars for {run_id!r} are not a dictionary")
            scalar_names.update(str(key) for key in scalars)
            parsed_rows.append((run_id, payload))

        columns = ["run_id", *sorted(scalar_names), "artifacts", "meta"]
        definitions = [f"{_quote_identifier('run_id')} TEXT PRIMARY KEY"]
        definitions.extend(
            f"{_quote_identifier(name)} NUMERIC" for name in sorted(scalar_names)
        )
        definitions.extend(
            [
                f"{_quote_identifier('artifacts')} TEXT",
                f"{_quote_identifier('meta')} TEXT",
            ]
        )

        with conn:
            conn.execute(f"DROP TABLE IF EXISTS {_quote_identifier(TABLE_NAME)}")
            conn.execute(
                f"CREATE TABLE {_quote_identifier(TABLE_NAME)} ({', '.join(definitions)})"
            )
            placeholders = ", ".join("?" for _ in columns)
            insert_sql = (
                f"INSERT INTO {_quote_identifier(TABLE_NAME)} "
                f"({', '.join(_quote_identifier(column) for column in columns)}) "
                f"VALUES ({placeholders})"
            )
            values = []
            for run_id, payload in parsed_rows:
                scalars = payload.get("scalars", {})
                values.append(
                    (
                        run_id,
                        *(scalars.get(name) for name in sorted(scalar_names)),
                        json.dumps(payload.get("artifacts", {})),
                        json.dumps(payload.get("meta", {})),
                    )
                )
            conn.executemany(insert_sql, values)

    return len(parsed_rows)


def main() -> None:
    """Parse command-line options and unpack the summary rows."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("db_path", nargs="?", type=Path, default=DEFAULT_DB_PATH)
    args = parser.parse_args()
    count = unpack_summaries(args.db_path)
    print(f"Unpacked {count} summary rows into {TABLE_NAME} in {args.db_path}")


if __name__ == "__main__":
    main()
