"""
storage.py
SQLite-based persistence for profiling run history and schema snapshots.
This is what powers the historical trend charts and predictive analytics.

Tracking key: dataset_id (a schema-based fingerprint, see
ingestion.generate_dataset_fingerprint) - NOT a user-typed dataset name.
dataset_label is kept purely for display (auto-derived from the uploaded
file name) and has no bearing on how history is matched up.

NOTE on schema drift: dataset_id is itself derived from the schema, so
looking up "the previous schema for this dataset_id" can only ever find
a schema identical to the current one (or find nothing at all, if the
schema has never been seen before). Detecting an actual schema CHANGE
therefore requires a lookup keyed on something schema-independent -
dataset_label - which is what get_latest_schema_by_label() is for.
Run-history tracking (dataset_id-based) is intentionally left alone.
"""

import sqlite3
import json
from datetime import datetime

DB_PATH = "dq_platform.db"


def init_db(db_path: str = DB_PATH):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS profiling_runs (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id TEXT NOT NULL,
            dataset_label TEXT,
            run_timestamp TEXT NOT NULL,
            row_count INTEGER,
            column_count INTEGER,
            quality_score REAL,
            completeness_pct REAL,
            duplicate_pct REAL,
            avg_null_pct REAL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS column_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER,
            column_name TEXT,
            null_pct REAL,
            outlier_pct REAL,
            health_score REAL,
            FOREIGN KEY (run_id) REFERENCES profiling_runs(run_id)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS schema_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id TEXT NOT NULL,
            dataset_label TEXT,
            snapshot_timestamp TEXT NOT NULL,
            schema_json TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


def save_profiling_run(dataset_id: str, dataset_label: str, profile: dict, row_count: int, column_count: int, db_path: str = DB_PATH) -> int:
    """
    Save one profiling run + its per-column metrics.
    `profile` is the dict returned by profiling.run_full_profile().
    Returns the run_id.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    nulls_df = profile["nulls"]
    avg_null_pct = round(nulls_df["null_pct"].mean(), 2) if not nulls_df.empty else 0.0

    cur.execute("""
        INSERT INTO profiling_runs
        (dataset_id, dataset_label, run_timestamp, row_count, column_count, quality_score, completeness_pct, duplicate_pct, avg_null_pct)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        dataset_id,
        dataset_label,
        datetime.utcnow().isoformat(),
        row_count,
        column_count,
        profile["quality_score"],
        profile["completeness_pct"],
        profile["duplicates"]["duplicate_pct"],
        avg_null_pct,
    ))
    run_id = cur.lastrowid

    health_df = profile["column_health"].merge(
        nulls_df[["column", "null_pct"]], on="column", how="left"
    )
    outliers_df = profile["outliers"]

    for _, row in health_df.iterrows():
        outlier_pct = 0.0
        if not outliers_df.empty and row["column"] in outliers_df["column"].values:
            outlier_pct = outliers_df.loc[outliers_df["column"] == row["column"], "outlier_pct"].values[0]

        cur.execute("""
            INSERT INTO column_metrics (run_id, column_name, null_pct, outlier_pct, health_score)
            VALUES (?, ?, ?, ?, ?)
        """, (run_id, row["column"], row["null_pct"], outlier_pct, row["health_score"]))

    conn.commit()
    conn.close()
    return run_id


def save_schema_snapshot(dataset_id: str, dataset_label: str, schema: dict, db_path: str = DB_PATH):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO schema_snapshots (dataset_id, dataset_label, snapshot_timestamp, schema_json)
        VALUES (?, ?, ?, ?)
    """, (dataset_id, dataset_label, datetime.utcnow().isoformat(), json.dumps(schema)))
    conn.commit()
    conn.close()


def get_run_history(dataset_id: str, db_path: str = DB_PATH):
    import pandas as pd
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(
        "SELECT * FROM profiling_runs WHERE dataset_id = ? ORDER BY run_timestamp",
        conn, params=(dataset_id,)
    )
    conn.close()
    return df


def get_column_metric_history(dataset_id: str, db_path: str = DB_PATH):
    import pandas as pd
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("""
        SELECT r.run_timestamp, r.dataset_id, c.column_name, c.null_pct, c.outlier_pct, c.health_score
        FROM column_metrics c
        JOIN profiling_runs r ON c.run_id = r.run_id
        WHERE r.dataset_id = ?
        ORDER BY r.run_timestamp
    """, conn, params=(dataset_id,))
    conn.close()
    return df


def get_latest_schema(dataset_id: str, db_path: str = DB_PATH):
    """
    Latest schema snapshot for an EXACT dataset_id match. Since dataset_id
    is itself derived from the schema, this can only ever return a schema
    identical to the one you'd compare it against - useful for "have I
    profiled this exact shape of data before", not for drift detection.
    Kept as-is; see get_latest_schema_by_label() for drift detection.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        SELECT schema_json FROM schema_snapshots
        WHERE dataset_id = ? ORDER BY snapshot_timestamp DESC LIMIT 1
    """, (dataset_id,))
    row = cur.fetchone()
    conn.close()
    return json.loads(row[0]) if row else None


def get_latest_schema_by_label(dataset_label: str, exclude_dataset_id: str = None, db_path: str = DB_PATH):
    """
    Latest schema snapshot for a given dataset_label (the uploaded file's
    name), optionally excluding a specific dataset_id - used to exclude
    the current run's own dataset_id so a same-schema re-upload doesn't
    just "find itself" and mask a real prior version under the same
    label. This is what actually enables schema DRIFT detection: it
    finds the most recent snapshot filed under the same label regardless
    of whether that snapshot's schema matches the current one.

    Returns (schema_dict, dataset_id_of_that_snapshot) or (None, None).
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    if exclude_dataset_id is not None:
        cur.execute("""
            SELECT schema_json, dataset_id FROM schema_snapshots
            WHERE dataset_label = ? AND dataset_id != ?
            ORDER BY snapshot_timestamp DESC LIMIT 1
        """, (dataset_label, exclude_dataset_id))
    else:
        cur.execute("""
            SELECT schema_json, dataset_id FROM schema_snapshots
            WHERE dataset_label = ?
            ORDER BY snapshot_timestamp DESC LIMIT 1
        """, (dataset_label,))
    row = cur.fetchone()
    conn.close()
    if row:
        return json.loads(row[0]), row[1]
    return None, None


def list_known_datasets(db_path: str = DB_PATH):
    """
    Returns the distinct datasets seen so far (dataset_id + most recent
    label + last run time) - useful for a future 'History' page where
    users pick a past dataset rather than needing to know its ID.
    """
    import pandas as pd
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("""
        SELECT dataset_id, dataset_label, MAX(run_timestamp) as last_run, COUNT(*) as run_count
        FROM profiling_runs
        GROUP BY dataset_id
        ORDER BY last_run DESC
    """, conn)
    conn.close()
    return df
