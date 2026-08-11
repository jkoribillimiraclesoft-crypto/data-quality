"""
storage.py
SQLite-based persistence for profiling run history and schema snapshots.
This is what powers the historical trend charts and predictive analytics.
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
            dataset_name TEXT NOT NULL,
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
            dataset_name TEXT NOT NULL,
            snapshot_timestamp TEXT NOT NULL,
            schema_json TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


def save_profiling_run(dataset_name: str, profile: dict, row_count: int, column_count: int, db_path: str = DB_PATH) -> int:
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
        (dataset_name, run_timestamp, row_count, column_count, quality_score, completeness_pct, duplicate_pct, avg_null_pct)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        dataset_name,
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


def save_schema_snapshot(dataset_name: str, schema: dict, db_path: str = DB_PATH):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO schema_snapshots (dataset_name, snapshot_timestamp, schema_json)
        VALUES (?, ?, ?)
    """, (dataset_name, datetime.utcnow().isoformat(), json.dumps(schema)))
    conn.commit()
    conn.close()


def get_run_history(dataset_name: str, db_path: str = DB_PATH):
    import pandas as pd
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(
        "SELECT * FROM profiling_runs WHERE dataset_name = ? ORDER BY run_timestamp",
        conn, params=(dataset_name,)
    )
    conn.close()
    return df


def get_column_metric_history(dataset_name: str, db_path: str = DB_PATH):
    import pandas as pd
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("""
        SELECT r.run_timestamp, r.dataset_name, c.column_name, c.null_pct, c.outlier_pct, c.health_score
        FROM column_metrics c
        JOIN profiling_runs r ON c.run_id = r.run_id
        WHERE r.dataset_name = ?
        ORDER BY r.run_timestamp
    """, conn, params=(dataset_name,))
    conn.close()
    return df


def get_latest_schema(dataset_name: str, db_path: str = DB_PATH):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        SELECT schema_json FROM schema_snapshots
        WHERE dataset_name = ? ORDER BY snapshot_timestamp DESC LIMIT 1
    """, (dataset_name,))
    row = cur.fetchone()
    conn.close()
    return json.loads(row[0]) if row else None
