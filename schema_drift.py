"""
schema_drift.py
Schema version comparison and basic data drift detection between two
snapshots of the same dataset.
"""

import pandas as pd
from scipy.stats import ks_2samp


def compare_schemas(old_schema: dict, new_schema: dict) -> dict:
    """
    Compare two schema dicts (column -> dtype) and report what changed.
    """
    old_cols, new_cols = set(old_schema.keys()), set(new_schema.keys())

    added = list(new_cols - old_cols)
    removed = list(old_cols - new_cols)
    common = old_cols & new_cols

    type_changes = [
        {"column": col, "old_type": old_schema[col], "new_type": new_schema[col]}
        for col in common
        if old_schema[col] != new_schema[col]
    ]

    return {
        "added_columns": added,
        "removed_columns": removed,
        "type_changes": type_changes,
        "has_changes": bool(added or removed or type_changes),
    }


def detect_data_drift(old_df: pd.DataFrame, new_df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """
    For each numeric column present in both dataframes, run a
    Kolmogorov-Smirnov test comparing the two distributions.
    p_value < alpha suggests the distribution has drifted.
    """
    common_numeric_cols = [
        c for c in old_df.columns
        if c in new_df.columns
        and pd.api.types.is_numeric_dtype(old_df[c])
        and pd.api.types.is_numeric_dtype(new_df[c])
    ]

    records = []
    for col in common_numeric_cols:
        old_vals = old_df[col].dropna()
        new_vals = new_df[col].dropna()
        if len(old_vals) < 2 or len(new_vals) < 2:
            continue
        stat, p_value = ks_2samp(old_vals, new_vals)
        records.append(
            {
                "column": col,
                "ks_statistic": round(stat, 4),
                "p_value": round(p_value, 4),
                "drift_detected": p_value < alpha,
            }
        )
    return pd.DataFrame(records)
