"""
profiling.py
Core data profiling logic: null analysis, duplicate detection, pattern
validation, cardinality, statistical profiling, outlier detection,
completeness, and an overall quality score.
"""

import pandas as pd
import numpy as np
import re

EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
PHONE_PATTERN = re.compile(r"^\+?\d{7,15}$")


def null_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Null count and percentage per column."""
    nulls = df.isnull().sum()
    pct = (nulls / len(df) * 100).round(2)
    return pd.DataFrame({"null_count": nulls, "null_pct": pct}).reset_index().rename(
        columns={"index": "column"}
    )


def duplicate_detection(df: pd.DataFrame) -> dict:
    """Full-row duplicates and duplicate percentage."""
    dup_count = int(df.duplicated().sum())
    return {
        "duplicate_rows": dup_count,
        "duplicate_pct": round(dup_count / len(df) * 100, 2) if len(df) else 0.0,
    }


def cardinality_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Unique value count and ratio per column (helps flag ID-like vs category-like columns)."""
    records = []
    for col in df.columns:
        unique_count = df[col].nunique(dropna=True)
        ratio = round(unique_count / len(df), 4) if len(df) else 0
        records.append({"column": col, "unique_count": unique_count, "unique_ratio": ratio})
    return pd.DataFrame(records)


def pattern_validation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Checks columns whose name suggests a known pattern (email, phone) and
    reports what percentage of non-null values match the expected pattern.
    """
    results = []
    for col in df.columns:
        col_lower = col.lower()
        series = df[col].dropna().astype(str)
        if len(series) == 0:
            continue
        if "email" in col_lower:
            match_pct = round(series.str.match(EMAIL_PATTERN).mean() * 100, 2)
            results.append({"column": col, "pattern": "email", "match_pct": match_pct})
        elif "phone" in col_lower:
            match_pct = round(series.str.match(PHONE_PATTERN).mean() * 100, 2)
            results.append({"column": col, "pattern": "phone", "match_pct": match_pct})
    return pd.DataFrame(results)


def statistical_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Mean/median/std/min/max for numeric columns."""
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return pd.DataFrame()
    stats = numeric_df.describe().T[["mean", "std", "min", "max"]]
    stats["median"] = numeric_df.median()
    return stats.reset_index().rename(columns={"index": "column"})


def outlier_detection(df: pd.DataFrame, method: str = "iqr") -> pd.DataFrame:
    """
    Outlier count per numeric column using IQR (default) or z-score method.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    records = []
    for col in numeric_df.columns:
        series = numeric_df[col].dropna()
        if len(series) == 0:
            continue
        if method == "iqr":
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outliers = series[(series < lower) | (series > upper)]
        else:  # z-score
            z_scores = (series - series.mean()) / series.std(ddof=0)
            outliers = series[z_scores.abs() > 3]
        records.append(
            {
                "column": col,
                "outlier_count": len(outliers),
                "outlier_pct": round(len(outliers) / len(series) * 100, 2),
            }
        )
    return pd.DataFrame(records)


def completeness_score(df: pd.DataFrame) -> float:
    """Overall % of non-null cells across the whole dataset."""
    total_cells = df.shape[0] * df.shape[1]
    if total_cells == 0:
        return 0.0
    non_null_cells = total_cells - df.isnull().sum().sum()
    return round(non_null_cells / total_cells * 100, 2)


def column_health_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-column health score (0-100), combining null pct and outlier pct.
    Simple weighted average - not a statistically rigorous measure, just
    a POC-level indicator to drive the dashboard heatmap.
    """
    nulls = null_analysis(df).set_index("column")["null_pct"]
    outliers = outlier_detection(df).set_index("column")["outlier_pct"] if not outlier_detection(df).empty else pd.Series(dtype=float)

    records = []
    for col in df.columns:
        null_pct = nulls.get(col, 0)
        outlier_pct = outliers.get(col, 0)
        score = 100 - (null_pct * 0.7 + outlier_pct * 0.3)
        records.append({"column": col, "health_score": round(max(score, 0), 2)})
    return pd.DataFrame(records)


def overall_quality_score(df: pd.DataFrame) -> float:
    """
    Single 0-100 score summarizing dataset health.
    Weighted blend of completeness, duplicate rate, and average column health.
    """
    completeness = completeness_score(df)
    dup_pct = duplicate_detection(df)["duplicate_pct"]
    health_df = column_health_score(df)
    avg_health = health_df["health_score"].mean() if not health_df.empty else 100

    score = (completeness * 0.4) + ((100 - dup_pct) * 0.2) + (avg_health * 0.4)
    return round(score, 2)


def run_full_profile(df: pd.DataFrame) -> dict:
    """Convenience function: runs every profiling check and returns one dict."""
    return {
        "nulls": null_analysis(df),
        "duplicates": duplicate_detection(df),
        "cardinality": cardinality_analysis(df),
        "patterns": pattern_validation(df),
        "statistics": statistical_profile(df),
        "outliers": outlier_detection(df),
        "completeness_pct": completeness_score(df),
        "column_health": column_health_score(df),
        "quality_score": overall_quality_score(df),
    }
