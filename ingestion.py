"""
ingestion.py
Handles loading CSV/Excel/JSON files, schema inference, and sampling
for large datasets.
"""

import pandas as pd
import os


def load_dataset(file_path: str, sample_size: int = 200_000) -> pd.DataFrame:
    """
    Load a dataset from CSV, Excel, or JSON.
    If the file has more rows than sample_size, a random sample is taken
    (profiling is then run on the sample, not the full file).
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".csv":
        # Count rows cheaply first (without loading full file) to decide on sampling
        row_count = sum(1 for _ in open(file_path, "r", encoding="utf-8", errors="ignore")) - 1
        if row_count > sample_size:
            # Sample by skipping rows randomly
            skip = sorted(
                pd.Series(range(1, row_count + 1)).sample(
                    row_count - sample_size, random_state=42
                )
            )
            df = pd.read_csv(file_path, skiprows=skip)
        else:
            df = pd.read_csv(file_path)

    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(file_path)
        if len(df) > sample_size:
            df = df.sample(sample_size, random_state=42)

    elif ext == ".json":
        df = pd.read_json(file_path)
        if len(df) > sample_size:
            df = df.sample(sample_size, random_state=42)

    else:
        raise ValueError(f"Unsupported file type: {ext}")

    return df.reset_index(drop=True)


def infer_schema(df: pd.DataFrame) -> dict:
    """
    Return a dict describing column name -> inferred dtype (as string).
    Used for schema comparison across dataset versions.
    """
    schema = {}
    for col in df.columns:
        dtype = str(df[col].dtype)
        schema[col] = dtype
    return schema


def dataset_summary(df: pd.DataFrame) -> dict:
    """High-level summary shown on the dashboard's Dataset Summary panel."""
    return {
        "row_count": len(df),
        "column_count": len(df.columns),
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
        "columns": list(df.columns),
    }
