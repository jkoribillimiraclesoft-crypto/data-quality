"""
predictive.py
Lightweight predictive analytics built on historical profiling runs.
Uses simple linear regression (scikit-learn) - sufficient for POC-level
trend forecasting, not intended to be production-grade forecasting.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def _fit_trend(history_df: pd.DataFrame, value_col: str):
    """
    Fit a simple linear trend: value_col vs run index (time order).
    Returns (model, last_index) or (None, None) if not enough data.
    """
    if len(history_df) < 2:
        return None, None

    x = np.arange(len(history_df)).reshape(-1, 1)
    y = history_df[value_col].values
    model = LinearRegression().fit(x, y)
    return model, len(history_df) - 1


def forecast_next_value(history_df: pd.DataFrame, value_col: str, steps_ahead: int = 1) -> float:
    """
    Predict the value of `value_col` `steps_ahead` runs into the future,
    based on the linear trend of past runs.
    """
    model, last_index = _fit_trend(history_df, value_col)
    if model is None:
        return float(history_df[value_col].iloc[-1]) if len(history_df) else None

    next_x = np.array([[last_index + steps_ahead]])
    prediction = model.predict(next_x)[0]
    return round(float(prediction), 2)


def predict_quality_score_next_month(run_history_df: pd.DataFrame) -> float:
    """Assumes run_history_df is ordered chronologically with a 'quality_score' column."""
    return forecast_next_value(run_history_df, "quality_score")


def predict_duplicate_trend(run_history_df: pd.DataFrame) -> float:
    return forecast_next_value(run_history_df, "duplicate_pct")


def predict_null_trend(run_history_df: pd.DataFrame) -> float:
    return forecast_next_value(run_history_df, "avg_null_pct")


def predict_volume_growth(run_history_df: pd.DataFrame) -> float:
    return forecast_next_value(run_history_df, "row_count")


def predict_columns_likely_to_fail(column_history_df: pd.DataFrame, health_threshold: float = 60.0) -> pd.DataFrame:
    """
    For each column, fit a trend on health_score over time and flag columns
    whose projected next health_score falls below health_threshold.
    Expects column_history_df with columns: column_name, health_score, run_timestamp
    (ordered chronologically - e.g. from storage.get_column_metric_history).
    """
    results = []
    for col_name, group in column_history_df.groupby("column_name"):
        group = group.sort_values("run_timestamp")
        predicted = forecast_next_value(group, "health_score")
        if predicted is not None:
            results.append({
                "column": col_name,
                "current_health_score": round(group["health_score"].iloc[-1], 2),
                "predicted_next_health_score": predicted,
                "at_risk": predicted < health_threshold,
            })
    return pd.DataFrame(results).sort_values("predicted_next_health_score")
