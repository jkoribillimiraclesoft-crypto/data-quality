"""
app.py
Streamlit dashboard for the Intelligent Data Quality & Profiling Platform.

Run with: streamlit run app.py
(In Colab: write this file, then run via subprocess + pyngrok/localtunnel
to expose port 8501.)
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import tempfile
import os

from ingestion import load_dataset, infer_schema, dataset_summary
from profiling import run_full_profile
from schema_drift import compare_schemas, detect_data_drift
from storage import (
    init_db, save_profiling_run, save_schema_snapshot,
    get_run_history, get_column_metric_history, get_latest_schema
)
from predictive import (
    predict_quality_score_next_month, predict_duplicate_trend,
    predict_null_trend, predict_volume_growth, predict_columns_likely_to_fail
)

st.set_page_config(page_title="Data Quality & Profiling Platform", layout="wide")
init_db()

# ---------------------------------------------------------------
# Sidebar: logo (top-left) + dataset upload (below logo)
# ---------------------------------------------------------------
with st.sidebar:
    st.image("assets/miracle_logo.png", use_container_width=True)
    st.markdown("---")
    st.subheader("Upload Dataset")
    uploaded_file = st.file_uploader("Upload a dataset", type=["csv", "xlsx", "xls", "json"])
    dataset_name = st.text_input("Dataset name (used to track history)", value="")

st.title("Intelligent Data Quality & Profiling Platform")

if uploaded_file and dataset_name:
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    df = load_dataset(tmp_path)

    # ---------------------------------------------------------------
    # 2. Dataset Summary
    # ---------------------------------------------------------------
    st.header("Dataset Summary")
    summary = dataset_summary(df)
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", summary["row_count"])
    col2.metric("Columns", summary["column_count"])
    col3.metric("Memory (MB)", summary["memory_usage_mb"])

    # ---------------------------------------------------------------
    # 3. Run profiling
    # ---------------------------------------------------------------
    profile = run_full_profile(df)
    schema = infer_schema(df)

    st.header("Data Quality Score")
    st.metric("Overall Quality Score", f"{profile['quality_score']} / 100")

    # ---------------------------------------------------------------
    # 4. Column Health Score (heatmap-style)
    # ---------------------------------------------------------------
    st.header("Column Health Score")
    health_df = profile["column_health"]
    fig = px.bar(health_df, x="column", y="health_score", color="health_score",
                 color_continuous_scale="RdYlGn", range_y=[0, 100])
    st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------------------------
    # 5. Missing Value Heatmap
    # ---------------------------------------------------------------
    st.header("Missing Value Analysis")
    nulls_df = profile["nulls"]
    fig_null = px.bar(nulls_df, x="column", y="null_pct", title="Null % by column")
    st.plotly_chart(fig_null, use_container_width=True)

    # ---------------------------------------------------------------
    # 6. Duplicates
    # ---------------------------------------------------------------
    st.header("Duplicate Analysis")
    st.write(profile["duplicates"])

    # ---------------------------------------------------------------
    # 7. Outlier Distribution
    # ---------------------------------------------------------------
    st.header("Outlier Distribution")
    outliers_df = profile["outliers"]
    if not outliers_df.empty:
        fig_out = px.bar(outliers_df, x="column", y="outlier_pct", title="Outlier % by column")
        st.plotly_chart(fig_out, use_container_width=True)
    else:
        st.info("No numeric columns found for outlier detection.")

    # ---------------------------------------------------------------
    # 8. Statistical Profile / Cardinality / Patterns (tables)
    # ---------------------------------------------------------------
    with st.expander("Statistical Profile"):
        st.dataframe(profile["statistics"])
    with st.expander("Cardinality Analysis"):
        st.dataframe(profile["cardinality"])
    with st.expander("Pattern Validation"):
        st.dataframe(profile["patterns"])

    # ---------------------------------------------------------------
    # 9. Schema Comparison
    # ---------------------------------------------------------------
    st.header("Schema Comparison")
    previous_schema = get_latest_schema(dataset_name)
    if previous_schema:
        diff = compare_schemas(previous_schema, schema)
        st.write(diff)
    else:
        st.info("No previous schema found - this is the first run for this dataset.")

    save_schema_snapshot(dataset_name, schema)

    # ---------------------------------------------------------------
    # 10. Save this run to history
    # ---------------------------------------------------------------
    save_profiling_run(dataset_name, profile, summary["row_count"], summary["column_count"])

    # ---------------------------------------------------------------
    # 11. Historical Quality Trend
    # ---------------------------------------------------------------
    st.header("Historical Quality Trend")
    run_history = get_run_history(dataset_name)
    if len(run_history) > 1:
        fig_trend = px.line(run_history, x="run_timestamp", y="quality_score",
                             title="Quality Score Over Time", markers=True)
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.info("Run this dataset a few more times to build a trend.")

    # ---------------------------------------------------------------
    # 12. Predictive Analytics
    # ---------------------------------------------------------------
    st.header("Predictive Analytics")
    if len(run_history) >= 2:
        pcol1, pcol2, pcol3, pcol4 = st.columns(4)
        pcol1.metric("Predicted Quality Score (next run)", predict_quality_score_next_month(run_history))
        pcol2.metric("Predicted Duplicate %", predict_duplicate_trend(run_history))
        pcol3.metric("Predicted Avg Null %", predict_null_trend(run_history))
        pcol4.metric("Predicted Row Count", predict_volume_growth(run_history))

        col_history = get_column_metric_history(dataset_name)
        at_risk_df = predict_columns_likely_to_fail(col_history)
        st.subheader("Columns Likely to Fail")
        st.dataframe(at_risk_df)
    else:
        st.info("Need at least 2 historical runs of this dataset to generate predictions.")

else:
    st.info("Upload a dataset and provide a dataset name to begin.")
