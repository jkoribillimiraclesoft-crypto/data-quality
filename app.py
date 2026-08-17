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

from ingestion import load_dataset, infer_schema, dataset_summary, generate_dataset_fingerprint
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
from rule_engine import (
    load_rule_catalog, select_rules, apply_rules, summarize_by_dimension,
    get_exceptions, overall_rule_score
)

st.set_page_config(page_title="Data Quality & Profiling Platform", layout="wide")
init_db()

# ---------------------------------------------------------------
# Sidebar: logo (top-left), dataset upload, and rule count slider
# ---------------------------------------------------------------
with st.sidebar:
    st.image("assets/miracle_logo.png", use_container_width=True)
    st.markdown("---")
    st.subheader("Upload Dataset")
    uploaded_file = st.file_uploader("Upload a dataset", type=["csv", "xlsx", "xls", "json"])

    st.markdown("---")
    st.subheader("DQ Rule Selection")
    full_rule_catalog = load_rule_catalog("rule_catalog.json")
    total_available_rules = len(full_rule_catalog)
    rule_count = st.slider(
        "Number of DQ rules to apply",
        min_value=1,
        max_value=total_available_rules,
        value=total_available_rules,
    )
    st.caption(f"Applying {rule_count} of {total_available_rules} available rules.")

st.title("Intelligent Data Quality & Profiling Platform")

if uploaded_file:
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    df = load_dataset(tmp_path)
    schema = infer_schema(df)

    # Dataset is identified by a schema-based fingerprint, not a typed name.
    # The uploaded file name is kept only as a human-readable label.
    dataset_id = generate_dataset_fingerprint(schema)
    dataset_label = uploaded_file.name

    st.caption(f"Dataset: **{dataset_label}**  |  Dataset ID (auto-detected from schema): `{dataset_id}`")

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

    st.header("Data Quality Score")
    st.metric("Overall Quality Score", f"{profile['quality_score']} / 100")

    # ---------------------------------------------------------------
    # 3b. Rule-Based DQ Results (metadata-driven rule engine)
    # Only the rule_count rules selected via the sidebar slider are applied.
    # ---------------------------------------------------------------
    st.header("DQ Rule Results")
    active_rules = select_rules(full_rule_catalog, rule_count)
    rule_results = apply_rules(df, active_rules)
    score = overall_rule_score(rule_results)

    rcol1, rcol2, rcol3, rcol4, rcol5 = st.columns(5)
    rcol1.metric("Rules Executed", score["rules_executed"])
    rcol2.metric("Passed", score["passed"])
    rcol3.metric("Failed", score["failed"])
    rcol4.metric("Warnings", score["warnings"])
    rcol5.metric("Critical Issues", score["critical_issues"])

    st.subheader("DQ Dimensions")
    dim_summary = summarize_by_dimension(rule_results)
    if not dim_summary.empty:
        fig_dim = px.bar(dim_summary, x="dimension", y="pass_rate_pct",
                          color="pass_rate_pct", color_continuous_scale="RdYlGn",
                          range_y=[0, 100], title="Pass Rate % by DQ Dimension")
        st.plotly_chart(fig_dim, use_container_width=True)
        st.dataframe(dim_summary, use_container_width=True)
    else:
        st.info("No rules matched columns in this dataset.")

    st.subheader("Exceptions (Failed / Warning Records)")
    exceptions_df = get_exceptions(rule_results, df)
    if not exceptions_df.empty:
        st.dataframe(exceptions_df, use_container_width=True)
        st.caption(f"{len(exceptions_df)} rule violations found across {exceptions_df['record_id'].nunique()} records.")
    else:
        st.success("No rule violations found.")

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
    previous_schema = get_latest_schema(dataset_id)
    if previous_schema:
        diff = compare_schemas(previous_schema, schema)
        st.write(diff)
    else:
        st.info("No previous schema found - this is the first run for this dataset.")

    save_schema_snapshot(dataset_id, dataset_label, schema)

    # ---------------------------------------------------------------
    # 10. Save this run to history
    # ---------------------------------------------------------------
    save_profiling_run(dataset_id, dataset_label, profile, summary["row_count"], summary["column_count"])

    # ---------------------------------------------------------------
    # 11. Historical Quality Trend
    # ---------------------------------------------------------------
    st.header("Historical Quality Trend")
    run_history = get_run_history(dataset_id)
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

        col_history = get_column_metric_history(dataset_id)
        at_risk_df = predict_columns_likely_to_fail(col_history)
        st.subheader("Columns Likely to Fail")
        st.dataframe(at_risk_df)
    else:
        st.info("Need at least 2 historical runs of this dataset to generate predictions.")

else:
    st.info("Upload a dataset to begin.")
