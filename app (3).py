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
import plotly.io as pio
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

# =================================================================
# Brand palette (sampled from assets/miracle_logo.png) + chart theme
# =================================================================
BRAND_BLUE = "#00ABE8"
BRAND_DARK = "#1A1A1A"
BRAND_GRAY = "#6B7280"
BG_CARD = "#FFFFFF"
PASS_GREEN = "#1FAA59"
WARN_AMBER = "#F2A93B"
FAIL_RED = "#E5484D"

pio.templates["miracle"] = pio.templates["plotly_white"]
pio.templates["miracle"].layout.update(
    font=dict(family="Segoe UI, Helvetica, Arial, sans-serif", color=BRAND_DARK, size=13),
    colorway=[BRAND_BLUE, BRAND_DARK, WARN_AMBER, PASS_GREEN, FAIL_RED, BRAND_GRAY],
    title_font=dict(size=16, color=BRAND_DARK),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    margin=dict(t=50, l=10, r=10, b=10),
)
pio.templates.default = "miracle"

HEALTH_SCALE = [[0, FAIL_RED], [0.5, WARN_AMBER], [1, PASS_GREEN]]

st.set_page_config(page_title="Data Quality & Profiling Platform", layout="wide", page_icon="\U0001F4CA")

# =================================================================
# Global CSS - card-style metrics, brand color accents, tighter spacing
# =================================================================
st.markdown(f"""
<style>
    .block-container {{ padding-top: 1.5rem; }}

    h1, h2, h3 {{ color: {BRAND_DARK}; font-weight: 700; }}
    h2 {{ border-bottom: 2px solid {BRAND_BLUE}; padding-bottom: 6px; margin-top: 1.8rem; }}

    [data-testid="stMetric"] {{
        background: {BG_CARD};
        border: 1px solid #E5E7EB;
        border-left: 4px solid {BRAND_BLUE};
        border-radius: 8px;
        padding: 14px 16px 10px 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }}
    [data-testid="stMetricLabel"] {{ color: {BRAND_GRAY}; font-size: 0.8rem; }}
    [data-testid="stMetricValue"] {{ color: {BRAND_DARK}; }}

    section[data-testid="stSidebar"] {{
        background: #FAFBFC;
        border-right: 1px solid #E5E7EB;
    }}

    .stTabs [data-baseweb="tab-list"] {{ gap: 4px; }}
    .stTabs [data-baseweb="tab"] {{
        background: #F3F4F6;
        border-radius: 6px 6px 0 0;
        padding: 8px 18px;
        font-weight: 600;
        color: {BRAND_GRAY};
    }}
    .stTabs [aria-selected="true"] {{
        background: {BRAND_BLUE} !important;
        color: white !important;
    }}

    .dq-caption {{
        color: {BRAND_GRAY};
        font-size: 0.85rem;
        margin-top: -8px;
        margin-bottom: 12px;
    }}
</style>
""", unsafe_allow_html=True)

init_db()

# ---------------------------------------------------------------
# Sidebar: logo (top-left), dataset upload, and rule count slider
# ---------------------------------------------------------------
with st.sidebar:
    st.image("assets/miracle_logo.png", use_container_width=True)
    st.markdown("---")
    st.markdown("##### \U0001F4C1 Upload Dataset")
    uploaded_file = st.file_uploader("Upload a dataset", type=["csv", "xlsx", "xls", "json"], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("##### \U0001F39B\uFE0F DQ Rule Selection")
    full_rule_catalog = load_rule_catalog("rule_catalog.json")
    total_available_rules = len(full_rule_catalog)
    rule_count = st.slider(
        "Number of DQ rules to apply",
        min_value=1,
        max_value=total_available_rules,
        value=total_available_rules,
        label_visibility="visible",
    )
    st.caption(f"Applying **{rule_count}** of **{total_available_rules}** available rules.")

st.title("Intelligent Data Quality & Profiling Platform")
st.caption("Enterprise data quality assessment · profiling · rule engine · predictive trends")

if uploaded_file:
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    df = load_dataset(tmp_path)
    schema = infer_schema(df)

    # Dataset is identified by a schema-based fingerprint, not a typed name.
    dataset_id = generate_dataset_fingerprint(schema)
    dataset_label = uploaded_file.name

    st.markdown(
        f"<div class='dq-caption'>Dataset: <b>{dataset_label}</b> &nbsp;|&nbsp; "
        f"Dataset ID (auto-detected from schema): <code>{dataset_id}</code></div>",
        unsafe_allow_html=True,
    )

    profile = run_full_profile(df)
    summary = dataset_summary(df)
    active_rules = select_rules(full_rule_catalog, rule_count)
    rule_results = apply_rules(df, active_rules)
    score = overall_rule_score(rule_results)
    dim_summary = summarize_by_dimension(rule_results)
    exceptions_df = get_exceptions(rule_results, df)

    tab_overview, tab_rules, tab_profile, tab_trends = st.tabs(
        ["\U0001F4CB Overview", "\u2705 DQ Rules & Exceptions", "\U0001F50D Profiling Details", "\U0001F4C8 Trends & Predictions"]
    )

    # =============================================================
    # TAB 1: Overview
    # =============================================================
    with tab_overview:
        st.subheader("Dataset Summary")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", f"{summary['row_count']:,}")
        col2.metric("Columns", summary["column_count"])
        col3.metric("Memory (MB)", summary["memory_usage_mb"])
        col4.metric("Overall Quality Score", f"{profile['quality_score']}/100")

        st.subheader("Column Health Score")
        health_df = profile["column_health"]
        fig = px.bar(health_df, x="column", y="health_score", color="health_score",
                     color_continuous_scale=HEALTH_SCALE, range_y=[0, 100])
        fig.update_layout(coloraxis_showscale=False, xaxis_title="", yaxis_title="Health Score")
        st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Missing Values")
            nulls_df = profile["nulls"]
            fig_null = px.bar(nulls_df, x="column", y="null_pct")
            fig_null.update_traces(marker_color=BRAND_BLUE)
            fig_null.update_layout(xaxis_title="", yaxis_title="Null %")
            st.plotly_chart(fig_null, use_container_width=True)
        with c2:
            st.subheader("Outlier Distribution")
            outliers_df = profile["outliers"]
            if not outliers_df.empty:
                fig_out = px.bar(outliers_df, x="column", y="outlier_pct")
                fig_out.update_traces(marker_color=WARN_AMBER)
                fig_out.update_layout(xaxis_title="", yaxis_title="Outlier %")
                st.plotly_chart(fig_out, use_container_width=True)
            else:
                st.info("No numeric columns found for outlier detection.")

        st.subheader("Duplicate Analysis")
        dup = profile["duplicates"]
        dcol1, dcol2 = st.columns(2)
        dcol1.metric("Duplicate Rows", dup["duplicate_rows"])
        dcol2.metric("Duplicate %", f"{dup['duplicate_pct']}%")

    # =============================================================
    # TAB 2: DQ Rules & Exceptions
    # =============================================================
    with tab_rules:
        st.subheader("Rule Execution Summary")
        rcol1, rcol2, rcol3, rcol4, rcol5 = st.columns(5)
        rcol1.metric("Rules Executed", score["rules_executed"])
        rcol2.metric("Passed", score["passed"])
        rcol3.metric("Failed", score["failed"])
        rcol4.metric("Warnings", score["warnings"])
        rcol5.metric("Critical Issues", score["critical_issues"])

        st.subheader("DQ Dimensions")
        if not dim_summary.empty:
            fig_dim = px.bar(dim_summary, x="dimension", y="pass_rate_pct",
                              color="pass_rate_pct", color_continuous_scale=HEALTH_SCALE,
                              range_y=[0, 100], text="pass_rate_pct")
            fig_dim.update_traces(texttemplate="%{text}%", textposition="outside")
            fig_dim.update_layout(coloraxis_showscale=False, xaxis_title="", yaxis_title="Pass Rate %")
            st.plotly_chart(fig_dim, use_container_width=True)
            with st.expander("View dimension detail table"):
                st.dataframe(dim_summary, use_container_width=True)
        else:
            st.info("No rules matched columns in this dataset.")

        st.subheader("Exceptions (Failed / Warning Records)")
        if not exceptions_df.empty:
            st.dataframe(exceptions_df, use_container_width=True, height=320)
            st.caption(f"{len(exceptions_df)} rule violations found across {exceptions_df['record_id'].nunique()} records.")
        else:
            st.success("No rule violations found.")

    # =============================================================
    # TAB 3: Profiling Details
    # =============================================================
    with tab_profile:
        st.subheader("Statistical Profile")
        st.dataframe(profile["statistics"], use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Cardinality Analysis")
            st.dataframe(profile["cardinality"], use_container_width=True)
        with c2:
            st.subheader("Pattern Validation")
            st.dataframe(profile["patterns"], use_container_width=True)

        st.subheader("Schema Comparison")
        previous_schema = get_latest_schema(dataset_id)
        if previous_schema:
            diff = compare_schemas(previous_schema, schema)
            if diff["has_changes"]:
                d1, d2, d3 = st.columns(3)
                d1.metric("Columns Added", len(diff["added_columns"]))
                d2.metric("Columns Removed", len(diff["removed_columns"]))
                d3.metric("Type Changes", len(diff["type_changes"]))
                st.json(diff)
            else:
                st.success("No schema changes since the last run.")
        else:
            st.info("No previous schema found - this is the first run for this dataset.")

    save_schema_snapshot(dataset_id, dataset_label, schema)
    save_profiling_run(dataset_id, dataset_label, profile, summary["row_count"], summary["column_count"])

    # =============================================================
    # TAB 4: Trends & Predictions
    # =============================================================
    with tab_trends:
        run_history = get_run_history(dataset_id)

        st.subheader("Historical Quality Trend")
        if len(run_history) > 1:
            fig_trend = px.line(run_history, x="run_timestamp", y="quality_score", markers=True)
            fig_trend.update_traces(line_color=BRAND_BLUE, marker_color=BRAND_DARK)
            fig_trend.update_layout(xaxis_title="", yaxis_title="Quality Score")
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("Run this dataset a few more times to build a trend.")

        st.subheader("Predictive Analytics")
        if len(run_history) >= 2:
            pcol1, pcol2, pcol3, pcol4 = st.columns(4)
            pcol1.metric("Predicted Quality Score", predict_quality_score_next_month(run_history))
            pcol2.metric("Predicted Duplicate %", predict_duplicate_trend(run_history))
            pcol3.metric("Predicted Avg Null %", predict_null_trend(run_history))
            pcol4.metric("Predicted Row Count", predict_volume_growth(run_history))

            col_history = get_column_metric_history(dataset_id)
            at_risk_df = predict_columns_likely_to_fail(col_history)
            st.subheader("Columns Likely to Fail")
            st.dataframe(at_risk_df, use_container_width=True)
        else:
            st.info("Need at least 2 historical runs of this dataset to generate predictions.")

else:
    st.info("Upload a dataset from the sidebar to begin.")
