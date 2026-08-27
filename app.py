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
import plotly.graph_objects as go
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

HEALTH_SCALE = [[0, FAIL_RED], [1, PASS_GREEN]]  # higher value = better -> red to green

st.set_page_config(page_title="Data Quality & Profiling Platform", layout="wide", page_icon="\U0001F4CA")

# =================================================================
# Global CSS - card-style metrics, brand color accents, tighter spacing,
# and RESPONSIVE tabs (wrap instead of overflow/scroll on narrow screens)
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
    [data-testid="stMetricLabel"] {{
        color: {BRAND_GRAY};
        font-size: clamp(0.65rem, 1.1vw, 0.8rem);
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }}
    [data-testid="stMetricValue"] {{
        color: {BRAND_DARK};
        font-size: clamp(1rem, 1.8vw, 1.5rem);
        white-space: nowrap;
    }}
    [data-testid="stMetricDelta"] {{
        white-space: nowrap;
        font-size: clamp(0.65rem, 1vw, 0.8rem);
    }}

    section[data-testid="stSidebar"] {{
        background: #FAFBFC;
        border-right: 1px solid #E5E7EB;
    }}

    /* Responsive tabs: wrap onto multiple lines on narrow screens instead
       of overflowing or requiring horizontal scroll. Text shrinks slightly
       before it wraps, so labels stay fully visible at all widths. */
    div.stTabs [data-baseweb="tab-list"] {{
        display: flex !important;
        flex-wrap: wrap !important;
        row-gap: 6px;
        column-gap: 4px;
        width: 100% !important;
    }}
    div.stTabs [data-baseweb="tab-list"] button[data-baseweb="tab"] {{
        background: #F3F4F6 !important;
        border-radius: 6px 6px 0 0 !important;
        padding: 14px 16px !important;
        font-weight: 600 !important;
        color: {BRAND_GRAY} !important;
        flex: 1 1 0 !important;
        max-width: none !important;
        display: flex !important;
        justify-content: center !important;
        white-space: nowrap;
        height: auto !important;
    }}
    div.stTabs [data-baseweb="tab-list"] button[data-baseweb="tab"] p {{
        font-size: 1.15rem !important;
        font-weight: 600 !important;
    }}
    div.stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {{
        background: {BRAND_BLUE} !important;
        color: white !important;
    }}
    div.stTabs [data-baseweb="tab-list"] button[aria-selected="true"] p {{
        color: white !important;
    }}
    div.stTabs [data-baseweb="tab-highlight"] {{
        display: none !important;
    }}
    @media (max-width: 640px) {{
        div.stTabs [data-baseweb="tab-list"] button[data-baseweb="tab"] {{
            padding: 8px 10px !important;
            white-space: normal;
            text-align: center;
        }}
        div.stTabs [data-baseweb="tab-list"] button[data-baseweb="tab"] p {{
            font-size: 0.9rem !important;
        }}
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

        # -----------------------------------------------------------
        # Wide-dataset handling: past this many columns, vertical bars
        # / rotated x-axis labels become unreadable. Past the threshold
        # we switch to horizontal orientation (labels on the y-axis,
        # never rotated, chart grows downward instead of squeezing
        # sideways) and default to showing only the worst offenders,
        # with an option to expand to the full column list.
        # -----------------------------------------------------------
        WIDE_THRESHOLD = 15
        n_cols_total = len(health_df := profile["column_health"])
        is_wide = n_cols_total > WIDE_THRESHOLD

        # Continuous red-to-green colorscale, reversed: here a HIGH value
        # (more nulls / more outliers) is BAD, so red sits at the high end
        # and green at the low end - opposite of HEALTH_SCALE.
        RISK_SCALE = [[0, PASS_GREEN], [1, FAIL_RED]]

        def _limited(df, value_col, ascending, key, slider_label):
            """For wide datasets, let the user cap how many columns show
            (worst-first) instead of dumping everything into one chart."""
            if not is_wide or len(df) <= WIDE_THRESHOLD:
                return df.sort_values(value_col, ascending=ascending)
            max_n = len(df)
            default_n = min(20, max_n)
            top_n = st.slider(
                slider_label, min_value=5, max_value=max_n,
                value=default_n, key=f"topn_{key}",
            )
            return df.sort_values(value_col, ascending=ascending).head(top_n)

        def _dyn_height(n):
            return max(320, min(1400, 28 * n))

        st.subheader("Column Health Score")
        health_plot_df = _limited(
            health_df, "health_score", ascending=True, key="health",
            slider_label="Number of columns to display, starting with the lowest health scores",
        )
        if is_wide:
            health_sorted = health_plot_df.sort_values("health_score")
            fig = go.Figure(go.Bar(
                x=health_sorted["health_score"], y=health_sorted["column"], orientation="h",
                marker=dict(color=health_sorted["health_score"], colorscale=HEALTH_SCALE, cmin=0, cmax=100),
            ))
            fig.update_layout(xaxis=dict(range=[0, 100], title="Health Score"), yaxis_title="",
                               height=_dyn_height(len(health_sorted)), margin=dict(l=10, r=20, t=20, b=30))
        else:
            fig = px.bar(health_plot_df, x="column", y="health_score", color="health_score",
                         color_continuous_scale=HEALTH_SCALE, range_y=[0, 100])
            fig.update_layout(coloraxis_showscale=False, xaxis_title="", yaxis_title="Health Score")
        st.plotly_chart(fig, use_container_width=True)
        if is_wide:
            with st.expander(f"View all {n_cols_total} columns (table)"):
                st.dataframe(health_df.sort_values("health_score"), use_container_width=True)

        if is_wide:
            # Stack full-width when crowded rather than squeezing two
            # charts side by side, which halves the usable width.
            st.subheader("Missing Values")
            nulls_df = profile["nulls"]
            nulls_plot_df = _limited(
                nulls_df, "null_pct", ascending=False, key="nulls",
                slider_label="Number of columns to display, starting with the highest missing-value %",
            )
            nulls_sorted = nulls_plot_df.sort_values("null_pct")
            fig_null = go.Figure(go.Bar(
                x=nulls_sorted["null_pct"], y=nulls_sorted["column"], orientation="h",
                marker=dict(color=nulls_sorted["null_pct"], colorscale=RISK_SCALE, cmin=0, cmax=100),
            ))
            fig_null.update_layout(xaxis=dict(range=[0, 100], title="Null %"), yaxis_title="",
                                    height=_dyn_height(len(nulls_sorted)), margin=dict(l=10, r=20, t=20, b=30))
            st.plotly_chart(fig_null, use_container_width=True)
            st.caption("Color scale: \U0001F7E2 green = low missing %  \u2192  \U0001F534 red = high missing %")
            with st.expander(f"View all {len(nulls_df)} columns (table)"):
                st.dataframe(nulls_df.sort_values("null_pct", ascending=False), use_container_width=True)

            st.subheader("Outlier Distribution")
            outliers_df = profile["outliers"]
            if not outliers_df.empty:
                outliers_plot_df = _limited(
                    outliers_df, "outlier_pct", ascending=False, key="outliers",
                    slider_label="Number of columns to display, starting with the highest outlier %",
                )
                outliers_sorted = outliers_plot_df.sort_values("outlier_pct")
                fig_out = go.Figure(go.Bar(
                    x=outliers_sorted["outlier_pct"], y=outliers_sorted["column"], orientation="h",
                    marker=dict(color=outliers_sorted["outlier_pct"], colorscale=RISK_SCALE, cmin=0, cmax=100),
                ))
                fig_out.update_layout(xaxis=dict(range=[0, max(10, outliers_sorted["outlier_pct"].max())], title="Outlier %"),
                                       yaxis_title="",
                                       height=_dyn_height(len(outliers_sorted)), margin=dict(l=10, r=20, t=20, b=30))
                st.plotly_chart(fig_out, use_container_width=True)
                st.caption("Color scale: \U0001F7E2 green = low outlier %  \u2192  \U0001F534 red = high outlier %")
                with st.expander(f"View all {len(outliers_df)} columns (table)"):
                    st.dataframe(outliers_df.sort_values("outlier_pct", ascending=False), use_container_width=True)
            else:
                st.info("No numeric columns found for outlier detection.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Missing Values")
                nulls_df = profile["nulls"]
                fig_null = px.bar(nulls_df, x="column", y="null_pct", color="null_pct",
                                   color_continuous_scale=RISK_SCALE, range_color=[0, 100])
                fig_null.update_layout(coloraxis_showscale=False, xaxis_title="", yaxis_title="Null %")
                st.plotly_chart(fig_null, use_container_width=True)
                st.caption("Color scale: \U0001F7E2 green = low missing %  \u2192  \U0001F534 red = high missing %")
            with c2:
                st.subheader("Outlier Distribution")
                outliers_df = profile["outliers"]
                if not outliers_df.empty:
                    # Bar chart, colored by value (green -> red).
                    fig_out = px.bar(outliers_df, x="column", y="outlier_pct", color="outlier_pct",
                                      color_continuous_scale=RISK_SCALE, range_color=[0, 100])
                    fig_out.update_layout(coloraxis_showscale=False, xaxis_title="", yaxis_title="Outlier %")
                    st.plotly_chart(fig_out, use_container_width=True)
                    st.caption("Color scale: \U0001F7E2 green = low outlier %  \u2192  \U0001F534 red = high outlier %")
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

        # Passed/Failed/Warnings are counts from the full rules x records
        # cross-product (e.g. 11 rules x 5,250 records = up to ~57,750
        # individual checks) - a raw count like "Passed: 42,318" tells an
        # end user almost nothing on its own. Show each as a share of
        # total checks performed instead, plus one headline pass-rate
        # number, so the summary is readable at a glance.
        total_checks = score["passed"] + score["failed"] + score["warnings"]
        pct_passed = round(score["passed"] / total_checks * 100, 1) if total_checks else 0.0
        pct_failed = round(score["failed"] / total_checks * 100, 1) if total_checks else 0.0
        pct_warnings = round(score["warnings"] / total_checks * 100, 1) if total_checks else 0.0
        pct_critical_of_failed = round(score["critical_issues"] / score["failed"] * 100, 1) if score["failed"] else 0.0

        st.caption(
            f"Each **check** = one rule applied to one record ({score['rules_executed']} rules x records "
            f"= **{total_checks:,} checks** performed on this dataset)."
        )

        # All 6 score cards in a single row (one line), so nothing wraps
        # onto a second row even with the added headline pass-rate card.
        rcol0, rcol1, rcol2, rcol3, rcol4, rcol5 = st.columns(6)
        rcol0.metric(
            "\U0001F3AF Overall Pass Rate", f"{pct_passed}%",
            help="Share of all rule checks (rules x records) that passed. This is the single best "
                 "at-a-glance number for 'how healthy is this dataset overall'.",
        )
        rcol1.metric(
            "\U0001F4CB Rules Executed", score["rules_executed"],
            help="Number of distinct DQ rules that were evaluated against this dataset.",
        )
        rcol2.metric(
            "\u2705 Passed", f"{score['passed']:,}", delta=f"{pct_passed}% of checks", delta_color="off",
            help=f"Checks where the data met the rule's condition - {pct_passed}% of the {total_checks:,} total checks performed.",
        )
        rcol3.metric(
            "\u274C Failed", f"{score['failed']:,}", delta=f"{pct_failed}% of checks", delta_color="off",
            help=f"Checks where the data violated the rule's condition - {pct_failed}% of the {total_checks:,} total checks performed.",
        )
        rcol4.metric(
            "\u26A0\uFE0F Warnings", f"{score['warnings']:,}", delta=f"{pct_warnings}% of checks", delta_color="off",
            help=f"Lower-severity issues flagged, not hard failures - {pct_warnings}% of the {total_checks:,} total checks performed.",
        )
        rcol5.metric(
            "\U0001F534 Critical Issues", f"{score['critical_issues']:,}",
            delta=f"{pct_critical_of_failed}% of failures" if score["failed"] else None, delta_color="off",
            help="Failed checks marked CRITICAL severity - the highest-priority issues to fix first, "
                 "shown as a share of all failures above.",
        )

        st.subheader("Data Quality (DQ) Dimensions")
        if not dim_summary.empty:
            # Same line+marker chart as before - only the color changed:
            # Vertical bar chart, one bar per dimension, colored on a
            # continuous red-to-green scale tied to the pass rate value
            # (0% = red, 100% = green).
            dim_sorted = dim_summary.sort_values("dimension")
            fig_dim = go.Figure(go.Bar(
                x=dim_sorted["dimension"],
                y=dim_sorted["pass_rate_pct"],
                marker=dict(color=dim_sorted["pass_rate_pct"], colorscale=HEALTH_SCALE, cmin=0, cmax=100),
                text=dim_sorted["pass_rate_pct"].astype(str) + "%",
                textposition="outside",
                hovertemplate="%{x}: %{y}%<extra></extra>",
            ))
            fig_dim.update_layout(
                yaxis=dict(range=[0, 115], title="Pass Rate %"),
                xaxis_title="",
                height=380,
                margin=dict(l=10, r=20, t=30, b=30),
            )
            st.plotly_chart(fig_dim, use_container_width=True)
            st.caption("Color scale: \U0001F534 red = low pass rate  \u2192  \U0001F7E2 green = high pass rate")
            with st.expander("View dimension detail table"):
                st.dataframe(dim_summary, use_container_width=True)
        else:
            st.info("No rules matched columns in this dataset.")

        st.subheader("Exceptions (Failed / Warning Records)")
        if not exceptions_df.empty:
            # Bucket exceptions by rule: instead of a flat list of every
            # violated record, group by which rule was violated and show
            # what that violation actually means in plain language, plus
            # how many records are affected - e.g. "Customer ID Unique -
            # this value appears on more than one record - 42 records."
            def _explain_rule(rule):
                rtype = rule.get("rule_type")
                attr = rule.get("attribute")
                param = rule.get("parameter")
                if rtype == "NOT_NULL":
                    return f"'{attr}' is missing a value, but this field is required."
                if rtype == "UNIQUE":
                    return f"'{attr}' value appears on more than one record, but each value should be unique."
                if rtype == "REGEX":
                    return f"'{attr}' value doesn't match the required format."
                if rtype == "LENGTH":
                    mn = param.get("min") if isinstance(param, dict) else None
                    mx = param.get("max") if isinstance(param, dict) else None
                    return f"'{attr}' value length falls outside the allowed range ({mn}-{mx} characters)."
                if rtype == "RANGE":
                    mn = param.get("min") if isinstance(param, dict) else None
                    mx = param.get("max") if isinstance(param, dict) else None
                    return f"'{attr}' value falls outside the allowed range ({mn} to {mx})."
                if rtype == "ENUM":
                    allowed = ", ".join(param) if isinstance(param, list) else str(param)
                    return f"'{attr}' value isn't one of the allowed values ({allowed})."
                if rtype == "NOT_REPEATED_DIGITS":
                    return f"'{attr}' value is a single repeated digit (e.g. 1111111111) - usually a sign of fake or placeholder data."
                return f"'{attr}' failed the '{rule.get('rule_name', 'rule')}' check."

            SEVERITY_ICON = {"CRITICAL": "\U0001F534", "ERROR": "\U0001F7E0", "WARNING": "\U0001F7E1"}

            # get_exceptions() always returns:
            # record_id, rule_id, rule_name, attribute, value, severity, status
            # - confirmed against rule_engine.py, no guessing needed.
            rule_by_id = {r["rule_id"]: r for r in full_rule_catalog if "rule_id" in r}

            bucket_rows = []
            for rule_id, grp in exceptions_df.groupby("rule_id"):
                rule_meta = rule_by_id.get(rule_id)
                if rule_meta:
                    rule_name_disp = rule_meta.get("rule_name", rule_id)
                    dimension_disp = rule_meta.get("dimension", "")
                    severity_disp = rule_meta.get("severity", grp["severity"].iloc[0])
                    explanation = _explain_rule(rule_meta)
                else:
                    # Rule was removed/disabled from the catalog since this
                    # exception was generated - fall back to what's on the
                    # exceptions row itself so nothing silently disappears.
                    rule_name_disp = grp["rule_name"].iloc[0]
                    dimension_disp = ""
                    severity_disp = grp["severity"].iloc[0]
                    explanation = "Rule violation detected."
                record_count = grp["record_id"].nunique()
                icon = SEVERITY_ICON.get(str(severity_disp).upper(), "")
                bucket_rows.append({
                    "Rule": rule_name_disp,
                    "Dimension": dimension_disp,
                    "Severity": f"{icon} {severity_disp}".strip(),
                    "What it means": explanation,
                    "Records Affected": record_count,
                })

            bucket_df = pd.DataFrame(bucket_rows).sort_values("Records Affected", ascending=False)
            st.dataframe(bucket_df, use_container_width=True, hide_index=True)
            with st.expander("View raw exception records"):
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
            fig_trend = px.bar(run_history, x="run_timestamp", y="quality_score", color="quality_score",
                                color_continuous_scale=HEALTH_SCALE, range_color=[0, 100])
            fig_trend.update_layout(coloraxis_showscale=False, xaxis_title="", yaxis_title="Quality Score")
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
