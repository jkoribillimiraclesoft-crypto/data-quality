"""
rule_engine.py
A generic, metadata-driven rule engine. It does NOT contain any
domain-specific logic (no "check email", no "check phone" functions).
Instead it reads rule definitions from rule_catalog.json and executes
whichever generic rule_type each one specifies.

Adding a new rule = adding a row to rule_catalog.json, not writing new code.

Statuses used (per ISO-8000-style DQ practice, per team lead feedback):
  PASS            - rule satisfied
  FAIL            - rule violated
  WARNING         - potential issue, not a hard failure
  NOT_APPLICABLE  - rule doesn't apply to this record (e.g. column missing)
  ERROR           - rule could not be executed (bad config, bad data type)
"""

import json
import re
import pandas as pd

STATUS_PASS = "PASS"
STATUS_FAIL = "FAIL"
STATUS_WARNING = "WARNING"
STATUS_NA = "NOT_APPLICABLE"
STATUS_ERROR = "ERROR"


def load_rule_catalog(path: str = "rule_catalog.json") -> list:
    with open(path, "r") as f:
        catalog = json.load(f)
    return sorted(catalog, key=lambda r: r.get("priority", 999))


def select_rules(catalog: list, rule_count: int) -> list:
    """
    Returns the first `rule_count` rules from the catalog, ordered by
    priority. This is what the sidebar slider controls - as new rules
    get added to rule_catalog.json in future, the slider's max simply
    grows to match len(catalog); no UI code changes needed.
    """
    return catalog[:rule_count]


# ---------------------------------------------------------------
# Generic rule_type implementations.
# Each function takes a single cell value + the rule's parameter,
# and returns True (rule satisfied) or False (rule violated).
# ---------------------------------------------------------------

def _check_not_null(value, param):
    return pd.notna(value) and str(value).strip() != ""


def _check_unique(series: pd.Series, param):
    """Returns a boolean Series - True where value is NOT a duplicate."""
    return ~series.duplicated(keep=False)


def _check_regex(value, param):
    if pd.isna(value):
        return None  # let NOT_NULL rule catch missing values separately
    return bool(re.match(param, str(value)))


def _check_length(value, param):
    if pd.isna(value):
        return None
    length = len(str(value))
    return param.get("min", 0) <= length <= param.get("max", float("inf"))


def _check_range(value, param):
    if pd.isna(value):
        return None
    try:
        num = float(value)
    except (ValueError, TypeError):
        return "ERROR"
    return param.get("min", float("-inf")) <= num <= param.get("max", float("inf"))


def _check_enum(value, param):
    if pd.isna(value):
        return None
    return str(value) in param


def _check_not_repeated_digits(value, param):
    if pd.isna(value):
        return None
    digits = re.sub(r"\D", "", str(value))
    if len(digits) == 0:
        return None
    return len(set(digits)) > 1  # False if all digits are the same


RULE_FUNCTIONS = {
    "NOT_NULL": _check_not_null,
    "REGEX": _check_regex,
    "LENGTH": _check_length,
    "RANGE": _check_range,
    "ENUM": _check_enum,
    "NOT_REPEATED_DIGITS": _check_not_repeated_digits,
    # UNIQUE handled separately (needs the whole column, not one cell)
}


def apply_rules(df: pd.DataFrame, rule_catalog: list) -> pd.DataFrame:
    """
    Executes every enabled rule in rule_catalog against df.
    Returns a long-format results dataframe:
    row_index | rule_id | rule_name | dimension | attribute | severity | status | value
    """
    results = []

    for rule in rule_catalog:
        if not rule.get("enabled", True):
            continue

        attribute = rule["attribute"]
        rule_type = rule["rule_type"]
        param = rule.get("parameter")

        if attribute not in df.columns:
            # Column doesn't exist in this dataset - not applicable
            results.append({
                "row_index": None, "rule_id": rule["rule_id"], "rule_name": rule["rule_name"],
                "dimension": rule["dimension"], "attribute": attribute,
                "severity": rule["severity"], "status": STATUS_NA, "value": None,
            })
            continue

        if rule_type == "UNIQUE":
            not_dup_mask = _check_unique(df[attribute], param)
            for idx, is_unique in not_dup_mask.items():
                status = STATUS_PASS if is_unique else STATUS_FAIL
                results.append({
                    "row_index": idx, "rule_id": rule["rule_id"], "rule_name": rule["rule_name"],
                    "dimension": rule["dimension"], "attribute": attribute,
                    "severity": rule["severity"], "status": status, "value": df.at[idx, attribute],
                })
            continue

        func = RULE_FUNCTIONS.get(rule_type)
        if func is None:
            results.append({
                "row_index": None, "rule_id": rule["rule_id"], "rule_name": rule["rule_name"],
                "dimension": rule["dimension"], "attribute": attribute,
                "severity": rule["severity"], "status": STATUS_ERROR, "value": None,
            })
            continue

        for idx, value in df[attribute].items():
            outcome = func(value, param)

            if outcome is None:
                status = STATUS_NA
            elif outcome == "ERROR":
                status = STATUS_ERROR
            elif outcome is True:
                status = STATUS_PASS
            else:
                # Violated - severity determines whether it's a FAIL or WARNING
                status = STATUS_WARNING if rule["severity"] == "WARNING" else STATUS_FAIL

            results.append({
                "row_index": idx, "rule_id": rule["rule_id"], "rule_name": rule["rule_name"],
                "dimension": rule["dimension"], "attribute": attribute,
                "severity": rule["severity"], "status": status, "value": value,
            })

    return pd.DataFrame(results)


def summarize_by_dimension(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Rolls up rule results into a per-dimension pass rate, e.g.:
    Completeness 98.7%, Validity 99.8%, Uniqueness 96.2% ...
    """
    if results_df.empty:
        return pd.DataFrame(columns=["dimension", "pass_rate_pct", "total_checks", "failed", "warnings"])

    records = []
    for dim, group in results_df.groupby("dimension"):
        total = len(group)
        passed = (group["status"] == STATUS_PASS).sum()
        failed = (group["status"] == STATUS_FAIL).sum()
        warnings = (group["status"] == STATUS_WARNING).sum()
        applicable = total - (group["status"] == STATUS_NA).sum()
        pass_rate = round(passed / applicable * 100, 2) if applicable > 0 else 100.0
        records.append({
            "dimension": dim, "pass_rate_pct": pass_rate,
            "total_checks": total, "failed": int(failed), "warnings": int(warnings),
        })
    return pd.DataFrame(records)


def get_exceptions(results_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a record-level exceptions table: which row, which rule failed,
    what the value was, how severe. This is the 'Exceptions view' called
    out as the most important addition in the feedback.
    """
    bad = results_df[results_df["status"].isin([STATUS_FAIL, STATUS_WARNING, STATUS_ERROR])].copy()
    if bad.empty:
        return bad

    def _get_id(row_idx):
        if row_idx is None or pd.isna(row_idx):
            return None
        if "customer_id" in df.columns:
            return df.at[int(row_idx), "customer_id"]
        return row_idx

    bad["record_id"] = bad["row_index"].apply(_get_id)
    return bad[["record_id", "rule_id", "rule_name", "attribute", "value", "severity", "status"]].sort_values(
        by="severity"
    )


def overall_rule_score(results_df: pd.DataFrame) -> dict:
    """High-level counters shown at the top of the dashboard."""
    if results_df.empty:
        return {"rules_executed": 0, "passed": 0, "failed": 0, "warnings": 0, "critical_issues": 0}

    applicable = results_df[results_df["status"] != STATUS_NA]
    return {
        "rules_executed": results_df["rule_id"].nunique(),
        "passed": int((applicable["status"] == STATUS_PASS).sum()),
        "failed": int((applicable["status"] == STATUS_FAIL).sum()),
        "warnings": int((applicable["status"] == STATUS_WARNING).sum()),
        "critical_issues": int(((applicable["status"] == STATUS_FAIL) & (applicable["severity"] == "CRITICAL")).sum()),
    }
