"""
entity_resolution.py
Lightweight MDM-style entity resolution: scores similarity between record
pairs on chosen fields, groups records whose similarity crosses a
configurable threshold into clusters (the same real-world entity), then
merges each cluster into a single "golden record" using simple
survivorship rules (most complete / longest non-null value wins).

Uses only the standard library (difflib) - no extra dependencies.

Note on scale: pairwise comparison is O(n^2) within a block. For larger
datasets, pass `block_by` (e.g. last name, postal code) so only records
sharing that key are ever compared to each other - standard MDM
"blocking" technique, and the only thing that keeps this tractable on
more than a few thousand rows.
"""

import re
from difflib import SequenceMatcher
import pandas as pd


def _normalize(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def _digits_only(value) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\D", "", str(value))


def field_similarity(a, b, field_name: str = "") -> float:
    """Similarity score 0-100 between two values of the same field.
    Missing on either side -> 0 (can't confirm a match from nothing)."""
    a_n, b_n = _normalize(a), _normalize(b)
    if a_n == "" or b_n == "":
        return 0.0

    lname = field_name.lower()
    if "phone" in lname:
        da, db = _digits_only(a), _digits_only(b)
        if da and da == db:
            return 100.0
        return SequenceMatcher(None, da, db).ratio() * 100
    if "email" in lname:
        if a_n == b_n:
            return 100.0
        return SequenceMatcher(None, a_n, b_n).ratio() * 100

    return SequenceMatcher(None, a_n, b_n).ratio() * 100


def pair_score(row_a: pd.Series, row_b: pd.Series, fields: list):
    """Average similarity across fields that are actually comparable
    (non-missing on BOTH sides). A field missing on one side is excluded
    from the average rather than scored as 0 - missing data means "no
    signal", not "confirmed non-match", and scoring it as 0 would wrongly
    punish records that are a real match but incomplete on one field.
    Returns (score, fields_compared) - fields_compared lets the caller
    see how much evidence backed the score (2-of-3 fields is weaker
    evidence than 3-of-3, even at the same average).
    """
    scored = []
    for f in fields:
        if _normalize(row_a[f]) == "" or _normalize(row_b[f]) == "":
            continue
        scored.append(field_similarity(row_a[f], row_b[f], f))
    if not scored:
        return 0.0, 0
    return round(sum(scored) / len(scored), 1), len(scored)


def estimate_comparisons(df: pd.DataFrame, block_by: str = None, max_block_size: int = 300) -> int:
    """Estimates how many pairwise comparisons find_candidate_matches would
    actually perform, respecting the same block-size cap - so the UI can
    warn *before* running something that would take too long, instead of
    the user discovering it by waiting."""
    if block_by and block_by in df.columns:
        block_key = df[block_by].apply(_normalize)
        sizes = df.groupby(block_key).size().tolist()
    else:
        sizes = [len(df)]
    total = 0
    for n in sizes:
        n = min(n, max_block_size)
        if n > 1:
            total += n * (n - 1) // 2
    return total


def find_candidate_matches(df: pd.DataFrame, fields: list, threshold: float,
                            block_by: str = None, max_block_size: int = 300,
                            min_fields_compared: int = 1):
    """
    Returns (pairs, truncated_blocks, rows_skipped): pairs is a list of
    (index_a, index_b, score) for every pair scoring >= threshold on at
    least `min_fields_compared` comparable fields. truncated_blocks is
    how many blocks exceeded max_block_size and had to be capped, and
    rows_skipped is exactly how many rows were left out of comparison
    entirely as a result (e.g. with no blocking at all, a large dataset
    becomes one giant block and everything past max_block_size is never
    compared to anything - this number makes that impact explicit rather
    than a vague "some blocks were truncated" note).
    min_fields_compared guards against a "match" based on a single
    coincidentally-similar field when everything else was missing.
    """
    if block_by and block_by in df.columns:
        block_key = df[block_by].apply(_normalize)
        groups = [g.index.tolist() for _, g in df.groupby(block_key) if len(g) > 1]
    else:
        groups = [df.index.tolist()]

    pairs = []
    truncated_blocks = 0
    rows_skipped = 0
    for idxs in groups:
        n = len(idxs)
        if n < 2:
            continue
        if n > max_block_size:
            rows_skipped += n - max_block_size
            idxs = idxs[:max_block_size]
            n = len(idxs)
            truncated_blocks += 1
        for a in range(n):
            row_a = df.loc[idxs[a]]
            for b in range(a + 1, n):
                row_b = df.loc[idxs[b]]
                score, n_compared = pair_score(row_a, row_b, fields)
                if score >= threshold and n_compared >= min_fields_compared:
                    pairs.append((idxs[a], idxs[b], score))
    return pairs, truncated_blocks, rows_skipped


class _UnionFind:
    def __init__(self, items):
        self.parent = {i: i for i in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def build_clusters(df: pd.DataFrame, pairs: list) -> list:
    """Groups records into clusters via union-find over matched pairs.
    Every record appears in exactly one cluster, including singletons
    (records with no match - a cluster of one)."""
    uf = _UnionFind(df.index.tolist())
    for i, j, _ in pairs:
        uf.union(i, j)
    clusters = {}
    for idx in df.index:
        root = uf.find(idx)
        clusters.setdefault(root, []).append(idx)
    return list(clusters.values())


def _survive(values: list):
    """Survivorship rule: prefer the longest non-null value as 'most
    complete'. Simple and explainable; a real MDM system would let you
    configure per-field survivorship rules (most recent, source
    priority, etc.) - this is a reasonable default for a POC."""
    candidates = [v for v in values if pd.notna(v) and str(v).strip() != ""]
    if not candidates:
        return None
    return max(candidates, key=lambda v: len(str(v)))


def merge_cluster(df: pd.DataFrame, idxs: list) -> dict:
    row = {col: _survive([df.at[i, col] for i in idxs]) for col in df.columns}
    row["_merged_record_ids"] = ",".join(str(i) for i in idxs)
    row["_merged_record_count"] = len(idxs)
    return row


def build_golden_dataset(df: pd.DataFrame, clusters: list) -> pd.DataFrame:
    return pd.DataFrame([merge_cluster(df, idxs) for idxs in clusters])
