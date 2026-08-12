"""Causal, descriptive diagnostics for base-score relationship drift.

The functions in this module deliberately do not fit or modify a model.  They
answer a narrower question on an already frozen prediction ledger: did the
relationship between the base score and a resolved target change across
months, after separating within-query ranking from candidate composition?

``query_cols`` normally is ``("decision_ts", "side_name")``.  All ranking
metrics are computed *inside* those queries, so a changing score scale across
hours or sides cannot manufacture rank IC or winner recall.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


DEFAULT_QUERY_COLUMNS: tuple[str, ...] = ("decision_ts", "side_name")
DEFAULT_SIDE_COLUMN = "side_name"
DEFAULT_PERIOD_COLUMN = "month"


def _require(frame: pd.DataFrame, columns: Sequence[str], *, name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} lacks required columns: {missing}")


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)


def _canonical_period(values: pd.Series) -> pd.Series:
    """Return sortable calendar-month labels without depending on local TZ."""
    timestamp = pd.to_datetime(values, utc=True, errors="coerce")
    if timestamp.notna().any():
        return timestamp.dt.tz_localize(None).dt.to_period("M").astype(str)
    return values.astype(str)


def _query_ranks(
    work: pd.DataFrame,
    *,
    score_col: str,
    target_col: str,
    query_cols: Sequence[str],
    tie_breaker_col: str,
) -> pd.DataFrame:
    """Attach deterministic ascending ranks for score and target within query."""
    _require(work, [*query_cols, score_col, target_col, tie_breaker_col], name="ledger")
    result = work.loc[:, [*query_cols, score_col, target_col, tie_breaker_col]].copy()
    result["_score"] = _numeric(result, score_col)
    result["_target"] = _numeric(result, target_col)
    result = result.dropna(subset=["_score", "_target"])
    # A total order avoids row-order dependence when scores tie.  Higher score
    # gets a higher rank; winner selection subsequently uses the reverse order.
    result = result.sort_values(
        [*query_cols, "_score", tie_breaker_col],
        ascending=[True] * len(query_cols) + [True, True],
        kind="stable",
    )
    grouped = result.groupby(list(query_cols), sort=False, observed=True)
    result["_score_rank"] = grouped.cumcount().astype(np.float64) + 1.0
    # Target ties should remain ties for Spearman, unlike score selection ties.
    result["_target_rank"] = grouped["_target"].rank(method="average", ascending=True)
    result["_query_n"] = grouped["_score"].transform("size").astype(np.int64)
    return result


def _within_query_ic(ranked: pd.DataFrame, query_cols: Sequence[str]) -> tuple[float, int]:
    """Candidate-weighted mean within-query Spearman IC and usable query count."""
    if ranked.empty:
        return float("nan"), 0
    grouped = ranked.groupby(list(query_cols), observed=True, sort=False)
    x = ranked["_score_rank"]
    y = ranked["_target_rank"]
    stats = pd.DataFrame(
        {
            "n": grouped.size(),
            "sx": x.groupby([ranked[c] for c in query_cols], observed=True).sum(),
            "sy": y.groupby([ranked[c] for c in query_cols], observed=True).sum(),
            "sxx": (x * x).groupby([ranked[c] for c in query_cols], observed=True).sum(),
            "syy": (y * y).groupby([ranked[c] for c in query_cols], observed=True).sum(),
            "sxy": (x * y).groupby([ranked[c] for c in query_cols], observed=True).sum(),
        }
    )
    n = stats["n"].astype(float)
    numerator = n * stats["sxy"] - stats["sx"] * stats["sy"]
    x_var = n * stats["sxx"] - stats["sx"] * stats["sx"]
    y_var = n * stats["syy"] - stats["sy"] * stats["sy"]
    denominator = np.sqrt(np.maximum(x_var, 0.0) * np.maximum(y_var, 0.0))
    ic = numerator / denominator.where(denominator > 0.0)
    valid = ic.replace([np.inf, -np.inf], np.nan).dropna()
    if valid.empty:
        return float("nan"), 0
    # Candidate weighting makes a 1-row residual query harmless and estimates
    # the correlation of a randomly selected candidate's query relationship.
    weights = stats.loc[valid.index, "n"].astype(float)
    return float(np.average(valid.to_numpy(dtype=float), weights=weights.to_numpy(dtype=float))), int(len(valid))


def _top_query_mask(
    ranked: pd.DataFrame,
    *,
    query_cols: Sequence[str],
    fraction: float,
) -> pd.Series:
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    # score ranks ascend, therefore rank > n - ceil(fraction*n) is top tail.
    cutoff = ranked["_query_n"] - np.ceil(float(fraction) * ranked["_query_n"])
    return ranked["_score_rank"].gt(cutoff)


def _winner_recall(
    ranked: pd.DataFrame,
    original: pd.DataFrame,
    *,
    query_cols: Sequence[str],
    winner_col: str,
    fraction: float,
) -> float:
    # Align through the deterministic order established in _query_ranks.
    winners = original.loc[ranked.index, winner_col]
    winner = pd.to_numeric(winners, errors="coerce").fillna(0.0).gt(0.0)
    selected = _top_query_mask(ranked, query_cols=query_cols, fraction=fraction)
    denom = int(winner.sum())
    return float((selected & winner).sum() / denom) if denom else float("nan")


def _decile_response(
    ranked: pd.DataFrame,
    original: pd.DataFrame,
    *,
    target_col: str,
    winner_col: str,
) -> pd.DataFrame:
    work = ranked.copy()
    work["_winner"] = pd.to_numeric(original.loc[work.index, winner_col], errors="coerce").fillna(0.0).gt(0.0)
    # Highest score is decile 10.  The query-local rank is intentionally not
    # used here: deciles describe the reporting-slice score response curve.
    score = work["_score"]
    if score.nunique(dropna=True) < 2:
        work["decile"] = 1
    else:
        pct = score.rank(method="average", pct=True)
        work["decile"] = np.ceil(pct * 10.0).clip(1, 10).astype(np.int8)
    rows = (
        work.groupby("decile", observed=True)
        .agg(n=("_target", "size"), score_mean=("_score", "mean"), target_mean=("_target", "mean"), winner_rate=("_winner", "mean"))
        .reset_index()
    )
    return rows


def monthly_relationship_metrics(
    frame: pd.DataFrame,
    *,
    score_col: str,
    target_col: str,
    winner_col: str,
    timestamp_col: str = "decision_ts",
    side_col: str = DEFAULT_SIDE_COLUMN,
    query_cols: Sequence[str] = DEFAULT_QUERY_COLUMNS,
    tie_breaker_col: str = "candidate_id",
    top_fractions: Sequence[float] = (0.05, 0.30, 0.40),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return monthly pooled/side metrics and score-decile response curves.

    ``target_col`` can be R3 soft clear, R3 clear event, or exact realized
    economics.  ``winner_col`` must be a resolved binary opportunity outcome;
    it is never inferred from score percentiles.
    """
    _require(frame, [timestamp_col, side_col, score_col, target_col, winner_col, tie_breaker_col, *query_cols], name="ledger")
    work = frame.copy()
    work[DEFAULT_PERIOD_COLUMN] = _canonical_period(work[timestamp_col])
    metric_rows: list[dict[str, object]] = []
    decile_rows: list[pd.DataFrame] = []
    scopes: list[tuple[str, str, pd.DataFrame]] = [("pooled", "all", work)]
    scopes.extend(("side", str(side), local) for side, local in work.groupby(side_col, observed=True, sort=True))
    for scope, scope_value, scoped in scopes:
        for month, local in scoped.groupby(DEFAULT_PERIOD_COLUMN, observed=True, sort=True):
            ranked = _query_ranks(local, score_col=score_col, target_col=target_col, query_cols=query_cols, tie_breaker_col=tie_breaker_col)
            if ranked.empty:
                continue
            ic, usable_queries = _within_query_ic(ranked, query_cols)
            row: dict[str, object] = {
                "month": str(month), "scope": scope, "scope_value": scope_value,
                "n_rows": int(len(ranked)), "n_queries": int(ranked.groupby(list(query_cols), observed=True).ngroups),
                "n_usable_ic_queries": usable_queries, "within_query_rank_ic": ic,
                "pool_target_mean": float(ranked["_target"].mean()),
                "pool_winner_rate": float(pd.to_numeric(local.loc[ranked.index, winner_col], errors="coerce").fillna(0.0).gt(0.0).mean()),
            }
            for fraction in top_fractions:
                selected = _top_query_mask(ranked, query_cols=query_cols, fraction=float(fraction))
                suffix = str(int(round(100 * float(fraction))))
                top_mean = float(ranked.loc[selected, "_target"].mean()) if selected.any() else float("nan")
                row[f"top{suffix}_target_mean"] = top_mean
                row[f"top{suffix}_uplift"] = top_mean - float(ranked["_target"].mean())
                row[f"top{suffix}_winner_recall"] = _winner_recall(ranked, local, query_cols=query_cols, winner_col=winner_col, fraction=float(fraction))
            deciles = _decile_response(ranked, local, target_col=target_col, winner_col=winner_col)
            monotonicity = float(deciles["decile"].corr(deciles["target_mean"], method="spearman")) if len(deciles) >= 2 else float("nan")
            row["decile_monotonicity"] = monotonicity
            row["decile_adjacent_violations"] = int((deciles["target_mean"].diff().dropna() < 0.0).sum())
            deciles["month"] = str(month)
            deciles["scope"] = scope
            deciles["scope_value"] = scope_value
            decile_rows.append(deciles)
            metric_rows.append(row)
    return pd.DataFrame(metric_rows), pd.concat(decile_rows, ignore_index=True) if decile_rows else pd.DataFrame()


def adjacent_month_fixed_bin_decomposition(
    frame: pd.DataFrame,
    *,
    score_col: str,
    outcome_col: str,
    timestamp_col: str = "decision_ts",
    side_col: str = DEFAULT_SIDE_COLUMN,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Decompose adjacent-month outcome changes using prior-month fixed score bins.

    Within common ``side × fixed-score-bin`` support, ``composition_effect``
    is the change caused by score/side mix and ``relationship_effect`` is the
    change in realised payoff at fixed score-bin/side.  Their sum exactly
    equals ``observed_delta_common_support`` up to floating point error.
    """
    _require(frame, [timestamp_col, side_col, score_col, outcome_col], name="ledger")
    if int(n_bins) < 2:
        raise ValueError("n_bins must be at least 2")
    work = frame.loc[:, [timestamp_col, side_col, score_col, outcome_col]].copy()
    work["month"] = _canonical_period(work[timestamp_col])
    work["_score"] = _numeric(work, score_col)
    work["_outcome"] = _numeric(work, outcome_col)
    work = work.dropna(subset=["_score", "_outcome"])
    months = sorted(work["month"].unique())
    rows: list[dict[str, object]] = []
    for previous, current in zip(months[:-1], months[1:]):
        before = work.loc[work["month"].eq(previous)].copy()
        after = work.loc[work["month"].eq(current)].copy()
        if before.empty or after.empty:
            continue
        # Bins are fit only on the prior month.  Duplicate quantiles are
        # removed rather than inventing unstable zero-width score intervals.
        edges = np.unique(np.nanquantile(before["_score"].to_numpy(dtype=float), np.linspace(0.0, 1.0, int(n_bins) + 1)))
        if len(edges) < 2:
            continue
        edges[0], edges[-1] = -np.inf, np.inf
        before["score_bin"] = np.searchsorted(edges[1:-1], before["_score"], side="right").astype(np.int16)
        after["score_bin"] = np.searchsorted(edges[1:-1], after["_score"], side="right").astype(np.int16)
        keys = [side_col, "score_bin"]
        a = before.groupby(keys, observed=True)["_outcome"].agg(["size", "mean"]).rename(columns={"size": "n_a", "mean": "mu_a"})
        b = after.groupby(keys, observed=True)["_outcome"].agg(["size", "mean"]).rename(columns={"size": "n_b", "mean": "mu_b"})
        joined = a.join(b, how="inner")
        common_a, common_b = float(joined["n_a"].sum()), float(joined["n_b"].sum())
        if not common_a or not common_b:
            continue
        joined["w_a"] = joined["n_a"] / common_a
        joined["w_b"] = joined["n_b"] / common_b
        composition = float(((joined["w_b"] - joined["w_a"]) * joined["mu_a"]).sum())
        relationship = float((joined["w_b"] * (joined["mu_b"] - joined["mu_a"])).sum())
        observed_common = float((joined["w_b"] * joined["mu_b"]).sum() - (joined["w_a"] * joined["mu_a"]).sum())
        rows.append(
            {
                "month_from": str(previous), "month_to": str(current), "n_bins_requested": int(n_bins),
                "n_fixed_bins": int(len(edges) - 1), "common_cells": int(len(joined)),
                "prior_rows": int(len(before)), "current_rows": int(len(after)),
                "prior_common_support_share": common_a / float(len(before)),
                "current_common_support_share": common_b / float(len(after)),
                "observed_delta_common_support": observed_common,
                "composition_effect": composition, "relationship_effect": relationship,
                "decomposition_residual": observed_common - composition - relationship,
            }
        )
    return pd.DataFrame(rows)
