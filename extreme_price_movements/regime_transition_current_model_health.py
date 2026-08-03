"""Compact causal model-health panel for the current execution-EV lineage.

The old55 prediction-shard health panel is intentionally not used here.  This
module starts from the current execution-EV handoff and its matching, causal
recent-EV mapping ledger.  Five outcome fields are constructed sequentially:
at a decision timestamp they only contain labels whose recorded resolution time
is *strictly* earlier than that timestamp.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd


CURRENT_MODEL_HEALTH_COLUMNS = (
    "health__candidate_rows",
    "health__distinct_assets",
    "health__long_share",
    "health__candidate_rows_delta_24h",
    "health__mapped_ev_mean",
    "health__mapped_ev_std",
    "health__mapped_ev_p90",
    "health__mapped_ev_entropy",
    "health__mapped_ev_long_minus_short",
    "health__mapped_ev_global_side_abs_gap_mean",
    "health__base_score_mean",
    "health__base_score_std",
    "health__residual_abs_mean",
    "health__residual_std",
    "health__residual_long_minus_short",
    "health__base_residual_rank_spearman",
    "health__base_residual_rank_abs_gap",
    "health__cutoff_margin_z_mean",
    "health__cutoff_margin_z_std",
    "health__mapped_ev_coverage",
    "health__base_score_coverage",
    "health__residual_coverage",
    "health__catboost_entropy_mean",
    "health__alpha_uncertainty_mean",
    "health__recent_resolved_net_ev_hl3d",
    "health__recent_resolved_hit_rate_hl3d",
    "health__recent_resolved_mapping_error_hl3d",
    "health__recent_resolved_cost_bps_hl3d",
    "health__recent_resolved_effective_rows_hl3d",
)

_LEDGER_REQUIRED = {
    "candidate_id",
    "__ts__",
    "__symbol__",
    "side_name",
    "execution_decision_utc",
    "execution_label_end_utc",
    "execution_gross_ev_12h",
    "execution_net_ev_12h",
    "causal_recent_isotonic_ev",
    "causal_recent_side_isotonic_ev",
    "catboost__residual__without_hpo__all_features",
}
_RICH_REQUIRED = {
    "candidate_id",
    "__ts__",
    "execution_decision_utc",
    "base_oof_score",
    "base_margin_to_cutoff_z",
    "catboost_entropy",
    "alpha_prediction_uncertainty",
}


def _utc(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        result[column] = pd.to_datetime(result[column], utc=True, errors="coerce")
        if result[column].isna().any():
            raise ValueError(f"{column} contains non-UTC/invalid timestamps")
    return result


def _require(frame: pd.DataFrame, required: set[str], source: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {', '.join(missing)}")


def _entropy(values: pd.Series) -> float:
    """Entropy over fixed, economic EV bins; no future-derived bin edges."""

    numeric = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    if not len(numeric):
        return np.nan
    # Values are returns.  These fixed bins deliberately bracket the 1% cost
    # scale rather than fitting quantiles on the evaluation interval.
    edges = np.array([-np.inf, -0.02, -0.01, -0.005, -0.001, 0.0, 0.001, 0.005, 0.01, 0.02, np.inf])
    count, _ = np.histogram(numeric, bins=edges)
    probability = count[count > 0] / count.sum()
    return float(-(probability * np.log(probability)).sum())


def _rank_relationship(group: pd.DataFrame) -> pd.Series:
    local = group[["base_oof_score", "residual"]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(local) < 3 or local.iloc[:, 0].nunique() < 2 or local.iloc[:, 1].nunique() < 2:
        return pd.Series({
            "health__base_residual_rank_spearman": np.nan,
            "health__base_residual_rank_abs_gap": np.nan,
        })
    rank = local.rank(method="average", pct=True)
    return pd.Series({
        "health__base_residual_rank_spearman": float(rank.iloc[:, 0].corr(rank.iloc[:, 1], method="pearson")),
        "health__base_residual_rank_abs_gap": float((rank.iloc[:, 0] - rank.iloc[:, 1]).abs().mean()),
    })


def _causal_resolved_outcomes(work: pd.DataFrame, hourly: pd.DataFrame) -> pd.DataFrame:
    """Attach strictly-prior 3-day exponentially decayed outcome diagnostics."""

    outcome = pd.DataFrame({
        "resolved_at": pd.to_datetime(work["execution_label_end_utc"], utc=True),
        "net": pd.to_numeric(work["execution_net_ev_12h"], errors="coerce"),
        "hit": pd.to_numeric(work["execution_net_ev_12h"], errors="coerce").gt(0).astype(float),
        "mapping_error": (
            pd.to_numeric(work["execution_net_ev_12h"], errors="coerce")
            - pd.to_numeric(work["mapped_side_ev"], errors="coerce")
        ),
        "cost_bps": (
            pd.to_numeric(work["execution_gross_ev_12h"], errors="coerce")
            - pd.to_numeric(work["execution_net_ev_12h"], errors="coerce")
        ) * 10_000.0,
    }).dropna(subset=["resolved_at"])
    outcome = outcome.loc[outcome["resolved_at"].notna()].copy()
    if outcome.empty:
        for column in CURRENT_MODEL_HEALTH_COLUMNS[-5:]:
            hourly[column] = np.nan
        return hourly
    grouped = outcome.groupby("resolved_at", observed=True).agg(
        net_sum=("net", "sum"),
        hit_sum=("hit", "sum"),
        error_sum=("mapping_error", "sum"),
        cost_sum=("cost_bps", "sum"),
        rows=("net", "count"),
    )
    indexed = hourly.set_index("execution_decision_utc", drop=False).sort_index()
    # The panel is hourly, but preserve the no-lookahead rule even if a source
    # has a gap: outcome updates are applied only after recording that hour.
    start = min(indexed.index.min(), grouped.index.min()).floor("h")
    end = max(indexed.index.max(), grouped.index.max()).ceil("h")
    timeline = pd.date_range(start, end, freq="h", tz="UTC")
    rate = float(np.exp(-np.log(2.0) / 72.0))
    weight = 0.0
    sums = np.zeros(4, dtype=float)
    prior: dict[pd.Timestamp, tuple[float, float, float, float, float]] = {}
    for stamp in timeline:
        weight *= rate
        sums *= rate
        # Record before incorporating labels resolving at this exact decision
        # time.  Resolution must be strictly earlier, not merely equal.
        if weight > 0:
            prior[stamp] = (*tuple(sums / weight), weight)
        else:
            prior[stamp] = (np.nan, np.nan, np.nan, np.nan, 0.0)
        if stamp in grouped.index:
            row = grouped.loc[stamp]
            rows = float(row["rows"])
            if rows > 0:
                sums += row[["net_sum", "hit_sum", "error_sum", "cost_sum"]].to_numpy(float)
                weight += rows
    prior_frame = pd.DataFrame.from_dict(
        prior,
        orient="index",
        columns=list(CURRENT_MODEL_HEALTH_COLUMNS[-5:]),
    )
    prior_frame.index.name = "execution_decision_utc"
    result = indexed.join(prior_frame, how="left", rsuffix="__prior").reset_index(drop=True)
    return result


def build_hourly_current_model_health(
    mapping_ledger: pd.DataFrame,
    current_handoff: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build the 29-field compact, current-lineage hourly health panel.

    ``mapping_ledger`` supplies only current, causally mapped scores and exact
    label-resolution timestamps.  ``current_handoff`` supplies the matching
    base and residual-layer context.  Their candidate IDs must agree one-to-one.
    """

    _require(mapping_ledger, _LEDGER_REQUIRED, "mapping ledger")
    _require(current_handoff, _RICH_REQUIRED, "current handoff")
    ledger = _utc(mapping_ledger, ("__ts__", "execution_decision_utc", "execution_label_end_utc"))
    rich = _utc(current_handoff, ("__ts__", "execution_decision_utc"))
    if ledger["candidate_id"].duplicated().any() or rich["candidate_id"].duplicated().any():
        raise ValueError("candidate IDs must be unique in both current-lineage sources")
    right_columns = [
        "candidate_id", "__ts__", "execution_decision_utc", "base_oof_score",
        "base_margin_to_cutoff_z", "catboost_entropy", "alpha_prediction_uncertainty",
    ]
    handoff_columns = rich.loc[:, right_columns].rename(columns={
        "__ts__": "handoff_source_utc",
        "execution_decision_utc": "handoff_execution_decision_utc",
    })
    work = ledger.merge(
        handoff_columns, on="candidate_id", how="inner", validate="one_to_one"
    )
    if work.empty:
        raise ValueError("no matching current-lineage candidate IDs")
    for column, handoff_column in (
        ("__ts__", "handoff_source_utc"),
        ("execution_decision_utc", "handoff_execution_decision_utc"),
    ):
        if not work[column].eq(work[handoff_column]).all():
            raise ValueError(f"candidate timestamp mismatch between sources: {column}")
    work = work.drop(columns=["handoff_source_utc", "handoff_execution_decision_utc"])
    if not work["execution_label_end_utc"].gt(work["execution_decision_utc"]).all():
        raise ValueError("label resolution must be later than the candidate decision")
    work = work.rename(columns={
        "causal_recent_side_isotonic_ev": "mapped_side_ev",
        "causal_recent_isotonic_ev": "mapped_global_ev",
        "catboost__residual__without_hpo__all_features": "residual",
    })
    for column in (
        "mapped_side_ev", "mapped_global_ev", "residual", "base_oof_score",
        "base_margin_to_cutoff_z", "catboost_entropy", "alpha_prediction_uncertainty",
    ):
        work[column] = pd.to_numeric(work[column], errors="coerce")
    grouped = work.groupby("__ts__", sort=True, observed=True)
    hourly = grouped.agg(
        **{
            "health__candidate_rows": ("candidate_id", "size"),
            "health__distinct_assets": ("__symbol__", "nunique"),
            "health__long_share": ("side_name", lambda value: float(value.astype(str).eq("long").mean())),
            "health__mapped_ev_mean": ("mapped_side_ev", "mean"),
            "health__mapped_ev_std": ("mapped_side_ev", "std"),
            "health__mapped_ev_p90": ("mapped_side_ev", lambda value: value.quantile(0.90)),
            "health__mapped_ev_entropy": ("mapped_side_ev", _entropy),
            "health__mapped_ev_global_side_abs_gap_mean": ("mapped_side_ev", lambda value: np.nan),
            "health__base_score_mean": ("base_oof_score", "mean"),
            "health__base_score_std": ("base_oof_score", "std"),
            "health__residual_abs_mean": ("residual", lambda value: value.abs().mean()),
            "health__residual_std": ("residual", "std"),
            "health__cutoff_margin_z_mean": ("base_margin_to_cutoff_z", "mean"),
            "health__cutoff_margin_z_std": ("base_margin_to_cutoff_z", "std"),
            "health__mapped_ev_coverage": ("mapped_side_ev", lambda value: float(value.notna().mean())),
            "health__base_score_coverage": ("base_oof_score", lambda value: float(value.notna().mean())),
            "health__residual_coverage": ("residual", lambda value: float(value.notna().mean())),
            "health__catboost_entropy_mean": ("catboost_entropy", "mean"),
            "health__alpha_uncertainty_mean": ("alpha_prediction_uncertainty", "mean"),
        }
    )
    # Side disagreement is an observable same-timestamp comparison.  Keep it
    # separate from the global top-k policy; this panel never imposes a quota.
    side = work.groupby(["__ts__", "side_name"], observed=True).agg(
        mapped=("mapped_side_ev", "mean"), residual=("residual", "mean")
    ).unstack("side_name")
    for metric, name in (("mapped", "health__mapped_ev_long_minus_short"), ("residual", "health__residual_long_minus_short")):
        long = side[(metric, "long")] if (metric, "long") in side else pd.Series(index=hourly.index, dtype=float)
        short = side[(metric, "short")] if (metric, "short") in side else pd.Series(index=hourly.index, dtype=float)
        hourly[name] = long - short
    hourly["health__mapped_ev_global_side_abs_gap_mean"] = grouped.apply(
        lambda group: (group["mapped_side_ev"] - group["mapped_global_ev"]).abs().mean(), include_groups=False
    )
    relationship = grouped.apply(_rank_relationship, include_groups=False)
    hourly = hourly.join(relationship)
    hourly = hourly.reset_index().rename(columns={"__ts__": "source_utc"})
    hourly["execution_decision_utc"] = hourly["source_utc"] + pd.Timedelta(hours=1)
    hourly = hourly.sort_values("source_utc", kind="stable").reset_index(drop=True)
    hourly["health__candidate_rows_delta_24h"] = hourly["health__candidate_rows"].diff(24)
    hourly = _causal_resolved_outcomes(work, hourly)
    missing = sorted(set(CURRENT_MODEL_HEALTH_COLUMNS).difference(hourly.columns))
    if missing:
        raise AssertionError(f"health panel did not materialize: {', '.join(missing)}")
    hourly = hourly.loc[:, ["source_utc", "execution_decision_utc", *CURRENT_MODEL_HEALTH_COLUMNS]].copy()
    hourly.loc[:, CURRENT_MODEL_HEALTH_COLUMNS] = hourly.loc[:, CURRENT_MODEL_HEALTH_COLUMNS].astype(np.float32)
    flag_columns = [name for name in ledger if name.endswith("__is_oof") or name.endswith("__is_forward_oos")]
    report: dict[str, Any] = {
        "schema": "current_lineage_model_health_v1",
        "lineage": "current execution-EV repaired-heads handoff plus causal recent-EV mapping; old55 excluded",
        "candidate_rows": int(len(work)),
        "hourly_rows": int(len(hourly)),
        "start": hourly["source_utc"].min(),
        "end": hourly["source_utc"].max(),
        "health_feature_count": int(len(CURRENT_MODEL_HEALTH_COLUMNS)),
        "strict_resolution_contract": (
            "recent resolved outcome fields are recorded before labels resolving at the same decision timestamp are incorporated"
        ),
        "prediction_availability": {
            "base_oof_score": "carried from current handoff; source handoff is the point-in-time prediction contract",
            "residual": "current mapped-ledger CatBoost residual score",
            "mapping": "causal recent-EV score from current mapping ledger",
        },
        "mapping_provenance_counts": {
            column: int(ledger[column].fillna(False).astype(bool).sum())
            for column in flag_columns
        },
    }
    return hourly, report


__all__ = ["CURRENT_MODEL_HEALTH_COLUMNS", "build_hourly_current_model_health"]
