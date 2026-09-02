#!/usr/bin/env python3
"""Explain strict base+residual worst periods with frozen regime context.

The statistical unit is the UTC calendar week. Candidate rows are used only
to form the frozen pooled-global top-10 book. Feature, covariance and
interaction comparisons operate on weekly aggregates so hourly/candidate
pseudo-replication cannot manufacture significance.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.materialize_2022_2026_stack_performance_calendar import (
    ALPHA_TARGET,
    COST,
    DEFAULT_2025,
    DEFAULT_2026,
    END_EXCLUSIVE,
    GROSS,
    NET_TARGET,
    SCORE,
    START,
    _period_key,
    _rank_ic,
    json_safe,
    load_reconstructed_stack_rows,
    load_strict_stack_rows,
    sha256_file,
    stable_global_top_mask,
)


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "stack_regime_failure_analysis_2022_2026_v1"
DEFAULT_CALENDAR = ROOT / (
    "data_perp/artifacts/"
    "stack_performance_calendar_2022_2026_20260730_v2"
)
DEFAULT_REGIME = ROOT / (
    "data_perp/artifacts/regime_episode_ledger_2022_2026_20260730_v1"
)
DEFAULT_BACKFILL = ROOT / (
    "data_perp/artifacts/"
    "reconstructed_base_residual_stack_2022_2024q1_20260730_v2"
)
DEFAULT_OUTPUT = ROOT / (
    "data_perp/artifacts/"
    "stack_regime_failure_analysis_2022_2026_20260730_v2"
)
FORBIDDEN_PROFILE_PREFIXES = (
    "target__",
    "economic_",
    "execution_",
    "failure_",
    "outcome_",
)
NON_FEATURES = {
    "source_utc",
    "execution_decision_utc",
    "segment_id",
    "source_segment_id",
    "calendar_segment_id",
}


def bh_fdr(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    output = np.full(len(values), np.nan, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return output
    local = values[finite]
    order = np.argsort(local)
    ranked = local[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty(len(local), dtype=float)
    restored[order] = np.clip(adjusted, 0.0, 1.0)
    output[finite] = restored
    return output


def exact_label_permutation_pvalues(
    matrix: np.ndarray,
    labels: np.ndarray,
    *,
    max_exact: int = 50_000,
    monte_carlo_draws: int = 10_000,
    seed: int = 20260730,
) -> np.ndarray:
    """Two-sided exact or deterministic Monte-Carlo period-label p-values."""

    values = np.asarray(matrix, dtype=float)
    target = np.asarray(labels, dtype=bool)
    if values.ndim != 2 or len(values) != len(target):
        raise ValueError("matrix/labels are not aligned")
    n_bad = int(target.sum())
    if n_bad < 2 or len(target) - n_bad < 2:
        return np.full(values.shape[1], np.nan)
    if not np.isfinite(values).all():
        raise ValueError("permutation matrix must be finite")
    observed = np.abs(values[target].mean(axis=0) - values[~target].mean(axis=0))
    exceed = np.zeros(values.shape[1], dtype=int)
    total = 0
    all_positions = np.arange(len(target))
    if math.comb(len(target), n_bad) <= max_exact:
        for chosen in itertools.combinations(all_positions, n_bad):
            mask = np.zeros(len(target), dtype=bool)
            mask[list(chosen)] = True
            delta = np.abs(values[mask].mean(axis=0) - values[~mask].mean(axis=0))
            exceed += delta >= observed - 1e-14
            total += 1
        return exceed / max(total, 1)
    rng = np.random.default_rng(seed)
    for _ in range(int(monte_carlo_draws)):
        chosen = rng.choice(all_positions, size=n_bad, replace=False)
        mask = np.zeros(len(target), dtype=bool)
        mask[chosen] = True
        delta = np.abs(values[mask].mean(axis=0) - values[~mask].mean(axis=0))
        exceed += delta >= observed - 1e-14
        total += 1
    return (exceed + 1) / (total + 1)


def _feature_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    result = []
    for name in frame.columns:
        if name in NON_FEATURES or name.startswith(FORBIDDEN_PROFILE_PREFIXES):
            continue
        if not pd.api.types.is_numeric_dtype(frame[name]):
            continue
        result.append(name)
    if not result:
        raise ValueError("regime timeline exposes no eligible numeric features")
    return tuple(result)


def _robust_scale(
    frame: pd.DataFrame, reference_mask: np.ndarray,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    reference = frame.loc[reference_mask]
    center = reference.median(axis=0)
    mad = (reference - center).abs().median(axis=0) * 1.4826
    fallback = reference.std(axis=0).replace(0.0, np.nan)
    scale = mad.where(mad.gt(1e-9), fallback).fillna(1.0)
    return (frame - center) / scale, center, scale


def _oriented_auc(values: np.ndarray, labels: np.ndarray) -> float:
    if len(np.unique(labels)) < 2 or np.nanstd(values) <= 1e-12:
        return float("nan")
    raw = float(roc_auc_score(labels.astype(int), values))
    return max(raw, 1.0 - raw)


def identify_worst_weeks(
    performance: pd.DataFrame, *, quantile: float,
) -> pd.DataFrame:
    weeks = performance.loc[
        performance["period_type"].eq("week")
        & performance["complete_for_percentage"]
    ].copy()
    if len(weeks) < 8:
        raise ValueError("at least eight complete strict-stack weeks are required")
    count = max(2, int(math.ceil(len(weeks) * float(quantile))))
    weeks = weeks.sort_values(
        ["mean_net_bps", "period_start_utc"], kind="mergesort"
    ).reset_index(drop=True)
    weeks["worst_week"] = False
    weeks.loc[: count - 1, "worst_week"] = True
    weeks["worst_definition"] = (
        f"bottom {count}/{len(weeks)} complete lineage-qualified stack weeks by "
        "pooled-global top10 exact-policy mean net"
    )
    return weeks


def _load_regime(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != "regime_episode_calendar_v1"
        or int(manifest.get("counts", {}).get("transition_events", 0)) != 157
    ):
        raise ValueError("combined frozen 5-state regime episode ledger is required")
    hourly = pd.read_parquet(root / "hourly_state_calendar.parquet")
    events = pd.read_parquet(root / "transition_episode_ledger.parquet")
    hourly["source_utc"] = pd.to_datetime(
        hourly["source_utc"], utc=True, errors="raise"
    )
    if hourly["source_utc"].duplicated().any():
        raise ValueError("regime timeline is not unique by UTC hour")
    for name in (
        "anchor_source_utc",
        "transition_start_utc",
        "transition_end_utc",
        "target_available_utc",
    ):
        events[name] = pd.to_datetime(events[name], utc=True, errors="raise")
    if events["event_id"].duplicated().any():
        raise ValueError("transition event IDs are not unique")
    return hourly, events


def feature_shift_analysis(
    hourly: pd.DataFrame,
    weeks: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = _feature_columns(hourly)
    local = hourly.loc[:, ["source_utc", *features]].copy()
    local["week_start_utc"] = _period_key(local["source_utc"], "week")
    required_weeks = pd.to_datetime(weeks["period_start_utc"], utc=True)
    local = local.loc[local["week_start_utc"].isin(required_weeks)].copy()
    weekly = local.groupby("week_start_utc", sort=True)[list(features)].mean()
    labels = (
        weeks.set_index(pd.to_datetime(weeks["period_start_utc"], utc=True))[
            "worst_week"
        ]
        .reindex(weekly.index)
        .astype(bool)
        .to_numpy()
    )
    finite_columns = weekly.columns[
        np.isfinite(weekly.to_numpy(float)).all(axis=0)
        & weekly.nunique(dropna=False).gt(1).to_numpy()
    ]
    weekly = weekly.loc[:, finite_columns]
    scaled, _, _ = _robust_scale(weekly, ~labels)
    values = scaled.to_numpy(float)
    p_values = exact_label_permutation_pvalues(values, labels)
    bad_mean = values[labels].mean(axis=0)
    regular_mean = values[~labels].mean(axis=0)
    regular_low = np.quantile(values[~labels], 0.10, axis=0)
    regular_high = np.quantile(values[~labels], 0.90, axis=0)
    direction = np.sign(bad_mean - regular_mean)
    recurrence = np.where(
        direction >= 0,
        (values[labels] > regular_high).mean(axis=0),
        (values[labels] < regular_low).mean(axis=0),
    )
    shifts = pd.DataFrame(
        {
            "feature": weekly.columns,
            "feature_family": [
                (
                    "transition_dynamic"
                    if name.startswith(("transition_new__", "mkt_regime_change__"))
                    else "latent_state_context"
                    if name.startswith("state_context__")
                    else "market_level_or_composite"
                )
                for name in weekly.columns
            ],
            "worst_mean_robust_z": bad_mean,
            "regular_mean_robust_z": regular_mean,
            "worst_minus_regular_robust_z": bad_mean - regular_mean,
            "exact_week_permutation_p": p_values,
            "single_feature_oriented_auc": [
                _oriented_auc(values[:, position], labels)
                for position in range(values.shape[1])
            ],
            "worst_week_outside_regular_10_90_recurrence": recurrence,
            "worst_weeks": int(labels.sum()),
            "regular_weeks": int((~labels).sum()),
        }
    )
    shifts["bh_q"] = bh_fdr(shifts["exact_week_permutation_p"])
    shifts["significant_and_recurrent"] = (
        shifts["bh_q"].le(0.10)
        & shifts["worst_week_outside_regular_10_90_recurrence"].ge(0.60)
    )
    shifts = shifts.sort_values(
        [
            "significant_and_recurrent",
            "bh_q",
            "worst_minus_regular_robust_z",
        ],
        ascending=[False, True, False],
        key=lambda column: (
            column.abs()
            if column.name == "worst_minus_regular_robust_z"
            else column
        ),
        kind="mergesort",
    ).reset_index(drop=True)

    top = shifts.head(min(20, len(shifts)))["feature"].tolist()
    # Standardize hourly values using regular-week hours, then compute one
    # covariance estimate per week. Week remains the inferential unit.
    hourly_top = local.loc[:, ["week_start_utc", *top]].copy()
    hour_labels = hourly_top["week_start_utc"].map(
        dict(zip(weekly.index, labels))
    ).astype(bool)
    standardized, _, _ = _robust_scale(
        hourly_top[top], (~hour_labels).to_numpy()
    )
    standardized["week_start_utc"] = hourly_top["week_start_utc"].to_numpy()
    pair_names = list(itertools.combinations(top, 2))
    covariance_rows = []
    interaction_matrix = []
    covariance_matrix = []
    week_index = weekly.index
    for week in week_index:
        block = standardized.loc[standardized["week_start_utc"].eq(week), top]
        cov = block.cov()
        covariance_matrix.append(
            [float(cov.loc[left, right]) for left, right in pair_names]
        )
        means = block.mean(axis=0)
        interaction_matrix.append(
            [float((block[left] * block[right]).mean()) for left, right in pair_names]
        )
    covariance_values = np.asarray(covariance_matrix, dtype=float)
    interaction_values = np.asarray(interaction_matrix, dtype=float)
    for kind, matrix in (
        ("covariance", covariance_values),
        ("standardized_product_interaction", interaction_values),
    ):
        finite = np.isfinite(matrix).all(axis=0)
        p = np.full(len(pair_names), np.nan)
        p[finite] = exact_label_permutation_pvalues(matrix[:, finite], labels)
        q = bh_fdr(p)
        for position, (left, right) in enumerate(pair_names):
            if not finite[position]:
                continue
            bad = float(matrix[labels, position].mean())
            regular = float(matrix[~labels, position].mean())
            main_auc = max(
                float(
                    shifts.set_index("feature").loc[
                        [left, right], "single_feature_oriented_auc"
                    ].max()
                ),
                0.5,
            )
            pair_auc = _oriented_auc(matrix[:, position], labels)
            covariance_rows.append(
                {
                    "diagnostic_kind": kind,
                    "left_feature": left,
                    "right_feature": right,
                    "worst_mean": bad,
                    "regular_mean": regular,
                    "worst_minus_regular": bad - regular,
                    "exact_week_permutation_p": p[position],
                    "bh_q": q[position],
                    "oriented_pair_auc": pair_auc,
                    "best_single_feature_auc": main_auc,
                    "incremental_auc_vs_best_main": pair_auc - main_auc,
                    "significant_and_distinguishing": bool(
                        q[position] <= 0.10
                        and pair_auc >= 0.70
                        and pair_auc - main_auc >= 0.03
                    ),
                }
            )
    pair_table = pd.DataFrame(covariance_rows).sort_values(
        ["diagnostic_kind", "significant_and_distinguishing", "bh_q"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    covariance = pair_table.loc[
        pair_table["diagnostic_kind"].eq("covariance")
    ].reset_index(drop=True)
    interactions = pair_table.loc[
        pair_table["diagnostic_kind"].eq("standardized_product_interaction")
    ].reset_index(drop=True)
    return shifts, covariance, interactions


def regime_performance(
    rows: pd.DataFrame,
    hourly: pd.DataFrame,
    weeks: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = rows.copy()
    work["week_start_utc"] = _period_key(work["__ts__"], "week")
    eligible = set(pd.to_datetime(weeks["period_start_utc"], utc=True))
    work = work.loc[work["week_start_utc"].isin(eligible)].copy()
    selected_parts = []
    for week, local in work.groupby("week_start_utc", sort=True):
        mask = stable_global_top_mask(local, local[SCORE])
        part = local.loc[mask].copy()
        part["week_start_utc"] = week
        selected_parts.append(part)
    selected = pd.concat(selected_parts, ignore_index=True)
    context_columns = [
        "source_utc",
        "state_context__current_state",
        "state_context__nearest_distance",
        "state_context__top2_margin",
        "state_context__switch_count_6h",
        "state_context__state_age_hours",
        "target__phase",
        "target__transition_active",
        "target__destination_state",
        "target__transition_archetype",
    ]
    selected = selected.merge(
        hourly.loc[:, context_columns],
        left_on="__ts__",
        right_on="source_utc",
        how="left",
        validate="many_to_one",
    )
    selected = selected.loc[selected["state_context__current_state"].notna()].copy()
    if selected.empty:
        raise ValueError("no selected stack rows overlap frozen state context")
    selected["worst_week"] = selected["week_start_utc"].map(
        weeks.set_index(pd.to_datetime(weeks["period_start_utc"], utc=True))[
            "worst_week"
        ]
    ).astype(bool)

    records = []
    for dimension, column in (
        ("market_state", "state_context__current_state"),
        ("transition_phase", "target__phase"),
        ("transition_active", "target__transition_active"),
    ):
        for value, local in selected.groupby(column, dropna=False, sort=True):
            records.append(
                {
                    "dimension": dimension,
                    "value": str(value),
                    "selected_rows": len(local),
                    "weeks": int(local["week_start_utc"].nunique()),
                    "alpha_rank_ic": _rank_ic(local[SCORE], local[ALPHA_TARGET]),
                    "execution_net_rank_ic": _rank_ic(
                        local[SCORE], local[NET_TARGET]
                    ),
                    "mean_gross_bps": float(local[GROSS].mean() * 10_000.0),
                    "mean_net_bps": float(local[NET_TARGET].mean() * 10_000.0),
                    "positive_net_rate": float(local[NET_TARGET].gt(0.0).mean()),
                    "long_share": float(local["side_name"].eq("long").mean()),
                    "worst_week_share": float(local["worst_week"].mean()),
                }
            )
    performance = pd.DataFrame(records)
    state = performance.loc[performance["dimension"].eq("market_state")].copy()
    supported = state["selected_rows"].ge(500) & state["weeks"].ge(3)
    state["performance_qualification"] = "insufficient_support"
    if supported.any():
        baseline = float(
            np.average(
                state.loc[supported, "mean_net_bps"],
                weights=state.loc[supported, "selected_rows"],
            )
        )
        state.loc[
            supported & state["mean_net_bps"].ge(baseline + 15.0),
            "performance_qualification",
        ] = "relatively_better_conversion"
        state.loc[
            supported & state["mean_net_bps"].le(baseline - 15.0),
            "performance_qualification",
        ] = "relatively_poor_conversion"
        state.loc[
            supported
            & state["performance_qualification"].eq("insufficient_support"),
            "performance_qualification",
        ] = "near_stack_average"
        qualification = state.loc[
            :,
            [
                "value",
                "selected_rows",
                "weeks",
                "mean_net_bps",
                "alpha_rank_ic",
                "execution_net_rank_ic",
                "performance_qualification",
            ],
        ].copy()
    else:
        qualification = state.copy()
    return performance, qualification


def _period_start(period_type: str, value: object) -> pd.Timestamp:
    """Return the UTC start for a reporting period emitted by ``_period_key``."""

    if period_type == "week":
        return pd.Timestamp(value)
    if period_type == "month":
        return pd.Timestamp(f"{value}-01", tz="UTC")
    raise ValueError(f"unsupported period type: {period_type}")


def _selected_global_period_books(rows: pd.DataFrame) -> pd.DataFrame:
    """Select each weekly/monthly book globally before any category grouping."""

    selected_parts: list[pd.DataFrame] = []
    for period_type in ("week", "month"):
        work = rows.copy()
        work["__period__"] = _period_key(work["__ts__"], period_type)
        for period, local in work.groupby("__period__", sort=True):
            chosen = local.loc[stable_global_top_mask(local, local[SCORE])].copy()
            chosen["period_type"] = period_type
            chosen["period"] = str(period)
            chosen["period_start_utc"] = _period_start(period_type, period)
            chosen["period_candidate_rows"] = int(len(local))
            chosen["period_selected_rows"] = int(len(chosen))
            selected_parts.append(chosen.drop(columns="__period__"))
    if not selected_parts:
        raise ValueError("no candidate rows available for global period books")
    return pd.concat(selected_parts, ignore_index=True)


def _descriptive_context(
    selected: pd.DataFrame, hourly: pd.DataFrame) -> pd.DataFrame:
    """Attach observable state labels, preserving non-coverage as unavailable.

    ``target__phase`` and related transition fields remain attribution labels.
    They are explicitly not causal model inputs and never control selection.
    """

    optional = (
        "state_context__current_state",
        "target__phase",
        "target__transition_active",
        "target__transition_archetype",
    )
    context_columns = ["source_utc", *[name for name in optional if name in hourly]]
    result = selected.merge(
        hourly.loc[:, context_columns],
        left_on="__ts__",
        right_on="source_utc",
        how="left",
        validate="many_to_one",
    )
    result["regime_timeline_available"] = (
        result["source_utc"].notna()
        & result.get("state_context__current_state", pd.Series(index=result.index, dtype=float)).notna()
    )
    result["transition_attribution_available"] = (
        result["source_utc"].notna()
        & result.get("target__phase", pd.Series(index=result.index, dtype=object)).notna()
    )
    result["market_state"] = result.get(
        "state_context__current_state", pd.Series(index=result.index, dtype=object)
    ).where(result["regime_timeline_available"], "unavailable").astype(str)
    result["transition_phase_attribution"] = result.get(
        "target__phase", pd.Series(index=result.index, dtype=object)
    ).where(result["transition_attribution_available"], "unavailable").astype(str)
    active = result.get("target__transition_active", pd.Series(index=result.index, dtype=object))
    result["transition_active_attribution"] = active.where(
        result["transition_attribution_available"], "unavailable"
    ).astype(str)
    archetype = result.get("target__transition_archetype", pd.Series(index=result.index, dtype=object))
    result["transition_archetype_attribution"] = archetype.where(
        result["transition_attribution_available"], "unavailable"
    ).fillna("none_or_unlabeled").astype(str)
    return result


def side_state_transition_period_metrics(
    rows: pd.DataFrame, hourly: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Emit descriptive side/state/transition attribution after global selection.

    Categories are never used to form a book.  Every book is selected once
    globally at the reporting frequency, then partitioned for attribution.
    This preserves both the global top-k contract and unavailable state hours.
    """

    selected = _descriptive_context(_selected_global_period_books(rows), hourly)
    keys = [
        "period_type",
        "period",
        "period_start_utc",
        "lineage_id",
        "evidence_grade",
        "side_name",
        "regime_timeline_available",
        "transition_attribution_available",
        "market_state",
        "transition_phase_attribution",
        "transition_active_attribution",
        "transition_archetype_attribution",
    ]
    records: list[dict[str, Any]] = []
    for values, local in selected.groupby(keys, dropna=False, sort=True):
        record = dict(zip(keys, values))
        record.update(
            {
                "selected_rows": int(len(local)),
                "period_candidate_rows": int(local["period_candidate_rows"].iloc[0]),
                "period_selected_rows": int(local["period_selected_rows"].iloc[0]),
                "alpha_rank_ic": _rank_ic(local[SCORE], local[ALPHA_TARGET]),
                "execution_net_rank_ic": _rank_ic(local[SCORE], local[NET_TARGET]),
                "mean_gross_bps": float(local[GROSS].mean() * 10_000.0),
                "mean_cost_bps": float(local[COST].mean() * 10_000.0),
                "mean_net_bps": float(local[NET_TARGET].mean() * 10_000.0),
                "positive_net_rate": float(local[NET_TARGET].gt(0.0).mean()),
                "selection_scope": "one pooled-global top10 within period; category attribution after selection",
                "state_transition_role": "descriptive attribution only; never a selection or policy gate",
            }
        )
        records.append(record)
    metrics = pd.DataFrame.from_records(records).sort_values(keys, kind="mergesort")

    baseline = (
        selected.groupby(
            ["period_type", "period", "period_start_utc", "lineage_id", "evidence_grade", "side_name"],
            sort=True,
        )[NET_TARGET]
        .mean()
        .mul(10_000.0)
        .rename("side_period_baseline_net_bps")
        .reset_index()
    )
    metrics = metrics.merge(
        baseline,
        on=["period_type", "period", "period_start_utc", "lineage_id", "evidence_grade", "side_name"],
        how="left",
        validate="many_to_one",
    )
    metrics["net_bps_delta_vs_side_period_baseline"] = (
        metrics["mean_net_bps"] - metrics["side_period_baseline_net_bps"]
    )

    stability_keys = [
        "period_type",
        "evidence_grade",
        "side_name",
        "regime_timeline_available",
        "transition_attribution_available",
        "market_state",
        "transition_phase_attribution",
        "transition_active_attribution",
        "transition_archetype_attribution",
    ]
    stability_records: list[dict[str, Any]] = []
    for values, local in metrics.groupby(stability_keys, dropna=False, sort=True):
        record = dict(zip(stability_keys, values))
        delta = local["net_bps_delta_vs_side_period_baseline"]
        total_rows = int(local["selected_rows"].sum())
        period_count = int(local["period"].nunique())
        positive_delta_share = float(delta.gt(0.0).mean())
        negative_delta_share = float(delta.lt(0.0).mean())
        mean_delta = float(np.average(delta, weights=local["selected_rows"]))
        support_adequate = bool(period_count >= 3 and total_rows >= 500)
        direction_share = max(positive_delta_share, negative_delta_share)
        stable_effect = bool(
            support_adequate and direction_share >= 2.0 / 3.0 and abs(mean_delta) >= 15.0
        )
        if not support_adequate:
            qualification = "insufficient_support"
        elif not stable_effect:
            qualification = "unstable_or_small_relative_effect"
        elif mean_delta > 0.0:
            qualification = "stable_relatively_better_research_only"
        else:
            qualification = "stable_relatively_poor_research_only"
        record.update(
            {
                "period_count": period_count,
                "selected_rows": total_rows,
                "net_ev_bps_q10": float(local["mean_net_bps"].quantile(0.10)),
                "net_ev_bps_q50": float(local["mean_net_bps"].quantile(0.50)),
                "positive_period_share": float(local["mean_net_bps"].gt(0.0).mean()),
                "mean_delta_vs_side_period_baseline_bps": mean_delta,
                "positive_delta_period_share": positive_delta_share,
                "negative_delta_period_share": negative_delta_share,
                "support_adequate": support_adequate,
                "stable_directional_effect": stable_effect,
                "performance_qualification": qualification,
                "promotion_status": "diagnostic_only_requires_independent_strict_2026_confirmation",
            }
        )
        stability_records.append(record)
    stability = pd.DataFrame.from_records(stability_records).sort_values(
        stability_keys, kind="mergesort"
    )
    return metrics, stability, selected


def composition_within_category_decomposition(
    selected: pd.DataFrame, weeks: pd.DataFrame
) -> pd.DataFrame:
    """Decompose worst-week payoff shifts into composition and within-cell terms."""

    week = selected.loc[selected["period_type"].eq("week")].copy()
    labels = weeks.loc[:, ["period_start_utc", "worst_week"]].drop_duplicates()
    week = week.merge(labels, on="period_start_utc", how="inner", validate="many_to_one")
    if week["worst_week"].nunique() < 2:
        raise ValueError("worst/regular week labels are both required for decomposition")
    category = ["side_name", "market_state", "transition_phase_attribution"]
    output: list[dict[str, Any]] = []
    for metric_name, column in (("gross", GROSS), ("cost", COST), ("net", NET_TARGET)):
        bad = week.loc[week["worst_week"]]
        regular = week.loc[~week["worst_week"]]
        bad_group = bad.groupby(category, dropna=False)[column].agg(["size", "mean"])
        regular_group = regular.groupby(category, dropna=False)[column].agg(["size", "mean"])
        index = bad_group.index.union(regular_group.index)
        bad_count = bad_group.reindex(index)["size"].fillna(0.0)
        regular_count = regular_group.reindex(index)["size"].fillna(0.0)
        bad_mean = bad_group.reindex(index)["mean"]
        regular_mean = regular_group.reindex(index)["mean"]
        # Use the observed mean in the population that contains the cell.  This
        # yields an exact Oaxaca-style recomposition even for new/disappearing
        # state/transition cells.
        regular_reference = regular_mean.where(regular_count.gt(0.0), bad_mean)
        bad_reference = bad_mean.where(bad_count.gt(0.0), regular_mean)
        bad_share = bad_count / max(float(len(bad)), 1.0)
        regular_share = regular_count / max(float(len(regular)), 1.0)
        composition = float(((bad_share - regular_share) * regular_reference).sum() * 10_000.0)
        within = float((bad_share * (bad_reference - regular_reference)).sum() * 10_000.0)
        observed = float((bad[column].mean() - regular[column].mean()) * 10_000.0)
        output.append(
            {
                "metric": metric_name,
                "worst_week_selected_rows": int(len(bad)),
                "regular_week_selected_rows": int(len(regular)),
                "worst_minus_regular_bps": observed,
                "composition_effect_bps": composition,
                "within_category_payoff_effect_bps": within,
                "recomposition_error_bps": observed - composition - within,
                "category_contract": "side × observable market state × descriptive transition phase; unavailable is its own category",
                "selection_scope": "weekly pooled-global top10 before decomposition",
            }
        )
    return pd.DataFrame.from_records(output)


def asset_exit_attribution(
    rows: pd.DataFrame, weeks: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Attribute already-selected books to assets/exits without inventing exits.

    Selection is deliberately delegated to ``_selected_global_period_books``.
    Exit reasons are only described where the source actually carries an exact
    policy exit field; reconstructed/backcast rows remain an explicit
    ``unavailable`` category rather than a synthetic timeout/stop label.
    """
    selected = _selected_global_period_books(rows)
    selected["asset_attribution_available"] = selected["__symbol__"].notna()
    if "execution_exit_reason" in selected:
        selected["exit_reason_attribution_available"] = selected["execution_exit_reason"].notna()
        selected["exit_reason_attribution"] = selected["execution_exit_reason"].where(
            selected["exit_reason_attribution_available"], "unavailable"
        ).astype(str)
    else:
        selected["exit_reason_attribution_available"] = False
        selected["exit_reason_attribution"] = "unavailable"
    key = [
        "period_type", "period", "period_start_utc", "lineage_id",
        "evidence_grade",
    ]
    # A book is selected pooled-global at the reporting-period level, before
    # lineage/evidence attribution.  Compute that denominator once: repeating
    # a boolean scan of ``selected`` for every asset/exit group is quadratic
    # on a long history and, more importantly, would describe a local lineage
    # share rather than the globally selected book share.
    book_key = ["period_type", "period", "period_start_utc"]
    period_denominators = (
        selected.groupby(book_key, sort=True, dropna=False)
        .size()
        .rename("_selected_book_rows")
        .reset_index()
    )
    book_rows: list[dict[str, Any]] = []
    for values, local in selected.groupby(key, sort=True, dropna=False):
        record = dict(zip(key, values))
        asset_share = local["__symbol__"].value_counts(normalize=True, dropna=False)
        record.update(
            {
                "period_candidate_rows": int(local["period_candidate_rows"].iloc[0]),
                "period_selected_rows": int(len(local)),
                "asset_attribution_available": bool(local["asset_attribution_available"].all()),
                "exit_reason_attribution_available": bool(local["exit_reason_attribution_available"].all()),
                "selected_assets": int(local["__symbol__"].nunique(dropna=True)),
                "largest_asset_share": float(asset_share.iloc[0]) if len(asset_share) else np.nan,
                "asset_hhi": float((asset_share**2).sum()) if len(asset_share) else np.nan,
                "mean_gross_bps": float(local[GROSS].mean() * 10_000.0),
                "mean_cost_bps": float(local[COST].mean() * 10_000.0),
                "mean_net_bps": float(local[NET_TARGET].mean() * 10_000.0),
                "net_trade_q10_bps": float(local[NET_TARGET].quantile(.10) * 10_000.0),
                "net_trade_q50_bps": float(local[NET_TARGET].quantile(.50) * 10_000.0),
                "selection_scope": "one pooled-global top10 within period before asset/exit attribution",
            }
        )
        book_rows.append(record)
    book = pd.DataFrame.from_records(book_rows)

    records: list[dict[str, Any]] = []
    for kind, column, available in (
        ("asset", "__symbol__", "asset_attribution_available"),
        ("exit_reason", "exit_reason_attribution", "exit_reason_attribution_available"),
    ):
        for values, local in selected.groupby([*key, column, available], sort=True, dropna=False):
            record = dict(zip([*key, "attribution_value", f"{kind}_attribution_available"], values))
            record.update(
                {
                    "attribution_kind": kind,
                    "selected_rows": int(len(local)),
                    "period_candidate_rows": int(local["period_candidate_rows"].iloc[0]),
                    "period_selected_rows": int(local["period_selected_rows"].iloc[0]),
                    "mean_gross_bps": float(local[GROSS].mean() * 10_000.0),
                    "mean_cost_bps": float(local[COST].mean() * 10_000.0),
                    "mean_net_bps": float(local[NET_TARGET].mean() * 10_000.0),
                    "net_trade_q10_bps": float(local[NET_TARGET].quantile(.10) * 10_000.0),
                    "net_trade_q50_bps": float(local[NET_TARGET].quantile(.50) * 10_000.0),
                    "selection_scope": "one pooled-global top10 within period before asset/exit attribution",
                }
            )
            records.append(record)
    period = pd.DataFrame.from_records(records).merge(
        period_denominators, on=book_key, how="left", validate="many_to_one"
    )
    if period["_selected_book_rows"].isna().any():
        raise ValueError("missing pooled-global selected-book denominator")
    period["selected_book_share"] = (
        period["selected_rows"] / period["_selected_book_rows"]
    )
    period = period.drop(columns="_selected_book_rows")

    labels = weeks.loc[:, ["period_start_utc", "lineage_id", "evidence_grade", "worst_week"]].drop_duplicates()
    weekly = selected.loc[selected["period_type"].eq("week")].merge(
        labels, on=["period_start_utc", "lineage_id", "evidence_grade"],
        how="inner", validate="many_to_one",
    )
    decomposition: list[dict[str, Any]] = []
    for kind, column in (("asset", "__symbol__"), ("exit_reason", "exit_reason_attribution")):
        for (lineage, grade), scope in weekly.groupby(["lineage_id", "evidence_grade"], sort=True):
            if scope["worst_week"].nunique() < 2:
                continue
            for metric, value in (("gross", GROSS), ("cost", COST), ("net", NET_TARGET)):
                bad, regular = scope.loc[scope["worst_week"]], scope.loc[~scope["worst_week"]]
                bg, rg = bad.groupby(column, dropna=False)[value].agg(["size", "mean"]), regular.groupby(column, dropna=False)[value].agg(["size", "mean"])
                index = bg.index.union(rg.index); bc, rc = bg.reindex(index)["size"].fillna(0.), rg.reindex(index)["size"].fillna(0.)
                bm, rm = bg.reindex(index)["mean"], rg.reindex(index)["mean"]
                regular_reference = rm.where(rc.gt(0), bm); bad_reference = bm.where(bc.gt(0), rm)
                bshare, rshare = bc / len(bad), rc / len(regular)
                composition = float(((bshare-rshare)*regular_reference).sum()*10_000.)
                within = float((bshare*(bad_reference-regular_reference)).sum()*10_000.)
                observed = float((bad[value].mean()-regular[value].mean())*10_000.)
                decomposition.append({"attribution_kind":kind,"lineage_id":lineage,"evidence_grade":grade,"metric":metric,"worst_week_selected_rows":len(bad),"regular_week_selected_rows":len(regular),"worst_minus_regular_bps":observed,"composition_effect_bps":composition,"within_category_payoff_effect_bps":within,"recomposition_error_bps":observed-composition-within,"availability_contract":"exit reasons are unavailable rather than fabricated when not present in the source","selection_scope":"weekly pooled-global top10 before decomposition"})
    return period.sort_values(["attribution_kind", *key, "attribution_value"], kind="mergesort"), book.sort_values(key, kind="mergesort"), pd.DataFrame.from_records(decomposition)


def event_overlap(events: pd.DataFrame, weeks: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, week in weeks.iterrows():
        start = pd.Timestamp(week["period_start_utc"])
        end = pd.Timestamp(week["period_end_exclusive_utc"])
        local = events.loc[
            events["transition_start_utc"].lt(end)
            & events["transition_end_utc"].ge(start)
        ]
        records.append(
            {
                "week_start_utc": start,
                "worst_week": bool(week["worst_week"]),
                "transition_event_count": int(len(local)),
                "transition_archetypes": ",".join(
                    sorted(local["transition_archetype"].astype(str).unique())
                ),
                "robust_transition_events": int(
                    local["robust_pre_post_shift"].fillna(False).astype(bool).sum()
                ),
            }
        )
    return pd.DataFrame(records)


def run(args: argparse.Namespace) -> Path:
    if args.output.exists():
        raise FileExistsError(f"immutable output exists: {args.output}")
    performance = pd.read_parquet(
        args.performance_calendar / "performance_period_metrics.parquet"
    )
    weeks = identify_worst_weeks(performance, quantile=args.worst_quantile)
    rows, _ = load_strict_stack_rows(args.source_2025, args.source_2026)
    reconstructed, _ = load_reconstructed_stack_rows(args.source_backfill)
    rows = pd.concat([reconstructed, rows], ignore_index=True)
    hourly, events = _load_regime(args.regime_root)
    shifts, covariance, interactions = feature_shift_analysis(hourly, weeks)
    regime, qualification = regime_performance(rows, hourly, weeks)
    category_periods, category_stability, selected_books = side_state_transition_period_metrics(
        rows, hourly
    )
    decomposition = composition_within_category_decomposition(selected_books, weeks)
    asset_exit_periods, asset_exit_books, asset_exit_decomposition = asset_exit_attribution(
        rows, weeks
    )
    overlaps = event_overlap(events, weeks)
    worst_calendar = weeks.merge(
        overlaps, left_on="period_start_utc", right_on="week_start_utc",
        how="left", validate="one_to_one",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(dir=args.output.parent, prefix=f".{args.output.name}.")
    )
    try:
        frames = {
            "worst_week_calendar.csv": worst_calendar,
            "regime_performance.csv": regime,
            "state_performance_qualification.csv": qualification,
            "side_state_transition_period_metrics.csv": category_periods,
            "side_state_transition_stability_qualification.csv": category_stability,
            "composition_within_category_decomposition.csv": decomposition,
            "asset_exit_period_attribution.csv": asset_exit_periods,
            "asset_exit_book_concentration.csv": asset_exit_books,
            "asset_exit_worst_regular_decomposition.csv": asset_exit_decomposition,
            "weekly_feature_shifts.csv": shifts,
            "weekly_covariance_shifts.csv": covariance,
            "weekly_interaction_shifts.csv": interactions,
        }
        hashes = {}
        for name, frame in frames.items():
            path = temporary / name
            frame.to_csv(path, index=False)
            hashes[name] = sha256_file(path)
        manifest = {
            "schema": SCHEMA,
            "status": "MATERIALIZED_STRICT_STACK_WORST_PERIOD_DIAGNOSTIC",
            "strict_stack_complete_weeks": len(weeks),
            "worst_weeks": int(weeks["worst_week"].sum()),
            "worst_definition": weeks["worst_definition"].iloc[0],
            "inference_unit": "UTC calendar week",
            "feature_families_tested": sorted(shifts["feature_family"].unique()),
            "features_tested": len(shifts),
            "covariance_pairs_tested": len(covariance),
            "interaction_pairs_tested": len(interactions),
            "significance_contract": (
                "two-sided exact week-label permutation with Benjamini-Hochberg "
                "q<=0.10; exploratory effects remain reported when none pass"
            ),
            "regime_contract": {
                "source": str(args.regime_root.resolve()),
                "five_state_geometry": True,
                "market_timeline_status": "research_only_not_a_trade_gate",
                "target_phase_fields": "descriptive labels only, never model inputs",
            },
            "performance_contract": {
                "score": SCORE,
                "selection": "one pooled-global top10 within each complete week",
                "economics": "exact 1m deployed-policy 12h gross-cost=net",
                "promotion_eligible": False,
            },
            "category_reporting_contract": {
                "model_sample_cadence": "1h",
                "assessment_sample_cadence": "1h",
                "exact_replay_bar_cadence": "1m_labels_only",
                "selection": (
                    "one pooled-global top10 is selected before side/state/transition "
                    "attribution at each weekly or monthly reporting frequency"
                ),
                "unavailable_context": "retained as unavailable; never imputed as stable",
                "transition_fields": "descriptive attribution only; never causal model inputs",
                "qualification": (
                    "support >=3 periods and >=500 selected rows, directional relative "
                    "effect >=2/3 periods and >=15bps; research-only, not promotion"
                ),
            },
            "composition_decomposition_contract": {
                "unit": "selected candidate row in globally selected weekly book",
                "effects": "worst-minus-regular gross/cost/net = composition + within-category payoff",
                "category": "side × observable state × descriptive transition phase",
            },
            "asset_exit_reporting_contract": {
                "selection": "the same pooled-global top10 book is selected before asset and exit-reason attribution",
                "periods": "week and month; Q10/Q50 are selected-candidate net-return quantiles",
                "concentration": "largest selected-asset share and HHI are reported per book",
                "exit_reason_availability": "only exact-source rows with execution_exit_reason are attributed; unavailable reconstructed rows are retained as unavailable and never imputed",
                "decomposition": "worst-minus-regular gross/cost/net is reported separately per lineage/evidence grade and attribution kind; no cross-grade economic pooling",
            },
            "outputs_sha256": hashes,
            "runner_sha256": sha256_file(Path(__file__).resolve()),
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(
            json.dumps(json_safe(manifest), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / "manifest.sha256").write_text(
            f"{sha256_file(manifest_path)}  manifest.json\n", encoding="utf-8"
        )
        os.replace(temporary, args.output)
        return args.output
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--performance-calendar", type=Path, default=DEFAULT_CALENDAR)
    parser.add_argument("--regime-root", type=Path, default=DEFAULT_REGIME)
    parser.add_argument("--source-2025", type=Path, default=DEFAULT_2025)
    parser.add_argument("--source-2026", type=Path, default=DEFAULT_2026)
    parser.add_argument("--source-backfill", type=Path, default=DEFAULT_BACKFILL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--worst-quantile", type=float, default=0.25)
    args = parser.parse_args(argv)
    if not 0.10 <= args.worst_quantile <= 0.50:
        parser.error("--worst-quantile must be in [0.10, 0.50]")
    return args


if __name__ == "__main__":
    run(parse_args())
