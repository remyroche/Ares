#!/usr/bin/env python3
"""Audit market-state threshold-controller promotion gates.

This is a replay-free consistency audit. It reads the persisted walk-forward
selection table and selected-controller JSON, recomputes the conservative
promotion gates, and verifies that the stored non-promotion verdict follows
from the metrics.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


BASELINE_ARM = "S0_baseline_static_thresholds"
STAGE1_OBSERVED_ONLY_ARMS = {"S1_observed_axes_shared_response"}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def _normalise_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no", "", "nan", "none"}:
        return False
    return bool(value)


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame.fillna("").astype(str)
    columns = list(view.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(row[col] for col in columns) + " |")
    return "\n".join(lines)


def _controller_arm_complexity(arm: str) -> int:
    base = str(arm).replace("__post_selection_overlay", "")
    order = {
        "S1_observed_axes_shared_response": 1,
        "S2_observed_forecast_shared_response": 2,
        "S7_pruned_state_pack": 2,
        "S3_observed_forecast_latent_shared_response": 3,
        "S4_S3_plus_per_strategy_residual": 4,
    }
    return int(order.get(base, 99))


def _base_arm(arm: Any) -> str:
    return str(arm).replace("__post_selection_overlay", "")


def _filter_selection_scope(
    selection: pd.DataFrame,
    *,
    allowed_base_arms: set[str] | None,
) -> pd.DataFrame:
    if not allowed_base_arms:
        return selection.copy()
    allowed = {str(arm) for arm in allowed_base_arms}
    work = selection.copy()
    base = work["arm"].map(_base_arm)
    keep = work["arm"].astype(str).eq(BASELINE_ARM) | base.isin(allowed)
    return work.loc[keep].copy()


def _as_reason_set(value: Any) -> set[str]:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return set()
    text = str(value)
    if not text:
        return set()
    return {item for item in text.split(";") if item}


def _policy_value(policy: dict[str, Any], key: str, default: Any) -> Any:
    return policy.get(key, default)


def _enrich_selection_with_safety_metrics(selection: pd.DataFrame, artifact_dir: Path) -> pd.DataFrame:
    """Backfill safety metrics for older selection tables from fold summaries."""

    needed = {
        "median_trade_retention_share",
        "median_delta_full_sl_rate",
    }
    if needed.issubset(selection.columns):
        return _enrich_post_selection_safety_metrics(selection)
    summary_path = artifact_dir / "walkforward_summary.csv"
    if not summary_path.exists():
        return selection
    try:
        summary = pd.read_csv(summary_path)
    except Exception:
        return selection
    required = {"fold", "arm", "trade_count", "full_sl_rate"}
    if not required.issubset(summary.columns):
        return selection
    base = summary.loc[summary["arm"].astype(str).eq(BASELINE_ARM), ["fold", "trade_count", "full_sl_rate"]].copy()
    if base.empty:
        return selection
    base = base.rename(columns={"trade_count": "base_trade_count", "full_sl_rate": "base_full_sl_rate"})
    merged = summary.merge(base, on="fold", how="left")
    base_trade_count = pd.to_numeric(merged["base_trade_count"], errors="coerce").replace(0.0, np.nan)
    merged["_trade_retention_share"] = pd.to_numeric(merged["trade_count"], errors="coerce") / base_trade_count
    merged.loc[merged["arm"].astype(str).eq(BASELINE_ARM), "_trade_retention_share"] = 1.0
    merged["_delta_full_sl_rate"] = (
        pd.to_numeric(merged["full_sl_rate"], errors="coerce")
        - pd.to_numeric(merged["base_full_sl_rate"], errors="coerce")
    )
    safety = (
        merged.groupby("arm", sort=False)
        .agg(
            median_trade_retention_share=("_trade_retention_share", "median"),
            median_delta_full_sl_rate=("_delta_full_sl_rate", "median"),
        )
        .reset_index()
    )
    out = selection.copy()
    for column in ["median_trade_retention_share", "median_delta_full_sl_rate"]:
        if column in out.columns:
            safety = safety.drop(columns=[column])
    if len(safety.columns) <= 1:
        return out
    out = out.merge(safety, on="arm", how="left")
    return _enrich_post_selection_safety_metrics(out)


def _enrich_post_selection_safety_metrics(selection: pd.DataFrame) -> pd.DataFrame:
    out = selection.copy()
    if "base_arm" not in out.columns:
        out["base_arm"] = out["arm"].astype(str).str.replace("__post_selection_overlay", "", regex=False)
    if "is_post_selection_overlay" not in out.columns:
        out["is_post_selection_overlay"] = out["arm"].astype(str).str.endswith("__post_selection_overlay")
    safety_cols = [
        "median_delta_max_drawdown",
        "median_delta_worst_24h",
        "median_trade_retention_share",
        "median_delta_full_sl_rate",
    ]
    missing_post_cols = [f"post_selection_{col}" for col in safety_cols if f"post_selection_{col}" not in out.columns]
    if not missing_post_cols:
        return out
    overlays = out.loc[out["is_post_selection_overlay"].map(_normalise_bool), ["base_arm", *[c for c in safety_cols if c in out.columns]]].copy()
    if overlays.empty:
        return out
    rename = {col: f"post_selection_{col}" for col in safety_cols if col in overlays.columns}
    overlays = overlays.rename(columns=rename).drop_duplicates("base_arm", keep="first")
    overlays = overlays[["base_arm", *[col for col in rename.values() if col not in out.columns]]]
    if len(overlays.columns) <= 1:
        return out
    return out.merge(overlays, on="base_arm", how="left")


def _enrich_selection_with_action_metrics(selection: pd.DataFrame, artifact_dir: Path) -> pd.DataFrame:
    """Attach explicit replacement/freed-capacity metrics from action audit rows.

    Suppression-only defensive success answers whether raised thresholds avoided
    bad accepted trades.  The full portfolio effect also depends on whether
    freed capacity admitted later or alternate candidates.  These fields make
    that second component visible in the promotion audit while leaving the
    existing conservative selection gates unchanged.
    """

    out = selection.copy()
    action_path = artifact_dir / "strategy_threshold_action_audit.csv"
    if not action_path.exists():
        action_path = artifact_dir / "walkforward_threshold_action_edge_validation.csv"
    if not action_path.exists():
        return out
    try:
        action = pd.read_csv(action_path)
    except Exception:
        return out
    if action.empty or "arm" not in action.columns:
        return out

    num_cols = [
        "entrants",
        "removed",
        "entrant_net_pnl",
        "removed_net_pnl",
        "net_replacement_pnl",
        "same_key_net_pnl_delta",
        "net_action_pnl_delta",
    ]
    for column in num_cols:
        if column not in action.columns:
            action[column] = 0.0
        action[column] = pd.to_numeric(action[column], errors="coerce").fillna(0.0)

    group_cols = ["arm"]
    if "fold" in action.columns:
        group_cols.append("fold")
    fold_metrics = (
        action.groupby(group_cols, sort=False)
        .agg(
            freed_capacity_entrant_count=("entrants", "sum"),
            freed_capacity_removed_count=("removed", "sum"),
            freed_capacity_entrant_net_pnl=("entrant_net_pnl", "sum"),
            freed_capacity_removed_net_pnl=("removed_net_pnl", "sum"),
            freed_capacity_net_replacement_pnl=("net_replacement_pnl", "sum"),
            freed_capacity_same_key_net_pnl_delta=("same_key_net_pnl_delta", "sum"),
            freed_capacity_net_action_pnl_delta=("net_action_pnl_delta", "sum"),
        )
        .reset_index()
    )
    fold_metrics["_freed_capacity_positive_fold"] = (
        pd.to_numeric(fold_metrics["freed_capacity_net_action_pnl_delta"], errors="coerce").fillna(0.0)
        > 0.0
    )
    agg = (
        fold_metrics.groupby("arm", sort=False)
        .agg(
            freed_capacity_entrant_count=("freed_capacity_entrant_count", "sum"),
            freed_capacity_removed_count=("freed_capacity_removed_count", "sum"),
            freed_capacity_entrant_net_pnl=("freed_capacity_entrant_net_pnl", "sum"),
            freed_capacity_removed_net_pnl=("freed_capacity_removed_net_pnl", "sum"),
            freed_capacity_net_replacement_pnl=("freed_capacity_net_replacement_pnl", "sum"),
            freed_capacity_same_key_net_pnl_delta=("freed_capacity_same_key_net_pnl_delta", "sum"),
            freed_capacity_net_action_pnl_delta=("freed_capacity_net_action_pnl_delta", "sum"),
            median_freed_capacity_net_replacement_pnl=("freed_capacity_net_replacement_pnl", "median"),
            q25_freed_capacity_net_replacement_pnl=(
                "freed_capacity_net_replacement_pnl",
                lambda s: float(pd.to_numeric(s, errors="coerce").quantile(0.25)),
            ),
            positive_freed_capacity_fold_share=("_freed_capacity_positive_fold", "mean"),
        )
        .reset_index()
    )
    existing = [column for column in agg.columns if column in out.columns and column != "arm"]
    if existing:
        agg = agg.drop(columns=existing)
    if len(agg.columns) > 1:
        out = out.merge(agg, on="arm", how="left")

    if "base_arm" not in out.columns:
        out["base_arm"] = out["arm"].astype(str).str.replace("__post_selection_overlay", "", regex=False)
    if "is_post_selection_overlay" not in out.columns:
        out["is_post_selection_overlay"] = out["arm"].astype(str).str.endswith("__post_selection_overlay")
    action_cols = [
        column
        for column in [
            "freed_capacity_entrant_count",
            "freed_capacity_removed_count",
            "freed_capacity_entrant_net_pnl",
            "freed_capacity_removed_net_pnl",
            "freed_capacity_net_replacement_pnl",
            "freed_capacity_same_key_net_pnl_delta",
            "freed_capacity_net_action_pnl_delta",
            "median_freed_capacity_net_replacement_pnl",
            "q25_freed_capacity_net_replacement_pnl",
            "positive_freed_capacity_fold_share",
        ]
        if column in out.columns
    ]
    overlays = out.loc[
        out["is_post_selection_overlay"].map(_normalise_bool),
        ["base_arm", *action_cols],
    ].copy()
    if not overlays.empty:
        rename = {column: f"post_selection_{column}" for column in action_cols}
        overlays = overlays.rename(columns=rename).drop_duplicates("base_arm", keep="first")
        overlays = overlays[
            ["base_arm", *[column for column in rename.values() if column not in out.columns]]
        ]
        if len(overlays.columns) > 1:
            out = out.merge(overlays, on="base_arm", how="left")
    return out


def _compute_fail_reasons(row: pd.Series, policy: dict[str, Any]) -> list[str]:
    min_positive_delta_share = float(_policy_value(policy, "min_positive_delta_share", 0.50))
    min_median_delta_net_pnl = float(_policy_value(policy, "min_median_delta_net_pnl", 0.0))
    min_q25_delta_net_pnl = float(_policy_value(policy, "min_q25_delta_net_pnl", 0.0))
    min_defensive_success = float(_policy_value(policy, "min_defensive_success", 0.0))
    min_positive_suppression_share = float(_policy_value(policy, "min_positive_suppression_share", 0.50))
    max_mean_state_ood_share = float(_policy_value(policy, "max_mean_state_ood_share", 0.10))
    min_median_delta_max_drawdown = float(_policy_value(policy, "min_median_delta_max_drawdown", 0.0))
    min_median_delta_worst_24h = float(_policy_value(policy, "min_median_delta_worst_24h", 0.0))
    max_median_delta_full_sl_rate = float(_policy_value(policy, "max_median_delta_full_sl_rate", 0.0))
    min_median_trade_retention_share = float(_policy_value(policy, "min_median_trade_retention_share", 0.80))
    require_post_selection_confirmation = bool(_policy_value(policy, "require_post_selection_confirmation", True))

    arm = str(row.get("arm"))
    complexity = int(row.get("complexity", _controller_arm_complexity(arm)) or 99)
    is_baseline = _normalise_bool(row.get("is_baseline", arm == BASELINE_ARM))
    is_overlay = _normalise_bool(row.get("is_post_selection_overlay", arm.endswith("__post_selection_overlay")))
    select_no_backfill_overlay_only = bool(
        _policy_value(policy, "select_no_backfill_overlay_only", False)
    )
    reasons: list[str] = []
    if is_baseline:
        reasons.append("baseline_audit_arm")
    if is_overlay and not select_no_backfill_overlay_only:
        reasons.append("post_selection_overlay_audit_arm")
    if select_no_backfill_overlay_only and not is_overlay:
        reasons.append("full_replay_can_promote_replacements")
    if complexity >= 99:
        reasons.append("unknown_controller_arm")

    allowed_actions = {
        str(item).strip().lower()
        for item in _policy_value(
            policy,
            "allowed_controller_action_scopes",
            [
                "threshold_raises_only",
                "penalty_only_threshold_raises",
                "threshold_raise",
                "threshold_only",
                "no_backfill_threshold_raises_only",
            ],
        )
    }
    action_scope = row.get(
        "controller_action_scope",
        row.get("controller_action", row.get("action_scope", None)),
    )
    if action_scope is not None and not (isinstance(action_scope, float) and not np.isfinite(action_scope)):
        action_text = str(action_scope).strip().lower()
        if action_text and action_text not in allowed_actions:
            reasons.append("non_threshold_raise_action_scope")

    allowed_phases = {
        str(item).strip().lower()
        for item in _policy_value(
            policy,
            "allowed_controller_action_phases",
            ["c1", "phase1", "stage1", "threshold_only", "penalty_only_threshold_raises"],
        )
    }
    action_phase = row.get("controller_action_phase", row.get("action_phase", None))
    if action_phase is not None and not (isinstance(action_phase, float) and not np.isfinite(action_phase)):
        phase_text = str(action_phase).strip().lower()
        if phase_text and phase_text not in allowed_phases:
            reasons.append("later_action_phase_requires_prior_threshold_only_promotion")

    if _normalise_bool(row.get("changes_scores_or_ranks", False)) or _normalise_bool(
        row.get("changes_scores", False)
    ) or _normalise_bool(row.get("changes_ranks", False)):
        reasons.append("controller_changes_scores_or_ranks")
    if _normalise_bool(row.get("changes_auction_ordering", False)) or _normalise_bool(
        row.get("changes_auction_priority", False)
    ):
        reasons.append("controller_changes_auction_ordering")
    if _normalise_bool(row.get("changes_position_sizing", False)):
        reasons.append("position_sizing_requires_prior_threshold_only_promotion")
    if (
        _normalise_bool(row.get("allows_threshold_reductions", False))
        or _normalise_bool(row.get("can_lower_thresholds", False))
        or _normalise_bool(row.get("can_lower_threshold", False))
    ):
        reasons.append("controller_can_lower_thresholds")
    if _normalise_bool(row.get("promotes_replacement_candidates", False)) or _normalise_bool(
        row.get("directly_promotes_replacements", False)
    ):
        reasons.append("controller_promotes_replacement_candidates")

    if _finite_float(row.get("median_delta_net_pnl"), -np.inf) <= min_median_delta_net_pnl:
        reasons.append("median_delta_not_positive")
    if _finite_float(row.get("q25_delta_net_pnl"), -np.inf) < min_q25_delta_net_pnl:
        reasons.append("q25_delta_below_gate")
    if _finite_float(row.get("positive_delta_share"), 0.0) < min_positive_delta_share:
        reasons.append("insufficient_positive_folds")
    if _finite_float(row.get("median_delta_max_drawdown"), -np.inf) < min_median_delta_max_drawdown:
        reasons.append("max_drawdown_worsened")
    if _finite_float(row.get("median_delta_worst_24h"), -np.inf) < min_median_delta_worst_24h:
        reasons.append("worst_24h_worsened")
    if _finite_float(row.get("median_delta_full_sl_rate"), np.inf) > max_median_delta_full_sl_rate:
        reasons.append("full_sl_rate_worsened")
    if _finite_float(row.get("median_trade_retention_share"), 0.0) < min_median_trade_retention_share:
        reasons.append("insufficient_trade_retention")
    if _finite_float(row.get("realized_defensive_success"), 0.0) <= min_defensive_success:
        reasons.append("defensive_success_not_positive")
    if _finite_float(row.get("positive_suppression_fold_share"), 0.0) < min_positive_suppression_share:
        reasons.append("suppression_not_recurrent")
    if _finite_float(row.get("mean_state_ood_share"), 0.0) > max_mean_state_ood_share:
        reasons.append("mean_state_ood_share_too_high")
    if require_post_selection_confirmation and not is_baseline and not is_overlay:
        post_median = _finite_float(row.get("post_selection_median_delta_net_pnl"), -np.inf)
        if not np.isfinite(post_median):
            reasons.append("post_selection_confirmation_missing")
        if post_median <= min_median_delta_net_pnl:
            reasons.append("post_selection_median_delta_not_positive")
        if _finite_float(row.get("post_selection_q25_delta_net_pnl"), -np.inf) < min_q25_delta_net_pnl:
            reasons.append("post_selection_q25_delta_below_gate")
        if _finite_float(row.get("post_selection_positive_delta_share"), 0.0) < min_positive_delta_share:
            reasons.append("post_selection_insufficient_positive_folds")
        if _finite_float(row.get("post_selection_median_delta_max_drawdown"), -np.inf) < min_median_delta_max_drawdown:
            reasons.append("post_selection_max_drawdown_worsened")
        if _finite_float(row.get("post_selection_median_delta_worst_24h"), -np.inf) < min_median_delta_worst_24h:
            reasons.append("post_selection_worst_24h_worsened")
        if _finite_float(row.get("post_selection_median_delta_full_sl_rate"), np.inf) > max_median_delta_full_sl_rate:
            reasons.append("post_selection_full_sl_rate_worsened")
        if _finite_float(row.get("post_selection_median_trade_retention_share"), 0.0) < min_median_trade_retention_share:
            reasons.append("post_selection_insufficient_trade_retention")
        if _finite_float(row.get("post_selection_realized_defensive_success"), 0.0) <= min_defensive_success:
            reasons.append("post_selection_defensive_success_not_positive")
        if _finite_float(row.get("post_selection_positive_suppression_fold_share"), 0.0) < min_positive_suppression_share:
            reasons.append("post_selection_suppression_not_recurrent")
    return reasons


def _score_row(row: pd.Series) -> float:
    return (
        _finite_float(row.get("median_delta_net_pnl"), 0.0)
        + 0.25 * _finite_float(row.get("q25_delta_net_pnl"), 0.0)
        + 0.10 * _finite_float(row.get("mean_delta_net_pnl"), 0.0)
        + 10.0 * _finite_float(row.get("realized_defensive_success"), 0.0)
    )


def _add_action_attribution(selection: pd.DataFrame) -> pd.DataFrame:
    out = selection.copy()
    def _numeric_series(column: str, default: float = 0.0) -> pd.Series:
        if column in out.columns:
            values = out[column]
        else:
            values = pd.Series(default, index=out.index)
        return pd.to_numeric(values, errors="coerce").fillna(default)

    direct = _numeric_series("realized_defensive_success")
    freed = _numeric_series("freed_capacity_net_action_pnl_delta")
    entrants = _numeric_series("freed_capacity_entrant_count")
    median_delta = _numeric_series("median_delta_net_pnl")
    positive_freed = np.maximum(freed.to_numpy(dtype=float), 0.0)
    positive_direct = np.maximum(direct.to_numpy(dtype=float), 0.0)
    denom = positive_direct + positive_freed
    with np.errstate(divide="ignore", invalid="ignore"):
        direct_share = np.where(denom > 0.0, positive_direct / denom, np.nan)
    out["direct_suppression_defensive_success"] = direct
    out["replacement_or_backfill_pnl_delta"] = freed
    out["direct_suppression_value_share"] = direct_share
    out["replacement_dependent_lift"] = (
        (median_delta.to_numpy(dtype=float) > 0.0)
        & (entrants.to_numpy(dtype=float) > 0.0)
        & (positive_freed > positive_direct)
    )
    if "post_selection_realized_defensive_success" in out.columns:
        post_direct = pd.to_numeric(
            out["post_selection_realized_defensive_success"],
            errors="coerce",
        ).fillna(0.0)
        out["post_selection_direct_suppression_defensive_success"] = post_direct
    if "post_selection_freed_capacity_net_action_pnl_delta" in out.columns:
        post_freed = pd.to_numeric(
            out["post_selection_freed_capacity_net_action_pnl_delta"],
            errors="coerce",
        ).fillna(0.0)
        out["post_selection_replacement_or_backfill_pnl_delta"] = post_freed
    return out


def _action_attribution_gate(
    row: pd.Series | None,
    policy: dict[str, Any],
    *,
    expected_selected_arm: str | None,
) -> dict[str, Any]:
    if row is None:
        return {
            "evaluated_arm": None,
            "passed": False,
            "failures": ["no_selected_controller_candidate"],
            "interpretation": "No arm passed the conservative selection gates.",
        }

    min_defensive_success = float(_policy_value(policy, "min_defensive_success", 0.0))
    min_positive_suppression_share = float(_policy_value(policy, "min_positive_suppression_share", 0.50))
    direct_success = _finite_float(row.get("realized_defensive_success"), 0.0)
    suppression_share = _finite_float(row.get("positive_suppression_fold_share"), 0.0)
    freed_entrants = _finite_float(row.get("freed_capacity_entrant_count"), 0.0)
    freed_action_pnl = _finite_float(row.get("freed_capacity_net_action_pnl_delta"), 0.0)
    direct_share = _finite_float(row.get("direct_suppression_value_share"), float("nan"))
    replacement_dependent = _normalise_bool(row.get("replacement_dependent_lift", False))

    failures: list[str] = []
    if expected_selected_arm is None:
        failures.append("no_selected_controller_candidate")
    if direct_success <= min_defensive_success:
        failures.append("direct_suppression_defensive_success_not_positive")
    if suppression_share < min_positive_suppression_share:
        failures.append("direct_suppression_not_recurrent")
    if replacement_dependent:
        failures.append("replay_lift_depends_on_replacement_or_backfill")

    return {
        "evaluated_arm": str(row.get("arm")),
        "passed": not failures,
        "failures": failures,
        "direct_suppression_defensive_success": direct_success,
        "positive_suppression_fold_share": suppression_share,
        "freed_capacity_entrant_count": freed_entrants,
        "freed_capacity_net_action_pnl_delta": freed_action_pnl,
        "direct_suppression_value_share": direct_share,
        "replacement_dependent_lift": bool(replacement_dependent),
        "interpretation": (
            "A controller is deployable only if the direct accepted-trade suppression "
            "edge is positive and recurrent; full-replay replacement/backfill gains "
            "are treated as attribution evidence, not promotion evidence."
        ),
    }


def _select_expected_passing_arm(selection: pd.DataFrame, policy: dict[str, Any]) -> str | None:
    passed = selection.loc[selection["recomputed_passed_selection_gates"]].copy()
    if passed.empty:
        return None
    median_delta_tie_abs_tol = float(_policy_value(policy, "median_delta_tie_abs_tol", 1.0))
    median_delta_tie_rel_tol = float(_policy_value(policy, "median_delta_tie_rel_tol", 0.05))
    best_median = float(pd.to_numeric(passed["median_delta_net_pnl"], errors="coerce").max())
    tolerance = max(median_delta_tie_abs_tol, abs(best_median) * median_delta_tie_rel_tol)
    near_best = passed.loc[pd.to_numeric(passed["median_delta_net_pnl"], errors="coerce") >= best_median - tolerance].copy()
    near_best["_complexity"] = near_best["arm"].astype(str).map(_controller_arm_complexity)
    near_best["_selection_score"] = near_best.apply(_score_row, axis=1)
    near_best = near_best.sort_values(
        [
            "_complexity",
            "median_delta_net_pnl",
            "q25_delta_net_pnl",
            "realized_defensive_success",
            "_selection_score",
        ],
        ascending=[True, False, False, False, False],
    )
    return str(near_best.iloc[0]["arm"])


def audit_promotion_gates(
    artifact_dir: Path,
    *,
    allowed_base_arms: set[str] | None = None,
    audit_scope_name: str = "all_controller_arms",
) -> tuple[dict[str, Any], pd.DataFrame, list[str]]:
    failures: list[str] = []
    selection_path = artifact_dir / "walkforward_controller_candidate_selection.csv"
    selected_path = artifact_dir / "walkforward_selected_controller_candidate.json"
    config_path = artifact_dir / "strategy_threshold_controller_config.json"
    required_paths = [selection_path, selected_path, config_path]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        return {"artifact_dir": str(artifact_dir), "passed": False}, pd.DataFrame(), [f"missing promotion artifacts: {missing}"]

    selection = _enrich_selection_with_safety_metrics(pd.read_csv(selection_path), artifact_dir)
    selection = _enrich_selection_with_action_metrics(selection, artifact_dir)
    unfiltered_candidate_count = int(len(selection))
    selection = _filter_selection_scope(selection, allowed_base_arms=allowed_base_arms)
    if selection.empty:
        failures.append("candidate selection is empty after audit-scope filtering")
    selected = _load_json(selected_path)
    config = _load_json(config_path)
    policy = selected.get("selection_policy")
    if not isinstance(policy, dict):
        policy = config.get("selection", {}).get("selection_policy") if isinstance(config.get("selection"), dict) else {}
    if not isinstance(policy, dict) or not policy:
        failures.append("selection policy is missing")
        policy = {}

    required_columns = [
        "arm",
        "median_delta_net_pnl",
        "mean_delta_net_pnl",
        "q25_delta_net_pnl",
        "positive_delta_share",
        "median_delta_max_drawdown",
        "median_delta_worst_24h",
        "median_trade_retention_share",
        "median_delta_full_sl_rate",
        "realized_defensive_success",
        "positive_suppression_fold_share",
        "post_selection_median_delta_net_pnl",
        "post_selection_q25_delta_net_pnl",
        "post_selection_positive_delta_share",
        "post_selection_median_delta_max_drawdown",
        "post_selection_median_delta_worst_24h",
        "post_selection_median_trade_retention_share",
        "post_selection_median_delta_full_sl_rate",
        "post_selection_realized_defensive_success",
        "post_selection_positive_suppression_fold_share",
        "mean_state_ood_share",
        "passed_selection_gates",
        "selection_fail_reasons",
        "selection_score",
    ]
    missing_columns = [column for column in required_columns if column not in selection.columns]
    if missing_columns:
        failures.append(f"candidate selection missing columns: {missing_columns}")
        return {"artifact_dir": str(artifact_dir), "passed": False}, selection, failures

    selection = selection.copy()
    recomputed_reasons: list[list[str]] = []
    mismatched_reason_rows: list[str] = []
    mismatched_pass_rows: list[str] = []
    mismatched_score_rows: list[str] = []
    for index, row in selection.iterrows():
        reasons = _compute_fail_reasons(row, policy)
        recomputed_reasons.append(reasons)
        stored_reasons = _as_reason_set(row.get("selection_fail_reasons"))
        if stored_reasons != set(reasons):
            mismatched_reason_rows.append(str(row.get("arm", index)))
        stored_passed = _normalise_bool(row.get("passed_selection_gates"))
        recomputed_passed = not reasons
        if stored_passed != recomputed_passed:
            mismatched_pass_rows.append(str(row.get("arm", index)))
        stored_score = _finite_float(row.get("selection_score"), float("nan"))
        expected_score = _score_row(row)
        if np.isfinite(stored_score) and abs(stored_score - expected_score) > 1e-6:
            mismatched_score_rows.append(str(row.get("arm", index)))

    selection["recomputed_selection_fail_reasons"] = [";".join(reasons) for reasons in recomputed_reasons]
    selection["recomputed_passed_selection_gates"] = [not reasons for reasons in recomputed_reasons]
    selection["recomputed_selection_score"] = selection.apply(_score_row, axis=1)
    selection = _add_action_attribution(selection)
    if mismatched_reason_rows:
        failures.append(f"selection fail reasons mismatch for arms: {mismatched_reason_rows[:10]}")
    if mismatched_pass_rows:
        failures.append(f"selection pass flag mismatch for arms: {mismatched_pass_rows[:10]}")
    if mismatched_score_rows:
        failures.append(f"selection score mismatch for arms: {mismatched_score_rows[:10]}")

    expected_selected = _select_expected_passing_arm(selection, policy)
    selected_arm = selected.get("selected_arm")
    selected_arm_in_scope = (
        selected_arm is None
        or selected_arm == BASELINE_ARM
        or _base_arm(selected_arm) in set(allowed_base_arms or [])
        or not allowed_base_arms
    )
    if selected_arm_in_scope and selected_arm != expected_selected:
        failures.append(f"selected_arm {selected_arm!r} != expected {expected_selected!r}")
    if (
        selected_arm_in_scope
        and expected_selected is None
        and selected.get("reason") != "no_arm_passed_selection_gates"
    ):
        failures.append("selected_controller reason is not no_arm_passed_selection_gates")

    deployable = selection.loc[
        ~selection["arm"].astype(str).eq(BASELINE_ARM)
        & ~selection["arm"].astype(str).str.endswith("__post_selection_overlay")
    ].copy()
    pass_count = int(selection["recomputed_passed_selection_gates"].sum())
    reason_counter: Counter[str] = Counter()
    for reasons in recomputed_reasons:
        reason_counter.update(reasons)
    best_raw = (
        deployable.sort_values("recomputed_selection_score", ascending=False).iloc[0].to_dict()
        if not deployable.empty
        else {}
    )
    attribution_source: pd.Series | None = None
    if expected_selected is not None:
        expected_rows = selection.loc[selection["arm"].astype(str).eq(str(expected_selected))]
        if not expected_rows.empty:
            attribution_source = expected_rows.iloc[0]
    elif best_raw:
        attribution_source = pd.Series(best_raw)
    action_attribution_gate = _action_attribution_gate(
        attribution_source,
        policy,
        expected_selected_arm=expected_selected,
    )
    controller_promotion_ready = bool(
        expected_selected is not None and action_attribution_gate.get("passed")
    )
    payload = {
        "generated_by": "audit_market_state_controller_promotion_gates",
        "artifact_dir": str(artifact_dir),
        "audit_scope": {
            "name": str(audit_scope_name),
            "allowed_base_arms": sorted(str(arm) for arm in (allowed_base_arms or set())),
            "unfiltered_candidate_count": unfiltered_candidate_count,
            "filtered_candidate_count": int(len(selection)),
            "stored_selected_arm_in_scope": bool(selected_arm_in_scope),
        },
        "candidate_count": int(len(selection)),
        "deployable_candidate_count": int(len(deployable)),
        "passing_candidate_count": pass_count,
        "selected_arm": selected_arm,
        "expected_selected_arm": expected_selected,
        "promotion_gate_passed": expected_selected is not None,
        "action_attribution_gate": action_attribution_gate,
        "controller_promotion_ready": controller_promotion_ready,
        "controller_should_remain_disabled": not controller_promotion_ready,
        "selection_policy": policy,
        "failure_reason_counts": dict(sorted(reason_counter.items())),
        "best_raw_candidate": {
            "arm": best_raw.get("arm"),
            "controller_action_scope": best_raw.get("controller_action_scope")
            or best_raw.get("controller_action")
            or best_raw.get("action_scope"),
            "controller_action_phase": best_raw.get("controller_action_phase")
            or best_raw.get("action_phase"),
            "changes_scores_or_ranks": bool(
                _normalise_bool(best_raw.get("changes_scores_or_ranks", False))
                or _normalise_bool(best_raw.get("changes_scores", False))
                or _normalise_bool(best_raw.get("changes_ranks", False))
            ),
            "changes_auction_ordering": bool(
                _normalise_bool(best_raw.get("changes_auction_ordering", False))
                or _normalise_bool(best_raw.get("changes_auction_priority", False))
            ),
            "changes_position_sizing": bool(
                _normalise_bool(best_raw.get("changes_position_sizing", False))
            ),
            "allows_threshold_reductions": bool(
                _normalise_bool(best_raw.get("allows_threshold_reductions", False))
                or _normalise_bool(best_raw.get("can_lower_thresholds", False))
                or _normalise_bool(best_raw.get("can_lower_threshold", False))
            ),
            "promotes_replacement_candidates": bool(
                _normalise_bool(best_raw.get("promotes_replacement_candidates", False))
                or _normalise_bool(best_raw.get("directly_promotes_replacements", False))
            ),
            "median_delta_net_pnl": _finite_float(best_raw.get("median_delta_net_pnl")),
            "q25_delta_net_pnl": _finite_float(best_raw.get("q25_delta_net_pnl")),
            "positive_delta_share": _finite_float(best_raw.get("positive_delta_share")),
            "median_delta_max_drawdown": _finite_float(best_raw.get("median_delta_max_drawdown")),
            "median_delta_worst_24h": _finite_float(best_raw.get("median_delta_worst_24h")),
            "median_trade_retention_share": _finite_float(best_raw.get("median_trade_retention_share")),
            "median_delta_full_sl_rate": _finite_float(best_raw.get("median_delta_full_sl_rate")),
            "realized_defensive_success": _finite_float(best_raw.get("realized_defensive_success")),
            "positive_suppression_fold_share": _finite_float(best_raw.get("positive_suppression_fold_share")),
            "post_selection_realized_defensive_success": _finite_float(
                best_raw.get("post_selection_realized_defensive_success")
            ),
            "post_selection_positive_suppression_fold_share": _finite_float(
                best_raw.get("post_selection_positive_suppression_fold_share")
            ),
            "freed_capacity_entrant_count": _finite_float(best_raw.get("freed_capacity_entrant_count")),
            "freed_capacity_net_replacement_pnl": _finite_float(
                best_raw.get("freed_capacity_net_replacement_pnl")
            ),
            "freed_capacity_net_action_pnl_delta": _finite_float(
                best_raw.get("freed_capacity_net_action_pnl_delta")
            ),
            "direct_suppression_value_share": _finite_float(
                best_raw.get("direct_suppression_value_share")
            ),
            "replacement_dependent_lift": bool(
                _normalise_bool(best_raw.get("replacement_dependent_lift", False))
            ),
            "positive_freed_capacity_fold_share": _finite_float(
                best_raw.get("positive_freed_capacity_fold_share")
            ),
            "post_selection_freed_capacity_net_replacement_pnl": _finite_float(
                best_raw.get("post_selection_freed_capacity_net_replacement_pnl")
            ),
            "post_selection_freed_capacity_net_action_pnl_delta": _finite_float(
                best_raw.get("post_selection_freed_capacity_net_action_pnl_delta")
            ),
            "mean_state_ood_share": _finite_float(best_raw.get("mean_state_ood_share")),
            "selection_score": _finite_float(best_raw.get("selection_score")),
            "recomputed_fail_reasons": best_raw.get("recomputed_selection_fail_reasons", ""),
        },
        "failures": failures,
        "passed": not failures,
    }
    return payload, selection, failures


def _write_report(payload: dict[str, Any], selection: pd.DataFrame, output_dir: Path) -> str:
    best = payload.get("best_raw_candidate", {})
    scope = payload.get("audit_scope", {})
    attribution = payload.get("action_attribution_gate", {})
    lines = [
        "# Market-State Controller Promotion-Gate Audit",
        "",
        "This audit recomputes the controller promotion gates from persisted walk-forward selection metrics. It does not rerun replay and does not promote a controller.",
        "",
        "## Verdict",
        "",
        f"- Passed: `{payload['passed']}`",
        f"- Promotion gate passed: `{payload['promotion_gate_passed']}`",
        f"- Action-attribution gate passed: `{attribution.get('passed')}`",
        f"- Controller promotion ready: `{payload.get('controller_promotion_ready')}`",
        f"- Controller should remain disabled: `{payload['controller_should_remain_disabled']}`",
        f"- Selected arm: `{payload['selected_arm']}`",
        f"- Expected selected arm: `{payload['expected_selected_arm']}`",
        f"- Failures: `{len(payload['failures'])}`",
        f"- Audit scope: `{scope.get('name', 'all_controller_arms')}`",
        f"- Allowed base arms: `{', '.join(scope.get('allowed_base_arms') or []) or 'all'}`",
        "",
        "## Best Raw Candidate",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| arm | {best.get('arm')} |",
        f"| controller_action_scope | {best.get('controller_action_scope')} |",
        f"| controller_action_phase | {best.get('controller_action_phase')} |",
        f"| changes_scores_or_ranks | {best.get('changes_scores_or_ranks')} |",
        f"| changes_auction_ordering | {best.get('changes_auction_ordering')} |",
        f"| changes_position_sizing | {best.get('changes_position_sizing')} |",
        f"| allows_threshold_reductions | {best.get('allows_threshold_reductions')} |",
        f"| promotes_replacement_candidates | {best.get('promotes_replacement_candidates')} |",
        f"| median_delta_net_pnl | {best.get('median_delta_net_pnl')} |",
        f"| q25_delta_net_pnl | {best.get('q25_delta_net_pnl')} |",
        f"| positive_delta_share | {best.get('positive_delta_share')} |",
        f"| median_delta_max_drawdown | {best.get('median_delta_max_drawdown')} |",
        f"| median_delta_worst_24h | {best.get('median_delta_worst_24h')} |",
        f"| median_trade_retention_share | {best.get('median_trade_retention_share')} |",
        f"| median_delta_full_sl_rate | {best.get('median_delta_full_sl_rate')} |",
        f"| realized_defensive_success | {best.get('realized_defensive_success')} |",
        f"| post_selection_realized_defensive_success | {best.get('post_selection_realized_defensive_success')} |",
        f"| post_selection_positive_suppression_fold_share | {best.get('post_selection_positive_suppression_fold_share')} |",
        f"| freed_capacity_entrant_count | {best.get('freed_capacity_entrant_count')} |",
        f"| freed_capacity_net_replacement_pnl | {best.get('freed_capacity_net_replacement_pnl')} |",
        f"| freed_capacity_net_action_pnl_delta | {best.get('freed_capacity_net_action_pnl_delta')} |",
        f"| direct_suppression_value_share | {best.get('direct_suppression_value_share')} |",
        f"| replacement_dependent_lift | {best.get('replacement_dependent_lift')} |",
        f"| positive_freed_capacity_fold_share | {best.get('positive_freed_capacity_fold_share')} |",
        f"| post_selection_freed_capacity_net_replacement_pnl | {best.get('post_selection_freed_capacity_net_replacement_pnl')} |",
        f"| post_selection_freed_capacity_net_action_pnl_delta | {best.get('post_selection_freed_capacity_net_action_pnl_delta')} |",
        f"| mean_state_ood_share | {best.get('mean_state_ood_share')} |",
        f"| selection_score | {best.get('selection_score')} |",
        "",
        "## Action Attribution Gate",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| evaluated_arm | {attribution.get('evaluated_arm')} |",
        f"| passed | {attribution.get('passed')} |",
        f"| failures | {';'.join(attribution.get('failures') or [])} |",
        f"| direct_suppression_defensive_success | {attribution.get('direct_suppression_defensive_success')} |",
        f"| positive_suppression_fold_share | {attribution.get('positive_suppression_fold_share')} |",
        f"| freed_capacity_entrant_count | {attribution.get('freed_capacity_entrant_count')} |",
        f"| freed_capacity_net_action_pnl_delta | {attribution.get('freed_capacity_net_action_pnl_delta')} |",
        f"| direct_suppression_value_share | {attribution.get('direct_suppression_value_share')} |",
        f"| replacement_dependent_lift | {attribution.get('replacement_dependent_lift')} |",
        "",
        str(attribution.get("interpretation", "")),
        "",
        "## Candidate Gate Rows",
        "",
    ]
    report_cols = [
        "arm",
        "controller_action_scope",
        "controller_action_phase",
        "changes_scores_or_ranks",
        "changes_auction_ordering",
        "changes_position_sizing",
        "allows_threshold_reductions",
        "promotes_replacement_candidates",
        "median_delta_net_pnl",
        "q25_delta_net_pnl",
        "positive_delta_share",
        "median_delta_max_drawdown",
        "median_delta_worst_24h",
        "median_trade_retention_share",
        "median_delta_full_sl_rate",
        "realized_defensive_success",
        "post_selection_realized_defensive_success",
        "post_selection_positive_suppression_fold_share",
        "freed_capacity_entrant_count",
        "freed_capacity_net_replacement_pnl",
        "freed_capacity_net_action_pnl_delta",
        "direct_suppression_value_share",
        "replacement_dependent_lift",
        "positive_freed_capacity_fold_share",
        "post_selection_freed_capacity_net_replacement_pnl",
        "post_selection_freed_capacity_net_action_pnl_delta",
        "mean_state_ood_share",
        "recomputed_passed_selection_gates",
        "recomputed_selection_fail_reasons",
    ]
    available = [column for column in report_cols if column in selection.columns]
    lines.append(_markdown_table(selection.loc[:, available]) if available else "_No candidate rows._")
    lines.extend(
        [
            "",
            "Generated files:",
            f"- `{output_dir / 'market_state_controller_promotion_gate_audit.json'}`",
            f"- `{output_dir / 'market_state_controller_promotion_gate_selection.csv'}`",
            f"- `{output_dir / 'market_state_controller_promotion_gate_audit.md'}`",
        ]
    )
    if payload["failures"]:
        lines.extend(["", "## Failures", ""])
        for failure in payload["failures"]:
            lines.append(f"- {failure}")
    return "\n".join(lines) + "\n"


def write_promotion_gate_audit(payload: dict[str, Any], selection: pd.DataFrame, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": output_dir / "market_state_controller_promotion_gate_audit.json",
        "selection": output_dir / "market_state_controller_promotion_gate_selection.csv",
        "report": output_dir / "market_state_controller_promotion_gate_audit.md",
    }
    paths["json"].write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    selection.to_csv(paths["selection"], index=False)
    paths["report"].write_text(_write_report(payload, selection, output_dir), encoding="utf-8")
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--stage1-observed-only",
        action="store_true",
        help=(
            "Audit only the first permitted production challenger family: "
            "observed descriptive state axes with penalty-only threshold raises."
        ),
    )
    parser.add_argument(
        "--allowed-arm",
        action="append",
        default=[],
        help="Additional base controller arm allowed in the audit scope. Can be repeated.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    allowed = set(str(arm) for arm in args.allowed_arm)
    scope_name = "custom_allowed_arms" if allowed else "all_controller_arms"
    if bool(args.stage1_observed_only):
        allowed.update(STAGE1_OBSERVED_ONLY_ARMS)
        scope_name = "stage1_observed_only"
    payload, selection, failures = audit_promotion_gates(
        args.artifact_dir,
        allowed_base_arms=allowed or None,
        audit_scope_name=scope_name,
    )
    output_dir = args.output_dir or args.artifact_dir
    paths = write_promotion_gate_audit(payload, selection, output_dir)
    print(f"Wrote market-state controller promotion-gate audit: {paths['report']}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
