#!/usr/bin/env python3
"""Learn market-state head-priority modulation and replay it.

The model target is head/timestamp frontier utility or threshold-admission
utility, centered cross-sectionally within each timestamp.  The action is still
deliberately narrow: predicted state response becomes bounded rank-prior and/or
auction-priority action columns, so labels, thresholds, position sizing,
q-fail, HeadHealth, and the base T1 contract remain fixed.
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
import shutil
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.portfolio_policy_replay import (  # noqa: E402
    fit_hierarchical_ev_curves,
    replay_candidates,
)
from scripts import run_market_state_threshold_controller as mstc  # noqa: E402
from scripts.run_market_state_head_priority_modulation import (  # noqa: E402
    BASELINE_ARM,
    PRIORITY_ACTIONS,
    apply_head_priority_schedule,
    priority_action_values,
    rank_prior_values,
)


DEFAULT_WALKFORWARD_DIR = Path(
    "data_perp/reports/market_state_threshold_controller_walkforward_20260626_t1_lgbm_maturity_contract_v1"
)
DEFAULT_SCORE_DIR = Path(
    "data_perp/reports/market_state_controller_bundle_score_t1_lgbm_maturity_noop_20260626"
)
DEFAULT_TRAIN_DEPLOYABLE = Path(
    "data_perp/artifacts/reliability_blend_native_simple_policy_replay_20260624_floor070/"
    "simple_policy_optimiser/simple_policy_candidates.parquet"
)
DEFAULT_POLICY_MANIFEST = Path(
    "data_perp/reports/reliability_blend_component_arm_portfolio_ablation_20260625/"
    "A0_anchor_only/portfolio_policy_ablation_manifest.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "data_perp/reports/market_state_head_priority_learning_20260626"
)

LEARNED_ARMS = {
    "lgbm": "L1_lgbm_learned_priority",
    "xgb": "L2_xgb_learned_priority",
}
SELECTED_ARM_PREFIX = "L0_selected"
PORTFOLIO_ROUTING_REJECTION_REASONS = {
    "max_concurrent_positions_reached",
    "max_concurrent_per_side_reached",
    "max_concurrent_per_strategy_reached",
    "max_new_entries_per_bar_reached",
    "max_new_entries_per_strategy_per_bar_reached",
    "max_capital_allocation_reached",
}
TARGET_MODES = {
    "frontier_weighted_mean",
    "head_top_candidate",
    "rank_residual_frontier",
    "threshold_admission_mean",
}


def _validate_target_mode(target_mode: str) -> str:
    mode = str(target_mode)
    if mode not in TARGET_MODES:
        raise ValueError(f"unknown head-priority target mode: {target_mode}")
    return mode


def _frontier_weights(
    rank: pd.Series,
    threshold: pd.Series,
    *,
    target_mode: str,
    frontier_gamma: float,
    frontier_bandwidth: float,
) -> pd.Series:
    """Return candidate weights aligned to the configured priority target.

    The legacy frontier target keeps a unit baseline weight for all candidates
    above min_rank.  That is suitable for auction priority.  Rank-prior actions
    cross the deployment threshold, so the admission target removes the unit
    baseline and concentrates weight around the threshold frontier.
    """

    mode = _validate_target_mode(target_mode)
    bandwidth = max(float(frontier_bandwidth), 1e-6)
    distance = np.abs(pd.to_numeric(rank, errors="coerce") - pd.to_numeric(threshold, errors="coerce"))
    local = np.exp(-distance / bandwidth)
    if mode == "threshold_admission_mean":
        return pd.Series(local, index=rank.index).clip(lower=1e-9)
    return pd.Series(1.0 + float(frontier_gamma) * local, index=rank.index)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _sha256(path: Path | None) -> str | None:
    if path is None or not path.exists() or path.is_dir():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _prepare_staged_output_dir(final_dir: Path) -> tuple[Path, Path, dict[str, Any]]:
    """Return final and staging output dirs for all-or-nothing reports.

    The priority learner writes many intermediate parquet files before the final
    manifest exists.  Staging keeps interrupted runs from leaving report folders
    that look usable but contain only a partial static baseline replay.
    """

    final_dir = Path(final_dir)
    parent = final_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    if final_dir.exists():
        if any(final_dir.iterdir()):
            raise FileExistsError(
                f"output directory already exists and is not empty: {final_dir}"
            )
        final_dir.rmdir()
    staging_dir = parent / f".{final_dir.name}.staging-{os.getpid()}"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)
    return final_dir, staging_dir, {
        "enabled": True,
        "final_output_dir": str(final_dir),
        "staging_output_dir": str(staging_dir),
        "publish_mode": "atomic_replace_after_manifest",
    }


def _publish_staged_output_dir(staging_dir: Path, final_dir: Path) -> None:
    if final_dir.exists():
        if any(final_dir.iterdir()):
            raise FileExistsError(
                f"output directory already exists and is not empty: {final_dir}"
            )
        final_dir.rmdir()
    os.replace(staging_dir, final_dir)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _load_static_baseline_artifacts(
    manifest_path: Path | None,
    *,
    arm: str = BASELINE_ARM,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]] | None:
    """Load the frozen materialized static baseline for exact T1 parity.

    The priority experiments compare market-state action arms against the
    physically frozen T1 ledger.  Replaying the saved candidate parquet with a
    newer replay implementation can differ at tied frontier decisions, so the
    static arm must come from the manifest when a manifest is available.
    """
    if manifest_path is None:
        return None
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = _load_json(manifest_path)
    outputs = dict(manifest.get("outputs") or {})

    def _path(key: str) -> Path | None:
        raw = outputs.get(key)
        if raw in (None, ""):
            return None
        return Path(str(raw))

    summary_path = _path("summary")
    accepted_path = _path("accepted_trades")
    decisions_path = _path("decisions")
    equity_path = _path("equity_curve")
    by_head_path = _path("by_head")
    if summary_path is None or not summary_path.exists():
        summary_payload = dict(manifest.get("summary") or {})
        if not summary_payload:
            raise FileNotFoundError(f"missing static baseline summary for {manifest_path}")
        summary = pd.DataFrame([summary_payload])
    else:
        summary = pd.read_csv(summary_path)
    if summary.empty:
        raise ValueError(f"empty static baseline summary: {summary_path or manifest_path}")
    summary = summary.copy()
    summary["arm"] = str(arm)

    accepted = (
        pd.read_parquet(accepted_path)
        if accepted_path is not None and accepted_path.exists()
        else pd.DataFrame()
    )
    if not accepted.empty:
        accepted = accepted.copy()
        if "timestamp" in accepted.columns:
            accepted["timestamp"] = pd.to_datetime(accepted["timestamp"], utc=True, errors="coerce")
        accepted["arm"] = str(arm)

    by_head = (
        pd.read_csv(by_head_path)
        if by_head_path is not None and by_head_path.exists()
        else pd.DataFrame()
    )
    if not by_head.empty:
        by_head = by_head.copy()
        by_head["arm"] = str(arm)

    decisions = (
        pd.read_parquet(decisions_path)
        if decisions_path is not None and decisions_path.exists()
        else pd.DataFrame()
    )
    equity = (
        pd.read_parquet(equity_path)
        if equity_path is not None and equity_path.exists()
        else pd.DataFrame()
    )
    candidates_broad_path = _path("candidates_broad")
    candidates_deployable_path = _path("candidates_deployable")
    info = {
        "manifest_path": str(manifest_path),
        "summary_path": str(summary_path) if summary_path is not None else None,
        "accepted_trades_path": str(accepted_path) if accepted_path is not None else None,
        "decisions_path": str(decisions_path) if decisions_path is not None else None,
        "equity_curve_path": str(equity_path) if equity_path is not None else None,
        "by_head_path": str(by_head_path) if by_head_path is not None else None,
        "candidates_broad_path": (
            str(candidates_broad_path) if candidates_broad_path is not None else None
        ),
        "candidates_deployable_path": (
            str(candidates_deployable_path) if candidates_deployable_path is not None else None
        ),
        "input_eval_candidates_path": str(dict(manifest.get("inputs") or {}).get("eval_candidates") or "")
        or None,
        "generated_by": manifest.get("generated_by"),
        "rank_contract": dict(manifest.get("active_stack") or {}).get("rank_contract"),
        "rank_scope": dict(manifest.get("active_stack") or {}).get("rank_scope"),
        "manifest_candidate_rows": int(manifest.get("candidate_rows") or 0),
        "manifest_deployable_rows": int(manifest.get("deployable_rows") or 0),
        "manifest_accepted_rows": int(manifest.get("accepted_rows") or 0),
        "manifest_timestamp_min": manifest.get("timestamp_min"),
        "manifest_timestamp_max": manifest.get("timestamp_max"),
        "summary_trade_count": int(float(summary.iloc[0].get("trade_count", 0) or 0)),
        "summary_net_pnl": float(summary.iloc[0].get("net_pnl", 0.0) or 0.0),
        "accepted_rows": int(len(accepted)),
    }
    return decisions, equity, accepted, summary, by_head, info


def _load_candidates(path: Path) -> pd.DataFrame:
    out = pd.read_parquet(path)
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    if "head" not in out.columns:
        out["head"] = out["strategy_id"].astype(str).map(mstc._infer_head)
    return mstc.normalise_candidate_table(out)


def static_baseline_candidate_parity(
    candidates: pd.DataFrame,
    *,
    candidates_path: Path | None = None,
    static_baseline_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify whether a priority run uses the frozen deployable T1 universe.

    T1 manifests may keep both a broad diagnostic candidate table and a smaller
    deployable table used for the accepted-trade replay.  A learned-priority arm
    can look materially different if it is replayed on the broad table while the
    static arm is loaded from the deployable frozen ledger, so reports must make
    that scope explicit.
    """

    if not static_baseline_info:
        return {
            "checked": False,
            "promotion_grade_scope": False,
            "reason": "static_baseline_manifest_not_loaded",
        }
    rows = int(len(candidates))
    deployable_rows = int(static_baseline_info.get("manifest_deployable_rows") or 0)
    broad_rows = int(static_baseline_info.get("manifest_candidate_rows") or 0)
    deployable_rows_match = bool(deployable_rows and rows == deployable_rows)
    broad_rows_match = bool(broad_rows and rows == broad_rows)

    current_path = str(candidates_path) if candidates_path is not None else None
    deployable_path = static_baseline_info.get("candidates_deployable_path")
    broad_path = static_baseline_info.get("candidates_broad_path")
    current_resolved = str(Path(current_path).resolve()) if current_path else None

    def _path_matches(raw: Any) -> bool:
        if raw in (None, "") or current_resolved is None:
            return False
        try:
            return str(Path(str(raw)).resolve()) == current_resolved
        except OSError:
            return str(raw) == str(current_path)

    deployable_path_match = _path_matches(deployable_path)
    broad_path_match = _path_matches(broad_path)
    if deployable_rows_match or deployable_path_match:
        scope = "deployable_static_baseline"
    elif broad_rows_match or broad_path_match:
        scope = "broad_non_deployable_diagnostic"
    else:
        scope = "unknown_or_mismatched"

    ts = (
        pd.to_datetime(candidates["timestamp"], utc=True, errors="coerce")
        if "timestamp" in candidates.columns
        else pd.Series(dtype="datetime64[ns, UTC]")
    )
    failures: list[str] = []
    if scope != "deployable_static_baseline":
        failures.append("candidate_universe_not_deployable_static_baseline_scope")
    if rows <= 0:
        failures.append("empty_candidate_universe")
    return {
        "checked": True,
        "candidate_scope": scope,
        "promotion_grade_scope": scope == "deployable_static_baseline" and not failures,
        "current_rows": rows,
        "expected_deployable_rows": deployable_rows,
        "expected_broad_rows": broad_rows,
        "deployable_rows_match": deployable_rows_match,
        "broad_rows_match": broad_rows_match,
        "current_candidates_path": current_path,
        "expected_deployable_path": deployable_path,
        "expected_broad_path": broad_path,
        "deployable_path_match": deployable_path_match,
        "broad_path_match": broad_path_match,
        "timestamp_count": int(ts.nunique()) if len(ts) else 0,
        "timestamp_min": ts.min() if len(ts) else None,
        "timestamp_max": ts.max() if len(ts) else None,
        "heads": sorted(candidates.get("head", pd.Series(dtype=object)).dropna().astype(str).unique()),
        "failures": failures,
    }


def load_train_deployable_for_static_contract(
    path: Path,
    *,
    static_baseline_manifest: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load train-deployable rows under the same rank contract as static T1.

    The portfolio replay fits hierarchical EV curves from the historical
    deployable rows.  For global-rank T1, those training rows must receive the
    same frozen rank-reference transform as the evaluation rows; otherwise a
    no-op replay can differ from the materialized static baseline even when all
    priority actions are neutral.
    """
    default = _load_candidates(path)
    if static_baseline_manifest is None or not static_baseline_manifest.exists():
        return default, {
            "applied_static_rank_contract": False,
            "reason": "static_baseline_manifest_missing",
            "rows": int(len(default)),
        }
    manifest = _load_json(static_baseline_manifest)
    active_stack = dict(manifest.get("active_stack") or {})
    rank_contract = str(active_stack.get("rank_contract") or "")
    if rank_contract != "anchor_global_policy_rank_reference":
        return default, {
            "applied_static_rank_contract": False,
            "reason": "static_baseline_rank_contract_not_global_reference",
            "rank_contract": rank_contract or None,
            "rows": int(len(default)),
        }
    rank_reference_run_id = str(active_stack.get("rank_reference_run_id") or "")
    if not rank_reference_run_id:
        raise ValueError(
            f"{static_baseline_manifest} declares anchor_global_policy_rank_reference "
            "but has no active_stack.rank_reference_run_id"
        )
    disabled_heads = {str(x) for x in active_stack.get("disabled_heads") or []}
    from scripts.materialize_t1_repaired_static_baseline import _load_for_t1

    transformed, diag = _load_for_t1(
        path,
        rank_contract=rank_contract,
        disabled_heads=disabled_heads,
        data_root=Path("data_perp"),
        rank_reference_run_id=rank_reference_run_id,
    )
    return transformed, {
        "applied_static_rank_contract": True,
        "rank_contract": rank_contract,
        "rank_reference_run_id": rank_reference_run_id,
        "disabled_heads": sorted(disabled_heads),
        "rows_before": int(len(default)),
        "rows_after": int(len(transformed)),
        "rank_diagnostics": diag,
    }


def state_feature_columns(frame: pd.DataFrame) -> list[str]:
    blocked = {
        "timestamp",
        "fold",
        "split",
        "state_arm",
        "state_level",
        "prediction_contract",
        "state_feature_count",
    }
    cols = []
    for col in frame.columns:
        if col in blocked:
            continue
        if not (col.startswith("state_") or col.startswith("forecast_")):
            continue
        series = pd.to_numeric(frame[col], errors="coerce")
        if int(series.notna().sum()) > 0:
            cols.append(col)
    return cols


def _load_active_state_heads(
    activation_registry_path: Path | None,
    *,
    allowed_statuses: set[str] | None = None,
) -> tuple[set[str] | None, dict[str, Any]]:
    """Load state heads allowed to enter the priority design matrix.

    The market-state activation registry is produced by the threshold-controller
    walk-forward.  It encodes which state heads survived forecast-skill,
    response, action, leave-one-out and defensive-success gates.  Priority
    modulation is shadow-only, but it should not silently consume disabled
    state heads unless the caller explicitly opts out.
    """

    if activation_registry_path is None:
        return None, {
            "enabled": False,
            "reason": "activation_registry_not_configured",
            "allowed_statuses": sorted(allowed_statuses or []),
        }
    path = Path(activation_registry_path)
    statuses = set(allowed_statuses or {"active_candidate"})
    if not path.exists():
        return None, {
            "enabled": False,
            "reason": "activation_registry_missing",
            "path": str(path),
            "allowed_statuses": sorted(statuses),
        }
    registry = pd.read_csv(path)
    required = {"state_head", "recommended_status"}
    missing = sorted(required.difference(registry.columns))
    if missing:
        raise ValueError(f"activation registry missing columns: {missing}")
    status = registry["recommended_status"].astype(str)
    heads = set(
        registry.loc[status.isin(statuses), "state_head"]
        .dropna()
        .astype(str)
        .tolist()
    )
    return heads, {
        "enabled": True,
        "path": str(path),
        "allowed_statuses": sorted(statuses),
        "registry_rows": int(len(registry)),
        "allowed_state_head_count": int(len(heads)),
        "allowed_state_heads": sorted(heads),
        "status_counts": registry["recommended_status"].astype(str).value_counts().to_dict(),
    }


def _filter_state_feature_columns(
    feature_cols: list[str],
    allowed_state_heads: set[str] | None,
) -> list[str]:
    if allowed_state_heads is None:
        return list(feature_cols)
    allowed = set(map(str, allowed_state_heads))
    return [col for col in feature_cols if str(col) in allowed]


def _weighted_mean(value: pd.Series, weight: pd.Series) -> float:
    x = pd.to_numeric(value, errors="coerce")
    w = pd.to_numeric(weight, errors="coerce")
    valid = x.notna() & w.notna() & (w > 0.0)
    if not bool(valid.any()):
        return float("nan")
    return float(np.average(x.loc[valid].to_numpy(dtype=float), weights=w.loc[valid].to_numpy(dtype=float)))


def build_head_priority_targets(
    residual_ledger: pd.DataFrame,
    state_panel: pd.DataFrame,
    *,
    state_arm: str = "S1_observed_axes_shared_response",
    allowed_state_heads: set[str] | None = None,
    target_mode: str = "frontier_weighted_mean",
    min_rank: float = 0.50,
    frontier_gamma: float = 3.0,
    frontier_bandwidth: float = 0.06,
    sl_penalty: float = 0.010,
    timeout_penalty: float = 0.002,
    min_candidates_per_head_timestamp: int = 3,
    target_clip: float = 0.08,
    rank_residual_weight: float = 1.0,
) -> tuple[pd.DataFrame, list[str]]:
    """Create one train row per fold/timestamp/head from candidate residuals."""
    target_mode = _validate_target_mode(target_mode)
    if residual_ledger.empty:
        return pd.DataFrame(), []
    ledger = residual_ledger.copy()
    ledger["timestamp"] = pd.to_datetime(ledger["timestamp"], utc=True, errors="coerce")
    ledger = ledger.loc[ledger.get("arm", state_arm).astype(str).eq(str(state_arm))].copy()
    rank = pd.to_numeric(ledger.get("_rank"), errors="coerce")
    ledger = ledger.loc[(rank >= float(min_rank)).fillna(False)].copy()
    if ledger.empty:
        return pd.DataFrame(), []
    rank = pd.to_numeric(ledger["_rank"], errors="coerce")
    threshold = pd.to_numeric(ledger.get("_threshold"), errors="coerce").fillna(0.70)
    ledger["_frontier_weight"] = _frontier_weights(
        rank,
        threshold,
        target_mode=target_mode,
        frontier_gamma=float(frontier_gamma),
        frontier_bandwidth=float(frontier_bandwidth),
    )
    ledger["_target_component"] = (
        pd.to_numeric(ledger.get("resid_utility"), errors="coerce")
        - float(sl_penalty) * pd.to_numeric(ledger.get("resid_full_sl"), errors="coerce")
        - float(timeout_penalty) * pd.to_numeric(ledger.get("resid_timeout"), errors="coerce")
    )
    group_cols = ["fold", "arm", "timestamp", "head"]
    rows: list[dict[str, Any]] = []
    for key, group in ledger.groupby(group_cols, sort=True, observed=True):
        if len(group) < int(min_candidates_per_head_timestamp):
            continue
        if target_mode == "head_top_candidate":
            rank_values = pd.to_numeric(group["_rank"], errors="coerce")
            finite_rank = rank_values.replace([np.inf, -np.inf], np.nan).dropna()
            if finite_rank.empty:
                continue
            top = group.loc[finite_rank.idxmax()]
            weight_sum = float(pd.to_numeric(group["_frontier_weight"], errors="coerce").sum())
            rows.append(
                {
                    "fold": int(key[0]),
                    "arm": str(key[1]),
                    "timestamp": pd.Timestamp(key[2]),
                    "head": str(key[3]),
                    "candidate_count": int(len(group)),
                    "weight_sum": weight_sum if np.isfinite(weight_sum) and weight_sum > 0.0 else 1.0,
                    "raw_priority_target": float(
                        pd.to_numeric(pd.Series([top.get("_target_component")]), errors="coerce").iloc[0]
                    ),
                    "mean_resid_utility": float(
                        pd.to_numeric(pd.Series([top.get("resid_utility")]), errors="coerce").iloc[0]
                    ),
                    "mean_resid_full_sl": float(
                        pd.to_numeric(pd.Series([top.get("resid_full_sl")]), errors="coerce").iloc[0]
                    ),
                    "mean_resid_timeout": float(
                        pd.to_numeric(pd.Series([top.get("resid_timeout")]), errors="coerce").iloc[0]
                    ),
                    "mean_net_return": float(
                        pd.to_numeric(pd.Series([top.get("_net_return")]), errors="coerce").iloc[0]
                    ),
                    "baseline_rank_priority": float(finite_rank.max()),
                    "top_candidate_rank": float(finite_rank.max()),
                }
            )
            continue
        threshold_values = pd.to_numeric(group.get("_threshold"), errors="coerce").fillna(0.70)
        rank_values = pd.to_numeric(group["_rank"], errors="coerce")
        weight_values = pd.to_numeric(group["_frontier_weight"], errors="coerce")
        effective_weight = weight_values.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if target_mode == "threshold_admission_mean":
            finite = rank_values.notna() & threshold_values.notna() & (effective_weight > 0.0)
            if int(finite.sum()) < int(min_candidates_per_head_timestamp):
                continue
            rank_gap = rank_values.loc[finite] - threshold_values.loc[finite]
            below_share = float(
                np.average(
                    (rank_gap < 0.0).to_numpy(dtype=float),
                    weights=effective_weight.loc[finite].to_numpy(dtype=float),
                )
            )
        else:
            rank_gap = rank_values - threshold_values
            below_share = float("nan")
        rows.append(
            {
                "fold": int(key[0]),
                "arm": str(key[1]),
                "timestamp": pd.Timestamp(key[2]),
                "head": str(key[3]),
                "candidate_count": int(len(group)),
                "weight_sum": float(pd.to_numeric(group["_frontier_weight"], errors="coerce").sum()),
                "raw_priority_target": _weighted_mean(group["_target_component"], group["_frontier_weight"]),
                "mean_resid_utility": _weighted_mean(group["resid_utility"], group["_frontier_weight"]),
                "mean_resid_full_sl": _weighted_mean(group["resid_full_sl"], group["_frontier_weight"]),
                "mean_resid_timeout": _weighted_mean(group["resid_timeout"], group["_frontier_weight"]),
                "mean_net_return": _weighted_mean(group["_net_return"], group["_frontier_weight"]),
                "baseline_rank_priority": _weighted_mean(rank_values, effective_weight),
                "mean_rank_gap_to_threshold": _weighted_mean(rank_gap, effective_weight),
                "threshold_below_weighted_share": below_share,
            }
        )
    targets = pd.DataFrame(rows)
    if targets.empty:
        return targets, []
    targets = targets.replace([np.inf, -np.inf], np.nan).dropna(subset=["raw_priority_target"])
    targets["_heads_at_timestamp"] = targets.groupby(["fold", "timestamp"], observed=True)[
        "head"
    ].transform("nunique")
    targets = targets.loc[targets["_heads_at_timestamp"] >= 2].copy()
    targets["_centered_raw_priority_target"] = (
        targets["raw_priority_target"]
        - targets.groupby(["fold", "timestamp"], observed=True)["raw_priority_target"].transform("mean")
    )
    if target_mode == "rank_residual_frontier":
        rank_values = pd.to_numeric(targets.get("baseline_rank_priority"), errors="coerce")
        rank_centered = (
            rank_values
            - rank_values.groupby(
                [targets["fold"], targets["timestamp"]],
                observed=True,
            ).transform("mean")
        )
        utility_centered = pd.to_numeric(targets["_centered_raw_priority_target"], errors="coerce")
        weights = pd.to_numeric(targets.get("weight_sum"), errors="coerce").fillna(1.0).clip(lower=0.0)
        valid = rank_centered.notna() & utility_centered.notna() & (weights > 0.0)
        beta = 0.0
        if int(valid.sum()) >= 3:
            x = rank_centered.loc[valid].to_numpy(dtype=float)
            y = utility_centered.loc[valid].to_numpy(dtype=float)
            w = weights.loc[valid].to_numpy(dtype=float)
            w = w / max(float(w.sum()), 1e-12)
            x_bar = float(np.sum(w * x))
            y_bar = float(np.sum(w * y))
            var_x = float(np.sum(w * (x - x_bar) ** 2))
            if np.isfinite(var_x) and var_x > 1e-12:
                beta = float(np.sum(w * (x - x_bar) * (y - y_bar)) / var_x)
        targets["centered_baseline_rank_priority"] = rank_centered
        targets["rank_residual_beta"] = beta
        targets["rank_residual_component"] = float(rank_residual_weight) * beta * rank_centered
        targets["priority_target"] = utility_centered - targets["rank_residual_component"]
    else:
        targets["priority_target"] = targets["_centered_raw_priority_target"]
    targets["priority_target"] = targets["priority_target"].clip(
        lower=-abs(float(target_clip)),
        upper=abs(float(target_clip)),
    )

    state = state_panel.copy()
    state["timestamp"] = pd.to_datetime(state["timestamp"], utc=True, errors="coerce")
    state = state.loc[
        state.get("state_arm", state_arm).astype(str).eq(str(state_arm))
        & state.get("split", "train").astype(str).eq("train")
    ].copy()
    feature_cols = _filter_state_feature_columns(
        state_feature_columns(state),
        allowed_state_heads,
    )
    state_cols = ["fold", "timestamp", *feature_cols]
    state = state[state_cols].drop_duplicates(["fold", "timestamp"])
    merged = targets.merge(state, on=["fold", "timestamp"], how="inner", validate="many_to_one")
    return merged.reset_index(drop=True), feature_cols


def build_score_head_frame(
    score_state_panel: pd.DataFrame,
    candidates: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    state = score_state_panel.copy()
    state["timestamp"] = pd.to_datetime(state["timestamp"], utc=True, errors="coerce")
    if "split" in state.columns:
        state = state.loc[state["split"].astype(str).isin(["score", "valid", "eval"]) | state["split"].isna()].copy()
    present_feature_cols = [c for c in feature_cols if c in state.columns]
    if present_feature_cols:
        coverage = state[present_feature_cols].apply(pd.to_numeric, errors="coerce").notna().sum(axis=1)
    else:
        coverage = pd.Series(0, index=state.index)
    preferred_level = "forecast" if any(str(c).startswith("forecast_") for c in feature_cols) else "observed"
    if "state_level" in state.columns:
        level = state["state_level"].astype(str)
        state["_level_rank"] = np.where(level.eq(preferred_level), 0, np.where(level.isin(["score", "eval"]), 1, 2))
    else:
        state["_level_rank"] = 0
    state["_feature_coverage_count"] = coverage.astype(int)
    state = (
        state.sort_values(
            ["timestamp", "_feature_coverage_count", "_level_rank"],
            ascending=[True, False, True],
        )
        .drop_duplicates("timestamp", keep="first")
        .copy()
    )
    state = state[["timestamp", *present_feature_cols]]
    heads = sorted(candidates["head"].dropna().astype(str).unique())
    rows = []
    for head in heads:
        part = state.copy()
        part["head"] = head
        rows.append(part)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def make_design_matrix(
    frame: pd.DataFrame,
    *,
    feature_cols: list[str],
    heads: list[str],
    medians: dict[str, float] | None = None,
) -> tuple[np.ndarray, dict[str, float], list[str]]:
    out = frame.copy()
    medians_out: dict[str, float] = {} if medians is None else dict(medians)
    matrix_cols: list[str] = []
    parts: list[np.ndarray] = []
    for col in feature_cols:
        raw = out[col] if col in out.columns else pd.Series(np.nan, index=out.index)
        values = pd.to_numeric(raw, errors="coerce")
        if not isinstance(values, pd.Series):
            values = pd.Series(values, index=out.index)
        values = values.replace([np.inf, -np.inf], np.nan)
        if medians is None:
            median = float(values.median()) if int(values.notna().sum()) else 0.0
            if not np.isfinite(median):
                median = 0.0
            medians_out[col] = median
        filled = values.fillna(float(medians_out.get(col, 0.0))).to_numpy(dtype=np.float32)
        parts.append(filled.reshape(-1, 1))
        matrix_cols.append(col)
    head_values = out.get("head", pd.Series("", index=out.index)).astype(str)
    for head in heads:
        parts.append((head_values == head).astype(np.float32).to_numpy().reshape(-1, 1))
        matrix_cols.append(f"head__{head}")
    if not parts:
        return np.empty((len(out), 0), dtype=np.float32), medians_out, matrix_cols
    return np.concatenate(parts, axis=1).astype(np.float32, copy=False), medians_out, matrix_cols


def _timestamp_validation_mask(frame: pd.DataFrame, frac: float) -> pd.Series:
    ts = pd.Series(pd.to_datetime(frame["timestamp"], utc=True, errors="coerce").dropna().unique()).sort_values()
    if len(ts) < 4:
        return pd.Series(False, index=frame.index)
    n_valid = max(1, int(np.ceil(len(ts) * float(frac))))
    valid_ts = set(ts.iloc[-n_valid:])
    return pd.to_datetime(frame["timestamp"], utc=True, errors="coerce").isin(valid_ts)


def _spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 3:
        return float("nan")
    a = pd.Series(y_true).rank(method="average")
    b = pd.Series(y_pred).rank(method="average")
    if a.nunique(dropna=True) <= 1 or b.nunique(dropna=True) <= 1:
        return float("nan")
    return float(a.corr(b, method="pearson"))


def selection_objective(row: dict[str, Any], *, target_clip: float) -> float:
    """Bounded historical validation objective for learned priority models."""
    if "fold_count" in row or "fold_mean_spearman" in row:
        spearman_key = "fold_mean_spearman"
        directional_key = "fold_mean_directional_accuracy"
        mae_key = "fold_mean_mae"
    else:
        spearman_key = "validation_spearman"
        directional_key = "validation_directional_accuracy"
        mae_key = "validation_mae"
    spearman = float(row.get(spearman_key, 0.0) or 0.0)
    if not np.isfinite(spearman):
        spearman = 0.0
    directional = float(row.get(directional_key, 0.0) or 0.0)
    if not np.isfinite(directional):
        directional = 0.0
    mae = float(row.get(mae_key, np.inf))
    if not np.isfinite(mae):
        mae = float("inf")
    spearman_score = float(np.clip(spearman, 0.0, 1.0))
    directional_score = float(np.clip((directional - 0.5) * 2.0, 0.0, 1.0))
    mae_score = float(1.0 - np.clip(mae / max(2.0 * abs(float(target_clip)), 1e-9), 0.0, 1.0))
    base_score = float(0.45 * spearman_score + 0.35 * directional_score + 0.20 * mae_score)
    action_timestamps = int(float(row.get("fold_action_timestamps", 0) or 0))
    action_score = float("nan")
    if action_timestamps > 0:
        action_delta = float(row.get("fold_mean_action_utility_delta", np.nan))
        action_share = float(row.get("fold_action_positive_delta_share", np.nan))
        full_sl_delta = float(row.get("fold_mean_action_full_sl_delta", np.nan))
        if np.isfinite(action_delta) and np.isfinite(action_share):
            delta_score = float(
                np.clip(
                    action_delta / max(abs(float(target_clip)) * 0.25, 1e-9),
                    -1.0,
                    1.0,
                )
            )
            share_score = float(np.clip((action_share - 0.5) * 2.0, -1.0, 1.0))
            sl_score = 0.0
            if np.isfinite(full_sl_delta):
                sl_score = float(np.clip(-full_sl_delta / 0.05, -1.0, 1.0))
            action_score = float(
                np.clip(0.55 * delta_score + 0.30 * share_score + 0.15 * sl_score, 0.0, 1.0)
            )
    if "fold_incremental_objective" in row and pd.notna(row.get("fold_incremental_objective")):
        incremental = float(row.get("fold_incremental_objective", 0.0) or 0.0)
        incremental_score = float(np.clip(incremental / 0.25, 0.0, 1.0))
        if np.isfinite(action_score):
            return float(0.55 * base_score + 0.20 * incremental_score + 0.25 * action_score)
        return float(0.75 * base_score + 0.25 * incremental_score)
    if np.isfinite(action_score):
        return float(0.70 * base_score + 0.30 * action_score)
    return base_score


def selection_gate_passed(row: dict[str, Any], *, gate_mode: str = "defensive") -> bool:
    gate_mode = str(gate_mode or "defensive").strip().lower()
    if gate_mode not in {"defensive", "opportunity"}:
        raise ValueError(f"unknown selection gate mode: {gate_mode}")
    if "fold_count" in row and pd.notna(row.get("fold_count")):
        fold_count = int(row.get("fold_count", 0) or 0)
        validation_rows = int(row.get("fold_validation_rows", 0) or 0)
        spearman = float(row.get("fold_mean_spearman", np.nan))
        positive_spearman_share = float(row.get("fold_positive_spearman_share", np.nan))
        directional = float(row.get("fold_mean_directional_accuracy", np.nan))
        directional_ge_share = float(row.get("fold_directional_ge_50_share", np.nan))
        incremental_objective = float(row.get("fold_incremental_objective", np.nan))
        incremental_spearman = float(row.get("fold_incremental_spearman", np.nan))
        incremental_mae_reduction = float(row.get("fold_incremental_mae_reduction", np.nan))
        action_timestamps = int(float(row.get("fold_action_timestamps", 0) or 0))
        action_delta = float(row.get("fold_mean_action_utility_delta", np.nan))
        action_share = float(row.get("fold_action_positive_delta_share", np.nan))
        action_full_sl_delta = float(row.get("fold_mean_action_full_sl_delta", np.nan))
        action_gate = True
        if action_timestamps > 0:
            max_action_full_sl_delta = 0.03 if gate_mode == "opportunity" else 0.02
            action_gate = bool(
                np.isfinite(action_delta)
                and action_delta >= 0.0
                and np.isfinite(action_share)
                and action_share >= 0.50
                and (
                    not np.isfinite(action_full_sl_delta)
                    or action_full_sl_delta <= max_action_full_sl_delta
                )
            )
        trailing_rows = int(row.get("validation_rows", 0) or 0)
        trailing_spearman = float(row.get("validation_spearman", np.nan))
        trailing_directional = float(row.get("validation_directional_accuracy", np.nan))
        fold_gate = bool(
            fold_count >= 2
            and validation_rows >= 20
            and np.isfinite(spearman)
            and spearman > 0.0
            and np.isfinite(positive_spearman_share)
            and positive_spearman_share >= 0.50
            and np.isfinite(directional)
            and directional >= 0.50
            and np.isfinite(directional_ge_share)
            and directional_ge_share >= 0.50
            and np.isfinite(incremental_objective)
            and incremental_objective > 0.02
            and np.isfinite(incremental_spearman)
            and incremental_spearman > 0.0
            and np.isfinite(incremental_mae_reduction)
            and incremental_mae_reduction > 0.0
            and action_gate
        )
        if gate_mode == "opportunity":
            return fold_gate
        return bool(
            fold_gate
            and trailing_rows >= 10
            and np.isfinite(trailing_spearman)
            and trailing_spearman > -0.20
            and np.isfinite(trailing_directional)
            and trailing_directional >= 0.50
        )
    spearman = float(row.get("validation_spearman", np.nan))
    directional = float(row.get("validation_directional_accuracy", np.nan))
    validation_rows = int(row.get("validation_rows", 0) or 0)
    return bool(
        validation_rows >= 10
        and np.isfinite(spearman)
        and spearman > 0.0
        and np.isfinite(directional)
        and directional >= 0.50
    )


def replay_selection_score(row: dict[str, Any], *, gate_mode: str = "defensive") -> float:
    """Score a candidate using actual portfolio replay and accepted-swap evidence.

    Fit diagnostics decide whether a market-state priority model is coherent.
    This score decides whether a coherent model is worth preferring for the
    portfolio question: did it replace accepted trades with better accepted
    trades without worsening risk plumbing?
    """
    gate_mode = str(gate_mode or "defensive").strip().lower()
    if gate_mode not in {"defensive", "opportunity"}:
        raise ValueError(f"unknown selection gate mode: {gate_mode}")
    net_delta = float(row.get("replay_net_pnl_delta", np.nan))
    action_delta = float(row.get("replay_net_action_pnl_delta", np.nan))
    replacement = float(row.get("replay_net_replacement_pnl", np.nan))
    full_sl_delta = float(row.get("replay_full_sl_delta", np.nan))
    timeout_delta = float(row.get("replay_timeout_delta", np.nan))
    jaccard = float(row.get("replay_accepted_jaccard", np.nan))
    if not np.isfinite(net_delta):
        net_delta = -1e9
    if not np.isfinite(action_delta):
        action_delta = 0.0
    if not np.isfinite(replacement):
        replacement = 0.0
    if not np.isfinite(full_sl_delta):
        full_sl_delta = 0.0
    if not np.isfinite(timeout_delta):
        timeout_delta = 0.0
    if not np.isfinite(jaccard):
        jaccard = 0.0
    if gate_mode == "opportunity":
        risk_penalty = 50.0 * max(full_sl_delta, 0.0) + 25.0 * max(timeout_delta, 0.0)
    else:
        risk_penalty = 100.0 * max(full_sl_delta, 0.0) + 50.0 * max(timeout_delta, 0.0)
    overlap_penalty = 5.0 * max(0.90 - jaccard, 0.0)
    return float(net_delta + 0.50 * action_delta + 0.25 * replacement - risk_penalty - overlap_penalty)


def replay_selection_gate_passed(
    row: dict[str, Any],
    *,
    min_jaccard: float = 0.90,
    min_trade_retention: float = 0.90,
    max_full_sl_delta: float = 0.0,
    max_timeout_delta: float = 0.0,
    gate_mode: str = "defensive",
    relax_opportunity_risk_gates: bool = True,
) -> bool:
    gate_mode = str(gate_mode or "defensive").strip().lower()
    if gate_mode not in {"defensive", "opportunity"}:
        raise ValueError(f"unknown selection gate mode: {gate_mode}")
    if gate_mode == "opportunity" and bool(relax_opportunity_risk_gates):
        max_full_sl_delta = max(float(max_full_sl_delta), 0.02)
        max_timeout_delta = max(float(max_timeout_delta), 0.01)
    base_trades = float(row.get("replay_baseline_trade_count", np.nan))
    trade_delta = float(row.get("replay_trade_count_delta", np.nan))
    if not np.isfinite(base_trades) or base_trades <= 0.0:
        return False
    if not np.isfinite(trade_delta):
        return False
    retention = (base_trades + trade_delta) / max(base_trades, 1.0)
    full_sl_delta = float(row.get("replay_full_sl_delta", np.nan))
    timeout_delta = float(row.get("replay_timeout_delta", np.nan))
    entrants = int(float(row.get("replay_entrants", 0) or 0))
    removed = int(float(row.get("replay_removed", 0) or 0))
    return bool(
        float(row.get("replay_net_pnl_delta", np.nan)) > 0.0
        and retention >= float(min_trade_retention)
        and float(row.get("replay_accepted_jaccard", np.nan)) >= float(min_jaccard)
        and (not np.isfinite(full_sl_delta) or full_sl_delta <= float(max_full_sl_delta))
        and (not np.isfinite(timeout_delta) or timeout_delta <= float(max_timeout_delta))
        and entrants + removed > 0
        and float(row.get("replay_net_replacement_pnl", np.nan)) > 0.0
        and float(row.get("replay_net_action_pnl_delta", np.nan)) > 0.0
        and float(row.get("replay_entrant_net_pnl", np.nan))
        > float(row.get("replay_removed_net_pnl", np.nan))
    )


def _parse_float_list(value: str, default: list[float]) -> list[float]:
    raw = [part.strip() for part in str(value or "").split(",") if part.strip()]
    if not raw:
        return list(default)
    return [float(part) for part in raw]


def _priority_grid(args: argparse.Namespace, backends: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    raw_target_modes = [
        part.strip()
        for part in str(getattr(args, "grid_target_modes", "") or "").split(",")
        if part.strip()
    ]
    target_modes = raw_target_modes or [str(args.target_mode)]
    unknown_target_modes = sorted(set(target_modes).difference(TARGET_MODES))
    if unknown_target_modes:
        raise ValueError(f"unknown target modes: {unknown_target_modes}")
    min_ranks = _parse_float_list(str(args.grid_min_ranks), [float(args.min_rank)])
    gammas = _parse_float_list(str(args.grid_frontier_gammas), [float(args.frontier_gamma)])
    bandwidths = _parse_float_list(str(args.grid_frontier_bandwidths), [float(args.frontier_bandwidth)])
    sl_penalties = _parse_float_list(str(args.grid_sl_penalties), [float(args.sl_penalty)])
    timeout_penalties = _parse_float_list(str(args.grid_timeout_penalties), [float(args.timeout_penalty)])
    rank_residual_weights = _parse_float_list(
        str(getattr(args, "grid_rank_residual_weights", "")),
        [float(getattr(args, "rank_residual_weight", 1.0))],
    )
    max_adjustments = _parse_float_list(str(args.grid_max_adjustments), [float(args.max_adjustment)])
    max_priority_multipliers = _parse_float_list(
        str(args.grid_max_priority_multipliers),
        [float(args.max_priority_multiplier)],
    )
    max_rank_adjustments = _parse_float_list(
        str(getattr(args, "grid_max_rank_adjustments", "")),
        [float(getattr(args, "max_rank_adjustment", 0.0))],
    )
    raw_actions = [
        part.strip().lower()
        for part in str(getattr(args, "grid_priority_actions", "") or "").split(",")
        if part.strip()
    ]
    priority_actions = raw_actions or [str(args.priority_action)]
    unknown_actions = sorted(set(priority_actions).difference(PRIORITY_ACTIONS))
    if unknown_actions:
        raise ValueError(f"unknown priority actions: {unknown_actions}")
    for backend in backends:
        for target_mode in target_modes:
            for min_rank in min_ranks:
                for gamma in gammas:
                    for bandwidth in bandwidths:
                        for sl_penalty in sl_penalties:
                            for timeout_penalty in timeout_penalties:
                                target_rank_weights = (
                                    rank_residual_weights
                                    if str(target_mode) == "rank_residual_frontier"
                                    else [float(getattr(args, "rank_residual_weight", 1.0))]
                                )
                                for rank_residual_weight in target_rank_weights:
                                    for max_adjustment in max_adjustments:
                                        for max_priority_multiplier in max_priority_multipliers:
                                            for max_rank_adjustment in max_rank_adjustments:
                                                for priority_action in priority_actions:
                                                    rows.append(
                                                        {
                                                            "backend": backend,
                                                            "target_mode": str(target_mode),
                                                            "min_rank": float(min_rank),
                                                            "frontier_gamma": float(gamma),
                                                            "frontier_bandwidth": float(bandwidth),
                                                            "sl_penalty": float(sl_penalty),
                                                            "timeout_penalty": float(timeout_penalty),
                                                            "rank_residual_weight": float(rank_residual_weight),
                                                            "min_candidates_per_head_timestamp": int(args.min_candidates_per_head_timestamp),
                                                            "target_clip": float(args.target_clip),
                                                            "max_adjustment": float(max_adjustment),
                                                            "max_priority_multiplier": float(max_priority_multiplier),
                                                            "max_rank_adjustment": float(max_rank_adjustment),
                                                            "priority_action": str(priority_action),
                                                        }
                                                    )
    return rows


def _fit_model(
    backend: str,
    x: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
    *,
    seed: int,
) -> Any:
    if backend == "lgbm":
        import lightgbm as lgb

        min_child = max(10, int(0.05 * len(y)))
        model = lgb.LGBMRegressor(
            objective="huber",
            n_estimators=240,
            learning_rate=0.035,
            max_depth=3,
            num_leaves=8,
            min_child_samples=min_child,
            subsample=0.90,
            colsample_bytree=0.90,
            reg_lambda=3.0,
            random_state=int(seed),
            deterministic=True,
            force_col_wise=True,
            verbosity=-1,
        )
        model.fit(x, y, sample_weight=sample_weight)
        return model
    if backend == "xgb":
        import xgboost as xgb

        model = xgb.XGBRegressor(
            objective="reg:pseudohubererror",
            n_estimators=240,
            learning_rate=0.035,
            max_depth=2,
            min_child_weight=max(5.0, 0.025 * len(y)),
            subsample=0.90,
            colsample_bytree=0.90,
            reg_lambda=3.0,
            tree_method="hist",
            random_state=int(seed),
        )
        model.fit(x, y, sample_weight=sample_weight, verbose=False)
        return model
    raise ValueError(f"unknown backend: {backend}")


def _validation_metric_row(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if len(y_true) == 0:
        return {
            "validation_mae": float("nan"),
            "validation_spearman": float("nan"),
            "validation_directional_accuracy": float("nan"),
        }
    return {
        "validation_mae": float(np.nanmean(np.abs(np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)))),
        "validation_spearman": _spearman(np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)),
        "validation_directional_accuracy": float(np.mean(np.sign(y_pred) == np.sign(y_true))),
    }


def _frontier_candidate_utilities(
    residual_ledger: pd.DataFrame,
    *,
    state_arm: str,
    target_mode: str = "frontier_weighted_mean",
    min_rank: float,
    frontier_gamma: float,
    frontier_bandwidth: float,
    sl_penalty: float,
    timeout_penalty: float,
    min_candidates_per_head_timestamp: int,
) -> pd.DataFrame:
    """Summarize the actual auction-frontier utility by fold/timestamp/head.

    This is intentionally not a realized production replay.  It is a compact
    historical validation proxy for the exact question that the head-priority
    model is meant to answer: should a market state lift one head over another
    near the global auction frontier?
    """
    target_mode = _validate_target_mode(target_mode)
    if residual_ledger.empty:
        return pd.DataFrame()
    ledger = residual_ledger.copy()
    ledger["timestamp"] = pd.to_datetime(ledger["timestamp"], utc=True, errors="coerce")
    ledger = ledger.loc[ledger.get("arm", state_arm).astype(str).eq(str(state_arm))].copy()
    rank = pd.to_numeric(ledger.get("_rank"), errors="coerce")
    ledger = ledger.loc[(rank >= float(min_rank)).fillna(False)].copy()
    if ledger.empty:
        return pd.DataFrame()
    threshold = pd.to_numeric(ledger.get("_threshold"), errors="coerce").fillna(0.70)
    ledger["_frontier_weight"] = _frontier_weights(
        rank,
        threshold,
        target_mode=target_mode,
        frontier_gamma=float(frontier_gamma),
        frontier_bandwidth=float(frontier_bandwidth),
    )
    ledger["_actual_utility"] = (
        pd.to_numeric(ledger.get("resid_utility"), errors="coerce")
        - float(sl_penalty) * pd.to_numeric(ledger.get("resid_full_sl"), errors="coerce")
        - float(timeout_penalty) * pd.to_numeric(ledger.get("resid_timeout"), errors="coerce")
    )
    rows: list[dict[str, Any]] = []
    group_cols = ["fold", "timestamp", "head"]
    for key, group in ledger.groupby(group_cols, sort=True, observed=True):
        if len(group) < int(min_candidates_per_head_timestamp):
            continue
        if target_mode == "head_top_candidate":
            rank_values = pd.to_numeric(group["_rank"], errors="coerce")
            finite_rank = rank_values.replace([np.inf, -np.inf], np.nan).dropna()
            if finite_rank.empty:
                continue
            top = group.loc[finite_rank.idxmax()]
            rows.append(
                {
                    "fold": int(key[0]),
                    "timestamp": pd.Timestamp(key[1]),
                    "head": str(key[2]),
                    "frontier_candidate_count": int(len(group)),
                    "actual_head_utility": float(
                        pd.to_numeric(pd.Series([top.get("_actual_utility")]), errors="coerce").iloc[0]
                    ),
                    "actual_net_return": float(
                        pd.to_numeric(pd.Series([top.get("_net_return")]), errors="coerce").iloc[0]
                    ),
                    "baseline_rank_priority": float(finite_rank.max()),
                    "full_sl_rate": float(
                        pd.to_numeric(pd.Series([top.get("_is_full_sl")]), errors="coerce").iloc[0]
                    ),
                    "timeout_rate": float(
                        pd.to_numeric(pd.Series([top.get("_is_timeout")]), errors="coerce").iloc[0]
                    ),
                }
            )
            continue
        weights = pd.to_numeric(group["_frontier_weight"], errors="coerce")
        rank_values = pd.to_numeric(group["_rank"], errors="coerce")
        threshold_values = pd.to_numeric(group.get("_threshold"), errors="coerce").fillna(0.70)
        rank_gap = rank_values - threshold_values
        rows.append(
            {
                "fold": int(key[0]),
                "timestamp": pd.Timestamp(key[1]),
                "head": str(key[2]),
                "frontier_candidate_count": int(len(group)),
                "actual_head_utility": _weighted_mean(group["_actual_utility"], weights),
                "actual_net_return": _weighted_mean(group["_net_return"], weights),
                "baseline_rank_priority": _weighted_mean(group["_rank"], weights),
                "full_sl_rate": _weighted_mean(group["_is_full_sl"], weights),
                "timeout_rate": _weighted_mean(group["_is_timeout"], weights),
                "mean_rank_gap_to_threshold": _weighted_mean(rank_gap, weights),
                "threshold_below_weighted_share": _weighted_mean((rank_gap < 0.0).astype(float), weights),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["timestamp", "head", "actual_head_utility", "baseline_rank_priority"]
    )
    out["_heads_at_timestamp"] = out.groupby(["fold", "timestamp"], observed=True)[
        "head"
    ].transform("nunique")
    out = out.loc[out["_heads_at_timestamp"] >= 2].drop(columns=["_heads_at_timestamp"])
    return out.reset_index(drop=True)


def _head_selection_utility(frontier: pd.DataFrame, score_col: str) -> pd.DataFrame:
    if frontier.empty or score_col not in frontier.columns:
        return pd.DataFrame()
    work = frontier.copy()
    work["_choice_score"] = pd.to_numeric(work[score_col], errors="coerce")
    work = work.dropna(subset=["timestamp", "head", "_choice_score", "actual_head_utility"])
    if work.empty:
        return pd.DataFrame()
    idx = work.groupby(["fold", "timestamp"], observed=True)["_choice_score"].idxmax()
    selected = work.loc[idx].copy()
    return selected[
        [
            "fold",
            "timestamp",
            "head",
            "actual_head_utility",
            "actual_net_return",
            "full_sl_rate",
            "timeout_rate",
            "_choice_score",
        ]
    ].reset_index(drop=True)


def _head_share_metrics(static_head: pd.Series, model_head: pd.Series) -> dict[str, Any]:
    """Head-agnostic diagnostics for priority-action selection changes."""

    static = static_head.dropna().astype(str)
    model = model_head.dropna().astype(str)
    heads = sorted(set(static.unique()).union(set(model.unique())))
    if not heads:
        return {
            "action_selected_head_switch_share": float("nan"),
            "action_baseline_selected_head_max_share": float("nan"),
            "action_model_selected_head_max_share": float("nan"),
            "action_selected_head_share_l1_shift": float("nan"),
            "action_baseline_selected_head_entropy": float("nan"),
            "action_model_selected_head_entropy": float("nan"),
            "action_selected_head_share_by_head": {},
        }

    def _shares(values: pd.Series) -> dict[str, float]:
        counts = values.value_counts(normalize=True)
        return {head: float(counts.get(head, 0.0)) for head in heads}

    def _entropy(shares: dict[str, float]) -> float:
        vals = np.asarray([v for v in shares.values() if v > 0.0], dtype=float)
        if vals.size == 0:
            return float("nan")
        return float(-(vals * np.log(vals)).sum())

    static_shares = _shares(static)
    model_shares = _shares(model)
    return {
        "action_selected_head_switch_share": float((static_head.astype(str) != model_head.astype(str)).mean()),
        "action_baseline_selected_head_max_share": float(max(static_shares.values())),
        "action_model_selected_head_max_share": float(max(model_shares.values())),
        "action_selected_head_share_l1_shift": float(
            sum(abs(model_shares[head] - static_shares[head]) for head in heads)
        ),
        "action_baseline_selected_head_entropy": _entropy(static_shares),
        "action_model_selected_head_entropy": _entropy(model_shares),
        "action_selected_head_share_by_head": {
            head: {
                "baseline_share": static_shares[head],
                "model_share": model_shares[head],
                "delta": model_shares[head] - static_shares[head],
            }
            for head in heads
        },
    }


def _frontier_action_metric_row(
    frontier: pd.DataFrame,
    *,
    prediction_col: str = "predicted_priority_adjustment",
    multiplier_col: str = "predicted_priority_multiplier",
    rank_adjustment_col: str = "predicted_rank_adjustment",
) -> dict[str, Any]:
    if frontier.empty or prediction_col not in frontier.columns:
        return {
            "action_timestamps": 0,
            "action_mean_utility_delta": float("nan"),
            "action_sum_utility_delta": float("nan"),
            "action_positive_delta_share": float("nan"),
            "action_mean_net_return_delta": float("nan"),
            "action_full_sl_delta": float("nan"),
            "action_timeout_delta": float("nan"),
            "action_selected_head_switch_share": float("nan"),
            "action_baseline_selected_head_max_share": float("nan"),
            "action_model_selected_head_max_share": float("nan"),
            "action_selected_head_share_l1_shift": float("nan"),
            "action_baseline_selected_head_entropy": float("nan"),
            "action_model_selected_head_entropy": float("nan"),
            "action_selected_head_share_by_head": {},
            "action_baseline_short_boll_share": float("nan"),
            "action_model_short_boll_share": float("nan"),
        }
    work = frontier.copy()
    work["static_choice_score"] = pd.to_numeric(work["baseline_rank_priority"], errors="coerce")
    priority_multiplier = (
        pd.to_numeric(work[multiplier_col], errors="coerce").fillna(1.0)
        if multiplier_col in work.columns
        else pd.Series(1.0, index=work.index)
    )
    rank_adjustment = (
        pd.to_numeric(work[rank_adjustment_col], errors="coerce").fillna(0.0)
        if rank_adjustment_col in work.columns
        else pd.Series(0.0, index=work.index)
    )
    work["model_choice_score"] = (
        (pd.to_numeric(work["baseline_rank_priority"], errors="coerce") + rank_adjustment)
        * priority_multiplier
        + pd.to_numeric(work[prediction_col], errors="coerce").fillna(0.0)
    )
    static = _head_selection_utility(work, "static_choice_score")
    model = _head_selection_utility(work, "model_choice_score")
    if static.empty or model.empty:
        return {
            "action_timestamps": 0,
            "action_mean_utility_delta": float("nan"),
            "action_sum_utility_delta": float("nan"),
            "action_positive_delta_share": float("nan"),
            "action_mean_net_return_delta": float("nan"),
            "action_full_sl_delta": float("nan"),
            "action_timeout_delta": float("nan"),
            "action_selected_head_switch_share": float("nan"),
            "action_baseline_selected_head_max_share": float("nan"),
            "action_model_selected_head_max_share": float("nan"),
            "action_selected_head_share_l1_shift": float("nan"),
            "action_baseline_selected_head_entropy": float("nan"),
            "action_model_selected_head_entropy": float("nan"),
            "action_selected_head_share_by_head": {},
            "action_baseline_short_boll_share": float("nan"),
            "action_model_short_boll_share": float("nan"),
        }
    merged = static.merge(
        model,
        on=["fold", "timestamp"],
        how="inner",
        suffixes=("_static", "_model"),
        validate="one_to_one",
    )
    if merged.empty:
        return {
            "action_timestamps": 0,
            "action_mean_utility_delta": float("nan"),
            "action_sum_utility_delta": float("nan"),
            "action_positive_delta_share": float("nan"),
            "action_mean_net_return_delta": float("nan"),
            "action_full_sl_delta": float("nan"),
            "action_timeout_delta": float("nan"),
            "action_baseline_short_boll_share": float("nan"),
            "action_model_short_boll_share": float("nan"),
        }
    util_delta = (
        pd.to_numeric(merged["actual_head_utility_model"], errors="coerce")
        - pd.to_numeric(merged["actual_head_utility_static"], errors="coerce")
    )
    net_delta = (
        pd.to_numeric(merged["actual_net_return_model"], errors="coerce")
        - pd.to_numeric(merged["actual_net_return_static"], errors="coerce")
    )
    full_sl_delta = (
        pd.to_numeric(merged["full_sl_rate_model"], errors="coerce")
        - pd.to_numeric(merged["full_sl_rate_static"], errors="coerce")
    )
    timeout_delta = (
        pd.to_numeric(merged["timeout_rate_model"], errors="coerce")
        - pd.to_numeric(merged["timeout_rate_static"], errors="coerce")
    )
    head_metrics = _head_share_metrics(merged["head_static"], merged["head_model"])
    return {
        "action_timestamps": int(len(merged)),
        "action_mean_utility_delta": float(util_delta.mean()),
        "action_sum_utility_delta": float(util_delta.sum()),
        "action_positive_delta_share": float((util_delta > 0.0).mean()),
        "action_mean_net_return_delta": float(net_delta.mean()),
        "action_full_sl_delta": float(full_sl_delta.mean()),
        "action_timeout_delta": float(timeout_delta.mean()),
        **head_metrics,
        # Backward-compatible June attribution columns. These are diagnostic
        # only; selection and promotion gates must remain head-agnostic.
        "action_baseline_short_boll_share": float(merged["head_static"].astype(str).eq("short_boll").mean()),
        "action_model_short_boll_share": float(merged["head_model"].astype(str).eq("short_boll").mean()),
    }


def validate_priority_model_by_fold(
    train_frame: pd.DataFrame,
    *,
    feature_cols: list[str],
    backend: str,
    target_clip: float,
    seed: int,
    frontier_utilities: pd.DataFrame | None = None,
    max_adjustment: float = 0.20,
    max_priority_multiplier: float = 1.0,
    max_rank_adjustment: float = 0.0,
    priority_action: str = "adjustment",
    min_train_rows: int = 20,
    min_valid_rows: int = 6,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Validate the head-priority learner by holding out complete historical folds."""
    if "fold" not in train_frame.columns:
        return {}, pd.DataFrame()
    folds = sorted(pd.Series(train_frame["fold"]).dropna().unique())
    heads = sorted(train_frame["head"].dropna().astype(str).unique())
    rows: list[dict[str, Any]] = []
    for fold in folds:
        valid_mask = train_frame["fold"].eq(fold)
        train_mask = ~valid_mask
        if int(train_mask.sum()) < int(min_train_rows) or int(valid_mask.sum()) < int(min_valid_rows):
            continue
        x_train, medians, _matrix_cols = make_design_matrix(
            train_frame.loc[train_mask],
            feature_cols=feature_cols,
            heads=heads,
        )
        y_train = pd.to_numeric(
            train_frame.loc[train_mask, "priority_target"],
            errors="coerce",
        ).to_numpy(dtype=np.float32)
        weights = pd.to_numeric(train_frame.loc[train_mask, "weight_sum"], errors="coerce").fillna(1.0)
        weights = (weights / max(float(weights.mean()), 1e-9)).clip(lower=0.25, upper=8.0).to_numpy(dtype=np.float32)
        model = _fit_model(str(backend), x_train, y_train, weights, seed=int(seed) + int(fold))
        train_pred = np.asarray(model.predict(x_train), dtype=float)
        train_finite = train_pred[np.isfinite(train_pred)]
        pred_scale = float(np.nanpercentile(np.abs(train_finite), 75)) if train_finite.size else 1.0
        if not np.isfinite(pred_scale) or pred_scale <= 1e-9:
            pred_scale = float(np.nanstd(train_finite)) if train_finite.size else 1.0
        if not np.isfinite(pred_scale) or pred_scale <= 1e-9:
            pred_scale = 1.0
        x_valid, _medians, _cols = make_design_matrix(
            train_frame.loc[valid_mask],
            feature_cols=feature_cols,
            heads=heads,
            medians=medians,
        )
        y_valid = pd.to_numeric(
            train_frame.loc[valid_mask, "priority_target"],
            errors="coerce",
        ).to_numpy(dtype=np.float32)
        pred_valid = np.asarray(model.predict(x_valid), dtype=float)
        metrics = _validation_metric_row(y_valid, pred_valid)
        if frontier_utilities is not None and not frontier_utilities.empty:
            valid_scored = train_frame.loc[valid_mask, ["fold", "timestamp", "head"]].copy()
            valid_scored["timestamp"] = pd.to_datetime(valid_scored["timestamp"], utc=True, errors="coerce")
            valid_scored["raw_predicted_priority"] = pred_valid
            valid_scored["centered_predicted_priority"] = (
                pd.to_numeric(valid_scored["raw_predicted_priority"], errors="coerce")
                - valid_scored.groupby(["fold", "timestamp"], observed=True)["raw_predicted_priority"].transform("mean")
            )
            adjustment, multiplier = priority_action_values(
                valid_scored["centered_predicted_priority"],
                scale=pred_scale,
                max_adjustment=float(max_adjustment),
                max_priority_multiplier=float(max_priority_multiplier),
                priority_action=str(priority_action),
            )
            rank_adjustment = rank_prior_values(
                valid_scored["centered_predicted_priority"],
                scale=pred_scale,
                max_rank_adjustment=float(max_rank_adjustment),
            )
            valid_scored["predicted_priority_adjustment"] = adjustment
            valid_scored["predicted_priority_multiplier"] = multiplier
            valid_scored["predicted_rank_adjustment"] = rank_adjustment
            valid_frontier = frontier_utilities.loc[
                frontier_utilities["fold"].eq(fold)
            ].copy()
            valid_frontier["timestamp"] = pd.to_datetime(valid_frontier["timestamp"], utc=True, errors="coerce")
            valid_frontier = valid_frontier.merge(
                valid_scored[
                    [
                        "fold",
                        "timestamp",
                        "head",
                        "predicted_priority_adjustment",
                        "predicted_priority_multiplier",
                        "predicted_rank_adjustment",
                    ]
                ],
                on=["fold", "timestamp", "head"],
                how="inner",
                validate="one_to_one",
            )
            metrics.update(_frontier_action_metric_row(valid_frontier))
        metrics.update(
            {
                "validation_fold": int(fold),
                "validation_rows": int(valid_mask.sum()),
                "validation_timestamps": int(
                    pd.to_datetime(
                        train_frame.loc[valid_mask, "timestamp"],
                        utc=True,
                        errors="coerce",
                    ).nunique()
                ),
                "train_rows": int(train_mask.sum()),
            }
        )
        rows.append(metrics)
    fold_df = pd.DataFrame(rows)
    if fold_df.empty:
        return {
            "fold_count": 0,
            "fold_validation_rows": 0,
            "fold_mean_mae": float("nan"),
            "fold_mean_spearman": float("nan"),
            "fold_median_spearman": float("nan"),
            "fold_positive_spearman_share": float("nan"),
            "fold_mean_directional_accuracy": float("nan"),
            "fold_directional_ge_50_share": float("nan"),
            "fold_mean_objective": float("nan"),
            "fold_action_timestamps": 0,
            "fold_mean_action_utility_delta": float("nan"),
            "fold_median_action_utility_delta": float("nan"),
            "fold_action_positive_delta_share": float("nan"),
            "fold_mean_action_net_return_delta": float("nan"),
            "fold_mean_action_full_sl_delta": float("nan"),
            "fold_mean_action_timeout_delta": float("nan"),
            "fold_mean_action_short_boll_share_delta": float("nan"),
        }, fold_df
    objective_rows = [
        selection_objective(row, target_clip=float(target_clip))
        for row in fold_df.to_dict("records")
    ]
    summary = {
        "fold_count": int(len(fold_df)),
        "fold_validation_rows": int(fold_df["validation_rows"].sum()),
        "fold_validation_timestamps": int(fold_df["validation_timestamps"].sum()),
        "fold_mean_mae": float(pd.to_numeric(fold_df["validation_mae"], errors="coerce").mean()),
        "fold_median_mae": float(pd.to_numeric(fold_df["validation_mae"], errors="coerce").median()),
        "fold_mean_spearman": float(pd.to_numeric(fold_df["validation_spearman"], errors="coerce").mean()),
        "fold_median_spearman": float(pd.to_numeric(fold_df["validation_spearman"], errors="coerce").median()),
        "fold_positive_spearman_share": float(
            (pd.to_numeric(fold_df["validation_spearman"], errors="coerce") > 0.0).mean()
        ),
        "fold_mean_directional_accuracy": float(
            pd.to_numeric(fold_df["validation_directional_accuracy"], errors="coerce").mean()
        ),
        "fold_directional_ge_50_share": float(
            (pd.to_numeric(fold_df["validation_directional_accuracy"], errors="coerce") >= 0.50).mean()
        ),
        "fold_mean_objective": float(np.nanmean(objective_rows)),
        "fold_action_timestamps": int(
            pd.to_numeric(fold_df.get("action_timestamps"), errors="coerce").fillna(0).sum()
        )
        if "action_timestamps" in fold_df.columns
        else 0,
        "fold_mean_action_utility_delta": float(
            pd.to_numeric(fold_df.get("action_mean_utility_delta"), errors="coerce").mean()
        )
        if "action_mean_utility_delta" in fold_df.columns
        else float("nan"),
        "fold_median_action_utility_delta": float(
            pd.to_numeric(fold_df.get("action_mean_utility_delta"), errors="coerce").median()
        )
        if "action_mean_utility_delta" in fold_df.columns
        else float("nan"),
        "fold_action_positive_delta_share": float(
            (
                pd.to_numeric(fold_df.get("action_mean_utility_delta"), errors="coerce")
                > 0.0
            ).mean()
        )
        if "action_mean_utility_delta" in fold_df.columns
        else float("nan"),
        "fold_mean_action_net_return_delta": float(
            pd.to_numeric(fold_df.get("action_mean_net_return_delta"), errors="coerce").mean()
        )
        if "action_mean_net_return_delta" in fold_df.columns
        else float("nan"),
        "fold_mean_action_full_sl_delta": float(
            pd.to_numeric(fold_df.get("action_full_sl_delta"), errors="coerce").mean()
        )
        if "action_full_sl_delta" in fold_df.columns
        else float("nan"),
        "fold_mean_action_timeout_delta": float(
            pd.to_numeric(fold_df.get("action_timeout_delta"), errors="coerce").mean()
        )
        if "action_timeout_delta" in fold_df.columns
        else float("nan"),
        "fold_mean_action_selected_head_switch_share": float(
            pd.to_numeric(fold_df.get("action_selected_head_switch_share"), errors="coerce").mean()
        )
        if "action_selected_head_switch_share" in fold_df.columns
        else float("nan"),
        "fold_mean_action_selected_head_share_l1_shift": float(
            pd.to_numeric(fold_df.get("action_selected_head_share_l1_shift"), errors="coerce").mean()
        )
        if "action_selected_head_share_l1_shift" in fold_df.columns
        else float("nan"),
        "fold_mean_action_model_selected_head_max_share": float(
            pd.to_numeric(fold_df.get("action_model_selected_head_max_share"), errors="coerce").mean()
        )
        if "action_model_selected_head_max_share" in fold_df.columns
        else float("nan"),
        "fold_mean_action_short_boll_share_delta": float(
            (
                pd.to_numeric(fold_df.get("action_model_short_boll_share"), errors="coerce")
                - pd.to_numeric(fold_df.get("action_baseline_short_boll_share"), errors="coerce")
            ).mean()
        )
        if {
            "action_model_short_boll_share",
            "action_baseline_short_boll_share",
        }.issubset(fold_df.columns)
        else float("nan"),
    }
    return summary, fold_df


def add_head_only_incremental_validation(
    fold_diag: dict[str, Any],
    head_only_diag: dict[str, Any],
    *,
    target_clip: float,
) -> dict[str, Any]:
    """Attach incremental validation over a static head-only priority model."""
    def _num(mapping: dict[str, Any], key: str) -> float:
        value = float(mapping.get(key, np.nan))
        return value if np.isfinite(value) else float("nan")

    out = dict(fold_diag)
    for key, value in head_only_diag.items():
        out[f"head_only_{key}"] = value
    full_objective = selection_objective(fold_diag, target_clip=float(target_clip))
    head_objective = selection_objective(head_only_diag, target_clip=float(target_clip))
    out["fold_incremental_objective"] = float(full_objective - head_objective)
    out["fold_incremental_spearman"] = float(
        _num(fold_diag, "fold_mean_spearman") - _num(head_only_diag, "fold_mean_spearman")
    )
    out["fold_incremental_directional_accuracy"] = float(
        _num(fold_diag, "fold_mean_directional_accuracy")
        - _num(head_only_diag, "fold_mean_directional_accuracy")
    )
    out["fold_incremental_mae_reduction"] = float(
        _num(head_only_diag, "fold_mean_mae") - _num(fold_diag, "fold_mean_mae")
    )
    return out


def train_priority_model(
    train_frame: pd.DataFrame,
    *,
    feature_cols: list[str],
    backend: str,
    validation_frac: float,
    seed: int,
) -> tuple[Any, dict[str, Any]]:
    heads = sorted(train_frame["head"].dropna().astype(str).unique())
    valid_mask = _timestamp_validation_mask(train_frame, validation_frac)
    train_mask = ~valid_mask
    if int(train_mask.sum()) < 20 or int(valid_mask.sum()) < 6:
        train_mask = pd.Series(True, index=train_frame.index)
        valid_mask = pd.Series(False, index=train_frame.index)
    x_train, medians, matrix_cols = make_design_matrix(
        train_frame.loc[train_mask],
        feature_cols=feature_cols,
        heads=heads,
    )
    y_train = pd.to_numeric(train_frame.loc[train_mask, "priority_target"], errors="coerce").to_numpy(dtype=np.float32)
    weights = pd.to_numeric(train_frame.loc[train_mask, "weight_sum"], errors="coerce").fillna(1.0)
    weights = (weights / max(float(weights.mean()), 1e-9)).clip(lower=0.25, upper=8.0).to_numpy(dtype=np.float32)
    model = _fit_model(str(backend), x_train, y_train, weights, seed=seed)

    diagnostics: dict[str, Any] = {
        "backend": backend,
        "train_rows": int(train_mask.sum()),
        "validation_rows": int(valid_mask.sum()),
        "head_values": heads,
        "feature_count": int(len(matrix_cols)),
        "matrix_columns": matrix_cols,
        "feature_medians": medians,
    }
    if int(valid_mask.sum()) > 0:
        x_valid, _, _ = make_design_matrix(
            train_frame.loc[valid_mask],
            feature_cols=feature_cols,
            heads=heads,
            medians=medians,
        )
        y_valid = pd.to_numeric(train_frame.loc[valid_mask, "priority_target"], errors="coerce").to_numpy(dtype=np.float32)
        pred_valid = np.asarray(model.predict(x_valid), dtype=float)
        diagnostics.update(_validation_metric_row(y_valid, pred_valid))
    x_all, medians_all, matrix_cols_all = make_design_matrix(
        train_frame,
        feature_cols=feature_cols,
        heads=heads,
    )
    y_all = pd.to_numeric(train_frame["priority_target"], errors="coerce").to_numpy(dtype=np.float32)
    weights_all = pd.to_numeric(train_frame["weight_sum"], errors="coerce").fillna(1.0)
    weights_all = (weights_all / max(float(weights_all.mean()), 1e-9)).clip(lower=0.25, upper=8.0).to_numpy(dtype=np.float32)
    final_model = _fit_model(str(backend), x_all, y_all, weights_all, seed=seed)
    train_pred = np.asarray(final_model.predict(x_all), dtype=float)
    finite = train_pred[np.isfinite(train_pred)]
    pred_scale = float(np.nanpercentile(np.abs(finite), 75)) if finite.size else 1.0
    if not np.isfinite(pred_scale) or pred_scale <= 1e-9:
        pred_scale = float(np.nanstd(finite)) if finite.size else 1.0
    if not np.isfinite(pred_scale) or pred_scale <= 1e-9:
        pred_scale = 1.0
    diagnostics.update(
        {
            "all_train_mae": float(np.nanmean(np.abs(train_pred - y_all))),
            "all_train_spearman": _spearman(y_all, train_pred),
            "prediction_scale": pred_scale,
            "final_feature_medians": medians_all,
            "final_matrix_columns": matrix_cols_all,
        }
    )
    return final_model, diagnostics


def score_priority_schedule(
    model: Any,
    score_frame: pd.DataFrame,
    *,
    feature_cols: list[str],
    heads: list[str],
    medians: dict[str, float],
    pred_scale: float,
    max_adjustment: float,
    arm: str,
    max_priority_multiplier: float = 1.0,
    max_rank_adjustment: float = 0.0,
    priority_action: str = "adjustment",
) -> pd.DataFrame:
    x, _, _ = make_design_matrix(score_frame, feature_cols=feature_cols, heads=heads, medians=medians)
    out = score_frame[["timestamp", "head"]].copy()
    out["raw_head_score"] = np.asarray(model.predict(x), dtype=float)
    out["centered_head_score"] = (
        out["raw_head_score"]
        - out.groupby("timestamp", observed=True)["raw_head_score"].transform("mean")
    )
    scale = max(float(pred_scale), 1e-9)
    adjustment, multiplier = priority_action_values(
        out["centered_head_score"],
        scale=scale,
        max_adjustment=float(max_adjustment),
        max_priority_multiplier=float(max_priority_multiplier),
        priority_action=str(priority_action),
    )
    rank_adjustment = rank_prior_values(
        out["centered_head_score"],
        scale=scale,
        max_rank_adjustment=float(max_rank_adjustment),
    )
    out["portfolio_priority_adjustment"] = adjustment
    out["portfolio_priority_multiplier"] = multiplier
    out["portfolio_rank_adjustment"] = rank_adjustment
    out["priority_arm"] = arm
    out["score_column"] = "learned_head_priority_target"
    out["priority_scale"] = scale
    out["priority_action"] = str(priority_action)
    out["head_rows"] = 1
    return out.reset_index(drop=True)


def _replay_arm(
    *,
    arm: str,
    candidates: pd.DataFrame,
    train_deployable: pd.DataFrame,
    params: Any,
    market_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ev_curve = fit_hierarchical_ev_curves(train_deployable)
    decisions, equity, metrics = replay_candidates(
        candidates,
        params,
        mode="global_auction",
        ev_curve=ev_curve,
        market_mode=market_mode,
    )
    accepted = mstc._accepted_trades(candidates, decisions)
    if not accepted.empty:
        accepted["arm"] = arm
    summary = pd.DataFrame([mstc._metrics_row(arm, metrics, accepted, schedule=None)])
    by_head = mstc._by_head(arm, accepted)
    return decisions, equity, accepted, summary, by_head


def replay_selection_metrics(
    *,
    arm: str,
    candidate_summary: pd.DataFrame,
    candidate_accepted: pd.DataFrame,
    base_summary: pd.DataFrame,
    base_accepted: pd.DataFrame,
    gate_mode: str = "defensive",
    min_jaccard: float = 0.90,
    min_trade_retention: float = 0.90,
    max_full_sl_delta: float = 0.0,
    max_timeout_delta: float = 0.0,
    relax_opportunity_risk_gates: bool = True,
) -> dict[str, Any]:
    base_row = base_summary.iloc[0].to_dict() if not base_summary.empty else {}
    cand_row = candidate_summary.iloc[0].to_dict() if not candidate_summary.empty else {}
    overlap = _accepted_overlap({BASELINE_ARM: base_accepted, arm: candidate_accepted})
    overlap_row = _row_by_arm_like(overlap, arm)
    accepted_all = pd.concat([base_accepted, candidate_accepted], ignore_index=True)
    swap = mstc._threshold_action_utility(accepted_all, BASELINE_ARM)
    swap_row = _swap_row_like(swap, arm)
    out = {
        "replay_baseline_trade_count": int(float(base_row.get("trade_count", 0) or 0)),
        "replay_trade_count_delta": int(
            float(cand_row.get("trade_count", 0) or 0)
            - float(base_row.get("trade_count", 0) or 0)
        ),
        "replay_net_pnl_delta": float(cand_row.get("net_pnl", 0.0) or 0.0)
        - float(base_row.get("net_pnl", 0.0) or 0.0),
        "replay_full_sl_delta": float(cand_row.get("full_sl_rate", np.nan))
        - float(base_row.get("full_sl_rate", np.nan)),
        "replay_timeout_delta": float(cand_row.get("timeout_rate", np.nan))
        - float(base_row.get("timeout_rate", np.nan)),
        "replay_accepted_jaccard": float(overlap_row.get("jaccard_vs_baseline", np.nan)),
        "replay_entrants": int(float(swap_row.get("entrants", 0) or 0)),
        "replay_removed": int(float(swap_row.get("removed", 0) or 0)),
        "replay_entrant_net_pnl": float(swap_row.get("entrant_net_pnl", np.nan)),
        "replay_removed_net_pnl": float(swap_row.get("removed_net_pnl", np.nan)),
        "replay_net_replacement_pnl": float(swap_row.get("net_replacement_pnl", np.nan)),
        "replay_net_action_pnl_delta": float(swap_row.get("net_action_pnl_delta", np.nan)),
        "replay_defensive_success": float(swap_row.get("defensive_success", np.nan)),
    }
    out["replay_selection_score"] = replay_selection_score(out, gate_mode=gate_mode)
    out["replay_selection_gate_passed"] = replay_selection_gate_passed(
        out,
        gate_mode=gate_mode,
        min_jaccard=float(min_jaccard),
        min_trade_retention=float(min_trade_retention),
        max_full_sl_delta=float(max_full_sl_delta),
        max_timeout_delta=float(max_timeout_delta),
        relax_opportunity_risk_gates=bool(relax_opportunity_risk_gates),
    )
    out["replay_selection_min_jaccard"] = float(min_jaccard)
    out["replay_selection_min_trade_retention"] = float(min_trade_retention)
    out["replay_selection_max_full_sl_delta"] = float(max_full_sl_delta)
    out["replay_selection_max_timeout_delta"] = float(max_timeout_delta)
    out["replay_selection_relax_opportunity_risk_gates"] = bool(relax_opportunity_risk_gates)
    return out


def _row_by_arm_like(frame: pd.DataFrame, arm: str) -> dict[str, Any]:
    if frame.empty or "arm" not in frame.columns:
        return {}
    rows = frame.loc[frame["arm"].astype(str).eq(str(arm))]
    return rows.iloc[0].to_dict() if not rows.empty else {}


def _swap_row_like(frame: pd.DataFrame, arm: str) -> dict[str, Any]:
    if frame.empty or "arm" not in frame.columns:
        return {}
    work = frame.loc[frame["arm"].astype(str).eq(str(arm))]
    if "scope" in work.columns:
        work = work.loc[work["scope"].astype(str).eq("all")]
    return work.iloc[0].to_dict() if not work.empty else {}


def score_feature_coverage(score_state: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in feature_cols:
        if col in score_state.columns:
            values = pd.to_numeric(score_state[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            finite = int(values.notna().sum())
            rows.append(
                {
                    "feature": col,
                    "present": True,
                    "row_count": int(len(score_state)),
                    "finite_count": finite,
                    "finite_share": float(finite / max(len(score_state), 1)),
                    "filled_with_training_median": finite < len(score_state),
                }
            )
        else:
            rows.append(
                {
                    "feature": col,
                    "present": False,
                    "row_count": int(len(score_state)),
                    "finite_count": 0,
                    "finite_share": 0.0,
                    "filled_with_training_median": True,
                }
            )
    return pd.DataFrame(rows)


def _decision_key(df: pd.DataFrame) -> pd.Series:
    cols = [col for col in ("timestamp", "symbol", "strategy_id", "side", "head") if col in df.columns]
    if not cols:
        return pd.Series(np.arange(len(df)), index=df.index).astype(str)
    values = []
    for col in cols:
        if col == "timestamp":
            values.append(pd.to_datetime(df[col], utc=True, errors="coerce").astype(str))
        else:
            values.append(df[col].astype(str))
    out = values[0]
    for value in values[1:]:
        out = out.str.cat(value, sep="|")
    return out


def _accepted_overlap(accepted: dict[str, pd.DataFrame]) -> pd.DataFrame:
    base = accepted.get(BASELINE_ARM, pd.DataFrame())
    base_keys = set(_decision_key(base)) if not base.empty else set()
    rows = []
    for arm, frame in accepted.items():
        keys = set(_decision_key(frame)) if not frame.empty else set()
        union = base_keys | keys
        inter = base_keys & keys
        rows.append(
            {
                "arm": arm,
                "baseline_accepted": int(len(base_keys)),
                "arm_accepted": int(len(keys)),
                "intersection": int(len(inter)),
                "union": int(len(union)),
                "jaccard_vs_baseline": float(len(inter) / len(union)) if union else 1.0,
                "baseline_only": int(len(base_keys - keys)),
                "arm_only": int(len(keys - base_keys)),
            }
        )
    return pd.DataFrame(rows)


def _decisions_with_candidate_outcomes(
    candidates: pd.DataFrame,
    decisions: pd.DataFrame,
) -> pd.DataFrame:
    """Attach realized candidate economics to replay decisions by cache index."""

    if candidates.empty or decisions.empty:
        return pd.DataFrame()
    norm = mstc.normalise_candidate_table(candidates).reset_index(drop=True)
    keep_cols = [
        col
        for col in (
            "strategy_id",
            "symbol",
            "side",
            "timestamp",
            "net_return",
            "gross_return",
            "simple_policy_exit_reason",
            "portfolio_priority_adjustment",
            "portfolio_priority_multiplier",
        )
        if col in norm.columns
    ]
    candidate_view = norm[keep_cols].copy()
    candidate_view["candidate_index"] = np.arange(len(candidate_view), dtype=np.int64)
    rename = {
        col: f"candidate_{col}"
        for col in ("strategy_id", "symbol", "side", "timestamp")
        if col in candidate_view.columns
    }
    candidate_view = candidate_view.rename(columns=rename)
    dec = decisions.copy()
    if "candidate_index" not in dec.columns:
        return pd.DataFrame()
    dec["candidate_index"] = pd.to_numeric(dec["candidate_index"], errors="coerce").astype("Int64")
    dec = dec.dropna(subset=["candidate_index"]).copy()
    dec["candidate_index"] = dec["candidate_index"].astype(np.int64)
    out = dec.merge(candidate_view, on="candidate_index", how="left", validate="many_to_one")
    if "head" not in out.columns:
        out["head"] = out["strategy_id"].astype(str).map(mstc._infer_head)
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    return out


def priority_starvation_attribution(
    *,
    candidates_by_arm: dict[str, pd.DataFrame],
    decisions_by_arm: dict[str, pd.DataFrame],
    baseline_arm: str = BASELINE_ARM,
) -> pd.DataFrame:
    """Quantify head-level global-auction starvation under fixed thresholds.

    A row is treated as routing-starved when it passed its dynamic rank
    threshold but was rejected by portfolio capacity or auction constraints.
    The diagnostic is descriptive: it does not decide that such rows should
    have been accepted, but it exposes whether a priority arm actually shifts
    scarce auction slots toward useful threshold-passing candidates.
    """

    rows: list[dict[str, Any]] = []
    for arm, decisions in decisions_by_arm.items():
        candidates = candidates_by_arm.get(arm, pd.DataFrame())
        joined = _decisions_with_candidate_outcomes(candidates, decisions)
        if joined.empty:
            continue
        rank = pd.to_numeric(joined.get("normalized_rank_score"), errors="coerce")
        threshold = pd.to_numeric(joined.get("dynamic_threshold"), errors="coerce")
        accepted = joined.get("accepted", False).astype(bool)
        reason = joined.get("rejection_reason", "").astype(str)
        net = pd.to_numeric(joined.get("net_return"), errors="coerce")
        threshold_pass = (rank >= threshold).fillna(False)
        routing_rejected = (
            threshold_pass
            & ~accepted
            & reason.isin(PORTFOLIO_ROUTING_REJECTION_REASONS)
        )
        joined = joined.assign(
            _threshold_pass=threshold_pass,
            _routing_rejected=routing_rejected,
            _accepted=accepted,
            _net_return=net,
        )
        for head, part in joined.groupby(joined["head"].astype(str), sort=True, observed=True):
            tp = part["_threshold_pass"].astype(bool)
            acc = part["_accepted"].astype(bool)
            starved = part["_routing_rejected"].astype(bool)
            starved_net = pd.to_numeric(part.loc[starved, "_net_return"], errors="coerce")
            accepted_net = pd.to_numeric(part.loc[acc, "_net_return"], errors="coerce")
            threshold_pass_count = int(tp.sum())
            rows.append(
                {
                    "arm": arm,
                    "head": str(head),
                    "decision_rows": int(len(part)),
                    "threshold_pass_rows": threshold_pass_count,
                    "accepted_rows": int(acc.sum()),
                    "accepted_share_of_threshold_pass": float(
                        acc.sum() / max(threshold_pass_count, 1)
                    ),
                    "accepted_net_return_sum": float(accepted_net.sum()) if len(accepted_net) else 0.0,
                    "accepted_net_return_mean": float(accepted_net.mean()) if len(accepted_net) else np.nan,
                    "routing_rejected_rows": int(starved.sum()),
                    "routing_rejected_share_of_threshold_pass": float(
                        starved.sum() / max(threshold_pass_count, 1)
                    ),
                    "routing_rejected_net_return_sum": (
                        float(starved_net.sum()) if len(starved_net) else 0.0
                    ),
                    "routing_rejected_net_return_mean": (
                        float(starved_net.mean()) if len(starved_net) else np.nan
                    ),
                    "routing_rejected_positive_rows": int((starved_net > 0.0).sum())
                    if len(starved_net)
                    else 0,
                    "routing_rejected_positive_net_return_sum": float(
                        starved_net.loc[starved_net > 0.0].sum()
                    )
                    if len(starved_net)
                    else 0.0,
                    "mean_portfolio_priority": float(
                        pd.to_numeric(part.get("portfolio_priority"), errors="coerce").mean()
                    ),
                    "mean_rank_score": float(rank.loc[part.index].mean()),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    base = out.loc[out["arm"].astype(str).eq(str(baseline_arm))].copy()
    if base.empty:
        return out
    base = base.set_index("head")
    delta_cols = [
        "threshold_pass_rows",
        "accepted_rows",
        "accepted_net_return_sum",
        "routing_rejected_rows",
        "routing_rejected_net_return_sum",
        "routing_rejected_positive_rows",
        "routing_rejected_positive_net_return_sum",
    ]
    for col in delta_cols:
        out[f"delta_vs_baseline_{col}"] = np.nan
    for idx, row in out.iterrows():
        head = str(row["head"])
        if head not in base.index:
            continue
        for col in delta_cols:
            out.loc[idx, f"delta_vs_baseline_{col}"] = float(row[col]) - float(base.loc[head, col])
    return out.reset_index(drop=True)


def _render_report(
    *,
    manifest: dict[str, Any],
    summary: pd.DataFrame,
    by_head: pd.DataFrame,
    diagnostics: pd.DataFrame,
    overlap: pd.DataFrame,
    swap_attribution: pd.DataFrame,
    score_coverage: pd.DataFrame,
    starvation: pd.DataFrame,
) -> str:
    training_target = str(dict(manifest.get("training") or {}).get("target") or "")
    target_description = (
        "historical head/timestamp threshold-admission residual utility"
        if "threshold_admission" in training_target
        else "historical head/timestamp frontier residual utility"
    )
    lines = [
        "# Learned Market-State Head-Priority Modulation",
        "",
        (
            f"This is a shadow portfolio-routing ablation. It trains a small model on {target_description} and applies bounded pre-filter rank-prior plus auction-priority action columns."
            if dict(manifest.get("contract") or {}).get("rank_prior_layer") == "pre_filter_head_prior"
            else f"This is a shadow portfolio-routing ablation. It trains a small model on {target_description} and applies only bounded auction-priority action columns."
        ),
        "",
        "## Contract",
        "",
        f"- Training walk-forward dir: `{manifest['inputs']['walkforward_dir']}`",
        f"- Score dir: `{manifest['inputs']['score_dir']}`",
        f"- Candidate rows: `{manifest['candidate_universe']['rows']}`",
        f"- Timestamp range: `{manifest['candidate_universe']['timestamp_min']}` to `{manifest['candidate_universe']['timestamp_max']}`",
        f"- Scores/ranks adjusted before threshold: `{bool(manifest['contract'].get('changes_scores_or_ranks'))}`.",
        f"- Rank-prior layer: `{manifest['contract'].get('rank_prior_layer', 'disabled')}`.",
        f"- Head-specific rewards/penalties in selection: `{manifest['contract'].get('head_specific_selection_rewards')}`.",
        "- Thresholds/sizing unchanged: `true`.",
        "- q-fail and HeadHealth unchanged/inactive: `true`.",
        "",
    ]
    parity = dict(manifest.get("static_baseline_candidate_parity") or {})
    if parity.get("checked"):
        lines.extend(
            [
                "## Static Baseline Candidate Scope",
                "",
                f"- Candidate scope: `{parity.get('candidate_scope')}`",
                f"- Promotion-grade deployable scope: `{bool(parity.get('promotion_grade_scope'))}`",
                f"- Current rows: `{parity.get('current_rows')}`",
                f"- Expected deployable rows: `{parity.get('expected_deployable_rows')}`",
                f"- Expected broad rows: `{parity.get('expected_broad_rows')}`",
                f"- Failures: `{', '.join(parity.get('failures') or []) or 'none'}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Replay Summary",
            "",
            "| arm | trades | net_pnl | gross_pnl | cost_pnl | full_sl_rate | timeout_rate | worst_24h_net_pnl |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['arm']} | {int(row['trade_count'])} | "
            f"{float(row['net_pnl']):.6f} | {float(row['gross_pnl']):.6f} | "
            f"{float(row['cost_pnl']):.6f} | {float(row['full_sl_rate']):.6f} | "
            f"{float(row['timeout_rate']):.6f} | {float(row['worst_24h_net_pnl']):.6f} |"
        )
    lines.extend(["", "## By Head", ""])
    lines.append(by_head.to_markdown(index=False) if not by_head.empty else "_No by-head rows._")
    lines.extend(["", "## Model Diagnostics", ""])
    lines.append(diagnostics.to_markdown(index=False) if not diagnostics.empty else "_No diagnostics._")
    lines.extend(["", "## Accepted Overlap", ""])
    lines.append(overlap.to_markdown(index=False) if not overlap.empty else "_No overlap rows._")
    lines.extend(["", "## Accepted Swap Utility", ""])
    if swap_attribution.empty:
        lines.append("_No accepted-set swaps versus baseline._")
    else:
        view_cols = [
            "arm",
            "scope",
            "scope_value",
            "entrants",
            "removed",
            "entrant_net_pnl",
            "removed_net_pnl",
            "net_replacement_pnl",
            "same_key_net_pnl_delta",
            "net_action_pnl_delta",
            "removed_loss_avoided",
            "removed_winner_pnl_sacrificed",
            "defensive_success",
        ]
        view = swap_attribution[[c for c in view_cols if c in swap_attribution.columns]].copy()
        lines.append(view.to_markdown(index=False))
    lines.extend(["", "## Priority Starvation Attribution", ""])
    if starvation.empty:
        lines.append("_No priority-starvation rows._")
    else:
        view_cols = [
            "arm",
            "head",
            "threshold_pass_rows",
            "accepted_rows",
            "routing_rejected_rows",
            "routing_rejected_positive_rows",
            "accepted_net_return_sum",
            "routing_rejected_positive_net_return_sum",
            "delta_vs_baseline_accepted_rows",
            "delta_vs_baseline_routing_rejected_rows",
            "delta_vs_baseline_accepted_net_return_sum",
            "delta_vs_baseline_routing_rejected_positive_net_return_sum",
        ]
        view = starvation[[c for c in view_cols if c in starvation.columns]].copy()
        lines.append(view.to_markdown(index=False))
    lines.extend(["", "## Score Feature Coverage", ""])
    if score_coverage.empty:
        lines.append("_No feature-coverage rows._")
    else:
        view = score_coverage[["feature", "present", "finite_share", "filled_with_training_median"]].copy()
        lines.append(view.to_markdown(index=False))
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--walkforward-dir", type=Path, default=DEFAULT_WALKFORWARD_DIR)
    parser.add_argument("--score-dir", type=Path, default=DEFAULT_SCORE_DIR)
    parser.add_argument("--candidates", type=Path)
    parser.add_argument("--score-state-panel", type=Path)
    parser.add_argument("--train-deployable-candidates", type=Path, default=DEFAULT_TRAIN_DEPLOYABLE)
    parser.add_argument("--policy-manifest", type=Path, default=DEFAULT_POLICY_MANIFEST)
    parser.add_argument("--policy-variant", default="refit_bar4_strategy_bar2")
    parser.add_argument(
        "--static-baseline-manifest",
        type=Path,
        help=(
            "Optional materialized T1 manifest used to load the static P0 "
            "baseline exactly instead of recomputing it from the candidate "
            "parquet. This is required for promotion-grade parity audits."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--backends", default="lgbm,xgb")
    parser.add_argument("--select-config-grid", action="store_true")
    parser.add_argument("--force-select-failing-config", action="store_true")
    parser.add_argument(
        "--grid-target-modes",
        default="",
        help=(
            "Comma-separated target modes to compare during grid selection. "
            "Empty means use --target-mode only."
        ),
    )
    parser.add_argument("--grid-min-ranks", default="0.50,0.65")
    parser.add_argument("--grid-frontier-gammas", default="1.5,3.0")
    parser.add_argument("--grid-frontier-bandwidths", default="0.04,0.08")
    parser.add_argument("--grid-sl-penalties", default="0.0,0.01")
    parser.add_argument("--grid-timeout-penalties", default="0.002")
    parser.add_argument(
        "--grid-max-adjustments",
        default="0.20",
        help=(
            "Comma-separated bounded auction-priority adjustment amplitudes. "
            "This is part of action selection because global-over-time rank "
            "gaps can be wider than the default +/-0.20 tilt."
        ),
    )
    parser.add_argument(
        "--grid-max-priority-multipliers",
        default="1.0",
        help=(
            "Comma-separated bounded multiplicative auction-priority amplitudes. "
            "A value of 1.0 disables the multiplier channel."
        ),
    )
    parser.add_argument(
        "--grid-max-rank-adjustments",
        default="0.0",
        help=(
            "Comma-separated bounded pre-filter rank-prior amplitudes. "
            "A value of 0.0 preserves the legacy auction-only action."
        ),
    )
    parser.add_argument(
        "--grid-priority-actions",
        default="",
        help=(
            "Comma-separated priority action families to test: adjustment, "
            "multiplier, both. Empty means use --priority-action only."
        ),
    )
    parser.add_argument(
        "--grid-rank-residual-weights",
        default="0.5,1.0",
        help=(
            "Comma-separated residualization strengths for rank_residual_frontier. "
            "Ignored for other target modes."
        ),
    )
    parser.add_argument("--state-arm", default="S1_observed_axes_shared_response")
    parser.add_argument("--min-rank", type=float, default=0.50)
    parser.add_argument("--frontier-gamma", type=float, default=3.0)
    parser.add_argument("--frontier-bandwidth", type=float, default=0.06)
    parser.add_argument("--sl-penalty", type=float, default=0.010)
    parser.add_argument("--timeout-penalty", type=float, default=0.002)
    parser.add_argument("--min-candidates-per-head-timestamp", type=int, default=3)
    parser.add_argument(
        "--target-mode",
        choices=sorted(TARGET_MODES),
        default="frontier_weighted_mean",
        help=(
            "How to summarize per-head timestamp opportunity quality. "
            "frontier_weighted_mean uses all rank-frontier candidates; "
            "head_top_candidate uses the best ranked candidate per head, "
            "which better matches scarce global-auction priority; "
            "rank_residual_frontier predicts frontier utility after removing "
            "the part explained by current global rank, which targets market "
            "states where a head is useful but under-prioritized; "
            "threshold_admission_mean focuses weight on marginal rows around "
            "the deployment threshold, which is the appropriate target for "
            "pre-filter rank-prior admission tests."
        ),
    )
    parser.add_argument("--target-clip", type=float, default=0.08)
    parser.add_argument(
        "--rank-residual-weight",
        type=float,
        default=1.0,
        help=(
            "Strength of rank residualization for rank_residual_frontier. "
            "1.0 removes the fitted current-rank component from the priority target."
        ),
    )
    parser.add_argument("--max-adjustment", type=float, default=0.20)
    parser.add_argument("--max-priority-multiplier", type=float, default=1.0)
    parser.add_argument("--max-rank-adjustment", type=float, default=0.0)
    parser.add_argument(
        "--priority-action",
        choices=sorted(PRIORITY_ACTIONS),
        default="adjustment",
    )
    parser.add_argument("--validation-frac", type=float, default=0.25)
    parser.add_argument("--validation-mode", choices=["trailing", "fold_aware"], default="trailing")
    parser.add_argument(
        "--activation-registry",
        type=Path,
        default=None,
        help=(
            "Optional market_state_activation_registry.csv. When omitted and "
            "--use-all-state-heads is not set, defaults to "
            "<walkforward-dir>/market_state_activation_registry.csv if present."
        ),
    )
    parser.add_argument(
        "--state-head-statuses",
        default="active_candidate",
        help=(
            "Comma-separated recommended_status values allowed into the priority "
            "design matrix when an activation registry is available."
        ),
    )
    parser.add_argument(
        "--use-all-state-heads",
        action="store_true",
        help=(
            "Opt out of activation-registry pruning and allow all numeric "
            "state_/forecast_ columns into the shadow priority learner."
        ),
    )
    parser.add_argument(
        "--selection-gate-mode",
        choices=["defensive", "opportunity"],
        default="defensive",
        help=(
            "Gate used for model/config selection. 'defensive' preserves the older "
            "suppression-style no-risk-worsening contract; 'opportunity' is for "
            "head-priority routing and allows small full-SL/timeout drift only when "
            "accepted replacements improve portfolio utility."
        ),
    )
    parser.add_argument("--selection-min-accepted-jaccard", type=float, default=0.90)
    parser.add_argument("--selection-min-trade-retention", type=float, default=0.90)
    parser.add_argument("--selection-max-full-sl-delta", type=float, default=0.0)
    parser.add_argument("--selection-max-timeout-delta", type=float, default=0.0)
    parser.add_argument(
        "--strict-replay-risk-gates",
        action="store_true",
        help=(
            "Do not relax full-SL/timeout caps in opportunity mode. Use this "
            "for shadow-promotion-grade routing tests."
        ),
    )
    parser.add_argument(
        "--selection-replay-top-n",
        type=int,
        default=0,
        help=(
            "If >0, replay the top-N grid candidates after fit validation and "
            "select only from candidates with replay-level accepted-swap evidence."
        ),
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--allow-missing-schedule", action="store_true")
    parser.add_argument("--market-mode", default="perps")
    args = parser.parse_args()

    final_output_dir, staged_output_dir, output_staging_report = _prepare_staged_output_dir(
        args.output_dir
    )
    published = {"done": False}

    def _cleanup_staging() -> None:
        if not published["done"] and staged_output_dir.exists():
            shutil.rmtree(staged_output_dir, ignore_errors=True)

    def _handle_shutdown(signum: int, _frame: Any) -> None:
        _cleanup_staging()
        raise SystemExit(128 + int(signum))

    atexit.register(_cleanup_staging)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _handle_shutdown)
    args.output_dir = staged_output_dir
    candidates_path = args.candidates or (args.score_dir / "controller_scored_candidates.parquet")
    score_state_path = args.score_state_panel or (args.score_dir / "market_state_timestamp_panel.parquet")
    residual_path = args.walkforward_dir / "strategy_residual_target_ledger.parquet"
    train_state_path = args.walkforward_dir / "market_state_timestamp_panel.parquet"
    if not residual_path.exists():
        raise FileNotFoundError(residual_path)
    if not train_state_path.exists():
        raise FileNotFoundError(train_state_path)
    if not candidates_path.exists():
        raise FileNotFoundError(candidates_path)
    if not score_state_path.exists():
        raise FileNotFoundError(score_state_path)

    activation_registry_path = None
    if not bool(args.use_all_state_heads):
        activation_registry_path = args.activation_registry or (
            args.walkforward_dir / "market_state_activation_registry.csv"
        )
    allowed_statuses = {
        value.strip()
        for value in str(args.state_head_statuses).split(",")
        if value.strip()
    } or {"active_candidate"}
    allowed_state_heads, activation_registry_report = _load_active_state_heads(
        activation_registry_path,
        allowed_statuses=allowed_statuses,
    )
    if bool(args.use_all_state_heads):
        activation_registry_report = {
            "enabled": False,
            "reason": "explicit_use_all_state_heads",
            "allowed_statuses": sorted(allowed_statuses),
        }

    residual = pd.read_parquet(residual_path)
    train_state = pd.read_parquet(train_state_path)
    score_state = pd.read_parquet(score_state_path)
    candidates = _load_candidates(candidates_path)
    train_deployable, train_deployable_contract = load_train_deployable_for_static_contract(
        args.train_deployable_candidates,
        static_baseline_manifest=args.static_baseline_manifest,
    )
    params, policy_payload = mstc._load_policy_params(args.policy_manifest, args.policy_variant)

    base_train_frame, base_feature_cols = build_head_priority_targets(
        residual,
        train_state,
        state_arm=str(args.state_arm),
        allowed_state_heads=allowed_state_heads,
        target_mode=str(args.target_mode),
        min_rank=float(args.min_rank),
        frontier_gamma=float(args.frontier_gamma),
        frontier_bandwidth=float(args.frontier_bandwidth),
        sl_penalty=float(args.sl_penalty),
        timeout_penalty=float(args.timeout_penalty),
        min_candidates_per_head_timestamp=int(args.min_candidates_per_head_timestamp),
        target_clip=float(args.target_clip),
        rank_residual_weight=float(args.rank_residual_weight),
    )
    if base_train_frame.empty or not base_feature_cols:
        raise RuntimeError("no learned priority training rows/features were generated")
    base_frontier_utilities = _frontier_candidate_utilities(
        residual,
        state_arm=str(args.state_arm),
        target_mode=str(args.target_mode),
        min_rank=float(args.min_rank),
        frontier_gamma=float(args.frontier_gamma),
        frontier_bandwidth=float(args.frontier_bandwidth),
        sl_penalty=float(args.sl_penalty),
        timeout_penalty=float(args.timeout_penalty),
        min_candidates_per_head_timestamp=int(args.min_candidates_per_head_timestamp),
    )

    accepted_by_arm: dict[str, pd.DataFrame] = {}
    candidates_by_arm: dict[str, pd.DataFrame] = {}
    decisions_by_arm: dict[str, pd.DataFrame] = {}
    summary_frames: list[pd.DataFrame] = []
    by_head_frames: list[pd.DataFrame] = []
    schedule_frames: list[pd.DataFrame] = []
    diagnostic_rows: list[dict[str, Any]] = []

    static_baseline_info: dict[str, Any] | None = None
    static_baseline = _load_static_baseline_artifacts(
        args.static_baseline_manifest,
        arm=BASELINE_ARM,
    )
    if static_baseline is None:
        base_candidates = candidates.assign(
            portfolio_priority_adjustment=0.0,
            portfolio_priority_multiplier=1.0,
        )
        base_decisions, base_equity, base_accepted, base_summary, base_by_head = _replay_arm(
            arm=BASELINE_ARM,
            candidates=base_candidates,
            train_deployable=train_deployable,
            params=params,
            market_mode=str(args.market_mode),
        )
    else:
        (
            base_decisions,
            base_equity,
            base_accepted,
            base_summary,
            base_by_head,
            static_baseline_info,
        ) = static_baseline
        base_candidates = candidates.assign(
            portfolio_priority_adjustment=0.0,
            portfolio_priority_multiplier=1.0,
        )
    candidates_by_arm[BASELINE_ARM] = base_candidates
    decisions_by_arm[BASELINE_ARM] = base_decisions
    accepted_by_arm[BASELINE_ARM] = base_accepted
    summary_frames.append(base_summary)
    by_head_frames.append(base_by_head)
    base_decisions.to_parquet(args.output_dir / f"{BASELINE_ARM}_decisions.parquet", index=False)
    base_equity.to_parquet(args.output_dir / f"{BASELINE_ARM}_equity.parquet", index=False)
    base_accepted.to_parquet(args.output_dir / f"{BASELINE_ARM}_accepted_trades.parquet", index=False)

    backends = [b.strip() for b in str(args.backends).split(",") if b.strip()]
    model_specs: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    fold_validation_frames: list[pd.DataFrame] = []
    if bool(args.select_config_grid):
        for cfg in _priority_grid(args, backends):
            train_frame, feature_cols = build_head_priority_targets(
                residual,
                train_state,
                state_arm=str(args.state_arm),
                allowed_state_heads=allowed_state_heads,
                target_mode=str(cfg["target_mode"]),
                min_rank=float(cfg["min_rank"]),
                frontier_gamma=float(cfg["frontier_gamma"]),
                frontier_bandwidth=float(cfg["frontier_bandwidth"]),
                sl_penalty=float(cfg["sl_penalty"]),
                timeout_penalty=float(cfg["timeout_penalty"]),
                min_candidates_per_head_timestamp=int(cfg["min_candidates_per_head_timestamp"]),
                target_clip=float(cfg["target_clip"]),
                rank_residual_weight=float(cfg.get("rank_residual_weight", args.rank_residual_weight)),
            )
            if train_frame.empty or not feature_cols:
                continue
            model, diag = train_priority_model(
                train_frame,
                feature_cols=feature_cols,
                backend=str(cfg["backend"]),
                validation_frac=float(args.validation_frac),
                seed=int(args.seed),
            )
            fold_diag: dict[str, Any] = {}
            if str(args.validation_mode) == "fold_aware":
                fold_diag, fold_df = validate_priority_model_by_fold(
                    train_frame,
                    feature_cols=feature_cols,
                    backend=str(cfg["backend"]),
                    target_clip=float(cfg["target_clip"]),
                    seed=int(args.seed),
                    frontier_utilities=_frontier_candidate_utilities(
                        residual,
                        state_arm=str(args.state_arm),
                        target_mode=str(cfg["target_mode"]),
                        min_rank=float(cfg["min_rank"]),
                        frontier_gamma=float(cfg["frontier_gamma"]),
                        frontier_bandwidth=float(cfg["frontier_bandwidth"]),
                        sl_penalty=float(cfg["sl_penalty"]),
                        timeout_penalty=float(cfg["timeout_penalty"]),
                        min_candidates_per_head_timestamp=int(cfg["min_candidates_per_head_timestamp"]),
                    ),
                    max_adjustment=float(cfg["max_adjustment"]),
                    max_priority_multiplier=float(cfg.get("max_priority_multiplier", args.max_priority_multiplier)),
                    max_rank_adjustment=float(cfg.get("max_rank_adjustment", args.max_rank_adjustment)),
                    priority_action=str(cfg.get("priority_action", args.priority_action)),
                )
                head_only_diag, head_only_fold_df = validate_priority_model_by_fold(
                    train_frame,
                    feature_cols=[],
                    backend=str(cfg["backend"]),
                    target_clip=float(cfg["target_clip"]),
                    seed=int(args.seed),
                    frontier_utilities=_frontier_candidate_utilities(
                        residual,
                        state_arm=str(args.state_arm),
                        target_mode=str(cfg["target_mode"]),
                        min_rank=float(cfg["min_rank"]),
                        frontier_gamma=float(cfg["frontier_gamma"]),
                        frontier_bandwidth=float(cfg["frontier_bandwidth"]),
                        sl_penalty=float(cfg["sl_penalty"]),
                        timeout_penalty=float(cfg["timeout_penalty"]),
                        min_candidates_per_head_timestamp=int(cfg["min_candidates_per_head_timestamp"]),
                    ),
                    max_adjustment=float(cfg["max_adjustment"]),
                    max_priority_multiplier=float(cfg.get("max_priority_multiplier", args.max_priority_multiplier)),
                    max_rank_adjustment=float(cfg.get("max_rank_adjustment", args.max_rank_adjustment)),
                    priority_action=str(cfg.get("priority_action", args.priority_action)),
                )
                fold_diag = add_head_only_incremental_validation(
                    fold_diag,
                    head_only_diag,
                    target_clip=float(cfg["target_clip"]),
                )
                diag.update(fold_diag)
                if not fold_df.empty:
                    fold_df = fold_df.copy()
                    fold_df["config_id"] = int(len(selection_rows) + 1)
                    for key, value in cfg.items():
                        fold_df[key] = value
                    if not head_only_fold_df.empty:
                        head_cols = {
                            col: f"head_only_{col}"
                            for col in head_only_fold_df.columns
                            if col != "validation_fold"
                        }
                        head_only_view = head_only_fold_df.rename(columns=head_cols)
                        fold_df = fold_df.merge(
                            head_only_view,
                            on="validation_fold",
                            how="left",
                            validate="one_to_one",
                        )
                    fold_validation_frames.append(fold_df)
            objective = selection_objective(diag, target_clip=float(cfg["target_clip"]))
            row = {
                **cfg,
                **{
                    k: v
                    for k, v in diag.items()
                    if k
                    not in {
                        "feature_medians",
                        "matrix_columns",
                        "final_feature_medians",
                        "final_matrix_columns",
                    }
                },
                "selection_objective": objective,
                "training_rows_after_target_filters": int(len(train_frame)),
                "state_feature_count": int(len(feature_cols)),
            }
            row["selection_gate_passed"] = selection_gate_passed(
                row,
                gate_mode=str(args.selection_gate_mode),
            )
            selection_rows.append(row)
            model_specs.append(
                {
                    "arm": f"{SELECTED_ARM_PREFIX}_{cfg['backend']}_candidate_{len(selection_rows):03d}",
                    "backend": str(cfg["backend"]),
                    "config": cfg,
                    "model": model,
                    "diag": diag,
                    "train_frame": train_frame,
                    "feature_cols": feature_cols,
                }
            )
        if not selection_rows:
            raise RuntimeError("selection grid produced no trainable configs")
        selection_df = pd.DataFrame(selection_rows)
        if int(args.selection_replay_top_n) > 0 and not selection_df.empty:
            selection_df["pre_replay_selection_gate_passed"] = selection_df[
                "selection_gate_passed"
            ].astype(bool)
            selection_df["selection_replay_evaluated"] = False
            shortlist_source = selection_df.loc[
                selection_df["pre_replay_selection_gate_passed"].astype(bool)
            ].copy()
            if shortlist_source.empty and bool(args.force_select_failing_config):
                shortlist_source = selection_df.copy()
            shortlist = shortlist_source.sort_values(
                ["selection_objective", "validation_mae"],
                ascending=[False, True],
            ).head(int(args.selection_replay_top_n))
            for idx in shortlist.index:
                spec = model_specs[int(idx)]
                backend = str(spec["backend"])
                feature_cols = list(spec["feature_cols"])
                train_frame = spec["train_frame"]
                heads = sorted(train_frame["head"].dropna().astype(str).unique())
                score_frame = build_score_head_frame(score_state, candidates, feature_cols)
                if score_frame.empty:
                    continue
                diag = dict(spec["diag"])
                probe_arm = f"{SELECTED_ARM_PREFIX}_{backend}_candidate_{int(idx) + 1:03d}_replay_probe"
                schedule = score_priority_schedule(
                    spec["model"],
                    score_frame,
                    feature_cols=feature_cols,
                    heads=heads,
                    medians=diag["final_feature_medians"],
                    pred_scale=float(diag["prediction_scale"]),
                    max_adjustment=float(dict(spec["config"]).get("max_adjustment", args.max_adjustment)),
                    max_priority_multiplier=float(
                        dict(spec["config"]).get(
                            "max_priority_multiplier",
                            args.max_priority_multiplier,
                        )
                    ),
                    max_rank_adjustment=float(
                        dict(spec["config"]).get(
                            "max_rank_adjustment",
                            args.max_rank_adjustment,
                        )
                    ),
                    priority_action=str(
                        dict(spec["config"]).get("priority_action", args.priority_action)
                    ),
                    arm=probe_arm,
                )
                arm_candidates, _coverage = apply_head_priority_schedule(
                    candidates,
                    schedule,
                    fail_closed=not bool(args.allow_missing_schedule),
                )
                _decisions, _equity, accepted, summary_part, _by_head_part = _replay_arm(
                    arm=probe_arm,
                    candidates=arm_candidates,
                    train_deployable=train_deployable,
                    params=params,
                    market_mode=str(args.market_mode),
                )
                metrics = replay_selection_metrics(
                    arm=probe_arm,
                    candidate_summary=summary_part,
                    candidate_accepted=accepted,
                    base_summary=base_summary,
                    base_accepted=base_accepted,
                    gate_mode=str(args.selection_gate_mode),
                    min_jaccard=float(args.selection_min_accepted_jaccard),
                    min_trade_retention=float(args.selection_min_trade_retention),
                    max_full_sl_delta=float(args.selection_max_full_sl_delta),
                    max_timeout_delta=float(args.selection_max_timeout_delta),
                    relax_opportunity_risk_gates=not bool(args.strict_replay_risk_gates),
                )
                for key, value in metrics.items():
                    selection_df.loc[idx, key] = value
                selection_df.loc[idx, "selection_replay_evaluated"] = True
                replay_gate = bool(metrics.get("replay_selection_gate_passed"))
                pre_gate = bool(selection_df.loc[idx, "pre_replay_selection_gate_passed"])
                selection_df.loc[idx, "selection_gate_passed"] = bool(pre_gate and replay_gate)
                spec["diag"].update(metrics)
                spec["diag"]["pre_replay_selection_gate_passed"] = pre_gate
                spec["diag"]["selection_replay_evaluated"] = True
                spec["diag"]["selection_gate_passed"] = bool(pre_gate and replay_gate)
                spec["selection_replay_evaluated"] = True
                spec["pre_replay_selection_gate_passed"] = pre_gate
                spec["selection_gate_passed"] = bool(pre_gate and replay_gate)
            not_eval = ~selection_df["selection_replay_evaluated"].astype(bool)
            selection_df.loc[not_eval, "selection_gate_passed"] = False
            for idx in selection_df.index[not_eval]:
                model_specs[int(idx)]["selection_gate_passed"] = False
                model_specs[int(idx)]["selection_replay_evaluated"] = False
        selectable = selection_df.loc[selection_df["selection_gate_passed"].astype(bool)].copy()
        if selectable.empty and not bool(args.force_select_failing_config):
            sort_cols = ["selection_objective", "validation_mae"]
            sort_asc = [False, True]
            if "selection_replay_evaluated" in selection_df.columns:
                sort_cols.insert(0, "selection_replay_evaluated")
                sort_asc.insert(0, False)
            selection_df = selection_df.sort_values(sort_cols, ascending=sort_asc)
            model_specs = []
        else:
            rank_frame = selectable if not selectable.empty else selection_df
            if int(args.selection_replay_top_n) > 0 and "replay_selection_score" in rank_frame.columns:
                rank_frame = rank_frame.sort_values(
                    [
                        "selection_gate_passed",
                        "replay_selection_score",
                        "selection_objective",
                        "validation_mae",
                    ],
                    ascending=[False, False, False, True],
                )
            else:
                rank_frame = rank_frame.sort_values(
                    ["selection_objective", "validation_mae"],
                    ascending=[False, True],
                )
            best_index = int(rank_frame.index[0])
            best_spec = model_specs[best_index]
            best_spec["arm"] = f"{SELECTED_ARM_PREFIX}_{best_spec['backend']}_priority"
            best_spec["selection_gate_passed"] = bool(selection_rows[best_index].get("selection_gate_passed"))
            if int(args.selection_replay_top_n) > 0:
                best_spec["selection_gate_passed"] = bool(
                    selection_df.loc[best_index, "selection_gate_passed"]
                )
            best_spec["selection_objective"] = float(selection_rows[best_index].get("selection_objective", np.nan))
            model_specs = [best_spec]
            sort_cols = ["selection_gate_passed"]
            sort_asc = [False]
            if "replay_selection_score" in selection_df.columns:
                sort_cols.append("replay_selection_score")
                sort_asc.append(False)
            sort_cols.extend(["selection_objective", "validation_mae"])
            sort_asc.extend([False, True])
            selection_df = selection_df.sort_values(sort_cols, ascending=sort_asc)
    else:
        selection_df = pd.DataFrame()
        for backend in backends:
            cfg = {
                "backend": backend,
                "target_mode": str(args.target_mode),
                "min_rank": float(args.min_rank),
                "frontier_gamma": float(args.frontier_gamma),
                "frontier_bandwidth": float(args.frontier_bandwidth),
                "sl_penalty": float(args.sl_penalty),
                "timeout_penalty": float(args.timeout_penalty),
                "rank_residual_weight": float(args.rank_residual_weight),
                "min_candidates_per_head_timestamp": int(args.min_candidates_per_head_timestamp),
                "target_clip": float(args.target_clip),
                "max_adjustment": float(args.max_adjustment),
                "max_priority_multiplier": float(args.max_priority_multiplier),
                "max_rank_adjustment": float(args.max_rank_adjustment),
                "priority_action": str(args.priority_action),
            }
            model, diag = train_priority_model(
                base_train_frame,
                feature_cols=base_feature_cols,
                backend=backend,
                validation_frac=float(args.validation_frac),
                seed=int(args.seed),
            )
            if str(args.validation_mode) == "fold_aware":
                fold_diag, fold_df = validate_priority_model_by_fold(
                    base_train_frame,
                    feature_cols=base_feature_cols,
                    backend=backend,
                    target_clip=float(args.target_clip),
                    seed=int(args.seed),
                    frontier_utilities=base_frontier_utilities,
                    max_adjustment=float(cfg["max_adjustment"]),
                    max_priority_multiplier=float(cfg["max_priority_multiplier"]),
                    max_rank_adjustment=float(cfg.get("max_rank_adjustment", args.max_rank_adjustment)),
                    priority_action=str(cfg["priority_action"]),
                )
                head_only_diag, head_only_fold_df = validate_priority_model_by_fold(
                    base_train_frame,
                    feature_cols=[],
                    backend=backend,
                    target_clip=float(args.target_clip),
                    seed=int(args.seed),
                    frontier_utilities=base_frontier_utilities,
                    max_adjustment=float(cfg["max_adjustment"]),
                    max_priority_multiplier=float(cfg["max_priority_multiplier"]),
                    priority_action=str(cfg["priority_action"]),
                )
                fold_diag = add_head_only_incremental_validation(
                    fold_diag,
                    head_only_diag,
                    target_clip=float(args.target_clip),
                )
                diag.update(fold_diag)
                if not fold_df.empty:
                    fold_df = fold_df.copy()
                    fold_df["arm"] = LEARNED_ARMS[backend]
                    fold_df["backend"] = backend
                    if not head_only_fold_df.empty:
                        head_cols = {
                            col: f"head_only_{col}"
                            for col in head_only_fold_df.columns
                            if col != "validation_fold"
                        }
                        head_only_view = head_only_fold_df.rename(columns=head_cols)
                        fold_df = fold_df.merge(
                            head_only_view,
                            on="validation_fold",
                            how="left",
                            validate="one_to_one",
                        )
                    fold_validation_frames.append(fold_df)
            model_specs.append(
                {
                    "arm": LEARNED_ARMS[backend],
                    "backend": backend,
                    "config": cfg,
                    "model": model,
                    "diag": diag,
                    "train_frame": base_train_frame,
                    "feature_cols": base_feature_cols,
                }
            )

    for spec in model_specs:
        backend = str(spec["backend"])
        if backend not in LEARNED_ARMS:
            raise ValueError(f"unknown backend: {backend}")
        arm = str(spec["arm"])
        model = spec["model"]
        diag = dict(spec["diag"])
        feature_cols = list(spec["feature_cols"])
        train_frame = spec["train_frame"]
        score_frame = build_score_head_frame(score_state, candidates, feature_cols)
        if score_frame.empty:
            raise RuntimeError(f"no score head rows were generated for {arm}")
        heads = sorted(train_frame["head"].dropna().astype(str).unique())
        diag_row = {
            k: v
            for k, v in diag.items()
            if k
            not in {
                "feature_medians",
                "matrix_columns",
                "final_feature_medians",
                "final_matrix_columns",
            }
        }
        diag_row.update({f"config_{k}": v for k, v in dict(spec["config"]).items()})
        diag_row["selection_objective"] = selection_objective(
            diag,
            target_clip=float(dict(spec["config"]).get("target_clip", args.target_clip)),
        )
        diag_row["selection_gate_passed"] = bool(spec.get("selection_gate_passed", True))
        diagnostic_rows.append(diag_row)
        schedule = score_priority_schedule(
            model,
            score_frame,
            feature_cols=feature_cols,
            heads=heads,
            medians=diag["final_feature_medians"],
            pred_scale=float(diag["prediction_scale"]),
            max_adjustment=float(dict(spec["config"]).get("max_adjustment", args.max_adjustment)),
            max_priority_multiplier=float(
                dict(spec["config"]).get(
                    "max_priority_multiplier",
                    args.max_priority_multiplier,
                )
            ),
            max_rank_adjustment=float(
                dict(spec["config"]).get(
                    "max_rank_adjustment",
                    args.max_rank_adjustment,
                )
            ),
            priority_action=str(dict(spec["config"]).get("priority_action", args.priority_action)),
            arm=arm,
        )
        arm_candidates, coverage = apply_head_priority_schedule(
            candidates,
            schedule,
            fail_closed=not bool(args.allow_missing_schedule),
        )
        schedule["coverage"] = float(coverage["coverage"])
        schedule_frames.append(schedule)
        decisions, equity, accepted, summary_part, by_head_part = _replay_arm(
            arm=arm,
            candidates=arm_candidates,
            train_deployable=train_deployable,
            params=params,
            market_mode=str(args.market_mode),
        )
        candidates_by_arm[arm] = arm_candidates
        decisions_by_arm[arm] = decisions
        accepted_by_arm[arm] = accepted
        summary_frames.append(summary_part)
        by_head_frames.append(by_head_part)
        arm_candidates.to_parquet(args.output_dir / f"{arm}_candidates.parquet", index=False)
        decisions.to_parquet(args.output_dir / f"{arm}_decisions.parquet", index=False)
        equity.to_parquet(args.output_dir / f"{arm}_equity.parquet", index=False)
        accepted.to_parquet(args.output_dir / f"{arm}_accepted_trades.parquet", index=False)

    summary = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    by_head = pd.concat(by_head_frames, ignore_index=True) if by_head_frames else pd.DataFrame()
    diagnostics = pd.DataFrame(diagnostic_rows)
    schedules = pd.concat(schedule_frames, ignore_index=True) if schedule_frames else pd.DataFrame()
    overlap = _accepted_overlap(accepted_by_arm)
    starvation_attribution = priority_starvation_attribution(
        candidates_by_arm=candidates_by_arm,
        decisions_by_arm=decisions_by_arm,
        baseline_arm=BASELINE_ARM,
    )
    accepted_all = (
        pd.concat([frame for frame in accepted_by_arm.values() if frame is not None], ignore_index=True)
        if accepted_by_arm
        else pd.DataFrame()
    )
    swap_attribution = mstc._threshold_action_utility(accepted_all, BASELINE_ARM)
    selected_spec = model_specs[0] if model_specs else None
    selected_config_for_artifacts = (
        dict(selected_spec.get("config") or {})
        if selected_spec is not None
        else {
            "target_mode": str(args.target_mode),
            "min_rank": float(args.min_rank),
            "frontier_gamma": float(args.frontier_gamma),
            "frontier_bandwidth": float(args.frontier_bandwidth),
            "sl_penalty": float(args.sl_penalty),
            "timeout_penalty": float(args.timeout_penalty),
            "rank_residual_weight": float(args.rank_residual_weight),
            "min_candidates_per_head_timestamp": int(args.min_candidates_per_head_timestamp),
        }
    )
    selected_train_frame = (
        selected_spec["train_frame"] if selected_spec is not None else base_train_frame
    )
    selected_feature_cols = (
        list(selected_spec["feature_cols"]) if selected_spec is not None else list(base_feature_cols)
    )
    selected_target_mode = str(selected_config_for_artifacts.get("target_mode", args.target_mode))
    selected_frontier_utilities = _frontier_candidate_utilities(
        residual,
        state_arm=str(args.state_arm),
        target_mode=selected_target_mode,
        min_rank=float(selected_config_for_artifacts.get("min_rank", args.min_rank)),
        frontier_gamma=float(
            selected_config_for_artifacts.get("frontier_gamma", args.frontier_gamma)
        ),
        frontier_bandwidth=float(
            selected_config_for_artifacts.get("frontier_bandwidth", args.frontier_bandwidth)
        ),
        sl_penalty=float(selected_config_for_artifacts.get("sl_penalty", args.sl_penalty)),
        timeout_penalty=float(
            selected_config_for_artifacts.get("timeout_penalty", args.timeout_penalty)
        ),
        min_candidates_per_head_timestamp=int(
            selected_config_for_artifacts.get(
                "min_candidates_per_head_timestamp",
                args.min_candidates_per_head_timestamp,
            )
        ),
    )
    coverage_feature_cols = list(selected_feature_cols)
    score_coverage = score_feature_coverage(score_state, coverage_feature_cols)

    selected_train_frame.to_parquet(args.output_dir / "head_priority_training_targets.parquet", index=False)
    build_score_head_frame(score_state, candidates, coverage_feature_cols).to_parquet(
        args.output_dir / "head_priority_score_rows.parquet",
        index=False,
    )
    schedules.to_parquet(args.output_dir / "head_priority_learned_schedule.parquet", index=False)
    summary.to_csv(args.output_dir / "head_priority_learning_replay_summary.csv", index=False)
    by_head.to_csv(args.output_dir / "head_priority_learning_by_head.csv", index=False)
    diagnostics.to_csv(args.output_dir / "head_priority_learning_model_diagnostics.csv", index=False)
    overlap.to_csv(args.output_dir / "head_priority_learning_accepted_overlap.csv", index=False)
    starvation_attribution.to_csv(
        args.output_dir / "head_priority_learning_starvation_attribution.csv",
        index=False,
    )
    swap_attribution.to_csv(args.output_dir / "head_priority_learning_accepted_swap_utility.csv", index=False)
    score_coverage.to_csv(args.output_dir / "head_priority_score_feature_coverage.csv", index=False)
    selection_df.to_csv(args.output_dir / "head_priority_config_selection.csv", index=False)
    fold_validation = (
        pd.concat(fold_validation_frames, ignore_index=True)
        if fold_validation_frames
        else pd.DataFrame()
    )
    fold_validation.to_csv(args.output_dir / "head_priority_config_fold_validation.csv", index=False)

    score_ts = pd.to_datetime(candidates["timestamp"], utc=True, errors="coerce")
    static_candidate_parity = static_baseline_candidate_parity(
        candidates,
        candidates_path=candidates_path,
        static_baseline_info=static_baseline_info,
    )
    effective_max_adjustment = float(args.max_adjustment)
    effective_max_priority_multiplier = float(args.max_priority_multiplier)
    effective_max_rank_adjustment = float(args.max_rank_adjustment)
    effective_priority_action = str(args.priority_action)
    if model_specs:
        selected_config = dict(model_specs[0].get("config") or {})
        effective_max_adjustment = float(
            selected_config.get("max_adjustment", effective_max_adjustment)
        )
        effective_max_priority_multiplier = float(
            selected_config.get(
                "max_priority_multiplier",
                effective_max_priority_multiplier,
            )
        )
        effective_max_rank_adjustment = float(
            selected_config.get("max_rank_adjustment", effective_max_rank_adjustment)
        )
        effective_priority_action = str(selected_config.get("priority_action", effective_priority_action))
    manifest = {
        "generated_by": "run_market_state_head_priority_learning",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "learned_market_state_head_priority_modulation_shadow_ablation",
        "contract": {
            "changes_scores_or_ranks": bool(abs(float(effective_max_rank_adjustment)) > 0.0),
            "changes_thresholds": False,
            "changes_position_sizing": False,
            "changes_auction_ordering": True,
            "rank_prior_layer": (
                "pre_filter_head_prior"
                if abs(float(effective_max_rank_adjustment)) > 0.0
                else "disabled"
            ),
            "qfail_active": False,
            "head_health_active": False,
            "market_state_threshold_controller_active": False,
            "operational_status": "shadow_only",
            "execution_enabled": False,
            "production_eligible": False,
            "requires_promotion_gate": True,
            "market_state_encoder_uses_candidate_features": False,
            "priority_adjustment_column": "portfolio_priority_adjustment",
            "priority_multiplier_column": "portfolio_priority_multiplier",
            "rank_adjustment_column": "portfolio_rank_adjustment",
            "priority_action": effective_priority_action,
            "head_specific_selection_rewards": False,
            "head_mix_diagnostics": [
                "action_selected_head_switch_share",
                "action_selected_head_share_l1_shift",
                "action_baseline_selected_head_max_share",
                "action_model_selected_head_max_share",
                "action_baseline_selected_head_entropy",
                "action_model_selected_head_entropy",
            ],
            "static_baseline_source": (
                "materialized_t1_manifest"
                if static_baseline_info is not None
                else "recomputed_from_candidates"
            ),
        },
        "params": {
            "backends": backends,
            "select_config_grid": bool(args.select_config_grid),
            "force_select_failing_config": bool(args.force_select_failing_config),
            "grid_target_modes": str(args.grid_target_modes),
            "grid_min_ranks": str(args.grid_min_ranks),
            "grid_frontier_gammas": str(args.grid_frontier_gammas),
            "grid_frontier_bandwidths": str(args.grid_frontier_bandwidths),
            "grid_sl_penalties": str(args.grid_sl_penalties),
            "grid_timeout_penalties": str(args.grid_timeout_penalties),
            "grid_rank_residual_weights": str(args.grid_rank_residual_weights),
            "grid_max_adjustments": str(args.grid_max_adjustments),
            "grid_max_priority_multipliers": str(args.grid_max_priority_multipliers),
            "grid_max_rank_adjustments": str(args.grid_max_rank_adjustments),
            "grid_priority_actions": str(args.grid_priority_actions),
            "state_arm": str(args.state_arm),
            "min_rank": float(args.min_rank),
            "frontier_gamma": float(args.frontier_gamma),
            "frontier_bandwidth": float(args.frontier_bandwidth),
            "sl_penalty": float(args.sl_penalty),
            "timeout_penalty": float(args.timeout_penalty),
            "rank_residual_weight": float(args.rank_residual_weight),
            "min_candidates_per_head_timestamp": int(args.min_candidates_per_head_timestamp),
            "target_mode": str(args.target_mode),
            "target_clip": float(args.target_clip),
            "max_adjustment": float(effective_max_adjustment),
            "default_max_adjustment": float(args.max_adjustment),
            "max_priority_multiplier": float(effective_max_priority_multiplier),
            "default_max_priority_multiplier": float(args.max_priority_multiplier),
            "max_rank_adjustment": float(effective_max_rank_adjustment),
            "default_max_rank_adjustment": float(args.max_rank_adjustment),
            "priority_action": effective_priority_action,
            "validation_frac": float(args.validation_frac),
            "validation_mode": str(args.validation_mode),
            "selection_gate_mode": str(args.selection_gate_mode),
            "selection_min_accepted_jaccard": float(args.selection_min_accepted_jaccard),
            "selection_min_trade_retention": float(args.selection_min_trade_retention),
            "selection_max_full_sl_delta": float(args.selection_max_full_sl_delta),
            "selection_max_timeout_delta": float(args.selection_max_timeout_delta),
            "strict_replay_risk_gates": bool(args.strict_replay_risk_gates),
            "selection_replay_top_n": int(args.selection_replay_top_n),
            "use_all_state_heads": bool(args.use_all_state_heads),
            "state_head_statuses": sorted(allowed_statuses),
            "seed": int(args.seed),
        },
        "inputs": {
            "walkforward_dir": str(args.walkforward_dir),
            "score_dir": str(args.score_dir),
            "residual_ledger": str(residual_path),
            "residual_ledger_sha256": _sha256(residual_path),
            "train_state_panel": str(train_state_path),
            "train_state_panel_sha256": _sha256(train_state_path),
            "score_state_panel": str(score_state_path),
            "score_state_panel_sha256": _sha256(score_state_path),
            "candidates": str(candidates_path),
            "candidates_sha256": _sha256(candidates_path),
            "train_deployable_candidates": str(args.train_deployable_candidates),
            "train_deployable_candidates_sha256": _sha256(args.train_deployable_candidates),
            "train_deployable_rank_contract": train_deployable_contract,
            "policy_manifest": str(args.policy_manifest),
            "policy_manifest_sha256": _sha256(args.policy_manifest),
            "policy_manifest_run_id": policy_payload.get("run_id"),
            "static_baseline_manifest": (
                str(args.static_baseline_manifest)
                if args.static_baseline_manifest is not None
                else None
            ),
            "static_baseline_manifest_sha256": _sha256(args.static_baseline_manifest),
            "activation_registry": (
                str(activation_registry_path)
                if activation_registry_path is not None
                else None
            ),
            "activation_registry_sha256": _sha256(activation_registry_path),
            "walkforward_manifest": _load_json(args.walkforward_dir / "manifest.json").get("generated_by"),
            "score_manifest": _load_json(args.score_dir / "manifest.json").get("generated_by"),
        },
        "state_head_activation_filter": activation_registry_report,
        "static_baseline": static_baseline_info,
        "static_baseline_candidate_parity": static_candidate_parity,
        "candidate_universe": {
            "rows": int(len(candidates)),
            "timestamp_count": int(score_ts.nunique()),
            "timestamp_min": score_ts.min(),
            "timestamp_max": score_ts.max(),
            "heads": sorted(candidates["head"].dropna().astype(str).unique()),
        },
        "training": {
            "rows": int(len(selected_train_frame)),
            "timestamp_count": int(pd.to_datetime(selected_train_frame["timestamp"], utc=True, errors="coerce").nunique()),
            "feature_count": int(len(selected_feature_cols)),
            "features": selected_feature_cols,
            "frontier_action_validation_rows": int(len(selected_frontier_utilities)),
            "target": (
                "timestamp_centered_threshold_admission_residual_utility"
                if selected_target_mode == "threshold_admission_mean"
                else "timestamp_centered_rank_residual_frontier_utility"
                if selected_target_mode == "rank_residual_frontier"
                else "timestamp_centered_frontier_residual_utility"
            ),
            "target_mode": selected_target_mode,
            "rank_residual_weight": float(
                selected_config_for_artifacts.get("rank_residual_weight", args.rank_residual_weight)
            ),
        },
        "selection": {
            "enabled": bool(args.select_config_grid),
            "selected": diagnostics.iloc[0].to_dict() if bool(args.select_config_grid) and not diagnostics.empty else None,
            "gate_passed": (
                bool(diagnostics.iloc[0].get("selection_gate_passed"))
                if bool(args.select_config_grid) and not diagnostics.empty
                else False
                if bool(args.select_config_grid)
                else None
            ),
            "candidate_count": int(len(selection_df)) if bool(args.select_config_grid) else 0,
            "replay_aware_selection_top_n": int(args.selection_replay_top_n),
            "selection_gate_mode": str(args.selection_gate_mode),
            "objective": "head-agnostic fit quality plus incremental-over-head-only validation; when frontier validation is available, adds auction-frontier action score. If selection_replay_top_n>0, a top-N shortlist is replayed through the fixed T1 portfolio universe and final ranking uses replay_selection_score from net PnL delta, accepted-swap utility, replacement PnL, full-SL/timeout penalties and accepted overlap. No strategy/head receives a hard-coded reward or penalty.",
            "gate": "trailing: validation_rows>=10 and validation_spearman>0 and validation_directional_accuracy>=0.50; fold_aware defensive: fold_count>=2 and fold_validation_rows>=20 and fold_mean_spearman>0 and fold_positive_spearman_share>=0.50 and fold_mean_directional_accuracy>=0.50 and fold_directional_ge_50_share>=0.50 and incremental objective/spearman/mae beat head-only; if frontier action validation is present, require non-negative mean utility delta, positive action share>=0.50 and full-SL delta<=0.02; trailing validation_rows>=10 and validation_spearman>-0.20 and validation_directional_accuracy>=0.50. fold_aware opportunity uses the same recurrent fold and incremental requirements but does not require the small trailing validation split to pass, and permits full-SL delta<=0.03 in frontier action validation. If replay-aware selection is enabled, defensive candidates must improve replay net PnL, retain >=90% of trades, keep full-SL and timeout rates non-worse, keep accepted Jaccard >=90%, move the accepted set, and have entrants beat removed trades. Opportunity candidates use the same accepted-swap gates but allow small replay full-SL/timeout drift when net/action/replacement utility is positive.",
            "results": str(final_output_dir / "head_priority_config_selection.csv"),
            "fold_validation": str(final_output_dir / "head_priority_config_fold_validation.csv"),
        },
        "summary": summary.to_dict("records"),
        "by_head": by_head.to_dict("records"),
        "diagnostics": diagnostics.to_dict("records"),
        "accepted_swap_utility": swap_attribution.to_dict("records"),
        "priority_starvation_attribution": starvation_attribution.to_dict("records"),
        "outputs": {
            "manifest": str(final_output_dir / "manifest.json"),
            "report": str(final_output_dir / "market_state_head_priority_learning_report.md"),
            "training_targets": str(final_output_dir / "head_priority_training_targets.parquet"),
            "score_rows": str(final_output_dir / "head_priority_score_rows.parquet"),
            "schedule": str(final_output_dir / "head_priority_learned_schedule.parquet"),
            "summary": str(final_output_dir / "head_priority_learning_replay_summary.csv"),
            "by_head": str(final_output_dir / "head_priority_learning_by_head.csv"),
            "model_diagnostics": str(final_output_dir / "head_priority_learning_model_diagnostics.csv"),
            "accepted_overlap": str(final_output_dir / "head_priority_learning_accepted_overlap.csv"),
            "starvation_attribution": str(
                final_output_dir / "head_priority_learning_starvation_attribution.csv"
            ),
            "accepted_swap_utility": str(final_output_dir / "head_priority_learning_accepted_swap_utility.csv"),
            "score_feature_coverage": str(final_output_dir / "head_priority_score_feature_coverage.csv"),
            "config_selection": str(final_output_dir / "head_priority_config_selection.csv"),
            "config_fold_validation": str(final_output_dir / "head_priority_config_fold_validation.csv"),
        },
        "output_staging": output_staging_report,
    }
    report = _render_report(
        manifest=manifest,
        summary=summary,
        by_head=by_head,
        diagnostics=diagnostics,
        overlap=overlap,
        swap_attribution=swap_attribution,
        score_coverage=score_coverage,
        starvation=starvation_attribution,
    )
    (args.output_dir / "manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "market_state_head_priority_learning_report.md").write_text(report, encoding="utf-8")
    _publish_staged_output_dir(args.output_dir, final_output_dir)
    published["done"] = True
    print(json.dumps(_json_safe({"output_dir": str(final_output_dir), "summary": summary.to_dict("records"), "diagnostics": diagnostics.to_dict("records")}), indent=2))


if __name__ == "__main__":
    main()
