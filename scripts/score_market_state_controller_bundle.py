#!/usr/bin/env python3
"""Score candidates with a frozen market-state controller bundle.

This is the deployment-side companion to
``materialize_market_state_controller_bundle.py``.  It loads the frozen bundle,
builds the current timestamp-level market-state source from feature-store
aggregates, applies the persisted observed-axis, forecast-head and latent-state
transformers, predicts strategy response, and writes a deterministic threshold
schedule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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


DEFAULT_BUNDLE = Path(
    "data_perp/reports/market_state_controller_bundle_t1_lgbm_maturity_noop_20260626"
    "/market_state_controller_bundle.joblib"
)
DEFAULT_OUTPUT_DIR = Path("data_perp/reports/market_state_controller_bundle_score_t1_lgbm_maturity_noop_20260626")
NOOP_CONTROLLER_ARM = "S0_rejected_controller_noop"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(float(value)) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _file_sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _output_sha256(outputs: dict[str, str]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name, path_raw in outputs.items():
        digest = _file_sha256(Path(path_raw))
        if digest is not None:
            hashes[str(name)] = digest
    return hashes


def _accepted_trade_delta_report(
    baseline_accepted: pd.DataFrame,
    shadow_accepted: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    key_cols = [
        col
        for col in ("timestamp", "symbol", "strategy_id", "head", "side")
        if col in baseline_accepted.columns and col in shadow_accepted.columns
    ]
    if not key_cols:
        return pd.DataFrame(), {
            "available": False,
            "reason": "missing_common_decision_key_columns",
        }

    def _with_key(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        out["__decision_key__"] = out[key_cols].astype(str).agg("|".join, axis=1)
        return out

    base = _with_key(baseline_accepted)
    shadow = _with_key(shadow_accepted)
    base_keys = set(base["__decision_key__"])
    shadow_keys = set(shadow["__decision_key__"])
    removed = base.loc[~base["__decision_key__"].isin(shadow_keys)].copy()
    added = shadow.loc[~shadow["__decision_key__"].isin(base_keys)].copy()
    common = shadow.loc[shadow["__decision_key__"].isin(base_keys)].copy()
    removed["delta_action"] = "removed_by_shadow_no_backfill"
    added["delta_action"] = "added_by_shadow_no_backfill"
    common["delta_action"] = "common_accepted"
    delta = pd.concat([removed, added, common], ignore_index=True, sort=False)
    pnl_col = "net_pnl" if "net_pnl" in delta.columns else "net_return" if "net_return" in delta.columns else None

    def _pnl_sum(frame: pd.DataFrame) -> float:
        if pnl_col is None or pnl_col not in frame.columns:
            return 0.0
        return float(pd.to_numeric(frame[pnl_col], errors="coerce").fillna(0.0).sum())

    removed_pnl = _pnl_sum(removed)
    added_pnl = _pnl_sum(added)
    baseline_total_pnl = _pnl_sum(base)
    shadow_total_pnl = _pnl_sum(shadow)
    if pnl_col is not None and pnl_col in base.columns and pnl_col in shadow.columns:
        common_base = base.loc[base["__decision_key__"].isin(shadow_keys)]
        common_shadow = shadow.loc[shadow["__decision_key__"].isin(base_keys)]
        common_baseline_pnl = _pnl_sum(common_base)
        common_shadow_pnl = _pnl_sum(common_shadow)
    else:
        common_baseline_pnl = 0.0
        common_shadow_pnl = 0.0
    winner_sacrificed = max(0.0, removed_pnl)
    loss_avoided = max(0.0, -removed_pnl)
    action_only_delta = float(added_pnl - removed_pnl)
    total_delta = float(shadow_total_pnl - baseline_total_pnl)
    common_delta = float(common_shadow_pnl - common_baseline_pnl)
    summary = {
        "available": True,
        "key_columns": key_cols,
        "baseline_trade_count": int(len(base)),
        "shadow_trade_count": int(len(shadow)),
        "removed_trade_count": int(len(removed)),
        "added_trade_count": int(len(added)),
        "common_trade_count": int(len(common)),
        "shadow_subset_of_baseline": bool(shadow_keys.issubset(base_keys)),
        "baseline_net_pnl": float(baseline_total_pnl),
        "shadow_net_pnl": float(shadow_total_pnl),
        "total_net_pnl_delta": total_delta,
        "full_path_replay_net_pnl_delta": total_delta,
        "removed_net_pnl": float(removed_pnl),
        "added_net_pnl": float(added_pnl),
        "common_baseline_net_pnl": float(common_baseline_pnl),
        "common_shadow_net_pnl": float(common_shadow_pnl),
        "common_net_pnl_delta": common_delta,
        "path_dependent_common_trade_net_pnl_delta": common_delta,
        "action_only_fixed_common_size_net_pnl_delta": action_only_delta,
        "removed_loss_avoided": float(loss_avoided),
        "removed_winner_pnl_sacrificed": float(winner_sacrificed),
        "accepted_delta_defensive_success": float(loss_avoided - winner_sacrificed),
    }
    return delta, summary


def _direct_threshold_only_overlay(
    baseline_accepted: pd.DataFrame,
    proposed_schedule: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Apply proposed state thresholds directly to baseline accepted trades.

    The normal no-backfill shadow replay still runs the auction again on the
    baseline-accepted candidate subset. That can change the final accepted set
    through capacity/path mechanics. This counterfactual is intentionally
    replay-free: it starts from the baseline accepted trades and removes only
    rows whose reached rank is below the proposed state threshold. It therefore
    measures the controller's direct threshold action without accidental
    suppression from replay ordering, sizing, or capacity side effects.
    """

    if baseline_accepted.empty:
        empty = baseline_accepted.copy()
        return empty, pd.DataFrame(), {
            "available": True,
            "baseline_trade_count": 0,
            "shadow_trade_count": 0,
            "removed_trade_count": 0,
            "added_trade_count": 0,
            "direct_threshold_only": True,
            "no_path_or_capacity_replay": True,
        }
    if proposed_schedule.empty:
        kept = baseline_accepted.copy()
        delta, summary = _accepted_trade_delta_report(baseline_accepted, kept)
        summary.update(
            {
                "direct_threshold_only": True,
                "no_path_or_capacity_replay": True,
                "rank_column": None,
                "threshold_column": None,
                "reason": "empty_proposed_schedule",
            }
        )
        return kept, delta, summary

    rank_col = next(
        (
            col
            for col in (
                "effective_rank_score",
                "normalized_rank_score",
                "policy_rank_pct",
                "strategy_rank_pct",
                "rank_pct",
            )
            if col in baseline_accepted.columns
        ),
        None,
    )
    if rank_col is None:
        raise KeyError(
            "baseline accepted trades are missing a rank column required for "
            "direct threshold-only overlay"
        )

    sched_cols = [
        col
        for col in (
            "timestamp",
            "strategy_id",
            "head",
            "base_threshold",
            "state_threshold",
            "raw_state_threshold",
            "controller_reason",
            "risk_severity",
            "force_base_threshold",
        )
        if col in proposed_schedule.columns
    ]
    sched = proposed_schedule.loc[:, sched_cols].copy()
    sched["timestamp"] = pd.to_datetime(sched["timestamp"], utc=True, errors="coerce")
    join_cols = ["timestamp", "strategy_id"]
    if "head" in baseline_accepted.columns and "head" in sched.columns:
        join_cols.append("head")
    sched = sched.drop_duplicates(subset=join_cols, keep="last")

    work = baseline_accepted.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    work = work.merge(
        sched,
        on=join_cols,
        how="left",
        suffixes=("", "_schedule"),
        validate="many_to_one",
    )
    rank = pd.to_numeric(work[rank_col], errors="coerce")
    base_dynamic = (
        pd.to_numeric(work.get("dynamic_threshold"), errors="coerce")
        if "dynamic_threshold" in work.columns
        else pd.to_numeric(work.get("base_threshold"), errors="coerce")
    )
    state_threshold = pd.to_numeric(work.get("state_threshold"), errors="coerce")
    direct_threshold = np.maximum(
        base_dynamic.fillna(0.0).to_numpy(dtype=float),
        state_threshold.fillna(base_dynamic).fillna(0.0).to_numpy(dtype=float),
    )
    direct_threshold = np.clip(direct_threshold, 0.0, 1.01)
    work["direct_threshold_only_rank_column"] = rank_col
    work["direct_state_threshold"] = state_threshold
    work["direct_effective_threshold"] = direct_threshold
    work["direct_threshold_rank_margin"] = rank - direct_threshold
    missing_schedule = state_threshold.isna()
    removed = rank.notna() & ~missing_schedule & (rank < direct_threshold)
    work["direct_threshold_removed"] = removed.to_numpy(dtype=bool)
    work["direct_threshold_action"] = np.where(
        work["direct_threshold_removed"],
        "removed_by_direct_threshold_only",
        "kept_by_direct_threshold_only",
    )
    kept = work.loc[~work["direct_threshold_removed"]].copy()
    delta, summary = _accepted_trade_delta_report(work, kept)
    summary.update(
        {
            "direct_threshold_only": True,
            "no_path_or_capacity_replay": True,
            "rank_column": rank_col,
            "threshold_column": "direct_effective_threshold",
            "missing_schedule_count": int(missing_schedule.sum()),
            "direct_state_threshold_count": int(state_threshold.notna().sum()),
            "direct_threshold_removed_count": int(removed.sum()),
            "direct_threshold_kept_count": int((~removed).sum()),
        }
    )
    return kept, delta, summary


def _locked_accepted_overlay_from_direct(
    *,
    arm: str,
    direct_accepted: pd.DataFrame,
    direct_delta: pd.DataFrame,
    direct_summary: dict[str, Any],
) -> tuple[str, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Convert the direct threshold-only overlay into a locked action artifact.

    This is the production-safe no-backfill action semantics: start from the
    already accepted baseline trades, remove only rows directly below the
    proposed state threshold, and do not rerun auction/path/capacity mechanics.
    """

    locked_arm = f"{arm}__shadow_locked_accepted_overlay"
    locked_accepted = direct_accepted.copy()
    if not locked_accepted.empty:
        locked_accepted["arm"] = locked_arm
    locked_delta = direct_delta.copy()
    locked_summary = dict(direct_summary)
    locked_summary.update(
        {
            "locked_accepted_overlay": True,
            "direct_threshold_only": True,
            "no_path_or_capacity_replay": True,
            "no_replacement_candidates": True,
            "common_trade_sizing_locked": True,
            "auction_ordering_locked": True,
        }
    )
    return locked_arm, locked_accepted, locked_delta, locked_summary


def _controller_execution_enabled(bundle: dict[str, Any]) -> bool:
    params = dict(bundle.get("controller_params") or {})
    return bool(bundle.get("controller_execution_enabled", params.get("execution_enabled", True)))


def _shadow_controller_only(bundle: dict[str, Any]) -> bool:
    params = dict(bundle.get("controller_params") or {})
    return bool(bundle.get("shadow_controller_only", params.get("shadow_controller_only", False)))


def _requires_state_train_references(bundle: dict[str, Any]) -> bool:
    return _controller_execution_enabled(bundle) or (
        _shadow_controller_only(bundle)
        and bool(list(bundle.get("state_feature_columns") or []))
    )


def _validate_candidate_state_fallback_execution_contract(
    bundle: dict[str, Any],
    *,
    allow_candidate_state_fallback: bool,
) -> None:
    if bool(allow_candidate_state_fallback) and _requires_state_train_references(bundle):
        raise RuntimeError(
            "Refusing to score an executable or shadow market-state controller with "
            "allow_candidate_state_fallback=true. Candidate-population fallback "
            "is debug-only and is not live-equivalent for a threshold "
            "controller. Score with feature-store market aggregates or use a "
            "rejected/no-op audit bundle."
        )


def _validate_observed_axis_encoder_contract(
    encoder: Any,
    *,
    require_train_references: bool,
) -> dict[str, Any]:
    """Validate the frozen observed-axis artifact used by deployment scoring.

    Rejected/noop audit bundles may carry only a skeletal encoder because they
    are not executable. Any executable bundle must carry train-only robust
    references and the low-coverage fail-closed channel; otherwise live scoring
    can silently drift away from the walk-forward contract.
    """

    if not isinstance(encoder, dict):
        raise TypeError("observed_axis_encoder must be a dict")
    mode = str(encoder.get("mode", ""))
    if mode != "observed_axis_robust_z_v1":
        raise ValueError(f"observed_axis_encoder has unsupported mode={mode!r}")
    report = {
        "mode": mode,
        "require_train_references": bool(require_train_references),
        "minimum_input_coverage": encoder.get("minimum_input_coverage"),
        "column_ref_count": int(len(dict(encoder.get("column_refs") or {}))),
        "axis_count": int(len(dict(encoder.get("axes") or {}))),
        "reliability_column_count": int(
            len(list(dict(encoder.get("reliability") or {}).get("columns") or []))
        ),
    }
    if not require_train_references:
        return report

    min_cov = float(encoder.get("minimum_input_coverage", np.nan))
    if not np.isfinite(min_cov) or min_cov < 0.0 or min_cov > 1.0:
        raise ValueError("observed_axis_encoder.minimum_input_coverage must be finite in [0, 1]")
    column_refs = dict(encoder.get("column_refs") or {})
    if not column_refs:
        raise ValueError("executable observed_axis_encoder is missing train column_refs")
    for col, ref_raw in column_refs.items():
        ref = dict(ref_raw or {})
        med = float(ref.get("median", np.nan))
        scale = float(ref.get("scale", np.nan))
        q05 = float(ref.get("q05", np.nan))
        q95 = float(ref.get("q95", np.nan))
        if not np.isfinite(med) or not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"observed_axis_encoder column_ref {col!r} has invalid median/scale")
        if not np.isfinite(q05) or not np.isfinite(q95):
            raise ValueError(f"observed_axis_encoder column_ref {col!r} has invalid q05/q95")

    reliability = dict(encoder.get("reliability") or {})
    if reliability.get("mode") != "observed_reliability_train_reference_v1":
        raise ValueError("executable observed_axis_encoder has invalid reliability reference mode")
    reliability_cols = [str(c) for c in list(reliability.get("columns") or [])]
    if not reliability_cols:
        raise ValueError("executable observed_axis_encoder is missing reliability columns")
    missing_ref_cols = sorted(set(reliability_cols) - set(column_refs))
    if missing_ref_cols:
        raise ValueError(
            "observed_axis_encoder reliability columns missing column_refs: "
            + ", ".join(missing_ref_cols[:12])
        )
    source_validation = dict(encoder.get("source_validation") or {})
    train_validation = source_validation.get("train")
    if not isinstance(train_validation, dict) or not train_validation:
        raise ValueError("executable observed_axis_encoder is missing source_validation.train")
    axis_sources = dict(encoder.get("axis_sources") or {})
    for required_source in (
        "state_input_coverage",
        "state_uncertainty",
        "state_low_input_coverage",
    ):
        if required_source not in axis_sources:
            raise ValueError(
                f"executable observed_axis_encoder missing axis source {required_source!r}"
            )
    return report


def _validate_feature_store_tail_reference_contract(
    bundle: dict[str, Any],
    *,
    require_train_references: bool,
) -> dict[str, Any]:
    feature_store_cols = [str(c) for c in list(bundle.get("feature_store_columns") or [])]
    refs = _bundle_feature_store_tail_reference_quantiles(bundle)
    report = {
        "required": bool(require_train_references and feature_store_cols),
        "feature_store_column_count": int(len(feature_store_cols)),
        "tail_reference_quantile_count": int(len(refs)),
        "tail_reference_source": "bundle_train_reference" if refs else "missing",
    }
    if not require_train_references or not feature_store_cols:
        return report
    if not refs:
        raise ValueError(
            "executable bundle with feature_store_columns is missing "
            "feature_store_tail_reference_quantiles"
        )
    bad: list[str] = []
    for col, ref_raw in refs.items():
        ref = dict(ref_raw or {})
        q10 = float(ref.get("q10", np.nan))
        q90 = float(ref.get("q90", np.nan))
        if not np.isfinite(q10) or not np.isfinite(q90):
            bad.append(str(col))
    if bad:
        raise ValueError(
            "feature_store_tail_reference_quantiles contain invalid q10/q90 for: "
            + ", ".join(bad[:12])
        )
    return report


def _bundle_feature_store_tail_reference_quantiles(bundle: dict[str, Any]) -> dict[str, Any]:
    refs = bundle.get("feature_store_tail_reference_quantiles")
    if refs is None:
        refs = (
            dict(bundle.get("market_state_feature_contract") or {})
            .get("feature_store_tail_reference", {})
            .get("quantiles")
        )
    return dict(refs or {})


def _validate_state_activation_filter_contract(bundle: dict[str, Any]) -> dict[str, Any]:
    """Reject stale scored bundles that still consume disabled state heads."""

    feature_contract = dict(bundle.get("market_state_feature_contract") or {})
    activation_filter = bundle.get("state_activation_filter")
    if activation_filter is None:
        activation_filter = feature_contract.get("state_activation_filter")
    if activation_filter is None:
        return {
            "available": False,
            "enforced": False,
            "reason": "state_activation_filter_missing_legacy_bundle",
        }
    if not isinstance(activation_filter, dict):
        raise ValueError("state_activation_filter must be a dict when present")

    enforced = bool(activation_filter.get("enforced", False))
    active_cols = set(map(str, activation_filter.get("active_state_feature_columns") or []))
    dropped_cols = set(map(str, activation_filter.get("dropped_state_feature_columns") or []))
    bundle_state_cols = set(map(str, bundle.get("state_feature_columns") or []))
    bundle_response_cols = set(map(str, bundle.get("response_feature_columns") or []))

    source_schema = dict(feature_contract.get("source_schema") or {})
    contract_state_cols = set(map(str, source_schema.get("state_feature_columns") or []))
    contract_response_cols = set(map(str, source_schema.get("response_feature_columns") or []))

    report = {
        "available": True,
        "enforced": enforced,
        "reason": activation_filter.get("reason"),
        "active_state_feature_count": int(len(active_cols)),
        "dropped_state_feature_count": int(len(dropped_cols)),
        "bundle_state_feature_count": int(len(bundle_state_cols)),
        "bundle_response_feature_count": int(len(bundle_response_cols)),
        "contract_state_feature_count": int(len(contract_state_cols)),
        "contract_response_feature_count": int(len(contract_response_cols)),
    }
    if not enforced:
        return report

    allowed_empty_reasons = {
        "selected_controller_rejected_noop",
        "activation_registry_unavailable_fail_closed",
        "activation_registry_active_candidate_filter",
    }
    reason = str(activation_filter.get("reason") or "")
    failures: list[str] = []
    if not active_cols and reason not in allowed_empty_reasons:
        failures.append("state_activation_filter enforced with no active state features")

    outside_active = sorted(bundle_state_cols.difference(active_cols))
    if outside_active:
        failures.append(
            "bundle state_feature_columns outside activation filter: "
            + ", ".join(outside_active[:12])
        )
    contract_outside_active = sorted(contract_state_cols.difference(active_cols))
    if contract_outside_active:
        failures.append(
            "feature contract state_feature_columns outside activation filter: "
            + ", ".join(contract_outside_active[:12])
        )

    leaked_bundle = sorted(dropped_cols.intersection(bundle_state_cols.union(bundle_response_cols)))
    if leaked_bundle:
        failures.append(
            "bundle contains dropped activation-registry state features: "
            + ", ".join(leaked_bundle[:12])
        )
    leaked_contract = sorted(dropped_cols.intersection(contract_state_cols.union(contract_response_cols)))
    if leaked_contract:
        failures.append(
            "feature contract contains dropped activation-registry state features: "
            + ", ".join(leaked_contract[:12])
        )
    if failures:
        raise ValueError("; ".join(failures))
    return report


def _load_bundle(path: Path, *, allow_selected_arm_default_bundle: bool = False) -> dict[str, Any]:
    import joblib

    bundle = joblib.load(path)
    if not isinstance(bundle, dict):
        raise TypeError(f"Bundle at {path} did not deserialize to a dict")
    selected_controller = bundle.get("selected_controller")
    if isinstance(selected_controller, dict):
        default_used = bool(selected_controller.get("selected_arm_default_used", False))
        noop_used = bool(selected_controller.get("selected_arm_noop_used", False))
        selected_arm = selected_controller.get("selected_arm", bundle.get("selected_arm"))
        no_op_bundle = (
            noop_used
            and bundle.get("selected_arm") == NOOP_CONTROLLER_ARM
            and bundle.get("controller_execution_enabled") is False
        )
        if default_used and not allow_selected_arm_default_bundle:
            raise RuntimeError(
                f"Bundle {path} was materialized with selected_arm_default_used=true. "
                "Refusing to score a debug/default controller without "
                "--allow-selected-arm-default-bundle."
            )
        if (
            selected_arm is None or str(selected_arm).strip() == ""
        ) and not allow_selected_arm_default_bundle and not no_op_bundle:
            reason = selected_controller.get("reason") or "missing selected arm"
            raise RuntimeError(f"Bundle {path} has no promoted selected arm ({reason})")
    state_level = str(bundle.get("state_spec", {}).get("state_level", ""))
    if state_level not in {"observed", "forecast", "latent"}:
        raise NotImplementedError(f"Unsupported bundle state_level={state_level!r}")
    required = [
        "models",
        "response_feature_columns",
        "state_feature_columns",
        "feature_store_columns",
        "observed_axis_encoder",
    ]
    if state_level in {"forecast", "latent"}:
        required.append("forecast_artifact")
    if state_level == "latent":
        required.append("latent_artifact")
    missing = [key for key in required if key not in bundle]
    if missing:
        raise KeyError(f"Bundle missing required keys: {missing}")
    bundle["observed_axis_encoder_validation"] = _validate_observed_axis_encoder_contract(
        bundle["observed_axis_encoder"],
        require_train_references=_requires_state_train_references(bundle),
    )
    bundle["feature_store_tail_reference_validation"] = _validate_feature_store_tail_reference_contract(
        bundle,
        require_train_references=_requires_state_train_references(bundle),
    )
    bundle["state_activation_filter_validation"] = _validate_state_activation_filter_contract(bundle)
    return bundle


def _bundle_enabled_heads(value: Any) -> set[str] | None:
    if isinstance(value, str):
        return None if value == "all_active_heads" else mstc._parse_disabled_heads(value)
    if value is None:
        return None
    return set(value)


def score_candidates(
    *,
    bundle: dict[str, Any],
    candidates: pd.DataFrame,
    feature_store_dir: Path,
    feature_store_symbol_cap: int,
    allow_candidate_state_fallback: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    _validate_candidate_state_fallback_execution_contract(
        bundle,
        allow_candidate_state_fallback=bool(allow_candidate_state_fallback),
    )
    observed_axis_encoder_validation = _validate_observed_axis_encoder_contract(
        bundle.get("observed_axis_encoder"),
        require_train_references=_requires_state_train_references(bundle),
    )
    feature_store_tail_reference_validation = _validate_feature_store_tail_reference_contract(
        bundle,
        require_train_references=_requires_state_train_references(bundle),
    )
    if candidates.empty:
        state_feature_columns = list(bundle.get("state_feature_columns", []))
        response_model_names = [
            str(name)
            for name in dict(bundle.get("models") or {}).keys()
            if str(name) != "curves"
        ]
        state = pd.DataFrame(columns=["timestamp", *state_feature_columns])
        predictions = pd.DataFrame(columns=["timestamp", "strategy_id", *[f"pred_{name}" for name in response_model_names]])
        schedule = pd.DataFrame(
            columns=[
                "timestamp",
                "strategy_id",
                "base_threshold",
                "state_threshold",
                "raw_state_threshold",
                "threshold_action_enabled",
                "force_base_threshold",
                "controller_reason",
            ]
        )
        score_report = {
            "empty_eval_candidates": True,
            "empty_eval_reason": "no_candidate_rows_after_rank_contract_and_disabled_heads",
            "observed_axis_encoder_validation": observed_axis_encoder_validation,
            "feature_store_tail_reference_validation": feature_store_tail_reference_validation,
            "state_frame_validation": {
                "overall_passed": True,
                "context": "score_empty",
                "row_count": 0,
            },
            "state_join_validation": {
                "overall_passed": True,
                "context": "score_empty",
                "row_count": 0,
            },
            "state_source_report": {
                "source": "empty_eval_candidates",
                "row_count": 0,
            },
            "feature_store_report": {
                "row_count": 0,
                "tail_reference_role": "not_applicable_empty_eval_candidates",
            },
            "shadow_controller_only": _shadow_controller_only(bundle),
        }
        return (
            candidates.copy(),
            predictions,
            schedule,
            state,
            score_report,
            schedule.iloc[0:0].copy(),
        )
    candidate_feature_cols = [
        c for c in bundle.get("candidate_feature_columns", []) if c in candidates.columns
    ]
    candidate_agg = mstc._timestamp_aggregates(candidates, candidate_feature_cols)
    feature_store_cols = list(bundle.get("feature_store_columns", []))
    feature_store_eligible_symbols_raw = bundle.get("feature_store_eligible_symbols")
    feature_store_eligible_symbols = (
        [str(symbol) for symbol in list(feature_store_eligible_symbols_raw or [])]
        if feature_store_eligible_symbols_raw is not None
        else None
    )
    fs, fs_report = mstc._feature_store_timestamp_aggregates(
        feature_store_dir,
        candidate_agg["timestamp"],
        feature_store_cols,
        symbol_cap=int(feature_store_symbol_cap),
        tail_reference_quantiles=_bundle_feature_store_tail_reference_quantiles(bundle),
        eligible_symbols=feature_store_eligible_symbols,
    )
    if feature_store_cols:
        fs_report["tail_reference_role"] = "transformed_with_bundle_training_reference"
    state_source, state_source_report = mstc._state_source_aggregate_frame(
        candidate_agg,
        fs,
        allow_candidate_fallback=bool(allow_candidate_state_fallback),
    )
    state = mstc.transform_observed_axes(state_source, bundle["observed_axis_encoder"])
    state_level = str(bundle.get("state_spec", {}).get("state_level", "observed"))
    if state_level in {"forecast", "latent"}:
        state = mstc.transform_forecast_state_heads(
            state,
            bundle["forecast_artifact"],
            agg=state_source,
        )
    if state_level == "latent":
        state = mstc.transform_latent_state_probs(state, bundle["latent_artifact"])
    state_frame_validation = mstc.state_frame_contract_report(state, context=f"score_{state_level}")
    eval_frame = mstc.build_response_frame(candidates, state)
    state_join_validation = mstc.joined_state_invariance_report(
        eval_frame,
        list(bundle.get("state_feature_columns", [])),
        context=f"score_{state_level}",
    )
    predictions = mstc.predict_response(
        bundle["models"],
        eval_frame,
        list(bundle["response_feature_columns"]),
        list(bundle["state_feature_columns"]),
    )
    params = dict(bundle.get("controller_params", {}))
    if not bool(bundle.get("controller_execution_enabled", params.get("execution_enabled", True))):
        enabled_heads = set()
    else:
        enabled_heads = _bundle_enabled_heads(bundle.get("controller_enabled_heads"))
    schedule = mstc.threshold_schedule(
        eval_frame,
        predictions,
        bundle["models"]["curves"],
        delta_max=float(params.get("threshold_delta_max", 0.10)),
        max_down_step=float(params.get("max_threshold_up_step", 0.03)),
        relax_alpha=float(params.get("threshold_relax_alpha", 0.25)),
        controller_mode=str(params.get("controller_mode", "rank_grid")),
        min_lcb_utility=float(params.get("controller_min_lcb_utility", 0.0)),
        use_timeout_cap=bool(params.get("use_timeout_cap", False)),
        min_action_edge=float(params.get("controller_min_action_edge", 0.0)),
        winner_sacrifice_multiplier=float(params.get("controller_winner_sacrifice_multiplier", 1.0)),
        min_removed_full_sl=float(params.get("controller_min_removed_full_sl", 0.0)),
        max_removed_timeout=float(params.get("controller_max_removed_timeout", 1.0)),
        enabled_heads=enabled_heads,
        min_prediction_coverage=float(params.get("controller_min_prediction_coverage", 0.80)),
        min_usable_candidates=int(params.get("controller_min_usable_candidates", 1)),
        min_frontier_candidates=int(params.get("controller_min_frontier_candidates", 1)),
        max_state_ood_score=params.get("controller_max_state_ood_score"),
    )
    proposed_schedule = pd.DataFrame()
    if (
        not bool(bundle.get("controller_execution_enabled", params.get("execution_enabled", True)))
        and _shadow_controller_only(bundle)
        and bool(list(bundle.get("state_feature_columns", [])))
    ):
        proposed_schedule = mstc.threshold_schedule(
            eval_frame,
            predictions,
            bundle["models"]["curves"],
            delta_max=float(params.get("threshold_delta_max", 0.10)),
            max_down_step=float(params.get("max_threshold_up_step", 0.03)),
            relax_alpha=float(params.get("threshold_relax_alpha", 0.25)),
            controller_mode=str(params.get("controller_mode", "rank_grid")),
            min_lcb_utility=float(params.get("controller_min_lcb_utility", 0.0)),
            use_timeout_cap=bool(params.get("use_timeout_cap", False)),
            min_action_edge=float(params.get("controller_min_action_edge", 0.0)),
            winner_sacrifice_multiplier=float(
                params.get("controller_winner_sacrifice_multiplier", 1.0)
            ),
            min_removed_full_sl=float(params.get("controller_min_removed_full_sl", 0.0)),
            max_removed_timeout=float(params.get("controller_max_removed_timeout", 1.0)),
            enabled_heads=_bundle_enabled_heads(bundle.get("shadow_controller_enabled_heads")),
            min_prediction_coverage=float(params.get("controller_min_prediction_coverage", 0.80)),
            min_usable_candidates=int(params.get("controller_min_usable_candidates", 1)),
            min_frontier_candidates=int(params.get("controller_min_frontier_candidates", 1)),
            max_state_ood_score=params.get("controller_max_state_ood_score"),
        )
        proposed_schedule["arm"] = f"{bundle.get('selected_arm', 'market_state_controller')}__shadow_proposed"
    scored = mstc.apply_thresholds(candidates, schedule)
    proposed_audit = (
        mstc.threshold_action_audit(proposed_schedule)
        if not proposed_schedule.empty
        else pd.DataFrame()
    )
    report = {
        "candidate_feature_count": int(len(candidate_feature_cols)),
        "feature_store": fs_report,
        "market_state_source": state_source_report,
        "state_frame_validation": state_frame_validation,
        "state_join_validation": state_join_validation,
        "state_rows": int(len(state)),
        "state_feature_count": int(len([c for c in state.columns if c != "timestamp"])),
        "active_state_feature_count": int(len(list(bundle.get("state_feature_columns", [])))),
        "observed_axis_encoder_validation": observed_axis_encoder_validation,
        "feature_store_tail_reference_validation": feature_store_tail_reference_validation,
        "state_activation_filter_validation": _validate_state_activation_filter_contract(bundle),
        "controller_execution_enabled": bool(bundle.get("controller_execution_enabled", params.get("execution_enabled", True))),
        "controller_no_backfill_overlay": bool(
            bundle.get("controller_no_backfill_overlay")
            or dict(bundle.get("state_spec") or {}).get("controller_no_backfill_overlay")
            or params.get("controller_no_backfill_overlay", False)
        ),
        "shadow_controller_only": _shadow_controller_only(bundle),
        "shadow_proposed_schedule_rows": int(len(proposed_schedule)),
        "shadow_threshold_raised_count": (
            int(proposed_audit.loc[proposed_audit["scope"].eq("all"), "threshold_raised_count"].iloc[0])
            if not proposed_audit.empty and "threshold_raised_count" in proposed_audit.columns
            else 0
        ),
        "prediction_rows": int(len(predictions)),
        "schedule_rows": int(len(schedule)),
    }
    return scored, predictions, schedule, state, report, proposed_schedule


def _controller_config_payload(bundle: dict[str, Any]) -> dict[str, Any]:
    feature_contract = bundle.get("market_state_feature_contract")
    if not isinstance(feature_contract, dict):
        feature_contract = {}
    return {
        "config_version": "strategy_threshold_controller_config_v1",
        "generated_by": "score_market_state_controller_bundle",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_arm": bundle.get("selected_arm"),
        "selected_controller": bundle.get("selected_controller"),
        "state_spec": bundle.get("state_spec"),
        "rank_contract": bundle.get("rank_contract"),
        "rank_reference_run_id": bundle.get("rank_reference_run_id"),
        "data_root": bundle.get("data_root"),
        "controller_no_backfill_overlay": bool(
            bundle.get("controller_no_backfill_overlay")
            or dict(bundle.get("state_spec") or {}).get("controller_no_backfill_overlay")
            or dict(bundle.get("controller_params") or {}).get("controller_no_backfill_overlay", False)
        ),
        "disabled_heads": bundle.get("disabled_heads"),
        "active_heads": bundle.get("active_heads"),
        "controller_execution_enabled": bool(
            bundle.get(
                "controller_execution_enabled",
                dict(bundle.get("controller_params") or {}).get("execution_enabled", True),
            )
        ),
        "controller_enabled_heads": bundle.get("controller_enabled_heads"),
        "controller_enabled_scope": bundle.get("controller_enabled_scope"),
        "shadow_controller_only": _shadow_controller_only(bundle),
        "shadow_controller_enabled_heads": bundle.get("shadow_controller_enabled_heads"),
        "shadow_controller_enabled_scope": bundle.get("shadow_controller_enabled_scope"),
        "controller_params": dict(bundle.get("controller_params") or {}),
        "forecast_model_kind": bundle.get("forecast_model_kind"),
        "forecast_model_kind_resolution": bundle.get("forecast_model_kind_resolution"),
        "response_model_kind": bundle.get("response_model_kind"),
        "response_model_kind_resolution": bundle.get("response_model_kind_resolution"),
        "runtime_param_resolution": bundle.get("runtime_param_resolution"),
        "state_activation_filter": bundle.get("state_activation_filter"),
        "state_activation_filter_validation": bundle.get("state_activation_filter_validation"),
        "invariants": {
            "penalty_only": True,
            "controller_can_lower_thresholds": False,
            "controller_changes_scores_or_ranks": False,
            "controller_changes_auction_ordering": False,
        },
        "frozen_feature_contract_version": feature_contract.get("contract_version"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--eval-candidates", type=Path, default=mstc.DEFAULT_EVAL_CANDIDATES)
    parser.add_argument("--eval-feature-store-dir", type=Path, default=mstc.DEFAULT_EVAL_FEATURE_STORE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--feature-store-symbol-cap", type=int, default=220)
    parser.add_argument("--allow-candidate-state-fallback", action="store_true", default=False)
    parser.add_argument("--policy-manifest", type=Path, default=mstc.DEFAULT_POLICY_MANIFEST)
    parser.add_argument("--policy-variant", default="refit_bar4_strategy_bar2")
    parser.add_argument("--train-deployable-candidates", type=Path, default=mstc.DEFAULT_TRAIN_DEPLOYABLE)
    parser.add_argument("--market-mode", default="perps")
    parser.add_argument("--skip-replay", action="store_true")
    parser.add_argument("--window-start", default=None)
    parser.add_argument("--window-end", default=None)
    parser.add_argument(
        "--allow-selected-arm-default-bundle",
        action="store_true",
        default=False,
        help=(
            "Research/debug override: score a bundle that was materialized from "
            "a default selected arm rather than a promoted walk-forward arm."
        ),
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    bundle = _load_bundle(
        args.bundle,
        allow_selected_arm_default_bundle=bool(args.allow_selected_arm_default_bundle),
    )
    candidates = mstc._load_candidates(args.eval_candidates)
    bundle_data_root = Path(bundle.get("data_root") or mstc.DEFAULT_DATA_ROOT)
    bundle_rank_reference_run_id = str(
        bundle.get("rank_reference_run_id") or mstc.DEFAULT_RANK_REFERENCE_RUN_ID
    )
    candidates = mstc._apply_rank_contract(
        candidates,
        str(bundle.get("rank_contract", "strict")),
        data_root=bundle_data_root,
        rank_reference_run_id=bundle_rank_reference_run_id,
    )
    candidates = mstc._disable_heads(candidates, set(bundle.get("disabled_heads", [])))
    scored, predictions, schedule, state, score_report, proposed_schedule = score_candidates(
        bundle=bundle,
        candidates=candidates,
        feature_store_dir=args.eval_feature_store_dir,
        feature_store_symbol_cap=int(args.feature_store_symbol_cap),
        allow_candidate_state_fallback=bool(args.allow_candidate_state_fallback),
    )
    state_level = str(bundle.get("state_spec", {}).get("state_level", "observed"))
    timestamp_panel = mstc.market_state_timestamp_panel([("score", state_level, state)])
    feature_coverage = mstc.market_state_feature_coverage(timestamp_panel)

    scored.to_parquet(args.output_dir / "controller_scored_candidates.parquet", index=False)
    predictions.to_parquet(args.output_dir / "controller_predictions.parquet", index=False)
    schedule.to_csv(args.output_dir / "controller_schedule.csv", index=False)
    timestamp_panel.to_parquet(args.output_dir / "market_state_timestamp_panel.parquet", index=False)
    feature_coverage.to_csv(args.output_dir / "market_state_feature_coverage.csv", index=False)
    schedule.to_parquet(args.output_dir / "strategy_threshold_schedule.parquet", index=False)
    action_audit = mstc.threshold_action_audit(schedule)
    action_audit.to_csv(args.output_dir / "strategy_threshold_action_audit.csv", index=False)
    if bool(score_report.get("shadow_controller_only", False)):
        proposed_schedule.to_csv(args.output_dir / "shadow_controller_proposed_schedule.csv", index=False)
        proposed_schedule.to_parquet(
            args.output_dir / "shadow_controller_proposed_schedule.parquet",
            index=False,
        )
        shadow_action_audit = (
            mstc.threshold_action_audit(proposed_schedule)
            if not proposed_schedule.empty
            else pd.DataFrame()
        )
        shadow_action_audit.to_csv(args.output_dir / "shadow_threshold_action_audit.csv", index=False)
        shadow_suppression_utility = mstc._threshold_candidate_suppression_utility(
            candidates,
            proposed_schedule,
        )
        shadow_suppression_utility.to_csv(
            args.output_dir / "shadow_threshold_candidate_suppression_utility.csv",
            index=False,
        )
    controller_config = _controller_config_payload(bundle)
    (args.output_dir / "strategy_threshold_controller_config.json").write_text(
        json.dumps(_json_safe(controller_config), indent=2) + "\n",
        encoding="utf-8",
    )

    outputs = {
        "controller_scored_candidates": str(args.output_dir / "controller_scored_candidates.parquet"),
        "controller_predictions": str(args.output_dir / "controller_predictions.parquet"),
        "controller_schedule": str(args.output_dir / "controller_schedule.csv"),
        "market_state_timestamp_panel": str(args.output_dir / "market_state_timestamp_panel.parquet"),
        "market_state_feature_coverage": str(args.output_dir / "market_state_feature_coverage.csv"),
        "strategy_threshold_schedule": str(args.output_dir / "strategy_threshold_schedule.parquet"),
        "strategy_threshold_action_audit": str(args.output_dir / "strategy_threshold_action_audit.csv"),
        "strategy_threshold_controller_config": str(args.output_dir / "strategy_threshold_controller_config.json"),
    }
    if bool(score_report.get("shadow_controller_only", False)):
        outputs.update(
            {
                "shadow_controller_proposed_schedule": str(
                    args.output_dir / "shadow_controller_proposed_schedule.parquet"
                ),
                "shadow_controller_proposed_schedule_csv": str(
                    args.output_dir / "shadow_controller_proposed_schedule.csv"
                ),
                "shadow_threshold_action_audit": str(
                    args.output_dir / "shadow_threshold_action_audit.csv"
                ),
                "shadow_threshold_candidate_suppression_utility": str(
                    args.output_dir / "shadow_threshold_candidate_suppression_utility.csv"
                ),
            }
        )
    frozen_feature_contract = bundle.get("market_state_feature_contract")
    if isinstance(frozen_feature_contract, dict):
        feature_contract_path = args.output_dir / "market_state_feature_contract.json"
        feature_contract_path.write_text(
            json.dumps(_json_safe(frozen_feature_contract), indent=2) + "\n",
            encoding="utf-8",
        )
        outputs["market_state_feature_contract"] = str(feature_contract_path)
    replay_summary = pd.DataFrame()
    replay_by_head = pd.DataFrame()
    shadow_no_backfill_summary = pd.DataFrame()
    shadow_no_backfill_by_head = pd.DataFrame()
    shadow_no_backfill_accepted_delta_summary: dict[str, Any] = {}
    shadow_direct_threshold_only_summary = pd.DataFrame()
    shadow_direct_threshold_only_by_head = pd.DataFrame()
    shadow_direct_threshold_only_delta_summary: dict[str, Any] = {}
    shadow_locked_overlay_summary = pd.DataFrame()
    shadow_locked_overlay_by_head = pd.DataFrame()
    shadow_locked_overlay_delta_summary: dict[str, Any] = {}
    if not args.skip_replay and not candidates.empty:
        params, _ = mstc._load_policy_params(args.policy_manifest, args.policy_variant)
        train_deployable = mstc._load_candidates(args.train_deployable_candidates)
        train_deployable = mstc._apply_rank_contract(
            train_deployable,
            str(bundle.get("rank_contract", "strict")),
            data_root=bundle_data_root,
            rank_reference_run_id=bundle_rank_reference_run_id,
        )
        train_deployable = mstc._disable_heads(train_deployable, set(bundle.get("disabled_heads", [])))
        ev_curve = fit_hierarchical_ev_curves(train_deployable)
        scored_for_replay = scored
        baseline_allowed_keys: set[tuple[Any, ...]] = set()
        no_backfill_overlay = bool(
            bundle.get("controller_no_backfill_overlay")
            or dict(bundle.get("state_spec") or {}).get("controller_no_backfill_overlay")
            or dict(bundle.get("controller_params") or {}).get("controller_no_backfill_overlay", False)
        )
        if no_backfill_overlay:
            baseline_decisions, _, _baseline_metrics = replay_candidates(
                candidates,
                params,
                mode="global_auction",
                ev_curve=ev_curve,
                market_mode=args.market_mode,
            )
            baseline_accepted = mstc._accepted_trades(candidates, baseline_decisions)
            baseline_allowed_keys = mstc._accepted_key_set(baseline_accepted)
            scored_for_replay = mstc._restrict_to_allowed_decision_keys(
                scored,
                baseline_allowed_keys,
            )
            scored_for_replay.to_parquet(
                args.output_dir / "controller_replay_candidates.parquet",
                index=False,
            )
            outputs["controller_replay_candidates"] = str(
                args.output_dir / "controller_replay_candidates.parquet"
            )
        decisions, _, metrics = replay_candidates(
            scored_for_replay,
            params,
            mode="global_auction",
            ev_curve=ev_curve,
            market_mode=args.market_mode,
        )
        accepted = mstc._accepted_trades(scored_for_replay, decisions)
        arm = str(bundle.get("selected_arm", "market_state_controller"))
        schedule["arm"] = arm
        accepted["arm"] = arm
        if no_backfill_overlay:
            accepted_keys = mstc._accepted_key_set(accepted)
            if not accepted_keys.issubset(baseline_allowed_keys):
                raise RuntimeError("No-backfill overlay accepted trades outside baseline accepted keys")
        replay_summary = pd.DataFrame([mstc._metrics_row(arm, metrics, accepted, schedule)])
        replay_by_head = mstc._by_head(arm, accepted)
        if no_backfill_overlay and not proposed_schedule.empty:
            shadow_replay_candidates = mstc._restrict_to_allowed_decision_keys(
                candidates,
                baseline_allowed_keys,
            )
            shadow_scored = mstc.apply_thresholds(shadow_replay_candidates, proposed_schedule)
            shadow_decisions, _, shadow_metrics = replay_candidates(
                shadow_scored,
                params,
                mode="global_auction",
                ev_curve=ev_curve,
                market_mode=args.market_mode,
            )
            shadow_accepted = mstc._accepted_trades(shadow_scored, shadow_decisions)
            shadow_arm = f"{arm}__shadow_no_backfill"
            shadow_accepted["arm"] = shadow_arm
            proposed_schedule_for_metrics = proposed_schedule.copy()
            proposed_schedule_for_metrics["arm"] = shadow_arm
            shadow_accepted_keys = mstc._accepted_key_set(shadow_accepted)
            if not shadow_accepted_keys.issubset(baseline_allowed_keys):
                raise RuntimeError("Shadow no-backfill replay accepted trades outside baseline accepted keys")
            shadow_no_backfill_summary = pd.DataFrame(
                [mstc._metrics_row(shadow_arm, shadow_metrics, shadow_accepted, proposed_schedule_for_metrics)]
            )
            shadow_no_backfill_by_head = mstc._by_head(shadow_arm, shadow_accepted)
            shadow_scored.to_parquet(
                args.output_dir / "shadow_no_backfill_scored_candidates.parquet",
                index=False,
            )
            shadow_decisions.to_parquet(
                args.output_dir / "shadow_no_backfill_decisions.parquet",
                index=False,
            )
            shadow_accepted.to_parquet(
                args.output_dir / "shadow_no_backfill_accepted_trades.parquet",
                index=False,
            )
            shadow_no_backfill_summary.to_csv(
                args.output_dir / "shadow_no_backfill_replay_summary.csv",
                index=False,
            )
            shadow_no_backfill_by_head.to_csv(
                args.output_dir / "shadow_no_backfill_replay_by_head.csv",
                index=False,
            )
            shadow_no_backfill_suppression = mstc._threshold_candidate_suppression_utility(
                shadow_replay_candidates,
                proposed_schedule,
            )
            shadow_no_backfill_suppression.to_csv(
                args.output_dir / "shadow_no_backfill_threshold_candidate_suppression_utility.csv",
                index=False,
            )
            shadow_delta, shadow_no_backfill_accepted_delta_summary = _accepted_trade_delta_report(
                baseline_accepted,
                shadow_accepted,
            )
            shadow_delta.to_csv(
                args.output_dir / "shadow_no_backfill_accepted_trade_delta.csv",
                index=False,
            )
            (
                direct_accepted,
                direct_delta,
                shadow_direct_threshold_only_delta_summary,
            ) = _direct_threshold_only_overlay(baseline_accepted, proposed_schedule)
            direct_arm = f"{arm}__shadow_direct_threshold_only"
            if not direct_accepted.empty:
                direct_accepted["arm"] = direct_arm
            shadow_direct_threshold_only_summary = pd.DataFrame(
                [shadow_direct_threshold_only_delta_summary]
            )
            shadow_direct_threshold_only_by_head = mstc._by_head(
                direct_arm, direct_accepted
            )
            direct_accepted.to_parquet(
                args.output_dir / "shadow_direct_threshold_only_accepted_trades.parquet",
                index=False,
            )
            direct_delta.to_csv(
                args.output_dir / "shadow_direct_threshold_only_accepted_trade_delta.csv",
                index=False,
            )
            shadow_direct_threshold_only_summary.to_csv(
                args.output_dir / "shadow_direct_threshold_only_summary.csv",
                index=False,
            )
            shadow_direct_threshold_only_by_head.to_csv(
                args.output_dir / "shadow_direct_threshold_only_by_head.csv",
                index=False,
            )
            (
                locked_arm,
                locked_accepted,
                locked_delta,
                shadow_locked_overlay_delta_summary,
            ) = _locked_accepted_overlay_from_direct(
                arm=arm,
                direct_accepted=direct_accepted,
                direct_delta=direct_delta,
                direct_summary=shadow_direct_threshold_only_delta_summary,
            )
            shadow_locked_overlay_summary = pd.DataFrame(
                [shadow_locked_overlay_delta_summary]
            )
            shadow_locked_overlay_by_head = mstc._by_head(
                locked_arm,
                locked_accepted,
            )
            locked_accepted.to_parquet(
                args.output_dir / "shadow_locked_accepted_overlay_trades.parquet",
                index=False,
            )
            locked_delta.to_csv(
                args.output_dir / "shadow_locked_accepted_overlay_delta.csv",
                index=False,
            )
            shadow_locked_overlay_summary.to_csv(
                args.output_dir / "shadow_locked_accepted_overlay_summary.csv",
                index=False,
            )
            shadow_locked_overlay_by_head.to_csv(
                args.output_dir / "shadow_locked_accepted_overlay_by_head.csv",
                index=False,
            )
            outputs.update(
                {
                    "shadow_no_backfill_scored_candidates": str(
                        args.output_dir / "shadow_no_backfill_scored_candidates.parquet"
                    ),
                    "shadow_no_backfill_decisions": str(
                        args.output_dir / "shadow_no_backfill_decisions.parquet"
                    ),
                    "shadow_no_backfill_accepted_trades": str(
                        args.output_dir / "shadow_no_backfill_accepted_trades.parquet"
                    ),
                    "shadow_no_backfill_replay_summary": str(
                        args.output_dir / "shadow_no_backfill_replay_summary.csv"
                    ),
                    "shadow_no_backfill_replay_by_head": str(
                        args.output_dir / "shadow_no_backfill_replay_by_head.csv"
                    ),
                    "shadow_no_backfill_threshold_candidate_suppression_utility": str(
                        args.output_dir / "shadow_no_backfill_threshold_candidate_suppression_utility.csv"
                    ),
                    "shadow_no_backfill_accepted_trade_delta": str(
                        args.output_dir / "shadow_no_backfill_accepted_trade_delta.csv"
                    ),
                    "shadow_direct_threshold_only_accepted_trades": str(
                        args.output_dir / "shadow_direct_threshold_only_accepted_trades.parquet"
                    ),
                    "shadow_direct_threshold_only_accepted_trade_delta": str(
                        args.output_dir / "shadow_direct_threshold_only_accepted_trade_delta.csv"
                    ),
                    "shadow_direct_threshold_only_summary": str(
                        args.output_dir / "shadow_direct_threshold_only_summary.csv"
                    ),
                    "shadow_direct_threshold_only_by_head": str(
                        args.output_dir / "shadow_direct_threshold_only_by_head.csv"
                    ),
                    "shadow_locked_accepted_overlay_trades": str(
                        args.output_dir / "shadow_locked_accepted_overlay_trades.parquet"
                    ),
                    "shadow_locked_accepted_overlay_delta": str(
                        args.output_dir / "shadow_locked_accepted_overlay_delta.csv"
                    ),
                    "shadow_locked_accepted_overlay_summary": str(
                        args.output_dir / "shadow_locked_accepted_overlay_summary.csv"
                    ),
                    "shadow_locked_accepted_overlay_by_head": str(
                        args.output_dir / "shadow_locked_accepted_overlay_by_head.csv"
                    ),
                }
            )
        decisions.to_parquet(args.output_dir / "decisions.parquet", index=False)
        accepted.to_parquet(args.output_dir / "accepted_trades.parquet", index=False)
        replay_summary.to_csv(args.output_dir / "controller_replay_summary.csv", index=False)
        replay_by_head.to_csv(args.output_dir / "controller_replay_by_head.csv", index=False)
        outputs.update(
            {
                "decisions": str(args.output_dir / "decisions.parquet"),
                "accepted_trades": str(args.output_dir / "accepted_trades.parquet"),
                "controller_replay_summary": str(args.output_dir / "controller_replay_summary.csv"),
                "controller_replay_by_head": str(args.output_dir / "controller_replay_by_head.csv"),
            }
        )

    manifest = {
        "generated_by": "score_market_state_controller_bundle",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "score_manifest_contract_version": "market_state_controller_score_manifest_v2",
        "bundle": str(args.bundle),
        "bundle_sha256": _file_sha256(args.bundle),
        "policy_manifest": str(args.policy_manifest),
        "policy_manifest_sha256": _file_sha256(args.policy_manifest),
        "eval_candidates_sha256": _file_sha256(args.eval_candidates),
        "train_deployable_candidates": str(args.train_deployable_candidates),
        "train_deployable_candidates_sha256": _file_sha256(args.train_deployable_candidates),
        "selected_arm": bundle.get("selected_arm"),
        "state_level": bundle.get("state_spec", {}).get("state_level"),
        "rank_contract": bundle.get("rank_contract"),
        "rank_reference_run_id": bundle.get("rank_reference_run_id"),
        "data_root": bundle.get("data_root"),
        "controller_no_backfill_overlay": bool(
            bundle.get("controller_no_backfill_overlay")
            or dict(bundle.get("state_spec") or {}).get("controller_no_backfill_overlay")
            or dict(bundle.get("controller_params") or {}).get("controller_no_backfill_overlay", False)
        ),
        "disabled_heads": bundle.get("disabled_heads"),
        "active_heads": bundle.get("active_heads"),
        "source_contract_audit": (
            frozen_feature_contract.get("source_contract_audit")
            if isinstance(frozen_feature_contract, dict)
            else None
        ),
        "controller": {
            "penalty_only": True,
            "execution_enabled": bool(
                bundle.get(
                    "controller_execution_enabled",
                    dict(bundle.get("controller_params") or {}).get("execution_enabled", True),
                )
            ),
            "controller_execution_enabled": bool(
                bundle.get(
                    "controller_execution_enabled",
                    dict(bundle.get("controller_params") or {}).get("execution_enabled", True),
                )
            ),
            "controller_enabled_heads": bundle.get("controller_enabled_heads"),
            "controller_enabled_scope": bundle.get("controller_enabled_scope"),
            "shadow_controller_only": _shadow_controller_only(bundle),
            "changes_scores_or_ranks": False,
            "changes_auction_ordering": False,
        },
        "controller_enabled_heads": bundle.get("controller_enabled_heads"),
        "controller_enabled_scope": bundle.get("controller_enabled_scope"),
        "controller_execution_enabled": bool(
            bundle.get(
                "controller_execution_enabled",
                dict(bundle.get("controller_params") or {}).get("execution_enabled", True),
            )
        ),
        "shadow_controller_only": _shadow_controller_only(bundle),
        "shadow_controller_enabled_heads": bundle.get("shadow_controller_enabled_heads"),
        "shadow_controller_enabled_scope": bundle.get("shadow_controller_enabled_scope"),
        "shadow_no_backfill_replay_available": bool(not shadow_no_backfill_summary.empty),
        "shadow_no_backfill_replay_summary": (
            shadow_no_backfill_summary.iloc[0].to_dict()
            if not shadow_no_backfill_summary.empty
            else {}
        ),
        "shadow_no_backfill_accepted_delta_summary": shadow_no_backfill_accepted_delta_summary,
        "shadow_direct_threshold_only_available": bool(
            not shadow_direct_threshold_only_summary.empty
        ),
        "shadow_direct_threshold_only_summary": (
            shadow_direct_threshold_only_summary.iloc[0].to_dict()
            if not shadow_direct_threshold_only_summary.empty
            else {}
        ),
        "shadow_direct_threshold_only_delta_summary": shadow_direct_threshold_only_delta_summary,
        "shadow_locked_accepted_overlay_available": bool(
            not shadow_locked_overlay_summary.empty
        ),
        "shadow_locked_accepted_overlay_summary": (
            shadow_locked_overlay_summary.iloc[0].to_dict()
            if not shadow_locked_overlay_summary.empty
            else {}
        ),
        "shadow_locked_accepted_overlay_delta_summary": shadow_locked_overlay_delta_summary,
        "forecast_model_kind": bundle.get("forecast_model_kind"),
        "forecast_model_kind_resolution": bundle.get("forecast_model_kind_resolution"),
        "response_model_kind": bundle.get("response_model_kind"),
        "response_model_kind_resolution": bundle.get("response_model_kind_resolution"),
        "runtime_param_resolution": bundle.get("runtime_param_resolution"),
        "state_activation_filter": bundle.get("state_activation_filter"),
        "state_activation_filter_validation": bundle.get("state_activation_filter_validation"),
        "eval_candidates": str(args.eval_candidates),
        "period_start": args.window_start,
        "period_end": args.window_end,
        "window_start": args.window_start,
        "window_end": args.window_end,
        "eval_feature_store_dir": str(args.eval_feature_store_dir),
        "score_report": score_report,
        "market_state_feature_contract": (
            {
                "available": True,
                "contract_version": frozen_feature_contract.get("contract_version"),
                "state_level": frozen_feature_contract.get("state_level"),
                "state_feature_count": len(
                    frozen_feature_contract.get("source_schema", {}).get("state_feature_columns", [])
                ),
                "response_feature_count": len(
                    frozen_feature_contract.get("source_schema", {}).get("response_feature_columns", [])
                ),
            }
            if isinstance(frozen_feature_contract, dict)
            else {"available": False}
        ),
        "outputs": outputs,
        "output_sha256": _output_sha256(outputs),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(_json_safe(manifest), indent=2)[:6000])
    print(f"\nWrote {args.output_dir}")


if __name__ == "__main__":
    main()
