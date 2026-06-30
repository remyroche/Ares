#!/usr/bin/env python3
"""Materialize a full-fit market-state threshold-controller bundle.

This turns the selected walk-forward controller arm into a physical artifact
that can be scored prospectively.  It deliberately reuses the same encoder,
response-model and threshold-schedule functions as the ablation script so the
live bundle does not drift from the validated research path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
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


DEFAULT_SELECTED_CONTROLLER = Path(
    "data_perp/reports/market_state_threshold_controller_walkforward_20260626_t1_lgbm_maturity_contract_v1"
    "/walkforward_selected_controller_candidate.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "data_perp/reports/market_state_controller_bundle_t1_lgbm_maturity_noop_20260626"
)
NOOP_CONTROLLER_ARM = "S0_rejected_controller_noop"

MATERIALIZER_RUNTIME_DEFAULTS: dict[str, Any] = {
    "response_frontier_weight_gamma": 3.0,
    "response_frontier_weight_bandwidth": 0.06,
    "response_balance_timestamps": True,
    "response_balance_strategies": True,
    "threshold_delta_max": 0.10,
    "max_threshold_up_step": 0.03,
    "threshold_relax_alpha": 0.25,
    "controller_mode": "rank_grid",
    "controller_min_lcb_utility": 0.0,
    "controller_min_prediction_coverage": 0.80,
    "controller_min_usable_candidates": 1,
    "controller_min_frontier_candidates": 1,
    "controller_max_state_ood_score": None,
    "controller_min_action_edge": 0.0,
    "controller_winner_sacrifice_multiplier": 1.0,
    "controller_min_removed_full_sl": 0.0,
    "controller_max_removed_timeout": 1.0,
    "use_timeout_cap": False,
}

WALKFORWARD_RUNTIME_PARAM_PATHS: dict[str, tuple[str, ...]] = {
    "response_frontier_weight_gamma": ("response_weighting", "frontier_gamma"),
    "response_frontier_weight_bandwidth": ("response_weighting", "frontier_bandwidth"),
    "response_balance_timestamps": ("response_weighting", "timestamp_balanced"),
    "response_balance_strategies": ("response_weighting", "strategy_balanced"),
    "threshold_delta_max": ("threshold_delta_max",),
    "max_threshold_up_step": ("max_threshold_up_step",),
    "threshold_relax_alpha": ("threshold_relax_alpha",),
    "controller_mode": ("controller_mode",),
    "controller_min_lcb_utility": ("controller_min_lcb_utility",),
    "controller_min_prediction_coverage": ("controller_min_prediction_coverage",),
    "controller_min_usable_candidates": ("controller_min_usable_candidates",),
    "controller_min_frontier_candidates": ("controller_min_frontier_candidates",),
    "controller_max_state_ood_score": ("controller_max_state_ood_score",),
    "controller_min_action_edge": ("controller_min_action_edge",),
    "controller_winner_sacrifice_multiplier": ("controller_winner_sacrifice_multiplier",),
    "controller_min_removed_full_sl": ("controller_min_removed_full_sl",),
    "controller_max_removed_timeout": ("controller_max_removed_timeout",),
    "use_timeout_cap": ("use_timeout_cap",),
}


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


def _artifact_hashes(paths: dict[str, str]) -> dict[str, Any]:
    artifacts: dict[str, Any] = {}
    for name, raw_path in sorted(paths.items()):
        path = Path(raw_path)
        exists = bool(path.exists() and path.is_file())
        artifacts[str(name)] = {
            "path": str(path),
            "exists": exists,
            "bytes": int(path.stat().st_size) if exists else None,
            "sha256": _file_sha256(path) if exists else None,
        }
    return {
        "hash_version": "sha256_artifact_hashes_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": artifacts,
    }


def _load_selected_arm(
    path: Path,
    default: str,
    *,
    allow_default: bool = False,
    allow_null_noop: bool = False,
) -> tuple[str, dict[str, Any]]:
    if not path.exists():
        payload = {
            "path": str(path),
            "exists": False,
            "selected_arm": default,
            "source": "default_missing_selection_file",
        }
        if allow_default:
            payload["selected_arm_default_used"] = True
            return default, payload
        raise RuntimeError(
            f"Selected controller file is missing: {path}. "
            "Refusing to materialize a default controller without "
            "--allow-selected-arm-default."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload = dict(payload)
    payload["path"] = str(path)
    payload["exists"] = True
    selected_raw = payload.get("selected_arm")
    if selected_raw is None or str(selected_raw).strip() == "":
        reason = payload.get("reason") or "selection did not promote a controller arm"
        if allow_null_noop:
            payload["selected_arm"] = None
            payload["selected_arm_default_used"] = False
            payload["selected_arm_noop_used"] = True
            payload["noop_arm"] = NOOP_CONTROLLER_ARM
            payload["noop_reason"] = reason
            return NOOP_CONTROLLER_ARM, payload
        if allow_default:
            payload["selected_arm"] = default
            payload["selected_arm_default_used"] = True
            return default, payload
        raise RuntimeError(
            f"Selected controller {path} has no selected_arm ({reason}). "
            "Refusing to materialize a rejected/default controller without "
            "--allow-null-selection-noop or --allow-selected-arm-default."
        )
    selected = str(selected_raw)
    payload["selected_arm_default_used"] = False
    return selected, payload


def _apply_shadow_selected_arm_override(
    *,
    selected_arm: str,
    selected_payload: dict[str, Any],
    shadow_selected_arm: str,
) -> tuple[str, dict[str, Any]]:
    """Use a rejected/promoted arm for shadow logging while leaving execution off.

    The walk-forward selector can legitimately reject every controller.  Stage-2
    monitoring still needs a frozen controller to emit proposed schedules for
    matured-outcome analysis.  This helper makes that shadow intent explicit in
    the persisted selected-controller payload instead of pretending the arm was
    promoted.
    """

    shadow_selected_arm = str(shadow_selected_arm or "").strip()
    if not shadow_selected_arm:
        return selected_arm, selected_payload
    _arm_state_spec(shadow_selected_arm)
    original_payload = dict(selected_payload)
    original_promoted = (
        not bool(original_payload.get("selected_arm_noop_used", False))
        and original_payload.get("selected_arm") is not None
        and str(original_payload.get("selected_arm")).strip() != ""
    )
    if original_promoted and str(original_payload.get("selected_arm")) != shadow_selected_arm:
        raise RuntimeError(
            "Refusing to shadow-override an already promoted controller arm "
            f"{original_payload.get('selected_arm')!r} with {shadow_selected_arm!r}. "
            "Use --shadow-selected-arm only for rejected/null selections or for "
            "the same promoted arm."
        )
    payload = dict(original_payload)
    payload.update(
        {
            "selected_arm": shadow_selected_arm,
            "selected_arm_default_used": False,
            "selected_arm_noop_used": False,
            "selected_arm_shadow_used": True,
            "shadow_selected_arm": shadow_selected_arm,
            "shadow_controller_only": True,
            "shadow_source_selected_arm": original_payload.get("selected_arm"),
            "shadow_source_noop_arm": (
                selected_arm if selected_arm == NOOP_CONTROLLER_ARM else original_payload.get("noop_arm")
            ),
            "shadow_source_reason": original_payload.get("reason")
            or original_payload.get("noop_reason"),
        }
    )
    return shadow_selected_arm, payload


def _arm_state_spec(selected_arm: str) -> dict[str, Any]:
    arm = str(selected_arm)
    no_backfill_overlay = arm.endswith("__post_selection_overlay")
    base_arm = arm.replace("__post_selection_overlay", "")
    if arm == NOOP_CONTROLLER_ARM:
        return {
            "state_level": "observed",
            "per_strategy_residual": False,
            "controller_noop": True,
            "controller_no_backfill_overlay": False,
            "base_arm": arm,
        }
    if base_arm == "S1_observed_axes_shared_response":
        return {
            "state_level": "observed",
            "per_strategy_residual": False,
            "controller_noop": False,
            "controller_no_backfill_overlay": bool(no_backfill_overlay),
            "base_arm": base_arm,
        }
    if base_arm == "S2_observed_forecast_shared_response":
        return {
            "state_level": "forecast",
            "per_strategy_residual": False,
            "controller_noop": False,
            "controller_no_backfill_overlay": bool(no_backfill_overlay),
            "base_arm": base_arm,
        }
    if base_arm == "S3_observed_forecast_latent_shared_response":
        return {
            "state_level": "latent",
            "per_strategy_residual": False,
            "controller_noop": False,
            "controller_no_backfill_overlay": bool(no_backfill_overlay),
            "base_arm": base_arm,
        }
    if base_arm == "S4_S3_plus_per_strategy_residual":
        return {
            "state_level": "latent",
            "per_strategy_residual": True,
            "controller_noop": False,
            "controller_no_backfill_overlay": bool(no_backfill_overlay),
            "base_arm": base_arm,
        }
    raise ValueError(
        f"Selected controller arm {selected_arm!r} is not materializable by this bundle script"
    )


def _infer_walkforward_manifest_path(selected_controller: Path, explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit
    candidate = selected_controller.parent / "manifest.json"
    return candidate if candidate.exists() else None


def _load_walkforward_controller_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "available": False,
            "path": None,
            "reason": "walkforward_manifest_not_provided_or_not_found",
            "controller": {},
        }
    if not path.exists():
        raise FileNotFoundError(f"Walk-forward manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    controller = dict(payload.get("controller") or {})
    return {
        "available": True,
        "path": str(path),
        "generated_by": payload.get("generated_by"),
        "generated_at_utc": payload.get("generated_at_utc"),
        "controller": controller,
    }


def _resolve_backend_kind(
    requested: str,
    walkforward_config: dict[str, Any],
    key: str,
    default: str,
    valid: set[str],
) -> tuple[str, dict[str, Any]]:
    raw_requested = str(requested or "auto")
    source = "cli"
    value = raw_requested
    if raw_requested == "auto":
        controller = dict(walkforward_config.get("controller") or {})
        value = str(controller.get(key) or default)
        source = "walkforward_manifest" if controller.get(key) else "default"
    if value not in valid:
        raise ValueError(f"Invalid {key}={value!r}; expected one of {sorted(valid)}")
    return value, {
        "key": key,
        "value": value,
        "source": source,
        "requested": raw_requested,
        "default": default,
        "walkforward_manifest": walkforward_config.get("path"),
    }


def _nested_get(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    cur: Any = payload
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _cast_like(value: Any, default: Any) -> Any:
    if value is None:
        return None
    if isinstance(default, bool):
        return bool(value)
    if isinstance(default, int) and not isinstance(default, bool):
        return int(value)
    if isinstance(default, float):
        return float(value)
    if default is None:
        return value
    return str(value)


def _materializer_value_is_default(value: Any, default: Any) -> bool:
    if value is None or default is None:
        return value is None and default is None
    if isinstance(default, float):
        try:
            return abs(float(value) - float(default)) <= 1e-12
        except Exception:
            return False
    return value == default


def _apply_walkforward_runtime_defaults(
    args: argparse.Namespace,
    walkforward_config: dict[str, Any],
) -> dict[str, Any]:
    controller = dict(walkforward_config.get("controller") or {})
    inherit = bool(getattr(args, "inherit_walkforward_controller_config", True))
    report: dict[str, Any] = {
        "inherit_walkforward_controller_config": inherit,
        "walkforward_manifest": walkforward_config.get("path"),
        "params": {},
    }
    for attr, default in MATERIALIZER_RUNTIME_DEFAULTS.items():
        current = getattr(args, attr, default)
        if not hasattr(args, attr):
            setattr(args, attr, current)
        manifest_value = _nested_get(controller, WALKFORWARD_RUNTIME_PARAM_PATHS[attr])
        source = "cli_or_materializer_default"
        value = current
        if inherit and manifest_value is not None and _materializer_value_is_default(current, default):
            value = _cast_like(manifest_value, default)
            setattr(args, attr, value)
            source = "walkforward_manifest"
        elif inherit and manifest_value is None:
            source = "materializer_default_manifest_missing"
        elif not inherit:
            source = "cli_or_materializer_default_no_inherit"
        report["params"][attr] = {
            "value": _json_safe(value),
            "source": source,
            "manifest_value": _json_safe(manifest_value),
            "materializer_default": _json_safe(default),
        }
    return report


def _infer_activation_registry_path(selected_controller: Path, explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit
    candidate = selected_controller.parent / "market_state_activation_registry.csv"
    return candidate if candidate.exists() else None


def _load_activation_registry(path: Path | None) -> tuple[pd.DataFrame, dict[str, Any]]:
    if path is None:
        return pd.DataFrame(), {
            "available": False,
            "path": None,
            "reason": "activation_registry_not_provided_or_not_found",
            "active_state_heads": [],
            "shadow_state_heads": [],
            "disabled_state_heads": [],
        }
    if not path.exists():
        raise FileNotFoundError(f"Activation registry not found: {path}")
    registry = pd.read_csv(path)
    if "state_head" not in registry.columns or "recommended_status" not in registry.columns:
        raise ValueError(f"Activation registry {path} is missing state_head/recommended_status columns")
    status = registry["recommended_status"].astype(str)
    heads = registry["state_head"].astype(str)
    active = sorted(set(heads.loc[status.eq("active_candidate")]))
    shadow = sorted(set(heads.loc[status.eq("shadow") | status.eq("shadow_disabled")]))
    disabled = sorted(set(heads.loc[status.eq("disabled_candidate")]))
    return registry, {
        "available": True,
        "path": str(path),
        "rows": int(len(registry)),
        "active_state_heads": active,
        "shadow_state_heads": shadow,
        "disabled_state_heads": disabled,
        "recommended_status_counts": {
            str(k): int(v) for k, v in status.value_counts(dropna=False).to_dict().items()
        },
    }


def _filter_state_columns_by_activation_registry(
    state_cols: list[str],
    activation_report: dict[str, Any],
    *,
    fail_closed_when_unavailable: bool = True,
) -> tuple[list[str], dict[str, Any]]:
    if not bool(activation_report.get("available")):
        if bool(fail_closed_when_unavailable):
            return [], {
                "enforced": True,
                "reason": "activation_registry_unavailable_fail_closed",
                "input_state_feature_count": int(len(state_cols)),
                "active_state_feature_count": 0,
                "dropped_state_feature_count": int(len(state_cols)),
                "active_state_feature_columns": [],
                "dropped_state_feature_columns": list(state_cols),
            }
        return list(state_cols), {
            "enforced": False,
            "reason": "activation_registry_unavailable",
            "input_state_feature_count": int(len(state_cols)),
            "active_state_feature_count": int(len(state_cols)),
            "dropped_state_feature_count": 0,
            "active_state_feature_columns": list(state_cols),
            "dropped_state_feature_columns": [],
        }
    active_heads = set(map(str, activation_report.get("active_state_heads") or []))
    active_cols = [c for c in state_cols if str(c) in active_heads]
    dropped = [c for c in state_cols if str(c) not in active_heads]
    return active_cols, {
        "enforced": True,
        "reason": "activation_registry_active_candidate_filter",
        "input_state_feature_count": int(len(state_cols)),
        "active_state_feature_count": int(len(active_cols)),
        "dropped_state_feature_count": int(len(dropped)),
        "active_state_feature_columns": list(active_cols),
        "dropped_state_feature_columns": list(dropped),
    }


def _validate_candidate_state_fallback_execution_contract(
    *,
    allow_candidate_state_fallback: bool,
    controller_execution_enabled: bool,
    shadow_controller_only: bool = False,
) -> None:
    if bool(allow_candidate_state_fallback) and (
        bool(controller_execution_enabled) or bool(shadow_controller_only)
    ):
        raise RuntimeError(
            "Refusing to materialize an executable or shadow market-state controller with "
            "allow_candidate_state_fallback=true. Candidate-population fallback "
            "is a debug-only source because it can depend on strategy/model/rank "
            "candidate populations. Provide feature-store market aggregates or "
            "materialize a rejected/no-op audit bundle instead."
        )


def _validate_state_reference_materialization_contract(
    *,
    state_cols: list[str],
    state_artifacts: dict[str, Any],
    controller_execution_enabled: bool,
    shadow_controller_only: bool,
) -> None:
    """Fail closed when a retained state bundle lacks live-equivalent references.

    The scorer can only transform deployment feature-store rows with the
    train-fitted observed-axis encoder. A stale eval feature-store directory can
    otherwise make ``fit_observed_axis_encoder`` keep no common numeric columns,
    producing all-neutral state rows while the bundle still advertises active
    state columns.
    """

    if not state_cols or not (bool(controller_execution_enabled) or bool(shadow_controller_only)):
        return

    reports = dict(state_artifacts.get("reports") or {})
    observed_report = dict(reports.get("observed_axis_encoder") or {})
    market_source = dict(reports.get("market_state_source") or {})
    feature_store = dict(reports.get("feature_store") or {})
    train_source = dict(market_source.get("train") or {})
    eval_source = dict(market_source.get("eval") or {})
    eval_feature_store = dict(feature_store.get("eval") or {})

    source_column_count = int(observed_report.get("source_column_count") or 0)
    train_feature_count = int(train_source.get("feature_count") or 0)
    eval_feature_count = int(eval_source.get("feature_count") or 0)
    eval_coverage = float(eval_feature_store.get("timestamp_coverage") or 0.0)
    if source_column_count <= 0 or train_feature_count <= 0 or eval_feature_count <= 0:
        raise RuntimeError(
            "Refusing to materialize executable/shadow market-state bundle with "
            "retained state columns but no common train/eval observed-axis "
            "references. Check that --eval-feature-store-dir covers the eval "
            "candidate timestamps. "
            f"source_column_count={source_column_count}, "
            f"train_feature_count={train_feature_count}, "
            f"eval_feature_count={eval_feature_count}, "
            f"eval_timestamp_coverage={eval_coverage:.6f}."
        )
    if eval_coverage <= 0.0:
        raise RuntimeError(
            "Refusing to materialize executable/shadow market-state bundle with "
            "zero eval feature-store timestamp coverage. Check "
            "--eval-feature-store-dir."
        )


def _build_state_artifacts(
    train_broad: pd.DataFrame,
    eval_candidates: pd.DataFrame | None,
    *,
    train_feature_store_dir: Path,
    eval_feature_store_dir: Path | None,
    max_feature_cols: int,
    max_feature_store_cols: int,
    feature_store_symbol_cap: int,
    allow_candidate_state_fallback: bool,
    forecast_horizons_steps: tuple[int, ...],
    forecast_model_kind: str,
    latent_states: int,
) -> dict[str, Any]:
    eval_for_columns = eval_candidates if eval_candidates is not None else train_broad
    candidate_feature_cols = mstc._common_feature_columns(
        train_broad,
        eval_for_columns,
        max_feature_cols,
    )
    train_candidate_agg = mstc._timestamp_aggregates(train_broad, candidate_feature_cols)
    feature_store_cols = mstc._select_feature_store_columns(
        train_feature_store_dir,
        eval_feature_store_dir or train_feature_store_dir,
        int(max_feature_store_cols),
    )
    if eval_candidates is None:
        eval_candidate_agg = train_candidate_agg[["timestamp"]].copy()
        train_fs, train_fs_report = mstc._feature_store_timestamp_aggregates(
            train_feature_store_dir,
            train_candidate_agg["timestamp"],
            feature_store_cols,
            symbol_cap=int(feature_store_symbol_cap),
        )
        train_fs_report["tail_reference_role"] = "fit_on_training_timestamps"
        eval_fs = eval_candidate_agg.copy()
        eval_fs_report = {
            "enabled": False,
            "reason": "no_eval_candidates",
            "columns": [],
            "tail_reference_source": "not_applicable",
            "tail_reference_role": "not_applicable_no_eval_candidates",
        }
    else:
        eval_candidate_agg = mstc._timestamp_aggregates(eval_candidates, candidate_feature_cols)
        eval_fs_dir = eval_feature_store_dir or train_feature_store_dir
        train_fs, train_fs_report, eval_fs, eval_fs_report = mstc._feature_store_timestamp_aggregate_pair(
            train_feature_store_dir,
            eval_fs_dir,
            train_candidate_agg["timestamp"],
            eval_candidate_agg["timestamp"],
            feature_store_cols,
            symbol_cap=int(feature_store_symbol_cap),
        )
    train_source, train_source_report = mstc._state_source_aggregate_frame(
        train_candidate_agg,
        train_fs,
        allow_candidate_fallback=bool(allow_candidate_state_fallback),
    )
    eval_source, eval_source_report = mstc._state_source_aggregate_frame(
        eval_candidate_agg,
        eval_fs,
        allow_candidate_fallback=bool(allow_candidate_state_fallback),
    )
    observed_axis_encoder = mstc.fit_observed_axis_encoder(train_source, eval_source)
    train_observed = mstc.transform_observed_axes(train_source, observed_axis_encoder)
    eval_observed = mstc.transform_observed_axes(eval_source, observed_axis_encoder)
    axis_sources = dict(observed_axis_encoder.get("axis_sources", {}))
    axis_sources["state_transition_pressure"] = ["mean_abs_state_axis_diff"]
    train_forecast, forecast_artifact, forecast_report = mstc.fit_forecast_state_heads(
        train_observed,
        horizon_steps=list(forecast_horizons_steps),
        train_agg=train_source,
        forecast_model_kind=str(forecast_model_kind),
    )
    eval_forecast = mstc.transform_forecast_state_heads(
        eval_observed,
        forecast_artifact,
        agg=eval_source,
    )
    train_latent, latent_artifact, latent_report = mstc.fit_latent_state_probs(
        train_forecast,
        n_states=int(latent_states),
    )
    eval_latent = mstc.transform_latent_state_probs(eval_forecast, latent_artifact)
    states = {
        "observed": (
            train_observed,
            eval_observed,
            [c for c in train_observed.columns if c != "timestamp"],
        ),
        "forecast": (
            train_forecast,
            eval_forecast,
            [c for c in train_forecast.columns if c != "timestamp"],
        ),
        "latent": (
            train_latent,
            eval_latent,
            [c for c in train_latent.columns if c != "timestamp"],
        ),
    }
    state_frame_validation = {
        level: {
            "train": mstc.state_frame_contract_report(train_state, context=f"train_{level}"),
            "eval": mstc.state_frame_contract_report(eval_state, context=f"eval_{level}"),
        }
        for level, (train_state, eval_state, _cols) in states.items()
    }
    return {
        "candidate_feature_cols": candidate_feature_cols,
        "feature_store_cols": feature_store_cols,
        "feature_store_eligible_symbols": list(
            (train_fs_report.get("universe_contract") or {}).get("eligible_symbols") or []
        ),
        "feature_store_tail_reference_quantiles": dict(
            train_fs_report.get("tail_reference_quantiles") or {}
        ),
        "train_candidate_agg": train_candidate_agg,
        "eval_candidate_agg": eval_candidate_agg,
        "train_state_source": train_source,
        "eval_state_source": eval_source,
        "states": states,
        "reports": {
            "feature_store": {
                "selected_column_count": int(len(feature_store_cols)),
                "selected_columns": feature_store_cols,
                "train": train_fs_report,
                "eval": eval_fs_report,
            },
            "market_state_source": {
                "train": train_source_report,
                "eval": eval_source_report,
            },
            "state_frame_validation": state_frame_validation,
            "observed_axis_encoder": {
                "mode": observed_axis_encoder.get("mode"),
                "contract": observed_axis_encoder.get("contract"),
                "fit_rows": observed_axis_encoder.get("fit_rows"),
                "fit_timestamp_min": observed_axis_encoder.get("fit_timestamp_min"),
                "fit_timestamp_max": observed_axis_encoder.get("fit_timestamp_max"),
                "minimum_input_coverage": observed_axis_encoder.get("minimum_input_coverage"),
                "axis_count": int(len(observed_axis_encoder.get("axes", {}))),
                "source_column_count": int(len(observed_axis_encoder.get("column_refs", {}))),
                "ret_col": observed_axis_encoder.get("ret_col"),
                "transition_column_count": int(
                    len((observed_axis_encoder.get("transition", {}) or {}).get("columns", []) or [])
                ),
                "reliability_column_count": int(
                    len((observed_axis_encoder.get("reliability", {}) or {}).get("columns", []) or [])
                ),
                "reliability_mode": (observed_axis_encoder.get("reliability", {}) or {}).get("mode"),
                "reliability_train_novelty_q95": (observed_axis_encoder.get("reliability", {}) or {}).get(
                    "train_novelty_q95"
                ),
                "reliability_train_drift_q95": (observed_axis_encoder.get("reliability", {}) or {}).get(
                    "train_drift_q95"
                ),
                "low_input_coverage_fail_closed": bool(
                    "state_low_input_coverage" in dict(observed_axis_encoder.get("axis_sources", {}) or {})
                ),
                "source_validation_train_present": bool(
                    dict(observed_axis_encoder.get("source_validation", {}) or {}).get("train")
                ),
            },
            "axis_sources": axis_sources,
            "forecast_report": forecast_report,
            "latent_report": latent_report,
        },
        "observed_axis_encoder": observed_axis_encoder,
        "forecast_artifact": forecast_artifact,
        "latent_artifact": latent_artifact,
    }


def _make_manifest(
    *,
    args: argparse.Namespace,
    selected_arm: str,
    selected_payload: dict[str, Any],
    state_spec: dict[str, Any],
    state_artifacts: dict[str, Any],
    response_feature_cols: list[str],
    response_report: dict[str, Any],
    activation_report: dict[str, Any],
    state_activation_filter: dict[str, Any],
    controller_execution_enabled: bool,
    walkforward_config: dict[str, Any],
    forecast_model_kind_report: dict[str, Any],
    response_model_kind_report: dict[str, Any],
    runtime_param_resolution: dict[str, Any],
    state_join_validation: dict[str, Any],
    bundle_path: Path,
    outputs: dict[str, str],
) -> dict[str, Any]:
    disabled_heads = mstc._parse_disabled_heads(args.disable_heads)
    controller_enabled_manifest = mstc._controller_enabled_heads_manifest(
        mstc._parse_enabled_heads(args.controller_enabled_heads),
        disabled_heads,
    )
    shadow_controller_enabled_manifest = mstc._controller_enabled_heads_manifest(
        mstc._parse_enabled_heads(args.controller_enabled_heads),
        disabled_heads,
    )
    if not controller_execution_enabled:
        controller_enabled_manifest = {
            "controller_enabled_scope": "disabled_by_activation_registry",
            "controller_enabled_heads": [],
            "controller_enabled_heads_ignored_inactive": [],
        }
    if not bool(getattr(args, "shadow_controller_only", False)) or not bool(
        state_activation_filter.get("active_state_feature_columns", [])
    ):
        shadow_controller_enabled_manifest = {
            "shadow_controller_enabled_scope": "disabled_by_activation_registry",
            "shadow_controller_enabled_heads": [],
            "shadow_controller_enabled_heads_ignored_inactive": [],
        }
    else:
        shadow_controller_enabled_manifest = {
            "shadow_controller_enabled_scope": shadow_controller_enabled_manifest[
                "controller_enabled_scope"
            ],
            "shadow_controller_enabled_heads": shadow_controller_enabled_manifest[
                "controller_enabled_heads"
            ],
            "shadow_controller_enabled_heads_ignored_inactive": shadow_controller_enabled_manifest.get(
                "controller_enabled_heads_ignored_inactive",
                [],
            ),
        }
    source_contract_audit = _source_contract_audit(state_artifacts)
    return {
        "generated_by": "materialize_market_state_controller_bundle",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_arm": selected_arm,
        "selected_controller": selected_payload,
        "state_level": state_spec["state_level"],
        "per_strategy_residual": bool(state_spec["per_strategy_residual"]),
        "train_broad_candidates": str(args.train_broad_candidates),
        "train_broad_candidates_sha256": _file_sha256(args.train_broad_candidates),
        "train_deployable_candidates": str(args.train_deployable_candidates),
        "train_deployable_candidates_sha256": _file_sha256(args.train_deployable_candidates),
        "eval_candidates": str(args.eval_candidates) if args.eval_candidates else None,
        "eval_candidates_sha256": _file_sha256(args.eval_candidates) if args.eval_candidates else None,
        "policy_manifest": str(args.policy_manifest),
        "policy_manifest_sha256": _file_sha256(args.policy_manifest),
        "policy_variant": str(args.policy_variant),
        "rank_contract": str(args.rank_contract),
        "rank_reference_run_id": str(getattr(args, "rank_reference_run_id", mstc.DEFAULT_RANK_REFERENCE_RUN_ID)),
        "data_root": str(getattr(args, "data_root", mstc.DEFAULT_DATA_ROOT)),
        "disabled_heads": sorted(disabled_heads),
        "active_heads": mstc._active_heads(disabled_heads),
        **controller_enabled_manifest,
        **shadow_controller_enabled_manifest,
        "feature_store": state_artifacts["reports"]["feature_store"],
        "feature_store_tail_reference": {
            "role": "fit_on_training_timestamps",
            "source": "train_feature_store_aggregates",
            "quantile_count": int(len(state_artifacts.get("feature_store_tail_reference_quantiles", {}) or {})),
        },
        "market_state_source": state_artifacts["reports"]["market_state_source"],
        "source_contract_audit": source_contract_audit,
        "observed_axis_encoder": state_artifacts["reports"].get("observed_axis_encoder", {}),
        "axis_sources": state_artifacts["reports"]["axis_sources"],
        "forecast_report": state_artifacts["reports"]["forecast_report"],
        "latent_report": state_artifacts["reports"]["latent_report"],
        "state_frame_validation": state_artifacts["reports"].get("state_frame_validation", {}),
        "state_join_validation": state_join_validation,
        "activation_registry": activation_report,
        "state_activation_filter": state_activation_filter,
        "controller_execution_enabled": bool(controller_execution_enabled),
        "shadow_controller_only": bool(getattr(args, "shadow_controller_only", False)),
        "walkforward_manifest": walkforward_config,
        "forecast_model_kind": str(forecast_model_kind_report.get("value")),
        "forecast_model_kind_resolution": forecast_model_kind_report,
        "response_model_kind": str(response_model_kind_report.get("value")),
        "response_model_kind_resolution": response_model_kind_report,
        "runtime_param_resolution": runtime_param_resolution,
        "state_feature_count": int(len(state_activation_filter.get("active_state_feature_columns", []))),
        "state_feature_columns": list(state_activation_filter.get("active_state_feature_columns", [])),
        "response_feature_count": int(len(response_feature_cols)),
        "response_feature_columns": response_feature_cols,
        "response_report": response_report,
        "controller": {
            "penalty_only": True,
            "execution_enabled": bool(controller_execution_enabled),
            "controller_no_backfill_overlay": bool(
                state_spec.get("controller_no_backfill_overlay", False)
            ),
            "shadow_controller_only": bool(getattr(args, "shadow_controller_only", False)),
            "forecast_model_kind": str(forecast_model_kind_report.get("value")),
            "response_model_kind": str(response_model_kind_report.get("value")),
            "threshold_delta_max": float(args.threshold_delta_max),
            "max_threshold_up_step": float(args.max_threshold_up_step),
            "threshold_relax_alpha": float(args.threshold_relax_alpha),
            "controller_mode": str(args.controller_mode),
            "controller_min_lcb_utility": float(args.controller_min_lcb_utility),
            "controller_min_prediction_coverage": float(args.controller_min_prediction_coverage),
            "controller_min_usable_candidates": int(args.controller_min_usable_candidates),
            "controller_min_frontier_candidates": int(
                getattr(
                    args,
                    "controller_min_frontier_candidates",
                    MATERIALIZER_RUNTIME_DEFAULTS["controller_min_frontier_candidates"],
                )
            ),
            "controller_max_state_ood_score": (
                float(args.controller_max_state_ood_score)
                if args.controller_max_state_ood_score is not None
                else None
            ),
            "controller_min_action_edge": float(args.controller_min_action_edge),
            "controller_winner_sacrifice_multiplier": float(args.controller_winner_sacrifice_multiplier),
            "controller_min_removed_full_sl": float(
                getattr(
                    args,
                    "controller_min_removed_full_sl",
                    MATERIALIZER_RUNTIME_DEFAULTS["controller_min_removed_full_sl"],
                )
            ),
            "controller_max_removed_timeout": float(
                getattr(
                    args,
                    "controller_max_removed_timeout",
                    MATERIALIZER_RUNTIME_DEFAULTS["controller_max_removed_timeout"],
                )
            ),
            "use_timeout_cap": bool(args.use_timeout_cap),
            "changes_scores_or_ranks": False,
            "changes_auction_ordering": False,
        },
        "bundle_path": str(bundle_path),
        "outputs": outputs,
    }


def _make_market_state_feature_contract(
    *,
    args: argparse.Namespace,
    selected_arm: str,
    state_spec: dict[str, Any],
    state_artifacts: dict[str, Any],
    response_feature_cols: list[str],
    activation_report: dict[str, Any],
    state_activation_filter: dict[str, Any],
    controller_execution_enabled: bool,
    walkforward_config: dict[str, Any],
    forecast_model_kind_report: dict[str, Any],
    response_model_kind_report: dict[str, Any],
    runtime_param_resolution: dict[str, Any] | None = None,
    state_join_validation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    feature_store_report = dict(state_artifacts["reports"]["feature_store"])
    train_fs = dict(feature_store_report.get("train") or {})
    eval_fs = dict(feature_store_report.get("eval") or {})
    source_contract_audit = _source_contract_audit(state_artifacts)
    disabled_heads = mstc._parse_disabled_heads(args.disable_heads)
    controller_enabled_manifest = mstc._controller_enabled_heads_manifest(
        mstc._parse_enabled_heads(args.controller_enabled_heads),
        disabled_heads,
    )
    shadow_controller_enabled_manifest = mstc._controller_enabled_heads_manifest(
        mstc._parse_enabled_heads(args.controller_enabled_heads),
        disabled_heads,
    )
    if not controller_execution_enabled:
        controller_enabled_manifest = {
            "controller_enabled_scope": "disabled_by_activation_registry",
            "controller_enabled_heads": [],
            "controller_enabled_heads_ignored_inactive": [],
        }
    if not bool(getattr(args, "shadow_controller_only", False)) or not bool(
        state_activation_filter.get("active_state_feature_columns", [])
    ):
        shadow_controller_enabled_manifest = {
            "shadow_controller_enabled_scope": "disabled_by_activation_registry",
            "shadow_controller_enabled_heads": [],
            "shadow_controller_enabled_heads_ignored_inactive": [],
        }
    else:
        shadow_controller_enabled_manifest = {
            "shadow_controller_enabled_scope": shadow_controller_enabled_manifest[
                "controller_enabled_scope"
            ],
            "shadow_controller_enabled_heads": shadow_controller_enabled_manifest[
                "controller_enabled_heads"
            ],
            "shadow_controller_enabled_heads_ignored_inactive": shadow_controller_enabled_manifest.get(
                "controller_enabled_heads_ignored_inactive",
                [],
            ),
        }
    return {
        "contract_version": "market_state_feature_contract_v1",
        "generated_by": "materialize_market_state_controller_bundle",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_arm": selected_arm,
        "state_level": state_spec["state_level"],
        "per_strategy_residual": bool(state_spec["per_strategy_residual"]),
        "rank_contract": str(args.rank_contract),
        "rank_reference_run_id": str(getattr(args, "rank_reference_run_id", mstc.DEFAULT_RANK_REFERENCE_RUN_ID)),
        "data_root": str(getattr(args, "data_root", mstc.DEFAULT_DATA_ROOT)),
        "disabled_heads": sorted(disabled_heads),
        "active_heads": mstc._active_heads(disabled_heads),
        **controller_enabled_manifest,
        **shadow_controller_enabled_manifest,
        "activation_registry": activation_report,
        "state_activation_filter": state_activation_filter,
        "controller_execution_enabled": bool(controller_execution_enabled),
        "shadow_controller_only": bool(getattr(args, "shadow_controller_only", False)),
        "walkforward_manifest": walkforward_config,
        "forecast_model_kind": str(forecast_model_kind_report.get("value")),
        "forecast_model_kind_resolution": forecast_model_kind_report,
        "response_model_kind": str(response_model_kind_report.get("value")),
        "response_model_kind_resolution": response_model_kind_report,
        "runtime_param_resolution": runtime_param_resolution or {},
        "state_frame_validation": state_artifacts["reports"].get("state_frame_validation", {}),
        "state_join_validation": state_join_validation or {},
        "source_contract_audit": source_contract_audit,
        "invariants": {
            "one_market_state_row_per_timestamp": True,
            "state_join_timestamp_constant": True,
            "market_state_uses_strategy_ids": False,
            "market_state_uses_model_predictions": False,
            "market_state_uses_ranks": False,
            "market_state_uses_candidate_counts": False,
            "market_state_uses_portfolio_pnl": False,
            "market_state_uses_realized_strategy_outcomes": False,
            "actual_order_book_features_allowed": False,
            "candidate_population_fallback_enabled": bool(args.allow_candidate_state_fallback),
            "candidate_population_fallback_is_production_safe": False,
            "controller_changes_scores_or_ranks": False,
            "controller_changes_auction_ordering": False,
            "controller_can_lower_thresholds": False,
            "latent_gmm_active_controller_input": False,
        },
        "source_schema": {
            "candidate_feature_columns": list(state_artifacts["candidate_feature_cols"]),
            "feature_store_columns": list(state_artifacts["feature_store_cols"]),
            "state_feature_columns": list(state_activation_filter.get("active_state_feature_columns", [])),
            "response_feature_columns": list(response_feature_cols),
        },
        "feature_store_tail_reference": {
            "role": "fit_on_training_timestamps",
            "source": "train_feature_store_aggregates",
            "quantile_count": int(len(state_artifacts.get("feature_store_tail_reference_quantiles", {}) or {})),
            "quantiles": state_artifacts.get("feature_store_tail_reference_quantiles", {}),
        },
        "feature_store": {
            "selected_column_count": int(len(state_artifacts["feature_store_cols"])),
            "selected_columns": list(state_artifacts["feature_store_cols"]),
            "train_universe_contract": train_fs.get("universe_contract"),
            "eval_universe_contract": eval_fs.get("universe_contract"),
            "train_timestamp_coverage": train_fs.get("timestamp_coverage"),
            "eval_timestamp_coverage": eval_fs.get("timestamp_coverage"),
            "train_symbols_read": train_fs.get("symbols_read"),
            "eval_symbols_read": eval_fs.get("symbols_read"),
        },
        "market_state_source": state_artifacts["reports"]["market_state_source"],
        "observed_axis_encoder": state_artifacts["reports"].get("observed_axis_encoder", {}),
        "axis_sources": state_artifacts["reports"]["axis_sources"],
        "forecast_report": state_artifacts["reports"]["forecast_report"],
        "latent_report": state_artifacts["reports"]["latent_report"],
    }


def _make_market_state_universe_contract(state_artifacts: dict[str, Any]) -> dict[str, Any]:
    reports = dict(state_artifacts.get("reports") or {})
    feature_store = dict(reports.get("feature_store") or {})
    market_state_source = dict(reports.get("market_state_source") or {})
    payload = mstc._standalone_market_state_universe_contract(
        train_fs_report=dict(feature_store.get("train") or {}),
        eval_fs_report=dict(feature_store.get("eval") or {}),
        train_source_report=dict(market_state_source.get("train") or {}),
        eval_source_report=dict(market_state_source.get("eval") or {}),
    )
    payload["generated_by"] = "materialize_market_state_controller_bundle"
    return payload


def _source_contract_audit(state_artifacts: dict[str, Any]) -> dict[str, Any]:
    reports = state_artifacts.get("reports", {})
    market_state_source = reports.get("market_state_source", {})
    feature_store = reports.get("feature_store", {})
    splits: dict[str, Any] = {}
    overall_passed = True
    for split, source_report_raw in dict(market_state_source or {}).items():
        source_report = dict(source_report_raw or {})
        validation = dict(source_report.get("validation") or {})
        fs_report = dict((feature_store or {}).get(split) or {})
        forbidden_removed = list(source_report.get("forbidden_candidate_aggregate_columns_removed") or [])
        validation_forbidden_count = int(validation.get("forbidden_column_count") or 0)
        timestamp_unique = bool(validation.get("timestamp_unique") is True)
        market_wide = bool(validation.get("market_wide_one_row_per_timestamp") is True)
        production_safe = bool(source_report.get("production_safe") is True)
        candidate_fallback = bool(source_report.get("allow_candidate_fallback") is True)
        split_passed = (
            validation_forbidden_count == 0
            and timestamp_unique
            and market_wide
            and production_safe
            and not candidate_fallback
        )
        overall_passed = bool(overall_passed and split_passed)
        splits[str(split)] = {
            "source": source_report.get("source"),
            "production_safe": production_safe,
            "candidate_fallback_enabled": candidate_fallback,
            "feature_count": int(source_report.get("feature_count") or 0),
            "feature_store_feature_count": int(source_report.get("feature_store_aggregate_feature_count") or 0),
            "candidate_aggregate_feature_count": int(source_report.get("candidate_aggregate_feature_count") or 0),
            "forbidden_candidate_aggregate_columns_removed_count": int(len(forbidden_removed)),
            "forbidden_candidate_aggregate_columns_removed_sample": forbidden_removed[:20],
            "validation_forbidden_column_count": validation_forbidden_count,
            "timestamp_unique": timestamp_unique,
            "market_wide_one_row_per_timestamp": market_wide,
            "row_count": int(validation.get("row_count") or 0),
            "feature_store_enabled": bool(fs_report.get("enabled") is True),
            "feature_store_timestamp_coverage": fs_report.get("timestamp_coverage"),
            "feature_store_symbols_read": fs_report.get("symbols_read"),
            "passed": split_passed,
        }
    return {
        "audit_version": "market_state_source_contract_audit_v1",
        "overall_passed": bool(overall_passed and bool(splits)),
        "required_source": "feature_store_market_aggregates",
        "forbidden_inputs": [
            "strategy/model/rank/candidate-population fields",
            "portfolio PnL or accepted-trade fields",
            "realized strategy outcomes or labels",
            "actual order-book spread/depth/imbalance/microprice fields",
        ],
        "actual_order_book_features_allowed": False,
        "candidate_population_fallback_allowed_for_production": False,
        "splits": splits,
    }


def _render_report(manifest: dict[str, Any], summary: pd.DataFrame, by_head: pd.DataFrame) -> str:
    lines = [
        "# Market-State Controller Bundle",
        "",
        f"Generated: {manifest['generated_at_utc']}",
        "",
        f"Selected arm: `{manifest['selected_arm']}`",
        f"State level: `{manifest['state_level']}`",
        f"Rank contract: `{manifest['rank_contract']}`",
        f"Disabled heads: `{', '.join(manifest['disabled_heads']) or 'none'}`",
        f"Active heads: `{', '.join(manifest.get('active_heads', [])) or 'none'}`",
        (
            "Controller-enabled heads: "
            f"`{', '.join(manifest.get('controller_enabled_heads', [])) or 'none'}` "
            f"({manifest.get('controller_enabled_scope', 'unknown')})"
        ),
        "",
        "## Contract",
        "",
        "- One global state row per timestamp from feature-store market aggregates by default.",
        "- Response targets are residual utility, excess full-SL risk and excess timeout risk.",
        "- Threshold control is penalty-only and does not change scores, ranks or auction ordering.",
        "- Missing or OOD state inputs fail closed to the base threshold.",
        "",
        "## Eval Replay Summary",
        "",
        summary.to_markdown(index=False) if not summary.empty else "_No eval replay was requested._",
        "",
        "## Eval Replay By Head",
        "",
        by_head.to_markdown(index=False) if not by_head.empty else "_No accepted eval trades._",
        "",
    ]
    return "\n".join(lines) + "\n"


def _controller_config_payload(
    *,
    selected_arm: str,
    selected_payload: dict[str, Any],
    state_spec: dict[str, Any],
    feature_contract: dict[str, Any],
    bundle_path: Path | None,
    controller_params: dict[str, Any],
    forecast_model_kind_report: dict[str, Any],
    response_model_kind_report: dict[str, Any],
    runtime_param_resolution: dict[str, Any],
) -> dict[str, Any]:
    return {
        "config_version": "strategy_threshold_controller_config_v1",
        "generated_by": "materialize_market_state_controller_bundle",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_arm": selected_arm,
        "selected_controller": selected_payload,
        "state_spec": state_spec,
        "rank_contract": feature_contract.get("rank_contract"),
        "disabled_heads": feature_contract.get("disabled_heads"),
        "active_heads": feature_contract.get("active_heads"),
        "controller_execution_enabled": feature_contract.get("controller_execution_enabled"),
        "controller_enabled_heads": feature_contract.get("controller_enabled_heads"),
        "controller_enabled_scope": feature_contract.get("controller_enabled_scope"),
        "controller_params": controller_params,
        "forecast_model_kind": forecast_model_kind_report.get("value"),
        "forecast_model_kind_resolution": forecast_model_kind_report,
        "response_model_kind": response_model_kind_report.get("value"),
        "response_model_kind_resolution": response_model_kind_report,
        "runtime_param_resolution": runtime_param_resolution,
        "invariants": {
            "penalty_only": True,
            "controller_no_backfill_overlay": bool(
                state_spec.get("controller_no_backfill_overlay", False)
            ),
            "controller_can_lower_thresholds": False,
            "controller_changes_scores_or_ranks": False,
            "controller_changes_auction_ordering": False,
            "latent_gmm_active_controller_input": False,
        },
        "bundle_path": str(bundle_path) if bundle_path is not None else None,
    }


def _market_state_training_reference_payload(
    *,
    selected_arm: str,
    state_spec: dict[str, Any],
    state_artifacts: dict[str, Any],
    feature_contract: dict[str, Any],
    forecast_model_kind_report: dict[str, Any],
    response_model_kind_report: dict[str, Any],
    runtime_param_resolution: dict[str, Any],
) -> dict[str, Any]:
    return {
        "reference_version": "market_state_training_reference_v1",
        "generated_by": "materialize_market_state_controller_bundle",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_arm": selected_arm,
        "state_spec": state_spec,
        "rank_contract": feature_contract.get("rank_contract"),
        "disabled_heads": feature_contract.get("disabled_heads"),
        "active_heads": feature_contract.get("active_heads"),
        "source_schema": feature_contract.get("source_schema"),
        "feature_store_tail_reference": feature_contract.get("feature_store_tail_reference"),
        "feature_store": feature_contract.get("feature_store"),
        "market_state_source": feature_contract.get("market_state_source"),
        "source_contract_audit": feature_contract.get("source_contract_audit"),
        "state_frame_validation": feature_contract.get("state_frame_validation"),
        "state_join_validation": feature_contract.get("state_join_validation"),
        "observed_axis_encoder": state_artifacts.get("observed_axis_encoder"),
        "forecast_artifact": state_artifacts.get("forecast_artifact"),
        "latent_artifact": state_artifacts.get("latent_artifact"),
        "axis_sources": state_artifacts["reports"].get("axis_sources", {}),
        "forecast_report": state_artifacts["reports"].get("forecast_report", {}),
        "latent_report": state_artifacts["reports"].get("latent_report", {}),
        "forecast_model_kind_resolution": forecast_model_kind_report,
        "response_model_kind_resolution": response_model_kind_report,
        "runtime_param_resolution": runtime_param_resolution,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-controller", type=Path, default=DEFAULT_SELECTED_CONTROLLER)
    parser.add_argument(
        "--walkforward-manifest",
        type=Path,
        default=None,
        help=(
            "Optional walk-forward manifest.json. Defaults to manifest.json next to "
            "--selected-controller and is used to inherit forecast/response backend choices."
        ),
    )
    parser.add_argument(
        "--activation-registry",
        type=Path,
        default=None,
        help=(
            "Optional market_state_activation_registry.csv. If omitted, the script "
            "looks next to --selected-controller. Active controller state columns are "
            "restricted to rows with recommended_status=active_candidate."
        ),
    )
    parser.add_argument(
        "--use-all-state-heads-without-activation-registry",
        action="store_true",
        default=False,
        help=(
            "Research/debug override. If no activation registry is available, "
            "use all generated state heads instead of failing closed to a "
            "disabled controller."
        ),
    )
    parser.add_argument("--selected-arm-default", default="S1_observed_axes_shared_response")
    parser.add_argument(
        "--allow-selected-arm-default",
        action="store_true",
        default=False,
        help=(
            "Research/debug override: materialize --selected-arm-default when the "
            "selection file is missing or contains selected_arm=null. Production "
            "materialization should leave this disabled."
        ),
    )
    parser.add_argument(
        "--allow-null-selection-noop",
        action="store_true",
        default=True,
        help=(
            "Production-safe path for a rejected/null walk-forward selection: "
            "materialize a fail-closed audit bundle with threshold actions disabled."
        ),
    )
    parser.add_argument(
        "--no-allow-null-selection-noop",
        dest="allow_null_selection_noop",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--shadow-selected-arm",
        default="",
        help=(
            "Materialize this controller arm for Stage-2 shadow logging when "
            "the selected-controller file rejected/null-selected every arm. "
            "The arm is fitted and proposed schedules are persisted, but "
            "controller execution remains disabled."
        ),
    )
    parser.add_argument(
        "--shadow-controller-only",
        action="store_true",
        default=False,
        help=(
            "Fit and persist controller state/response artifacts for shadow "
            "schedules only. Applied thresholds remain base/T1 thresholds."
        ),
    )
    parser.add_argument(
        "--use-all-state-heads-for-shadow",
        action="store_true",
        default=False,
        help=(
            "Stage-2 monitoring override: when --shadow-controller-only is set, "
            "retain the selected arm's state columns even if the activation "
            "registry marks them disabled for production. Controller execution "
            "remains disabled and the override is persisted in the feature "
            "contract."
        ),
    )
    parser.add_argument("--train-broad-candidates", type=Path, default=mstc.DEFAULT_TRAIN_BROAD)
    parser.add_argument("--train-deployable-candidates", type=Path, default=mstc.DEFAULT_TRAIN_DEPLOYABLE)
    parser.add_argument("--eval-candidates", type=Path, default=mstc.DEFAULT_EVAL_CANDIDATES)
    parser.add_argument("--train-feature-store-dir", type=Path, default=mstc.DEFAULT_TRAIN_FEATURE_STORE)
    parser.add_argument("--eval-feature-store-dir", type=Path, default=mstc.DEFAULT_EVAL_FEATURE_STORE)
    parser.add_argument("--policy-manifest", type=Path, default=mstc.DEFAULT_POLICY_MANIFEST)
    parser.add_argument("--policy-variant", default="refit_bar4_strategy_bar2")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--data-root", type=Path, default=mstc.DEFAULT_DATA_ROOT)
    parser.add_argument("--rank-reference-run-id", default=mstc.DEFAULT_RANK_REFERENCE_RUN_ID)
    parser.add_argument(
        "--rank-contract",
        choices=("strict", "short_boll_timestamp_rank", "anchor_global_policy_rank_reference"),
        default="anchor_global_policy_rank_reference",
    )
    parser.add_argument("--disable-heads", default="long_bars,long_dist")
    parser.add_argument(
        "--controller-enabled-heads",
        default="",
        help="Comma-separated heads for threshold action. Empty means all active heads.",
    )
    parser.add_argument("--max-feature-cols", type=int, default=128)
    parser.add_argument("--max-feature-store-cols", type=int, default=96)
    parser.add_argument("--feature-store-symbol-cap", type=int, default=220)
    parser.add_argument("--allow-candidate-state-fallback", action="store_true", default=False)
    parser.add_argument("--forecast-horizons-steps", default="6,24")
    parser.add_argument(
        "--forecast-model-kind",
        choices=("auto", "lightgbm", "xgboost"),
        default="auto",
        help="Forecast backend. auto inherits the walk-forward manifest, falling back to lightgbm.",
    )
    parser.add_argument("--latent-states", type=int, default=4)
    parser.add_argument("--max-response-rows", type=int, default=6000)
    parser.add_argument("--max-response-keyword-cols", type=int, default=24)
    parser.add_argument(
        "--response-model-kind",
        choices=("auto", "additive_ebm", "hist_gradient_boosting", "xgboost"),
        default="auto",
        help="Strategy-response backend. auto inherits the walk-forward manifest, falling back to additive_ebm.",
    )
    parser.add_argument(
        "--no-inherit-walkforward-controller-config",
        dest="inherit_walkforward_controller_config",
        action="store_false",
        default=True,
        help=(
            "Use materializer CLI/default controller and response-weighting params "
            "instead of inheriting values from the walk-forward manifest when CLI "
            "values are still at materializer defaults."
        ),
    )
    parser.add_argument("--response-frontier-weight-gamma", type=float, default=3.0)
    parser.add_argument("--response-frontier-weight-bandwidth", type=float, default=0.06)
    parser.add_argument("--response-balance-timestamps", action="store_true", default=True)
    parser.add_argument("--no-response-balance-timestamps", dest="response_balance_timestamps", action="store_false")
    parser.add_argument("--response-balance-strategies", action="store_true", default=True)
    parser.add_argument("--no-response-balance-strategies", dest="response_balance_strategies", action="store_false")
    parser.add_argument("--threshold-delta-max", type=float, default=0.10)
    parser.add_argument("--max-threshold-up-step", type=float, default=0.03)
    parser.add_argument("--threshold-relax-alpha", type=float, default=0.25)
    parser.add_argument(
        "--controller-mode",
        choices=(
            "rank_grid",
            "action_aware_rank_grid",
            "frontier_rank_grid",
            "frontier_action_rank_grid",
            "accepted_frontier_action_rank_grid",
            "severity",
        ),
        default="rank_grid",
    )
    parser.add_argument("--controller-min-lcb-utility", type=float, default=0.0)
    parser.add_argument("--controller-min-prediction-coverage", type=float, default=0.80)
    parser.add_argument("--controller-min-usable-candidates", type=int, default=1)
    parser.add_argument("--controller-min-frontier-candidates", type=int, default=1)
    parser.add_argument("--controller-max-state-ood-score", type=float, default=None)
    parser.add_argument("--controller-min-action-edge", type=float, default=0.0)
    parser.add_argument("--controller-min-removed-full-sl", type=float, default=0.0)
    parser.add_argument("--controller-max-removed-timeout", type=float, default=1.0)
    parser.add_argument("--controller-winner-sacrifice-multiplier", type=float, default=1.0)
    parser.add_argument("--enable-timeout-cap", dest="use_timeout_cap", action="store_true", default=False)
    parser.add_argument("--market-mode", default="perps")
    parser.add_argument("--skip-eval-replay", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_arm, selected_payload = _load_selected_arm(
        args.selected_controller,
        args.selected_arm_default,
        allow_default=bool(args.allow_selected_arm_default),
        allow_null_noop=bool(args.allow_null_selection_noop),
    )
    shadow_selected_arm = str(args.shadow_selected_arm or "").strip()
    if shadow_selected_arm:
        selected_arm, selected_payload = _apply_shadow_selected_arm_override(
            selected_arm=selected_arm,
            selected_payload=selected_payload,
            shadow_selected_arm=shadow_selected_arm,
        )
        args.shadow_controller_only = True
    state_spec = _arm_state_spec(selected_arm)
    disabled_heads = mstc._parse_disabled_heads(args.disable_heads)
    controller_enabled_heads = mstc._parse_enabled_heads(args.controller_enabled_heads)
    walkforward_manifest_path = _infer_walkforward_manifest_path(args.selected_controller, args.walkforward_manifest)
    walkforward_config = _load_walkforward_controller_config(walkforward_manifest_path)
    runtime_param_resolution = _apply_walkforward_runtime_defaults(args, walkforward_config)
    forecast_model_kind, forecast_model_kind_report = _resolve_backend_kind(
        args.forecast_model_kind,
        walkforward_config,
        "forecast_model_kind",
        "lightgbm",
        {"lightgbm", "xgboost"},
    )
    response_model_kind, response_model_kind_report = _resolve_backend_kind(
        args.response_model_kind,
        walkforward_config,
        "response_model_kind",
        "additive_ebm",
        {"additive_ebm", "hist_gradient_boosting", "xgboost"},
    )
    activation_registry_path = _infer_activation_registry_path(args.selected_controller, args.activation_registry)
    _activation_registry, activation_report = _load_activation_registry(activation_registry_path)

    train_broad = mstc._disable_heads(
        mstc._apply_rank_contract(
            mstc._load_candidates(args.train_broad_candidates),
            args.rank_contract,
            data_root=args.data_root,
            rank_reference_run_id=args.rank_reference_run_id,
        ),
        disabled_heads,
    )
    train_deployable = mstc._disable_heads(
        mstc._apply_rank_contract(
            mstc._load_candidates(args.train_deployable_candidates),
            args.rank_contract,
            data_root=args.data_root,
            rank_reference_run_id=args.rank_reference_run_id,
        ),
        disabled_heads,
    )
    eval_candidates = None
    if not args.skip_eval_replay and args.eval_candidates is not None:
        eval_candidates = mstc._disable_heads(
            mstc._apply_rank_contract(
                mstc._load_candidates(args.eval_candidates),
                args.rank_contract,
                data_root=args.data_root,
                rank_reference_run_id=args.rank_reference_run_id,
            ),
            disabled_heads,
        )

    state_artifacts = _build_state_artifacts(
        train_broad,
        eval_candidates,
        train_feature_store_dir=args.train_feature_store_dir,
        eval_feature_store_dir=args.eval_feature_store_dir,
        max_feature_cols=int(args.max_feature_cols),
        max_feature_store_cols=int(args.max_feature_store_cols),
        feature_store_symbol_cap=int(args.feature_store_symbol_cap),
        allow_candidate_state_fallback=bool(args.allow_candidate_state_fallback),
        forecast_horizons_steps=mstc._parse_int_grid(args.forecast_horizons_steps, (6, 24)),
        forecast_model_kind=forecast_model_kind,
        latent_states=int(args.latent_states),
    )
    state_level = state_spec["state_level"]
    train_state, eval_state, state_cols = state_artifacts["states"][state_level]
    raw_state_cols = list(state_cols)
    shadow_controller_only = bool(args.shadow_controller_only)
    if bool(state_spec.get("controller_noop", False)):
        state_cols = []
        state_activation_filter = {
            "enforced": True,
            "reason": "selected_controller_rejected_noop",
            "input_state_feature_count": int(len(raw_state_cols)),
            "active_state_feature_count": 0,
            "dropped_state_feature_count": int(len(raw_state_cols)),
            "active_state_feature_columns": [],
            "dropped_state_feature_columns": raw_state_cols,
        }
    elif shadow_controller_only and bool(args.use_all_state_heads_for_shadow):
        state_cols = list(raw_state_cols)
        state_activation_filter = {
            "enforced": False,
            "reason": "shadow_controller_all_state_heads_override",
            "input_state_feature_count": int(len(raw_state_cols)),
            "active_state_feature_count": int(len(raw_state_cols)),
            "dropped_state_feature_count": 0,
            "active_state_feature_columns": list(raw_state_cols),
            "dropped_state_feature_columns": [],
            "production_execution_allowed": False,
        }
    else:
        state_cols, state_activation_filter = _filter_state_columns_by_activation_registry(
            state_cols,
            activation_report,
            fail_closed_when_unavailable=not bool(
                args.use_all_state_heads_without_activation_registry
            ),
        )
    controller_execution_enabled = (
        len(state_cols) > 0
        and not bool(state_spec.get("controller_noop", False))
        and not shadow_controller_only
    )
    _validate_candidate_state_fallback_execution_contract(
        allow_candidate_state_fallback=bool(args.allow_candidate_state_fallback),
        controller_execution_enabled=controller_execution_enabled,
        shadow_controller_only=shadow_controller_only,
    )
    _validate_state_reference_materialization_contract(
        state_cols=list(state_cols),
        state_artifacts=state_artifacts,
        controller_execution_enabled=controller_execution_enabled,
        shadow_controller_only=shadow_controller_only,
    )
    effective_controller_enabled_heads = controller_enabled_heads if controller_execution_enabled else set()
    shadow_controller_enabled_manifest = mstc._controller_enabled_heads_manifest(
        controller_enabled_heads,
        disabled_heads,
    )
    train_frame = mstc.build_response_frame(train_broad, train_state)
    state_join_validation: dict[str, Any] = {
        "train": mstc.joined_state_invariance_report(
            train_frame,
            state_cols,
            context=f"train_{state_level}",
        )
    }
    eval_frame_for_replay: pd.DataFrame | None = None
    if eval_candidates is not None:
        eval_frame_for_replay = mstc.build_response_frame(eval_candidates, eval_state)
        state_join_validation["eval"] = mstc.joined_state_invariance_report(
            eval_frame_for_replay,
            state_cols,
            context=f"eval_{state_level}",
        )
    models, response_features, response_report = mstc.fit_response_models(
        train_frame,
        state_cols,
        per_strategy_residual=bool(state_spec["per_strategy_residual"]),
        max_rows=int(args.max_response_rows),
        max_keyword_cols=int(args.max_response_keyword_cols),
        response_frontier_weight_gamma=float(args.response_frontier_weight_gamma),
        response_frontier_weight_bandwidth=float(args.response_frontier_weight_bandwidth),
        response_balance_timestamps=bool(args.response_balance_timestamps),
        response_balance_strategies=bool(args.response_balance_strategies),
        response_model_kind=response_model_kind,
    )
    feature_contract = _make_market_state_feature_contract(
        args=args,
        selected_arm=selected_arm,
        state_spec=state_spec,
        state_artifacts=state_artifacts,
        response_feature_cols=response_features,
        activation_report=activation_report,
        state_activation_filter=state_activation_filter,
        controller_execution_enabled=controller_execution_enabled,
        walkforward_config=walkforward_config,
        forecast_model_kind_report=forecast_model_kind_report,
        response_model_kind_report=response_model_kind_report,
        runtime_param_resolution=runtime_param_resolution,
        state_join_validation=state_join_validation,
    )
    controller_params = {
        "execution_enabled": bool(controller_execution_enabled),
        "shadow_controller_only": bool(shadow_controller_only),
        "controller_no_backfill_overlay": bool(state_spec.get("controller_no_backfill_overlay", False)),
        "forecast_model_kind": forecast_model_kind,
        "response_model_kind": response_model_kind,
        "threshold_delta_max": float(args.threshold_delta_max),
        "max_threshold_up_step": float(args.max_threshold_up_step),
        "threshold_relax_alpha": float(args.threshold_relax_alpha),
        "controller_mode": str(args.controller_mode),
        "controller_min_lcb_utility": float(args.controller_min_lcb_utility),
        "controller_min_prediction_coverage": float(args.controller_min_prediction_coverage),
        "controller_min_usable_candidates": int(args.controller_min_usable_candidates),
        "controller_min_frontier_candidates": int(args.controller_min_frontier_candidates),
        "controller_max_state_ood_score": (
            float(args.controller_max_state_ood_score)
            if args.controller_max_state_ood_score is not None
            else None
        ),
        "controller_min_action_edge": float(args.controller_min_action_edge),
        "controller_winner_sacrifice_multiplier": float(args.controller_winner_sacrifice_multiplier),
        "controller_min_removed_full_sl": float(args.controller_min_removed_full_sl),
        "controller_max_removed_timeout": float(args.controller_max_removed_timeout),
        "use_timeout_cap": bool(args.use_timeout_cap),
    }
    bundle = {
        "selected_arm": selected_arm,
        "selected_controller": selected_payload,
        "market_state_feature_contract": feature_contract,
        "state_spec": state_spec,
        "controller_no_backfill_overlay": bool(state_spec.get("controller_no_backfill_overlay", False)),
        "rank_contract": str(args.rank_contract),
        "rank_reference_run_id": str(args.rank_reference_run_id),
        "data_root": str(args.data_root),
        "models": models,
        "response_feature_columns": response_features,
        "state_feature_columns": state_cols,
        "activation_registry": activation_report,
        "state_activation_filter": state_activation_filter,
        "controller_execution_enabled": bool(controller_execution_enabled),
        "candidate_feature_columns": state_artifacts["candidate_feature_cols"],
        "feature_store_columns": state_artifacts["feature_store_cols"],
        "feature_store_eligible_symbols": state_artifacts.get("feature_store_eligible_symbols", []),
        "feature_store_tail_reference_quantiles": state_artifacts.get(
            "feature_store_tail_reference_quantiles",
            {},
        ),
        "axis_sources": state_artifacts["reports"]["axis_sources"],
        "observed_axis_encoder": state_artifacts["observed_axis_encoder"],
        "forecast_artifact": state_artifacts["forecast_artifact"],
        "latent_artifact": state_artifacts["latent_artifact"],
        "response_report": response_report,
        "walkforward_manifest": walkforward_config,
        "forecast_model_kind": forecast_model_kind,
        "forecast_model_kind_resolution": forecast_model_kind_report,
        "response_model_kind": response_model_kind,
        "response_model_kind_resolution": response_model_kind_report,
        "runtime_param_resolution": runtime_param_resolution,
        "state_frame_validation": state_artifacts["reports"].get("state_frame_validation", {}),
        "state_join_validation": state_join_validation,
        "rank_contract": str(args.rank_contract),
        "disabled_heads": sorted(disabled_heads),
        "active_heads": feature_contract["active_heads"],
        "shadow_controller_only": bool(shadow_controller_only),
        "shadow_controller_enabled_heads": (
            shadow_controller_enabled_manifest["controller_enabled_heads"]
            if shadow_controller_only and len(state_cols) > 0
            else []
        ),
        "shadow_controller_enabled_scope": (
            shadow_controller_enabled_manifest["controller_enabled_scope"]
            if shadow_controller_only and len(state_cols) > 0
            else "disabled_by_activation_registry"
        ),
        "controller_enabled_heads": (
            feature_contract["controller_enabled_heads"] if controller_execution_enabled else []
        ),
        "controller_enabled_scope": (
            feature_contract["controller_enabled_scope"] if controller_execution_enabled else "disabled_by_activation_registry"
        ),
        "controller_enabled_heads_ignored_inactive": feature_contract.get(
            "controller_enabled_heads_ignored_inactive",
            [],
        ),
        "controller_params": controller_params,
    }

    import joblib

    bundle_path = args.output_dir / "market_state_controller_bundle.joblib"
    joblib.dump(bundle, bundle_path)

    outputs: dict[str, str] = {
        "bundle": str(bundle_path),
        "train_state_features": str(args.output_dir / "train_market_state_features.csv"),
        "market_state_feature_contract": str(args.output_dir / "market_state_feature_contract.json"),
        "market_state_universe_contract": str(args.output_dir / "market_state_universe_contract.json"),
        "strategy_threshold_controller_config": str(args.output_dir / "strategy_threshold_controller_config.json"),
        "artifact_hashes": str(args.output_dir / "artifact_hashes.json"),
        "manifest": str(args.output_dir / "manifest.json"),
        "report": str(args.output_dir / "market_state_controller_bundle_report.md"),
    }
    controller_config = _controller_config_payload(
        selected_arm=selected_arm,
        selected_payload=selected_payload,
        state_spec=state_spec,
        feature_contract=feature_contract,
        bundle_path=bundle_path,
        controller_params=controller_params,
        forecast_model_kind_report=forecast_model_kind_report,
        response_model_kind_report=response_model_kind_report,
        runtime_param_resolution=runtime_param_resolution,
    )
    timestamp_panel = mstc.market_state_timestamp_panel(
        [
            (split, level, frame)
            for level, (train_level_state, eval_level_state, _cols) in state_artifacts["states"].items()
            for split, frame in (("train", train_level_state), ("eval", eval_level_state))
        ]
    )
    feature_coverage = mstc.market_state_feature_coverage(timestamp_panel)
    training_reference = _market_state_training_reference_payload(
        selected_arm=selected_arm,
        state_spec=state_spec,
        state_artifacts=state_artifacts,
        feature_contract=feature_contract,
        forecast_model_kind_report=forecast_model_kind_report,
        response_model_kind_report=response_model_kind_report,
        runtime_param_resolution=runtime_param_resolution,
    )
    universe_contract = _make_market_state_universe_contract(state_artifacts)
    train_state.to_csv(args.output_dir / "train_market_state_features.csv", index=False)
    timestamp_panel.to_parquet(args.output_dir / "market_state_timestamp_panel.parquet", index=False)
    feature_coverage.to_csv(args.output_dir / "market_state_feature_coverage.csv", index=False)
    joblib.dump(training_reference, args.output_dir / "market_state_training_reference.joblib")
    outputs.update(
        {
            "market_state_timestamp_panel": str(args.output_dir / "market_state_timestamp_panel.parquet"),
            "market_state_feature_coverage": str(args.output_dir / "market_state_feature_coverage.csv"),
            "market_state_training_reference": str(args.output_dir / "market_state_training_reference.joblib"),
        }
    )
    eval_summary = pd.DataFrame()
    eval_by_head = pd.DataFrame()
    if eval_candidates is not None:
        params, _ = mstc._load_policy_params(args.policy_manifest, args.policy_variant)
        ev_curve = fit_hierarchical_ev_curves(train_deployable)
        eval_frame = eval_frame_for_replay if eval_frame_for_replay is not None else mstc.build_response_frame(eval_candidates, eval_state)
        predictions = mstc.predict_response(models, eval_frame, response_features, state_cols)
        replay_candidates_base = eval_candidates
        schedule_frame = eval_frame
        schedule_predictions = predictions
        baseline_allowed_keys: set[tuple[Any, ...]] = set()
        if bool(state_spec.get("controller_no_backfill_overlay", False)):
            baseline_decisions, _, _baseline_metrics = replay_candidates(
                eval_candidates,
                params,
                mode="global_auction",
                ev_curve=ev_curve,
                market_mode=args.market_mode,
            )
            baseline_accepted = mstc._accepted_trades(eval_candidates, baseline_decisions)
            baseline_allowed_keys = mstc._accepted_key_set(baseline_accepted)
            replay_candidates_base = mstc._restrict_to_allowed_decision_keys(
                eval_candidates,
                baseline_allowed_keys,
            )
            overlay_mask = mstc._allowed_decision_key_mask(eval_frame, baseline_allowed_keys)
            schedule_frame = eval_frame.loc[overlay_mask].copy()
            schedule_predictions = predictions.loc[overlay_mask].copy()
        schedule = mstc.threshold_schedule(
            schedule_frame,
            schedule_predictions,
            models["curves"],
            delta_max=float(args.threshold_delta_max),
            max_down_step=float(args.max_threshold_up_step),
            relax_alpha=float(args.threshold_relax_alpha),
            controller_mode=str(args.controller_mode),
            min_lcb_utility=float(args.controller_min_lcb_utility),
            use_timeout_cap=bool(args.use_timeout_cap),
            min_action_edge=float(args.controller_min_action_edge),
            winner_sacrifice_multiplier=float(args.controller_winner_sacrifice_multiplier),
            min_removed_full_sl=float(args.controller_min_removed_full_sl),
            max_removed_timeout=float(args.controller_max_removed_timeout),
            enabled_heads=effective_controller_enabled_heads,
            min_prediction_coverage=float(args.controller_min_prediction_coverage),
            min_usable_candidates=int(args.controller_min_usable_candidates),
            min_frontier_candidates=int(args.controller_min_frontier_candidates),
            max_state_ood_score=args.controller_max_state_ood_score,
        )
        scored_candidates = mstc.apply_thresholds(replay_candidates_base, schedule)
        decisions, _, metrics = replay_candidates(
            scored_candidates,
            params,
            mode="global_auction",
            ev_curve=ev_curve,
            market_mode=args.market_mode,
        )
        accepted = mstc._accepted_trades(scored_candidates, decisions)
        schedule["arm"] = selected_arm
        accepted["arm"] = selected_arm
        if bool(state_spec.get("controller_no_backfill_overlay", False)):
            accepted_keys = mstc._accepted_key_set(accepted)
            if not accepted_keys.issubset(baseline_allowed_keys):
                raise RuntimeError("No-backfill overlay accepted trades outside baseline accepted keys")
        eval_summary = pd.DataFrame([mstc._metrics_row(selected_arm, metrics, accepted, schedule)])
        eval_by_head = mstc._by_head(selected_arm, accepted)
        eval_state.to_csv(args.output_dir / "eval_market_state_features.csv", index=False)
        predictions.to_parquet(args.output_dir / "controller_predictions.parquet", index=False)
        schedule.to_csv(args.output_dir / "controller_schedule.csv", index=False)
        schedule.to_parquet(args.output_dir / "strategy_threshold_schedule.parquet", index=False)
        action_audit = mstc.threshold_action_audit(schedule)
        action_audit.to_csv(args.output_dir / "strategy_threshold_action_audit.csv", index=False)
        if shadow_controller_only and len(state_cols) > 0:
            shadow_schedule = mstc.threshold_schedule(
                eval_frame,
                predictions,
                models["curves"],
                delta_max=float(args.threshold_delta_max),
                max_down_step=float(args.max_threshold_up_step),
                relax_alpha=float(args.threshold_relax_alpha),
                controller_mode=str(args.controller_mode),
                min_lcb_utility=float(args.controller_min_lcb_utility),
                use_timeout_cap=bool(args.use_timeout_cap),
                min_action_edge=float(args.controller_min_action_edge),
                winner_sacrifice_multiplier=float(args.controller_winner_sacrifice_multiplier),
                min_removed_full_sl=float(args.controller_min_removed_full_sl),
                max_removed_timeout=float(args.controller_max_removed_timeout),
                enabled_heads=controller_enabled_heads,
                min_prediction_coverage=float(args.controller_min_prediction_coverage),
                min_usable_candidates=int(args.controller_min_usable_candidates),
                min_frontier_candidates=int(args.controller_min_frontier_candidates),
                max_state_ood_score=args.controller_max_state_ood_score,
            )
            shadow_schedule["arm"] = f"{selected_arm}__shadow_proposed"
            shadow_schedule.to_csv(
                args.output_dir / "shadow_controller_proposed_schedule.csv",
                index=False,
            )
            shadow_schedule.to_parquet(
                args.output_dir / "shadow_controller_proposed_schedule.parquet",
                index=False,
            )
            shadow_action_audit = mstc.threshold_action_audit(shadow_schedule)
            shadow_action_audit.to_csv(
                args.output_dir / "shadow_threshold_action_audit.csv",
                index=False,
            )
            shadow_suppression_utility = mstc._threshold_candidate_suppression_utility(
                eval_candidates,
                shadow_schedule,
            )
            shadow_suppression_utility.to_csv(
                args.output_dir / "shadow_threshold_candidate_suppression_utility.csv",
                index=False,
            )
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
        scored_candidates.to_parquet(args.output_dir / "controller_scored_candidates.parquet", index=False)
        decisions.to_parquet(args.output_dir / "decisions.parquet", index=False)
        accepted.to_parquet(args.output_dir / "accepted_trades.parquet", index=False)
        eval_summary.to_csv(args.output_dir / "controller_replay_summary.csv", index=False)
        eval_by_head.to_csv(args.output_dir / "controller_replay_by_head.csv", index=False)
        outputs.update(
            {
                "eval_state_features": str(args.output_dir / "eval_market_state_features.csv"),
                "controller_predictions": str(args.output_dir / "controller_predictions.parquet"),
                "controller_schedule": str(args.output_dir / "controller_schedule.csv"),
                "strategy_threshold_schedule": str(args.output_dir / "strategy_threshold_schedule.parquet"),
                "strategy_threshold_action_audit": str(args.output_dir / "strategy_threshold_action_audit.csv"),
                "controller_scored_candidates": str(args.output_dir / "controller_scored_candidates.parquet"),
                "decisions": str(args.output_dir / "decisions.parquet"),
                "accepted_trades": str(args.output_dir / "accepted_trades.parquet"),
                "controller_replay_summary": str(args.output_dir / "controller_replay_summary.csv"),
                "controller_replay_by_head": str(args.output_dir / "controller_replay_by_head.csv"),
            }
        )

    (args.output_dir / "market_state_feature_contract.json").write_text(
        json.dumps(_json_safe(feature_contract), indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "market_state_universe_contract.json").write_text(
        json.dumps(_json_safe(universe_contract), indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "strategy_threshold_controller_config.json").write_text(
        json.dumps(_json_safe(controller_config), indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = _make_manifest(
        args=args,
        selected_arm=selected_arm,
        selected_payload=selected_payload,
        state_spec=state_spec,
        state_artifacts=state_artifacts,
        response_feature_cols=response_features,
        response_report=response_report,
        activation_report=activation_report,
        state_activation_filter=state_activation_filter,
        controller_execution_enabled=controller_execution_enabled,
        walkforward_config=walkforward_config,
        forecast_model_kind_report=forecast_model_kind_report,
        response_model_kind_report=response_model_kind_report,
        runtime_param_resolution=runtime_param_resolution,
        state_join_validation=state_join_validation,
        bundle_path=bundle_path,
        outputs=outputs,
    )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "market_state_controller_bundle_report.md").write_text(
        _render_report(manifest, eval_summary, eval_by_head),
        encoding="utf-8",
    )
    hash_inputs = {
        key: path
        for key, path in outputs.items()
        if key not in {"artifact_hashes", "manifest", "report"} and path
    }
    (args.output_dir / "artifact_hashes.json").write_text(
        json.dumps(_json_safe(_artifact_hashes(hash_inputs)), indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(_json_safe(manifest), indent=2)[:6000])
    print(f"\nWrote {args.output_dir}")


if __name__ == "__main__":
    main()
