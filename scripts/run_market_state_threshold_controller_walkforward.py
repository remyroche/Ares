#!/usr/bin/env python3
"""Chronological walk-forward validation for the market-state controller."""

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
    portfolio_policy_params_from_live_config,
    replay_candidates,
)
from scripts import run_market_state_threshold_controller as mstc  # noqa: E402


DEFAULT_OUTPUT_DIR = Path(
    "data_perp/reports/market_state_threshold_controller_walkforward_20260625"
)
PRUNED_STATE_ARM = "S7_pruned_state_pack"


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
    rows: dict[str, Any] = {}
    for name, raw_path in sorted(paths.items()):
        path = Path(raw_path)
        digest = _file_sha256(path)
        rows[name] = {
            "path": str(path),
            "exists": bool(path.exists()),
            "bytes": int(path.stat().st_size) if path.exists() and path.is_file() else None,
            "sha256": digest,
        }
    return {
        "hash_version": "sha256_artifact_hashes_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": rows,
    }


def _parse_csv_list(value: str | None) -> list[str]:
    return [part.strip() for part in str(value or "").split(",") if part.strip()]


def _add_pruned_state_pack(
    states: dict[str, tuple[pd.DataFrame, pd.DataFrame, list[str]]],
    *,
    state_head_allowlist: list[str],
    arm_name: str = PRUNED_STATE_ARM,
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame, list[str]]]:
    """Add an explicit pruned state-pack arm without mutating S1/S2 arms."""
    allow = {str(col) for col in state_head_allowlist if str(col)}
    if not allow:
        return states
    candidates: list[tuple[str, pd.DataFrame, pd.DataFrame, list[str]]] = []
    for source_arm, (train_state, valid_state, state_cols) in states.items():
        cols = [col for col in state_cols if col in allow]
        if cols:
            candidates.append((source_arm, train_state, valid_state, cols))
    if not candidates:
        return states
    # Prefer the broadest source containing the allowlisted heads; this keeps
    # selected forecast heads with their observed-axis context source while
    # preserving the original S1/S2 arms for comparison.
    source_arm, train_state, valid_state, cols = max(candidates, key=lambda item: len(item[3]))
    out = dict(states)
    out[str(arm_name)] = (
        train_state[["timestamp", *cols]].copy(),
        valid_state[["timestamp", *cols]].copy(),
        cols,
    )
    return out


def _load_policy_params(path: Path, variant: str):
    payload = json.loads(path.read_text(encoding="utf-8"))
    params = payload.get("variant_params", {}).get(variant)
    if not isinstance(params, dict):
        raise KeyError(f"Missing variant_params[{variant!r}] in {path}")
    return portfolio_policy_params_from_live_config(params), payload


def _build_time_folds(
    timestamps: pd.Series,
    *,
    n_folds: int,
    min_train_days: int,
    valid_days: int,
    embargo_hours: int,
    min_valid_rows: int = 1,
    min_valid_timestamps: int = 1,
) -> list[dict[str, pd.Timestamp]]:
    all_ts = pd.to_datetime(timestamps, utc=True, errors="coerce").dropna().sort_values()
    if all_ts.empty:
        return []
    ts = all_ts.drop_duplicates().sort_values()
    start = ts.min() + pd.Timedelta(days=int(min_train_days))
    end_latest = ts.max() - pd.Timedelta(days=int(valid_days))
    candidates = ts.loc[(ts >= start) & (ts <= end_latest)]
    if candidates.empty:
        return []
    supported: list[pd.Timestamp] = []
    support: dict[pd.Timestamp, tuple[int, int]] = {}
    for valid_start in candidates:
        valid_end = min(
            valid_start + pd.Timedelta(days=int(valid_days)),
            ts.max() + pd.Timedelta(nanoseconds=1),
        )
        valid_mask = (all_ts >= valid_start) & (all_ts < valid_end)
        valid_rows = int(valid_mask.sum())
        valid_timestamps = int(all_ts.loc[valid_mask].nunique())
        support[pd.Timestamp(valid_start)] = (valid_rows, valid_timestamps)
        if valid_rows >= int(min_valid_rows) and valid_timestamps >= int(min_valid_timestamps):
            supported.append(pd.Timestamp(valid_start))
    if not supported:
        return []
    candidates = pd.Series(supported).sort_values(ignore_index=True)
    if len(candidates) <= n_folds:
        starts = candidates
    else:
        idx = np.linspace(0, len(candidates) - 1, int(n_folds)).round().astype(int)
        starts = candidates.iloc[np.unique(idx)]
    folds: list[dict[str, pd.Timestamp]] = []
    for i, valid_start in enumerate(starts, start=1):
        valid_end = valid_start + pd.Timedelta(days=int(valid_days))
        train_end = valid_start - pd.Timedelta(hours=int(embargo_hours))
        if train_end <= ts.min():
            continue
        valid_rows, valid_timestamps = support[pd.Timestamp(valid_start)]
        folds.append(
            {
                "fold": i,
                "train_start": ts.min(),
                "train_end": train_end,
                "valid_start": valid_start,
                "valid_end": min(valid_end, ts.max() + pd.Timedelta(nanoseconds=1)),
                "valid_rows_available": int(valid_rows),
                "valid_timestamps_available": int(valid_timestamps),
            }
        )
    return folds


def _filter_time(df: pd.DataFrame, start: pd.Timestamp | None, end: pd.Timestamp) -> pd.DataFrame:
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    mask = ts < end
    if start is not None:
        mask &= ts >= start
    return df.loc[mask].copy()


def _filter_matured_training_time(
    df: pd.DataFrame,
    start: pd.Timestamp | None,
    entry_end: pd.Timestamp,
    outcome_available_end: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Filter training rows by entry time and known outcome availability.

    Entry timestamps still obey the purged train/validation split, but rows
    whose realized trade outcome is unavailable before validation starts are
    removed from response/EV training. This prevents a training fold from
    learning from labels that would not have been known at the validation
    decision time.
    """

    entry_filtered = _filter_time(df, start, entry_end)
    diag: dict[str, Any] = {
        "entry_filtered_rows": int(len(entry_filtered)),
        "outcome_available_end": outcome_available_end,
        "uses_outcome_available_timestamp": "exit_timestamp" in entry_filtered.columns,
        "dropped_immature_outcome_rows": 0,
        "missing_outcome_available_rows": 0,
        "matured_rows": int(len(entry_filtered)),
    }
    if entry_filtered.empty or "exit_timestamp" not in entry_filtered.columns:
        return entry_filtered, diag
    available_ts = pd.to_datetime(entry_filtered["exit_timestamp"], utc=True, errors="coerce")
    known = available_ts.notna() & (available_ts < outcome_available_end)
    diag.update(
        {
            "missing_outcome_available_rows": int(available_ts.isna().sum()),
            "dropped_immature_outcome_rows": int((~known).sum()),
            "matured_rows": int(known.sum()),
            "max_outcome_available_timestamp": available_ts.loc[known].max()
            if bool(known.any())
            else None,
        }
    )
    return entry_filtered.loc[known].copy(), diag


def _state_features_for_fold(
    train_broad: pd.DataFrame,
    valid_broad: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[dict[str, tuple[pd.DataFrame, pd.DataFrame, list[str]]], dict[str, Any], dict[str, Any]]:
    feature_cols = mstc._common_feature_columns(train_broad, valid_broad, args.max_feature_cols)
    train_candidate_agg = mstc._timestamp_aggregates(train_broad, feature_cols)
    valid_candidate_agg = mstc._timestamp_aggregates(valid_broad, feature_cols)
    feature_store_cols = mstc._select_feature_store_columns(
        args.feature_store_dir,
        args.feature_store_dir,
        int(args.max_feature_store_cols),
    )
    train_fs, train_fs_report, valid_fs, valid_fs_report = mstc._feature_store_timestamp_aggregate_pair(
        args.feature_store_dir,
        args.feature_store_dir,
        train_candidate_agg["timestamp"],
        valid_candidate_agg["timestamp"],
        feature_store_cols,
        symbol_cap=int(args.feature_store_symbol_cap),
    )
    train_state_source, train_state_source_report = mstc._state_source_aggregate_frame(
        train_candidate_agg,
        train_fs,
        allow_candidate_fallback=bool(args.allow_candidate_state_fallback),
    )
    valid_state_source, valid_state_source_report = mstc._state_source_aggregate_frame(
        valid_candidate_agg,
        valid_fs,
        allow_candidate_fallback=bool(args.allow_candidate_state_fallback),
    )
    observed_axis_encoder = mstc.fit_observed_axis_encoder(train_state_source, valid_state_source)
    train_observed = mstc.transform_observed_axes(train_state_source, observed_axis_encoder)
    valid_observed = mstc.transform_observed_axes(valid_state_source, observed_axis_encoder)
    axis_sources = dict(observed_axis_encoder.get("axis_sources") or {})
    train_forecast, forecast_artifact, forecast_report = mstc.fit_forecast_state_heads(
        train_observed,
        horizon_steps=list(
            mstc._parse_int_grid(
                args.forecast_horizons_steps,
                (max(1, int(args.forecast_horizon_steps)),),
            )
        ),
        train_agg=train_state_source,
        forecast_model_kind=str(args.forecast_model_kind),
    )
    valid_forecast = mstc.transform_forecast_state_heads(
        valid_observed,
        forecast_artifact,
        agg=valid_state_source,
    )
    observed_cols = [c for c in train_observed.columns if c != "timestamp"]
    forecast_cols = [c for c in train_forecast.columns if c != "timestamp"]
    states = {
        "S1_observed_axes_shared_response": (train_observed, valid_observed, observed_cols),
        "S2_observed_forecast_shared_response": (train_forecast, valid_forecast, forecast_cols),
    }
    allowlist = _parse_csv_list(getattr(args, "state_head_allowlist", ""))
    pruned_arm_name = str(getattr(args, "pruned_state_arm_name", PRUNED_STATE_ARM) or PRUNED_STATE_ARM)
    states = _add_pruned_state_pack(
        states,
        state_head_allowlist=allowlist,
        arm_name=pruned_arm_name,
    )
    if bool(getattr(args, "include_latent_shadow_arms", False)):
        train_latent, latent_artifact, latent_report = mstc.fit_latent_state_probs(
            train_forecast,
            n_states=int(args.latent_states),
        )
        valid_latent = mstc.transform_latent_state_probs(valid_forecast, latent_artifact)
        latent_cols = [c for c in train_latent.columns if c != "timestamp"]
        states.update(
            {
                "S3_observed_forecast_latent_shared_response": (train_latent, valid_latent, latent_cols),
                "S4_S3_plus_per_strategy_residual": (train_latent, valid_latent, latent_cols),
            }
        )
    else:
        latent_artifact = None
        latent_report = {
            "mode": "shadow_disabled_by_default",
            "reason": "latent_gmm_outputs_removed_from_active_controller_architecture",
        }
    report = {
        "candidate_feature_count": int(len(feature_cols)),
        "feature_store": {
            "selected_column_count": int(len(feature_store_cols)),
            "train": train_fs_report,
            "valid": valid_fs_report,
        },
        "market_state_source": {
            "train": train_state_source_report,
            "valid": valid_state_source_report,
        },
        "axis_sources": axis_sources,
        "forecast_report": forecast_report,
        "latent_report": latent_report,
        "pruned_state_pack": {
            "enabled": bool(allowlist),
            "allowlist": allowlist,
            "arm": pruned_arm_name,
            "present": pruned_arm_name in states,
            "selected_columns": list(states.get(pruned_arm_name, (None, None, []))[2]),
        },
    }
    reference_bundle = {
        "reference_version": "market_state_fold_training_reference_v1",
        "feature_store_columns": feature_store_cols,
        "feature_store_tail_reference_quantiles": dict(
            train_fs_report.get("tail_reference_quantiles") or {}
        ),
        "feature_store_reports": {
            "train": train_fs_report,
            "valid": valid_fs_report,
        },
        "market_state_source_reports": {
            "train": train_state_source_report,
            "valid": valid_state_source_report,
        },
        "observed_axis_encoder": observed_axis_encoder,
        "forecast_artifact": forecast_artifact,
        "latent_artifact": latent_artifact,
        "latent_report": latent_report,
    }
    return states, report, reference_bundle


def _state_timestamp_panel(
    states: dict[str, tuple[pd.DataFrame, pd.DataFrame, list[str]]],
    *,
    fold: int,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for arm, (train_state, valid_state, state_cols) in states.items():
        for split, frame in (("train", train_state), ("valid", valid_state)):
            if frame.empty:
                continue
            cols = ["timestamp", *[c for c in state_cols if c in frame.columns]]
            out = frame[cols].copy()
            out.insert(0, "fold", int(fold))
            out.insert(1, "split", split)
            out.insert(2, "state_arm", str(arm))
            out["state_feature_count"] = int(len(state_cols))
            frames.append(out)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _strategy_residual_target_ledger(
    frame: pd.DataFrame,
    curves: Any,
    *,
    fold: int,
    arm: str,
    state_training_input_contract: str = "",
    response_training_uses_oof_state_scores: bool | None = None,
) -> pd.DataFrame:
    cols = [
        c
        for c in (
            "timestamp",
            "strategy_id",
            "head",
            "side",
            "symbol",
            "_rank",
            "_threshold",
            "_net_return",
            "_is_full_sl",
            "_is_timeout",
            "normalized_rank_score",
            "calibrated_score",
            "base_strategy_threshold",
            "deployment_rank_threshold",
        )
        if c in frame.columns
    ]
    out = frame[cols].copy()
    out["fold"] = int(fold)
    out["arm"] = str(arm)
    out["state_training_input_contract"] = str(state_training_input_contract or "")
    out["response_training_uses_oof_state_scores"] = response_training_uses_oof_state_scores
    out["base_mu"] = curves.predict(frame["strategy_id"], frame["_rank"], "mu")
    out["base_psl"] = curves.predict(frame["strategy_id"], frame["_rank"], "psl")
    out["base_pto"] = curves.predict(frame["strategy_id"], frame["_rank"], "pto")
    out["resid_utility"] = pd.to_numeric(frame["_net_return"], errors="coerce") - out["base_mu"]
    out["resid_full_sl"] = pd.to_numeric(frame["_is_full_sl"], errors="coerce") - out["base_psl"]
    out["resid_timeout"] = pd.to_numeric(frame["_is_timeout"], errors="coerce") - out["base_pto"]
    return out


def _strategy_response_prediction_ledger(
    frame: pd.DataFrame,
    pred: pd.DataFrame,
    *,
    fold: int,
    arm: str,
) -> pd.DataFrame:
    key_cols = [
        c
        for c in (
            "timestamp",
            "strategy_id",
            "head",
            "side",
            "symbol",
            "_rank",
            "_threshold",
            "_net_return",
            "_is_full_sl",
            "_is_timeout",
        )
        if c in frame.columns
    ]
    out = frame[key_cols].reset_index(drop=True).copy()
    pred_cols = [
        c
        for c in pred.columns
        if c
        not in {
            "timestamp",
            "strategy_id",
            "head",
            "_rank",
            "_threshold",
        }
    ]
    out = pd.concat([out, pred[pred_cols].reset_index(drop=True)], axis=1)
    out["fold"] = int(fold)
    out["arm"] = str(arm)
    out["state_prediction_contract"] = "outer_fold_validation_state_scores"
    out["actual_resid_utility"] = pd.to_numeric(frame["_net_return"], errors="coerce").reset_index(drop=True) - out["base_mu"]
    out["actual_resid_full_sl"] = pd.to_numeric(frame["_is_full_sl"], errors="coerce").reset_index(drop=True) - out["base_psl"]
    out["actual_resid_timeout"] = pd.to_numeric(frame["_is_timeout"], errors="coerce").reset_index(drop=True) - out["base_pto"]
    out["pred_resid_utility"] = out["pred_eu_mean"]
    out["pred_resid_utility_lcb"] = out["pred_eu_q10"]
    out["pred_resid_full_sl"] = out["pred_excess_full_sl"]
    out["pred_resid_timeout"] = out["pred_excess_timeout"]
    return out


def _corr_or_nan(a: pd.Series, b: pd.Series, method: str) -> float:
    x = pd.to_numeric(a, errors="coerce").replace([np.inf, -np.inf], np.nan)
    y = pd.to_numeric(b, errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = x.notna() & y.notna()
    if int(valid.sum()) < 8:
        return float("nan")
    if x.loc[valid].nunique(dropna=True) <= 1 or y.loc[valid].nunique(dropna=True) <= 1:
        return float("nan")
    return float(x.loc[valid].corr(y.loc[valid], method=method))


def _strategy_state_effect_matrix(
    frame: pd.DataFrame,
    pred: pd.DataFrame,
    state_cols: list[str],
    *,
    fold: int,
    arm: str,
) -> pd.DataFrame:
    if frame.empty or not state_cols:
        return pd.DataFrame()
    work = frame.reset_index(drop=True).copy()
    for col in (
        "pred_resid_utility",
        "pred_resid_utility_lcb",
        "pred_resid_full_sl",
        "pred_resid_timeout",
        "pred_mean_utility",
        "pred_lcb_utility",
        "pred_full_sl",
        "pred_timeout",
    ):
        source = {
            "pred_resid_utility": "pred_eu_mean",
            "pred_resid_utility_lcb": "pred_eu_q10",
            "pred_resid_full_sl": "pred_excess_full_sl",
            "pred_resid_timeout": "pred_excess_timeout",
        }.get(col, col)
        if source in pred.columns:
            work[col] = pd.to_numeric(pred[source], errors="coerce").reset_index(drop=True)
    targets = [
        "pred_resid_utility",
        "pred_resid_utility_lcb",
        "pred_resid_full_sl",
        "pred_resid_timeout",
        "pred_mean_utility",
        "pred_full_sl",
        "pred_timeout",
    ]
    rows: list[dict[str, Any]] = []
    groups: list[tuple[str, str, pd.DataFrame]] = [("all", "all", work)]
    if "strategy_id" in work.columns:
        groups.extend((("strategy_id", str(k), g) for k, g in work.groupby("strategy_id", sort=False)))
    if "head" in work.columns:
        groups.extend((("head", str(k), g) for k, g in work.groupby("head", sort=False)))
    for scope, scope_value, g in groups:
        for state_col in state_cols:
            if state_col not in g.columns:
                continue
            state = pd.to_numeric(g[state_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            valid_state = state.dropna()
            if len(valid_state) < 8 or valid_state.nunique(dropna=True) <= 1:
                continue
            q10 = float(valid_state.quantile(0.10))
            q90 = float(valid_state.quantile(0.90))
            low_mask = state <= q10
            high_mask = state >= q90
            for target in targets:
                if target not in g.columns:
                    continue
                y = pd.to_numeric(g[target], errors="coerce").replace([np.inf, -np.inf], np.nan)
                valid = state.notna() & y.notna()
                if int(valid.sum()) < 8 or y.loc[valid].nunique(dropna=True) <= 1:
                    continue
                low_mean = float(y.loc[low_mask & y.notna()].mean()) if (low_mask & y.notna()).any() else np.nan
                high_mean = float(y.loc[high_mask & y.notna()].mean()) if (high_mask & y.notna()).any() else np.nan
                rows.append(
                    {
                        "fold": int(fold),
                        "arm": str(arm),
                        "scope": scope,
                        "scope_value": scope_value,
                        "state_feature": state_col,
                        "target": target,
                        "rows": int(valid.sum()),
                        "state_q10": q10,
                        "state_q90": q90,
                        "target_mean_state_q10": low_mean,
                        "target_mean_state_q90": high_mean,
                        "target_q90_minus_q10": (
                            float(high_mean - low_mean)
                            if np.isfinite(high_mean) and np.isfinite(low_mean)
                            else np.nan
                        ),
                        "pearson": _corr_or_nan(state, y, "pearson"),
                        "spearman": _corr_or_nan(state, y, "spearman"),
                    }
                )
    return pd.DataFrame(rows)


def _accepted_jaccard(a: pd.DataFrame, b: pd.DataFrame) -> float:
    left = mstc._accepted_key_set(a) if not a.empty else set()
    right = mstc._accepted_key_set(b) if not b.empty else set()
    union = left | right
    if not union:
        return 1.0
    return float(len(left & right) / len(union))


def _schedule_threshold_raise_count(schedule: pd.DataFrame) -> int:
    if schedule.empty:
        return 0
    delta = (
        pd.to_numeric(schedule.get("state_threshold"), errors="coerce")
        - pd.to_numeric(schedule.get("base_threshold"), errors="coerce")
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return int((delta > 1e-9).sum())


def _state_leave_one_out_replay(
    *,
    fold: int,
    arm: str,
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    valid_broad: pd.DataFrame,
    models: dict[str, Any],
    response_features: list[str],
    state_cols: list[str],
    full_summary_row: dict[str, Any],
    full_schedule: pd.DataFrame,
    full_accepted: pd.DataFrame,
    params: Any,
    ev_curve: Any,
    args: argparse.Namespace,
    baseline_keys: set[tuple[Any, ...]] | None = None,
) -> pd.DataFrame:
    """Replay the controller after neutralizing one state head at a time.

    This is a bounded leave-one-state-head-out diagnostic. It reuses the
    fold-fitted response model and replaces the held-out state value with its
    training median before prediction. It is intentionally labelled no-refit:
    the artifact tests execution dependence on a state head without paying the
    cost or instability of refitting the response model once per state.
    """

    if not state_cols or valid_frame.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    full_net = float(full_summary_row.get("net_pnl", 0.0) or 0.0)
    full_trade_count = int(full_summary_row.get("trade_count", 0) or 0)
    full_full_sl = float(full_summary_row.get("full_sl_rate", np.nan))
    full_timeout = float(full_summary_row.get("timeout_rate", np.nan))
    full_mean_delta = float(full_summary_row.get("mean_threshold_delta", 0.0) or 0.0)
    full_raise_count = _schedule_threshold_raise_count(full_schedule)
    for state_col in state_cols:
        if state_col not in valid_frame.columns or state_col not in train_frame.columns:
            continue
        train_values = pd.to_numeric(train_frame[state_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if train_values.notna().any():
            neutral_value = float(train_values.median())
        else:
            neutral_value = 0.0
        loo_frame = valid_frame.copy()
        loo_frame[state_col] = neutral_value
        loo_pred = mstc.predict_response(models, loo_frame, response_features, state_cols)
        loo_schedule = mstc.threshold_schedule(
            loo_frame,
            loo_pred,
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
            enabled_heads=args.controller_enabled_heads_parsed,
            min_prediction_coverage=float(args.controller_min_prediction_coverage),
            min_usable_candidates=int(args.controller_min_usable_candidates),
            min_frontier_candidates=int(args.controller_min_frontier_candidates),
            max_state_ood_score=args.controller_max_state_ood_score,
            accepted_decision_keys=(
                baseline_keys
                if str(args.controller_mode) == "accepted_frontier_action_rank_grid"
                and baseline_keys
                else None
            ),
        )
        loo_candidates = mstc.apply_thresholds(valid_broad, loo_schedule)
        decisions, _equity, metrics = replay_candidates(
            loo_candidates,
            params,
            mode="global_auction",
            ev_curve=ev_curve,
            market_mode=args.market_mode,
        )
        loo_accepted = mstc._accepted_trades(loo_candidates, decisions)
        loo_row = mstc._metrics_row(f"{arm}__leave_one_out_{state_col}", metrics, loo_accepted, loo_schedule)
        loo_net = float(loo_row.get("net_pnl", 0.0) or 0.0)
        state_action_delta = mstc._schedule_action_delta(loo_accepted, full_accepted)
        rows.append(
            {
                "fold": int(fold),
                "arm": str(arm),
                "state_head": str(state_col),
                "leave_one_out_mode": "neutralized_valid_state_no_refit",
                "neutral_value": neutral_value,
                "full_net_pnl": full_net,
                "loo_net_pnl": loo_net,
                "increment_net_pnl": full_net - loo_net,
                "full_trade_count": full_trade_count,
                "loo_trade_count": int(loo_row.get("trade_count", 0) or 0),
                "delta_trade_count": full_trade_count - int(loo_row.get("trade_count", 0) or 0),
                "full_full_sl_rate": full_full_sl,
                "loo_full_sl_rate": float(loo_row.get("full_sl_rate", np.nan)),
                "full_timeout_rate": full_timeout,
                "loo_timeout_rate": float(loo_row.get("timeout_rate", np.nan)),
                "full_mean_threshold_delta": full_mean_delta,
                "loo_mean_threshold_delta": float(loo_row.get("mean_threshold_delta", 0.0) or 0.0),
                "full_threshold_raise_count": full_raise_count,
                "loo_threshold_raise_count": _schedule_threshold_raise_count(loo_schedule),
                "accepted_jaccard_vs_full": _accepted_jaccard(full_accepted, loo_accepted),
                "state_head_removed_loss_avoided": float(state_action_delta.get("removed_loss_avoided", 0.0) or 0.0),
                "state_head_removed_winner_pnl_sacrificed": float(
                    state_action_delta.get("removed_winner_pnl_sacrificed", 0.0) or 0.0
                ),
                "state_head_defensive_success": float(state_action_delta.get("defensive_success", 0.0) or 0.0),
                "state_head_added_net_pnl": float(state_action_delta.get("entrant_net_pnl", 0.0) or 0.0),
                "state_head_removed_net_pnl": float(state_action_delta.get("removed_net_pnl", 0.0) or 0.0),
                "state_head_net_action_pnl_delta": float(state_action_delta.get("net_action_pnl_delta", 0.0) or 0.0),
            }
        )
    return pd.DataFrame(rows)


def _response_state_training_contract(
    *,
    arm: str,
    state_cols: list[str],
    state_report: dict[str, Any],
) -> dict[str, Any]:
    learned_cols = [str(col) for col in state_cols if str(col).startswith("forecast_")]
    forecast_targets = dict(dict(state_report.get("forecast_report") or {}).get("targets") or {})
    mode_by_col: dict[str, str] = {}
    non_oof_learned_cols: list[str] = []
    fallback_cols: list[str] = []
    for col in learned_cols:
        target_report = dict(forecast_targets.get(col) or {})
        mode = str(target_report.get("train_prediction_mode") or target_report.get("mode") or "")
        mode_by_col[col] = mode
        if mode == "chronological_expanding_oof_or_fallback":
            continue
        if mode == "bounded_current_axis_fallback" or "fallback" in mode:
            fallback_cols.append(col)
            continue
        non_oof_learned_cols.append(col)

    if not learned_cols:
        contract = "fold_fitted_descriptive_state_axes_no_learned_state_oof_required"
        passed = True
    elif non_oof_learned_cols:
        contract = "learned_state_training_not_oof"
        passed = False
    elif fallback_cols and len(fallback_cols) == len(learned_cols):
        contract = "learned_state_training_all_bounded_current_axis_fallback"
        passed = True
    else:
        contract = "learned_state_training_chronological_expanding_oof_or_fallback"
        passed = True

    return {
        "state_training_input_contract": contract,
        "response_training_state_prediction_contract": contract,
        "response_training_uses_oof_state_scores": bool(passed),
        "response_training_state_contract_passed": bool(passed),
        "response_training_arm": str(arm),
        "learned_state_column_count": int(len(learned_cols)),
        "learned_state_columns": learned_cols,
        "learned_state_train_prediction_modes": mode_by_col,
        "learned_state_fallback_columns": fallback_cols,
        "learned_state_non_oof_columns": non_oof_learned_cols,
    }


def _run_fold(
    fold: dict[str, Any],
    train_broad_all: pd.DataFrame,
    train_deployable_all: pd.DataFrame,
    params: Any,
    args: argparse.Namespace,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
    dict[str, Any],
]:
    train_broad, train_broad_maturity_diag = _filter_matured_training_time(
        train_broad_all,
        None,
        fold["train_end"],
        fold["valid_start"],
    )
    train_deployable, train_deployable_maturity_diag = _filter_matured_training_time(
        train_deployable_all,
        None,
        fold["train_end"],
        fold["valid_start"],
    )
    valid_broad = _filter_time(train_broad_all, fold["valid_start"], fold["valid_end"])
    if train_broad.empty or train_deployable.empty or valid_broad.empty:
        raise RuntimeError(f"Fold {fold['fold']} has empty train/deployable/valid split")
    ev_curve = fit_hierarchical_ev_curves(train_deployable)
    states, state_report, state_reference_bundle = _state_features_for_fold(train_broad, valid_broad, args)
    state_timestamp_panel = _state_timestamp_panel(states, fold=int(fold["fold"]))
    arms = {
        "S0_baseline_static_thresholds": {
            "state_train": None,
            "state_valid": None,
            "state_cols": [],
            "per_strategy_residual": False,
        },
        "S1_observed_axes_shared_response": {
            "state_train": states["S1_observed_axes_shared_response"][0],
            "state_valid": states["S1_observed_axes_shared_response"][1],
            "state_cols": states["S1_observed_axes_shared_response"][2],
            "per_strategy_residual": False,
        },
        "S2_observed_forecast_shared_response": {
            "state_train": states["S2_observed_forecast_shared_response"][0],
            "state_valid": states["S2_observed_forecast_shared_response"][1],
            "state_cols": states["S2_observed_forecast_shared_response"][2],
            "per_strategy_residual": False,
        },
    }
    pruned_arm = str(getattr(args, "pruned_state_arm_name", PRUNED_STATE_ARM) or PRUNED_STATE_ARM)
    if pruned_arm in states:
        arms[pruned_arm] = {
            "state_train": states[pruned_arm][0],
            "state_valid": states[pruned_arm][1],
            "state_cols": states[pruned_arm][2],
            "per_strategy_residual": False,
        }
    if bool(getattr(args, "include_latent_shadow_arms", False)):
        arms.update(
            {
                "S3_observed_forecast_latent_shared_response": {
                    "state_train": states["S3_observed_forecast_latent_shared_response"][0],
                    "state_valid": states["S3_observed_forecast_latent_shared_response"][1],
                    "state_cols": states["S3_observed_forecast_latent_shared_response"][2],
                    "per_strategy_residual": False,
                },
                "S4_S3_plus_per_strategy_residual": {
                    "state_train": states["S4_S3_plus_per_strategy_residual"][0],
                    "state_valid": states["S4_S3_plus_per_strategy_residual"][1],
                    "state_cols": states["S4_S3_plus_per_strategy_residual"][2],
                    "per_strategy_residual": True,
                },
            }
        )
    summary_rows: list[dict[str, Any]] = []
    by_head_frames: list[pd.DataFrame] = []
    accepted_frames: list[pd.DataFrame] = []
    schedule_frames: list[pd.DataFrame] = []
    rank_curve_frames: list[pd.DataFrame] = []
    residual_ledger_frames: list[pd.DataFrame] = []
    response_prediction_frames: list[pd.DataFrame] = []
    state_effect_frames: list[pd.DataFrame] = []
    state_loo_frames: list[pd.DataFrame] = []
    response_model_bundles: dict[str, Any] = {}
    model_reports: dict[str, Any] = {}
    baseline_keys: set[tuple[Any, ...]] = set()
    for arm, spec in arms.items():
        valid_frame_for_overlay: pd.DataFrame | None = None
        pred_for_overlay: pd.DataFrame | None = None
        curves_for_overlay: Any | None = None
        if spec["state_train"] is None:
            candidate_arm = valid_broad.copy()
            schedule = pd.DataFrame()
            model_reports[arm] = {"mode": "baseline_no_state_controller"}
        else:
            train_frame = mstc.build_response_frame(train_broad, spec["state_train"])
            valid_frame = mstc.build_response_frame(valid_broad, spec["state_valid"])
            models, response_features, model_report = mstc.fit_response_models(
                train_frame,
                spec["state_cols"],
                per_strategy_residual=bool(spec["per_strategy_residual"]),
                max_rows=int(args.max_response_rows),
                max_keyword_cols=int(args.max_response_keyword_cols),
                response_model_kind=str(args.response_model_kind),
                response_frontier_weight_gamma=float(args.response_frontier_weight_gamma),
                response_frontier_weight_bandwidth=float(args.response_frontier_weight_bandwidth),
                response_balance_timestamps=bool(args.response_balance_timestamps),
                response_balance_strategies=bool(args.response_balance_strategies),
            )
            state_training_contract = _response_state_training_contract(
                arm=arm,
                state_cols=list(spec["state_cols"]),
                state_report=state_report,
            )
            model_report = model_report | state_training_contract
            pred = mstc.predict_response(models, valid_frame, response_features, spec["state_cols"])
            valid_frame_for_overlay = valid_frame
            pred_for_overlay = pred
            curves_for_overlay = models["curves"]
            curves = models["curves"]
            rank_curve = curves.table.copy()
            rank_curve["fold"] = int(fold["fold"])
            rank_curve["arm"] = arm
            for key, value in dict(curves.global_values).items():
                rank_curve[f"global_{key}"] = value
            rank_curve_frames.append(rank_curve)
            residual_ledger_frames.append(
                _strategy_residual_target_ledger(
                    train_frame,
                    curves,
                    fold=int(fold["fold"]),
                    arm=arm,
                    state_training_input_contract=str(
                        state_training_contract.get("state_training_input_contract", "")
                    ),
                    response_training_uses_oof_state_scores=bool(
                        state_training_contract.get("response_training_uses_oof_state_scores")
                    ),
                )
            )
            response_prediction_frames.append(
                _strategy_response_prediction_ledger(
                    valid_frame,
                    pred,
                    fold=int(fold["fold"]),
                    arm=arm,
                )
            )
            effect = _strategy_state_effect_matrix(
                valid_frame,
                pred,
                spec["state_cols"],
                fold=int(fold["fold"]),
                arm=arm,
            )
            if not effect.empty:
                state_effect_frames.append(effect)
            response_model_bundles[arm] = {
                "fold": int(fold["fold"]),
                "arm": arm,
                "model_type": model_report.get("risk_model", "rank_curve_plus_additive_ebm_response"),
                "state_columns": list(spec["state_cols"]),
                "response_feature_columns": list(response_features),
                "model_report": model_report,
                "models": models,
            }
            accepted_keys_for_schedule = (
                baseline_keys
                if str(args.controller_mode) == "accepted_frontier_action_rank_grid"
                and arm != "S0_baseline_static_thresholds"
                and baseline_keys
                else None
            )
            schedule = mstc.threshold_schedule(
                valid_frame,
                pred,
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
                enabled_heads=args.controller_enabled_heads_parsed,
                min_prediction_coverage=float(args.controller_min_prediction_coverage),
                min_usable_candidates=int(args.controller_min_usable_candidates),
                min_frontier_candidates=int(args.controller_min_frontier_candidates),
                max_state_ood_score=args.controller_max_state_ood_score,
                accepted_decision_keys=accepted_keys_for_schedule,
            )
            candidate_arm = mstc.apply_thresholds(valid_broad, schedule)
            schedule["arm"] = arm
            schedule["fold"] = int(fold["fold"])
            schedule_frames.append(schedule)
            model_reports[arm] = model_report | {
                "response_feature_count": int(len(response_features)),
                "state_feature_count": int(len(spec["state_cols"])),
            }
        decisions, equity, metrics = replay_candidates(
            candidate_arm,
            params,
            mode="global_auction",
            ev_curve=ev_curve,
            market_mode=args.market_mode,
        )
        decisions["arm"] = arm
        decisions["fold"] = int(fold["fold"])
        accepted = mstc._accepted_trades(candidate_arm, decisions)
        accepted["arm"] = arm
        accepted["fold"] = int(fold["fold"])
        if arm == "S0_baseline_static_thresholds":
            baseline_keys = mstc._accepted_key_set(accepted)
        row = mstc._metrics_row(arm, metrics, accepted, schedule)
        row["fold"] = int(fold["fold"])
        summary_rows.append(row)
        if (
            spec["state_train"] is not None
            and not bool(getattr(args, "skip_state_leave_one_out", False))
            and valid_frame_for_overlay is not None
            and pred_for_overlay is not None
        ):
            loo = _state_leave_one_out_replay(
                fold=int(fold["fold"]),
                arm=arm,
                train_frame=train_frame,
                valid_frame=valid_frame_for_overlay,
                valid_broad=valid_broad,
                models=models,
                response_features=response_features,
                state_cols=list(spec["state_cols"]),
                full_summary_row=row,
                full_schedule=schedule,
                full_accepted=accepted,
                params=params,
                ev_curve=ev_curve,
                args=args,
                baseline_keys=baseline_keys,
            )
            if not loo.empty:
                state_loo_frames.append(loo)
        by_head = mstc._by_head(arm, accepted)
        if not by_head.empty:
            by_head["fold"] = int(fold["fold"])
            by_head_frames.append(by_head)
        accepted_frames.append(accepted)
        if args.include_post_selection_overlay_arms and arm != "S0_baseline_static_thresholds" and baseline_keys:
            overlay_arm = mstc._post_selection_overlay_arm_name(arm)
            overlay_candidates_base = mstc._restrict_to_allowed_decision_keys(valid_broad, baseline_keys)
            if valid_frame_for_overlay is not None and pred_for_overlay is not None and curves_for_overlay is not None:
                overlay_mask = mstc._allowed_decision_key_mask(valid_frame_for_overlay, baseline_keys)
                overlay_frame = valid_frame_for_overlay.loc[overlay_mask].copy()
                overlay_pred = pred_for_overlay.loc[overlay_mask].copy()
                schedule_g = mstc.threshold_schedule(
                    overlay_frame,
                    overlay_pred,
                    curves_for_overlay,
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
                    enabled_heads=args.controller_enabled_heads_parsed,
                    min_prediction_coverage=float(args.controller_min_prediction_coverage),
                    min_usable_candidates=int(args.controller_min_usable_candidates),
                    min_frontier_candidates=int(args.controller_min_frontier_candidates),
                    max_state_ood_score=args.controller_max_state_ood_score,
                    accepted_decision_keys=(
                        baseline_keys
                        if str(args.controller_mode) == "accepted_frontier_action_rank_grid"
                        and baseline_keys
                        else None
                    ),
                )
            else:
                schedule_g = schedule.copy()
            overlay_candidates = mstc.apply_thresholds(overlay_candidates_base, schedule_g)
            decisions_g, equity_g, metrics_g = replay_candidates(
                overlay_candidates,
                params,
                mode="global_auction",
                ev_curve=ev_curve,
                market_mode=args.market_mode,
            )
            decisions_g["arm"] = overlay_arm
            decisions_g["fold"] = int(fold["fold"])
            accepted_g = mstc._accepted_trades(overlay_candidates, decisions_g)
            accepted_g["arm"] = overlay_arm
            accepted_g["fold"] = int(fold["fold"])
            if not schedule_g.empty:
                schedule_g["arm"] = overlay_arm
                schedule_g["fold"] = int(fold["fold"])
                schedule_frames.append(schedule_g)
            row_g = mstc._metrics_row(overlay_arm, metrics_g, accepted_g, schedule_g)
            row_g["fold"] = int(fold["fold"])
            summary_rows.append(row_g)
            by_head_g = mstc._by_head(overlay_arm, accepted_g)
            if not by_head_g.empty:
                by_head_g["fold"] = int(fold["fold"])
                by_head_frames.append(by_head_g)
            accepted_frames.append(accepted_g)
    summary = pd.DataFrame(summary_rows)
    accepted_all = pd.concat(accepted_frames, ignore_index=True) if accepted_frames else pd.DataFrame()
    by_head = pd.concat(by_head_frames, ignore_index=True) if by_head_frames else pd.DataFrame()
    schedules = pd.concat(schedule_frames, ignore_index=True) if schedule_frames else pd.DataFrame()
    rank_curves = pd.concat(rank_curve_frames, ignore_index=True) if rank_curve_frames else pd.DataFrame()
    residual_ledger = pd.concat(residual_ledger_frames, ignore_index=True) if residual_ledger_frames else pd.DataFrame()
    response_predictions = pd.concat(response_prediction_frames, ignore_index=True) if response_prediction_frames else pd.DataFrame()
    state_effect_matrix = pd.concat(state_effect_frames, ignore_index=True) if state_effect_frames else pd.DataFrame()
    state_loo = pd.concat(state_loo_frames, ignore_index=True) if state_loo_frames else pd.DataFrame()
    overlap = mstc._accepted_overlap(accepted_all, "S0_baseline_static_thresholds")
    overlap["fold"] = int(fold["fold"])
    controller_diag = mstc._controller_state_diagnostics(accepted_all, schedules)
    if not controller_diag.empty:
        controller_diag["fold"] = int(fold["fold"])
    action_utility = mstc._threshold_action_utility(accepted_all, "S0_baseline_static_thresholds")
    if not action_utility.empty:
        action_utility["fold"] = int(fold["fold"])
    action_edge_validation = mstc._threshold_action_edge_validation(accepted_all, schedules, "S0_baseline_static_thresholds")
    if not action_edge_validation.empty:
        action_edge_validation["fold"] = int(fold["fold"])
    action_edge_bucket = mstc._threshold_action_edge_bucket_performance(action_edge_validation)
    if not action_edge_bucket.empty and "fold" not in action_edge_bucket.columns:
        action_edge_bucket["fold"] = int(fold["fold"])
    suppression_utility = mstc._threshold_candidate_suppression_utility(valid_broad, schedules)
    if not suppression_utility.empty:
        suppression_utility["fold"] = int(fold["fold"])
    baseline_accepted_suppression_utility = mstc._threshold_candidate_suppression_utility(
        valid_broad,
        schedules,
        eligible_decision_keys=baseline_keys,
    )
    if not baseline_accepted_suppression_utility.empty:
        baseline_accepted_suppression_utility["fold"] = int(fold["fold"])
    state_bucket_perf = mstc._state_bucket_performance(accepted_all, schedules)
    if not state_bucket_perf.empty:
        state_bucket_perf["fold"] = int(fold["fold"])
    fold_report = {
        "fold": int(fold["fold"]),
        "train_start": fold["train_start"],
        "train_end": fold["train_end"],
        "valid_start": fold["valid_start"],
        "valid_end": fold["valid_end"],
        "train_rows": int(len(train_broad)),
        "valid_rows": int(len(valid_broad)),
        "valid_rows_available": int(fold.get("valid_rows_available", len(valid_broad))),
        "valid_timestamps_available": int(fold.get("valid_timestamps_available", valid_broad["timestamp"].nunique())),
        "split_maturity_contract": {
            "training_entry_end": fold["train_end"],
            "training_outcome_available_before": fold["valid_start"],
            "train_broad": train_broad_maturity_diag,
            "train_deployable": train_deployable_maturity_diag,
            "uses_matured_training_outcomes_only": True,
        },
        "model_reports": model_reports,
        "state_report": state_report,
    }
    return (
        summary,
        by_head,
        overlap,
        controller_diag,
        action_utility,
        action_edge_validation,
        action_edge_bucket,
        suppression_utility,
        baseline_accepted_suppression_utility,
        state_bucket_perf,
        accepted_all,
        schedules,
        rank_curves,
        residual_ledger,
        response_predictions,
        state_effect_matrix,
        state_loo,
        state_timestamp_panel,
        response_model_bundles,
        state_reference_bundle,
        fold_report,
    )


def _aggregate_delta(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    base_cols = [
        "fold",
        "net_pnl",
        "max_drawdown",
        "worst_24h_net_pnl",
        "trade_count",
        "full_sl_rate",
    ]
    available_base_cols = [col for col in base_cols if col in summary.columns]
    base = summary.loc[summary["arm"].eq("S0_baseline_static_thresholds"), available_base_cols].copy()
    base = base.rename(
        columns={
            "net_pnl": "base_net_pnl",
            "max_drawdown": "base_max_drawdown",
            "worst_24h_net_pnl": "base_worst_24h",
            "trade_count": "base_trade_count",
            "full_sl_rate": "base_full_sl_rate",
        }
    )
    merged = summary.merge(base, on="fold", how="left")
    merged["delta_net_pnl"] = merged["net_pnl"] - merged["base_net_pnl"]
    merged["delta_max_drawdown"] = merged["max_drawdown"] - merged["base_max_drawdown"]
    merged["delta_worst_24h"] = merged["worst_24h_net_pnl"] - merged["base_worst_24h"]
    if "base_full_sl_rate" in merged.columns and "full_sl_rate" in merged.columns:
        merged["delta_full_sl_rate"] = merged["full_sl_rate"] - merged["base_full_sl_rate"]
    else:
        merged["delta_full_sl_rate"] = np.nan
    if "base_trade_count" in merged.columns and "trade_count" in merged.columns:
        base_trade_count = pd.to_numeric(merged["base_trade_count"], errors="coerce").replace(0.0, np.nan)
        merged["trade_retention_share"] = pd.to_numeric(merged["trade_count"], errors="coerce") / base_trade_count
        merged.loc[merged["arm"].eq("S0_baseline_static_thresholds"), "trade_retention_share"] = 1.0
    else:
        merged["trade_retention_share"] = np.nan
    rows = []
    for arm, g in merged.groupby("arm", sort=False):
        rows.append(
            {
                "arm": arm,
                "folds": int(g["fold"].nunique()),
                "median_delta_net_pnl": float(g["delta_net_pnl"].median()),
                "mean_delta_net_pnl": float(g["delta_net_pnl"].mean()),
                "q25_delta_net_pnl": float(g["delta_net_pnl"].quantile(0.25)),
                "positive_delta_share": float((g["delta_net_pnl"] > 0.0).mean()),
                "median_delta_max_drawdown": float(g["delta_max_drawdown"].median()),
                "median_delta_worst_24h": float(g["delta_worst_24h"].median()),
                "median_trade_count": float(g["trade_count"].median()),
                "median_trade_retention_share": float(pd.to_numeric(g["trade_retention_share"], errors="coerce").median()),
                "median_delta_full_sl_rate": float(pd.to_numeric(g["delta_full_sl_rate"], errors="coerce").median()),
            }
        )
    return pd.DataFrame(rows)


def _aggregate_suppression_utility(suppression_utility: pd.DataFrame) -> pd.DataFrame:
    result_cols = [
        "arm",
        "scope",
        "scope_value",
        "folds_with_suppression",
        "suppressed_candidates",
        "suppressed_net_return_sum",
        "suppressed_loss_avoided",
        "suppressed_winner_pnl_sacrificed",
        "realized_defensive_success",
        "positive_suppression_fold_share",
        "mean_suppressed_full_sl_rate",
        "mean_suppressed_timeout_rate",
    ]
    if suppression_utility.empty:
        return pd.DataFrame(columns=result_cols)
    rows: list[dict[str, Any]] = []
    for (arm, scope, scope_value), g in suppression_utility.groupby(["arm", "scope", "scope_value"], sort=False):
        defensive = pd.to_numeric(g["realized_defensive_success"], errors="coerce").fillna(0.0)
        rows.append(
            {
                "arm": arm,
                "scope": scope,
                "scope_value": scope_value,
                "folds_with_suppression": int(g["fold"].nunique()) if "fold" in g else int(len(g)),
                "suppressed_candidates": int(pd.to_numeric(g["suppressed_candidates"], errors="coerce").fillna(0).sum()),
                "suppressed_net_return_sum": float(pd.to_numeric(g["suppressed_net_return_sum"], errors="coerce").fillna(0.0).sum()),
                "suppressed_loss_avoided": float(pd.to_numeric(g["suppressed_loss_avoided"], errors="coerce").fillna(0.0).sum()),
                "suppressed_winner_pnl_sacrificed": float(
                    pd.to_numeric(g["suppressed_winner_pnl_sacrificed"], errors="coerce").fillna(0.0).sum()
                ),
                "realized_defensive_success": float(defensive.sum()),
                "positive_suppression_fold_share": float((defensive > 0.0).mean()),
                "mean_suppressed_full_sl_rate": float(pd.to_numeric(g["suppressed_full_sl_rate"], errors="coerce").mean()),
                "mean_suppressed_timeout_rate": float(pd.to_numeric(g["suppressed_timeout_rate"], errors="coerce").mean()),
            }
        )
    return pd.DataFrame(rows, columns=result_cols)


def _aggregate_action_utility(action_utility: pd.DataFrame) -> pd.DataFrame:
    result_cols = [
        "arm",
        "scope",
        "scope_value",
        "folds_with_action",
        "action_entrants",
        "action_removed",
        "action_removed_loss_avoided",
        "action_removed_winner_pnl_sacrificed",
        "action_defensive_success",
        "positive_action_fold_share",
        "mean_action_net_pnl_delta",
    ]
    if action_utility.empty:
        return pd.DataFrame(columns=result_cols)
    rows: list[dict[str, Any]] = []
    for (arm, scope, scope_value), g in action_utility.groupby(["arm", "scope", "scope_value"], sort=False):
        defensive = pd.to_numeric(g["defensive_success"], errors="coerce").fillna(0.0)
        rows.append(
            {
                "arm": arm,
                "scope": scope,
                "scope_value": scope_value,
                "folds_with_action": int(g["fold"].nunique()) if "fold" in g else int(len(g)),
                "action_entrants": int(pd.to_numeric(g["entrants"], errors="coerce").fillna(0).sum()),
                "action_removed": int(pd.to_numeric(g["removed"], errors="coerce").fillna(0).sum()),
                "action_removed_loss_avoided": float(
                    pd.to_numeric(g["removed_loss_avoided"], errors="coerce").fillna(0.0).sum()
                ),
                "action_removed_winner_pnl_sacrificed": float(
                    pd.to_numeric(g["removed_winner_pnl_sacrificed"], errors="coerce").fillna(0.0).sum()
                ),
                "action_defensive_success": float(defensive.sum()),
                "positive_action_fold_share": float((defensive > 0.0).mean()),
                "mean_action_net_pnl_delta": float(
                    pd.to_numeric(g["net_action_pnl_delta"], errors="coerce").fillna(0.0).mean()
                ),
            }
        )
    return pd.DataFrame(rows, columns=result_cols)


def _state_component_group(name: str) -> str:
    text = str(name).lower()
    if "latent" in text or "gmm" in text:
        return "latent_shadow"
    if "reliability" in text or "coverage" in text or "novelty" in text or "ood" in text or "uncertainty" in text:
        return "state_reliability"
    if "delever" in text or "oi" in text or "fund" in text:
        return "leverage_oi_funding"
    if "liq" in text or "spread" in text or "amihud" in text:
        return "liquidity_proxy"
    if "rv" in text or "vol" in text or "tail" in text:
        return "volatility_tail"
    if "shock" in text or "ret" in text or "down" in text or "up" in text:
        return "return_shock"
    if "trend" in text:
        return "trend"
    if "compression" in text or "consolidation" in text:
        return "compression_consolidation"
    if "transition" in text or "drift" in text:
        return "transition_drift"
    return "other_market_state"


def _state_head_registry(fold_reports: list[dict[str, Any]]) -> pd.DataFrame:
    """Aggregate fold-local state-head health into a flat audit registry.

    The registry is intentionally descriptive.  It records whether state heads
    were active, fallback-only, or shadow-disabled without using per-head PnL
    attribution or strategy-specific outcomes.
    """

    rows: list[dict[str, Any]] = []
    for fold_report in fold_reports:
        fold = int(fold_report.get("fold", 0) or 0)
        state_report = dict(fold_report.get("state_report") or {})

        for axis, sources in dict(state_report.get("axis_sources") or {}).items():
            source_count = len(list(sources or []))
            rows.append(
                {
                    "fold": fold,
                    "state_level": "observed_axis",
                    "state_head": str(axis),
                    "component_group": _state_component_group(str(axis)),
                    "status": "active",
                    "mode": "observed_axis_robust_z",
                    "trained": 1,
                    "fallback": 0,
                    "shadow": 0,
                    "source_count": int(source_count),
                    "validation_rows": np.nan,
                    "validation_top_decile_lift": np.nan,
                    "validation_tail_average_precision": np.nan,
                    "validation_tail_ap_lift_p90": np.nan,
                    "validation_tail_brier_p90": np.nan,
                    "validation_tail_ece_5bin": np.nan,
                    "validation_tail_false_alarm_rate_p90": np.nan,
                    "validation_tail_recall_p90": np.nan,
                    "validation_collapsed": np.nan,
                    "oof_coverage": np.nan,
                    "target_rows": np.nan,
                    "target_std": np.nan,
                    "disable_reason": "",
                }
            )

        forecast_report = dict(state_report.get("forecast_report") or {})
        for target_name, target_report_any in dict(forecast_report.get("targets") or {}).items():
            target_report = dict(target_report_any or {})
            mode = str(target_report.get("mode", "unknown"))
            is_trained = mode == "gbm_soft_empirical_cdf_target"
            is_fallback = "fallback" in mode
            reason = str(target_report.get("oof_reason") or "")
            if is_fallback and not reason:
                reason = "current_axis_fallback"
            rows.append(
                {
                    "fold": fold,
                    "state_level": "forecast",
                    "state_head": str(target_name),
                    "component_group": _state_component_group(str(target_name)),
                    "status": "active" if is_trained else "fallback" if is_fallback else "unknown",
                    "mode": mode,
                    "trained": int(is_trained),
                    "fallback": int(is_fallback),
                    "shadow": 0,
                    "source_count": int(forecast_report.get("features_used", 0) or 0),
                    "validation_rows": target_report.get("validation_rows"),
                    "validation_top_decile_lift": target_report.get("validation_top_decile_lift"),
                    "validation_tail_average_precision": target_report.get("validation_tail_average_precision"),
                    "validation_tail_ap_lift_p90": target_report.get("validation_tail_ap_lift_p90"),
                    "validation_tail_brier_p90": target_report.get("validation_tail_brier_p90"),
                    "validation_tail_ece_5bin": target_report.get("validation_tail_ece_5bin"),
                    "validation_tail_false_alarm_rate_p90": target_report.get("validation_tail_false_alarm_rate_p90"),
                    "validation_tail_recall_p90": target_report.get("validation_tail_recall_p90"),
                    "validation_collapsed": target_report.get("validation_collapsed"),
                    "oof_coverage": target_report.get("oof_coverage"),
                    "target_rows": target_report.get("rows"),
                    "target_std": target_report.get("target_std"),
                    "disable_reason": reason,
                }
            )

        latent_report = dict(state_report.get("latent_report") or {})
        if latent_report:
            latent_mode = str(latent_report.get("mode", "unknown"))
            if latent_mode == "shadow_disabled_by_default":
                status = "shadow_disabled"
                trained = 0
                fallback = 0
                shadow = 1
            elif latent_mode == "uniform_fallback":
                status = "fallback"
                trained = 0
                fallback = 1
                shadow = 0
            else:
                status = "shadow" if "latent" in latent_mode or "gmm" in latent_mode else "unknown"
                trained = int(status == "shadow")
                fallback = 0
                shadow = int(status == "shadow")
            rows.append(
                {
                    "fold": fold,
                    "state_level": "latent",
                    "state_head": "latent_gmm_probabilities",
                    "component_group": "latent_shadow",
                    "status": status,
                    "mode": latent_mode,
                    "trained": trained,
                    "fallback": fallback,
                    "shadow": shadow,
                    "source_count": int(len(list(latent_report.get("state_cols") or []))),
                    "validation_rows": np.nan,
                    "validation_top_decile_lift": np.nan,
                    "validation_tail_average_precision": np.nan,
                    "validation_tail_ap_lift_p90": np.nan,
                    "validation_tail_brier_p90": np.nan,
                    "validation_tail_ece_5bin": np.nan,
                    "validation_tail_false_alarm_rate_p90": np.nan,
                    "validation_tail_recall_p90": np.nan,
                    "validation_collapsed": np.nan,
                    "oof_coverage": np.nan,
                    "target_rows": np.nan,
                    "target_std": np.nan,
                    "disable_reason": str(latent_report.get("reason") or ""),
                }
            )

    if not rows:
        return pd.DataFrame()

    raw = pd.DataFrame(rows)
    numeric_cols = [
        "validation_rows",
        "validation_top_decile_lift",
        "validation_tail_average_precision",
        "validation_tail_ap_lift_p90",
        "validation_tail_brier_p90",
        "validation_tail_ece_5bin",
        "validation_tail_false_alarm_rate_p90",
        "validation_tail_recall_p90",
        "oof_coverage",
        "target_rows",
        "target_std",
        "source_count",
    ]
    for col in numeric_cols:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw["validation_collapsed_bool"] = raw["validation_collapsed"].map(
        lambda x: bool(x) if isinstance(x, (bool, np.bool_)) else False
    )

    grouped_rows: list[dict[str, Any]] = []
    keys = ["state_level", "state_head", "component_group"]
    for (state_level, state_head, component_group), g in raw.groupby(keys, sort=False):
        status_counts = g["status"].astype(str).value_counts().to_dict()
        disable_reasons = sorted({x for x in g["disable_reason"].astype(str).tolist() if x})
        trained_folds = int(pd.to_numeric(g["trained"], errors="coerce").fillna(0).sum())
        fallback_folds = int(pd.to_numeric(g["fallback"], errors="coerce").fillna(0).sum())
        shadow_folds = int(pd.to_numeric(g["shadow"], errors="coerce").fillna(0).sum())
        folds_seen = int(g["fold"].nunique())
        if trained_folds > 0:
            aggregate_status = "active"
        elif fallback_folds > 0:
            aggregate_status = "fallback"
        elif shadow_folds > 0:
            aggregate_status = "shadow_disabled"
        else:
            aggregate_status = "unknown"
        grouped_rows.append(
            {
                "state_level": state_level,
                "state_head": state_head,
                "component_group": component_group,
                "aggregate_status": aggregate_status,
                "folds_seen": folds_seen,
                "trained_folds": trained_folds,
                "fallback_folds": fallback_folds,
                "shadow_disabled_folds": shadow_folds,
                "active_fold_share": float(trained_folds / max(folds_seen, 1)),
                "fallback_fold_share": float(fallback_folds / max(folds_seen, 1)),
                "mean_source_count": float(g["source_count"].mean()) if g["source_count"].notna().any() else np.nan,
                "mean_validation_rows": float(g["validation_rows"].mean()) if g["validation_rows"].notna().any() else np.nan,
                "mean_validation_top_decile_lift": (
                    float(g["validation_top_decile_lift"].mean())
                    if g["validation_top_decile_lift"].notna().any()
                    else np.nan
                ),
                "mean_tail_average_precision": (
                    float(g["validation_tail_average_precision"].mean())
                    if g["validation_tail_average_precision"].notna().any()
                    else np.nan
                ),
                "mean_tail_ap_lift_p90": (
                    float(g["validation_tail_ap_lift_p90"].mean())
                    if g["validation_tail_ap_lift_p90"].notna().any()
                    else np.nan
                ),
                "mean_tail_brier_p90": (
                    float(g["validation_tail_brier_p90"].mean())
                    if g["validation_tail_brier_p90"].notna().any()
                    else np.nan
                ),
                "mean_tail_ece_5bin": (
                    float(g["validation_tail_ece_5bin"].mean())
                    if g["validation_tail_ece_5bin"].notna().any()
                    else np.nan
                ),
                "mean_tail_false_alarm_rate_p90": (
                    float(g["validation_tail_false_alarm_rate_p90"].mean())
                    if g["validation_tail_false_alarm_rate_p90"].notna().any()
                    else np.nan
                ),
                "mean_tail_recall_p90": (
                    float(g["validation_tail_recall_p90"].mean())
                    if g["validation_tail_recall_p90"].notna().any()
                    else np.nan
                ),
                "collapsed_folds": int(g["validation_collapsed_bool"].sum()),
                "positive_validation_lift_share": (
                    float((g["validation_top_decile_lift"].dropna() > 0.0).mean())
                    if g["validation_top_decile_lift"].notna().any()
                    else np.nan
                ),
                "mean_oof_coverage": float(g["oof_coverage"].mean()) if g["oof_coverage"].notna().any() else np.nan,
                "min_oof_coverage": float(g["oof_coverage"].min()) if g["oof_coverage"].notna().any() else np.nan,
                "mean_target_rows": float(g["target_rows"].mean()) if g["target_rows"].notna().any() else np.nan,
                "mean_target_std": float(g["target_std"].mean()) if g["target_std"].notna().any() else np.nan,
                "status_counts": json.dumps(status_counts, sort_keys=True),
                "disable_reasons": ";".join(disable_reasons),
            }
        )
    out = pd.DataFrame(grouped_rows)
    sort_cols = ["state_level", "component_group", "state_head"]
    return out.sort_values(sort_cols).reset_index(drop=True)


def _state_output_redundancy(state_panel: pd.DataFrame, *, threshold: float = 0.96) -> pd.DataFrame:
    if state_panel.empty:
        return pd.DataFrame()
    work = state_panel.copy()
    if "split" in work.columns:
        work = work.loc[work["split"].astype(str).eq("valid")].copy()
    state_cols = [
        c
        for c in work.columns
        if (str(c).startswith("state_") or str(c).startswith("forecast_"))
        and pd.api.types.is_numeric_dtype(work[c])
    ]
    if len(state_cols) < 2:
        return pd.DataFrame()
    corr = work[state_cols].apply(pd.to_numeric, errors="coerce").corr(method="spearman").abs()
    rows: list[dict[str, Any]] = []
    for col in state_cols:
        others = corr[col].drop(index=col, errors="ignore").replace([np.inf, -np.inf], np.nan).dropna()
        if others.empty:
            rows.append(
                {
                    "state_head": col,
                    "max_abs_spearman_corr": np.nan,
                    "redundant_with": "",
                    "redundancy_group": _state_component_group(col),
                    "redundancy_flag": False,
                }
            )
            continue
        best = str(others.idxmax())
        value = float(others.loc[best])
        rows.append(
            {
                "state_head": col,
                "max_abs_spearman_corr": value,
                "redundant_with": best,
                "redundancy_group": _state_component_group(col),
                "redundancy_flag": bool(value >= float(threshold)),
            }
        )
    return pd.DataFrame(rows)


def _state_response_gate(state_effect_matrix: pd.DataFrame) -> pd.DataFrame:
    if state_effect_matrix.empty:
        return pd.DataFrame()
    work = state_effect_matrix.copy()
    work["state_feature"] = work["state_feature"].astype(str)
    work["target"] = work["target"].astype(str)
    useful_targets = {
        "pred_resid_utility",
        "pred_resid_utility_lcb",
        "pred_resid_full_sl",
        "pred_resid_timeout",
    }
    work = work.loc[work["target"].isin(useful_targets)].copy()
    if work.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for state_feature, g in work.groupby("state_feature", sort=False):
        effect = pd.to_numeric(g["target_q90_minus_q10"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        spearman = pd.to_numeric(g.get("spearman"), errors="coerce").replace([np.inf, -np.inf], np.nan)
        valid_effect = effect.dropna()
        signs = np.sign(valid_effect.to_numpy(dtype=float))
        signs = signs[signs != 0]
        if signs.size:
            pos_share = float((signs > 0).mean())
            sign_stability = max(pos_share, 1.0 - pos_share)
        else:
            sign_stability = np.nan
        mean_abs_effect = float(valid_effect.abs().mean()) if not valid_effect.empty else np.nan
        mean_abs_spearman = float(spearman.abs().mean()) if spearman.notna().any() else np.nan
        response_pass = bool(
            (np.isfinite(mean_abs_effect) and mean_abs_effect >= 0.005)
            or (np.isfinite(mean_abs_spearman) and mean_abs_spearman >= 0.02)
        )
        rows.append(
            {
                "state_head": str(state_feature),
                "response_effect_rows": int(len(g)),
                "response_mean_abs_q90_q10": mean_abs_effect,
                "response_max_abs_q90_q10": float(valid_effect.abs().max()) if not valid_effect.empty else np.nan,
                "response_mean_abs_spearman": mean_abs_spearman,
                "response_sign_stability": sign_stability,
                "response_gate_pass": response_pass,
            }
        )
    return pd.DataFrame(rows)


def _arm_action_summary(schedules: pd.DataFrame) -> pd.DataFrame:
    if schedules.empty or "arm" not in schedules.columns:
        return pd.DataFrame()
    work = schedules.copy()
    base = pd.to_numeric(work.get("base_threshold"), errors="coerce")
    state = pd.to_numeric(work.get("state_threshold"), errors="coerce")
    raw = pd.to_numeric(work.get("raw_state_threshold"), errors="coerce")
    suppressed = pd.to_numeric(work.get("suppressed_candidate_count"), errors="coerce").fillna(0.0)
    work["_threshold_raised"] = ((state - base) > 1e-9) | ((raw - base) > 1e-9)
    work["_suppressed_candidates"] = suppressed
    rows = []
    for arm, g in work.groupby("arm", sort=False):
        rows.append(
            {
                "arm": str(arm),
                "schedule_rows": int(len(g)),
                "threshold_raise_count": int(g["_threshold_raised"].sum()),
                "threshold_raise_share": float(g["_threshold_raised"].mean()) if len(g) else 0.0,
                "suppressed_candidate_count": int(g["_suppressed_candidates"].sum()),
                "mean_state_ood_share": (
                    float(pd.to_numeric(g.get("state_ood_share"), errors="coerce").mean())
                    if "state_ood_share" in g
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _aggregate_state_leave_one_out(state_loo: pd.DataFrame) -> pd.DataFrame:
    if state_loo.empty:
        return pd.DataFrame()
    work = state_loo.copy()
    work["state_head"] = work["state_head"].astype(str)
    work["arm"] = work["arm"].astype(str)
    numeric_cols = [
        "increment_net_pnl",
        "accepted_jaccard_vs_full",
        "delta_trade_count",
        "full_threshold_raise_count",
        "loo_threshold_raise_count",
        "state_head_removed_loss_avoided",
        "state_head_removed_winner_pnl_sacrificed",
        "state_head_defensive_success",
        "state_head_net_action_pnl_delta",
    ]
    for col in numeric_cols:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    rows: list[dict[str, Any]] = []
    for (state_head, arm), g in work.groupby(["state_head", "arm"], sort=False):
        inc = pd.to_numeric(g["increment_net_pnl"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        defensive = (
            pd.to_numeric(g.get("state_head_defensive_success"), errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        loss_avoided = (
            pd.to_numeric(g.get("state_head_removed_loss_avoided"), errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
        winners_sacrificed = (
            pd.to_numeric(g.get("state_head_removed_winner_pnl_sacrificed"), errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
        action_delta = (
            pd.to_numeric(g.get("state_head_net_action_pnl_delta"), errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        rows.append(
            {
                "state_head": state_head,
                "action_arm_hint": arm,
                "loo_replay_folds": int(g["fold"].nunique()) if "fold" in g else int(len(g)),
                "loo_mode": ";".join(sorted(set(g.get("leave_one_out_mode", pd.Series("", index=g.index)).astype(str)))),
                "loo_median_increment_net_pnl": float(inc.median()) if not inc.empty else np.nan,
                "loo_mean_increment_net_pnl": float(inc.mean()) if not inc.empty else np.nan,
                "loo_q25_increment_net_pnl": float(inc.quantile(0.25)) if not inc.empty else np.nan,
                "loo_positive_increment_share": float((inc > 0.0).mean()) if not inc.empty else np.nan,
                "loo_mean_accepted_jaccard": (
                    float(pd.to_numeric(g.get("accepted_jaccard_vs_full"), errors="coerce").mean())
                    if "accepted_jaccard_vs_full" in g
                    else np.nan
                ),
                "loo_mean_delta_trade_count": (
                    float(pd.to_numeric(g.get("delta_trade_count"), errors="coerce").mean())
                    if "delta_trade_count" in g
                    else np.nan
                ),
                "loo_mean_threshold_raise_delta": (
                    float(
                        (
                            pd.to_numeric(g.get("full_threshold_raise_count"), errors="coerce")
                            - pd.to_numeric(g.get("loo_threshold_raise_count"), errors="coerce")
                        ).mean()
                    )
                    if {"full_threshold_raise_count", "loo_threshold_raise_count"}.issubset(g.columns)
                    else np.nan
                ),
                "loo_state_head_defensive_success": float(defensive.sum()) if not defensive.empty else np.nan,
                "loo_state_head_median_defensive_success": float(defensive.median()) if not defensive.empty else np.nan,
                "loo_state_head_positive_defensive_share": float((defensive > 0.0).mean()) if not defensive.empty else np.nan,
                "loo_state_head_loss_avoided": float(loss_avoided.sum()) if len(loss_avoided) else np.nan,
                "loo_state_head_winner_pnl_sacrificed": float(winners_sacrificed.sum()) if len(winners_sacrificed) else np.nan,
                "loo_state_head_net_action_pnl_delta": float(action_delta.sum()) if not action_delta.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _state_head_arm_hint(state_head: str) -> str:
    if str(state_head).startswith("forecast_"):
        return "S2_observed_forecast_shared_response"
    return "S1_observed_axes_shared_response"


def _market_state_activation_registry(
    state_head_registry: pd.DataFrame,
    state_panel: pd.DataFrame,
    state_effect_matrix: pd.DataFrame,
    schedules: pd.DataFrame,
    state_loo_aggregate: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if state_head_registry.empty:
        return pd.DataFrame()
    out = state_head_registry.copy()
    out["state_head"] = out["state_head"].astype(str)
    redundancy = _state_output_redundancy(state_panel)
    response = _state_response_gate(state_effect_matrix)
    action = _arm_action_summary(schedules)
    loo = state_loo_aggregate.copy() if state_loo_aggregate is not None and not state_loo_aggregate.empty else pd.DataFrame()
    if not redundancy.empty:
        out = out.merge(redundancy, on="state_head", how="left", validate="one_to_one")
    else:
        out["max_abs_spearman_corr"] = np.nan
        out["redundant_with"] = ""
        out["redundancy_group"] = out["component_group"]
        out["redundancy_flag"] = False
    if not response.empty:
        out = out.merge(response, on="state_head", how="left", validate="one_to_one")
    else:
        out["response_effect_rows"] = 0
        out["response_mean_abs_q90_q10"] = np.nan
        out["response_max_abs_q90_q10"] = np.nan
        out["response_mean_abs_spearman"] = np.nan
        out["response_sign_stability"] = np.nan
        out["response_gate_pass"] = False
    out["action_arm_hint"] = out["state_head"].map(_state_head_arm_hint)
    if not action.empty:
        out = out.merge(action, left_on="action_arm_hint", right_on="arm", how="left", validate="many_to_one")
        out = out.drop(columns=["arm"])
    else:
        out["schedule_rows"] = 0
        out["threshold_raise_count"] = 0
        out["threshold_raise_share"] = 0.0
        out["suppressed_candidate_count"] = 0
        out["mean_state_ood_share"] = np.nan
    if not loo.empty:
        out = out.merge(loo, on=["state_head", "action_arm_hint"], how="left", validate="one_to_one")
    else:
        out["loo_replay_folds"] = 0
        out["loo_mode"] = ""
        out["loo_median_increment_net_pnl"] = np.nan
        out["loo_mean_increment_net_pnl"] = np.nan
        out["loo_q25_increment_net_pnl"] = np.nan
        out["loo_positive_increment_share"] = np.nan
        out["loo_mean_accepted_jaccard"] = np.nan
        out["loo_mean_delta_trade_count"] = np.nan
        out["loo_mean_threshold_raise_delta"] = np.nan
        out["loo_state_head_defensive_success"] = np.nan
        out["loo_state_head_median_defensive_success"] = np.nan
        out["loo_state_head_positive_defensive_share"] = np.nan
        out["loo_state_head_loss_avoided"] = np.nan
        out["loo_state_head_winner_pnl_sacrificed"] = np.nan
        out["loo_state_head_net_action_pnl_delta"] = np.nan

    out["forecast_skill_gate_pass"] = True
    is_forecast = out["state_level"].astype(str).eq("forecast")
    out.loc[is_forecast, "forecast_skill_gate_pass"] = (
        out.loc[is_forecast, "aggregate_status"].astype(str).eq("active")
        & (pd.to_numeric(out.loc[is_forecast, "active_fold_share"], errors="coerce").fillna(0.0) >= 0.50)
        & (pd.to_numeric(out.loc[is_forecast, "min_oof_coverage"], errors="coerce").fillna(0.0) >= 0.50)
        & (pd.to_numeric(out.loc[is_forecast, "collapsed_folds"], errors="coerce").fillna(0.0) == 0.0)
        & (
            (pd.to_numeric(out.loc[is_forecast, "positive_validation_lift_share"], errors="coerce").fillna(0.0) >= 0.50)
            | (pd.to_numeric(out.loc[is_forecast, "mean_tail_ap_lift_p90"], errors="coerce").fillna(0.0) > 0.0)
        )
    )
    out["response_gate_pass"] = out["response_gate_pass"].eq(True)
    out["redundancy_flag"] = out["redundancy_flag"].eq(True)
    out["action_gate_pass"] = (
        (pd.to_numeric(out["threshold_raise_count"], errors="coerce").fillna(0) > 0)
        | (pd.to_numeric(out["suppressed_candidate_count"], errors="coerce").fillna(0) > 0)
    )
    out["leave_one_out_gate_pass"] = (
        (pd.to_numeric(out.get("loo_replay_folds"), errors="coerce").fillna(0) > 0)
        & (pd.to_numeric(out.get("loo_median_increment_net_pnl"), errors="coerce").fillna(-np.inf) > 0.0)
        & (pd.to_numeric(out.get("loo_positive_increment_share"), errors="coerce").fillna(0.0) >= 0.50)
    )
    out["defensive_action_gate_pass"] = (
        (pd.to_numeric(out.get("loo_replay_folds"), errors="coerce").fillna(0) > 0)
        & (pd.to_numeric(out.get("loo_state_head_defensive_success"), errors="coerce").fillna(-np.inf) > 0.0)
        & (
            pd.to_numeric(out.get("loo_state_head_loss_avoided"), errors="coerce").fillna(0.0)
            > pd.to_numeric(out.get("loo_state_head_winner_pnl_sacrificed"), errors="coerce").fillna(0.0)
        )
    )

    statuses: list[str] = []
    reasons_col: list[str] = []
    loo_status: list[str] = []
    for _, row in out.iterrows():
        reasons: list[str] = []
        aggregate_status = str(row.get("aggregate_status", ""))
        if aggregate_status != "active":
            reasons.append(aggregate_status or "not_active")
        if bool(row.get("state_level") == "forecast") and not bool(row.get("forecast_skill_gate_pass")):
            reasons.append("weak_or_unstable_forecast_skill")
        if bool(row.get("redundancy_flag")) and not bool(row.get("response_gate_pass")):
            reasons.append("redundant_without_response_effect")
        if not bool(row.get("response_gate_pass")):
            reasons.append("weak_response_effect")
        if not bool(row.get("action_gate_pass")):
            reasons.append("no_material_threshold_action")
        if not bool(row.get("leave_one_out_gate_pass")):
            reasons.append("no_positive_leave_one_out_increment")
        if not bool(row.get("defensive_action_gate_pass")):
            reasons.append("state_action_sacrifices_winners")

        if aggregate_status != "active":
            status = "shadow"
        elif (
            "redundant_without_response_effect" in reasons
            or "weak_or_unstable_forecast_skill" in reasons
            or "no_positive_leave_one_out_increment" in reasons
            or "state_action_sacrifices_winners" in reasons
        ):
            status = "disabled_candidate"
        elif "weak_response_effect" in reasons and "no_material_threshold_action" in reasons:
            status = "shadow"
        else:
            status = "active_candidate"
        statuses.append(status)
        reasons_col.append(";".join(dict.fromkeys(reasons)))
        loo_status.append(
            "required_before_promotion"
            if status == "active_candidate"
            else "not_required_for_shadow_or_disabled_candidate"
        )
    out["recommended_status"] = statuses
    out["activation_disable_reason"] = reasons_col
    out["leave_one_head_out_status"] = loo_status
    out["activation_registry_version"] = "market_state_activation_registry_v1"
    sort_cols = ["recommended_status", "state_level", "component_group", "state_head"]
    return out.sort_values(sort_cols).reset_index(drop=True)


def _source_contract_audit(fold_reports: list[dict[str, Any]]) -> dict[str, Any]:
    splits: dict[str, Any] = {}
    overall_passed = True
    for fold_report in fold_reports:
        fold = int(fold_report.get("fold", 0) or 0)
        state_report = dict(fold_report.get("state_report") or {})
        market_source = dict(state_report.get("market_state_source") or {})
        feature_store = dict(state_report.get("feature_store") or {})
        for split, source_report_raw in dict(market_source or {}).items():
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
            splits[f"fold_{fold}_{split}"] = {
                "fold": fold,
                "split": str(split),
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
                "feature_store_aggregation_contract": fs_report.get("aggregation_contract"),
                "feature_store_tail_reference_source": fs_report.get("tail_reference_source"),
                "feature_store_tail_reference_role": fs_report.get("tail_reference_role"),
                "passed": split_passed,
            }
    return {
        "audit_version": "market_state_walkforward_source_contract_audit_v1",
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


def _fold_split_universe_contracts(fold_reports: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    splits: dict[str, dict[str, Any]] = {}
    for fold_report in fold_reports:
        fold = int(fold_report.get("fold", 0) or 0)
        state_report = dict(fold_report.get("state_report") or {})
        feature_store = dict(state_report.get("feature_store") or {})
        market_source = dict(state_report.get("market_state_source") or {})
        for split in ("train", "valid"):
            fs_report = dict(feature_store.get(split) or {})
            source_report = dict(market_source.get(split) or {})
            universe = dict(fs_report.get("universe_contract") or {})
            eligible_symbols = [str(symbol) for symbol in universe.get("eligible_symbols") or []]
            excluded_symbols = [str(symbol) for symbol in universe.get("excluded_symbols") or []]
            excluded_reasons = {
                str(symbol): str(reason)
                for symbol, reason in dict(universe.get("excluded_reasons") or {}).items()
            }
            key = f"fold_{fold}_{split}"
            splits[key] = {
                "fold": fold,
                "split": split,
                "source": source_report.get("source"),
                "production_safe": bool(source_report.get("production_safe") is True),
                "candidate_fallback_enabled": bool(source_report.get("allow_candidate_fallback") is True),
                "strategy_independent": True,
                "candidate_independent": source_report.get("source") == "feature_store_market_aggregates"
                and not bool(source_report.get("allow_candidate_fallback") is True),
                "actual_order_book_features_allowed": False,
                "universe_definition_version": universe.get("universe_definition_version"),
                "universe_source": universe.get("source"),
                "feature_dir": universe.get("feature_dir"),
                "minimum_history": universe.get("minimum_history"),
                "minimum_volume": universe.get("minimum_volume"),
                "oi_coverage_requirements": universe.get("oi_coverage_requirements"),
                "funding_coverage_requirements": universe.get("funding_coverage_requirements"),
                "symbol_cap": universe.get("symbol_cap"),
                "available_symbol_count": universe.get("available_symbol_count"),
                "eligible_symbol_count": universe.get("eligible_symbol_count"),
                "eligible_symbols": eligible_symbols,
                "excluded_symbols": excluded_symbols,
                "excluded_symbols_and_reasons": excluded_reasons,
                "selection_reason": universe.get("selection_reason"),
                "feature_store_timestamp_coverage": fs_report.get("timestamp_coverage"),
                "feature_store_symbols_read": fs_report.get("symbols_read"),
            }
    return splits


def _market_state_universe_contract(fold_reports: list[dict[str, Any]]) -> dict[str, Any]:
    splits = _fold_split_universe_contracts(fold_reports)
    failures: list[str] = []
    if not splits:
        failures.append("no_fold_split_universe_contracts")
    eligible_sets = {
        tuple(row.get("eligible_symbols") or [])
        for row in splits.values()
        if isinstance(row.get("eligible_symbols"), list)
    }
    common_eligible_symbols = list(next(iter(eligible_sets))) if len(eligible_sets) == 1 else []
    excluded_union: dict[str, str] = {}
    versions: set[str] = set()
    feature_dirs: set[str] = set()
    symbol_caps: set[int] = set()
    minimum_history: set[str] = set()
    minimum_volume: set[str] = set()
    oi_requirements: set[str] = set()
    funding_requirements: set[str] = set()
    available_counts: set[int] = set()
    eligible_counts: set[int] = set()
    for key, row in splits.items():
        eligible_symbols = list(row.get("eligible_symbols") or [])
        excluded_symbols = list(row.get("excluded_symbols") or [])
        excluded_reasons = dict(row.get("excluded_symbols_and_reasons") or {})
        available_count = row.get("available_symbol_count")
        eligible_count = row.get("eligible_symbol_count")
        if row.get("source") != "feature_store_market_aggregates":
            failures.append(f"{key}_source_not_feature_store_market_aggregates")
        if row.get("production_safe") is not True:
            failures.append(f"{key}_not_production_safe")
        if row.get("candidate_fallback_enabled") is not False:
            failures.append(f"{key}_candidate_fallback_enabled")
        if row.get("strategy_independent") is not True:
            failures.append(f"{key}_not_strategy_independent")
        if row.get("candidate_independent") is not True:
            failures.append(f"{key}_not_candidate_independent")
        if row.get("actual_order_book_features_allowed") is not False:
            failures.append(f"{key}_allows_actual_order_book_features")
        if not eligible_symbols:
            failures.append(f"{key}_missing_eligible_symbols")
        if len(set(eligible_symbols)) != len(eligible_symbols):
            failures.append(f"{key}_duplicate_eligible_symbols")
        if eligible_count is None or int(eligible_count) != len(eligible_symbols):
            failures.append(f"{key}_eligible_symbol_count_mismatch")
        if available_count is None or int(available_count) < len(eligible_symbols):
            failures.append(f"{key}_available_symbol_count_lt_eligible")
        missing_excluded_reasons = sorted(symbol for symbol in excluded_symbols if symbol not in excluded_reasons)
        if missing_excluded_reasons:
            failures.append(f"{key}_excluded_symbols_missing_reasons")
        if row.get("universe_definition_version"):
            versions.add(str(row.get("universe_definition_version")))
        if row.get("feature_dir"):
            feature_dirs.add(str(row.get("feature_dir")))
        if row.get("symbol_cap") is not None:
            symbol_caps.add(int(row.get("symbol_cap")))
        if row.get("minimum_history"):
            minimum_history.add(str(row.get("minimum_history")))
        if row.get("minimum_volume"):
            minimum_volume.add(str(row.get("minimum_volume")))
        if row.get("oi_coverage_requirements"):
            oi_requirements.add(str(row.get("oi_coverage_requirements")))
        if row.get("funding_coverage_requirements"):
            funding_requirements.add(str(row.get("funding_coverage_requirements")))
        if available_count is not None:
            available_counts.add(int(available_count))
        if eligible_count is not None:
            eligible_counts.add(int(eligible_count))
        excluded_union.update({str(symbol): str(reason) for symbol, reason in excluded_reasons.items()})
    if len(eligible_sets) > 1:
        failures.append("eligible_symbol_list_not_constant_across_fold_splits")
    return {
        "contract_version": "market_state_universe_contract_v1",
        "generated_by": "run_market_state_threshold_controller_walkforward",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "required_source": "feature_store_market_aggregates",
        "universe_definition_versions": sorted(versions),
        "strategy_independent": True,
        "candidate_independent": bool(splits) and all(row.get("candidate_independent") is True for row in splits.values()),
        "actual_order_book_features_allowed": False,
        "candidate_population_fallback_enabled": any(
            bool(row.get("candidate_fallback_enabled")) for row in splits.values()
        ),
        "feature_dirs": sorted(feature_dirs),
        "symbol_caps": sorted(symbol_caps),
        "available_symbol_counts": sorted(available_counts),
        "eligible_symbol_counts": sorted(eligible_counts),
        "eligible_symbols": common_eligible_symbols,
        "eligible_symbol_count": len(common_eligible_symbols),
        "minimum_history": sorted(minimum_history),
        "minimum_volume": sorted(minimum_volume),
        "oi_coverage_requirements": sorted(oi_requirements),
        "funding_coverage_requirements": sorted(funding_requirements),
        "excluded_symbols_and_reasons": dict(sorted(excluded_union.items())),
        "fold_split_contracts": splits,
        "validation": {
            "passed": not failures,
            "failures": failures,
            "fold_split_count": int(len(splits)),
            "eligible_symbol_list_constant": len(eligible_sets) == 1,
        },
    }


def _market_state_feature_contract(
    *,
    args: argparse.Namespace,
    folds: list[dict[str, Any]],
    disabled_heads: set[str],
    controller_enabled_manifest: dict[str, Any],
    fold_reports: list[dict[str, Any]],
    state_head_registry: pd.DataFrame,
) -> dict[str, Any]:
    first_state_report = dict(fold_reports[0].get("state_report") or {}) if fold_reports else {}
    first_feature_store = dict(first_state_report.get("feature_store") or {})
    first_train_fs = dict(first_feature_store.get("train") or {})
    first_valid_fs = dict(first_feature_store.get("valid") or {})
    first_market_source = dict(first_state_report.get("market_state_source") or {})
    first_train_source = dict(first_market_source.get("train") or {})
    first_valid_source = dict(first_market_source.get("valid") or {})
    axis_sources = dict(first_state_report.get("axis_sources") or {})
    forecast_report = dict(first_state_report.get("forecast_report") or {})
    latent_report = dict(first_state_report.get("latent_report") or {})
    feature_store_columns = list(first_train_fs.get("columns") or [])
    validation_failures: list[str] = []
    maturity_failures: list[str] = []
    immature_rows_dropped = 0
    for fold_report in fold_reports:
        fold = int(fold_report.get("fold", 0) or 0)
        state_report = dict(fold_report.get("state_report") or {})
        market_source = dict(state_report.get("market_state_source") or {})
        for split in ("train", "valid"):
            split_report = dict(market_source.get(split) or {})
            validation = dict(split_report.get("validation") or {})
            if not bool(validation.get("timestamp_unique", True)):
                validation_failures.append(f"fold_{fold}_{split}_timestamp_not_unique")
            if not bool(validation.get("market_wide_one_row_per_timestamp", True)):
                validation_failures.append(f"fold_{fold}_{split}_not_one_market_row_per_timestamp")
            if int(validation.get("forbidden_column_count", 0) or 0) != 0:
                validation_failures.append(f"fold_{fold}_{split}_forbidden_state_columns")
            if not bool(split_report.get("production_safe", True)):
                validation_failures.append(f"fold_{fold}_{split}_state_source_not_production_safe")
        maturity_contract = fold_report.get("split_maturity_contract")
        if not isinstance(maturity_contract, dict):
            maturity_failures.append(f"fold_{fold}_split_maturity_contract_missing")
        elif maturity_contract.get("uses_matured_training_outcomes_only") is not True:
            maturity_failures.append(f"fold_{fold}_does_not_use_matured_training_outcomes_only")
        else:
            valid_start = pd.Timestamp(maturity_contract.get("training_outcome_available_before"))
            for split in ("train_broad", "train_deployable"):
                split_diag = maturity_contract.get(split)
                if not isinstance(split_diag, dict):
                    maturity_failures.append(f"fold_{fold}_{split}_maturity_diag_missing")
                    continue
                immature_rows_dropped += int(split_diag.get("dropped_immature_outcome_rows") or 0)
                max_available = split_diag.get("max_outcome_available_timestamp")
                if max_available is not None and pd.Timestamp(max_available) >= valid_start:
                    maturity_failures.append(f"fold_{fold}_{split}_uses_outcome_available_at_or_after_valid_start")

    validation_failures.extend(maturity_failures)

    registry_rows = int(len(state_head_registry))
    active_state_heads = (
        state_head_registry.loc[state_head_registry["aggregate_status"].astype(str).eq("active"), "state_head"].astype(str).tolist()
        if not state_head_registry.empty and "aggregate_status" in state_head_registry
        else []
    )
    fallback_state_heads = (
        state_head_registry.loc[state_head_registry["aggregate_status"].astype(str).eq("fallback"), "state_head"].astype(str).tolist()
        if not state_head_registry.empty and "aggregate_status" in state_head_registry
        else []
    )
    shadow_state_heads = (
        state_head_registry.loc[
            state_head_registry["aggregate_status"].astype(str).str.contains("shadow", regex=False),
            "state_head",
        ].astype(str).tolist()
        if not state_head_registry.empty and "aggregate_status" in state_head_registry
        else []
    )

    source_contract_audit = _source_contract_audit(fold_reports)
    universe_contract = _market_state_universe_contract(fold_reports)
    return {
        "contract_version": "market_state_walkforward_feature_contract_v1",
        "generated_by": "run_market_state_threshold_controller_walkforward",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rank_contract": str(args.rank_contract),
        "rank_reference_run_id": str(getattr(args, "rank_reference_run_id", mstc.DEFAULT_RANK_REFERENCE_RUN_ID)),
        "data_root": str(getattr(args, "data_root", mstc.DEFAULT_DATA_ROOT)),
        "policy_variant": str(args.policy_variant),
        "disabled_heads": sorted(disabled_heads),
        "active_heads": mstc._active_heads(disabled_heads),
        **controller_enabled_manifest,
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
            "latent_gmm_active_controller_input": bool(args.include_latent_shadow_arms),
        },
        "validation": {
            "passed": not validation_failures,
            "failures": validation_failures,
            "fold_count": int(len(fold_reports)),
            "state_head_registry_rows": registry_rows,
            "training_outcome_maturity_contract_passed": not maturity_failures,
            "training_immature_outcome_rows_dropped": int(immature_rows_dropped),
            "training_outcome_maturity_failures": maturity_failures,
        },
        "source_contract_audit": source_contract_audit,
        "universe_contract": universe_contract,
        "fold_definition": {
            "n_folds_requested": int(args.n_folds),
            "folds_built": folds,
            "min_train_days": int(args.min_train_days),
            "valid_days": int(args.valid_days),
            "embargo_hours": int(args.embargo_hours),
            "min_valid_rows": int(args.min_valid_rows),
            "min_valid_timestamps": int(args.min_valid_timestamps),
        },
        "source_schema": {
            "candidate_feature_count": int(first_state_report.get("candidate_feature_count", 0) or 0),
            "feature_store_selected_column_count": int(first_feature_store.get("selected_column_count", 0) or 0),
            "feature_store_columns": feature_store_columns,
            "observed_axis_columns": list(axis_sources.keys()),
            "forecast_horizons_steps": list(forecast_report.get("horizon_steps") or []),
            "forecast_target_count": int(len(dict(forecast_report.get("targets") or {}))),
            "latent_report_mode": latent_report.get("mode"),
        },
        "feature_store": {
            "train_feature_dir": first_train_fs.get("feature_dir"),
            "valid_feature_dir": first_valid_fs.get("feature_dir"),
            "selected_column_count": int(first_feature_store.get("selected_column_count", 0) or 0),
            "train_universe_contract": first_train_fs.get("universe_contract"),
            "valid_universe_contract": first_valid_fs.get("universe_contract"),
            "universe_contract_artifact": "market_state_universe_contract.json",
            "train_timestamp_coverage": first_train_fs.get("timestamp_coverage"),
            "valid_timestamp_coverage": first_valid_fs.get("timestamp_coverage"),
            "train_symbols_read": first_train_fs.get("symbols_read"),
            "valid_symbols_read": first_valid_fs.get("symbols_read"),
        },
        "market_state_source": {
            "train": first_train_source,
            "valid": first_valid_source,
        },
        "axis_sources": axis_sources,
        "state_head_summary": {
            "active_state_heads": active_state_heads,
            "fallback_state_heads": fallback_state_heads,
            "shadow_state_heads": shadow_state_heads,
        },
    }


def _market_state_target_definitions(fold_reports: list[dict[str, Any]]) -> dict[str, Any]:
    target_rows: dict[str, list[dict[str, Any]]] = {}
    target_sources: dict[str, dict[str, Any]] = {}
    for fold_report in fold_reports:
        fold = int(fold_report.get("fold", 0) or 0)
        forecast_report = dict((fold_report.get("state_report") or {}).get("forecast_report") or {})
        for source_key, source_report in dict(forecast_report.get("target_source_reports") or {}).items():
            target_sources[str(source_key)] = dict(source_report or {})
        for target_name, target_report_any in dict(forecast_report.get("targets") or {}).items():
            target_report = dict(target_report_any or {})
            target_rows.setdefault(str(target_name), []).append(
                {
                    "fold": fold,
                    "mode": target_report.get("mode"),
                    "horizon_steps": target_report.get("horizon_steps"),
                    "raw_target": target_report.get("raw_target"),
                    "fallback_axis": target_report.get("fallback_axis"),
                    "rows": target_report.get("rows"),
                    "target_std": target_report.get("target_std"),
                    "soft_target_mean": target_report.get("soft_target_mean"),
                    "hard_tail_rate_p90": target_report.get("hard_tail_rate_p90"),
                    "train_prediction_mode": target_report.get("train_prediction_mode"),
                    "oof_rows": target_report.get("oof_rows"),
                    "oof_coverage": target_report.get("oof_coverage"),
                }
            )

    targets: dict[str, Any] = {}
    for target_name, rows in sorted(target_rows.items()):
        frame = pd.DataFrame(rows)
        numeric = {}
        for col in ("rows", "target_std", "soft_target_mean", "hard_tail_rate_p90", "oof_rows", "oof_coverage"):
            vals = pd.to_numeric(frame.get(col), errors="coerce") if col in frame else pd.Series(dtype=float)
            numeric[f"mean_{col}"] = float(vals.mean()) if vals.notna().any() else None
        modes = sorted({str(x) for x in frame.get("mode", pd.Series(dtype=object)).dropna().tolist()})
        raw_targets = sorted({str(x) for x in frame.get("raw_target", pd.Series(dtype=object)).dropna().tolist()})
        horizons = sorted({int(x) for x in pd.to_numeric(frame.get("horizon_steps"), errors="coerce").dropna().tolist()}) if "horizon_steps" in frame else []
        targets[target_name] = {
            "modes": modes,
            "horizon_steps": horizons,
            "raw_targets": raw_targets,
            "fold_count": int(frame["fold"].nunique()) if "fold" in frame else int(len(frame)),
            **numeric,
            "folds": rows,
        }

    return {
        "contract_version": "market_state_target_definitions_v1",
        "generated_by": "run_market_state_threshold_controller_walkforward",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_type": "training_cdf_normalized_future_market_geometry_soft_severity",
        "hard_tail_definition": "soft_target >= 0.90",
        "signed_targets": "not_active_in_current_lgbm_pack",
        "forecast_targets": targets,
        "target_source_reports": target_sources,
    }


def _market_state_target_cdfs(state_reference_bundles: dict[str, Any]) -> dict[str, Any]:
    folds: dict[str, Any] = {}
    target_count = 0
    missing_reference_count = 0
    for fold_key, bundle_any in dict(state_reference_bundles or {}).items():
        bundle = dict(bundle_any or {})
        forecast_artifact = dict(bundle.get("forecast_artifact") or {})
        fold_targets: dict[str, Any] = {}
        for target_name, spec_any in dict(forecast_artifact.get("targets") or {}).items():
            spec = dict(spec_any or {})
            cdf = spec.get("target_cdf_reference")
            if isinstance(cdf, dict):
                fold_targets[str(target_name)] = {
                    "mode": spec.get("mode"),
                    "horizon_steps": spec.get("horizon_steps"),
                    "raw_target": spec.get("raw_target"),
                    "fallback_axis": spec.get("fallback_axis"),
                    "target_cdf_reference": cdf,
                }
                target_count += 1
            else:
                missing_reference_count += 1
        folds[str(fold_key)] = {
            "forecast_model_kind": forecast_artifact.get("forecast_model_kind"),
            "model_backend": forecast_artifact.get("model_backend"),
            "horizon_steps": forecast_artifact.get("horizon_steps"),
            "target_count": int(len(fold_targets)),
            "targets": fold_targets,
        }
    return {
        "artifact_version": "market_state_target_cdfs_v1",
        "generated_by": "run_market_state_threshold_controller_walkforward",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "normalization": "training_fold_empirical_cdf_raw_future_market_geometry_targets",
        "target_count": int(target_count),
        "missing_reference_count": int(missing_reference_count),
        "folds": folds,
    }


def _market_state_oof_predictions(state_panel: pd.DataFrame) -> pd.DataFrame:
    if state_panel.empty or "split" not in state_panel.columns:
        return pd.DataFrame()
    out = state_panel.loc[state_panel["split"].astype(str).eq("valid")].copy()
    if out.empty:
        return out.reset_index(drop=True)
    out["prediction_contract"] = "outer_fold_validation_state_scores"
    return out.reset_index(drop=True)


def _market_state_feature_coverage(fold_reports: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold_report in fold_reports:
        fold = int(fold_report.get("fold", 0) or 0)
        state_report = dict(fold_report.get("state_report") or {})
        feature_store = dict(state_report.get("feature_store") or {})
        market_source = dict(state_report.get("market_state_source") or {})
        for split in ("train", "valid"):
            fs = dict(feature_store.get(split) or {})
            source = dict(market_source.get(split) or {})
            validation = dict(source.get("validation") or {})
            universe = dict(fs.get("universe_contract") or {})
            rows.append(
                {
                    "fold": fold,
                    "split": split,
                    "feature_store_enabled": bool(fs.get("enabled", False)),
                    "feature_store_selected_column_count": int(feature_store.get("selected_column_count", 0) or 0),
                    "feature_store_timestamp_coverage": fs.get("timestamp_coverage"),
                    "feature_store_symbols_read": fs.get("symbols_read"),
                    "feature_store_available_symbol_count": universe.get("available_symbol_count"),
                    "feature_store_eligible_symbol_count": universe.get("eligible_symbol_count"),
                    "feature_store_aggregation_contract": fs.get("aggregation_contract"),
                    "feature_store_tail_reference_source": fs.get("tail_reference_source"),
                    "feature_store_tail_reference_role": fs.get("tail_reference_role"),
                    "market_state_source": source.get("source"),
                    "market_state_feature_count": source.get("feature_count"),
                    "market_state_production_safe": source.get("production_safe"),
                    "validation_row_count": validation.get("row_count"),
                    "validation_feature_count": validation.get("feature_count"),
                    "validation_forbidden_column_count": validation.get("forbidden_column_count"),
                    "validation_timestamp_unique": validation.get("timestamp_unique"),
                    "validation_market_wide_one_row_per_timestamp": validation.get("market_wide_one_row_per_timestamp"),
                    "candidate_aggregate_feature_count": source.get("candidate_aggregate_feature_count"),
                    "feature_store_aggregate_feature_count": source.get("feature_store_aggregate_feature_count"),
                }
            )
    return pd.DataFrame(rows)


def _controller_arm_complexity(arm: str) -> int:
    base = str(arm).replace("__post_selection_overlay", "")
    order = {
        "S1_observed_axes_shared_response": 1,
        "S2_observed_forecast_shared_response": 2,
        PRUNED_STATE_ARM: 2,
        "S3_observed_forecast_latent_shared_response": 3,
        "S4_S3_plus_per_strategy_residual": 4,
    }
    return int(order.get(base, 99))


def _select_controller_candidate(
    aggregate: pd.DataFrame,
    suppression_aggregate: pd.DataFrame,
    controller_diagnostics: pd.DataFrame,
    action_utility_aggregate: pd.DataFrame | None = None,
    baseline_accepted_suppression_aggregate: pd.DataFrame | None = None,
    *,
    min_positive_delta_share: float = 0.50,
    min_median_delta_net_pnl: float = 0.0,
    min_q25_delta_net_pnl: float = 0.0,
    min_defensive_success: float = 0.0,
    min_positive_suppression_share: float = 0.50,
    max_mean_state_ood_share: float = 0.10,
    min_median_delta_max_drawdown: float = 0.0,
    min_median_delta_worst_24h: float = 0.0,
    max_median_delta_full_sl_rate: float = 0.0,
    min_median_trade_retention_share: float = 0.80,
    median_delta_tie_abs_tol: float = 1.0,
    median_delta_tie_rel_tol: float = 0.05,
    require_post_selection_confirmation: bool = True,
    select_no_backfill_overlay_only: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Select a deployable controller arm from walk-forward diagnostics.

    Selection is deliberately conservative: post-selection overlays and the
    baseline are audit arms, not deployable controller specifications.  The
    gates require recurrent fold lift, positive accepted-frontier suppression
    utility, fixed-universe/no-backfill confirmation, and live-state health.
    If multiple arms pass, choose the simplest arm within a small median-delta
    tolerance of the best arm.
    """

    if aggregate.empty:
        payload = {"selected_arm": None, "reason": "missing_aggregate_metrics"}
        return pd.DataFrame(), payload

    work = aggregate.copy()
    work["arm"] = work["arm"].astype(str)
    work["base_arm"] = work["arm"].str.replace("__post_selection_overlay", "", regex=False)
    work["is_post_selection_overlay"] = work["arm"].str.endswith("__post_selection_overlay")
    work["is_baseline"] = work["arm"].eq("S0_baseline_static_thresholds")
    work["complexity"] = work["arm"].map(_controller_arm_complexity).astype(int)

    if not suppression_aggregate.empty:
        sup = suppression_aggregate.copy()
        sup = sup.loc[
            sup.get("scope", pd.Series(index=sup.index, dtype=object)).astype(str).eq("all")
            & sup.get("scope_value", pd.Series(index=sup.index, dtype=object)).astype(str).eq("all")
        ].copy()
        sup_cols = [
            "arm",
            "suppressed_candidates",
            "realized_defensive_success",
            "positive_suppression_fold_share",
            "suppressed_loss_avoided",
            "suppressed_winner_pnl_sacrificed",
        ]
        sup = sup[[c for c in sup_cols if c in sup.columns]]
        sup = sup.rename(
            columns={
                c: f"candidate_{c}"
                for c in sup.columns
                if c != "arm"
            }
        )
        work = work.merge(sup, on="arm", how="left")

    accepted_sup_provided = baseline_accepted_suppression_aggregate is not None
    accepted_sup_source = (
        baseline_accepted_suppression_aggregate
        if accepted_sup_provided
        else suppression_aggregate
    )
    accepted_sup_source_name = (
        "baseline_accepted_suppression"
        if accepted_sup_provided
        else "candidate_suppression_fallback"
    )
    if accepted_sup_source is not None and not accepted_sup_source.empty:
        accepted_sup = accepted_sup_source.copy()
        accepted_sup = accepted_sup.loc[
            accepted_sup.get("scope", pd.Series(index=accepted_sup.index, dtype=object)).astype(str).eq("all")
            & accepted_sup.get("scope_value", pd.Series(index=accepted_sup.index, dtype=object)).astype(str).eq("all")
        ].copy()
        accepted_sup_cols = [
            "arm",
            "suppressed_candidates",
            "realized_defensive_success",
            "positive_suppression_fold_share",
            "suppressed_loss_avoided",
            "suppressed_winner_pnl_sacrificed",
        ]
        accepted_sup = accepted_sup[[c for c in accepted_sup_cols if c in accepted_sup.columns]]
        work = work.merge(accepted_sup, on="arm", how="left")
    else:
        for col in (
            "suppressed_candidates",
            "realized_defensive_success",
            "positive_suppression_fold_share",
            "suppressed_loss_avoided",
            "suppressed_winner_pnl_sacrificed",
        ):
            work[col] = np.nan
    work["suppression_gate_source"] = accepted_sup_source_name

    if action_utility_aggregate is not None and not action_utility_aggregate.empty:
        au = action_utility_aggregate.copy()
        au = au.loc[
            au.get("scope", pd.Series(index=au.index, dtype=object)).astype(str).eq("all")
            & au.get("scope_value", pd.Series(index=au.index, dtype=object)).astype(str).eq("all")
        ].copy()
        au_cols = [
            "arm",
            "folds_with_action",
            "action_entrants",
            "action_removed",
            "action_removed_loss_avoided",
            "action_removed_winner_pnl_sacrificed",
            "action_defensive_success",
            "positive_action_fold_share",
            "mean_action_net_pnl_delta",
        ]
        au = au[[c for c in au_cols if c in au.columns]]
        work = work.merge(au, on="arm", how="left")
    else:
        for col in (
            "folds_with_action",
            "action_entrants",
            "action_removed",
            "action_removed_loss_avoided",
            "action_removed_winner_pnl_sacrificed",
            "action_defensive_success",
            "positive_action_fold_share",
            "mean_action_net_pnl_delta",
        ):
            work[col] = np.nan

    if not controller_diagnostics.empty:
        diag = controller_diagnostics.copy()
        diag = diag.loc[~diag["arm"].astype(str).eq("S0_baseline_static_thresholds")].copy()
        diag_agg = (
            diag.groupby("arm", sort=False)
            .agg(
                mean_prediction_coverage=("mean_prediction_coverage", "mean"),
                mean_state_ood_share=("mean_state_ood_share", "mean"),
                max_state_ood_share=("mean_state_ood_share", "max"),
                mean_force_base_share=("force_base_share", "mean"),
            )
            .reset_index()
        )
        work = work.merge(diag_agg, on="arm", how="left")
    else:
        for col in (
            "mean_prediction_coverage",
            "mean_state_ood_share",
            "max_state_ood_share",
            "mean_force_base_share",
        ):
            work[col] = np.nan

    numeric_defaults = {
        "median_delta_net_pnl": -np.inf,
        "q25_delta_net_pnl": -np.inf,
        "positive_delta_share": 0.0,
        "median_delta_max_drawdown": -np.inf,
        "median_delta_worst_24h": -np.inf,
        "median_delta_full_sl_rate": np.inf,
        "median_trade_retention_share": 0.0,
        "realized_defensive_success": 0.0,
        "positive_suppression_fold_share": 0.0,
        "mean_state_ood_share": 0.0,
        "max_state_ood_share": 0.0,
        "mean_prediction_coverage": 1.0,
        "post_selection_median_delta_net_pnl": -np.inf,
        "post_selection_q25_delta_net_pnl": -np.inf,
        "post_selection_positive_delta_share": 0.0,
        "post_selection_median_delta_max_drawdown": -np.inf,
        "post_selection_median_delta_worst_24h": -np.inf,
        "post_selection_median_delta_full_sl_rate": np.inf,
        "post_selection_median_trade_retention_share": 0.0,
        "post_selection_realized_defensive_success": 0.0,
        "post_selection_positive_suppression_fold_share": 0.0,
        "post_selection_suppressed_loss_avoided": 0.0,
        "post_selection_suppressed_winner_pnl_sacrificed": 0.0,
        "action_entrants": 0.0,
        "action_removed": 0.0,
        "action_removed_loss_avoided": 0.0,
        "action_removed_winner_pnl_sacrificed": 0.0,
        "action_defensive_success": 0.0,
        "positive_action_fold_share": 0.0,
        "mean_action_net_pnl_delta": 0.0,
        "candidate_suppressed_candidates": 0.0,
        "candidate_realized_defensive_success": 0.0,
        "candidate_positive_suppression_fold_share": 0.0,
        "candidate_suppressed_loss_avoided": 0.0,
        "candidate_suppressed_winner_pnl_sacrificed": 0.0,
    }
    for col, default in numeric_defaults.items():
        if col not in work.columns:
            work[col] = default
        work[col] = pd.to_numeric(work[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)

    overlay_mask = work["is_post_selection_overlay"].astype(bool)
    use_action_metrics_for_overlay_gate = accepted_sup_source_name != "baseline_accepted_suppression"
    if overlay_mask.any() and use_action_metrics_for_overlay_gate:
        work.loc[overlay_mask, "suppressed_candidates"] = work.loc[overlay_mask, "action_removed"]
        work.loc[overlay_mask, "realized_defensive_success"] = work.loc[
            overlay_mask, "action_defensive_success"
        ]
        work.loc[overlay_mask, "positive_suppression_fold_share"] = work.loc[
            overlay_mask, "positive_action_fold_share"
        ]
        work.loc[overlay_mask, "suppressed_loss_avoided"] = work.loc[
            overlay_mask, "action_removed_loss_avoided"
        ]
        work.loc[overlay_mask, "suppressed_winner_pnl_sacrificed"] = work.loc[
            overlay_mask, "action_removed_winner_pnl_sacrificed"
        ]

    overlay_cols = [
        "base_arm",
        "median_delta_net_pnl",
        "mean_delta_net_pnl",
        "q25_delta_net_pnl",
        "positive_delta_share",
        "median_delta_max_drawdown",
        "median_delta_worst_24h",
        "median_delta_full_sl_rate",
        "median_trade_retention_share",
        "realized_defensive_success",
        "positive_suppression_fold_share",
        "suppressed_loss_avoided",
        "suppressed_winner_pnl_sacrificed",
    ]
    overlays = work.loc[overlay_mask, [c for c in overlay_cols if c in work.columns]].copy()
    if not overlays.empty:
        overlays = overlays.rename(
            columns={
                "median_delta_net_pnl": "post_selection_median_delta_net_pnl",
                "mean_delta_net_pnl": "post_selection_mean_delta_net_pnl",
                "q25_delta_net_pnl": "post_selection_q25_delta_net_pnl",
                "positive_delta_share": "post_selection_positive_delta_share",
                "median_delta_max_drawdown": "post_selection_median_delta_max_drawdown",
                "median_delta_worst_24h": "post_selection_median_delta_worst_24h",
                "median_delta_full_sl_rate": "post_selection_median_delta_full_sl_rate",
                "median_trade_retention_share": "post_selection_median_trade_retention_share",
                "realized_defensive_success": "post_selection_realized_defensive_success",
                "positive_suppression_fold_share": "post_selection_positive_suppression_fold_share",
                "suppressed_loss_avoided": "post_selection_suppressed_loss_avoided",
                "suppressed_winner_pnl_sacrificed": "post_selection_suppressed_winner_pnl_sacrificed",
            }
        )
        overlays = overlays.drop_duplicates("base_arm", keep="first")
        overlay_update_cols = [c for c in overlays.columns if c != "base_arm"]
        work = work.drop(columns=[c for c in overlay_update_cols if c in work.columns])
        work = work.merge(overlays, on="base_arm", how="left")
        for col, default in numeric_defaults.items():
            if col.startswith("post_selection_") and col in work.columns:
                work[col] = (
                    pd.to_numeric(work[col], errors="coerce")
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(default)
                )

    fail_reasons: list[list[str]] = []
    passed: list[bool] = []
    for _, row in work.iterrows():
        reasons: list[str] = []
        if bool(row["is_baseline"]):
            reasons.append("baseline_audit_arm")
        if bool(row["is_post_selection_overlay"]) and not bool(select_no_backfill_overlay_only):
            reasons.append("post_selection_overlay_audit_arm")
        if bool(select_no_backfill_overlay_only) and not bool(row["is_post_selection_overlay"]):
            reasons.append("full_replay_can_promote_replacements")
        if bool(row["is_post_selection_overlay"]) and float(row["action_entrants"]) > 0.0:
            reasons.append("no_backfill_contract_violated")
        if int(row["complexity"]) >= 99:
            reasons.append("unknown_controller_arm")
        if float(row["median_delta_net_pnl"]) <= float(min_median_delta_net_pnl):
            reasons.append("median_delta_not_positive")
        if float(row["q25_delta_net_pnl"]) < float(min_q25_delta_net_pnl):
            reasons.append("q25_delta_below_gate")
        if float(row["positive_delta_share"]) < float(min_positive_delta_share):
            reasons.append("insufficient_positive_folds")
        if float(row["median_delta_max_drawdown"]) < float(min_median_delta_max_drawdown):
            reasons.append("max_drawdown_worsened")
        if float(row["median_delta_worst_24h"]) < float(min_median_delta_worst_24h):
            reasons.append("worst_24h_worsened")
        if float(row["median_delta_full_sl_rate"]) > float(max_median_delta_full_sl_rate):
            reasons.append("full_sl_rate_worsened")
        if float(row["median_trade_retention_share"]) < float(min_median_trade_retention_share):
            reasons.append("insufficient_trade_retention")
        if float(row["realized_defensive_success"]) <= float(min_defensive_success):
            reasons.append("defensive_success_not_positive")
        if float(row["positive_suppression_fold_share"]) < float(min_positive_suppression_share):
            reasons.append("suppression_not_recurrent")
        if float(row["mean_state_ood_share"]) > float(max_mean_state_ood_share):
            reasons.append("mean_state_ood_share_too_high")
        if (
            bool(require_post_selection_confirmation)
            and not bool(row["is_baseline"])
            and not bool(row["is_post_selection_overlay"])
        ):
            if not np.isfinite(float(row["post_selection_median_delta_net_pnl"])):
                reasons.append("post_selection_confirmation_missing")
            if float(row["post_selection_median_delta_net_pnl"]) <= float(min_median_delta_net_pnl):
                reasons.append("post_selection_median_delta_not_positive")
            if float(row["post_selection_q25_delta_net_pnl"]) < float(min_q25_delta_net_pnl):
                reasons.append("post_selection_q25_delta_below_gate")
            if float(row["post_selection_positive_delta_share"]) < float(min_positive_delta_share):
                reasons.append("post_selection_insufficient_positive_folds")
            if float(row["post_selection_median_delta_max_drawdown"]) < float(min_median_delta_max_drawdown):
                reasons.append("post_selection_max_drawdown_worsened")
            if float(row["post_selection_median_delta_worst_24h"]) < float(min_median_delta_worst_24h):
                reasons.append("post_selection_worst_24h_worsened")
            if float(row["post_selection_median_delta_full_sl_rate"]) > float(max_median_delta_full_sl_rate):
                reasons.append("post_selection_full_sl_rate_worsened")
            if float(row["post_selection_median_trade_retention_share"]) < float(min_median_trade_retention_share):
                reasons.append("post_selection_insufficient_trade_retention")
            if float(row["post_selection_realized_defensive_success"]) <= float(min_defensive_success):
                reasons.append("post_selection_defensive_success_not_positive")
            if float(row["post_selection_positive_suppression_fold_share"]) < float(min_positive_suppression_share):
                reasons.append("post_selection_suppression_not_recurrent")
        passed.append(not reasons)
        fail_reasons.append(reasons)

    work["passed_selection_gates"] = passed
    work["selection_fail_reasons"] = [";".join(r) if r else "" for r in fail_reasons]
    work["selection_score"] = (
        work["median_delta_net_pnl"]
        + 0.25 * work["q25_delta_net_pnl"]
        + 0.10 * work["mean_delta_net_pnl"]
        + 10.0 * work["realized_defensive_success"]
    )

    passed_df = work.loc[work["passed_selection_gates"]].copy()
    selection_policy = {
        "min_positive_delta_share": float(min_positive_delta_share),
        "min_median_delta_net_pnl": float(min_median_delta_net_pnl),
        "min_q25_delta_net_pnl": float(min_q25_delta_net_pnl),
        "min_defensive_success": float(min_defensive_success),
        "min_positive_suppression_share": float(min_positive_suppression_share),
        "max_mean_state_ood_share": float(max_mean_state_ood_share),
        "min_median_delta_max_drawdown": float(min_median_delta_max_drawdown),
        "min_median_delta_worst_24h": float(min_median_delta_worst_24h),
        "max_median_delta_full_sl_rate": float(max_median_delta_full_sl_rate),
        "min_median_trade_retention_share": float(min_median_trade_retention_share),
        "median_delta_tie_abs_tol": float(median_delta_tie_abs_tol),
        "median_delta_tie_rel_tol": float(median_delta_tie_rel_tol),
        "require_post_selection_confirmation": bool(require_post_selection_confirmation),
        "select_no_backfill_overlay_only": bool(select_no_backfill_overlay_only),
        "suppression_gate_source": accepted_sup_source_name,
        "overlay_gate_uses_action_metrics": bool(use_action_metrics_for_overlay_gate),
    }
    if passed_df.empty:
        payload = {
            "selected_arm": None,
            "reason": "no_arm_passed_selection_gates",
            "selection_policy": selection_policy,
        }
        return work, payload

    best_median = float(passed_df["median_delta_net_pnl"].max())
    tolerance = max(float(median_delta_tie_abs_tol), abs(best_median) * float(median_delta_tie_rel_tol))
    near_best = passed_df.loc[passed_df["median_delta_net_pnl"] >= best_median - tolerance].copy()
    near_best = near_best.sort_values(
        [
            "complexity",
            "median_delta_net_pnl",
            "q25_delta_net_pnl",
            "realized_defensive_success",
            "selection_score",
        ],
        ascending=[True, False, False, False, False],
    )
    selected = near_best.iloc[0].to_dict()
    payload = {
        "selected_arm": selected.get("arm"),
        "selection_basis": (
            "simplest passing arm within median-delta tie tolerance of the best passing arm"
        ),
        "best_passing_median_delta_net_pnl": best_median,
        "median_delta_tolerance": tolerance,
        "selection_policy": selection_policy,
        "selected_metrics": _json_safe(selected),
    }
    return work, payload


def _render_report(
    summary: pd.DataFrame,
    aggregate: pd.DataFrame,
    selection: pd.DataFrame,
    state_head_registry: pd.DataFrame,
    activation_registry: pd.DataFrame,
    overlap: pd.DataFrame,
    action_utility: pd.DataFrame,
    action_edge_bucket: pd.DataFrame,
    suppression_utility: pd.DataFrame,
    suppression_aggregate: pd.DataFrame,
    baseline_accepted_suppression_utility: pd.DataFrame,
    baseline_accepted_suppression_aggregate: pd.DataFrame,
    state_bucket_perf: pd.DataFrame,
    manifest: dict[str, Any],
) -> str:
    lines = [
        "# Market-State Threshold Controller Walk-Forward",
        "",
        f"Generated: {manifest['generated_at_utc']}",
        "",
        "## Aggregate Delta",
        "",
        aggregate.to_markdown(index=False) if not aggregate.empty else "_No aggregate metrics._",
        "",
        "## Selected Controller Candidate",
        "",
        (
            f"Selected arm: `{manifest.get('selected_controller_candidate', {}).get('selected_arm')}`"
            if manifest.get("selected_controller_candidate", {}).get("selected_arm")
            else "_No controller arm passed selection gates._"
        ),
        "",
        selection.to_markdown(index=False) if not selection.empty else "_No selection metrics._",
        "",
        "## State Head Registry",
        "",
        state_head_registry.to_markdown(index=False)
        if not state_head_registry.empty
        else "_No state-head registry metrics._",
        "",
        "## Market-State Activation Registry",
        "",
        (
            activation_registry[
                [
                    "state_head",
                    "recommended_status",
                    "activation_disable_reason",
                    "redundant_with",
                    "response_mean_abs_spearman",
                    "loo_median_increment_net_pnl",
                    "loo_positive_increment_share",
                    "threshold_raise_count",
                    "leave_one_head_out_status",
                ]
            ].to_markdown(index=False)
            if not activation_registry.empty
            else "_No activation registry metrics._"
        ),
        "",
        "## Fold Summary",
        "",
        summary.to_markdown(index=False) if not summary.empty else "_No fold metrics._",
        "",
        "## Replacement Diagnostics",
        "",
        overlap.to_markdown(index=False) if not overlap.empty else "_No overlap metrics._",
        "",
        "## Threshold Action Utility",
        "",
        action_utility.loc[action_utility["scope"].eq("all")].to_markdown(index=False)
        if not action_utility.empty and "scope" in action_utility
        else "_No action utility metrics._",
        "",
        "## Predicted Edge Bucket Validation",
        "",
        action_edge_bucket.to_markdown(index=False)
        if not action_edge_bucket.empty
        else "_No action-edge validation metrics._",
        "",
        "## Candidate Suppression Utility",
        "",
        suppression_utility.loc[suppression_utility["scope"].eq("all")].to_markdown(index=False)
        if not suppression_utility.empty and "scope" in suppression_utility
        else "_No suppressed candidate metrics._",
        "",
        "## Candidate Suppression Utility Aggregate",
        "",
        suppression_aggregate.loc[suppression_aggregate["scope"].eq("all")].to_markdown(index=False)
        if not suppression_aggregate.empty and "scope" in suppression_aggregate
        else "_No aggregate suppressed candidate metrics._",
        "",
        "## Baseline-Accepted Candidate Suppression Utility",
        "",
        baseline_accepted_suppression_utility.loc[baseline_accepted_suppression_utility["scope"].eq("all")].to_markdown(index=False)
        if not baseline_accepted_suppression_utility.empty and "scope" in baseline_accepted_suppression_utility
        else "_No baseline-accepted suppressed candidate metrics._",
        "",
        "## Baseline-Accepted Candidate Suppression Utility Aggregate",
        "",
        baseline_accepted_suppression_aggregate.loc[baseline_accepted_suppression_aggregate["scope"].eq("all")].to_markdown(index=False)
        if not baseline_accepted_suppression_aggregate.empty and "scope" in baseline_accepted_suppression_aggregate
        else "_No aggregate baseline-accepted suppressed candidate metrics._",
        "",
        "## State Bucket Performance",
        "",
        state_bucket_perf.to_markdown(index=False) if not state_bucket_perf.empty else "_No state bucket metrics._",
        "",
        "## Contract",
        "",
        f"- Active heads: `{', '.join(manifest.get('active_heads', [])) or 'none'}`.",
        f"- Disabled heads: `{', '.join(manifest.get('disabled_heads', [])) or 'none'}`.",
        (
            "- Controller-enabled heads: "
            f"`{', '.join(manifest.get('controller', {}).get('controller_enabled_heads', [])) or 'none'}` "
            f"({manifest.get('controller', {}).get('controller_enabled_scope', 'unknown')})."
        ),
        "- Chronological folds split by complete timestamps.",
        "- Training entries end before validation start with an embargo.",
        "- Training labels use only outcomes whose `exit_timestamp` is known before validation starts.",
        "- Market-state transforms, response models and EV curves are fit inside each fold.",
        "- Scores, rank references and auction ordering are unchanged.",
        "- Threshold action is penalty-only: state thresholds may not fall below the existing base threshold.",
        "- Post-selection overlay arms preserve the baseline accepted decision-key universe to isolate sizing/suppression without freed-capacity backfill.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=mstc.DEFAULT_TRAIN_BROAD)
    parser.add_argument("--deployable-candidates", type=Path, default=mstc.DEFAULT_TRAIN_DEPLOYABLE)
    parser.add_argument("--policy-manifest", type=Path, default=mstc.DEFAULT_POLICY_MANIFEST)
    parser.add_argument("--policy-variant", default="refit_bar4_strategy_bar2")
    parser.add_argument("--feature-store-dir", type=Path, default=mstc.DEFAULT_TRAIN_FEATURE_STORE)
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
        help=(
            "Comma-separated heads for which the state controller may raise thresholds. "
            "Empty means all active heads."
        ),
    )
    parser.add_argument("--n-folds", type=int, default=3)
    parser.add_argument("--min-train-days", type=int, default=21)
    parser.add_argument("--valid-days", type=int, default=7)
    parser.add_argument(
        "--min-valid-rows",
        type=int,
        default=30,
        help=(
            "Minimum active-candidate rows required in a validation fold. "
            "Keep this aligned with the response-quality audit min_fold_rows "
            "so generated folds are not structurally under-supported."
        ),
    )
    parser.add_argument(
        "--min-valid-timestamps",
        type=int,
        default=4,
        help="Minimum active-candidate timestamps required in a validation fold.",
    )
    parser.add_argument("--embargo-hours", type=int, default=96)
    parser.add_argument("--max-feature-cols", type=int, default=128)
    parser.add_argument("--max-feature-store-cols", type=int, default=96)
    parser.add_argument("--feature-store-symbol-cap", type=int, default=80)
    parser.add_argument(
        "--allow-candidate-state-fallback",
        action="store_true",
        default=False,
        help=(
            "Allow candidate-ledger timestamp aggregates as a fallback source for "
            "market-state axes when feature-store aggregates are unavailable. "
            "Disabled by default to keep market state independent of the candidate population."
        ),
    )
    parser.add_argument("--forecast-horizon-steps", type=int, default=24)
    parser.add_argument("--forecast-horizons-steps", default="6,24")
    parser.add_argument(
        "--state-head-allowlist",
        default="",
        help=(
            "Comma-separated state/forecast heads to expose as an additional "
            "pruned-state-pack arm. Leaves the standard S1/S2 arms unchanged."
        ),
    )
    parser.add_argument(
        "--pruned-state-arm-name",
        default=PRUNED_STATE_ARM,
        help="Arm name for --state-head-allowlist experiments.",
    )
    parser.add_argument(
        "--forecast-model-kind",
        choices=("lightgbm", "xgboost"),
        default="lightgbm",
        help="Prospective market-state forecast backend. LightGBM is primary; XGBoost is the challenger.",
    )
    parser.add_argument("--latent-states", type=int, default=4)
    parser.add_argument(
        "--include-latent-shadow-arms",
        action="store_true",
        default=False,
        help=(
            "Include latent/GMM probability arms as shadow research benchmarks. "
            "Disabled by default because active controller selection uses observed/forecast market states."
        ),
    )
    parser.add_argument("--max-response-rows", type=int, default=5000)
    parser.add_argument("--max-response-keyword-cols", type=int, default=24)
    parser.add_argument(
        "--response-model-kind",
        choices=("additive_ebm", "hist_gradient_boosting", "xgboost"),
        default="additive_ebm",
        help=(
            "Strategy-response model family. additive_ebm uses deterministic "
            "training-fitted feature bins plus linear shape effects; "
            "hist_gradient_boosting and xgboost keep shallow tree challengers."
        ),
    )
    parser.add_argument(
        "--skip-state-leave-one-out",
        action="store_true",
        help=(
            "Skip neutralized no-refit leave-one-state-head-out replay diagnostics. "
            "By default the walk-forward run persists this evidence for the activation registry."
        ),
    )
    parser.add_argument("--response-frontier-weight-gamma", type=float, default=3.0)
    parser.add_argument("--response-frontier-weight-bandwidth", type=float, default=0.06)
    parser.add_argument(
        "--response-timestamp-balance",
        dest="response_balance_timestamps",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-response-timestamp-balance",
        dest="response_balance_timestamps",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--response-strategy-balance",
        dest="response_balance_strategies",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-response-strategy-balance",
        dest="response_balance_strategies",
        action="store_false",
        help=argparse.SUPPRESS,
    )
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
    parser.add_argument("--selection-min-positive-delta-share", type=float, default=0.50)
    parser.add_argument("--selection-min-median-delta-net-pnl", type=float, default=0.0)
    parser.add_argument("--selection-min-q25-delta-net-pnl", type=float, default=0.0)
    parser.add_argument("--selection-min-defensive-success", type=float, default=0.0)
    parser.add_argument("--selection-min-positive-suppression-share", type=float, default=0.50)
    parser.add_argument("--selection-max-mean-state-ood-share", type=float, default=0.10)
    parser.add_argument("--selection-min-median-delta-max-drawdown", type=float, default=0.0)
    parser.add_argument("--selection-min-median-delta-worst-24h", type=float, default=0.0)
    parser.add_argument("--selection-max-median-delta-full-sl-rate", type=float, default=0.0)
    parser.add_argument("--selection-min-median-trade-retention-share", type=float, default=0.80)
    parser.add_argument("--selection-median-delta-tie-abs-tol", type=float, default=1.0)
    parser.add_argument("--selection-median-delta-tie-rel-tol", type=float, default=0.05)
    parser.add_argument(
        "--require-post-selection-confirmation",
        dest="require_post_selection_confirmation",
        action="store_true",
        default=True,
        help=(
            "Require the no-backfill/post-selection overlay to pass defensive-success "
            "and fold-stability gates before a full-replay controller arm can be selected."
        ),
    )
    parser.add_argument(
        "--no-require-post-selection-confirmation",
        dest="require_post_selection_confirmation",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--select-no-backfill-overlay-only",
        dest="select_no_backfill_overlay_only",
        action="store_true",
        default=True,
        help=(
            "Select only no-backfill post-selection overlay arms. Full replay "
            "arms remain diagnostics because they can admit replacement trades."
        ),
    )
    parser.add_argument(
        "--allow-full-replay-controller-selection",
        dest="select_no_backfill_overlay_only",
        action="store_false",
        help="Research override: allow full replay threshold arms to be selected.",
    )
    parser.add_argument(
        "--enable-timeout-cap",
        dest="use_timeout_cap",
        action="store_true",
        default=False,
        help="Also require timeout risk to stay below its strategy cap. Disabled by default; timeout remains diagnostic.",
    )
    parser.add_argument(
        "--disable-timeout-cap",
        dest="use_timeout_cap",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--include-post-selection-overlay-arms",
        dest="include_post_selection_overlay_arms",
        action="store_true",
        default=True,
        help=(
            "Replay post-selection overlay arms restricted to the S0 accepted decision keys. "
            "Enabled by default for walk-forward validation."
        ),
    )
    parser.add_argument(
        "--include-guarded-arms",
        dest="include_post_selection_overlay_arms",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-post-selection-overlay-arms",
        dest="include_post_selection_overlay_arms",
        action="store_false",
    )
    parser.add_argument("--market-mode", default="perps")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    params, _ = _load_policy_params(args.policy_manifest, args.policy_variant)
    disabled_heads = mstc._parse_disabled_heads(args.disable_heads)
    args.controller_enabled_heads_parsed = mstc._parse_enabled_heads(args.controller_enabled_heads)
    active_heads = mstc._active_heads(disabled_heads)
    controller_enabled_manifest = mstc._controller_enabled_heads_manifest(
        args.controller_enabled_heads_parsed,
        disabled_heads,
    )
    candidates = mstc._disable_heads(
        mstc._apply_rank_contract(
            mstc._load_candidates(args.candidates),
            args.rank_contract,
            data_root=args.data_root,
            rank_reference_run_id=args.rank_reference_run_id,
        ),
        disabled_heads,
    )
    deployable = mstc._disable_heads(
        mstc._apply_rank_contract(
            mstc._load_candidates(args.deployable_candidates),
            args.rank_contract,
            data_root=args.data_root,
            rank_reference_run_id=args.rank_reference_run_id,
        ),
        disabled_heads,
    )
    folds = _build_time_folds(
        candidates["timestamp"],
        n_folds=int(args.n_folds),
        min_train_days=int(args.min_train_days),
        valid_days=int(args.valid_days),
        embargo_hours=int(args.embargo_hours),
        min_valid_rows=int(args.min_valid_rows),
        min_valid_timestamps=int(args.min_valid_timestamps),
    )
    if not folds:
        raise RuntimeError("No valid chronological folds could be built")

    summary_frames: list[pd.DataFrame] = []
    by_head_frames: list[pd.DataFrame] = []
    overlap_frames: list[pd.DataFrame] = []
    controller_frames: list[pd.DataFrame] = []
    action_utility_frames: list[pd.DataFrame] = []
    action_edge_validation_frames: list[pd.DataFrame] = []
    action_edge_bucket_frames: list[pd.DataFrame] = []
    suppression_utility_frames: list[pd.DataFrame] = []
    baseline_accepted_suppression_utility_frames: list[pd.DataFrame] = []
    state_bucket_frames: list[pd.DataFrame] = []
    accepted_frames: list[pd.DataFrame] = []
    schedule_frames: list[pd.DataFrame] = []
    rank_curve_frames: list[pd.DataFrame] = []
    residual_ledger_frames: list[pd.DataFrame] = []
    response_prediction_frames: list[pd.DataFrame] = []
    state_effect_frames: list[pd.DataFrame] = []
    state_loo_frames: list[pd.DataFrame] = []
    state_panel_frames: list[pd.DataFrame] = []
    response_model_bundles: dict[str, Any] = {}
    state_reference_bundles: dict[str, Any] = {}
    fold_reports: list[dict[str, Any]] = []
    for fold in folds:
        (
            summary,
            by_head,
            overlap,
            controller_diag,
            action_utility,
            action_edge_validation,
            action_edge_bucket,
            suppression_utility,
            baseline_accepted_suppression_utility,
            state_bucket_perf,
            accepted,
            schedules,
            rank_curves,
            residual_ledger,
            response_predictions,
            state_effect_matrix,
            state_loo,
            state_timestamp_panel,
            fold_response_models,
            fold_state_reference,
            fold_report,
        ) = _run_fold(
            fold,
            candidates,
            deployable,
            params,
            args,
        )
        summary_frames.append(summary)
        if not by_head.empty:
            by_head_frames.append(by_head)
        if not overlap.empty:
            overlap_frames.append(overlap)
        if not controller_diag.empty:
            controller_frames.append(controller_diag)
        if not action_utility.empty:
            action_utility_frames.append(action_utility)
        if not action_edge_validation.empty:
            action_edge_validation_frames.append(action_edge_validation)
        if not action_edge_bucket.empty:
            action_edge_bucket_frames.append(action_edge_bucket)
        if not suppression_utility.empty:
            suppression_utility_frames.append(suppression_utility)
        if not baseline_accepted_suppression_utility.empty:
            baseline_accepted_suppression_utility_frames.append(baseline_accepted_suppression_utility)
        if not state_bucket_perf.empty:
            state_bucket_frames.append(state_bucket_perf)
        if not accepted.empty:
            accepted_frames.append(accepted)
        if not schedules.empty:
            schedule_frames.append(schedules)
        if not rank_curves.empty:
            rank_curve_frames.append(rank_curves)
        if not residual_ledger.empty:
            residual_ledger_frames.append(residual_ledger)
        if not response_predictions.empty:
            response_prediction_frames.append(response_predictions)
        if not state_effect_matrix.empty:
            state_effect_frames.append(state_effect_matrix)
        if not state_loo.empty:
            state_loo_frames.append(state_loo)
        if not state_timestamp_panel.empty:
            state_panel_frames.append(state_timestamp_panel)
        response_model_bundles.update(
            {
                f"fold_{int(fold['fold'])}__{arm}": payload
                for arm, payload in dict(fold_response_models).items()
            }
        )
        state_reference_bundles[f"fold_{int(fold['fold'])}"] = fold_state_reference
        fold_reports.append(fold_report)

    summary_all = pd.concat(summary_frames, ignore_index=True)
    by_head_all = pd.concat(by_head_frames, ignore_index=True) if by_head_frames else pd.DataFrame()
    overlap_all = pd.concat(overlap_frames, ignore_index=True) if overlap_frames else pd.DataFrame()
    controller_all = pd.concat(controller_frames, ignore_index=True) if controller_frames else pd.DataFrame()
    action_utility_all = pd.concat(action_utility_frames, ignore_index=True) if action_utility_frames else pd.DataFrame()
    action_edge_validation_all = (
        pd.concat(action_edge_validation_frames, ignore_index=True)
        if action_edge_validation_frames
        else pd.DataFrame()
    )
    action_edge_bucket_all = (
        pd.concat(action_edge_bucket_frames, ignore_index=True)
        if action_edge_bucket_frames
        else pd.DataFrame()
    )
    empty_suppression_utility = mstc._threshold_candidate_suppression_utility(pd.DataFrame(), pd.DataFrame())
    if "fold" not in empty_suppression_utility.columns:
        empty_suppression_utility = empty_suppression_utility.copy()
        empty_suppression_utility["fold"] = pd.Series(dtype="int64")
    suppression_utility_all = (
        pd.concat(suppression_utility_frames, ignore_index=True)
        if suppression_utility_frames
        else empty_suppression_utility.copy()
    )
    baseline_accepted_suppression_utility_all = (
        pd.concat(baseline_accepted_suppression_utility_frames, ignore_index=True)
        if baseline_accepted_suppression_utility_frames
        else empty_suppression_utility.copy()
    )
    state_bucket_all = pd.concat(state_bucket_frames, ignore_index=True) if state_bucket_frames else pd.DataFrame()
    accepted_all = pd.concat(accepted_frames, ignore_index=True) if accepted_frames else pd.DataFrame()
    schedules_all = pd.concat(schedule_frames, ignore_index=True) if schedule_frames else pd.DataFrame()
    state_panel_all = pd.concat(state_panel_frames, ignore_index=True) if state_panel_frames else pd.DataFrame()
    rank_curves_all = pd.concat(rank_curve_frames, ignore_index=True) if rank_curve_frames else pd.DataFrame()
    residual_ledger_all = pd.concat(residual_ledger_frames, ignore_index=True) if residual_ledger_frames else pd.DataFrame()
    response_predictions_all = (
        pd.concat(response_prediction_frames, ignore_index=True)
        if response_prediction_frames
        else pd.DataFrame()
    )
    state_effect_matrix_all = pd.concat(state_effect_frames, ignore_index=True) if state_effect_frames else pd.DataFrame()
    state_loo_all = pd.concat(state_loo_frames, ignore_index=True) if state_loo_frames else pd.DataFrame()
    state_loo_aggregate = _aggregate_state_leave_one_out(state_loo_all)
    aggregate = _aggregate_delta(summary_all)
    suppression_aggregate = _aggregate_suppression_utility(suppression_utility_all)
    baseline_accepted_suppression_aggregate = _aggregate_suppression_utility(baseline_accepted_suppression_utility_all)
    action_utility_aggregate = _aggregate_action_utility(action_utility_all)
    state_head_registry = _state_head_registry(fold_reports)
    activation_registry = _market_state_activation_registry(
        state_head_registry,
        state_panel_all,
        state_effect_matrix_all,
        schedules_all,
        state_loo_aggregate,
    )
    market_state_feature_contract = _market_state_feature_contract(
        args=args,
        folds=folds,
        disabled_heads=disabled_heads,
        controller_enabled_manifest=controller_enabled_manifest,
        fold_reports=fold_reports,
        state_head_registry=state_head_registry,
    )
    market_state_universe_contract = _market_state_universe_contract(fold_reports)
    market_state_target_definitions = _market_state_target_definitions(fold_reports)
    market_state_target_cdfs = _market_state_target_cdfs(state_reference_bundles)
    market_state_feature_coverage = _market_state_feature_coverage(fold_reports)
    market_state_oof_predictions = _market_state_oof_predictions(state_panel_all)
    source_contract_audit = dict(market_state_feature_contract.get("source_contract_audit") or {})
    selection, selected_payload = _select_controller_candidate(
        aggregate,
        suppression_aggregate,
        controller_all,
        action_utility_aggregate,
        baseline_accepted_suppression_aggregate,
        min_positive_delta_share=float(args.selection_min_positive_delta_share),
        min_median_delta_net_pnl=float(args.selection_min_median_delta_net_pnl),
        min_q25_delta_net_pnl=float(args.selection_min_q25_delta_net_pnl),
        min_defensive_success=float(args.selection_min_defensive_success),
        min_positive_suppression_share=float(args.selection_min_positive_suppression_share),
        max_mean_state_ood_share=float(args.selection_max_mean_state_ood_share),
        min_median_delta_max_drawdown=float(args.selection_min_median_delta_max_drawdown),
        min_median_delta_worst_24h=float(args.selection_min_median_delta_worst_24h),
        max_median_delta_full_sl_rate=float(args.selection_max_median_delta_full_sl_rate),
        min_median_trade_retention_share=float(args.selection_min_median_trade_retention_share),
        median_delta_tie_abs_tol=float(args.selection_median_delta_tie_abs_tol),
        median_delta_tie_rel_tol=float(args.selection_median_delta_tie_rel_tol),
        require_post_selection_confirmation=bool(args.require_post_selection_confirmation),
        select_no_backfill_overlay_only=bool(args.select_no_backfill_overlay_only),
    )

    summary_all.to_csv(args.output_dir / "walkforward_summary.csv", index=False)
    aggregate.to_csv(args.output_dir / "walkforward_aggregate_delta.csv", index=False)
    selection.to_csv(args.output_dir / "walkforward_controller_candidate_selection.csv", index=False)
    state_head_registry.to_csv(args.output_dir / "walkforward_state_head_registry.csv", index=False)
    state_head_registry.to_csv(args.output_dir / "market_state_head_diagnostics.csv", index=False)
    activation_registry.to_csv(args.output_dir / "market_state_activation_registry.csv", index=False)
    market_state_feature_coverage.to_csv(args.output_dir / "market_state_feature_coverage.csv", index=False)
    state_panel_all.to_parquet(args.output_dir / "market_state_timestamp_panel.parquet", index=False)
    market_state_oof_predictions.to_parquet(args.output_dir / "market_state_oof_predictions.parquet", index=False)
    rank_curves_all.to_csv(args.output_dir / "strategy_rank_outcome_curves.csv", index=False)
    state_effect_matrix_all.to_csv(args.output_dir / "strategy_state_effect_matrix.csv", index=False)
    state_loo_all.to_csv(args.output_dir / "market_state_leave_one_head_out_replay.csv", index=False)
    state_loo_aggregate.to_csv(args.output_dir / "market_state_leave_one_head_out_aggregate.csv", index=False)
    residual_ledger_all.to_parquet(args.output_dir / "strategy_residual_target_ledger.parquet", index=False)
    response_predictions_all.to_parquet(args.output_dir / "strategy_response_oof_predictions.parquet", index=False)
    import joblib

    joblib.dump(
        {
            "generated_by": "run_market_state_threshold_controller_walkforward",
            "reference_version": "market_state_training_reference_bundle_v1",
            "fold_references": state_reference_bundles,
        },
        args.output_dir / "market_state_training_reference.joblib",
    )
    forecast_model_artifact = {
        "generated_by": "run_market_state_threshold_controller_walkforward",
        "artifact_version": "market_state_forecast_models_v1",
        "forecast_model_kind": str(args.forecast_model_kind),
        "fold_forecast_artifacts": {
            fold_key: bundle.get("forecast_artifact")
            for fold_key, bundle in state_reference_bundles.items()
        },
    }
    if str(args.forecast_model_kind) == "xgboost":
        joblib.dump(forecast_model_artifact, args.output_dir / "market_state_xgb_models.joblib")
    else:
        joblib.dump(forecast_model_artifact, args.output_dir / "market_state_lgbm_models.joblib")
    joblib.dump(market_state_target_cdfs, args.output_dir / "market_state_target_cdfs.joblib")
    joblib.dump(
        {
            "generated_by": "run_market_state_threshold_controller_walkforward",
            "artifact_version": "strategy_rank_outcome_curves_v1",
            "rank_curve_table": rank_curves_all,
        },
        args.output_dir / "strategy_rank_outcome_curves.joblib",
    )
    response_model_artifact = {
        "generated_by": "run_market_state_threshold_controller_walkforward",
        "model_type": (
            "rank_curve_plus_additive_ebm_response"
            if str(args.response_model_kind) == "additive_ebm"
            else (
                "rank_curve_plus_xgboost_response"
                if str(args.response_model_kind) == "xgboost"
                else "rank_curve_plus_hist_gradient_boosting_response"
            )
        ),
        "response_model_kind": str(args.response_model_kind),
        "fold_models": response_model_bundles,
    }
    joblib.dump(response_model_artifact, args.output_dir / "strategy_response_models.joblib")
    if str(args.response_model_kind) == "additive_ebm":
        joblib.dump(response_model_artifact, args.output_dir / "strategy_response_ebm_models.joblib")
    if str(args.response_model_kind) == "xgboost":
        joblib.dump(response_model_artifact, args.output_dir / "strategy_response_xgb_models.joblib")
    schedules_all.to_parquet(args.output_dir / "strategy_threshold_schedule.parquet", index=False)
    accepted_all.to_parquet(args.output_dir / "accepted_trades.parquet", index=False)
    action_edge_validation_all.to_csv(args.output_dir / "strategy_threshold_action_audit.csv", index=False)
    summary_all.to_csv(args.output_dir / "portfolio_replay_summary.csv", index=False)
    by_head_all.to_csv(args.output_dir / "portfolio_replay_by_head.csv", index=False)
    by_head_all.to_csv(args.output_dir / "walkforward_by_head.csv", index=False)
    overlap_all.to_csv(args.output_dir / "walkforward_overlap.csv", index=False)
    controller_all.to_csv(args.output_dir / "walkforward_controller_state_diagnostics.csv", index=False)
    action_utility_all.to_csv(args.output_dir / "walkforward_threshold_action_utility.csv", index=False)
    action_utility_aggregate.to_csv(
        args.output_dir / "walkforward_threshold_action_utility_aggregate.csv",
        index=False,
    )
    action_edge_validation_all.to_csv(args.output_dir / "walkforward_threshold_action_edge_validation.csv", index=False)
    action_edge_bucket_all.to_csv(args.output_dir / "walkforward_threshold_action_edge_bucket_performance.csv", index=False)
    suppression_utility_all.to_csv(args.output_dir / "walkforward_threshold_candidate_suppression_utility.csv", index=False)
    suppression_aggregate.to_csv(args.output_dir / "walkforward_threshold_candidate_suppression_aggregate.csv", index=False)
    baseline_accepted_suppression_utility_all.to_csv(
        args.output_dir / "walkforward_threshold_baseline_accepted_suppression_utility.csv",
        index=False,
    )
    baseline_accepted_suppression_aggregate.to_csv(
        args.output_dir / "walkforward_threshold_baseline_accepted_suppression_aggregate.csv",
        index=False,
    )
    state_bucket_all.to_csv(args.output_dir / "walkforward_state_bucket_performance.csv", index=False)
    (args.output_dir / "walkforward_selected_controller_candidate.json").write_text(
        json.dumps(_json_safe(selected_payload), indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "market_state_feature_contract.json").write_text(
        json.dumps(_json_safe(market_state_feature_contract), indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "market_state_universe_contract.json").write_text(
        json.dumps(_json_safe(market_state_universe_contract), indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "market_state_target_definitions.json").write_text(
        json.dumps(_json_safe(market_state_target_definitions), indent=2) + "\n",
        encoding="utf-8",
    )

    controller_config = {
        "config_version": "strategy_threshold_controller_config_v1",
        "generated_by": "run_market_state_threshold_controller_walkforward",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_contract": {
            "rank_contract": str(args.rank_contract),
            "policy_variant": str(args.policy_variant),
            "active_heads": active_heads,
            "disabled_heads": sorted(disabled_heads),
            "q_fail_enabled": False,
            "changes_scores_or_ranks": False,
            "changes_auction_ordering": False,
        },
        "controller": {
            "penalty_only": True,
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
            "response_weighting": {
                "timestamp_balanced": bool(args.response_balance_timestamps),
                "strategy_balanced": bool(args.response_balance_strategies),
                "frontier_gamma": float(args.response_frontier_weight_gamma),
                "frontier_bandwidth": float(args.response_frontier_weight_bandwidth),
            },
            "response_model_kind": str(args.response_model_kind),
            "forecast_model_kind": str(args.forecast_model_kind),
            "state_leave_one_out_enabled": not bool(args.skip_state_leave_one_out),
            "include_post_selection_overlay_arms": bool(args.include_post_selection_overlay_arms),
            "post_selection_overlay_contract": "baseline accepted decision keys only; no freed-capacity backfill",
            "require_post_selection_confirmation": bool(args.require_post_selection_confirmation),
            "select_no_backfill_overlay_only": bool(args.select_no_backfill_overlay_only),
            **controller_enabled_manifest,
        },
        "selection": selected_payload,
        "validation": {
            "chronological_complete_timestamp_folds": True,
            "embargo_hours": int(args.embargo_hours),
            "valid_days": int(args.valid_days),
            "min_valid_rows": int(args.min_valid_rows),
            "min_valid_timestamps": int(args.min_valid_timestamps),
            "selected_controller_is_null": selected_payload.get("selected_arm") is None,
        },
        "outputs": {
            "strategy_threshold_schedule": str(args.output_dir / "strategy_threshold_schedule.parquet"),
            "accepted_trades": str(args.output_dir / "accepted_trades.parquet"),
            "strategy_threshold_action_audit": str(args.output_dir / "strategy_threshold_action_audit.csv"),
            "market_state_leave_one_head_out_replay": str(args.output_dir / "market_state_leave_one_head_out_replay.csv"),
            "market_state_leave_one_head_out_aggregate": str(args.output_dir / "market_state_leave_one_head_out_aggregate.csv"),
            "threshold_action_utility_aggregate": str(args.output_dir / "walkforward_threshold_action_utility_aggregate.csv"),
            "portfolio_replay_summary": str(args.output_dir / "portfolio_replay_summary.csv"),
            "portfolio_replay_by_head": str(args.output_dir / "portfolio_replay_by_head.csv"),
        },
    }
    (args.output_dir / "strategy_threshold_controller_config.json").write_text(
        json.dumps(_json_safe(controller_config), indent=2) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "generated_by": "run_market_state_threshold_controller_walkforward",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidates": str(args.candidates),
        "deployable_candidates": str(args.deployable_candidates),
        "policy_manifest": str(args.policy_manifest),
        "policy_variant": str(args.policy_variant),
        "policy_params": asdict(params),
        "rank_contract": str(args.rank_contract),
        "rank_reference_run_id": str(args.rank_reference_run_id),
        "data_root": str(args.data_root),
        "disabled_heads": sorted(disabled_heads),
        "active_heads": active_heads,
        "source_contract_audit": source_contract_audit,
        "folds": fold_reports,
        "selected_controller_candidate": selected_payload,
        "controller": {
            "penalty_only": True,
            "include_post_selection_overlay_arms": bool(args.include_post_selection_overlay_arms),
            "post_selection_overlay_contract": "baseline accepted decision keys only; no freed-capacity backfill",
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
            "response_weighting": {
                "timestamp_balanced": bool(args.response_balance_timestamps),
                "strategy_balanced": bool(args.response_balance_strategies),
                "frontier_gamma": float(args.response_frontier_weight_gamma),
                "frontier_bandwidth": float(args.response_frontier_weight_bandwidth),
            },
            "response_model_kind": str(args.response_model_kind),
            "forecast_model_kind": str(args.forecast_model_kind),
            "state_leave_one_out_enabled": not bool(args.skip_state_leave_one_out),
            "embargo_hours": int(args.embargo_hours),
            "valid_days": int(args.valid_days),
            "min_valid_rows": int(args.min_valid_rows),
            "min_valid_timestamps": int(args.min_valid_timestamps),
            "allow_candidate_state_fallback": bool(args.allow_candidate_state_fallback),
            "include_latent_shadow_arms": bool(args.include_latent_shadow_arms),
            "require_post_selection_confirmation": bool(args.require_post_selection_confirmation),
            "select_no_backfill_overlay_only": bool(args.select_no_backfill_overlay_only),
            "changes_scores_or_ranks": False,
            "changes_auction_ordering": False,
            **controller_enabled_manifest,
        },
        "outputs": {
            "summary": str(args.output_dir / "walkforward_summary.csv"),
            "aggregate_delta": str(args.output_dir / "walkforward_aggregate_delta.csv"),
            "by_head": str(args.output_dir / "walkforward_by_head.csv"),
            "overlap": str(args.output_dir / "walkforward_overlap.csv"),
            "controller_state_diagnostics": str(args.output_dir / "walkforward_controller_state_diagnostics.csv"),
            "threshold_action_utility": str(args.output_dir / "walkforward_threshold_action_utility.csv"),
            "threshold_action_utility_aggregate": str(args.output_dir / "walkforward_threshold_action_utility_aggregate.csv"),
            "threshold_action_edge_validation": str(args.output_dir / "walkforward_threshold_action_edge_validation.csv"),
            "threshold_action_edge_bucket_performance": str(args.output_dir / "walkforward_threshold_action_edge_bucket_performance.csv"),
            "threshold_candidate_suppression_utility": str(args.output_dir / "walkforward_threshold_candidate_suppression_utility.csv"),
            "threshold_candidate_suppression_aggregate": str(args.output_dir / "walkforward_threshold_candidate_suppression_aggregate.csv"),
            "threshold_baseline_accepted_suppression_utility": str(args.output_dir / "walkforward_threshold_baseline_accepted_suppression_utility.csv"),
            "threshold_baseline_accepted_suppression_aggregate": str(args.output_dir / "walkforward_threshold_baseline_accepted_suppression_aggregate.csv"),
            "state_bucket_performance": str(args.output_dir / "walkforward_state_bucket_performance.csv"),
            "state_head_registry": str(args.output_dir / "walkforward_state_head_registry.csv"),
            "market_state_head_diagnostics": str(args.output_dir / "market_state_head_diagnostics.csv"),
            "market_state_activation_registry": str(args.output_dir / "market_state_activation_registry.csv"),
            "market_state_feature_contract": str(args.output_dir / "market_state_feature_contract.json"),
            "market_state_universe_contract": str(args.output_dir / "market_state_universe_contract.json"),
            "market_state_target_definitions": str(args.output_dir / "market_state_target_definitions.json"),
            "market_state_target_cdfs": str(args.output_dir / "market_state_target_cdfs.joblib"),
            "market_state_feature_coverage": str(args.output_dir / "market_state_feature_coverage.csv"),
            "market_state_training_reference": str(args.output_dir / "market_state_training_reference.joblib"),
            "market_state_lgbm_models": (
                str(args.output_dir / "market_state_lgbm_models.joblib")
                if str(args.forecast_model_kind) == "lightgbm"
                else None
            ),
            "market_state_xgb_models": (
                str(args.output_dir / "market_state_xgb_models.joblib")
                if str(args.forecast_model_kind) == "xgboost"
                else None
            ),
            "market_state_timestamp_panel": str(args.output_dir / "market_state_timestamp_panel.parquet"),
            "market_state_oof_predictions": str(args.output_dir / "market_state_oof_predictions.parquet"),
            "strategy_rank_outcome_curves": str(args.output_dir / "strategy_rank_outcome_curves.csv"),
            "strategy_rank_outcome_curves_joblib": str(args.output_dir / "strategy_rank_outcome_curves.joblib"),
            "strategy_residual_target_ledger": str(args.output_dir / "strategy_residual_target_ledger.parquet"),
            "strategy_response_oof_predictions": str(args.output_dir / "strategy_response_oof_predictions.parquet"),
            "strategy_response_models": str(args.output_dir / "strategy_response_models.joblib"),
            "strategy_response_ebm_models": (
                str(args.output_dir / "strategy_response_ebm_models.joblib")
                if str(args.response_model_kind) == "additive_ebm"
                else None
            ),
            "strategy_response_xgb_models": (
                str(args.output_dir / "strategy_response_xgb_models.joblib")
                if str(args.response_model_kind) == "xgboost"
                else None
            ),
            "strategy_state_effect_matrix": str(args.output_dir / "strategy_state_effect_matrix.csv"),
            "market_state_leave_one_head_out_replay": str(args.output_dir / "market_state_leave_one_head_out_replay.csv"),
            "market_state_leave_one_head_out_aggregate": str(args.output_dir / "market_state_leave_one_head_out_aggregate.csv"),
            "strategy_threshold_schedule": str(args.output_dir / "strategy_threshold_schedule.parquet"),
            "accepted_trades": str(args.output_dir / "accepted_trades.parquet"),
            "strategy_threshold_controller_config": str(args.output_dir / "strategy_threshold_controller_config.json"),
            "strategy_threshold_action_audit": str(args.output_dir / "strategy_threshold_action_audit.csv"),
            "portfolio_replay_summary": str(args.output_dir / "portfolio_replay_summary.csv"),
            "portfolio_replay_by_head": str(args.output_dir / "portfolio_replay_by_head.csv"),
            "artifact_hashes": str(args.output_dir / "artifact_hashes.json"),
            "controller_candidate_selection": str(args.output_dir / "walkforward_controller_candidate_selection.csv"),
            "selected_controller_candidate": str(args.output_dir / "walkforward_selected_controller_candidate.json"),
            "report": str(args.output_dir / "market_state_threshold_controller_walkforward_report.md"),
        },
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2) + "\n")
    hash_inputs = {
        k: v
        for k, v in dict(manifest.get("outputs", {})).items()
        if k not in {"manifest", "report", "artifact_hashes"} and v
    }
    (args.output_dir / "artifact_hashes.json").write_text(
        json.dumps(_json_safe(_artifact_hashes(hash_inputs)), indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "market_state_threshold_controller_walkforward_report.md").write_text(
        _render_report(
            summary_all,
            aggregate,
            selection,
            state_head_registry,
            activation_registry,
            overlap_all,
            action_utility_all,
            action_edge_bucket_all,
            suppression_utility_all,
            suppression_aggregate,
            baseline_accepted_suppression_utility_all,
            baseline_accepted_suppression_aggregate,
            state_bucket_all,
            manifest,
        ),
        encoding="utf-8",
    )
    print(json.dumps(_json_safe(manifest), indent=2)[:6000])
    print(f"\nWrote {args.output_dir}")


if __name__ == "__main__":
    main()
