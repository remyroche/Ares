#!/usr/bin/env python3
"""Materialize a guarded meta handoff for frozen replay review.

The script replays one fixed scenario/method from the execution-guard
walk-forward, writes the guard-kept candidate stream, writes the portfolio
accepted execution plan, and emits an explicit leakage/feature audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.simple_policy_optimiser import _json_safe  # noqa: E402
from scripts.validate_meta_handoff_execution_guard_walkforward import (  # noqa: E402
    DEFAULT_CANDIDATES,
    _filter_by_train_quantile,
    _fit_predict_scores,
    _fold_result,
    _parse_float_grid,
    _prepare_frame,
    _replay_with_train_curve,
    _scenario_folds,
    _select_keep_fraction,
    _summarise,
)


DEFAULT_OUT_DIR = Path(
    "data_perp/reports/gmm_cluster_policy_smoke_20260702_wide_sidebalanced/"
    "meta_handoff_guarded_execution_h9x4_execnet_v1"
)

FUTURE_OR_LABEL_TOKENS = (
    "net_return",
    "gross_return",
    "exit_timestamp",
    "exit_price",
    "simple_policy_exit_reason",
    "mtm_path",
    "holding_bars",
    "archetype_label_",
    "oracle",
    "path_adverse",
    "path_favorable",
)
DELAY_DECISION_FEATURES = {
    "entry_reanchor_bps",
    "entry_gap_bps",
    "entry_slippage_proxy_bps",
    "price_gap_bps",
    "entry_delay_actual_minutes",
    "delay_window_range_bps",
    "delay_entry_ref_gap_bps",
    "delay_close_gap_bps",
    "delay_max_adverse_bps",
    "delay_max_favorable_bps",
}
COST_ASSUMPTION_FEATURES = {
    "expected_spread_bps",
    "expected_half_spread_bps",
    "exit_spread_cost_bps",
    "expected_friction_bps",
}


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_payload(payload: Any) -> str:
    raw = json.dumps(_json_safe(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _thresholds_from_args(args: argparse.Namespace) -> dict[str, float | int]:
    return {
        "min_net_pnl": float(args.min_net_pnl),
        "min_mean_objective": float(args.min_mean_objective),
        "min_positive_fold_share": float(args.min_positive_fold_share),
        "max_full_sl_rate": float(args.max_full_sl_rate),
        "max_timeout_rate": float(args.max_timeout_rate),
        "min_worst_fold_net_pnl": float(args.min_worst_fold_net_pnl),
        "max_no_trade_folds": int(args.max_no_trade_folds),
    }


def _feature_audit_rows(
    features: Iterable[str],
    *,
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    prohibited: list[str] = []
    conditional: list[str] = []
    cost_assumption: list[str] = []
    for col in features:
        reason = "pre_entry_or_model_context"
        status = "safe"
        if any(token in col for token in FUTURE_OR_LABEL_TOKENS):
            status = "prohibited"
            reason = "future_outcome_or_label_token"
            prohibited.append(col)
        elif col in DELAY_DECISION_FEATURES:
            status = "conditional"
            reason = "safe only if guard decision is made after delayed-entry window is observed"
            conditional.append(col)
        elif col in COST_ASSUMPTION_FEATURES:
            status = "cost_assumption"
            reason = "spread/friction estimate or deterministic policy cost assumption"
            cost_assumption.append(col)
        rows.append({"feature": col, "status": status, "reason": reason})

    exit_matches_expected = None
    if {"exit_spread_cost_bps", "expected_half_spread_bps"}.issubset(frame.columns):
        diff = (
            pd.to_numeric(frame["exit_spread_cost_bps"], errors="coerce")
            - pd.to_numeric(frame["expected_half_spread_bps"], errors="coerce")
        ).abs()
        exit_matches_expected = bool(diff.fillna(0.0).le(1e-9).all())

    audit = {
        "guard_feature_count": len(list(features)),
        "prohibited_guard_features": prohibited,
        "conditional_delay_decision_features": conditional,
        "cost_assumption_features": cost_assumption,
        "all_guard_inputs_signal_time_safe": not prohibited and not conditional,
        "all_guard_inputs_decision_time_safe": not prohibited,
        "requires_guard_after_delayed_entry_window": bool(conditional),
        "exit_spread_cost_matches_expected_half_spread": exit_matches_expected,
        "realized_replay_columns_used_as_guard_features": prohibited,
    }
    return pd.DataFrame(rows), audit


def _decision_time_columns(frame: pd.DataFrame) -> list[str]:
    desired = [
        "signal_timestamp",
        "guard_decision_timestamp",
        "guard_decision_contract",
        "requires_delayed_entry_observation",
        "timestamp",
        "symbol",
        "side",
        "strategy_id",
        "archetype_handoff_row_id",
        "entry_price",
        "policy_executable_entry_price",
        "delayed_entry_ts",
        "delayed_entry_effective_ts",
        "entry_execution_source",
        "base_strategy_threshold",
        "strategy_rank_pct",
        "rank_pct",
        "normalized_rank_score",
        "calibrated_score",
        "oof_regime_centroid_similarity_train",
        "archetype_meta_bad_risk",
        "archetype_meta_timeout_risk",
        "archetype_joint_bad_risk",
        "archetype_joint_timeout_risk",
        "barrier_pct",
        "policy_effective_barrier_pct",
        "policy_sl_mult",
        "policy_sl_return",
        "policy_target_holding_hours",
        "policy_trailing_activation_return",
        "policy_uncapped_trailing_activation_return",
        "expected_spread_bps",
        "expected_half_spread_bps",
        "spread_cost_bps",
        "exit_spread_cost_bps",
        "expected_friction_bps",
        "entry_reanchor_bps",
        "entry_gap_bps",
        "entry_slippage_proxy_bps",
        "price_gap_bps",
        "liquidity_capacity_weight",
        "entry_delay_minutes",
        "entry_delay_target_minutes",
        "entry_delay_actual_minutes",
        "delay_window_range_bps",
        "delay_entry_ref_gap_bps",
        "delay_close_gap_bps",
        "delay_max_adverse_bps",
        "delay_max_favorable_bps",
        "market_mode",
        "scenario",
        "path_len",
        "horizon_hours",
        "barrier_multiplier",
        "delayed_entry_enabled",
    ]
    return [col for col in desired if col in frame.columns]


def _attach_candidates(decisions: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    if decisions.empty:
        return pd.DataFrame()
    out = decisions.copy()
    if "candidate_index" not in out.columns:
        return out
    out["candidate_index"] = pd.to_numeric(out["candidate_index"], errors="coerce").astype("Int64")
    payload_cols = [
        "timestamp",
        "symbol",
        "side",
        "strategy_id",
        "scenario",
        "archetype_handoff_row_id",
        "rank_pct",
        "strategy_rank_pct",
        "normalized_rank_score",
        "calibrated_score",
        "net_return",
        "gross_return",
        "simple_policy_exit_reason",
        "holding_bars",
    ]
    payload_cols = [col for col in payload_cols if col in candidates.columns]
    payload = candidates[payload_cols].reset_index(names="candidate_index")
    return out.merge(payload, on="candidate_index", how="left", suffixes=("", "_candidate"))


def _add_guard_metadata(
    frame: pd.DataFrame,
    *,
    fold_id: int,
    validation_week: pd.Timestamp,
    train: pd.DataFrame,
    scenario: str,
    method: str,
    model_name: str,
    keep_frac: float,
    threshold: float,
    feature_set_hash: str,
    model_spec_hash: str,
) -> pd.DataFrame:
    out = frame.copy()
    out["base_selector"] = out.get("strategy_id", "")
    out["base_score_oof"] = pd.to_numeric(out.get("rank_pct", np.nan), errors="coerce")
    out["meta_score_oof"] = pd.to_numeric(out.get("calibrated_score", np.nan), errors="coerce")
    out["scenario_id"] = scenario
    out["horizon_bars"] = pd.to_numeric(out.get("path_len", np.nan), errors="coerce")
    out["barrier_mult"] = pd.to_numeric(out.get("barrier_multiplier", np.nan), errors="coerce")
    out["stop_mult"] = pd.to_numeric(out.get("policy_sl_mult", np.nan), errors="coerce")
    out["accepted"] = True
    out["guard_accepted"] = True
    out["signal_timestamp"] = pd.to_datetime(out.get("timestamp"), utc=True, errors="coerce")
    if "delayed_entry_effective_ts" in out.columns:
        decision_ts = pd.to_datetime(
            out["delayed_entry_effective_ts"],
            utc=True,
            errors="coerce",
        )
    else:
        decision_ts = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns, UTC]")
    has_delayed_observation = decision_ts.notna()
    out["requires_delayed_entry_observation"] = has_delayed_observation
    out["guard_decision_timestamp"] = decision_ts.fillna(out["signal_timestamp"])
    out["guard_decision_contract"] = np.where(
        has_delayed_observation.to_numpy(dtype=bool),
        "post_delayed_entry_window",
        "signal_time_no_delay_window",
    )
    out["decision_fold"] = int(fold_id)
    out["validation_week"] = validation_week.date().isoformat()
    out["train_window_end"] = validation_week.isoformat()
    out["train_rows"] = int(len(train))
    out["exec_guard_method"] = method
    out["exec_guard_model_name"] = model_name
    out["exec_guard_keep_frac"] = float(keep_frac)
    out["exec_guard_threshold"] = float(threshold)
    out["exec_guard_feature_set_hash"] = feature_set_hash
    out["exec_guard_model_hash"] = model_spec_hash
    out["threshold_source"] = "prior_train_keep_fraction_selection"
    return out


def _fmt_table(frame: pd.DataFrame, cols: list[str], max_rows: int = 30) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame[[col for col in cols if col in frame.columns]].head(max_rows).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:,.4f}")
    return view.to_markdown(index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--scenario", default="h9_delay_1_barrier_x4")
    parser.add_argument("--method", default="exec_net_regressor")
    parser.add_argument(
        "--feature-mode",
        choices=["base", "execution_known"],
        default="execution_known",
    )
    parser.add_argument("--market-mode", default="perps", choices=["spot", "perps"])
    parser.add_argument("--global-threshold-floor", type=float, default=0.0)
    parser.add_argument("--min-train-weeks", type=int, default=2)
    parser.add_argument("--min-train-trades", type=int, default=12)
    parser.add_argument("--seed", type=int, default=104729)
    parser.add_argument("--keep-fracs", default="0.25,0.35,0.50,0.65,0.80,1.00")
    parser.add_argument("--min-net-pnl", type=float, default=0.0)
    parser.add_argument("--min-mean-objective", type=float, default=0.0)
    parser.add_argument("--min-positive-fold-share", type=float, default=0.67)
    parser.add_argument("--max-full-sl-rate", type=float, default=0.22)
    parser.add_argument("--max-timeout-rate", type=float, default=0.55)
    parser.add_argument("--min-worst-fold-net-pnl", type=float, default=-250.0)
    parser.add_argument("--max-no-trade-folds", type=int, default=0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame, features = _prepare_frame(args.candidates, feature_mode=str(args.feature_mode))
    scenario_frame = frame.loc[frame["scenario"].eq(str(args.scenario))].copy().reset_index(drop=True)
    if scenario_frame.empty:
        raise ValueError(f"No candidates found for scenario={args.scenario!r}")

    feature_set_hash = _hash_payload({"feature_mode": args.feature_mode, "features": features})
    feature_audit_df, feature_audit = _feature_audit_rows(features, frame=scenario_frame)
    keep_fracs = _parse_float_grid(str(args.keep_fracs))
    thresholds = _thresholds_from_args(args)

    guarded_candidate_frames: list[pd.DataFrame] = []
    decision_frames: list[pd.DataFrame] = []
    accepted_decision_frames: list[pd.DataFrame] = []
    fold_rows = []
    train_selection_rows: list[dict[str, Any]] = []

    for fold_id, validation_week in _scenario_folds(
        scenario_frame,
        min_train_weeks=int(args.min_train_weeks),
    ):
        validation_end = validation_week + pd.Timedelta(days=7)
        train = scenario_frame.loc[scenario_frame["timestamp"].lt(validation_week)].copy().reset_index(drop=True)
        validation = scenario_frame.loc[
            scenario_frame["timestamp"].ge(validation_week)
            & scenario_frame["timestamp"].lt(validation_end)
        ].copy().reset_index(drop=True)
        if train.empty or validation.empty:
            continue

        train_scores, validation_scores, model_name = _fit_predict_scores(
            str(args.method),
            train,
            validation,
            features,
            seed=int(args.seed) + int(fold_id),
        )
        keep_frac, threshold, selector_score, train_metrics = _select_keep_fraction(
            train,
            train_scores,
            keep_fracs,
            market_mode=str(args.market_mode),
            global_threshold_floor=float(args.global_threshold_floor),
            min_train_trades=int(args.min_train_trades),
        )
        model_spec = {
            "scenario": str(args.scenario),
            "method": str(args.method),
            "model_name": model_name,
            "feature_set_hash": feature_set_hash,
            "fold_id": int(fold_id),
            "validation_week": validation_week.date().isoformat(),
            "seed": int(args.seed) + int(fold_id),
            "train_rows": int(len(train)),
            "keep_frac": float(keep_frac),
            "threshold": float(threshold),
        }
        model_spec_hash = _hash_payload(model_spec)

        validation = validation.copy()
        validation["exec_guard_score_oof"] = validation_scores.astype(np.float32)
        validation["exec_guard_threshold"] = np.float32(threshold)
        validation["exec_guard_keep"] = validation_scores >= float(threshold)
        filtered_validation = _filter_by_train_quantile(
            validation,
            validation_scores,
            threshold=threshold,
        )
        guarded = _add_guard_metadata(
            filtered_validation,
            fold_id=int(fold_id),
            validation_week=validation_week,
            train=train,
            scenario=str(args.scenario),
            method=str(args.method),
            model_name=model_name,
            keep_frac=float(keep_frac),
            threshold=float(threshold),
            feature_set_hash=feature_set_hash,
            model_spec_hash=model_spec_hash,
        )
        guarded_candidate_frames.append(guarded)

        decisions, _equity, metrics = _replay_with_train_curve(
            train_candidates=train,
            eval_candidates=filtered_validation,
            market_mode=str(args.market_mode),
            global_threshold_floor=float(args.global_threshold_floor),
        )
        merged_decisions = _attach_candidates(decisions, filtered_validation)
        if not merged_decisions.empty:
            merged_decisions["decision_fold"] = int(fold_id)
            merged_decisions["validation_week"] = validation_week.date().isoformat()
            merged_decisions["exec_guard_method"] = str(args.method)
            merged_decisions["exec_guard_threshold"] = float(threshold)
            merged_decisions["exec_guard_keep_frac"] = float(keep_frac)
            decision_frames.append(merged_decisions)
            accepted = merged_decisions.loc[
                merged_decisions.get("accepted", pd.Series(False, index=merged_decisions.index)).astype(bool)
            ].copy()
            if not accepted.empty:
                accepted_decision_frames.append(accepted)

        fold_rows.append(
            _fold_result(
                scenario=str(args.scenario),
                fold_id=int(fold_id),
                validation_week=validation_week.date().isoformat(),
                variant=str(args.method),
                train_rows=len(train),
                validation_rows=len(validation),
                keep_frac=keep_frac,
                score_threshold=threshold,
                filtered_validation=filtered_validation,
                decisions=decisions,
                metrics=metrics,
                train_selector_score=selector_score,
            )
        )
        train_selection_rows.append(
            {
                "scenario": str(args.scenario),
                "fold_id": int(fold_id),
                "validation_week": validation_week.date().isoformat(),
                "method": str(args.method),
                "model_name": model_name,
                "keep_frac": float(keep_frac),
                "score_threshold": float(threshold),
                "train_selector_score": float(selector_score),
                "train_objective": float(train_metrics.get("objective", np.nan)),
                "train_net_pnl": float(train_metrics.get("net_pnl", np.nan)),
                "train_trade_count": int(train_metrics.get("trade_count", 0) or 0),
                "train_full_sl_rate": float(train_metrics.get("full_sl_rate", np.nan)),
                "train_timeout_rate": float(train_metrics.get("timeout_rate", np.nan)),
                "model_spec_hash": model_spec_hash,
            }
        )

    guarded_all = (
        pd.concat(guarded_candidate_frames, ignore_index=True)
        if guarded_candidate_frames
        else pd.DataFrame()
    )
    decision_all = (
        pd.concat(decision_frames, ignore_index=True) if decision_frames else pd.DataFrame()
    )
    accepted_all = (
        pd.concat(accepted_decision_frames, ignore_index=True)
        if accepted_decision_frames
        else pd.DataFrame()
    )
    folds_df = pd.DataFrame([row.__dict__ for row in fold_rows])
    summary_df = _summarise(folds_df, thresholds=thresholds)
    train_selection_df = pd.DataFrame(train_selection_rows)

    handoff_cols = _decision_time_columns(guarded_all)
    metadata_cols = [
        "base_selector",
        "base_score_oof",
        "meta_score_oof",
        "scenario_id",
        "horizon_bars",
        "barrier_mult",
        "stop_mult",
        "accepted",
        "guard_accepted",
        "exec_guard_score_oof",
        "exec_guard_threshold",
        "exec_guard_keep_frac",
        "exec_guard_method",
        "exec_guard_model_name",
        "exec_guard_feature_set_hash",
        "exec_guard_model_hash",
        "threshold_source",
        "decision_fold",
        "validation_week",
        "train_window_end",
        "train_rows",
    ]
    handoff_cols = [col for col in dict.fromkeys(handoff_cols + metadata_cols) if col in guarded_all.columns]
    guarded_handoff = guarded_all[handoff_cols].copy() if not guarded_all.empty else pd.DataFrame()

    duplicate_keys = ["timestamp", "symbol", "side", "strategy_id", "scenario_id"]
    duplicate_count = (
        int(guarded_handoff.duplicated([c for c in duplicate_keys if c in guarded_handoff.columns]).sum())
        if not guarded_handoff.empty
        else 0
    )
    if not guarded_handoff.empty and {
        "signal_timestamp",
        "guard_decision_timestamp",
    }.issubset(guarded_handoff.columns):
        signal_ts = pd.to_datetime(
            guarded_handoff["signal_timestamp"],
            utc=True,
            errors="coerce",
        )
        decision_ts = pd.to_datetime(
            guarded_handoff["guard_decision_timestamp"],
            utc=True,
            errors="coerce",
        )
        requires_delay = (
            guarded_handoff.get(
                "requires_delayed_entry_observation",
                pd.Series(False, index=guarded_handoff.index),
            )
            .fillna(False)
            .astype(bool)
        )
        missing_decision_ts = int(decision_ts.isna().sum())
        decision_before_signal = int((decision_ts < signal_ts).fillna(False).sum())
        same_time_decision = int((decision_ts == signal_ts).fillna(False).sum())
        delayed_observation_rows = int(requires_delay.sum())
        signal_time_no_delay_rows = int((~requires_delay).sum())
        missing_required_delay_ts = int((requires_delay & decision_ts.isna()).sum())
    else:
        missing_decision_ts = 0
        decision_before_signal = 0
        same_time_decision = 0
        delayed_observation_rows = 0
        signal_time_no_delay_rows = 0
        missing_required_delay_ts = 0
    fold_week_count = int(folds_df["validation_week"].nunique()) if "validation_week" in folds_df else 0
    feature_audit.update(
        {
            "guarded_candidate_rows": int(len(guarded_handoff)),
            "portfolio_accepted_rows": int(len(accepted_all)),
            "duplicate_timestamp_symbol_side_strategy_scenario_rows": duplicate_count,
            "missing_guard_decision_timestamp_rows": missing_decision_ts,
            "missing_required_delayed_entry_timestamp_rows": missing_required_delay_ts,
            "guard_decision_before_signal_rows": decision_before_signal,
            "guard_decision_equals_signal_rows": same_time_decision,
            "delayed_observation_decision_rows": delayed_observation_rows,
            "signal_time_no_delay_rows": signal_time_no_delay_rows,
            "validation_folds": fold_week_count,
            "base_meta_predictions_source": "source candidate artifact columns; expected OOF/prior-fold from upstream meta handoff",
            "guard_label_source": "prior-week replay net_return only; validation outcomes are not used for score fitting or threshold selection",
            "scenario_selection": "fixed before materialization",
            "scenario": str(args.scenario),
            "method": str(args.method),
            "feature_mode": str(args.feature_mode),
        }
    )
    feature_audit["leakage_safe_for_frozen_replay"] = bool(
        feature_audit["all_guard_inputs_decision_time_safe"]
        and duplicate_count == 0
        and missing_decision_ts == 0
        and missing_required_delay_ts == 0
        and decision_before_signal == 0
        and feature_audit["exit_spread_cost_matches_expected_half_spread"] is not False
    )
    feature_audit["leakage_safe_caveat"] = (
        "Guard is safe only if evaluated after the delayed-entry window, because "
        "delay-window features are included."
        if feature_audit["requires_guard_after_delayed_entry_window"]
        else ""
    )

    paths = {
        "guarded_candidates": args.out_dir / "meta_handoff_guarded_candidates.parquet",
        "offline_replay_candidates": args.out_dir
        / "meta_handoff_guarded_offline_replay_candidates.parquet",
        "execution_plan": args.out_dir / "meta_handoff_guarded_execution_plan.csv",
        "replay_decisions": args.out_dir / "meta_handoff_guarded_replay_decisions.parquet",
        "fold_summary": args.out_dir / "meta_handoff_guarded_fold_summary.csv",
        "summary": args.out_dir / "meta_handoff_guarded_summary.csv",
        "train_selection": args.out_dir / "meta_handoff_guarded_train_selection.csv",
        "feature_audit": args.out_dir / "meta_handoff_guarded_feature_audit.csv",
        "leakage_audit": args.out_dir / "meta_handoff_guarded_leakage_audit.json",
        "manifest": args.out_dir / "meta_handoff_guarded_manifest.json",
        "report": args.out_dir / "meta_handoff_guarded_report.md",
    }
    guarded_handoff.to_parquet(paths["guarded_candidates"], index=False)
    guarded_all.to_parquet(paths["offline_replay_candidates"], index=False)
    accepted_all.to_csv(paths["execution_plan"], index=False)
    decision_all.to_parquet(paths["replay_decisions"], index=False)
    folds_df.to_csv(paths["fold_summary"], index=False)
    summary_df.to_csv(paths["summary"], index=False)
    train_selection_df.to_csv(paths["train_selection"], index=False)
    feature_audit_df.to_csv(paths["feature_audit"], index=False)
    paths["leakage_audit"].write_text(
        json.dumps(_json_safe(feature_audit), indent=2),
        encoding="utf-8",
    )

    manifest = {
        "generated_by": "materialize_meta_handoff_guarded_execution",
        "source_candidates": str(args.candidates),
        "source_candidates_sha256": _sha256_path(args.candidates),
        "out_dir": str(args.out_dir),
        "scenario": str(args.scenario),
        "method": str(args.method),
        "feature_mode": str(args.feature_mode),
        "feature_columns": features,
        "feature_set_hash": feature_set_hash,
        "keep_fracs": keep_fracs,
        "thresholds": thresholds,
        "market_mode": str(args.market_mode),
        "global_threshold_floor": float(args.global_threshold_floor),
        "min_train_weeks": int(args.min_train_weeks),
        "min_train_trades": int(args.min_train_trades),
        "seed": int(args.seed),
        "guarded_candidate_rows": int(len(guarded_handoff)),
        "portfolio_accepted_rows": int(len(accepted_all)),
        "gate_summary": summary_df.to_dict(orient="records"),
        "leakage_audit": feature_audit,
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")

    report_lines = [
        "# Meta Handoff Guarded Execution Materialization",
        "",
        "Fixed scenario/materialized guard replay. Guard scores and thresholds are fit/selected on prior weeks only; replay decisions use guard-kept validation rows.",
        "",
        "## Summary",
        "",
        _fmt_table(
            summary_df,
            [
                "scenario",
                "variant",
                "pass_simple_policy_gate",
                "sum_net_pnl",
                "mean_objective",
                "worst_fold_net_pnl",
                "positive_fold_share",
                "accepted_trades",
                "mean_keep_frac",
                "weighted_full_sl_rate",
                "weighted_timeout_rate",
            ],
        ),
        "",
        "## Leakage Audit",
        "",
        _fmt_table(
            pd.DataFrame([feature_audit]),
            [
                "leakage_safe_for_frozen_replay",
                "all_guard_inputs_signal_time_safe",
                "all_guard_inputs_decision_time_safe",
                "requires_guard_after_delayed_entry_window",
                "guarded_candidate_rows",
                "portfolio_accepted_rows",
                "duplicate_timestamp_symbol_side_strategy_scenario_rows",
                "missing_guard_decision_timestamp_rows",
                "missing_required_delayed_entry_timestamp_rows",
                "guard_decision_before_signal_rows",
                "delayed_observation_decision_rows",
                "signal_time_no_delay_rows",
            ],
        ),
        "",
        "## Fold Detail",
        "",
        _fmt_table(
            folds_df,
            [
                "validation_week",
                "net_pnl",
                "objective",
                "accepted_trades",
                "full_sl_rate",
                "timeout_rate",
                "hit_rate",
                "keep_frac",
            ],
            max_rows=80,
        ),
    ]
    paths["report"].write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            _json_safe(
                {
                    "summary": summary_df.to_dict(orient="records"),
                    "leakage_audit": feature_audit,
                    "outputs": {key: str(value) for key, value in paths.items()},
                }
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
