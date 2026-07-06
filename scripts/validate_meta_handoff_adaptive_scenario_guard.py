#!/usr/bin/env python3
"""Walk-forward adaptive scenario guard for meta handoff candidates.

The fixed guarded policy has already passed for h9 x4. This script tests the
next, more flexible architecture: choose among pre-registered scenarios per
opportunity using only prior-fold guard models and thresholds, then replay the
chosen guarded rows. Promotion requires beating the fixed guarded replay.
"""

from __future__ import annotations

import argparse
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
    _parse_csv,
    _parse_float_grid,
    _prepare_frame,
    _replay_with_train_curve,
    _select_keep_fraction,
    _summarise,
)


DEFAULT_OUT_DIR = Path(
    "data_perp/reports/gmm_cluster_policy_smoke_20260702_wide_sidebalanced/"
    "meta_handoff_adaptive_scenario_guard_h9x4_h10x3_h12x3_v1"
)
DEFAULT_FIXED_SUMMARY = Path(
    "data_perp/reports/gmm_cluster_policy_smoke_20260702_wide_sidebalanced/"
    "meta_handoff_guarded_execution_h9x4_execnet_v1/frozen_replay_v1/"
    "frozen_replay_summary.csv"
)
BASE_KEY_COLUMNS = ["timestamp", "symbol", "side", "strategy_id"]


def _thresholds(args: argparse.Namespace) -> dict[str, float | int]:
    return {
        "min_net_pnl": float(args.min_net_pnl),
        "min_mean_objective": float(args.min_mean_objective),
        "min_positive_fold_share": float(args.min_positive_fold_share),
        "max_full_sl_rate": float(args.max_full_sl_rate),
        "max_timeout_rate": float(args.max_timeout_rate),
        "min_worst_fold_net_pnl": float(args.min_worst_fold_net_pnl),
        "max_no_trade_folds": int(args.max_no_trade_folds),
    }


def _filter_candidates(frame: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = frame.copy()
    start_rows = int(len(out))
    filters: list[str] = []

    def apply_max(col: str, value: float | None, name: str) -> None:
        nonlocal out
        if value is None or not np.isfinite(float(value)) or col not in out.columns:
            return
        out = out.loc[pd.to_numeric(out[col], errors="coerce").le(float(value)).fillna(False)].copy()
        filters.append(f"{name}<={float(value):.6g}")

    def apply_min(col: str, value: float | None, name: str) -> None:
        nonlocal out
        if value is None or not np.isfinite(float(value)) or col not in out.columns:
            return
        out = out.loc[pd.to_numeric(out[col], errors="coerce").ge(float(value)).fillna(False)].copy()
        filters.append(f"{name}>={float(value):.6g}")

    apply_max("archetype_joint_bad_risk", args.max_joint_bad_risk, "archetype_joint_bad_risk")
    apply_max("archetype_meta_bad_risk", args.max_meta_bad_risk, "archetype_meta_bad_risk")
    apply_max("archetype_joint_timeout_risk", args.max_joint_timeout_risk, "archetype_joint_timeout_risk")
    apply_max("archetype_meta_timeout_risk", args.max_meta_timeout_risk, "archetype_meta_timeout_risk")
    apply_min("rank_minus_joint_bad", args.min_rank_minus_joint_bad, "rank_minus_joint_bad")
    return out.reset_index(drop=True), {
        "input_rows": start_rows,
        "output_rows": int(len(out)),
        "removed_rows": int(start_rows - len(out)),
        "filters": filters,
    }


def _base_key(frame: pd.DataFrame) -> pd.Series:
    parts = []
    for col in BASE_KEY_COLUMNS:
        if col == "timestamp":
            values = pd.to_datetime(frame[col], utc=True, errors="coerce").astype(str)
        else:
            values = frame[col].astype(str)
        parts.append(values)
    key = parts[0]
    for values in parts[1:]:
        key = key + "|" + values
    return key


def _scenario_validation_predictions(
    *,
    scenario: str,
    scenario_frame: pd.DataFrame,
    validation_week: pd.Timestamp,
    method: str,
    features: list[str],
    keep_fracs: list[float],
    market_mode: str,
    global_threshold_floor: float,
    min_train_trades: int,
    seed: int,
    fold_id: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    validation_end = validation_week + pd.Timedelta(days=7)
    train = scenario_frame.loc[scenario_frame["timestamp"].lt(validation_week)].copy().reset_index(drop=True)
    validation = scenario_frame.loc[
        scenario_frame["timestamp"].ge(validation_week)
        & scenario_frame["timestamp"].lt(validation_end)
    ].copy().reset_index(drop=True)
    if train.empty or validation.empty:
        return pd.DataFrame(), pd.DataFrame(), {
            "scenario": scenario,
            "fold_id": int(fold_id),
            "validation_week": validation_week.date().isoformat(),
            "train_rows": int(len(train)),
            "validation_rows": int(len(validation)),
            "reason": "empty_train_or_validation",
        }
    train_scores, validation_scores, model_name = _fit_predict_scores(
        method,
        train,
        validation,
        features,
        seed=int(seed) + int(fold_id),
    )
    keep_frac, threshold, selector_score, train_metrics = _select_keep_fraction(
        train,
        train_scores,
        keep_fracs,
        market_mode=market_mode,
        global_threshold_floor=float(global_threshold_floor),
        min_train_trades=int(min_train_trades),
    )
    out = validation.copy()
    out["adaptive_guard_score_oof"] = validation_scores.astype(np.float32)
    out["adaptive_guard_threshold"] = np.float32(threshold)
    out["adaptive_guard_margin"] = (
        out["adaptive_guard_score_oof"].astype(float) - float(threshold)
    ).astype(np.float32)
    out["adaptive_guard_keep_frac"] = float(keep_frac)
    out["adaptive_guard_keep"] = out["adaptive_guard_margin"].ge(0.0)
    out["adaptive_guard_model_name"] = model_name
    out["adaptive_guard_method"] = method
    out["base_opportunity_key"] = _base_key(out)
    train_out = train.copy()
    train_out["adaptive_guard_score_oof"] = train_scores.astype(np.float32)
    train_out["adaptive_guard_threshold"] = np.float32(threshold)
    train_out["adaptive_guard_margin"] = (
        train_out["adaptive_guard_score_oof"].astype(float) - float(threshold)
    ).astype(np.float32)
    train_out["adaptive_guard_keep_frac"] = float(keep_frac)
    train_out["adaptive_guard_keep"] = train_out["adaptive_guard_margin"].ge(0.0)
    train_out["adaptive_guard_model_name"] = model_name
    train_out["adaptive_guard_method"] = method
    train_out["base_opportunity_key"] = _base_key(train_out)
    train_row = {
        "scenario": scenario,
        "fold_id": int(fold_id),
        "validation_week": validation_week.date().isoformat(),
        "method": method,
        "model_name": model_name,
        "keep_frac": float(keep_frac),
        "score_threshold": float(threshold),
        "train_selector_score": float(selector_score),
        "train_objective": float(train_metrics.get("objective", np.nan)),
        "train_net_pnl": float(train_metrics.get("net_pnl", np.nan)),
        "train_trade_count": int(train_metrics.get("trade_count", 0) or 0),
        "train_full_sl_rate": float(train_metrics.get("full_sl_rate", np.nan)),
        "train_timeout_rate": float(train_metrics.get("timeout_rate", np.nan)),
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
    }
    return out, train_out, train_row


def _choose_adaptive_rows(
    predictions: pd.DataFrame,
    *,
    mode: str,
    anchor_scenario: str = "h9_delay_1_barrier_x4",
    switch_margin_buffer: float = 0.0,
    require_anchor_admission: bool = False,
    require_keep: bool = True,
) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    work = predictions.copy()
    if mode == "anchor_margin_buffer":
        rows: list[pd.Series] = []
        for _key, group in work.groupby("base_opportunity_key", sort=False):
            anchor = group.loc[group["scenario"].astype(str).eq(str(anchor_scenario))]
            anchor_row = anchor.iloc[0] if not anchor.empty else None
            alternatives = group.loc[~group["scenario"].astype(str).eq(str(anchor_scenario))].copy()
            if require_keep:
                alternatives = alternatives.loc[alternatives["adaptive_guard_keep"].astype(bool)]
            if anchor_row is not None:
                anchor_keep = bool(anchor_row.get("adaptive_guard_keep", False))
                if require_anchor_admission and not anchor_keep:
                    continue
                if (not require_keep) or anchor_keep:
                    base_margin = float(anchor_row.get("adaptive_guard_margin", -np.inf))
                    if not alternatives.empty:
                        alternatives = alternatives.assign(
                            _switch_edge=alternatives["adaptive_guard_margin"].astype(float)
                            - base_margin
                        )
                        candidate = alternatives.sort_values(
                            ["_switch_edge", "adaptive_guard_margin", "adaptive_guard_score_oof"],
                            ascending=[False, False, False],
                        ).iloc[0]
                        if float(candidate["_switch_edge"]) >= float(switch_margin_buffer):
                            rows.append(candidate.drop(labels=["_switch_edge"], errors="ignore"))
                        else:
                            rows.append(anchor_row)
                    else:
                        rows.append(anchor_row)
                    continue
            pool = group.loc[group["adaptive_guard_keep"].astype(bool)] if require_keep else group
            if not pool.empty:
                rows.append(
                    pool.sort_values(
                        ["adaptive_guard_margin", "adaptive_guard_score_oof"],
                        ascending=[False, False],
                    ).iloc[0]
                )
        return pd.DataFrame(rows).reset_index(drop=True) if rows else pd.DataFrame()

    if mode == "best_score_then_threshold":
        sort_cols = ["base_opportunity_key", "adaptive_guard_score_oof", "adaptive_guard_margin"]
    else:
        sort_cols = ["base_opportunity_key", "adaptive_guard_margin", "adaptive_guard_score_oof"]
    chosen = (
        work.sort_values(sort_cols, ascending=[True, False, False])
        .groupby("base_opportunity_key", sort=False, as_index=False)
        .head(1)
        .copy()
    )
    if require_keep:
        chosen = chosen.loc[chosen["adaptive_guard_keep"].astype(bool)]
    return chosen.reset_index(drop=True)


def _attach_candidates(decisions: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    if decisions.empty or "candidate_index" not in decisions.columns:
        return decisions.copy()
    out = decisions.copy()
    out["candidate_index"] = pd.to_numeric(out["candidate_index"], errors="coerce").astype("Int64")
    payload_cols = [
        "timestamp",
        "symbol",
        "side",
        "strategy_id",
        "scenario",
        "base_opportunity_key",
        "adaptive_guard_score_oof",
        "adaptive_guard_threshold",
        "adaptive_guard_margin",
        "adaptive_guard_keep_frac",
        "net_return",
        "gross_return",
        "simple_policy_exit_reason",
    ]
    payload_cols = [col for col in payload_cols if col in candidates.columns]
    payload = candidates[payload_cols].reset_index(names="candidate_index")
    return out.merge(payload, on="candidate_index", how="left", suffixes=("", "_candidate"))


def _fixed_comparator(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"available": False}
    df = pd.read_csv(path)
    if df.empty:
        return {"available": False}
    row = df.iloc[0].to_dict()
    row["available"] = True
    return row


def _adaptive_beats_fixed(
    summary: pd.DataFrame,
    fixed: Mapping[str, Any],
    *,
    tolerance: float = 1e-12,
) -> dict[str, Any]:
    if summary.empty or not bool(fixed.get("available")):
        return {"available": bool(fixed.get("available")), "passes": False}
    row = summary.iloc[0]
    tol = float(max(tolerance, 0.0))
    checks = {
        "sum_net_pnl_gte_fixed": float(row["sum_net_pnl"]) + tol >= float(fixed["sum_net_pnl"]),
        "mean_objective_gte_fixed": float(row["mean_objective"]) + tol >= float(fixed["mean_objective"]),
        "positive_fold_share_gte_fixed": float(row["positive_fold_share"])
        + tol
        >= float(fixed["positive_fold_share"]),
        "worst_fold_net_pnl_gte_fixed": float(row["worst_fold_net_pnl"])
        + tol
        >= float(fixed["worst_fold_net_pnl"]),
        "weighted_full_sl_rate_lte_fixed": float(row["weighted_full_sl_rate"])
        <= float(fixed["weighted_full_sl_rate"]) + tol,
        "weighted_timeout_rate_lte_fixed": float(row["weighted_timeout_rate"])
        <= float(fixed["weighted_timeout_rate"]) + tol,
        "no_trade_folds_lte_fixed": int(row["no_trade_folds"]) <= int(fixed["no_trade_folds"]),
    }
    return {
        "available": True,
        "fixed_summary": {k: _json_safe(v) for k, v in fixed.items()},
        "checks": checks,
        "passes": bool(all(checks.values())),
    }


def _scenario_distribution(frame: pd.DataFrame, *, label: str) -> pd.DataFrame:
    if frame.empty or "scenario" not in frame.columns:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    group_cols = ["validation_week", "scenario"] if "validation_week" in frame.columns else ["scenario"]
    for keys, group in frame.groupby(group_cols, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {"row_type": label, "rows": int(len(group))}
        for col, value in zip(group_cols, keys):
            row[col] = value
        if "position_net_return" in group.columns:
            row["accepted_net_pnl"] = float(
                (
                    pd.to_numeric(group.get("position_size", 0.0), errors="coerce").fillna(0.0)
                    * pd.to_numeric(group["position_net_return"], errors="coerce").fillna(0.0)
                ).sum()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _fmt_table(frame: pd.DataFrame, cols: list[str], max_rows: int = 40) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame[[col for col in cols if col in frame.columns]].head(max_rows).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:,.6f}")
    return view.to_markdown(index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--scenarios",
        default="h9_delay_1_barrier_x4,h10_delay_1_barrier_x3,h12_delay_1_barrier_x3",
    )
    parser.add_argument("--method", default="exec_net_regressor")
    parser.add_argument(
        "--selection-mode",
        default="best_margin",
        choices=["best_margin", "best_score_then_threshold", "anchor_margin_buffer"],
    )
    parser.add_argument("--anchor-scenario", default="h9_delay_1_barrier_x4")
    parser.add_argument("--switch-margin-buffer", type=float, default=0.0)
    parser.add_argument("--require-anchor-admission", action="store_true")
    parser.add_argument("--feature-mode", choices=["base", "execution_known"], default="execution_known")
    parser.add_argument("--market-mode", default="perps", choices=["spot", "perps"])
    parser.add_argument("--global-threshold-floor", type=float, default=0.0)
    parser.add_argument("--min-train-weeks", type=int, default=2)
    parser.add_argument("--min-train-trades", type=int, default=12)
    parser.add_argument("--seed", type=int, default=104729)
    parser.add_argument("--keep-fracs", default="0.25,0.35,0.50,0.65,0.80,1.00")
    parser.add_argument("--fixed-summary", type=Path, default=DEFAULT_FIXED_SUMMARY)
    parser.add_argument("--min-net-pnl", type=float, default=0.0)
    parser.add_argument("--min-mean-objective", type=float, default=0.0)
    parser.add_argument("--min-positive-fold-share", type=float, default=0.67)
    parser.add_argument("--max-full-sl-rate", type=float, default=0.22)
    parser.add_argument("--max-timeout-rate", type=float, default=0.55)
    parser.add_argument("--min-worst-fold-net-pnl", type=float, default=-250.0)
    parser.add_argument("--max-no-trade-folds", type=int, default=0)
    parser.add_argument("--max-joint-bad-risk", type=float, default=None)
    parser.add_argument("--max-meta-bad-risk", type=float, default=None)
    parser.add_argument("--max-joint-timeout-risk", type=float, default=None)
    parser.add_argument("--max-meta-timeout-risk", type=float, default=None)
    parser.add_argument("--min-rank-minus-joint-bad", type=float, default=None)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame, features = _prepare_frame(args.candidates, feature_mode=str(args.feature_mode))
    frame, candidate_filter = _filter_candidates(frame, args)
    scenarios = _parse_csv(str(args.scenarios))
    keep_fracs = _parse_float_grid(str(args.keep_fracs))
    thresholds = _thresholds(args)
    scenario_frames = {
        scenario: frame.loc[frame["scenario"].eq(scenario)].copy().reset_index(drop=True)
        for scenario in scenarios
    }
    scenario_frames = {key: value for key, value in scenario_frames.items() if not value.empty}
    if not scenario_frames:
        raise ValueError(f"No candidates found for scenarios={scenarios}")
    all_weeks = sorted(
        pd.to_datetime(
            pd.concat([sf["week_start"] for sf in scenario_frames.values()], ignore_index=True),
            utc=True,
        )
        .dropna()
        .unique()
    )

    fold_rows = []
    selected_frames: list[pd.DataFrame] = []
    train_ev_frames: list[pd.DataFrame] = []
    decision_frames: list[pd.DataFrame] = []
    train_selection_rows: list[dict[str, Any]] = []
    for fold_id, validation_week in enumerate(all_weeks[int(args.min_train_weeks) :]):
        validation_week = pd.Timestamp(validation_week)
        prediction_frames: list[pd.DataFrame] = []
        train_prediction_frames: list[pd.DataFrame] = []
        for scenario, scenario_frame in scenario_frames.items():
            preds, train_preds, train_row = _scenario_validation_predictions(
                scenario=scenario,
                scenario_frame=scenario_frame,
                validation_week=validation_week,
                method=str(args.method),
                features=features,
                keep_fracs=keep_fracs,
                market_mode=str(args.market_mode),
                global_threshold_floor=float(args.global_threshold_floor),
                min_train_trades=int(args.min_train_trades),
                seed=int(args.seed),
                fold_id=int(fold_id),
            )
            train_selection_rows.append(train_row)
            if not preds.empty:
                prediction_frames.append(preds)
            if not train_preds.empty:
                train_prediction_frames.append(train_preds)
        validation_predictions = (
            pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
        )
        train_predictions = (
            pd.concat(train_prediction_frames, ignore_index=True)
            if train_prediction_frames
            else pd.DataFrame()
        )
        selected = _choose_adaptive_rows(
            validation_predictions,
            mode=str(args.selection_mode),
            anchor_scenario=str(args.anchor_scenario),
            switch_margin_buffer=float(args.switch_margin_buffer),
            require_anchor_admission=bool(args.require_anchor_admission),
            require_keep=True,
        )
        if not selected.empty:
            selected["validation_week"] = validation_week.date().isoformat()
            selected_frames.append(selected)

        train_all = _choose_adaptive_rows(
            train_predictions,
            mode=str(args.selection_mode),
            anchor_scenario=str(args.anchor_scenario),
            switch_margin_buffer=float(args.switch_margin_buffer),
            require_anchor_admission=bool(args.require_anchor_admission),
            require_keep=False,
        )
        if not train_all.empty:
            train_all = train_all.copy()
            train_all["validation_week"] = validation_week.date().isoformat()
            train_all["decision_fold"] = int(fold_id)
            train_ev_frames.append(train_all)
        decisions, _equity, metrics = _replay_with_train_curve(
            train_candidates=train_all,
            eval_candidates=selected,
            market_mode=str(args.market_mode),
            global_threshold_floor=float(args.global_threshold_floor),
        )
        attached = _attach_candidates(decisions, selected)
        if not attached.empty:
            attached["fold_id"] = int(fold_id)
            attached["validation_week"] = validation_week.date().isoformat()
            decision_frames.append(attached)
        fold_rows.append(
            _fold_result(
                scenario="adaptive_scenario",
                fold_id=int(fold_id),
                validation_week=validation_week.date().isoformat(),
                variant=f"{args.method}_{args.selection_mode}",
                train_rows=len(train_all),
                validation_rows=len(validation_predictions),
                keep_frac=float(len(selected) / max(validation_predictions["base_opportunity_key"].nunique(), 1))
                if not validation_predictions.empty
                else 0.0,
                score_threshold=0.0,
                filtered_validation=selected,
                decisions=decisions,
                metrics=metrics,
                train_selector_score=float("nan"),
            )
        )

    folds_df = pd.DataFrame([row.__dict__ for row in fold_rows])
    summary_df = _summarise(folds_df, thresholds=thresholds)
    selected_df = pd.concat(selected_frames, ignore_index=True) if selected_frames else pd.DataFrame()
    train_ev_df = pd.concat(train_ev_frames, ignore_index=True) if train_ev_frames else pd.DataFrame()
    decisions_df = pd.concat(decision_frames, ignore_index=True) if decision_frames else pd.DataFrame()
    train_selection_df = pd.DataFrame(train_selection_rows)
    fixed = _fixed_comparator(args.fixed_summary)
    fixed_gate = _adaptive_beats_fixed(summary_df, fixed)
    scenario_dist = pd.concat(
        [
            _scenario_distribution(selected_df, label="selected_candidates"),
            _scenario_distribution(
                decisions_df.loc[decisions_df.get("accepted", pd.Series(False, index=decisions_df.index)).astype(bool)]
                if not decisions_df.empty and "accepted" in decisions_df.columns
                else pd.DataFrame(),
                label="portfolio_accepted",
            ),
        ],
        ignore_index=True,
    )
    adaptive_pass = bool(
        not summary_df.empty
        and bool(summary_df["pass_simple_policy_gate"].iloc[0])
        and bool(fixed_gate.get("passes"))
    )

    paths = {
        "folds": args.out_dir / "adaptive_scenario_guard_folds.csv",
        "summary": args.out_dir / "adaptive_scenario_guard_summary.csv",
        "selected_candidates": args.out_dir / "adaptive_scenario_guard_selected_candidates.parquet",
        "train_ev_candidates": args.out_dir / "adaptive_scenario_guard_train_ev_candidates.parquet",
        "decisions": args.out_dir / "adaptive_scenario_guard_decisions.parquet",
        "train_selection": args.out_dir / "adaptive_scenario_guard_train_selection.csv",
        "scenario_distribution": args.out_dir / "adaptive_scenario_guard_scenario_distribution.csv",
        "manifest": args.out_dir / "manifest.json",
        "report": args.out_dir / "adaptive_scenario_guard_report.md",
    }
    folds_df.to_csv(paths["folds"], index=False)
    summary_df.to_csv(paths["summary"], index=False)
    selected_df.to_parquet(paths["selected_candidates"], index=False)
    train_ev_df.to_parquet(paths["train_ev_candidates"], index=False)
    decisions_df.to_parquet(paths["decisions"], index=False)
    train_selection_df.to_csv(paths["train_selection"], index=False)
    scenario_dist.to_csv(paths["scenario_distribution"], index=False)
    manifest = {
        "generated_by": "validate_meta_handoff_adaptive_scenario_guard",
        "candidates": str(args.candidates),
        "out_dir": str(args.out_dir),
        "scenarios": scenarios,
        "method": str(args.method),
        "selection_mode": str(args.selection_mode),
        "anchor_scenario": str(args.anchor_scenario),
        "switch_margin_buffer": float(args.switch_margin_buffer),
        "require_anchor_admission": bool(args.require_anchor_admission),
        "feature_mode": str(args.feature_mode),
        "feature_columns": features,
        "candidate_filter": candidate_filter,
        "keep_fracs": keep_fracs,
        "min_train_weeks": int(args.min_train_weeks),
        "min_train_trades": int(args.min_train_trades),
        "market_mode": str(args.market_mode),
        "global_threshold_floor": float(args.global_threshold_floor),
        "seed": int(args.seed),
        "thresholds": thresholds,
        "fixed_summary": str(args.fixed_summary),
        "fixed_comparator_gate": fixed_gate,
        "pass_adaptive_scenario_gate": adaptive_pass,
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    lines = [
        "# Adaptive Scenario Guard Walk-Forward",
        "",
        "Per fold, guard models and scenario thresholds are fit on prior rows only. The validation opportunity chooses one of the pre-registered scenarios by guard margin and is replayed only if it clears that scenario threshold.",
        "",
        "## Gate",
        "",
        pd.DataFrame(
            [
                {
                    "pass_adaptive_scenario_gate": adaptive_pass,
                    "passes_base_thresholds": bool(summary_df["pass_simple_policy_gate"].iloc[0]) if not summary_df.empty else False,
                    "beats_fixed_guard": bool(fixed_gate.get("passes")),
                    "fixed_available": bool(fixed_gate.get("available")),
                }
            ]
        ).to_markdown(index=False),
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
                "weighted_full_sl_rate",
                "weighted_timeout_rate",
            ],
        ),
        "",
        "## Fixed Comparator Checks",
        "",
        pd.DataFrame(
            [
                {"check": key, "pass": value}
                for key, value in (fixed_gate.get("checks") or {}).items()
            ]
        ).to_markdown(index=False)
        if fixed_gate.get("checks")
        else "_No fixed comparator available._",
        "",
        "## Scenario Distribution",
        "",
        _fmt_table(
            scenario_dist,
            ["row_type", "validation_week", "scenario", "rows", "accepted_net_pnl"],
            max_rows=80,
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
    paths["report"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            _json_safe(
                {
                    "pass_adaptive_scenario_gate": adaptive_pass,
                    "summary": summary_df.to_dict(orient="records"),
                    "fixed_comparator_gate": fixed_gate,
                    "outputs": {key: str(value) for key, value in paths.items()},
                }
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
