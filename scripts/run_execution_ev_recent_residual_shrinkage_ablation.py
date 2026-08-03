#!/usr/bin/env python3
"""Strict weekly forward-OOS shrinkage for the recent execution-EV adapter."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.execution_ev_meta import execution_ev_metrics  # noqa: E402
from scripts.run_execution_ev_mixed_period_remedies import (  # noqa: E402
    ARCHETYPE_COLUMN,
    BASELINE_COLUMN,
    DECISION_COLUMN,
    RESOLUTION_COLUMN,
    SIDE_COLUMN,
    TARGET_COLUMN,
    Arm,
    ForwardWindow,
    apply_canonical_recent_mapping,
    build_forward_split,
    fit_arm_scores,
    recent_residual_correction,
)


SCHEMA = "execution_ev_recent_residual_shrinkage_ablation_v1"
FIXED_SHRINKAGES = (0.0, 0.10, 0.25, 0.50, 0.75, 1.0)
DRIFT_COLUMNS = (
    BASELINE_COLUMN,
    "base_oof_score",
    "base_margin_to_cutoff_z",
    "oof_clean_favorable_probability",
    "alpha_prediction_uncertainty",
    "catboost_entropy",
    "catboost_p_0",
    "catboost_p_1",
    "catboost_p_2",
    "catboost_p_3",
    "catboost_p_4",
    "catboost_p_5",
    "catboost_p_6",
)


def weekly_forward_windows(
    *,
    start: str = "2026-06-01T00:00:00Z",
    end: str = "2026-07-20T00:00:00Z",
) -> tuple[ForwardWindow, ...]:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    windows: list[ForwardWindow] = []
    cutoff = start_ts
    while cutoff < end_ts:
        evaluation_end = min(cutoff + pd.Timedelta(days=7), end_ts)
        windows.append(
            ForwardWindow(
                name=f"week_{cutoff.strftime('%Y%m%d')}",
                train_start="2026-05-01T00:00:00Z",
                cutoff=cutoff.isoformat(),
                evaluation_end=evaluation_end.isoformat(),
                retention_role=(
                    "may_to_june_weekly_forward_retention"
                    if cutoff.month == 6
                    else "july_weekly_forward_research_oos"
                ),
            )
        )
        cutoff += pd.Timedelta(days=7)
    return tuple(windows)


def _weighted_ess(report: dict[str, Any]) -> float:
    values = [
        float(side.get("weights", {}).get("effective_sample_size", 0.0))
        for side in report.values()
        if isinstance(side, dict)
    ]
    return float(sum(values))


def training_only_adaptive_shrinkage(
    train: pd.DataFrame,
    residual_report: dict[str, Any],
    *,
    base_shrink: float = 0.50,
    min_side_ess: float = 500.0,
    min_period_rows: int = 500,
    zero_drift_threshold: float = 3.0,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Compute bounded side shrinkage from training support and covariate drift.

    No realized evaluation label or evaluation-period regime count is consumed.
    An unavailable adapter, weak support, missing prior window, or extreme
    training-only shift falls back explicitly to zero.
    """
    decision = pd.to_datetime(train[DECISION_COLUMN], utc=True, errors="raise")
    snapshot = decision.max() + pd.Timedelta(microseconds=1)
    columns = [column for column in DRIFT_COLUMNS if column in train]
    shrink: dict[str, float] = {}
    audit: dict[str, Any] = {}
    for side in ("long", "short"):
        side_mask = train[SIDE_COLUMN].astype(str).str.lower().eq(side)
        recent_mask = (
            side_mask
            & decision.ge(snapshot - pd.Timedelta(days=7))
            & decision.lt(snapshot)
        )
        prior_mask = (
            side_mask
            & decision.ge(snapshot - pd.Timedelta(days=21))
            & decision.lt(snapshot - pd.Timedelta(days=7))
        )
        side_report = residual_report.get(side, {})
        ess = float(
            side_report.get("weights", {}).get("effective_sample_size", 0.0)
        )
        fallback_reason = None
        shifts: list[float] = []
        if side_report.get("status") != "fit_on_recent_oof_residuals":
            fallback_reason = "adapter_unavailable"
        elif ess < float(min_side_ess):
            fallback_reason = "insufficient_effective_sample_size"
        elif int(recent_mask.sum()) < int(min_period_rows):
            fallback_reason = "insufficient_recent_rows"
        elif int(prior_mask.sum()) < int(min_period_rows):
            fallback_reason = "insufficient_prior_rows"
        else:
            for column in columns:
                recent = pd.to_numeric(
                    train.loc[recent_mask, column], errors="coerce"
                ).to_numpy(dtype=float)
                prior = pd.to_numeric(
                    train.loc[prior_mask, column], errors="coerce"
                ).to_numpy(dtype=float)
                if not (np.isfinite(recent).all() and np.isfinite(prior).all()):
                    continue
                scale = max(
                    float(np.sqrt((np.var(recent) + np.var(prior)) / 2.0)),
                    1e-6,
                )
                shifts.append(abs(float(np.mean(recent) - np.mean(prior))) / scale)
            if not shifts:
                fallback_reason = "no_finite_drift_features"
        median_shift = float(np.median(shifts)) if shifts else float("nan")
        if fallback_reason is None and median_shift >= float(zero_drift_threshold):
            fallback_reason = "extreme_training_drift"
        if fallback_reason is not None:
            value = 0.0
        else:
            ess_factor = min(1.0, ess / 5_000.0)
            drift_factor = float(np.exp(-max(median_shift, 0.0)))
            value = float(np.clip(base_shrink * ess_factor * drift_factor, 0.0, 1.0))
        shrink[side] = value
        audit[side] = {
            "shrink": value,
            "fallback_reason": fallback_reason,
            "effective_sample_size": ess,
            "recent_rows": int(recent_mask.sum()),
            "prior_rows": int(prior_mask.sum()),
            "drift_feature_count": int(len(shifts)),
            "median_standardized_mean_shift": (
                median_shift if np.isfinite(median_shift) else None
            ),
            "contract": "training_only_support_and_covariate_drift",
        }
    audit["combined_effective_sample_size"] = _weighted_ess(residual_report)
    return shrink, audit


def _apply_side_shrink(
    frame: pd.DataFrame,
    base: np.ndarray,
    delta: np.ndarray,
    shrink: float | dict[str, float],
) -> np.ndarray:
    out = np.asarray(base, dtype=float).copy()
    sides = frame[SIDE_COLUMN].astype(str).str.lower().to_numpy()
    if isinstance(shrink, dict):
        multiplier = np.asarray([float(shrink.get(side, 0.0)) for side in sides])
    else:
        multiplier = np.full(len(out), float(shrink))
    return out + multiplier * np.asarray(delta, dtype=float)


def _arm_name(shrink: float | str) -> str:
    if isinstance(shrink, str):
        return shrink
    return f"shrink_{float(shrink):.2f}".replace(".", "p")


def policy_global_topk_mask(
    frame: pd.DataFrame,
    score_col: str,
    fraction: float = 0.10,
) -> np.ndarray:
    values = pd.to_numeric(frame[score_col], errors="raise").to_numpy(dtype=float)
    if not 0.0 < float(fraction) <= 1.0 or not np.isfinite(values).all():
        raise ValueError("global top-k requires finite scores and fraction in (0, 1]")
    take = max(1, int(np.ceil(float(fraction) * len(values))))
    tie_columns = [
        column
        for column in (DECISION_COLUMN, "__symbol__", SIDE_COLUMN, "candidate_id")
        if column in frame
    ]
    order = (
        frame.assign(__policy_score__=values, __position__=np.arange(len(frame)))
        .sort_values(
            ["__policy_score__", *tie_columns],
            ascending=[False, *([True] * len(tie_columns))],
            kind="stable",
        )
        .head(take)["__position__"]
        .to_numpy(dtype=int)
    )
    mask = np.zeros(len(values), dtype=bool)
    mask[order] = True
    return mask


def _metrics(
    actual: np.ndarray,
    score: np.ndarray,
    *,
    scope: str,
    arm: str,
    week: str,
    stage: str,
) -> dict[str, Any]:
    result = execution_ev_metrics(actual, score, top_k_fraction=0.10)
    return {
        "scope": scope,
        "week": week,
        "arm": arm,
        "stage": stage,
        "promotion_eligible": False,
        "eligible_rows": int(len(actual)),
        "coverage_rate": float(np.isfinite(score).mean()),
        **result,
        "top_k_mean_net_ev_bps": float(10_000.0 * result["top_k_mean_net_ev"]),
    }


def _period_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for arm, arm_frame in predictions.groupby("arm", sort=True):
        decision = pd.to_datetime(arm_frame[DECISION_COLUMN], utc=True)
        periods = {
            "calendar_june_forward_retention": decision.dt.month.eq(6).to_numpy(),
            "calendar_july_forward": decision.dt.month.eq(7).to_numpy(),
            "all_weekly_forward_oos": np.ones(len(arm_frame), dtype=bool),
        }
        for scope, mask in periods.items():
            if not mask.any():
                continue
            rows.append(
                _metrics(
                    arm_frame.loc[mask, TARGET_COLUMN].to_numpy(dtype=float),
                    arm_frame.loc[
                        mask, "prediction_canonical_recent_ev_mapping"
                    ].to_numpy(dtype=float),
                    scope=scope,
                    arm=str(arm),
                    week="pooled_across_weekly_oos_rows",
                    stage="canonical_recent_ev_mapping",
                )
            )
    return pd.DataFrame(rows)


def _select_robust_arm(
    weekly: pd.DataFrame,
    periods: pd.DataFrame,
    *,
    latest_week: str,
) -> tuple[str, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    mapped = weekly.loc[
        weekly["stage"].eq("canonical_recent_ev_mapping")
        & weekly["scope"].eq("weekly_global_top10")
    ]
    for arm, group in mapped.groupby("arm", sort=True):
        values = group["top_k_mean_net_ev"].to_numpy(dtype=float)
        june = periods.loc[
            periods["arm"].eq(arm)
            & periods["scope"].eq("calendar_june_forward_retention")
        ]
        latest = group.loc[group["week"].eq(latest_week)]
        objective = float(
            np.mean(values) - 0.5 * np.std(values) + 0.25 * np.min(values)
        )
        rows.append(
            {
                "arm": arm,
                "weekly_mean_net_ev": float(np.mean(values)),
                "weekly_std_net_ev": float(np.std(values)),
                "worst_week_net_ev": float(np.min(values)),
                "stability_objective": objective,
                "june_pooled_top10_net_ev": (
                    float(june["top_k_mean_net_ev"].iloc[0]) if len(june) else np.nan
                ),
                "latest_week": latest_week,
                "latest_week_top10_net_ev": (
                    float(latest["top_k_mean_net_ev"].iloc[0])
                    if len(latest)
                    else np.nan
                ),
                "latest_week_coverage": (
                    float(latest["coverage_rate"].iloc[0]) if len(latest) else 0.0
                ),
                "promotion_eligible": False,
            }
        )
    leaderboard = pd.DataFrame(rows).sort_values(
        ["stability_objective", "june_pooled_top10_net_ev"],
        ascending=False,
        kind="stable",
    )
    viable = leaderboard.loc[
        leaderboard["june_pooled_top10_net_ev"].gt(0.0)
        & leaderboard["latest_week_coverage"].ge(1.0)
    ]
    winner = str((viable if len(viable) else leaderboard).iloc[0]["arm"])
    leaderboard["research_winner"] = leaderboard["arm"].eq(winner)
    return winner, leaderboard


def run(args: argparse.Namespace) -> dict[str, Path]:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    frame = pd.read_parquet(args.input)
    feature_manifest = json.loads(
        Path(args.feature_manifest).read_text(encoding="utf-8")
    )
    feature_columns = list(feature_manifest["feature_columns"])
    for column in feature_columns:
        prefix = "catboost_archetype__"
        if column.startswith(prefix) and column not in frame:
            frame[column] = (
                frame[ARCHETYPE_COLUMN]
                .astype(str)
                .eq(column[len(prefix) :])
                .astype("float32")
            )
    frame = frame.sort_values(
        [DECISION_COLUMN, "candidate_id"], kind="stable"
    ).reset_index(drop=True)
    prediction_parts: list[pd.DataFrame] = []
    weekly_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    adapter_audit: dict[str, Any] = {}
    windows = weekly_forward_windows()
    for window in windows:
        train_pos, eval_pos, split_audit = build_forward_split(
            frame, window, purge_hours=args.purge_hours
        )
        train = frame.iloc[train_pos].copy().reset_index(drop=True)
        evaluation = frame.iloc[eval_pos].copy().reset_index(drop=True)
        split_rows.append(split_audit)
        base_arm = Arm("uniform_all_available")
        train_oof, eval_base, base_report = fit_arm_scores(
            train,
            evaluation,
            feature_columns,
            base_arm,
            iterations=args.n_estimators,
            seed=args.random_state,
            n_jobs=args.n_jobs,
        )
        train_full, eval_full, residual_report = recent_residual_correction(
            train,
            evaluation,
            train_oof,
            eval_base,
            feature_columns,
            shrink=1.0,
            seed=args.random_state,
            n_jobs=args.n_jobs,
        )
        train_delta = train_full - train_oof
        eval_delta = eval_full - eval_base
        adaptive, adaptive_audit = training_only_adaptive_shrinkage(
            train, residual_report
        )
        adapter_audit[window.name] = {
            "base_model": base_report,
            "residual_adapter": residual_report,
            "adaptive_shrinkage": adaptive_audit,
        }
        arms: list[tuple[str, float | dict[str, float]]] = [
            (_arm_name(value), value) for value in FIXED_SHRINKAGES
        ]
        arms.append(("adaptive_ess_drift", adaptive))
        for arm_name, shrink in arms:
            corrected_train = _apply_side_shrink(
                train, train_oof, train_delta, shrink
            )
            corrected_eval = _apply_side_shrink(
                evaluation, eval_base, eval_delta, shrink
            )
            mapped, mapping_report = apply_canonical_recent_mapping(
                train, evaluation, corrected_train, corrected_eval
            )
            weekly_rows.append(
                _metrics(
                    evaluation[TARGET_COLUMN].to_numpy(dtype=float),
                    mapped,
                    scope="weekly_global_top10",
                    arm=arm_name,
                    week=window.name,
                    stage="canonical_recent_ev_mapping",
                )
            )
            identity = [
                column
                for column in (
                    "__ts__",
                    "__symbol__",
                    SIDE_COLUMN,
                    "candidate_id",
                    DECISION_COLUMN,
                    RESOLUTION_COLUMN,
                    TARGET_COLUMN,
                )
                if column in evaluation
            ]
            prediction_parts.append(
                evaluation.loc[:, identity].assign(
                    week=window.name,
                    arm=arm_name,
                    prediction_pre_recent_mapping=corrected_eval,
                    prediction_canonical_recent_ev_mapping=mapped,
                    promotion_eligible=False,
                )
            )
            adapter_audit[window.name].setdefault("mapping", {})[
                arm_name
            ] = mapping_report
    predictions = pd.concat(prediction_parts, ignore_index=True)
    weekly = pd.DataFrame(weekly_rows)
    periods = _period_metrics(predictions)
    latest_week = windows[-1].name
    winner, leaderboard = _select_robust_arm(
        weekly, periods, latest_week=latest_week
    )
    portfolio = predictions.loc[predictions["arm"].eq(winner)].copy()
    # Match the downstream exact portfolio replay tie contract: descending
    # score, then chronological/symbol/side/candidate identity.
    portfolio["global_top10_selected"] = policy_global_topk_mask(
        portfolio,
        "prediction_canonical_recent_ev_mapping",
        0.10,
    )
    portfolio["score_contract"] = (
        "weekly forward OOS model; temporal correction OOF; canonical recent EV; "
        "one pooled global rank across timestamps and sides"
    )
    weekly.to_csv(output / "weekly_metrics.csv", index=False)
    periods.to_csv(output / "period_metrics.csv", index=False)
    leaderboard.to_csv(output / "shrinkage_leaderboard.csv", index=False)
    pd.DataFrame(split_rows).to_csv(output / "weekly_splits.csv", index=False)
    predictions.to_parquet(output / "all_predictions.parquet", index=False)
    portfolio.to_parquet(output / "portfolio_ready_winner_scores.parquet", index=False)
    manifest = {
        "schema": SCHEMA,
        "status": "completed",
        "input": {"path": str(args.input), "rows": int(len(frame))},
        "feature_manifest": str(args.feature_manifest),
        "feature_columns": feature_columns,
        "regime_contract": (
            "regime definitions may be model inputs/supporting labels only; "
            "no regime field is used in sample weighting"
        ),
        "weights": "uniform base; residual adapter uses training-only recency; no regime weights",
        "shrinkages": list(FIXED_SHRINKAGES),
        "adaptive_contract": (
            "per-side 0.5 base shrink x ESS factor x exp(-training-only drift); "
            "explicit zero fallback for missing adapter, ESS, prior/recent support, "
            "or extreme drift"
        ),
        "selection_contract": (
            "canonical 21d causal side x predicted-archetype recent-EV score; "
            "one pooled global top10; never per timestamp"
        ),
        "evidence_contract": (
            "strict weekly past-to-future research OOS; labels resolved before "
            "each cutoff; research-selected, not untouched or promotion eligible"
        ),
        "windows": [asdict(window) for window in windows],
        "research_winner": winner,
        "latest_week": latest_week,
        "portfolio_ready_score_rows": int(len(portfolio)),
        "adapter_audit": adapter_audit,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return {
        "weekly_metrics": output / "weekly_metrics.csv",
        "period_metrics": output / "period_metrics.csv",
        "leaderboard": output / "shrinkage_leaderboard.csv",
        "portfolio_scores": output / "portfolio_ready_winner_scores.parquet",
        "manifest": output / "manifest.json",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--purge-hours", type=float, default=12.0)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=3)
    return parser


def main() -> None:
    paths = run(_parser().parse_args())
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))


if __name__ == "__main__":
    main()
