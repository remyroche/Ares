#!/usr/bin/env python3
"""Causal transition-context overlay on the frozen direct-EV global-top10 book.

Transition probabilities are never treated as a rank or an admission score.
They enter only through continuous state-risk interactions, uncertainty terms,
and an OOF abstention-risk estimate.  The output remains a direct-EV score in
the original units and is selected once over the pooled global population.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_execution_ev_recent_residual_shrinkage_ablation import policy_global_topk_mask


SCHEMA = "execution_ev_transition_context_overlay_v1"
SCORE = "catboost__residual__without_hpo__all_features__recent_ev_catboost_predicted_archetype"
TARGET = "execution_net_ev_12h"
DECISION = "execution_decision_utc"
RESOLUTION = "execution_label_end_utc"
SIDE = "side_name"
HORIZONS = (1, 3, 6, 12)
DEFAULT_SCORES = ROOT / "data_perp/artifacts/execution_ev_context_clean_exact_recent_correction_forward_july19_20260726_v2/mapped_oof_and_forward.parquet"
DEFAULT_TRANSITION = ROOT / "data_perp/artifacts/execution_ev_raw_market_state_transition_heads_20260726_v2/strict_weekly_oof_transition_predictions.parquet"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/execution_ev_transition_context_overlay_20260726_v1"


def _utc(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, utc=True, errors="raise")


def build_transition_context_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Continuous transition context; deliberately no ranks, deciles or IDs."""

    out = pd.DataFrame(index=frame.index)
    direct_ev = pd.to_numeric(frame[SCORE], errors="coerce")
    probabilities: list[pd.Series] = []
    for horizon in HORIZONS:
        column = f"transition_probability_h{horizon}"
        probability = pd.to_numeric(frame[column], errors="coerce").clip(0.0, 1.0)
        probabilities.append(probability)
        uncertainty = probability * (1.0 - probability)
        out[f"transition_p_h{horizon}"] = probability
        out[f"transition_uncertainty_h{horizon}"] = uncertainty
        out[f"direct_ev_x_transition_h{horizon}"] = direct_ev * probability
        out[f"abs_direct_ev_x_uncertainty_h{horizon}"] = direct_ev.abs() * uncertainty
    probability_frame = pd.concat(probabilities, axis=1)
    out["transition_probability_mean"] = probability_frame.mean(axis=1)
    out["transition_probability_max"] = probability_frame.max(axis=1)
    out["transition_probability_range"] = probability_frame.max(axis=1) - probability_frame.min(axis=1)
    out["transition_uncertainty_mean"] = pd.concat(
        [out[f"transition_uncertainty_h{horizon}"] for horizon in HORIZONS], axis=1
    ).mean(axis=1)
    out["direct_ev_x_transition_mean"] = direct_ev * out["transition_probability_mean"]
    return out.replace([np.inf, -np.inf], np.nan)


def load_strict_oof_frame(scores_path: Path, transition_path: Path) -> pd.DataFrame:
    scores = pd.read_parquet(scores_path)
    transitions = pd.read_parquet(transition_path)
    required_scores = {"candidate_id", SIDE, DECISION, RESOLUTION, TARGET, SCORE, "evaluation_origin"}
    if missing := required_scores.difference(scores.columns):
        raise ValueError(f"score file missing {sorted(missing)}")
    if not scores[SCORE].notna().all():
        raise ValueError("frozen direct-EV score must be populated")
    transitions = transitions.loc[
        transitions["feature_set"].eq("combined") & transitions["horizon_hours"].isin(HORIZONS)
    ].copy()
    if transitions.duplicated(["candidate_id", "horizon_hours"]).any():
        raise ValueError("transition OOF rows must be unique per candidate/horizon")
    wide = transitions.pivot(index="candidate_id", columns="horizon_hours", values="oof_transition_probability")
    wide = wide.rename(columns={horizon: f"transition_probability_h{horizon}" for horizon in HORIZONS}).reset_index()
    if set(f"transition_probability_h{horizon}" for horizon in HORIZONS).difference(wide.columns):
        raise ValueError("strict OOF transition coverage lacks a required horizon")
    work = scores.merge(wide, on="candidate_id", how="inner", validate="one_to_one")
    work[DECISION] = _utc(work[DECISION])
    work[RESOLUTION] = _utc(work[RESOLUTION])
    # The scores already record outer OOF or frozen-final forward OOS.  Do not
    # mix in training-only/promotion-ineligible score rows.
    is_eval = work.get(f"{SCORE}__is_evaluation", pd.Series(True, index=work.index)).astype(bool)
    work = work.loc[is_eval].copy()
    work = work.sort_values([DECISION, "candidate_id"], kind="stable").reset_index(drop=True)
    return work


def _fit_side_overlay(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    features: list[str],
    *,
    min_rows: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return residual correction and abstention probability with hard zero fallback."""

    correction = np.zeros(len(evaluation), dtype=np.float32)
    abstention = np.zeros(len(evaluation), dtype=np.float32)
    report: dict[str, Any] = {}
    for side_index, side in enumerate(("long", "short")):
        fit = train.loc[train[SIDE].eq(side)].copy()
        eval_mask = evaluation[SIDE].eq(side).to_numpy()
        if not eval_mask.any():
            continue
        if len(fit) < int(min_rows):
            report[side] = {"status": "zero_fallback", "reason": "insufficient_resolved_prior_rows", "train_rows": int(len(fit))}
            continue
        x_train = fit.loc[:, features].apply(pd.to_numeric, errors="coerce")
        x_eval = evaluation.loc[eval_mask, features].apply(pd.to_numeric, errors="coerce")
        medians = x_train.median(axis=0).fillna(0.0)
        x_train = x_train.fillna(medians).replace([np.inf, -np.inf], 0.0)
        x_eval = x_eval.fillna(medians).replace([np.inf, -np.inf], 0.0)
        residual = fit[TARGET].to_numpy(dtype=float) - fit[SCORE].to_numpy(dtype=float)
        regressor = HistGradientBoostingRegressor(
            learning_rate=0.05, max_iter=48, max_leaf_nodes=8, min_samples_leaf=80,
            l2_regularization=3.0, random_state=random_state + side_index,
        ).fit(x_train, residual)
        # A separate risk model is an abstention *input* to the EV correction,
        # not a direct rank replacement.  Its target is fully resolved prior EV.
        bad = fit[TARGET].le(0.0).astype(np.int8).to_numpy()
        if bad.min() == bad.max():
            risk = np.full(eval_mask.sum(), float(bad.mean()), dtype=float)
        else:
            classifier = HistGradientBoostingClassifier(
                learning_rate=0.05, max_iter=48, max_leaf_nodes=8, min_samples_leaf=80,
                l2_regularization=3.0, random_state=random_state + 100 + side_index,
            ).fit(x_train, bad)
            risk = classifier.predict_proba(x_eval)[:, 1]
        raw_correction = np.asarray(regressor.predict(x_eval), dtype=float)
        # Train-derived cost scale: how much adverse residual remains per bad
        # trade. It cannot use evaluation EV and is zero-safe if unavailable.
        adverse_residual = np.maximum(-residual[bad.astype(bool)], 0.0)
        risk_scale = float(np.median(adverse_residual)) if len(adverse_residual) else 0.0
        # Bounded correction prevents a learned context head from replacing the
        # frozen direct-EV model.  Risk is an explicit continuous abstention
        # penalty, not a hard selector.
        applied = np.clip(raw_correction - risk_scale * risk, -0.01, 0.01)
        correction[np.flatnonzero(eval_mask)] = applied.astype(np.float32)
        abstention[np.flatnonzero(eval_mask)] = np.asarray(risk, dtype=np.float32)
        report[side] = {
            "status": "fit_on_resolved_prior_oof_rows",
            "train_rows": int(len(fit)),
            "bad_rate": float(bad.mean()),
            "risk_scale": risk_scale,
            "mean_abs_applied_correction": float(np.abs(applied).mean()),
        }
    return correction, abstention, report


def _metrics(frame: pd.DataFrame, selected: pd.Series) -> dict[str, Any]:
    trade = frame.loc[selected]
    return {
        "eligible_rows": int(len(frame)),
        "selected_rows": int(selected.sum()),
        "mean_net_ev": float(trade[TARGET].mean()) if len(trade) else None,
        "mean_net_ev_bps": float(10000.0 * trade[TARGET].mean()) if len(trade) else None,
        "positive_rate": float(trade[TARGET].gt(0.0).mean()) if len(trade) else None,
        "sum_net_ev": float(trade[TARGET].sum()) if len(trade) else 0.0,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    frame = load_strict_oof_frame(args.scores, args.transitions)
    features = build_transition_context_features(frame)
    feature_columns = list(features.columns)
    frame = pd.concat([frame, features], axis=1)
    weeks = pd.date_range(frame[DECISION].min().floor("D"), frame[DECISION].max().ceil("D"), freq="7D")
    parts: list[pd.DataFrame] = []
    reports: dict[str, Any] = {}
    for week_start in weeks:
        week_end = min(week_start + pd.Timedelta(days=7), frame[DECISION].max() + pd.Timedelta("1ns"))
        evaluation = frame.loc[frame[DECISION].ge(week_start) & frame[DECISION].lt(week_end)].copy()
        if evaluation.empty:
            continue
        train = frame.loc[frame[RESOLUTION].lt(week_start)].copy()
        correction, abstention, report = _fit_side_overlay(
            train, evaluation, feature_columns, min_rows=args.min_prior_rows, random_state=args.random_state
        )
        # Exact zero fallback is observable in these columns, rather than a
        # merely descriptive report.
        evaluation["transition_context_correction"] = correction
        evaluation["transition_abstention_risk"] = abstention
        evaluation["frozen_direct_ev_score"] = evaluation[SCORE].to_numpy(dtype=float)
        evaluation["transition_context_score"] = evaluation["frozen_direct_ev_score"] + correction
        evaluation["week_start"] = week_start
        parts.append(evaluation)
        reports[str(week_start.date())] = {
            "evaluation_rows": int(len(evaluation)),
            "resolved_prior_rows": int(len(train)),
            "max_prior_resolution": str(train[RESOLUTION].max()) if len(train) else None,
            "overlay": report,
        }
    predictions = pd.concat(parts, ignore_index=True)
    metric_rows: list[dict[str, Any]] = []
    for arm, score_column in (("frozen_direct_ev", "frozen_direct_ev_score"), ("transition_context_overlay", "transition_context_score")):
        for scope, local in [("pooled_global_top10", predictions), *[(f"week_{pd.Timestamp(w).date()}", g) for w, g in predictions.groupby("week_start", sort=True)]]:
            selected = policy_global_topk_mask(local, score_column, args.top_k_fraction)
            metric_rows.append({"arm": arm, "scope": scope, **_metrics(local, selected)})
    metrics = pd.DataFrame(metric_rows)
    args.output_dir.mkdir(parents=True)
    predictions.to_parquet(args.output_dir / "strict_oof_transition_context_predictions.parquet", index=False)
    metrics.to_csv(args.output_dir / "global_top10_metrics.csv", index=False)
    report = {
        "schema": SCHEMA,
        "contract": {
            "base": "frozen direct EV mapped score; one pooled global top10 after score correction",
            "transition": "strict weekly OOF combined probabilities at 1/3/6/12h only",
            "context": "continuous transition probabilities, their uncertainty, and direct-EV interactions; no rank/decile/timestamp-relative score field",
            "abstention": "prior-resolved side-local P(netEV<=0) used only as a bounded direct-EV risk penalty",
            "fallback": "exact zero correction and zero abstention risk for first week or under-supported side",
            "training": "side-local train labels resolve before evaluation week starts; no calendar/regime weights",
        },
        "sources": {"scores": str(args.scores), "transition_oof": str(args.transitions)},
        "rows": int(len(predictions)),
        "features": feature_columns,
        "weekly_reports": reports,
        "outputs": {
            "predictions": str(args.output_dir / "strict_oof_transition_context_predictions.parquet"),
            "metrics": str(args.output_dir / "global_top10_metrics.csv"),
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    result.add_argument("--transitions", type=Path, default=DEFAULT_TRANSITION)
    result.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--top-k-fraction", type=float, default=0.10)
    result.add_argument("--min-prior-rows", type=int, default=1500)
    result.add_argument("--random-state", type=int, default=20260726)
    return result


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args()), indent=2, sort_keys=True, default=str))
