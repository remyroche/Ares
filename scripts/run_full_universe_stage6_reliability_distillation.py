#!/usr/bin/env python3
"""Stage 6: OOF future-path teacher distillation for the P30 reliability head.

The future-path fields are *teacher-only*.  Each training row receives a
teacher target from a model which was not fitted on that row.  The deployable
student is then fitted solely on the frozen causal base-state/context feature
contract and is the only model used to score development or OOS candidates.

This deliberately does not re-use the historical TP2/SL1 base-distillation
artifact: its shuffled folds and target/geometry do not match this stack.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_full_universe_round_b_meta_targets import (  # noqa: E402
    _attach_prequential_expected, _attach_prequential_population,
    _base_predictions, _state_features,
)
from scripts.run_full_universe_residual_meta import select  # noqa: E402

# ``event`` is the renamed TP3/SL2 first-touch outcome in the working frame.
FUTURE = ("t2_path_mfe_atr", "t2_path_mae_atr", "event", "t2_tp3_sl2_exit_minute")
TOPS = (.01, .05, .10, .20)


def _matrix(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return frame.loc[:, columns].replace([np.inf, -np.inf], np.nan).fillna(0.).to_numpy("float32")


def _teacher_classifier() -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary", n_estimators=140, learning_rate=.05, num_leaves=20,
        min_child_samples=300, colsample_bytree=.9, subsample=.8, reg_lambda=12.,
        random_state=20260808, n_jobs=1, verbosity=-1,
    )


def _teacher_quantile(alpha: float) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="quantile", alpha=alpha, n_estimators=140, learning_rate=.05,
        num_leaves=20, min_child_samples=300, colsample_bytree=.9, subsample=.8,
        reg_lambda=12., random_state=20260808 + int(alpha * 100), n_jobs=1, verbosity=-1,
    )


def _student() -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="huber", alpha=.9, n_estimators=180, learning_rate=.05,
        num_leaves=24, min_child_samples=400, colsample_bytree=.8, subsample=.8,
        reg_lambda=10., random_state=20260808, n_jobs=1, verbosity=-1,
    )


def _chronological_oof_teacher(train: pd.DataFrame, *, folds: int) -> tuple[pd.DataFrame, list[dict]]:
    """Cross-fit every teacher output; no row is predicted in its fit fold."""
    x = _matrix(train.assign(side_is_long=train.side_name.eq("long").astype(float)), list(FUTURE) + ["side_is_long"])
    net = train.net_bps.to_numpy(float)
    # Consecutive time blocks preserve contemporaneous candidates together.
    day = train["__ts__"].dt.floor("D")
    days = np.array(sorted(day.unique()))
    day_blocks = np.array_split(days, folds)
    out = pd.DataFrame(index=train.index, data={
        "teacher_p_net_gt_0": np.nan, "teacher_p_net_gt_25": np.nan,
        "teacher_p_net_gt_50": np.nan, "teacher_net_q10_bps": np.nan,
        "teacher_net_q50_bps": np.nan, "teacher_net_q90_bps": np.nan,
        "teacher_oof": False,
    })
    lineage: list[dict] = []
    for k, hold_days in enumerate(day_blocks):
        hold = day.isin(hold_days).to_numpy()
        fit = ~hold
        if int(fit.sum()) < 1000 or int(hold.sum()) < 100:
            raise ValueError("insufficient fold support for OOF future teacher")
        for threshold, name in ((0., "teacher_p_net_gt_0"), (25., "teacher_p_net_gt_25"), (50., "teacher_p_net_gt_50")):
            out.loc[train.index[hold], name] = _teacher_classifier().fit(x[fit], net[fit] > threshold).predict_proba(x[hold])[:, 1]
        for quantile, name in ((.10, "teacher_net_q10_bps"), (.50, "teacher_net_q50_bps"), (.90, "teacher_net_q90_bps")):
            out.loc[train.index[hold], name] = _teacher_quantile(quantile).fit(x[fit], net[fit]).predict(x[hold])
        out.loc[train.index[hold], "teacher_oof"] = True
        lineage.append({"fold": k, "fit_rows": int(fit.sum()), "holdout_rows": int(hold.sum()),
                        "holdout_start": str(pd.Timestamp(hold_days[0])), "holdout_end": str(pd.Timestamp(hold_days[-1])),
                        "row_excluded_from_its_teacher_fit": True})
    if not out.teacher_oof.all() or out.isna().any().any():
        raise ValueError("incomplete OOF teacher output")
    return out, lineage


def _metrics(frame: pd.DataFrame, score: np.ndarray) -> list[dict]:
    ranked = frame.assign(_score=score).sort_values(["_score", "candidate_id"], ascending=[False, True], kind="mergesort")
    rows = []
    for fraction in TOPS:
        z = ranked.head(int(np.ceil(len(ranked) * fraction)))
        rows.append({"top_fraction": fraction, "n": int(len(z)), "gross_bps": float(z.gross_bps.mean()),
                     "net_bps": float(z.net_bps.mean()), "long_n": int(z.side_name.eq("long").sum()),
                     "short_n": int(z.side_name.eq("short").sum())})
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panel", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--base-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--train-start", default="2024-05-01")
    p.add_argument("--eval-start", default="2024-06-15")
    p.add_argument("--eval-end", default="2024-08-01")
    p.add_argument("--population", type=float, default=.30)
    p.add_argument("--folds", type=int, default=3)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    train_start, eval_start, eval_end = (pd.Timestamp(x, tz="UTC") for x in (a.train_start, a.eval_start, a.eval_end))
    context = json.loads(a.audit.read_text())["meta"]["coverage_ge_90pct"]
    required = ["candidate_id", "__ts__", "__label_available_at__", "side_name", "t4_tp3_sl2_net_bps", "t4_tp3_sl2_gross_bps", "t2_tp3_sl2_event", "t2_path_mfe_atr", "t2_path_mae_atr", "t2_tp3_sl2_exit_minute", *context]
    raw = pd.concat([pd.read_parquet(part, columns=list(dict.fromkeys(required))) for part in sorted((a.panel / "parts").glob("*.parquet"))], ignore_index=True)
    raw["__ts__"] = pd.to_datetime(raw["__ts__"], utc=True)
    raw["__label_available_at__"] = pd.to_datetime(raw["__label_available_at__"], utc=True)
    raw = raw.rename(columns={"t4_tp3_sl2_net_bps": "net_bps", "t4_tp3_sl2_gross_bps": "gross_bps", "t2_tp3_sl2_event": "event"})
    data = raw.merge(_base_predictions(a.base_root, "tp3_sl2"), on="candidate_id", validate="one_to_one")
    # The map can only use labels resolved before the outer evaluation starts.
    mapped = _attach_prequential_expected(data, pd.Timestamp("2024-04-15", tz="UTC"), eval_start)
    mapped = _attach_prequential_population(mapped, a.population, train_start, eval_start)
    train = mapped[mapped.__ts__.ge(train_start) & mapped.__ts__.lt(eval_start) & mapped.__label_available_at__.lt(eval_start) & mapped.high_base_eligible].copy()
    evaluation = mapped[mapped.__ts__.ge(eval_start) & mapped.__ts__.lt(eval_end)].copy()
    eligible = evaluation[evaluation.high_base_eligible].copy()
    if min(len(train), len(eligible)) < 1000:
        raise ValueError("insufficient P30 train/evaluation population")
    teacher, lineage = _chronological_oof_teacher(train, folds=a.folds)
    # Teacher target matches the selected robust P(net>25) reliability head.
    y = train.net_bps.gt(25.).to_numpy(float)
    chosen = select(train, context, y, n=30)
    x_train = _state_features(train, chosen)
    x_eval = _state_features(eligible, chosen)
    variants: list[dict] = []
    variant_frames: dict[int, pd.DataFrame] = {}
    predictions = evaluation[["candidate_id", "__ts__", "side_name", "gross_bps", "net_bps", "base_expected_net_bps", "base_expected_gross_bps", "base_payoff_mixture_sd_bps", "high_base_eligible", "high_base_cutoff"]].copy()
    teacher_auc = {name: float(roc_auc_score((train.net_bps > threshold).astype(int), teacher[name])) for threshold, name in ((0., "teacher_p_net_gt_0"), (25., "teacher_p_net_gt_25"), (50., "teacher_p_net_gt_50"))}
    for alpha in (.50, .75):
        target = alpha * y + (1. - alpha) * teacher.teacher_p_net_gt_25.to_numpy(float)
        model = _student().fit(x_train, target)
        probability = np.clip(model.predict(x_eval), 0., 1.)
        y_eval = eligible.net_bps.gt(25.).to_numpy(int)
        variants.append({"alpha": alpha, "target": "alpha*I(net>25)+(1-alpha)*OOF_teacher_P(net>25)",
                         "target_auc": float(roc_auc_score(y_eval, probability)), "target_brier": float(brier_score_loss(y_eval, probability)),
                         "cost_clear_auc": float(roc_auc_score(eligible.net_bps.gt(0).to_numpy(int), probability)),
                         "cost_clear_brier": float(brier_score_loss(eligible.net_bps.gt(0).to_numpy(int), probability)),
                         "eligible_only_metrics": _metrics(eligible, probability)})
        predictions[f"reliability_alpha_{int(alpha * 100)}"] = np.nan
        predictions.loc[eligible.index, f"reliability_alpha_{int(alpha * 100)}"] = probability
        # Drop-in compatibility with the existing value/reliability overlay
        # replay: exactly one probability on admitted rows, no implicit score
        # or veto elsewhere.
        replay = predictions[["candidate_id", "__ts__", "side_name", "gross_bps", "net_bps", "base_expected_net_bps", "base_expected_gross_bps", "base_payoff_mixture_sd_bps", "high_base_eligible", "high_base_cutoff"]].copy()
        replay["reliability_score"] = np.nan
        replay.loc[eligible.index, "reliability_score"] = probability
        variant_frames[int(alpha * 100)] = replay
    a.out.mkdir(parents=True)
    train[["candidate_id", "__ts__", "net_bps"]].join(teacher).to_parquet(a.out / "teacher_oof_predictions.parquet", index=False)
    predictions.to_parquet(a.out / "predictions.parquet", index=False)
    for alpha, replay in variant_frames.items():
        replay.to_parquet(a.out / f"predictions_alpha_{alpha}.parquet", index=False)
    manifest = {"schema": "full_universe_stage6_reliability_distillation_v1", "status": "COMPLETED",
                "geometry": "TP3_SL2", "reliability_target": "I(realised net > 25 bps)",
                "outer_train_window": [str(train_start), str(eval_start)], "outer_evaluation_window": [str(eval_start), str(eval_end)],
                "population": "causal B2 P30; no reliability score outside admission", "teacher_inputs": list(FUTURE) + ["side_is_long"],
                "teacher_outputs": ["P(net>0)", "P(net>25)", "P(net>50)", "net q10/q50/q90"],
                "teacher_oof_only": True, "teacher_lineage": lineage, "teacher_oof_auc": teacher_auc,
                "student_inputs": [*chosen, "frozen causal base-state fields from _state_features"],
                "student_never_receives_future_path_columns": True, "variants": variants}
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"rows": {"train": len(train), "eligible_eval": len(eligible)}, "teacher_oof_auc": teacher_auc, "variants": variants}, indent=2))


if __name__ == "__main__":
    main()
