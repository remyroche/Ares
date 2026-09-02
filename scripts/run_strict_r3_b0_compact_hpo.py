#!/usr/bin/env python3
"""Short strict-OOF HPO and fixed rescore for the frozen compact B0 contract."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

from run_strict_r3_b0_fulluniverse_mda import _evaluate
from run_strict_r3_b0_fulluniverse_screen import _read_window, _valid_held, _valid_train
from run_strict_r3_b0_replacement_ranker_screen import GAIN_SCHEDULES, SEED, _groups, _sample_queries, _utc
from run_strict_r3_routed_et_fulluniverse_screen import _selected_feature_matrix


def _exclusive(path: Path, payload: object) -> None:
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _params(trial: optuna.Trial, train_rows: int) -> dict[str, Any]:
    fraction = trial.suggest_float("min_data_fraction", .005, .03, log=True)
    return {
        "n_estimators": 2000, "learning_rate": trial.suggest_float("learning_rate", .02, .10, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 6), "num_leaves": trial.suggest_int("num_leaves", 15, 61),
        "min_child_samples": max(40, int(round(train_rows * fraction))),
        "subsample": trial.suggest_float("bagging_fraction", .70, .90), "subsample_freq": 1,
        "colsample_bytree": trial.suggest_float("feature_fraction", .70, .90),
        "reg_alpha": trial.suggest_float("lambda_l1", 1e-4, 5.0, log=True),
        "reg_lambda": trial.suggest_float("lambda_l2", .1, 30.0, log=True),
        "min_split_gain": trial.suggest_float("min_gain_to_split", 1e-4, .01, log=True),
        "lambdarank_truncation_level": trial.suggest_categorical("truncation", (8, 10, 12, 16)),
        "sigmoid": trial.suggest_float("sigmoid", .5, 1.5),
    }


def _selection(rows: list[dict[str, float]]) -> float:
    values = pd.DataFrame(rows)
    top10 = values.blend_top10_ev
    stable = .50 * top10.mean() + .20 * top10.median() + .15 * top10.quantile(.25) + .15 * top10.quantile(.10)
    # Top-10 stability remains primary.  The tail term rewards a helpful
    # precision improvement but cannot dominate a broad-ranking deterioration.
    return float(stable + .10 * (values.blend_top01_ev.mean() - top10.mean()))


def _prepare(args: argparse.Namespace, fields: list[str]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    target, valid = "policy_ordinal_base_grade", "policy_ordinal_base_valid"
    for held_text in args.held_months:
        held_month = _utc(held_text)
        reserve = held_month - pd.Timedelta(days=args.reserve_days)
        window = _read_window(args.feature_root, args.score_root, args.router_root, args.label_root, reserve - pd.DateOffset(months=args.train_months), held_month + pd.offsets.MonthBegin(1), target)
        train = _sample_queries(_valid_train(window.loc[window.__decision_ts__.lt(reserve)], valid, target, reserve), args.train_cap).sort_values(["__decision_ts__", "candidate_id"], kind="stable")
        held = _sample_queries(_valid_held(window.loc[window.__decision_ts__.ge(held_month)], valid), args.held_cap).sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
        selected = pd.concat([train, held], ignore_index=True)
        values = _selected_feature_matrix(args.feature_root, selected, fields)
        medians = np.nanmedian(values[:len(train)], axis=0).astype(np.float32)
        medians[~np.isfinite(medians)] = 0.0
        missing = ~np.isfinite(values)
        if missing.any():
            values[missing] = np.broadcast_to(medians, values.shape)[missing]
        output.append({
            "held_month": f"{held_month:%Y-%m}", "x_train": values[:len(train)],
            "y_train": pd.to_numeric(train[target], errors="coerce").to_numpy(np.int32), "group_train": _groups(train),
            "x_held": values[len(train):], "y_held": pd.to_numeric(held[target], errors="coerce").to_numpy(np.int32), "group_held": _groups(held),
            "held": held.loc[:, ["candidate_id", "__decision_ts__", "policy_net_bps", "efficiency_bps", "timing_bps"]].copy(),
        })
    return output


def _fit(prepared: list[dict[str, Any]], params: dict[str, Any], *, model_jobs: int) -> tuple[list[dict[str, object]], pd.DataFrame]:
    rows: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    for fold_index, item in enumerate(prepared):
        model = lgb.LGBMRanker(
            objective="lambdarank", metric="ndcg", label_gain=GAIN_SCHEDULES["g3_clipped_economic"], lambdarank_norm=True,
            random_state=SEED + fold_index, deterministic=True, force_col_wise=True, verbosity=-1, n_jobs=model_jobs, **params,
        )
        model.fit(item["x_train"], item["y_train"], group=item["group_train"], eval_set=[(item["x_held"], item["y_held"])], eval_group=[item["group_held"]], callbacks=[lgb.early_stopping(30, verbose=False)])
        score = model.predict(item["x_held"])
        result = _evaluate(item["held"], score)
        rows.append({"held_month": item["held_month"], "best_iteration": int(model.best_iteration_), **result})
        predictions.append(item["held"].assign(x_score=score))
    return rows, pd.concat(predictions, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--score-root", type=Path, required=True)
    parser.add_argument("--router-root", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--feature-contract", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--held-months", nargs="+", default=("2026-02-01", "2026-03-01", "2026-04-01"))
    parser.add_argument("--train-months", type=int, default=3)
    parser.add_argument("--reserve-days", type=int, default=28)
    parser.add_argument("--train-cap", type=int, default=60_000)
    parser.add_argument("--held-cap", type=int, default=15_000)
    parser.add_argument("--trials", type=int, default=28)
    parser.add_argument("--model-jobs", type=int, default=4)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    contract = json.loads(args.feature_contract.read_text())
    fields = list(contract["selected_features"])
    if not 1 <= len(fields) <= 120:
        raise AssertionError("compact HPO needs a frozen <=120 contract")
    args.out.mkdir(parents=True)
    prepared = _prepare(args, fields)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED, multivariate=True), pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1, interval_steps=1))
    trial_rows: list[dict[str, object]] = []

    def objective(trial: optuna.Trial) -> float:
        results: list[dict[str, float]] = []
        for fold_index, item in enumerate(prepared):
            params = _params(trial, len(item["x_train"]))
            model = lgb.LGBMRanker(objective="lambdarank", metric="ndcg", label_gain=GAIN_SCHEDULES["g3_clipped_economic"], lambdarank_norm=True, random_state=SEED + fold_index, deterministic=True, force_col_wise=True, verbosity=-1, n_jobs=args.model_jobs, **params)
            model.fit(item["x_train"], item["y_train"], group=item["group_train"], eval_set=[(item["x_held"], item["y_held"])], eval_group=[item["group_held"]], callbacks=[lgb.early_stopping(30, verbose=False)])
            result = _evaluate(item["held"], model.predict(item["x_held"]))
            results.append(result)
            trial.report(_selection(results), step=fold_index)
            if trial.should_prune():
                raise optuna.TrialPruned()
        value = _selection(results)
        trial_rows.append({"trial": trial.number, "state": "complete", "selection_score": value, **{key: float(np.mean([row[key] for row in results])) for key in results[0]}, **trial.params})
        return value

    study.optimize(objective, n_trials=args.trials, n_jobs=1, gc_after_trial=True)
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.PRUNED:
            trial_rows.append({"trial": trial.number, "state": "pruned", **trial.params})
    if not any(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials):
        raise RuntimeError("no complete HPO trials")
    best = study.best_trial
    best_params = _params(best, int(np.median([len(item["x_train"]) for item in prepared])))
    # The stored fraction is the portable source of min_child_samples; use the
    # fold-specific conversion for the fixed rescore below.
    rows: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    for fold_index, item in enumerate(prepared):
        params = _params(best, len(item["x_train"]))
        fold_rows, fold_pred = _fit([item], params, model_jobs=args.model_jobs)
        rows.extend(fold_rows); predictions.append(fold_pred)
    pd.DataFrame(trial_rows).sort_values("trial").to_parquet(args.out / "trials.parquet", index=False, compression="zstd")
    pd.DataFrame(rows).to_parquet(args.out / "best_fold_metrics.parquet", index=False, compression="zstd")
    pd.concat(predictions, ignore_index=True).to_parquet(args.out / "best_oof_predictions.parquet", index=False, compression="zstd")
    _exclusive(args.out / "run_manifest.json", {"schema": "strict_r3_b0_compact_hpo_v1", "scope": "offline B0 candidate only; live unchanged", "feature_contract": str(args.feature_contract), "features": fields, "feature_count": len(fields), "target": "policy_ordinal_base", "gain_schedule": "g3_clipped_economic", "objective": "lambdarank", "strict_oof": True, "query": "decision timestamp × long side", "hpo": {"trials": args.trials, "pruner": "MedianPruner(startup=8,warmup_fold=1)", "early_stopping_rounds": 30, "best_trial": best.number, "best_value": best.value, "best_params": best.params, "representative_params": best_params}})


if __name__ == "__main__":
    main()
