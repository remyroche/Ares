#!/usr/bin/env python3
"""Short, strict-OOF HPO for the selected candidate B0 replacement.

This is intentionally a development-only HPO.  It preserves the frozen
top-50 router, three-month/reserve ledger rule, candidate-ID identities and
120-field base contract.  Its objective is timestamp-local Top-10 precision
of E+T+X, evaluated against policy-net outcomes, with explicit stability and
tail-economic terms.  It never writes or reads a live model bundle.
"""

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

from run_strict_r3_b0_replacement_ranker_screen import (
    GAIN_SCHEDULES,
    SEED,
    TARGETS,
    _features,
    _groups,
    _metrics,
    _rank,
    _read_window,
    _route,
    _sample_queries,
    _utc,
)


def _precision_stability(frame: pd.DataFrame, score: str, fraction: float) -> dict[str, float]:
    work = frame.loc[:, ["__decision_ts__", "candidate_id", "policy_net_bps", score]].copy()
    work = work.sort_values(["__decision_ts__", score, "candidate_id"], ascending=[True, False, True], kind="stable")
    ordinal = work.groupby("__decision_ts__", sort=False).cumcount() + 1
    count = work.groupby("__decision_ts__", sort=False).candidate_id.transform("size")
    selected = work.loc[ordinal.le(np.ceil(count.to_numpy(float) * fraction))].copy()
    per_timestamp = selected.assign(win=selected.policy_net_bps.gt(50)).groupby("__decision_ts__", sort=False).win.mean()
    week = per_timestamp.groupby(per_timestamp.index.isocalendar().year.astype(str) + "-" + per_timestamp.index.isocalendar().week.astype(str)).mean()
    month = per_timestamp.groupby(per_timestamp.index.tz_localize(None).to_period("M")).mean()
    return {
        "mean": float(per_timestamp.mean()),
        "q10_week": float(week.quantile(.10)),
        "q25_month": float(month.quantile(.25)),
    }


def _evaluate(frame: pd.DataFrame) -> dict[str, float]:
    baseline = _metrics(frame, "et_rank")
    blend = _metrics(frame, "etx_rank")
    p_et_10 = _precision_stability(frame, "et_rank", .10)
    p_x_10 = _precision_stability(frame, "etx_rank", .10)
    p_et_01 = _precision_stability(frame, "et_rank", .01)
    p_x_01 = _precision_stability(frame, "etx_rank", .01)
    values = {
        "delta_p10_mean": p_x_10["mean"] - p_et_10["mean"],
        "delta_p10_q10_week": p_x_10["q10_week"] - p_et_10["q10_week"],
        "delta_p10_q25_month": p_x_10["q25_month"] - p_et_10["q25_month"],
        "delta_p01_mean": p_x_01["mean"] - p_et_01["mean"],
        "delta_ev_top01": blend["top01_ev"] - baseline["top01_ev"],
        "delta_ev_top05": blend["top05_ev"] - baseline["top05_ev"],
        "delta_ev_top10": blend["top10_ev"] - baseline["top10_ev"],
        "blend_top01_ev": blend["top01_ev"],
        "blend_top10_ev": blend["top10_ev"],
    }
    # Primary: timestamp-local precision and its lower temporal quantiles.
    # Secondary: economics, with slightly greater emphasis at Top-1.
    values["selection_score"] = float(
        100.0 * (
            .45 * values["delta_p10_mean"]
            + .20 * values["delta_p10_q10_week"]
            + .15 * values["delta_p10_q25_month"]
            + .20 * values["delta_p01_mean"]
        )
        + .035 * values["delta_ev_top01"]
        + .020 * values["delta_ev_top05"]
        + .015 * values["delta_ev_top10"]
    )
    return values


def _suggest(trial: optuna.Trial, train_rows: int) -> dict[str, Any]:
    min_fraction = trial.suggest_float("min_data_fraction", .005, .03, log=True)
    return {
        "n_estimators": 2000,
        "learning_rate": trial.suggest_float("learning_rate", .02, .10, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 6),
        "num_leaves": trial.suggest_int("num_leaves", 15, 61),
        "min_child_samples": max(40, int(round(train_rows * min_fraction))),
        "subsample": trial.suggest_float("bagging_fraction", .70, .90),
        "colsample_bytree": trial.suggest_float("feature_fraction", .70, .90),
        "reg_alpha": trial.suggest_float("lambda_l1", 1e-4, 5.0, log=True),
        "reg_lambda": trial.suggest_float("lambda_l2", .1, 30.0, log=True),
        "min_split_gain": trial.suggest_float("min_gain_to_split", 1e-4, .01, log=True),
        "lambdarank_truncation_level": trial.suggest_categorical("truncation", (8, 10, 12, 16)),
        "sigmoid": trial.suggest_float("sigmoid", .5, 1.5),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--router-root", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--target", choices=tuple(TARGETS), default="policy_ordinal_base")
    parser.add_argument("--gain-schedule", choices=tuple(GAIN_SCHEDULES), default="g3_clipped_economic")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--held-months", nargs="+", default=("2026-02-01", "2026-03-01", "2026-04-01"))
    parser.add_argument("--train-months", type=int, default=3)
    parser.add_argument("--reserve-days", type=int, default=28)
    parser.add_argument("--train-cap", type=int, default=35000)
    parser.add_argument("--trials", type=int, default=32)
    parser.add_argument("--study-jobs", type=int, default=2)
    parser.add_argument("--model-jobs", type=int, default=2)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    target = TARGETS[args.target]
    valid = target.replace("_grade", "_valid")
    fields = _features(args.source_root)
    prepared: list[dict[str, Any]] = []
    for held_text in args.held_months:
        held_month = _utc(held_text)
        reserve = held_month - pd.Timedelta(days=args.reserve_days)
        window = _read_window(
            args.source_root, args.router_root, args.label_root,
            reserve - pd.DateOffset(months=args.train_months),
            held_month + pd.offsets.MonthBegin(1), fields, target,
        )
        train = window.loc[
            window.router_selected & window[valid].fillna(False).astype(bool)
            & window.label_available_ts.lt(reserve)
            & np.isfinite(pd.to_numeric(window[target], errors="coerce"))
        ].copy()
        train = _sample_queries(train, args.train_cap).sort_values(["__decision_ts__", "candidate_id"], kind="stable")
        held = window.loc[
            window.__decision_ts__.ge(held_month) & window.router_selected
            & window[valid].fillna(False).astype(bool)
            & np.isfinite(pd.to_numeric(window.policy_net_bps, errors="coerce"))
        ].copy().sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
        if len(train) < 8000 or len(held) < 2000:
            raise AssertionError(f"{held_month:%Y-%m}: insufficient strict query support")
        medians = train.loc[:, fields].apply(pd.to_numeric, errors="coerce").median().fillna(0.0)
        prepared.append({
            "month": f"{held_month:%Y-%m}",
            "x_train": train.loc[:, fields].apply(pd.to_numeric, errors="coerce").fillna(medians).to_numpy(np.float32),
            "y_train": pd.to_numeric(train[target], errors="coerce").to_numpy(np.int32),
            "group_train": _groups(train),
            "x_held": held.loc[:, fields].apply(pd.to_numeric, errors="coerce").fillna(medians).to_numpy(np.float32),
            "y_held": pd.to_numeric(held[target], errors="coerce").to_numpy(np.int32),
            "group_held": _groups(held),
            "held": held.loc[:, ["candidate_id", "__decision_ts__", "policy_net_bps", "efficiency_bps", "timing_bps"]].copy(),
        })

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=SEED, multivariate=True)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1, interval_steps=1)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    trial_rows: list[dict[str, Any]] = []

    def objective(trial: optuna.Trial) -> float:
        fold_rows: list[dict[str, float]] = []
        for fold_index, item in enumerate(prepared):
            params = _suggest(trial, len(item["x_train"]))
            model = lgb.LGBMRanker(
                objective="lambdarank", metric="ndcg", label_gain=GAIN_SCHEDULES[args.gain_schedule],
                lambdarank_norm=True, subsample_freq=1, random_state=SEED + fold_index,
                deterministic=True, force_col_wise=True, verbosity=-1, n_jobs=args.model_jobs,
                **params,
            )
            model.fit(
                item["x_train"], item["y_train"], group=item["group_train"],
                eval_set=[(item["x_held"], item["y_held"])], eval_group=[item["group_held"]],
                callbacks=[lgb.early_stopping(30, verbose=False)],
            )
            held = item["held"].copy()
            held["x_rank"] = _rank(held.assign(x_score=model.predict(item["x_held"])), "x_score")
            held["e_rank"] = _rank(held, "efficiency_bps")
            held["t_rank"] = _rank(held, "timing_bps")
            held["et_rank"] = .5 * (held.e_rank + held.t_rank)
            held["etx_rank"] = (held.e_rank + held.t_rank + held.x_rank) / 3.0
            measure = _evaluate(held)
            fold_rows.append(measure)
            partial = float(np.mean([row["selection_score"] for row in fold_rows]))
            trial.report(partial, step=fold_index)
            if trial.should_prune():
                raise optuna.TrialPruned()
        aggregate = {key: float(np.mean([row[key] for row in fold_rows])) for key in fold_rows[0]}
        trial_rows.append({"trial": trial.number, "state": "complete", **aggregate, **trial.params})
        return aggregate["selection_score"]

    study.optimize(objective, n_trials=args.trials, n_jobs=args.study_jobs, gc_after_trial=True)
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.PRUNED:
            trial_rows.append({"trial": trial.number, "state": "pruned", **trial.params})
    complete = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if not complete:
        raise RuntimeError("all HPO trials pruned or failed")
    best = study.best_trial
    pd.DataFrame(trial_rows).sort_values("trial").to_parquet(args.out / "trials.parquet", index=False, compression="zstd")
    payload = {
        "schema": "strict_r3_b0_replacement_hpo_v1", "target": args.target,
        "target_column": target, "valid_column": valid, "gain_schedule": args.gain_schedule,
        "label_gain": GAIN_SCHEDULES[args.gain_schedule], "objective": "lambdarank",
        "query": "decision timestamp × long side", "router": "frozen top50",
        "train_contract": {"train_months": args.train_months, "reserve_days": args.reserve_days, "query_safe_cap": args.train_cap},
        "held_months": list(args.held_months), "feature_count": len(fields), "features": fields,
        "best_trial": best.number, "best_value": best.value, "best_params": best.params,
        "pruner": "MedianPruner(startup=8,warmup_folds=1)", "early_stopping_rounds": 30,
        "scope": "offline development HPO; does not modify E/T/B0 live contracts",
    }
    fd = os.open(args.out / "run_manifest.json", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


if __name__ == "__main__":
    main()
