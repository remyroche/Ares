#!/usr/bin/env python3
"""Bounded strict-OOF local HPO for a frozen Router50 single-Base finalist.

The target, ranking objective, gain schedule, truncation, sigmoid, feature
contract, Router50 gate, and 28-day outer reserve are supplied and immutable.
Only tree geometry and regularisation vary.  Every trial is evaluated on
timestamp-balanced outer held months; early stopping uses only an inner,
chronological slice of the already eligible training rows.

This is an offline development stage.  It does not run R/U, MC1, a portfolio,
live scoring, or exchange I/O.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMRanker, early_stopping

import run_strict_r3_router_single_base_prescreen_v1 as base


SEED = 1729


@dataclass(frozen=True)
class Fold:
    month: str
    train: pd.DataFrame
    held: pd.DataFrame
    labels: np.ndarray
    fields: tuple[str, ...]


def _months(value: str) -> tuple[pd.Timestamp, ...]:
    months = tuple(pd.Timestamp(f"{token.strip()}-01", tz="UTC") for token in value.split(",") if token.strip())
    if not months or tuple(sorted(months)) != months:
        raise ValueError("held months must be a non-empty chronological comma-separated sequence")
    return months


def _groups(frame: pd.DataFrame) -> np.ndarray:
    return frame.groupby("__decision_ts__", sort=False).size().to_numpy(np.int32)


def _inner_split(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    queries = frame.loc[:, ["__decision_ts__"]].drop_duplicates().sort_values("__decision_ts__", kind="stable")
    count = len(queries)
    cut = max(1, min(count - 1, int(np.floor(.80 * count))))
    fit_ts = set(queries.iloc[:cut]["__decision_ts__"])
    fit = frame["__decision_ts__"].isin(fit_ts).to_numpy(bool)
    if not fit.any() or fit.all():
        raise AssertionError("inner chronological split lacks fit or validation queries")
    return fit, ~fit


def _metrics(scored: pd.DataFrame, held: pd.DataFrame) -> dict[str, float]:
    labels = held.loc[:, ["candidate_id", "policy_ordinal_valid", "policy_net_bps"]]
    return base._metrics(scored, labels)


def _summary(rows: list[dict[str, float]]) -> dict[str, float]:
    d = pd.DataFrame(rows)
    mean = d.mean(numeric_only=True)
    tip = float(.25 * mean.dtp1_bps + .25 * mean.dtp2_bps + .50 * mean.dtp5_bps)
    breadth = float(.30 * mean.er50_at20 + .25 * mean.recall50_at20 + .25 * mean.recall100_at20 + .20 * mean.er100_at20)
    stability = float(.50 * mean.q10_week_dtp5_bps + .50 * mean.q25_month_dtp5_bps)
    return {
        "dtp1_bps": float(mean.dtp1_bps), "dtp2_bps": float(mean.dtp2_bps), "dtp5_bps": float(mean.dtp5_bps),
        "dtp10_bps": float(mean.dtp10_bps), "dtp20_bps": float(mean.dtp20_bps),
        "tip_bps": tip, "breadth": breadth, "stability_bps": stability,
        "q10_week_dtp5_bps": float(mean.q10_week_dtp5_bps), "q25_month_dtp5_bps": float(mean.q25_month_dtp5_bps),
    }


def _fit_predict(fold: Fold, params: dict[str, Any], seed: int) -> pd.DataFrame:
    x_train, medians = base._numeric_matrix(fold.train, fold.fields)
    x_held, _ = base._numeric_matrix(fold.held, fold.fields, medians)
    fit, valid = _inner_split(fold.train)
    train_groups = _groups(fold.train)
    # Query blocks remain contiguous because every fold is sorted by timestamp
    # and candidate ID.  Build fit/valid group sizes rather than slicing an
    # arbitrary row matrix across a query boundary.
    query = fold.train["__decision_ts__"].to_numpy()
    change = np.r_[True, query[1:] != query[:-1]]
    starts = np.flatnonzero(change)
    ends = np.r_[starts[1:], len(query)]
    fit_groups: list[int] = []
    valid_groups: list[int] = []
    for start, end in zip(starts, ends, strict=True):
        block = fit[start:end]
        if block.all():
            fit_groups.append(int(end - start))
        elif (~block).all():
            valid_groups.append(int(end - start))
        else:
            raise AssertionError("inner split broke a query")
    if sum(fit_groups) != int(fit.sum()) or sum(valid_groups) != int(valid.sum()):
        raise AssertionError("inner groups mismatch")
    model = LGBMRanker(
        objective=params["objective"], metric="ndcg", n_estimators=2000,
        learning_rate=params["learning_rate"], max_depth=params["max_depth"],
        num_leaves=params["num_leaves"], min_child_samples=params["min_child_samples"],
        subsample=params["bagging_fraction"], subsample_freq=1,
        colsample_bytree=params["feature_fraction"], reg_alpha=params["lambda_l1"],
        reg_lambda=params["lambda_l2"], min_split_gain=params["min_gain_to_split"],
        random_state=seed, n_jobs=params["model_jobs"], deterministic=True,
        force_col_wise=True, verbosity=-1,
        **({"label_gain": params["label_gain"], "lambdarank_truncation_level": params["truncation"], "sigmoid": params["sigmoid"]}
           if params["objective"] == "lambdarank" else {}),
    )
    model.fit(
        x_train[fit], fold.labels[fit], group=fit_groups,
        eval_set=[(x_train[valid], fold.labels[valid])], eval_group=[valid_groups],
        callbacks=[early_stopping(30, verbose=False)],
    )
    output = fold.held.loc[:, list(base.IDENTITY)].copy()
    output["base_score"] = model.predict(x_held, num_iteration=model.best_iteration_).astype(np.float32)
    output["base_rank_ts"] = base._rank_desc(output, "base_score")
    return output


def _prepare(args: argparse.Namespace) -> tuple[list[Fold], tuple[str, ...]]:
    fields = base._load_f72_fields(args.selection_receipt)
    spec = base.TARGETS[args.target]
    folds: list[Fold] = []
    for month in _months(args.held_months):
        reserve = month - pd.Timedelta(days=args.reserve_days)
        end = month + pd.offsets.MonthBegin(1)
        window, _ = base._load_window(
            candidate_root=None, feature_root=tuple(args.feature_roots), label_root=args.label_root,
            router_root=args.router_root, start=reserve - pd.DateOffset(months=args.train_months), end=end, fields=fields,
        )
        valid = window[spec.valid_column].fillna(False).astype(bool)
        available = pd.to_datetime(window[spec.available_column], utc=True, errors="coerce")
        numeric = np.isfinite(pd.to_numeric(window[spec.value_column], errors="coerce"))
        train = window.loc[window.__decision_ts__.lt(reserve) & valid & numeric & available.lt(reserve)].copy()
        train = base._sample_complete_queries(train, args.train_cap).sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
        held = window.loc[window.__decision_ts__.ge(month) & window.__decision_ts__.lt(end)].copy()
        held = held.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
        if len(train) < 8_000 or train.__decision_ts__.nunique() < 40 or len(held) < 10_000:
            raise AssertionError(f"{month:%Y-%m}: insufficient strict Base HPO support")
        labels, _ = base._target_labels(train, held, spec)
        folds.append(Fold(f"{month:%Y-%m}", train, held, labels, fields))
    return folds, fields


def _default_params(args: argparse.Namespace, model_jobs: int) -> dict[str, Any]:
    return {
        "objective": args.objective, "label_gain": base.GAIN_SCHEDULES[args.gain_name],
        "truncation": args.truncation, "sigmoid": args.sigmoid,
        "learning_rate": .05, "max_depth": 4, "num_leaves": 15,
        "min_child_samples": 260, "feature_fraction": .80, "bagging_fraction": .80,
        "lambda_l1": .05, "lambda_l2": 8.0, "min_gain_to_split": .001,
        "model_jobs": model_jobs,
    }


def _suggest(trial: optuna.Trial, *, train_rows: int, frozen: dict[str, Any]) -> dict[str, Any]:
    depth = trial.suggest_int("max_depth", 3, 6)
    requested = trial.suggest_categorical("num_leaves", [7, 15, 31, 61])
    leaves = max(item for item in [7, 15, 31, 61] if item <= min(requested, 2 ** depth - 1))
    return {
        **frozen,
        "learning_rate": trial.suggest_float("learning_rate", .02, .10, log=True),
        "max_depth": depth, "num_leaves": leaves,
        "min_child_samples": max(40, int(round(train_rows * trial.suggest_float("min_data_fraction", .005, .03, log=True)))),
        "feature_fraction": trial.suggest_float("feature_fraction", .70, .90),
        "bagging_fraction": trial.suggest_float("bagging_fraction", .70, .90),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-4, 5.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", .1, 30.0, log=True),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 1e-4, .01, log=True),
    }


def _write_once(path: Path, value: object) -> None:
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, default=str)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-roots", required=True, help="comma-separated immutable causal roots")
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--router-root", type=Path, required=True)
    parser.add_argument("--selection-receipt", type=Path, required=True)
    parser.add_argument("--target", choices=tuple(base.TARGETS), required=True)
    parser.add_argument("--gain-name", choices=tuple(base.GAIN_SCHEDULES), required=True)
    parser.add_argument("--truncation", type=int, required=True)
    parser.add_argument("--sigmoid", type=float, default=1.0)
    parser.add_argument("--objective", choices=("lambdarank", "rank_xendcg"), required=True)
    parser.add_argument("--held-months", default="2025-11,2026-01,2026-03")
    parser.add_argument("--train-months", type=int, default=3)
    parser.add_argument("--reserve-days", type=int, default=28)
    parser.add_argument("--train-cap", type=int, default=60_000)
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--study-jobs", type=int, default=2)
    parser.add_argument("--model-jobs", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.feature_roots = tuple(Path(item.strip()).resolve() for item in args.feature_roots.split(",") if item.strip())
    args.label_root, args.router_root, args.selection_receipt, args.out = (args.label_root.resolve(), args.router_root.resolve(), args.selection_receipt.resolve(), args.out.resolve())
    if args.out.exists():
        raise FileExistsError(f"immutable output exists: {args.out}")
    args.out.mkdir(parents=True)
    folds, fields = _prepare(args)
    frozen = _default_params(args, args.model_jobs)
    control_rows = [_metrics(_fit_predict(fold, frozen, SEED + index), fold.held) for index, fold in enumerate(folds)]
    control = _summary(control_rows)
    rows: list[dict[str, object]] = []
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED, multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=1, interval_steps=1),
    )

    def objective(trial: optuna.Trial) -> float:
        params = _suggest(trial, train_rows=len(folds[0].train), frozen=frozen)
        metrics: list[dict[str, float]] = []
        for index, fold in enumerate(folds):
            metrics.append(_metrics(_fit_predict(fold, params, SEED + 10_000 * (trial.number + 1) + index), fold.held))
            partial = _summary(metrics)
            score = .50 * partial["tip_bps"] / control["tip_bps"] + .35 * partial["breadth"] / control["breadth"] + .15 * partial["stability_bps"] / control["stability_bps"]
            trial.report(float(score), index)
            if index >= 1 and trial.should_prune():
                raise optuna.TrialPruned()
        result = _summary(metrics)
        passes = (result["dtp1_bps"] >= .97 * control["dtp1_bps"] and result["dtp2_bps"] >= .98 * control["dtp2_bps"] and result["dtp5_bps"] >= .98 * control["dtp5_bps"] and result["q10_week_dtp5_bps"] >= control["q10_week_dtp5_bps"] and result["q25_month_dtp5_bps"] >= control["q25_month_dtp5_bps"])
        score = .50 * result["tip_bps"] / control["tip_bps"] + .35 * result["breadth"] / control["breadth"] + .15 * result["stability_bps"] / control["stability_bps"]
        # ``num_leaves`` is constrained by the sampled depth before fitting.
        # Keep the raw Optuna suggestion for audit, but publish the *effective*
        # value that reached LightGBM so a frozen receipt can be replayed
        # exactly by the full OOF producer.
        result.update({
            "trial": trial.number, "state": "complete", "passes_base_gates": passes,
            "selection_score": float(score), **trial.params,
            "requested_num_leaves": trial.params["num_leaves"],
            "num_leaves": params["num_leaves"],
            "min_child_samples_reference": params["min_child_samples"],
        })
        rows.append(result)
        return float(score if passes else score - 1_000.0)

    study.optimize(objective, n_trials=args.trials, n_jobs=args.study_jobs, gc_after_trial=True)
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.PRUNED:
            rows.append({"trial": trial.number, "state": "pruned", **trial.params})
    result = pd.DataFrame(rows).sort_values("trial", kind="stable")
    result.to_parquet(args.out / "trials.parquet", index=False, compression="zstd")
    complete = result.loc[(result.state == "complete") & result.passes_base_gates.fillna(False)].sort_values(["selection_score", "trial"], ascending=[False, True], kind="stable")
    if complete.empty:
        raise RuntimeError("no HPO trial passed frozen Base gates")
    winner = complete.iloc[0].to_dict()
    _write_once(args.out / "run_manifest.json", {
        "schema": "strict_r3_router_single_base_local_hpo_v1", "scope": "offline development HPO; no R/U, MC1, portfolio, live, or exchange mutation",
        "target": args.target, "objective": args.objective, "gain_name": args.gain_name, "truncation": args.truncation, "sigmoid": args.sigmoid,
        "query": "decision timestamp x long side", "router": "exact frozen Router50 identities only", "fields": list(fields),
        "train_contract": {"outer_train_months": args.train_months, "outer_reserve_days": args.reserve_days, "query_safe_cap": args.train_cap, "inner_validation": "latest 20% eligible training queries", "early_stopping_rounds": 30},
        "held_months": [fold.month for fold in folds], "control_metrics": control,
        "selection": "0.50 TIP + 0.35 BREADTH + 0.15 STABILITY, all normalised to frozen common-parameter control; original tip and stability gates bind",
        "pruner": "MedianPruner(startup=4,warmup_folds=1)", "trials": int(args.trials), "winner": winner,
    })


if __name__ == "__main__":
    main()
