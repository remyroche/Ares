#!/usr/bin/env python3
"""Cross-year, strict-OOF HPO for one retained direct B/E/T base head.

Research-only.  This tool never reads a target into the feature matrix and
never modifies inference, consensus, MC1, admission, portfolio or execution
artifacts.  It selects *parameters* for an already frozen feature contract.
All reported economics are timestamp-local policy-net outcomes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

from run_strict_r3_base_stability_selector_v2 import (
    GAIN_G3, HEADS, IDENTITY, SCORE_FIELDS, SEED, _apply_base_override,
    _enhanced_score, _groups, _held_rows, _impute, _materialize,
    _next_month, _read_policy, _sample_whole_queries, _timestamp_metrics,
    _train_rows, _utc, _window,
)


def _exclusive(path: Path, payload: object) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _feature_contract(path: Path) -> list[str]:
    source = json.loads(path.read_text())
    fields = source.get("selected_features", source.get("causal_features"))
    if not isinstance(fields, list) or not fields or len(fields) > 120:
        raise AssertionError(f"{path}: expected a non-empty <=120 field contract")
    if len(set(fields)) != len(fields):
        raise AssertionError(f"{path}: duplicate features")
    return [str(value) for value in fields]


def _split_early(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological final-20%-of-train early-stop reserve; never held data."""
    timestamps = frame.__decision_ts__.drop_duplicates().sort_values().to_numpy()
    cut = max(1, int(np.floor(.80 * len(timestamps))))
    if cut >= len(timestamps):
        raise AssertionError("insufficient train timestamps for early stopping")
    boundary = timestamps[cut]
    fit = frame.loc[frame.__decision_ts__.lt(boundary)].copy()
    valid = frame.loc[frame.__decision_ts__.ge(boundary)].copy()
    if fit.empty or valid.empty:
        raise AssertionError("empty chronological early-stop split")
    return fit, valid


def _objective_value(metrics: dict[str, float]) -> float:
    """Timestamp-local tail quality, deliberately more tail than Top-10 led."""
    return float(
        .27 * metrics["ts_top01_ev"]
        + .23 * metrics["ts_top02_ev"]
        + .20 * metrics["ts_top05_ev"]
        + .10 * metrics["ts_top10_ev"]
        + .08 * metrics["monthly_median_top10"]
        + .05 * metrics["monthly_q25_top05"]
        + .07 * metrics["weekly_q10_top02"]
    )


def _summarize(frame: pd.DataFrame) -> dict[str, float]:
    row: dict[str, float] = {}
    keys = [
        "ts_top01_ev", "ts_top02_ev", "ts_top05_ev", "ts_top10_ev",
        "monthly_median_top10", "monthly_q25_top05", "weekly_q10_top02",
        "weekly_q10_top10", "worst_month_top10", "positive_month_fraction_top10",
        "fixed_k1_ev", "fixed_k2_ev", "fixed_k3_ev", "fixed_k5_ev", "fixed_k10_ev",
        "stable_top10_5_2",
    ]
    for name, grouped in frame.groupby("contract", sort=False):
        values = {key: float(grouped[key].mean()) for key in keys}
        values["hpo_tail_stability"] = _objective_value(values)
        row.update({f"{name}_{key}": value for key, value in values.items()})
    return row


def _params(trial: optuna.Trial, *, head: str, rows: int) -> dict[str, Any]:
    fraction = trial.suggest_float("min_data_fraction", .005, .035, log=True)
    result: dict[str, Any] = {
        "n_estimators": 2000,
        "learning_rate": trial.suggest_float("learning_rate", .015, .10, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 5),
        "num_leaves": trial.suggest_int("num_leaves", 7, 47),
        "min_child_samples": max(40, int(round(fraction * rows))),
        "subsample": trial.suggest_float("bagging_fraction", .65, .92),
        "subsample_freq": 1,
        "colsample_bytree": trial.suggest_float("feature_fraction", .65, .92),
        "reg_alpha": trial.suggest_float("lambda_l1", 1e-4, 5.0, log=True),
        "reg_lambda": trial.suggest_float("lambda_l2", .1, 30.0, log=True),
        "min_split_gain": trial.suggest_float("min_gain_to_split", 1e-4, .01, log=True),
    }
    if head == "B":
        result.update({
            "lambdarank_truncation_level": trial.suggest_categorical("truncation", (5, 8, 10, 12, 16)),
            "sigmoid": trial.suggest_float("sigmoid", .5, 1.5),
        })
    else:
        result["alpha"] = trial.suggest_float("huber_alpha", .80, .95)
    return result


def _model(params: dict[str, Any], *, head: str, seed: int, jobs: int) -> lgb.LGBMModel:
    common: dict[str, Any] = {
        **params, "random_state": seed, "deterministic": True, "force_col_wise": True,
        "verbosity": -1, "n_jobs": jobs,
    }
    if head == "B":
        return lgb.LGBMRanker(
            objective="lambdarank", metric="ndcg", label_gain=GAIN_G3,
            lambdarank_norm=True, **common,
        )
    return lgb.LGBMRegressor(objective="huber", **common)


def _fit_predict(
    *, prepared: dict[str, Any], params: dict[str, Any], head: str, seed: int, jobs: int,
) -> tuple[np.ndarray, int]:
    model = _model(params, head=head, seed=seed, jobs=jobs)
    callbacks = [lgb.early_stopping(30, verbose=False)]
    if head == "B":
        model.fit(
            prepared["fit_x"], prepared["fit_y"], group=prepared["fit_group"],
            eval_set=[(prepared["valid_x"], prepared["valid_y"])],
            eval_group=[prepared["valid_group"]], callbacks=callbacks,
        )
    else:
        model.fit(
            prepared["fit_x"], prepared["fit_y"],
            eval_set=[(prepared["valid_x"], prepared["valid_y"])], callbacks=callbacks,
        )
    score = np.asarray(model.predict(prepared["held_x"]), dtype=float)
    return float(HEADS[head]["direction"]) * score, int(model.best_iteration_ or 0)


def _evaluate(prepared: dict[str, Any], score: np.ndarray, *, head: str, contract: str) -> dict[str, Any]:
    result = prepared["held"].copy()
    result["candidate_score"] = score
    result["enhanced_score"] = _enhanced_score(result, head, score)
    values = _timestamp_metrics(result, "enhanced_score")
    return {"contract": contract, "held_month": prepared["held_month"], **values}


def _prepare(args: argparse.Namespace, fields: list[str], base_override: pd.DataFrame | None) -> list[dict[str, Any]]:
    policy = _read_policy(args.policy_path)
    result: list[dict[str, Any]] = []
    for fold, held_month in enumerate(_utc(value) for value in args.held_months):
        reserve = held_month - pd.Timedelta(days=args.reserve_days)
        start = reserve - pd.DateOffset(months=args.train_months)
        window = _window(
            head=args.head, feature_root=args.feature_root, router_root=args.router_root,
            score_root=args.score_root, label_root=args.label_root, policy=policy,
            start=start, end=_next_month(held_month), route_fraction=args.route_fraction,
        )
        train = _train_rows(window.loc[window.__decision_ts__.lt(reserve)].copy(), args.head, reserve, args.train_cap)
        held = _sample_whole_queries(
            _held_rows(window.loc[window.__decision_ts__.ge(held_month)].copy()),
            args.held_cap, seed=SEED + 77 * fold,
        ).sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
        if len(train) < args.min_train_rows or len(held) < args.min_held_rows:
            raise AssertionError(f"{held_month:%Y-%m}: insufficient strict support")
        fit, valid = _split_early(train.sort_values(["__decision_ts__", "candidate_id"], kind="stable"))
        selected = pd.concat([fit, valid, held], ignore_index=True)
        matrix = _impute(_materialize(args.feature_root, selected, fields), len(fit))
        fit_end = len(fit); valid_end = fit_end + len(valid)
        outcome = str(HEADS[args.head]["target"])
        held_out = held.loc[:, [*IDENTITY, "policy_net_bps", *SCORE_FIELDS.values()]].copy()
        held_out = _apply_base_override(held_out, base_override)
        result.append({
            "held_month": f"{held_month:%Y-%m}",
            "fit_x": matrix[:fit_end], "fit_y": pd.to_numeric(fit[outcome], errors="raise").to_numpy(np.int32 if args.head == "B" else float),
            "fit_group": _groups(fit), "valid_x": matrix[fit_end:valid_end],
            "valid_y": pd.to_numeric(valid[outcome], errors="raise").to_numpy(np.int32 if args.head == "B" else float),
            "valid_group": _groups(valid), "held_x": matrix[valid_end:], "held": held_out,
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--router-root", type=Path, required=True)
    parser.add_argument("--score-root", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--feature-contract", type=Path, required=True)
    parser.add_argument("--base-score-oof", type=Path)
    parser.add_argument("--head", choices=tuple(HEADS), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--held-months", nargs="+", default=("2025-11-01", "2026-01-01", "2026-03-01", "2026-05-01", "2026-07-01"))
    parser.add_argument("--route-fraction", type=float, default=.50)
    parser.add_argument("--train-months", type=int, default=3)
    parser.add_argument("--reserve-days", type=int, default=28)
    parser.add_argument("--train-cap", type=int, default=60_000)
    parser.add_argument("--held-cap", type=int, default=36_000)
    parser.add_argument("--min-train-rows", type=int, default=8_000)
    parser.add_argument("--min-held-rows", type=int, default=2_000)
    parser.add_argument("--trials", type=int, default=32)
    parser.add_argument("--model-jobs", type=int, default=2)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    held_months = tuple(_utc(value) for value in args.held_months)
    span = (held_months[-1].year - held_months[0].year) * 12 + held_months[-1].month - held_months[0].month
    if len(held_months) < 5 or len({value.year for value in held_months}) < 2 or span < 8:
        raise ValueError("HPO requires >=5 chronological held months spanning >=8 months and two years")
    fields = _feature_contract(args.feature_contract)
    base_override = None
    if args.base_score_oof is not None:
        base_override = pd.read_parquet(args.base_score_oof, columns=[*IDENTITY, "b0_f72_score"])
        base_override["__decision_ts__"] = pd.to_datetime(base_override["__decision_ts__"], utc=True, errors="raise")
        if base_override.duplicated(list(IDENTITY)).any():
            raise AssertionError("base override has duplicate identities")
    args.out.mkdir(parents=True)
    _exclusive(args.out / "run_manifest.json", {
        "schema": "strict_r3_direct_head_crossyear_hpo_v1", "scope": "offline research only; no live mutation",
        "head": args.head, "feature_contract": str(args.feature_contract), "feature_contract_sha256": _sha(args.feature_contract),
        "features": fields, "held_months": [f"{value:%Y-%m}" for value in held_months],
        "cross_year_portability": True, "early_stopping": "final 20% chronological train timestamps; 30 rounds",
        "sampling": "whole decision timestamps only", "base_override": str(args.base_score_oof) if args.base_score_oof else None,
        "objective": "0.27 Top1 + 0.23 Top2 + 0.20 Top5 + 0.10 Top10 + temporal stability terms",
    })
    prepared = _prepare(args, fields, base_override)
    baseline = [_evaluate(item, pd.to_numeric(item["held"][SCORE_FIELDS[args.head]], errors="coerce").to_numpy(float), head=args.head, contract="incumbent") for item in prepared]
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED, multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1, interval_steps=1),
    )
    trial_rows: list[dict[str, Any]] = []

    def objective(trial: optuna.Trial) -> float:
        folds: list[dict[str, Any]] = []
        for fold, item in enumerate(prepared):
            params = _params(trial, head=args.head, rows=len(item["fit_x"]))
            score, iteration = _fit_predict(prepared=item, params=params, head=args.head, seed=SEED + 101 * fold, jobs=args.model_jobs)
            row = _evaluate(item, score, head=args.head, contract="challenger")
            row["best_iteration"] = iteration
            folds.append(row)
            partial = _objective_value({key: float(np.mean([entry[key] for entry in folds])) for key in ("ts_top01_ev", "ts_top02_ev", "ts_top05_ev", "ts_top10_ev", "monthly_median_top10", "monthly_q25_top05", "weekly_q10_top02")})
            trial.report(partial, step=fold)
            if trial.should_prune():
                raise optuna.TrialPruned()
        summary = _summarize(pd.DataFrame([*baseline, *folds]))
        value = summary["challenger_hpo_tail_stability"] - summary["incumbent_hpo_tail_stability"]
        trial_rows.append({"trial": trial.number, "state": "complete", "selection_delta": value, **summary, **trial.params})
        return value

    study.optimize(objective, n_trials=args.trials, n_jobs=1, gc_after_trial=True)
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.PRUNED:
            trial_rows.append({"trial": trial.number, "state": "pruned", **trial.params})
    complete = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if not complete:
        raise RuntimeError("no complete HPO trial")
    best = study.best_trial
    best_folds: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    for fold, item in enumerate(prepared):
        params = _params(best, head=args.head, rows=len(item["fit_x"]))
        score, iteration = _fit_predict(prepared=item, params=params, head=args.head, seed=SEED + 101 * fold, jobs=args.model_jobs)
        row = _evaluate(item, score, head=args.head, contract="challenger")
        row["best_iteration"] = iteration
        best_folds.append(row)
        prediction = item["held"].loc[:, list(IDENTITY)].copy()
        prediction["head_score"] = score
        prediction["enhanced_score"] = _enhanced_score(item["held"], args.head, score)
        prediction["held_month"] = item["held_month"]
        predictions.append(prediction)
    final_summary = _summarize(pd.DataFrame([*baseline, *best_folds]))
    _exclusive(args.out / "winner.json", {
        "head": args.head, "feature_contract_sha256": _sha(args.feature_contract), "features": fields,
        "target": HEADS[args.head], "best_trial": best.number, "best_params": best.params,
        "selection_delta": final_summary["challenger_hpo_tail_stability"] - final_summary["incumbent_hpo_tail_stability"],
        "pruner": "MedianPruner(startup=8,warmup_fold=1)", "early_stopping_rounds": 30,
    })
    pd.DataFrame(trial_rows).sort_values("trial").to_parquet(args.out / "trials.parquet", index=False, compression="zstd")
    pd.DataFrame([*baseline, *best_folds]).to_parquet(args.out / "fold_metrics.parquet", index=False, compression="zstd")
    pd.DataFrame([final_summary]).to_parquet(args.out / "summary.parquet", index=False, compression="zstd")
    pd.concat(predictions, ignore_index=True).to_parquet(args.out / "best_oof_predictions.parquet", index=False, compression="zstd")


if __name__ == "__main__":
    main()
