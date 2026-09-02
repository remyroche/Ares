#!/usr/bin/env python3
"""Strict-OOF single-head 120-field tail-broadening parameter ablation.

This is intentionally a *single-head* alternative to the retained E/T 50-50
upstream.  It does not blend E, T, R3, or any other score into the challenger.
The only supervised model is a policy-ordinal LambdaRank head fitted on the
frozen 120-field causal source.  Its staged search first compares three
predeclared booster geometries, then changes label gains, truncation level,
and sigmoid.  It never adds a second base model.

Selection labels are read only for training and diagnostics.  The final score
ledger is materialised from target-free candidate/feature panels and contains
no policy, path, target, or label-availability column.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from run_strict_r3_b0_replacement_ranker_screen import (
    SEED,
    _groups,
    _read_window,
    _route,
    _sample_queries,
    _utc,
)

# These known library warnings are non-semantic and otherwise emit once per
# fold/arm.  Suppressing them keeps the long strict-OOF process observable
# through its explicit stage receipts rather than overwhelming the managed
# worker output buffer.
warnings.filterwarnings("ignore", message="Downcasting object dtype arrays on .fillna")
warnings.filterwarnings("ignore", message="X does not have valid feature names")


IDENTITY = ("candidate_id", "__decision_ts__", "side_name")
TARGET = "policy_ordinal_base_grade"
VALID = "policy_ordinal_base_valid"
GAINS: dict[str, list[float]] = {
    "g0_control": [0.0, 0.5, 2.0, 3.0, 6.0, 8.0],
    "g1_soft": [0.0, 0.75, 2.0, 3.0, 5.0, 6.5],
    "g2_medium": [0.0, 0.5, 2.0, 3.5, 7.0, 10.0],
}
CONTROL = {"gain": "g0_control", "truncation": 10, "sigmoid": 1.4293}
# A deliberately small, interpretable geometry sweep around the frozen F120
# HPO winner.  These are model *parameterisations* of the same 120-field,
# same-label, same-router, same-query head.  There is no E/T/R3 coordinate in
# any arm.  Larger head sweeps belong in a later HPO phase only if a candidate
# survives the downstream MC1/portfolio test.
MODEL_CONFIGS: dict[str, dict[str, float | int]] = {
    "f120_hpo_reference": {},
    "medium_regularised": {
        "learning_rate": 0.05, "max_depth": 5, "num_leaves": 31,
        "min_data_fraction": 0.012, "feature_fraction": 0.85,
        "bagging_fraction": 0.85, "lambda_l1": 0.01, "lambda_l2": 10.0,
        "min_gain_to_split": 0.0005,
    },
    "shallow_stable": {
        "learning_rate": 0.05, "max_depth": 4, "num_leaves": 15,
        "min_data_fraction": 0.018, "feature_fraction": 0.80,
        "bagging_fraction": 0.85, "lambda_l1": 0.02, "lambda_l2": 15.0,
        "min_gain_to_split": 0.001,
    },
}


def _exclusive(path: Path, payload: object) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _month_end(value: pd.Timestamp) -> pd.Timestamp:
    return value + pd.offsets.MonthBegin(1)


def _month_range(start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.Timestamp, ...]:
    return tuple(pd.date_range(start.normalize().replace(day=1), (end - pd.Timedelta(nanoseconds=1)).normalize().replace(day=1), freq="MS", tz="UTC"))


def _fields(source_root: Path) -> list[str]:
    probe = source_root / "target_free_monthly" / "month=2026-02" / "scores_features.parquet"
    names = pq.ParquetFile(probe).schema_arrow.names
    try:
        start = names.index("side_name") + 1
    except ValueError as exc:
        raise AssertionError("target-free 120-field source lacks side_name") from exc
    fields = list(names[start:])
    if len(fields) != 120:
        raise AssertionError(f"expected exactly 120 frozen causal fields, got {len(fields)}")
    return fields


def _strict_train(frame: pd.DataFrame, reserve: pd.Timestamp, cap: int) -> pd.DataFrame:
    mask = (
        frame.router_selected.fillna(False).astype(bool)
        & frame[VALID].fillna(False).astype(bool)
        & frame.label_available_ts.lt(reserve)
        & np.isfinite(pd.to_numeric(frame[TARGET], errors="coerce"))
    )
    return _sample_queries(frame.loc[mask].copy(), cap).sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)


def _diagnostic_held(frame: pd.DataFrame) -> pd.DataFrame:
    mask = (
        frame.router_selected.fillna(False).astype(bool)
        & frame[VALID].fillna(False).astype(bool)
        & np.isfinite(pd.to_numeric(frame.policy_net_bps, errors="coerce"))
    )
    return frame.loc[mask].copy().sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)


def _internal_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reserve the last 20% of *prior-resolved* timestamp queries for ES."""
    timestamps = frame.__decision_ts__.drop_duplicates().sort_values().to_numpy()
    cut = max(1, int(math.floor(.80 * len(timestamps))))
    if cut >= len(timestamps):
        raise AssertionError("need at least two chronological train timestamps")
    boundary = timestamps[cut]
    fit = frame.loc[frame.__decision_ts__.lt(boundary)].copy()
    valid = frame.loc[frame.__decision_ts__.ge(boundary)].copy()
    if fit.empty or valid.empty:
        raise AssertionError("empty chronological early-stopping split")
    return fit, valid


def _matrix(frame: pd.DataFrame, fields: list[str], medians: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    values = frame.loc[:, fields].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(np.float32)
    if medians is None:
        medians = np.nanmedian(values, axis=0).astype(np.float32)
        medians[~np.isfinite(medians)] = 0.0
    bad = ~np.isfinite(values)
    if bad.any():
        values[bad] = np.broadcast_to(medians, values.shape)[bad]
    return values, medians


def _rank_desc(frame: pd.DataFrame, field: str) -> np.ndarray:
    work = frame.loc[:, ["candidate_id", "__decision_ts__", field]].copy()
    work["__pos__"] = np.arange(len(work), dtype=np.int64)
    work = work.sort_values(["__decision_ts__", field, "candidate_id"], ascending=[True, False, True], kind="stable")
    ordinal = work.groupby("__decision_ts__", sort=False).cumcount().to_numpy(np.int32) + 1
    output = np.empty(len(frame), dtype=np.int32)
    output[work["__pos__"].to_numpy(np.int64)] = ordinal
    return output


def _weekly_monthly(per_timestamp: pd.Series) -> tuple[pd.Series, pd.Series]:
    week = per_timestamp.groupby(per_timestamp.index.isocalendar().year.astype(str) + "-" + per_timestamp.index.isocalendar().week.astype(str)).mean()
    month = per_timestamp.groupby(per_timestamp.index.tz_localize(None).to_period("M")).mean()
    return week, month


def _topk(frame: pd.DataFrame, score: str, k: int) -> dict[str, float]:
    ranks = _rank_desc(frame, score)
    selected = frame.loc[ranks <= k, ["__decision_ts__", "policy_net_bps"]].copy()
    per_timestamp = selected.groupby("__decision_ts__", sort=False).policy_net_bps.mean()
    precision = selected.assign(__win__=selected.policy_net_bps.gt(50.0)).groupby("__decision_ts__", sort=False).__win__.mean()
    week, month = _weekly_monthly(per_timestamp)
    return {
        "ev": float(per_timestamp.mean()),
        "precision50": float(precision.mean()),
        "rows": float(len(selected)),
        "timestamps": float(len(per_timestamp)),
        "q10_week": float(week.quantile(.10)),
        "q25_month": float(month.quantile(.25)),
        "worst_month": float(month.min()),
    }


def _dtp(frame: pd.DataFrame, score: str, k: int) -> dict[str, float]:
    ranks = _rank_desc(frame, score)
    work = frame.loc[ranks <= k, ["__decision_ts__", "policy_net_bps"]].copy()
    work["__rank__"] = ranks[ranks <= k]
    work["__weight__"] = 1.0 / np.log2(work.__rank__.to_numpy(float) + 1.0)
    numerator = (work.policy_net_bps.to_numpy(float) * work.__weight__.to_numpy(float))
    per_timestamp = pd.Series(numerator, index=work.__decision_ts__).groupby(level=0, sort=False).sum() / work.groupby("__decision_ts__", sort=False).__weight__.sum()
    week, month = _weekly_monthly(per_timestamp)
    return {
        "value": float(per_timestamp.mean()),
        "q10_week": float(week.quantile(.10)),
        "q25_month": float(month.quantile(.25)),
        "median_month": float(month.median()),
        "worst_month": float(month.min()),
    }


def _metrics(frame: pd.DataFrame, score: str) -> dict[str, float]:
    output: dict[str, float] = {}
    for k in (1, 2, 3, 5, 10, 15, 20):
        values = _topk(frame, score, k)
        output.update({f"top{k}_{key}": value for key, value in values.items()})
    for k in (2, 5, 10, 15, 20):
        values = _dtp(frame, score, k)
        output.update({f"dtp{k}_{key}": value for key, value in values.items()})
    output["base_selection_score"] = float(
        .30 * output["dtp2_value"] + .25 * output["dtp5_value"] + .20 * output["dtp10_value"]
        + .06 * output["dtp15_value"] + .03 * output["dtp20_value"]
        + .07 * output["dtp5_median_month"] + .05 * output["dtp10_q25_month"]
        + .04 * output["dtp5_q10_week"]
    )
    return output


def _params(base: dict[str, Any], *, rows: int, seed: int, jobs: int, gain: str, truncation: int, sigmoid: float) -> dict[str, Any]:
    return {
        "objective": "lambdarank", "metric": "ndcg", "n_estimators": 2000,
        "learning_rate": float(base["learning_rate"]), "max_depth": int(base["max_depth"]),
        "num_leaves": int(base["num_leaves"]),
        "min_child_samples": max(40, int(round(rows * float(base["min_data_fraction"])))),
        "subsample": float(base["bagging_fraction"]), "subsample_freq": 1,
        "colsample_bytree": float(base["feature_fraction"]),
        "reg_alpha": float(base["lambda_l1"]), "reg_lambda": float(base["lambda_l2"]),
        "min_split_gain": float(base["min_gain_to_split"]),
        "lambdarank_truncation_level": int(truncation), "sigmoid": float(sigmoid),
        "label_gain": GAINS[gain], "lambdarank_norm": True, "random_state": int(seed),
        "deterministic": True, "force_col_wise": True, "verbosity": -1, "n_jobs": int(jobs),
    }


def _model_base(base: dict[str, Any], arm: dict[str, Any]) -> dict[str, Any]:
    config = str(arm.get("model_config", "f120_hpo_reference"))
    if config not in MODEL_CONFIGS:
        raise AssertionError(f"unknown single-head model configuration: {config}")
    return {**base, **MODEL_CONFIGS[config]}


@dataclass(frozen=True)
class PreparedFold:
    month: str
    fit: pd.DataFrame
    validation: pd.DataFrame
    held: pd.DataFrame
    x_fit: np.ndarray
    x_validation: np.ndarray
    x_held: np.ndarray


def _prepare_fold(args: argparse.Namespace, fields: list[str], held_text: str) -> PreparedFold:
    """Materialise one fold only.

    The staged arms share this read-only matrix, then it is released before
    the next month.  Holding all five full-width folds was needlessly large
    and can trigger macOS memory termination when LightGBM arms run in
    parallel.
    """
    held_month = _utc(held_text)
    reserve = held_month - pd.Timedelta(days=args.reserve_days)
    window = _read_window(
        args.source_root, args.router_root, args.label_root,
        reserve - pd.DateOffset(months=args.train_months), _month_end(held_month), fields, TARGET,
    )
    training = _strict_train(window.loc[window.__decision_ts__.lt(reserve)].copy(), reserve, args.train_cap)
    held = _diagnostic_held(window.loc[window.__decision_ts__.ge(held_month)].copy())
    if len(training) < 8_000 or len(held) < 2_000:
        raise AssertionError(f"{held_month:%Y-%m}: insufficient strict single-head support")
    fit, validation = _internal_split(training)
    x_fit, medians = _matrix(fit, fields)
    x_validation, _ = _matrix(validation, fields, medians)
    x_held, _ = _matrix(held, fields, medians)
    return PreparedFold(
        month=f"{held_month:%Y-%m}", fit=fit, validation=validation, held=held,
        x_fit=x_fit, x_validation=x_validation, x_held=x_held,
    )


def _evaluate_one_fold(arm: dict[str, Any], item: PreparedFold, base: dict[str, Any], model_jobs: int, fold_index: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    model = lgb.LGBMRanker(**_params(
        _model_base(base, arm), rows=len(item.fit), seed=SEED + 10_000 * int(arm["seed_offset"]) + fold_index,
        jobs=model_jobs, gain=str(arm["gain"]), truncation=int(arm["truncation"]), sigmoid=float(arm["sigmoid"]),
    ))
    model.fit(
        item.x_fit, pd.to_numeric(item.fit[TARGET], errors="raise").to_numpy(np.int32), group=_groups(item.fit),
        eval_set=[(item.x_validation, pd.to_numeric(item.validation[TARGET], errors="raise").to_numpy(np.int32))],
        eval_group=[_groups(item.validation)], callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    held = item.held.loc[:, [*IDENTITY, "policy_net_bps"]].copy()
    held["head_score"] = model.predict(item.x_held)
    held["held_month"] = item.month
    values = _metrics(held, "head_score")
    fold = {
        **arm, "held_month": item.month, "fit_rows": len(item.fit), "internal_validation_rows": len(item.validation),
        "held_rows": len(item.held), "best_iteration": int(model.best_iteration_), **values,
    }
    return held, fold


def _guardrails(metrics: dict[str, Any], control: dict[str, Any]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    checks = (
        ("top1_ev", .97, "top-1"), ("top2_ev", .98, "top-2"),
        ("dtp5_value", .98, "DTP5"), ("dtp5_q10_week", .95, "DTP5 weekly q10"),
    )
    for field, retention, name in checks:
        if float(metrics[field]) < retention * float(control[field]):
            failures.append(name)
    # "No catastrophic month" is frozen here as: cannot turn an otherwise
    # positive control month negative and cannot lose more than 10 bps on the
    # worst monthly DTP5 value.
    control_worst = float(control["dtp5_worst_month"])
    candidate_worst = float(metrics["dtp5_worst_month"])
    if control_worst > 0.0 and candidate_worst <= 0.0:
        failures.append("catastrophic monthly sign reversal")
    if candidate_worst < control_worst - 10.0:
        failures.append("worst monthly DTP5")
    return not failures, failures


def _select(rows: list[dict[str, Any]], control: dict[str, Any]) -> dict[str, Any]:
    evaluated: list[dict[str, Any]] = []
    for row in rows:
        passed, failures = _guardrails(row, control)
        evaluated.append({**row, "guardrail_pass": passed, "guardrail_failures": "|".join(failures)})
    safe = [row for row in evaluated if row["guardrail_pass"]]
    if not safe:
        return next(row for row in evaluated if row["name"] == "control")
    return sorted(safe, key=lambda row: (-float(row["base_selection_score"]), -float(row["dtp5_q10_week"]), -float(row["top1_ev"]), str(row["name"])))[0]


def _target_free_month(source_root: Path, router_root: Path, month: pd.Timestamp, fields: list[str]) -> pd.DataFrame:
    token = f"{month:%Y-%m}"
    source = source_root / "target_free_monthly" / f"month={token}" / "scores_features.parquet"
    router = router_root / "target_free_scores" / f"month={token}.parquet"
    if not source.exists() or not router.exists():
        raise FileNotFoundError(f"missing target-free source/router for {token}")
    panel = pd.read_parquet(source, columns=[*IDENTITY, *fields])
    rank = pd.read_parquet(router, columns=[*IDENTITY, "router_primary_rank"])
    panel["__decision_ts__"] = pd.to_datetime(panel["__decision_ts__"], utc=True, errors="raise")
    rank["__decision_ts__"] = pd.to_datetime(rank["__decision_ts__"], utc=True, errors="raise")
    output = panel.merge(rank, on=list(IDENTITY), how="inner", validate="one_to_one")
    if len(output) != len(panel) or output.duplicated(list(IDENTITY)).any():
        raise AssertionError(f"{token}: target-free router identity mismatch")
    output["router_selected"] = _route(output).to_numpy(bool)
    return output.loc[output.router_selected].drop(columns="router_selected").sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)


def _final_target_free_scores(args: argparse.Namespace, fields: list[str], base: dict[str, Any], winner: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    target_root = args.out / "target_free_scores"
    for fold_index, month in enumerate(_month_range(_utc(args.score_start), _utc(args.score_end))):
        reserve = month - pd.Timedelta(days=args.reserve_days)
        window = _read_window(
            args.source_root, args.router_root, args.label_root,
            reserve - pd.DateOffset(months=args.train_months), reserve, fields, TARGET,
        )
        training = _strict_train(window, reserve, args.train_cap)
        if len(training) < 8_000:
            raise AssertionError(f"{month:%Y-%m}: insufficient strict training support for target-free score")
        fit, validation = _internal_split(training)
        x_fit, medians = _matrix(fit, fields)
        x_validation, _ = _matrix(validation, fields, medians)
        held = _target_free_month(args.source_root, args.router_root, month, fields)
        x_held, _ = _matrix(held, fields, medians)
        model = lgb.LGBMRanker(**_params(
            _model_base(base, winner), rows=len(fit), seed=SEED + 50_000 + fold_index, jobs=args.final_model_jobs,
            gain=str(winner["gain"]), truncation=int(winner["truncation"]), sigmoid=float(winner["sigmoid"]),
        ))
        model.fit(
            x_fit, pd.to_numeric(fit[TARGET], errors="raise").to_numpy(np.int32), group=_groups(fit),
            eval_set=[(x_validation, pd.to_numeric(validation[TARGET], errors="raise").to_numpy(np.int32))],
            eval_group=[_groups(validation)], callbacks=[lgb.early_stopping(30, verbose=False)],
        )
        output = held.loc[:, list(IDENTITY)].copy()
        output["head_score"] = model.predict(x_held).astype(np.float32)
        output["held_month"] = f"{month:%Y-%m}"
        target = target_root / f"month={month:%Y-%m}"
        target.mkdir(parents=True, exist_ok=False)
        output.to_parquet(target / "target_free_scores.parquet", index=False, compression="zstd")
        rows.append({
            "month": f"{month:%Y-%m}", "rows": len(output), "timestamps": int(output.__decision_ts__.nunique()),
            "fit_rows": len(fit), "internal_validation_rows": len(validation), "best_iteration": int(model.best_iteration_),
            "feature_complete_fraction": float(np.isfinite(x_held).all(axis=1).mean()), "target_free": True,
        })
    result = pd.DataFrame(rows)
    result.to_parquet(args.out / "target_free_coverage_audit.parquet", index=False, compression="zstd")
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--router-root", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--base-manifest", type=Path, required=True, help="frozen 120-feature common model configuration")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--held-months", nargs="+", default=("2025-10-01", "2025-12-01", "2026-02-01", "2026-04-01", "2026-06-01"))
    parser.add_argument("--train-months", type=int, default=3)
    parser.add_argument("--reserve-days", type=int, default=28)
    parser.add_argument("--train-cap", type=int, default=60_000)
    parser.add_argument("--arm-workers", type=int, default=3)
    parser.add_argument("--model-jobs", type=int, default=4)
    parser.add_argument("--score-start", default="2025-10-01")
    parser.add_argument("--score-end", default="2026-08-01", help="exclusive; includes the July-2026 held score month by default")
    parser.add_argument("--final-model-jobs", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if args.arm_workers * args.model_jobs > (os.cpu_count() or 1):
        raise ValueError("arm-workers × model-jobs exceeds detected CPU count")
    if _utc(args.score_start) >= _utc(args.score_end):
        raise ValueError("score start must precede score end")
    base_manifest = json.loads(args.base_manifest.read_text())
    base = dict(base_manifest["best_params"])
    fields = _fields(args.source_root)
    forbidden_score_inputs = {"base_bps", "efficiency_bps", "timing_bps", "enhanced_base_bps"}
    if forbidden_score_inputs.intersection(fields):
        raise AssertionError("single-head feature matrix unexpectedly contains incumbent score coordinates")
    if list(base_manifest.get("features", fields)) != fields:
        raise AssertionError("the provided frozen F120 HPO manifest does not match the 120-field source order")
    args.out.mkdir(parents=True)
    _exclusive(args.out / "run_manifest.json", {
        "schema": "strict_r3_single_head_tail_broadening_v1",
        "scope": "offline single-head challenger; incumbent E/T, live inference, exchange, and policy contracts are unchanged",
        "architecture": "one policy-ordinal LambdaRank base head only; E/T/R3 are neither inputs nor blend coordinates",
        "source_root": str(args.source_root), "router_root": str(args.router_root), "label_root": str(args.label_root),
        "base_manifest": str(args.base_manifest), "base_manifest_sha256": _sha(args.base_manifest),
        "feature_contract": {"count": len(fields), "sha256": hashlib.sha256("\n".join(fields).encode()).hexdigest(), "fields": fields},
        "target": TARGET, "validity": VALID, "query": "decision timestamp × long side", "router": "frozen timestamp-local top 50%",
        "train_contract": {"train_months": args.train_months, "reserve_days": args.reserve_days, "train_cap": args.train_cap, "early_stopping": "last 20% chronological prior-resolved queries only; 30 rounds"},
        "search": {"staged": ["model_geometry", "gain", "truncation", "sigmoid"], "model_configs": MODEL_CONFIGS, "gains": GAINS, "truncation": [8, 10, 12, 14, 18], "sigmoid": [0.8, 1.0, 1.4293], "control": CONTROL},
        "held_months": list(args.held_months), "score_months": [f"{item:%Y-%m}" for item in _month_range(_utc(args.score_start), _utc(args.score_end))],
        "target_free_final_scores": True,
    })
    parameter_arms = [
        {"name": "control", "model_config": "f120_hpo_reference", **CONTROL, "seed_offset": 0},
        {"name": "model_medium_regularised", "model_config": "medium_regularised", **CONTROL, "seed_offset": 1},
        {"name": "model_shallow_stable", "model_config": "shallow_stable", **CONTROL, "seed_offset": 2},
    ]

    def run_arms(arms: list[dict[str, Any]], *, stage: str) -> tuple[list[dict[str, Any]], list[pd.DataFrame]]:
        """Evaluate a stage fold-by-fold, parallelising only independent arms.

        This preserves exact models/rows/seeds while bounding peak memory to
        one full-width fold plus at most ``arm_workers`` boosters.
        """
        scored: dict[str, list[pd.DataFrame]] = {str(arm["name"]): [] for arm in arms}
        fold_rows: list[dict[str, Any]] = []
        for fold_index, held_text in enumerate(args.held_months):
            prepared = _prepare_fold(args, fields, str(held_text))
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(args.arm_workers, len(arms))) as pool:
                future = {
                    pool.submit(_evaluate_one_fold, arm, prepared, base, args.model_jobs, fold_index): arm
                    for arm in arms
                }
                for task in concurrent.futures.as_completed(future):
                    held, fold = task.result()
                    scored[str(fold["name"])].append(held)
                    fold_rows.append(fold)
            del prepared
            import gc
            gc.collect()
            print(json.dumps({
                "event": "fold_complete", "stage": stage,
                "held_month": str(held_text), "arms": len(arms),
            }), flush=True)
        metrics: list[dict[str, Any]] = []
        for arm in arms:
            arm_folds = [row for row in fold_rows if row["name"] == arm["name"]]
            output = pd.concat(scored[str(arm["name"])], ignore_index=True)
            values = _metrics(output, "head_score")
            metrics.append({
                **arm, **values, "folds": len(arm_folds),
                "mean_best_iteration": float(np.mean([row["best_iteration"] for row in arm_folds])),
                "mean_fit_rows": float(np.mean([row["fit_rows"] for row in arm_folds])),
            })
        return metrics, [pd.DataFrame(fold_rows)]

    all_metrics: list[dict[str, Any]] = []
    all_folds: list[pd.DataFrame] = []
    stage0, folds0 = run_arms(parameter_arms, stage="model_geometry")
    all_metrics.extend([{**row, "stage": "model_geometry"} for row in stage0])
    all_folds.extend([item.assign(stage="model_geometry") for item in folds0])
    control = next(row for row in stage0 if row["name"] == "control")
    model_winner = _select(stage0, control)
    print(json.dumps({"event": "stage_complete", "stage": "model_geometry", "winner": model_winner["name"]}), flush=True)

    gain_arms = [
        {"name": "control", "model_config": model_winner["model_config"], **CONTROL, "seed_offset": 10},
        {"name": "gain_g1", "model_config": model_winner["model_config"], "gain": "g1_soft", "truncation": 10, "sigmoid": CONTROL["sigmoid"], "seed_offset": 11},
        {"name": "gain_g2", "model_config": model_winner["model_config"], "gain": "g2_medium", "truncation": 10, "sigmoid": CONTROL["sigmoid"], "seed_offset": 12},
    ]
    stage1, folds1 = run_arms(gain_arms, stage="gain")
    all_metrics.extend([{**row, "stage": "gain"} for row in stage1])
    all_folds.extend([item.assign(stage="gain") for item in folds1])
    gain_winner = _select(stage1, control)
    print(json.dumps({"event": "stage_complete", "stage": "gain", "winner": gain_winner["name"]}), flush=True)

    truncation_arms = [{"name": f"trunc_{value}", "model_config": model_winner["model_config"], "gain": gain_winner["gain"], "truncation": value, "sigmoid": CONTROL["sigmoid"], "seed_offset": 30 + value} for value in (8, 10, 12, 14, 18)]
    stage2, folds2 = run_arms(truncation_arms, stage="truncation")
    all_metrics.extend([{**row, "stage": "truncation"} for row in stage2])
    all_folds.extend([item.assign(stage="truncation") for item in folds2])
    trunc_winner = _select(stage2, control)
    print(json.dumps({"event": "stage_complete", "stage": "truncation", "winner": trunc_winner["name"]}), flush=True)

    sigmoid_arms = [{"name": f"sigmoid_{value:g}", "model_config": model_winner["model_config"], "gain": trunc_winner["gain"], "truncation": trunc_winner["truncation"], "sigmoid": value, "seed_offset": 50 + index} for index, value in enumerate((.8, 1.0, 1.4293))]
    stage3, folds3 = run_arms(sigmoid_arms, stage="sigmoid")
    all_metrics.extend([{**row, "stage": "sigmoid"} for row in stage3])
    all_folds.extend([item.assign(stage="sigmoid") for item in folds3])
    winner = _select(stage3, control)
    print(json.dumps({"event": "stage_complete", "stage": "sigmoid", "winner": winner["name"]}), flush=True)
    winner = {**winner, "selection_path": {"model_geometry": model_winner["name"], "gain": gain_winner["name"], "truncation": trunc_winner["name"], "sigmoid": winner["name"]}}

    selection_rows: list[dict[str, Any]] = []
    for row in all_metrics:
        passed, failures = _guardrails(row, control)
        selection_rows.append({**row, "guardrail_pass": passed, "guardrail_failures": "|".join(failures)})
    selection = pd.DataFrame(selection_rows)
    selection.to_parquet(args.out / "selection_summary.parquet", index=False, compression="zstd")
    pd.concat(all_folds, ignore_index=True).to_parquet(args.out / "fold_metrics.parquet", index=False, compression="zstd")
    _exclusive(args.out / "winner.json", winner)
    target_free = _final_target_free_scores(args, fields, base, winner)
    correctness = {
        "schema": "strict_r3_single_head_tail_broadening_correctness_v1",
        "single_head_only": True,
        "no_et_or_r3_input": True,
        "strict_prior_resolved_train": True,
        "internal_early_stopping_is_prior_only": True,
        "held_outcomes_used_only_for_diagnostics": True,
        "final_score_outputs_are_target_free": True,
        "target_free_months": int(len(target_free)),
        "target_free_min_feature_complete_fraction": float(target_free.feature_complete_fraction.min()),
        "staged_search_arms": int(len(selection)),
        "winner": {key: winner[key] for key in ("model_config", "gain", "truncation", "sigmoid", "base_selection_score", "guardrail_pass")},
    }
    _exclusive(args.out / "correctness_report.json", correctness)
    print(json.dumps({"event": "complete", "out": str(args.out), "winner": winner["name"], "selection_score": winner["base_selection_score"]}), flush=True)


if __name__ == "__main__":
    main()
