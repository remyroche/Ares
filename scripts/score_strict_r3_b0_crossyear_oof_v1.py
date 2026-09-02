#!/usr/bin/env python3
"""Build a strict cross-year OOF ledger for the frozen B0 F72/HPO contract.

Research-only.  Each held month is scored by a B0 model whose supervised fit,
including early stopping, uses only rows resolved before that month's 28-day
reserve.  The result is a reusable counterpart ledger for conditional E/T
selection; it is not an inference or live-trading bundle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from run_strict_r3_base_stability_selector_v2 import (
    IDENTITY, SEED, _held_rows, _impute, _materialize, _next_month,
    _read_policy, _sample_whole_queries, _train_rows, _utc, _window,
)


def _exclusive(path: Path, payload: object) -> None:
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _groups(frame: pd.DataFrame) -> np.ndarray:
    return frame.groupby("__decision_ts__", sort=False).size().to_numpy(np.int32)


def _train_validation_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reserve the final 20% complete decision timestamps for early stopping."""
    times = frame.__decision_ts__.drop_duplicates().sort_values().to_numpy()
    cut = max(1, int(np.floor(.80 * len(times))))
    if cut >= len(times):
        raise AssertionError("not enough chronological B0 train queries for internal validation")
    boundary = times[cut]
    fit = frame.loc[frame.__decision_ts__.lt(boundary)].copy()
    valid = frame.loc[frame.__decision_ts__.ge(boundary)].copy()
    if fit.empty or valid.empty:
        raise AssertionError("empty chronological B0 internal validation split")
    return fit, valid


def _params(config: dict[str, object], train_rows: int, seed: int, n_jobs: int) -> dict[str, object]:
    raw = dict(config["b0_challenger"]["hpo"]["params"])
    return {
        "objective": "lambdarank", "metric": "ndcg", "n_estimators": 2000,
        "learning_rate": float(raw["learning_rate"]), "max_depth": int(raw["max_depth"]),
        "num_leaves": int(raw["num_leaves"]),
        "min_child_samples": max(40, int(round(float(raw["min_data_fraction"]) * train_rows))),
        "subsample": float(raw["bagging_fraction"]), "subsample_freq": 1,
        "colsample_bytree": float(raw["feature_fraction"]),
        "reg_alpha": float(raw["lambda_l1"]), "reg_lambda": float(raw["lambda_l2"]),
        "min_split_gain": float(raw["min_gain_to_split"]),
        "lambdarank_truncation_level": int(raw["lambdarank_truncation_level"]),
        "sigmoid": float(raw["sigmoid"]), "label_gain": [0., .5, 2., 3., 6., 8.],
        "lambdarank_norm": True, "random_state": seed, "deterministic": True,
        "force_col_wise": True, "verbosity": -1, "n_jobs": n_jobs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--router-root", type=Path, required=True)
    parser.add_argument("--score-root", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--held-head", choices=("B", "E", "T"), default="B")
    parser.add_argument("--held-label-root", type=Path, help="Support-label root for E/T held universes")
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--b0-config", type=Path, required=True)
    parser.add_argument("--feature-contract", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--held-months", nargs="+", default=("2025-11-01", "2026-01-01", "2026-03-01", "2026-05-01", "2026-07-01"))
    parser.add_argument("--train-months", type=int, default=3)
    parser.add_argument("--reserve-days", type=int, default=28)
    parser.add_argument("--train-cap", type=int, default=60_000)
    parser.add_argument("--held-cap", type=int, default=36_000)
    parser.add_argument("--n-jobs", type=int, default=8)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    held_months = tuple(_utc(value) for value in args.held_months)
    span = (held_months[-1].year - held_months[0].year) * 12 + held_months[-1].month - held_months[0].month
    if len(held_months) < 5 or len({item.year for item in held_months}) < 2 or span < 8:
        raise ValueError("B0 OOF ledger requires >=5 held months across >=8 months and two years")
    config = json.loads(args.b0_config.read_text())
    fields = list(json.loads(args.feature_contract.read_text())["selected_features"])
    if len(fields) != 72:
        raise AssertionError(f"expected frozen B0 F72, got {len(fields)} fields")
    if args.held_head != "B" and args.held_label_root is None:
        raise ValueError("--held-label-root is required for E/T held candidate universes")
    args.out.mkdir(parents=True)
    _exclusive(args.out / "run_manifest.json", {
        "schema": "strict_r3_b0_crossyear_oof_v1", "scope": "offline research only",
        "held_months": [f"{item:%Y-%m}" for item in held_months], "features": fields,
        "b0_config_sha256": hashlib.sha256(args.b0_config.read_bytes()).hexdigest(),
        "feature_contract_sha256": hashlib.sha256(args.feature_contract.read_bytes()).hexdigest(),
        "held_candidate_head": args.held_head,
        "early_stopping": "final 20% chronological pre-reserve B0 train queries only",
        "target_fields_in_feature_matrix": False,
    })
    policy = _read_policy(args.policy_path)
    outputs: list[pd.DataFrame] = []
    metrics: list[dict[str, object]] = []
    for fold, held_month in enumerate(held_months):
        reserve = held_month - pd.Timedelta(days=args.reserve_days)
        start = reserve - pd.DateOffset(months=args.train_months)
        window = _window(head="B", feature_root=args.feature_root, router_root=args.router_root,
                         score_root=args.score_root, label_root=args.label_root, policy=policy,
                         start=start, end=_next_month(held_month), route_fraction=.50)
        train = _train_rows(window.loc[window.__decision_ts__.lt(reserve)].copy(), "B", reserve, args.train_cap)
        held_window = window
        if args.held_head != "B":
            held_window = _window(head=args.held_head, feature_root=args.feature_root, router_root=args.router_root,
                                  score_root=args.score_root, label_root=args.held_label_root, policy=policy,
                                  start=start, end=_next_month(held_month), route_fraction=.50)
        held = _sample_whole_queries(_held_rows(held_window.loc[held_window.__decision_ts__.ge(held_month)].copy()), args.held_cap, seed=SEED + 77 * fold)
        fit, valid = _train_validation_split(train.sort_values(["__decision_ts__", "candidate_id"], kind="stable"))
        selected = pd.concat([fit, valid, held], ignore_index=True)
        values = _impute(_materialize(args.feature_root, selected, fields), len(fit))
        fit_x, valid_x, held_x = values[:len(fit)], values[len(fit):len(fit)+len(valid)], values[len(fit)+len(valid):]
        model = lgb.LGBMRanker(**_params(config, len(fit), SEED + fold, args.n_jobs))
        target = "policy_ordinal_base_grade"
        model.fit(fit_x, pd.to_numeric(fit[target], errors="raise").to_numpy(np.int32), group=_groups(fit),
                  eval_set=[(valid_x, pd.to_numeric(valid[target], errors="raise").to_numpy(np.int32))],
                  eval_group=[_groups(valid)], callbacks=[lgb.early_stopping(30, verbose=False)])
        score = model.predict(held_x)
        outputs.append(held.loc[:, list(IDENTITY)].assign(b0_f72_score=score, held_month=f"{held_month:%Y-%m}", best_iteration=int(model.best_iteration_)))
        metrics.append({"held_month": f"{held_month:%Y-%m}", "fit_rows": len(fit), "internal_validation_rows": len(valid), "held_rows": len(held), "best_iteration": int(model.best_iteration_)})
    pd.concat(outputs, ignore_index=True).to_parquet(args.out / "b0_oof_scores.parquet", index=False, compression="zstd")
    pd.DataFrame(metrics).to_parquet(args.out / "fold_metrics.parquet", index=False, compression="zstd")


if __name__ == "__main__":
    main()
