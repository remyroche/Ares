#!/usr/bin/env python3
"""Evaluate two-seed E/T feature subsets by their three-way base contribution."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_strict_r3_routed_et_fulluniverse_screen import (  # noqa: E402
    HEADS, SEED, _held_eval, _impute_from_train, _metric_suite, _params,
    _selected_feature_matrix, _strict_train, _time_balanced_sample, _utc,
)
from run_strict_r3_routed_et_conditional_mda import _blend_score, _window


def _exclusive(path: Path, payload: object) -> None:
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _summary(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for size, data in detail.groupby("features", sort=False):
        item: dict[str, object] = {"features": int(size), "observations": len(data)}
        for metric in ("blend_top01_ev", "blend_top02_ev", "blend_top05_ev", "blend_top10_ev", "blend_stable", "blend_precision50", "blend_q10_week", "blend_q25_month", "x_top10_ev", "x_stable"):
            values = pd.to_numeric(data[metric], errors="coerce")
            item[f"mean_{metric}"] = float(values.mean())
            item[f"q10_{metric}"] = float(values.quantile(.10))
            item[f"worst_{metric}"] = float(values.min())
        item["selection_score"] = float(
            .50 * item["mean_blend_stable"] + .20 * item["mean_blend_top10_ev"]
            + .15 * item["mean_blend_q25_month"] + .15 * item["mean_blend_q10_week"]
        )
        rows.append(item)
    return pd.DataFrame(rows).sort_values("selection_score", ascending=False, kind="stable")


def _metrics(held: pd.DataFrame, head: str, score: np.ndarray) -> dict[str, float]:
    blend = _metric_suite(held.assign(__blend__=_blend_score(held, head, score)), "__blend__")
    standalone = _metric_suite(held.assign(__score__=score), "__score__")
    return {
        "x_top10_ev": standalone["ts_top10_ev"], "x_stable": standalone["base_stable_p10"],
        "blend_top01_ev": blend["ts_top01_ev"], "blend_top02_ev": blend["ts_top02_ev"],
        "blend_top05_ev": blend["ts_top05_ev"], "blend_top10_ev": blend["ts_top10_ev"],
        "blend_stable": blend["base_stable_p10"], "blend_precision50": blend["ts_top10_precision50"],
        "blend_q10_week": blend["q10_week_top10_ev"], "blend_q25_month": blend["q25_month_top10_ev"],
    }


def _control(held: pd.DataFrame) -> dict[str, float]:
    work = held.loc[:, ["__decision_ts__", "candidate_id", "base_bps", "efficiency_bps", "timing_bps"]].copy()
    score = _blend_score(work, "E", work.efficiency_bps.to_numpy(float))
    return _metric_suite(held.assign(__blend__=score), "__blend__")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--router-root", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--score-root", type=Path, required=True)
    parser.add_argument("--mda-root", type=Path, required=True)
    parser.add_argument("--head", choices=("E", "T"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sizes", nargs="+", type=int, default=(120, 90, 70, 50, 35, 25))
    parser.add_argument("--held-months", nargs="+", default=("2026-02-01", "2026-03-01", "2026-04-01"))
    parser.add_argument("--train-months", type=int, default=3)
    parser.add_argument("--reserve-days", type=int, default=28)
    parser.add_argument("--train-cap", type=int, default=35000)
    parser.add_argument("--held-cap", type=int, default=15000)
    parser.add_argument("--n-jobs", type=int, default=3)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    contracts: dict[int, list[str]] = {}
    for size in args.sizes:
        path = args.mda_root / f"{args.head.lower()}_conditional_subset{size}_contract.json"
        contracts[size] = list(json.loads(path.read_text())["features"])
    args.out.mkdir(parents=True)
    _exclusive(args.out / "run_manifest.json", {
        "schema": "strict_r3_routed_et_conditional_subset_ladder_v1", "scope": "offline E/T research only; B0/live unchanged",
        "head": args.head, "mda_root": str(args.mda_root), "strict_oof": True,
        "seeds": [SEED, SEED + 70000], "selection": "conditional equal B0/E/T timestamp-rank blend",
    })
    policy = pd.read_parquet(args.policy_path, columns=["candidate_id", "policy_path_valid", "policy_net_bps", "policy_label_available_ts"])
    policy.policy_path_valid = policy.policy_path_valid.fillna(False).astype(bool)
    policy.policy_label_available_ts = pd.to_datetime(policy.policy_label_available_ts, utc=True, errors="coerce")
    target, direction = str(HEADS[args.head]["target"]), float(HEADS[args.head]["direction"])
    rows: list[dict[str, object]] = []
    controls: list[dict[str, object]] = []
    for fold, held_value in enumerate(args.held_months):
        held_month = _utc(held_value)
        reserve = held_month - pd.Timedelta(days=args.reserve_days)
        window = _window(args, reserve - pd.DateOffset(months=args.train_months), held_month + pd.offsets.MonthBegin(1), policy)
        train = _strict_train(window.loc[window.__decision_ts__.lt(reserve)].copy(), reserve, target, args.train_cap)
        held = _time_balanced_sample(_held_eval(window.loc[window.__decision_ts__.ge(held_month)].copy()), args.held_cap, seed=SEED + fold).reset_index(drop=True)
        controls.append({"held_month": f"{held_month:%Y-%m}", **_control(held)})
        for size, fields in contracts.items():
            selected = pd.concat([train, held], ignore_index=True)
            values = _selected_feature_matrix(args.feature_root, selected, fields)
            values, _ = _impute_from_train(values, len(train))
            for seed in (SEED, SEED + 70000):
                model = LGBMRegressor(**_params(seed=seed + fold + (0 if args.head == "E" else 10000), n_jobs=args.n_jobs, cheap=False))
                model.fit(values[:len(train)], pd.to_numeric(train[target], errors="coerce").to_numpy(float))
                score = direction * model.predict(values[len(train):])
                rows.append({"head": args.head, "features": size, "held_month": f"{held_month:%Y-%m}", "seed": seed, **_metrics(held, args.head, score)})
                del model
            del values, selected
            gc.collect()
        del train, held, window
        gc.collect()
    detail = pd.DataFrame(rows)
    detail.to_parquet(args.out / "conditional_subset_fold_seed_metrics.parquet", index=False, compression="zstd")
    _summary(detail).to_parquet(args.out / "conditional_subset_summary.parquet", index=False, compression="zstd")
    pd.DataFrame(controls).to_parquet(args.out / "frozen_threeway_control_metrics.parquet", index=False, compression="zstd")


if __name__ == "__main__":
    main()
