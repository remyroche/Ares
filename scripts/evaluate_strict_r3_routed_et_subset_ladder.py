#!/usr/bin/env python3
"""Evaluate nested routed E/T MDA feature subsets on strict OOF folds."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from lightgbm import LGBMRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_strict_r3_routed_et_fulluniverse_screen import (  # noqa: E402
    HEADS, SEED, _held_eval, _impute_from_train, _joined, _metric_suite,
    _params, _selected_feature_matrix, _strict_train, _time_balanced_sample,
    _utc,
)


def _exclusive(path: Path, payload: object) -> None:
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _policy(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=["candidate_id", "policy_path_valid", "policy_net_bps", "policy_label_available_ts"])
    frame.policy_path_valid = frame.policy_path_valid.fillna(False).astype(bool)
    frame.policy_label_available_ts = pd.to_datetime(frame.policy_label_available_ts, utc=True, errors="coerce")
    return frame


def _evaluate(head: str, fields: list[str], args: argparse.Namespace, policy: pd.DataFrame) -> list[dict[str, object]]:
    target, direction = str(HEADS[head]["target"]), float(HEADS[head]["direction"])
    rows: list[dict[str, object]] = []
    for fold, held_month_value in enumerate(args.held_months):
        held_month = _utc(held_month_value)
        reserve = held_month - pd.Timedelta(days=args.reserve_days)
        window = _joined(
            feature_root=args.feature_root, router_root=args.router_root, labels_root=args.labels_root,
            policy=policy, start=reserve - pd.DateOffset(months=args.train_months),
            end=held_month + pd.offsets.MonthBegin(1), fields=(), route_fraction=.50,
        )
        train = _strict_train(window.loc[window.__decision_ts__.lt(reserve)].copy(), reserve, target, args.train_cap)
        held = _time_balanced_sample(_held_eval(window.loc[window.__decision_ts__.ge(held_month)].copy()), args.held_cap, seed=SEED + fold)
        selected = pd.concat([train, held], ignore_index=True)
        values = _selected_feature_matrix(args.feature_root, selected, fields)
        values, _ = _impute_from_train(values, len(train))
        model = LGBMRegressor(**_params(seed=SEED + fold + (0 if head == "E" else 10_000), n_jobs=args.n_jobs, cheap=False))
        model.fit(values[:len(train)], pd.to_numeric(train[target], errors="coerce").to_numpy(float))
        held = held.reset_index(drop=True)
        held["score"] = direction * model.predict(values[len(train):])
        metrics = _metric_suite(held, "score")
        rows.append({"head": head, "held_month": f"{held_month:%Y-%m}", "features": len(fields), "train_rows": len(train), "held_rows": len(held), **metrics})
        del model, values, selected, train, held, window
        gc.collect()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--router-root", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--mda-root", type=Path, required=True)
    parser.add_argument("--feature-source-parquet", type=Path, default=None, help="optional existing 120-field source-contract control")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--head", choices=("E", "T"), required=True)
    parser.add_argument("--sizes", nargs="+", type=int, default=(120, 90, 70, 50, 35, 25))
    parser.add_argument("--held-months", nargs="+", default=("2026-02-01", "2026-03-01", "2026-04-01"))
    parser.add_argument("--train-months", type=int, default=3)
    parser.add_argument("--reserve-days", type=int, default=28)
    parser.add_argument("--train-cap", type=int, default=35000)
    parser.add_argument("--held-cap", type=int, default=15000)
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    policy = _policy(args.policy_path)
    outputs: list[dict[str, object]] = []
    if args.feature_source_parquet is not None:
        names = pq.ParquetFile(args.feature_source_parquet).schema_arrow.names
        # Scores/features have 13 stable non-feature fields before the frozen
        # 120 causal base fields; validate rather than infer a moving schema.
        fields = list(names[13:])
        if len(fields) != 120:
            raise AssertionError(f"expected 120 baseline fields, found {len(fields)}")
        outputs.extend(_evaluate(args.head, fields, args, policy))
    else:
        for size in args.sizes:
            contract = json.loads((args.mda_root / f"{args.head.lower()}_mda_subset{size}_contract.json").read_text())
            outputs.extend(_evaluate(args.head, list(contract["features"]), args, policy))
    detail = pd.DataFrame(outputs)
    summary = detail.groupby(["head", "features"], sort=False).agg(
        folds=("held_month", "nunique"), ts_top01_ev=("ts_top01_ev", "mean"), ts_top02_ev=("ts_top02_ev", "mean"),
        ts_top05_ev=("ts_top05_ev", "mean"), ts_top10_ev=("ts_top10_ev", "mean"),
        top10_precision50=("ts_top10_precision50", "mean"), stable_p10=("base_stable_p10", "mean"),
        q10_week=("q10_week_top10_ev", "mean"), q25_month=("q25_month_top10_ev", "mean"),
        worst_month_top10=("ts_top10_ev", "min"),
    ).reset_index()
    summary["selection_score"] = .50 * summary.stable_p10 + .20 * summary.ts_top01_ev + .15 * summary.q10_week + .15 * summary.q25_month
    detail.to_parquet(args.out / "subset_ladder_fold_metrics.parquet", index=False, compression="zstd")
    summary.sort_values("selection_score", ascending=False, kind="stable").to_parquet(args.out / "subset_ladder_summary.parquet", index=False, compression="zstd")
    _exclusive(args.out / "run_manifest.json", {"schema": "strict_r3_routed_et_subset_ladder_v1", "head": args.head, "strict_oof": True, "scope": "offline E/T only; B0/live untouched", "sizes": args.sizes})


if __name__ == "__main__":
    main()
