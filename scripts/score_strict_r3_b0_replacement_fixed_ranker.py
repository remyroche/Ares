#!/usr/bin/env python3
"""Score one frozen B0-replacement LambdaRank contract on strict OOF folds."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from run_strict_r3_b0_replacement_ranker_screen import (
    GAIN_SCHEDULES, SEED, TARGETS, _conditional_ic, _features, _groups,
    _metrics, _rank, _read_window, _sample_queries, _utc,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--router-root", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--hpo-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--held-months", nargs="+", default=("2026-02-01", "2026-03-01", "2026-04-01"))
    parser.add_argument("--train-months", type=int, default=3)
    parser.add_argument("--reserve-days", type=int, default=28)
    parser.add_argument("--train-cap", type=int, default=35000)
    parser.add_argument("--model-jobs", type=int, default=4)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    manifest = json.loads(args.hpo_manifest.read_text())
    if manifest.get("objective") != "lambdarank":
        raise ValueError("fixed scorer supports the frozen LambdaRank HPO contract only")
    target_name = str(manifest["target"])
    target = TARGETS[target_name]
    valid = target.replace("_grade", "_valid")
    gains = GAIN_SCHEDULES[str(manifest["gain_schedule"])]
    choice = dict(manifest["best_params"])
    args.out.mkdir(parents=True)
    fields = _features(args.source_root)
    audits: list[dict[str, object]] = []
    outputs: list[pd.DataFrame] = []
    for fold_index, held_text in enumerate(args.held_months):
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
        x_train = train.loc[:, fields].apply(pd.to_numeric, errors="coerce").fillna(medians).to_numpy(np.float32)
        x_held = held.loc[:, fields].apply(pd.to_numeric, errors="coerce").fillna(medians).to_numpy(np.float32)
        actual = {
            "n_estimators": 2000, "learning_rate": float(choice["learning_rate"]),
            "max_depth": int(choice["max_depth"]), "num_leaves": int(choice["num_leaves"]),
            "min_child_samples": max(40, int(round(len(train) * float(choice["min_data_fraction"]))),),
            "subsample": float(choice["bagging_fraction"]), "subsample_freq": 1,
            "colsample_bytree": float(choice["feature_fraction"]),
            "reg_alpha": float(choice["lambda_l1"]), "reg_lambda": float(choice["lambda_l2"]),
            "min_split_gain": float(choice["min_gain_to_split"]),
            "lambdarank_truncation_level": int(choice["truncation"]), "sigmoid": float(choice["sigmoid"]),
        }
        model = lgb.LGBMRanker(
            objective="lambdarank", metric="ndcg", label_gain=gains, lambdarank_norm=True,
            random_state=SEED + fold_index, deterministic=True, force_col_wise=True,
            verbosity=-1, n_jobs=args.model_jobs, **actual,
        )
        model.fit(
            x_train, pd.to_numeric(train[target], errors="coerce").to_numpy(np.int32), group=_groups(train),
            eval_set=[(x_held, pd.to_numeric(held[target], errors="coerce").to_numpy(np.int32))],
            eval_group=[_groups(held)], callbacks=[lgb.early_stopping(30, verbose=False)],
        )
        held["x_score"] = model.predict(x_held)
        held["x_rank"] = _rank(held, "x_score")
        held["e_rank"] = _rank(held, "efficiency_bps")
        held["t_rank"] = _rank(held, "timing_bps")
        held["et_rank"] = .5 * (held.e_rank + held.t_rank)
        held["etx_equal_rank"] = (held.e_rank + held.t_rank + held.x_rank) / 3.0
        measures = {name: _metrics(held, field) for name, field in (("x", "x_rank"), ("et", "et_rank"), ("etx_equal", "etx_equal_rank"))}
        audits.append({
            "held_month": f"{held_month:%Y-%m}", "train_rows": len(train), "held_rows": len(held),
            "best_iteration": int(model.best_iteration_), "conditional_ic_x_given_et": _conditional_ic(held),
            "actual_min_child_samples": actual["min_child_samples"],
            **{f"{name}_{key}": value for name, metric in measures.items() for key, value in metric.items()},
            "delta_dtp10_equal": measures["etx_equal"]["top10_precision50"] - measures["et"]["top10_precision50"],
            "delta_top10_ev_equal": measures["etx_equal"]["top10_ev"] - measures["et"]["top10_ev"],
        })
        outputs.append(held.loc[:, ["candidate_id", "__decision_ts__", "side_name", "policy_net_bps", "e_rank", "t_rank", "x_score", "x_rank", "et_rank", "etx_equal_rank"]])
    pd.DataFrame(audits).to_parquet(args.out / "fold_metrics.parquet", index=False, compression="zstd")
    pd.concat(outputs, ignore_index=True).to_parquet(args.out / "oof_predictions.parquet", index=False, compression="zstd")
    receipt = {
        "schema": "strict_r3_b0_replacement_fixed_ranker_v1", "hpo_manifest": str(args.hpo_manifest),
        "hpo_manifest_sha256": __import__("hashlib").sha256(args.hpo_manifest.read_bytes()).hexdigest(),
        "target": target_name, "target_column": target, "gain_schedule": manifest["gain_schedule"],
        "best_params": choice, "features": fields, "feature_count": len(fields),
        "query": "decision timestamp × long side", "router": "frozen top50", "strict_oof": True,
        "scope": "offline B0 replacement scorer; does not modify E/T/B0 live contracts",
    }
    fd = os.open(args.out / "run_manifest.json", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
