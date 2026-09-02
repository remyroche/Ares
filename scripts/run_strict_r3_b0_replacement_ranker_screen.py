#!/usr/bin/env python3
"""Cheap strict-OOF LambdaRank screen for B0-replacement target candidates.

Every candidate is tested on the frozen current 120 causal feature contract,
with the frozen E/T OOF scores retained only as downstream comparison inputs.
The primary result is the candidate's conditional delta over E+T, not its
standalone score.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from lightgbm import LGBMRanker

ROOT = Path(__file__).resolve().parents[1]
SEED = 1729
PREFIX_COLUMNS = 13
TARGETS = {
    "tbm_a": "tbm_a_grade", "tbm_b": "tbm_b_grade", "tbm_c": "tbm_c_grade",
    "policy_ordinal_base": "policy_ordinal_base_grade",
    "policy_ordinal_floor50": "policy_ordinal_floor50_grade",
    "path_quality": "path_quality_grade",
}
GAIN_SCHEDULES = {
    "g1_moderate_convex": [0, 1, 2, 4, 7, 11],
    "g2_stronger_top_tail": [0, 1, 3, 6, 11, 18],
    "g3_clipped_economic": [0, .5, 2, 3, 6, 8],
}


def _utc(value: object) -> pd.Timestamp:
    item = pd.Timestamp(value)
    return item.tz_localize("UTC") if item.tzinfo is None else item.tz_convert("UTC")


def _exclusive(path: Path, payload: object) -> None:
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _route(frame: pd.DataFrame) -> pd.Series:
    work = frame.loc[:, ["candidate_id", "__decision_ts__", "router_primary_rank"]].copy()
    work["pos"] = np.arange(len(work), dtype=np.int64)
    work = work.sort_values(["__decision_ts__", "router_primary_rank", "candidate_id"], ascending=[True, False, True], kind="stable")
    order = work.groupby("__decision_ts__", sort=False).cumcount() + 1
    count = work.groupby("__decision_ts__", sort=False).candidate_id.transform("size")
    selected = order.le(np.ceil(count.to_numpy(float) * .50)).to_numpy()
    out = pd.Series(False, index=np.arange(len(frame)))
    out.iloc[work.pos.to_numpy(np.int64)] = selected
    return out


def _rank(frame: pd.DataFrame, field: str) -> np.ndarray:
    work = frame.loc[:, ["__decision_ts__", "candidate_id", field]].copy()
    work["pos"] = np.arange(len(work), dtype=np.int64)
    work = work.sort_values(["__decision_ts__", field, "candidate_id"], ascending=[True, False, True], kind="stable")
    position = work.groupby("__decision_ts__", sort=False).cumcount() + 1
    count = work.groupby("__decision_ts__", sort=False).candidate_id.transform("size")
    output = np.empty(len(frame), dtype=np.float32)
    output[work.pos.to_numpy(np.int64)] = (1.0 - (position.to_numpy(float) - .5) / count.to_numpy(float)).astype(np.float32)
    return output


def _sample_queries(frame: pd.DataFrame, cap: int) -> pd.DataFrame:
    if len(frame) <= cap:
        return frame.copy()
    work = frame.copy()
    work["month"] = work.__decision_ts__.dt.strftime("%Y-%m")
    query = work.loc[:, ["__decision_ts__", "month"]].drop_duplicates().copy()
    query["hash"] = pd.util.hash_pandas_object(query.__decision_ts__.astype(str) + f"|{SEED}", index=False).to_numpy(np.uint64)
    counts = work.groupby("month", sort=False).size()
    allocation = {month: max(1, int(math.ceil(cap * count / len(work)))) for month, count in counts.items()}
    keep_ts = []
    for month, group in query.sort_values(["month", "hash"], kind="stable").groupby("month", sort=False):
        # Use complete timestamp queries until the approximate row allocation
        # is met; no query is split by the subsampler.
        target = allocation[str(month)]
        cumulative = 0
        for stamp in group["__decision_ts__"]:
            n = int((work["__decision_ts__"] == stamp).sum())
            if cumulative and cumulative + n > target:
                break
            keep_ts.append(stamp)
            cumulative += n
    return work.loc[work["__decision_ts__"].isin(keep_ts)].drop(columns="month").copy()


def _features(source_root: Path) -> list[str]:
    path = source_root / "target_free_monthly" / "month=2026-02" / "scores_features.parquet"
    fields = pq.ParquetFile(path).schema_arrow.names[PREFIX_COLUMNS:]
    if len(fields) != 120:
        raise AssertionError(f"expected frozen 120-field source, got {len(fields)}")
    return list(fields)


def _read_window(source_root: Path, router_root: Path, label_root: Path, start: pd.Timestamp, end: pd.Timestamp, fields: list[str], target: str) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for month in pd.date_range(start.normalize().replace(day=1), (end - pd.Timedelta(nanoseconds=1)).normalize().replace(day=1), freq="MS", tz="UTC"):
        token = f"{month:%Y-%m}"
        source_path = source_root / "target_free_monthly" / f"month={token}" / "scores_features.parquet"
        router_path = router_root / "target_free_scores" / f"month={token}.parquet"
        target_path = label_root / f"month={token}" / "b0_replacement_targets.parquet"
        if not (source_path.exists() and router_path.exists() and target_path.exists()):
            raise FileNotFoundError(f"missing source/router/target partition for {token}")
        source = pd.read_parquet(source_path, columns=["candidate_id", "__decision_ts__", "side_name", "efficiency_bps", "timing_bps", *fields])
        router = pd.read_parquet(router_path, columns=["candidate_id", "__decision_ts__", "side_name", "router_primary_rank"])
        labels = pd.read_parquet(target_path, columns=["candidate_id", "label_available_ts", f"{target[:-6]}_valid" if target.endswith("_grade") else "", target, "policy_net_bps"])
        labels = labels.loc[:, [item for item in labels.columns if item]].copy()
        source.__decision_ts__ = pd.to_datetime(source.__decision_ts__, utc=True, errors="raise")
        router.__decision_ts__ = pd.to_datetime(router.__decision_ts__, utc=True, errors="raise")
        labels.label_available_ts = pd.to_datetime(labels.label_available_ts, utc=True, errors="coerce")
        frame = source.merge(router, on=["candidate_id", "__decision_ts__", "side_name"], how="inner", validate="one_to_one")
        if len(frame) != len(source):
            raise AssertionError(f"{token}: target-free router identity mismatch")
        frame = frame.merge(labels, on="candidate_id", how="left", validate="one_to_one")
        frame = frame.loc[frame.__decision_ts__.ge(start) & frame.__decision_ts__.lt(end)].copy()
        frame["router_selected"] = _route(frame).to_numpy(bool)
        parts.append(frame)
    return pd.concat(parts, ignore_index=True)


def _groups(frame: pd.DataFrame) -> np.ndarray:
    return frame.groupby("__decision_ts__", sort=False).size().to_numpy(np.int32)


def _metrics(frame: pd.DataFrame, score: str) -> dict[str, float]:
    output: dict[str, float] = {}
    for fraction, name in ((.01, "01"), (.02, "02"), (.05, "05"), (.10, "10")):
        work = frame.loc[:, ["__decision_ts__", "candidate_id", "policy_net_bps", score]].copy()
        work = work.sort_values(["__decision_ts__", score, "candidate_id"], ascending=[True, False, True], kind="stable")
        order = work.groupby("__decision_ts__", sort=False).cumcount() + 1
        count = work.groupby("__decision_ts__", sort=False).candidate_id.transform("size")
        selected = work.loc[order.le(np.ceil(count.to_numpy(float) * fraction))]
        per_ts = selected.groupby("__decision_ts__", sort=False).policy_net_bps.mean()
        output[f"top{name}_ev"] = float(per_ts.mean())
        output[f"top{name}_precision50"] = float(selected.assign(win=selected.policy_net_bps.gt(50)).groupby("__decision_ts__", sort=False).win.mean().mean())
        if name == "10":
            weekly = per_ts.groupby(per_ts.index.isocalendar().year.astype(str) + "-" + per_ts.index.isocalendar().week.astype(str)).mean()
            monthly = per_ts.groupby(per_ts.index.tz_localize(None).to_period("M")).mean()
            output["q10_week_top10"] = float(weekly.quantile(.10))
            output["q25_month_top10"] = float(monthly.quantile(.25))
            output["stable_dtp10"] = float(.50 * per_ts.mean() + .20 * monthly.median() + .15 * monthly.quantile(.25) + .15 * weekly.quantile(.10))
    return output


def _conditional_ic(frame: pd.DataFrame) -> float:
    x = frame.loc[:, ["e_rank", "t_rank", "x_rank"]].to_numpy(float)
    y = pd.to_numeric(frame.policy_net_bps, errors="coerce").to_numpy(float)
    mask = np.isfinite(x).all(axis=1) & np.isfinite(y)
    if mask.sum() < 100:
        return float("nan")
    design = np.column_stack((np.ones(mask.sum()), x[mask, :2]))
    beta, *_ = np.linalg.lstsq(design, x[mask, 2], rcond=None)
    residual = x[mask, 2] - design @ beta
    return float(pd.Series(residual).corr(pd.Series(y[mask]), method="spearman"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--router-root", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--target", choices=tuple(TARGETS), required=True)
    parser.add_argument("--objective", choices=("lambdarank", "rank_xendcg"), default="lambdarank")
    parser.add_argument("--gain-schedule", choices=tuple(GAIN_SCHEDULES), default="g1_moderate_convex")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--held-months", nargs="+", default=("2026-02-01", "2026-03-01", "2026-04-01"))
    parser.add_argument("--train-months", type=int, default=3)
    parser.add_argument("--reserve-days", type=int, default=28)
    parser.add_argument("--train-cap", type=int, default=60000)
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    target = TARGETS[args.target]
    valid = target.replace("_grade", "_valid")
    fields = _features(args.source_root)
    outputs: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []
    for fold_index, held_text in enumerate(args.held_months):
        held_month = _utc(held_text)
        reserve = held_month - pd.Timedelta(days=args.reserve_days)
        window = _read_window(args.source_root, args.router_root, args.label_root, reserve - pd.DateOffset(months=args.train_months), held_month + pd.offsets.MonthBegin(1), fields, target)
        train = window.loc[
            window.router_selected & window[valid].fillna(False).astype(bool)
            & window.label_available_ts.lt(reserve) & np.isfinite(pd.to_numeric(window[target], errors="coerce"))
        ].copy()
        train = _sample_queries(train, args.train_cap).sort_values(["__decision_ts__", "candidate_id"], kind="stable")
        held = window.loc[
            window.__decision_ts__.ge(held_month) & window.router_selected & window[valid].fillna(False).astype(bool)
            & np.isfinite(pd.to_numeric(window.policy_net_bps, errors="coerce"))
        ].copy().sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
        if len(train) < 8000 or len(held) < 2000:
            raise AssertionError(f"{args.target}/{held_month:%Y-%m}: insufficient strict query support")
        medians = train.loc[:, fields].apply(pd.to_numeric, errors="coerce").median().fillna(0.0)
        x_train = train.loc[:, fields].apply(pd.to_numeric, errors="coerce").fillna(medians).to_numpy(np.float32)
        x_held = held.loc[:, fields].apply(pd.to_numeric, errors="coerce").fillna(medians).to_numpy(np.float32)
        model = LGBMRanker(
            objective=args.objective, metric="ndcg", n_estimators=140, learning_rate=.05,
            max_depth=4, num_leaves=15, min_child_samples=260, subsample=.80, subsample_freq=1,
            colsample_bytree=.80, reg_alpha=.05, reg_lambda=8.0, min_split_gain=.001,
            lambdarank_truncation_level=12, label_gain=GAIN_SCHEDULES[args.gain_schedule],
            random_state=SEED + fold_index, n_jobs=args.n_jobs, deterministic=True, force_col_wise=True, verbosity=-1,
        )
        model.fit(x_train, pd.to_numeric(train[target], errors="coerce").to_numpy(np.int32), group=_groups(train))
        held["x_score"] = model.predict(x_held)
        held["x_rank"] = _rank(held, "x_score")
        held["e_rank"] = _rank(held, "efficiency_bps")
        held["t_rank"] = _rank(held, "timing_bps")
        held["et_rank"] = .5 * (held.e_rank + held.t_rank)
        held["etx_equal_rank"] = (held.e_rank + held.t_rank + held.x_rank) / 3.0
        held["etx_quarter_rank"] = .375 * held.e_rank + .375 * held.t_rank + .25 * held.x_rank
        measures = {name: _metrics(held, field) for name, field in (("x", "x_rank"), ("et", "et_rank"), ("etx_equal", "etx_equal_rank"), ("etx_quarter", "etx_quarter_rank"))}
        audits.append({
            "target": args.target, "objective": args.objective, "gain_schedule": args.gain_schedule,
            "held_month": f"{held_month:%Y-%m}", "train_rows": len(train), "held_rows": len(held),
            "conditional_ic_x_given_et": _conditional_ic(held),
            **{f"{name}_{key}": value for name, metric in measures.items() for key, value in metric.items()},
            "delta_dtp10_equal": measures["etx_equal"]["top10_precision50"] - measures["et"]["top10_precision50"],
            "delta_top10_ev_equal": measures["etx_equal"]["top10_ev"] - measures["et"]["top10_ev"],
        })
        outputs.append(held.loc[:, ["candidate_id", "__decision_ts__", "side_name", "policy_net_bps", "e_rank", "t_rank", "x_score", "x_rank", "et_rank", "etx_equal_rank", "etx_quarter_rank"]])
        del model, x_train, x_held, train, held, window
        gc.collect()
    audit = pd.DataFrame(audits)
    audit.to_parquet(args.out / "fold_metrics.parquet", index=False, compression="zstd")
    pd.concat(outputs, ignore_index=True).to_parquet(args.out / "oof_predictions.parquet", index=False, compression="zstd")
    _exclusive(args.out / "run_manifest.json", {"schema": "strict_r3_b0_replacement_ranker_screen_v2", "target": args.target, "target_column": target, "valid_column": valid, "features": fields, "feature_count": len(fields), "model": f"cheap_common_{args.objective}", "objective": args.objective, "gain_schedule": args.gain_schedule, "label_gain": GAIN_SCHEDULES[args.gain_schedule], "query": "decision timestamp × long side", "strict_oof": True, "route": "frozen top50", "scope": "offline B0 candidate only; E/T/B0 live contracts unchanged"})


if __name__ == "__main__":
    main()
