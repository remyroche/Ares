#!/usr/bin/env python3
"""Reproduce the original MC1_d2 historical mapper mechanics, offline only.

This narrow producer exists solely to establish a fair MC1 control.  It has
the original full-universe day-balanced history, monthly static depth-2 fit,
daily structural curve, daily 21-day trimmed residual shift, +50 bps admission
and final-score-only auction inputs.  It intentionally contains no new feature,
model, target, or external mapper dependency.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression


ROOT = Path(__file__).resolve().parents[1]
CORE = (
    "final_score", "base_rank42", "conditional_consensus_rank", "upstream",
    "ordinary_shadow_consensus_rank", "correctness_rank",
)
FEATURE_SETS: dict[str, tuple[str, ...]] = {
    # C0: nonlinear score calibration only.  This is the matched mapper null.
    "score_only": ("final_score",),
    # C1: contemporaneous corroboration / agreement, without recent correctness.
    "score_agreement": (
        "final_score", "base_rank42", "conditional_consensus_rank", "upstream",
        "ordinary_shadow_consensus_rank",
    ),
    # C2: temporal validity state, without cross-model agreement geometry.
    "score_correctness": ("final_score", "correctness_rank"),
    # C3/C5: frozen MC1_d2 feature contract.
    "full": CORE,
}
SEED = 1729


def utc(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def robust(values: pd.Series, trim: float = .10) -> float:
    data = np.sort(pd.to_numeric(values, errors="coerce").dropna().to_numpy(float))
    if not len(data):
        return float("nan")
    count = int(trim * len(data))
    if count and len(data) > 2 * count:
        data = data[count:-count]
    return float(data.mean())


def score_bands(frame: pd.DataFrame) -> np.ndarray:
    work = frame.loc[:, ["candidate_id", "__decision_ts__", "final_score"]].copy()
    work["order"] = np.arange(len(work), dtype=np.int64)
    work = work.sort_values(
        ["__decision_ts__", "final_score", "candidate_id"],
        ascending=[True, False, True], kind="stable", na_position="last",
    )
    rank = work.groupby("__decision_ts__", sort=False).cumcount().to_numpy(float) + 1.0
    count = work.groupby("__decision_ts__", sort=False)["candidate_id"].transform("size").to_numpy(float)
    work["score_band"] = np.minimum(9, ((rank - .5) / count * 10.0).astype(np.int8))
    return work.sort_values("order", kind="stable").score_band.to_numpy(np.int8)


def day_balanced(frame: pd.DataFrame) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for _, group in frame.groupby("day", sort=True):
        ordered = group.sort_values(
            ["__decision_ts__", "final_score", "candidate_id"],
            ascending=[True, False, True], kind="stable",
        ).copy()
        ordered["rank_n"] = ordered.groupby("__decision_ts__", sort=False).cumcount() + 1
        top = ordered.loc[ordered.rank_n.le(50)]
        rest = ordered.loc[ordered.rank_n.gt(50)]
        if len(rest):
            rest = rest.sample(min(250, len(rest)), random_state=SEED)
        pieces.append(pd.concat([top, rest], ignore_index=False))
    return pd.concat(pieces, ignore_index=True).sort_values(
        ["policy_label_available_ts", "candidate_id"], kind="stable",
    )


def history(source: pd.DataFrame, decision: pd.Timestamp, days: int | None, inclusive: bool) -> pd.DataFrame:
    resolved = (
        source.policy_label_available_ts.le(decision)
        if inclusive else source.policy_label_available_ts.lt(decision)
    )
    result = source.loc[
        resolved & source.policy_path_valid.fillna(False).astype(bool) & source.policy_net_bps.notna()
    ].copy()
    if days is not None:
        result = result.loc[result.__decision_ts__.ge(decision - pd.Timedelta(days=days))].copy()
    if len(result):
        low, high = result.policy_net_bps.quantile([.02, .98]).to_numpy(float)
        result["net"] = pd.to_numeric(result.policy_net_bps, errors="coerce").clip(low, high)
    else:
        result["net"] = np.nan
    return result


def structural(train: pd.DataFrame) -> tuple[np.ndarray, float]:
    global_mean = robust(train.net)
    values = np.full(10, global_mean, dtype=float)
    for band, group in train.groupby("score_band", sort=True):
        y = pd.to_numeric(group.net, errors="coerce").dropna().to_numpy(float)
        if not len(y):
            continue
        mean = float(y.mean())
        std = max(float(y.std(ddof=0)), 1.0)
        precision = len(y) / (std * std + 1.0)
        prior = 80.0 / (250.0**2)
        values[int(band)] = (precision * mean + prior * global_mean) / (precision + prior)
    return -IsotonicRegression(increasing=True).fit_transform(np.arange(10), -values), float(global_mean)


def fit_model(
    train: pd.DataFrame,
    fields: tuple[str, ...],
) -> tuple[HistGradientBoostingRegressor, pd.Series]:
    medians = train.loc[:, fields].apply(pd.to_numeric, errors="coerce").median(numeric_only=True)
    x = train.loc[:, fields].apply(pd.to_numeric, errors="coerce").fillna(medians)
    y = pd.to_numeric(train.net, errors="coerce")
    if len(x) > 50_000:
        take = x.sample(50_000, random_state=SEED).index
        x, y = x.loc[take], y.loc[take]
    model = HistGradientBoostingRegressor(
        max_depth=2, max_iter=80, learning_rate=.04, l2_regularization=20.0,
        min_samples_leaf=100, random_state=SEED,
    ).fit(x, y)
    return model, medians


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2026-08-01")
    parser.add_argument("--inclusive-label-boundary", action="store_true",
                        help="legacy parity: allow a label resolved exactly at day start")
    parser.add_argument(
        "--feature-set", choices=tuple(FEATURE_SETS), default="full",
        help="predeclared MC1 component ablation; default is frozen full MC1_d2",
    )
    parser.add_argument(
        "--disable-recent-shift", action="store_true",
        help="diagnostic only: removes the causal 21-day global residual shift",
    )
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output exists: {args.out_dir}")
    args.out_dir.mkdir(parents=True)
    fields = FEATURE_SETS[args.feature_set]
    columns = [
        "candidate_id", "__decision_ts__", "__symbol__", "side_name",
        "policy_label_available_ts", "policy_path_valid", "policy_net_bps",
        "policy_gross_bps", "policy_exit_bar_15m", "policy_entry_price",
        "policy_exit_price", "policy_exit_reason", *CORE,
    ]
    data = pd.read_parquet(args.ledger, columns=columns)
    data["__decision_ts__"] = pd.to_datetime(data["__decision_ts__"], utc=True)
    data["policy_label_available_ts"] = pd.to_datetime(data["policy_label_available_ts"], utc=True)
    if not data.side_name.astype(str).str.lower().eq("long").all():
        raise ValueError("MC1 historical parity is long-only")
    data["score_band"] = score_bands(data)
    data["day"] = data.__decision_ts__.dt.normalize()
    source = day_balanced(data)
    start, end = utc(args.start), utc(args.end)
    boundary_equal = int(source.policy_label_available_ts.isin(pd.date_range(start, end, freq="D", tz="UTC")).sum())
    output: list[pd.DataFrame] = []
    models: dict[str, tuple[HistGradientBoostingRegressor, pd.Series]] = {}
    for day in pd.date_range(start, end, freq="D", inclusive="left", tz="UTC"):
        rows = data.loc[data.day.eq(day)].copy()
        if rows.empty:
            continue
        key = day.strftime("%Y-%m")
        long_history = history(source, day, None, args.inclusive_label_boundary)
        if len(long_history) < 5_000:
            continue
        if key not in models:
            models[key] = fit_model(long_history, fields)
        curve, global_mean = structural(long_history)
        recent = history(source, day, 21, args.inclusive_label_boundary)
        shift = robust(recent.net - curve[recent.score_band.to_numpy(int)]) if len(recent) else float("nan")
        if args.disable_recent_shift:
            shift = 0.0
        model, medians = models[key]
        x = rows.loc[:, fields].apply(pd.to_numeric, errors="coerce").fillna(medians)
        rows["static_expected_bps"] = model.predict(x)
        rows["recent_shift_bps"] = shift
        rows["mc1_expected_bps"] = rows.static_expected_bps + shift
        rows["fold_start"] = pd.Timestamp(day.year, day.month, 1, tz="UTC")
        output.append(rows)
        if day.day == 1:
            print(json.dumps({"event": "month_complete", "month": key, "history_rows": len(long_history), "shift": shift}), flush=True)
        del rows, long_history, recent
        gc.collect()
    prediction = pd.concat(output, ignore_index=True)
    prediction.to_parquet(args.out_dir / "predictions_mc1_d2_historical_parity.parquet", index=False, compression="zstd")
    (args.out_dir / "run_manifest.json").write_text(json.dumps({
        "schema": "strict_r3_mc1_d2_historical_parity_v1", "status": "complete",
        "purpose": "fair original-mechanics MC1_d2 control; no challenger selection",
        "features": list(fields),
        "model": "HistGradientBoostingRegressor depth=2 iter=80 lr=.04 l2=20 min_leaf=100 seed=1729",
        "training": "full-universe day-balanced source; 50k deterministic cap; static refit each calendar month",
        "structural_curve": "daily full resolved day-balanced history, isotonic score-band curve",
        "recent_shift": (
            "disabled diagnostic control" if args.disable_recent_shift else
            "daily 21d 10%-trimmed residual against that daily curve"
        ),
        "admission": "MC1 expected policy net >= +50 bps; not applied before predictions",
        "auction": "not applied in this producer; final-score-only report replay required",
        "label_boundary": "<= day start" if args.inclusive_label_boundary else "< day start",
        "labels_equal_to_calendar_boundary_in_source": boundary_equal,
        "exclusions": ["R5", "live state", "exchange I/O", "held outcome inputs"],
        "period": {"start": start.isoformat(), "end_exclusive": end.isoformat()},
    }, indent=2) + "\n")
    print(json.dumps({"event": "complete", "rows": len(prediction)}))


if __name__ == "__main__":
    main()
