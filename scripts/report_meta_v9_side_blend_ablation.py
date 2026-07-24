#!/usr/bin/env python3
"""Evaluate side-model corrections on top of the historical V9 meta backbone."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["__ts__", "__symbol__", "side_name"]
OUTCOMES = [
    "calendar_month",
    "ev_after_1pct",
    "clean_exec",
    "dirty_positive",
    "full_path_bad_mae_1r",
    "timeout",
]
SCORE = "score_meta_base_soft_label"


def _read(path: Path, score_name: str, *, include_outcomes: bool) -> pd.DataFrame:
    columns = KEYS + [SCORE]
    if include_outcomes:
        columns += OUTCOMES
    frame = pd.read_parquet(path, columns=columns)
    frame = frame.rename(columns={SCORE: score_name})
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    frame["side_name"] = frame["side_name"].astype(str).str.lower()
    return frame


def _fit_reference(values: np.ndarray) -> np.ndarray:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.asarray([0.0], dtype=np.float64)
    return np.sort(finite, kind="stable")


def _cdf(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    out = np.full(arr.shape, 0.5, dtype=np.float32)
    finite = np.isfinite(arr)
    if finite.any():
        out[finite] = (
            np.searchsorted(reference, arr[finite], side="right")
            / max(int(reference.size), 1)
        ).astype(np.float32)
    return out


def _metric_row(frame: pd.DataFrame, score_col: str, scope: str) -> dict[str, object]:
    n = len(frame)
    score = pd.to_numeric(frame[score_col], errors="coerce")
    finite = score.notna() & np.isfinite(score.to_numpy(np.float64))
    if not bool(finite.any()):
        raise ValueError(f"No finite scores for {score_col} in {scope}")
    rank = score.groupby(frame["calendar_month"], sort=False).rank(
        pct=True, method="average"
    )
    selected = frame.loc[finite & rank.ge(0.90)]
    ev = pd.to_numeric(selected["ev_after_1pct"], errors="coerce")
    normalized_day = selected["__ts__"].dt.normalize()
    week = normalized_day - pd.to_timedelta(normalized_day.dt.weekday, unit="D")
    weekly = ev.groupby(week).mean()
    return {
        "scope": scope,
        "candidate_rows": int(n),
        "scoreable_rows": int(finite.sum()),
        "selected_rows": int(len(selected)),
        "mean_ev_after_1pct": float(ev.mean()),
        "worst_week_ev_after_1pct": float(weekly.min()),
        "clean_exec_precision": float(
            pd.to_numeric(selected["clean_exec"], errors="coerce").mean()
        ),
        "dirty_positive_rate": float(
            pd.to_numeric(selected["dirty_positive"], errors="coerce").mean()
        ),
        "full_path_bad_mae_rate": float(
            pd.to_numeric(selected["full_path_bad_mae_1r"], errors="coerce").mean()
        ),
        "timeout_rate": float(
            pd.to_numeric(selected["timeout"], errors="coerce").mean()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--global-predictions", type=Path, required=True)
    parser.add_argument("--side-predictions", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--calibration-month", default="2026-06")
    parser.add_argument("--alpha-step", type=float, default=0.1)
    parser.add_argument(
        "--score-space",
        choices=["raw", "side_cdf"],
        default="raw",
        help="Blend common soft-label scores directly, or use a diagnostic side-local CDF.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    global_frame = _read(args.global_predictions, "score_global", include_outcomes=True)
    side_frame = _read(args.side_predictions, "score_side", include_outcomes=False)
    frame = global_frame.merge(side_frame, on=KEYS, how="inner", validate="one_to_one")
    frame["calendar_month"] = frame["calendar_month"].astype(str)

    calibration = frame["calendar_month"].eq(str(args.calibration_month))
    if args.score_space == "side_cdf":
        for side in ("long", "short"):
            side_mask = frame["side_name"].eq(side)
            fit_mask = calibration & side_mask
            for source in ("global", "side"):
                reference = _fit_reference(
                    frame.loc[fit_mask, f"score_{source}"].to_numpy()
                )
                frame.loc[side_mask, f"rank_{source}"] = _cdf(
                    frame.loc[side_mask, f"score_{source}"].to_numpy(), reference
                )
    else:
        frame["rank_global"] = pd.to_numeric(
            frame["score_global"], errors="coerce"
        ).astype(np.float32)
        frame["rank_side"] = pd.to_numeric(
            frame["score_side"], errors="coerce"
        ).astype(np.float32)

    alpha_values = np.arange(0.0, 1.0001, float(args.alpha_step), dtype=np.float32)
    grid_rows: list[dict[str, object]] = []
    calibration_frame = frame.loc[calibration].copy()
    long_mask = calibration_frame["side_name"].eq("long").to_numpy()
    rank_global = calibration_frame["rank_global"].to_numpy(np.float32)
    rank_side = calibration_frame["rank_side"].to_numpy(np.float32)
    for alpha_long in alpha_values:
        for alpha_short in alpha_values:
            alpha = np.where(long_mask, alpha_long, alpha_short).astype(np.float32)
            side_delta = np.where(
                np.isfinite(rank_side), rank_side - rank_global, 0.0
            ).astype(np.float32)
            calibration_frame["score_blend"] = rank_global + alpha * side_delta
            row = _metric_row(calibration_frame, "score_blend", str(args.calibration_month))
            row["alpha_long"] = float(alpha_long)
            row["alpha_short"] = float(alpha_short)
            grid_rows.append(row)

    grid = pd.DataFrame(grid_rows)
    baseline = grid.loc[grid["alpha_long"].eq(0.0) & grid["alpha_short"].eq(0.0)].iloc[0]
    ev_gain = grid["mean_ev_after_1pct"] - float(baseline["mean_ev_after_1pct"])
    worst_delta = grid["worst_week_ev_after_1pct"] - float(
        baseline["worst_week_ev_after_1pct"]
    )
    grid["admissible"] = worst_delta.ge(-np.maximum(ev_gain, 0.0) / 5.0)
    eligible = grid.loc[grid["admissible"] & ev_gain.ge(0.0)]
    best = (eligible if not eligible.empty else grid).sort_values(
        ["mean_ev_after_1pct", "worst_week_ev_after_1pct"], ascending=False
    ).iloc[0]

    alpha_by_side = {
        "long": float(best["alpha_long"]),
        "short": float(best["alpha_short"]),
    }
    alpha = frame["side_name"].map(alpha_by_side).fillna(0.0).to_numpy(np.float32)
    rank_global_all = frame["rank_global"].to_numpy(np.float32)
    rank_side_all = frame["rank_side"].to_numpy(np.float32)
    side_delta_all = np.where(
        np.isfinite(rank_side_all), rank_side_all - rank_global_all, 0.0
    ).astype(np.float32)
    frame["score_blend"] = rank_global_all + alpha * side_delta_all

    report_rows: list[dict[str, object]] = []
    for model, score_col in (
        ("global_backbone", "rank_global"),
        ("side_model", "rank_side"),
        ("side_blend", "score_blend"),
    ):
        overall = _metric_row(frame, score_col, "all")
        overall["model"] = model
        report_rows.append(overall)
        for month, month_frame in frame.groupby("calendar_month", sort=True):
            row = _metric_row(month_frame, score_col, str(month))
            row["model"] = model
            report_rows.append(row)

    grid.to_csv(args.out_dir / "alpha_grid_calibration.csv", index=False)
    pd.DataFrame(report_rows).to_csv(args.out_dir / "metrics.csv", index=False)
    manifest = {
        "generated_by": Path(__file__).name,
        "global_predictions": str(args.global_predictions),
        "side_predictions": str(args.side_predictions),
        "calibration_month": str(args.calibration_month),
        "score_space": str(args.score_space),
        "selection_contract": (
            f"{args.score_space} score blend; finite predictions only; maximize "
            "calibration-month top10 EV; worst-week degradation no worse than "
            "one fifth of EV gain"
        ),
        "alpha_by_side": alpha_by_side,
        "baseline_calibration_ev": float(baseline["mean_ev_after_1pct"]),
        "best_calibration_ev": float(best["mean_ev_after_1pct"]),
        "best_calibration_worst_week": float(best["worst_week_ev_after_1pct"]),
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
