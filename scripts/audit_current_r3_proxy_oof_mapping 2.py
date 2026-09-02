#!/usr/bin/env python3
"""Audit causal side-local score-to-bps mapping on strict R3 OOF rows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _fit_map(train: pd.DataFrame, score: str, target: str, bins: int = 20) -> tuple[np.ndarray, np.ndarray, float]:
    x = train[[score, target]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) < 100:
        return np.array([-np.inf, np.inf]), np.array([float(train[target].mean())]), float(train[target].mean())
    edges = np.unique(np.nanquantile(x[score].to_numpy(float), np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 2:
        return np.array([-np.inf, np.inf]), np.array([float(x[target].mean())]), float(x[target].mean())
    idx = np.clip(np.searchsorted(edges, x[score].to_numpy(float), side="right") - 1, 0, len(edges) - 2)
    means = np.array([x.loc[idx == i, target].mean() if np.any(idx == i) else x[target].mean() for i in range(len(edges) - 1)], dtype=float)
    means = np.maximum.accumulate(means)
    return edges, means, float(x[target].mean())


def _apply(values: np.ndarray, edges: np.ndarray, means: np.ndarray) -> np.ndarray:
    idx = np.clip(np.searchsorted(edges, values, side="right") - 1, 0, len(means) - 1)
    return means[idx]


def _tail(frame: pd.DataFrame, score: str, fraction: float) -> dict[str, object]:
    n = max(1, int(np.ceil(len(frame) * fraction)))
    top = frame.nlargest(n, score)
    return {"rows": len(top), "gross_bps": float(top.gross_bps.mean()), "net_bps": float(top.net_bps.mean()), "long_net_bps": float(top.loc[top.side_name.eq("long"), "net_bps"].mean()), "short_net_bps": float(top.loc[top.side_name.eq("short"), "net_bps"].mean()), "long_rows": int(top.side_name.eq("long").sum()), "short_rows": int(top.side_name.eq("short").sum())}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--oof", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    frame = pd.read_parquet(args.oof, columns=["candidate_id", "decision_ts", "label_available_ts", "side_name", "score", "net_bps", "gross_bps", "robust_clear_event_b25"])
    frame.decision_ts = pd.to_datetime(frame.decision_ts, utc=True)
    frame.label_available_ts = pd.to_datetime(frame.label_available_ts, utc=True)
    frame["month"] = frame.decision_ts.dt.strftime("%Y-%m")
    frame["mapped_score_bps"] = np.nan
    frame["mapping_train_rows"] = 0
    frame["mapping_valid"] = False
    frame["mapping_contract"] = "prior-resolved side-local 20-bin monotone map"
    months = sorted(frame.month.unique())
    for month in months:
        start = pd.Timestamp(month + "-01", tz="UTC")
        test_mask = frame.month.eq(month)
        prior = frame.label_available_ts.lt(start)
        for side in ("long", "short"):
            test_idx = frame.index[test_mask & frame.side_name.eq(side)]
            train = frame.loc[prior & frame.side_name.eq(side)]
            if len(test_idx) == 0:
                continue
            # Do not fill an early month from a pooled/future mean.  It has no
            # causal map until enough prior-resolved labels exist.
            if len(train) < 100:
                continue
            edges, means, fallback = _fit_map(train, "score", "net_bps")
            frame.loc[test_idx, "mapped_score_bps"] = _apply(frame.loc[test_idx, "score"].to_numpy(float), edges, means)
            frame.loc[test_idx, "mapping_train_rows"] = len(train)
            frame.loc[test_idx, "mapping_valid"] = True

    rows = []
    for month in months:
        for score in ("score", "mapped_score_bps"):
            part = frame[frame.month.eq(month)]
            if score == "mapped_score_bps":
                part = part[part.mapping_valid]
            if len(part) == 0:
                continue
            for fraction in (0.01, 0.05, 0.10, 0.20):
                rec = {"month": month, "score": score, "tail_fraction": fraction, **_tail(part, score, fraction)}
                rows.append(rec)
    monthly = pd.DataFrame(rows)
    pooled_rows = []
    for score in ("score", "mapped_score_bps"):
        part = frame if score == "score" else frame[frame.mapping_valid]
        for fraction in (0.01, 0.05, 0.10, 0.20):
            if len(part):
                pooled_rows.append({"scope": "pooled", "score": score, "tail_fraction": fraction, **_tail(part, score, fraction)})
    pooled = pd.DataFrame(pooled_rows)
    side_rows = []
    for side in ("long", "short"):
        for score in ("score", "mapped_score_bps"):
            part = frame[frame.side_name.eq(side)]
            if score == "mapped_score_bps":
                part = part[part.mapping_valid]
            if len(part) == 0:
                continue
            for fraction in (0.01, 0.05, 0.10, 0.20):
                side_rows.append({"scope": f"side:{side}", "score": score, "tail_fraction": fraction, **_tail(part, score, fraction)})
    by_side = pd.DataFrame(side_rows)
    args.out.mkdir(parents=True)
    frame.to_parquet(args.out / "mapped_oof_predictions.parquet", index=False, compression="zstd")
    monthly.to_parquet(args.out / "monthly_metrics.parquet", index=False, compression="zstd")
    pooled.to_parquet(args.out / "pooled_metrics.parquet", index=False, compression="zstd")
    by_side.to_parquet(args.out / "side_metrics.parquet", index=False, compression="zstd")
    manifest = {"schema": "r3_causal_side_local_mapping_audit_v2", "status": "complete", "input": str(args.oof), "mapping": "prior-resolved labels before month start, side-local 20-bin monotone net-bps map; rows without >=100 prior labels excluded", "rows": len(frame), "mapped_rows": int(frame.mapping_valid.sum()), "latest_month": months[-1], "mapping_train_rows_min_mapped": int(frame.loc[frame.mapping_valid, "mapping_train_rows"].min()) if frame.mapping_valid.any() else 0}
    (args.out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
