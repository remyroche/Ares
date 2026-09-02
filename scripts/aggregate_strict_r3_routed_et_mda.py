#!/usr/bin/env python3
"""Aggregate isolated routed E/T MDA receipts and emit nested subset contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _exclusive(path: Path, payload: object) -> None:
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _rank(frame: pd.DataFrame, name: str) -> pd.Series:
    return frame[name].rank(method="average", pct=True).fillna(.5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=ROOT / "data_perp" / "artifacts")
    parser.add_argument("--screen-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prefix", default="strict_r3_routed_et_mda_20260826_v6")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    months = ("202602", "202603", "202604")
    parents: dict[str, list[str]] = {}
    for head in ("E", "T"):
        paths = [args.artifacts / f"{args.prefix}_{head.lower()}_{month}" for month in months]
        required = [path / f"{head.lower()}_economic_mda.parquet" for path in paths]
        if not all(path.exists() for path in required):
            raise FileNotFoundError(f"missing {head} MDA receipt")
        parents[head] = [str(path) for path in paths]
        feature = pd.concat([pd.read_parquet(path) for path in required], ignore_index=True)
        family = pd.concat([pd.read_parquet(path / f"{head.lower()}_family_mda.parquet") for path in paths], ignore_index=True)
        screen = pd.read_parquet(args.screen_root / f"{head.lower()}_crossfold_feature_summary.parquet")
        summary = feature.groupby(["feature", "family"], sort=False).agg(
            folds=("folds", "sum"),
            mda_top10_ev=("mda_top10_ev", "mean"),
            mda_top01_ev=("mda_top01_ev", "mean"),
            mda_stable_p10=("mda_stable_p10", "mean"),
            mda_precision50=("mda_precision50", "mean"),
            mda_top10_jaccard=("mda_top10_jaccard", "mean"),
            mda_worst_month_top10_ev=("mda_worst_month_top10_ev", "min"),
        ).reset_index()
        summary = summary.merge(screen[["feature", "screen_score", "precision_shap", "selected_screen120"]], on="feature", how="left", validate="one_to_one")
        summary["mda_top10_rank"] = _rank(summary, "mda_top10_ev")
        summary["mda_top01_rank"] = _rank(summary, "mda_top01_ev")
        summary["mda_stable_rank"] = _rank(summary, "mda_stable_p10")
        summary["mda_precision_rank"] = _rank(summary, "mda_precision50")
        summary["mda_boundary_rank"] = _rank(summary.assign(boundary=1.0 - summary.mda_top10_jaccard), "boundary")
        summary["mda_score"] = (
            .30 * summary.mda_top10_rank + .25 * summary.mda_stable_rank +
            .20 * summary.mda_top01_rank + .10 * summary.mda_precision_rank +
            .15 * summary.mda_boundary_rank
        )
        summary["combined_screen_mda_score"] = .60 * summary.mda_score + .40 * summary.screen_score.fillna(.0)
        summary = summary.sort_values(["combined_screen_mda_score", "feature"], ascending=[False, True], kind="stable")
        summary.to_parquet(args.out / f"{head.lower()}_crossfold_mda_feature_summary.parquet", index=False, compression="zstd")
        family_summary = family.groupby("family", sort=False).agg(
            folds=("held_month", "size"), fields=("fields", "max"),
            mda_top10_ev=("delta_top10_ev", "mean"), mda_top01_ev=("delta_top01_ev", "mean"),
            mda_stable_p10=("delta_stable_p10", "mean"), mda_precision50=("delta_precision50", "mean"),
            mda_top10_jaccard=("top10_jaccard", "mean"),
        ).reset_index().sort_values("mda_stable_p10", ascending=False, kind="stable")
        family_summary.to_parquet(args.out / f"{head.lower()}_crossfold_mda_family_summary.parquet", index=False, compression="zstd")
        ordered = summary.feature.tolist()
        for size in (120, 90, 70, 50, 35, 25):
            chosen = ordered[:size]
            _exclusive(args.out / f"{head.lower()}_mda_subset{size}_contract.json", {
                "head": head, "features": chosen,
                "sha256": hashlib.sha256("\n".join(chosen).encode()).hexdigest(),
                "selection": "cross-fold OOF within-timestamp economic + Top10-boundary MDA blended with prior screen evidence",
                "parents": parents[head],
            })
    _exclusive(args.out / "run_manifest.json", {
        "schema": "strict_r3_routed_et_crossfold_mda_aggregate_v1", "strict_oof": True,
        "scope": "offline E/T selection only; B0 and live artifacts unchanged", "parents": parents,
        "subset_ladder": [120, 90, 70, 50, 35, 25],
    })


if __name__ == "__main__":
    main()
