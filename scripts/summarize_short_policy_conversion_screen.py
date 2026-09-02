#!/usr/bin/env python3
"""Summarise a chronological short policy-conversion formulation screen.

Selection consumes only development-fold prediction artifacts.  It does not
open or inspect a later confirmation period, and reports the economic uplift
of the top K candidates relative to the contemporaneous valid policy pool.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TOP_KS = (1, 2, 4, 8, 16, 32)
SELECTION_KS = ((1, 0.40), (2, 0.25), (4, 0.20), (8, 0.10), (16, 0.05))


def _one_fold(path: Path, spec: str) -> dict[str, float | str]:
    frame = pd.read_parquet(path)
    valid = frame.loc[
        frame.policy_path_valid.astype(bool)
        & pd.to_numeric(frame.p0_canonical_net_bps, errors="coerce").notna()
    ].copy()
    if valid.empty:
        raise ValueError(f"no valid policy rows: {path}")
    valid["p0_canonical_net_bps"] = pd.to_numeric(valid.p0_canonical_net_bps, errors="raise")
    valid["score"] = pd.to_numeric(valid.score, errors="raise")
    ics: list[float] = []
    values: dict[int, list[float]] = {k: [] for k in TOP_KS}
    uplifts: dict[int, list[float]] = {k: [] for k in TOP_KS}
    for _, group in valid.groupby("__ts__", sort=False):
        if len(group) < 2:
            continue
        ics.append(float(group.score.corr(group.p0_canonical_net_bps, method="spearman")))
        baseline = float(group.p0_canonical_net_bps.mean())
        ordered = group.sort_values(["score", "candidate_id"], ascending=[False, True], kind="stable")
        for k in TOP_KS:
            chosen = ordered.iloc[: min(k, len(ordered))]
            value = float(chosen.p0_canonical_net_bps.mean())
            values[k].append(value)
            uplifts[k].append(value - baseline)
    result: dict[str, float | str] = {
        "fold": path.parent.name,
        "spec": spec,
        "query_count": float(len(ics)),
        "policy_ic_mean": float(np.nanmean(ics)),
        "policy_ic_positive_fraction": float(np.mean(np.asarray(ics) > 0.0)),
        "valid_rows": float(len(valid)),
    }
    for k in TOP_KS:
        result[f"top_{k}_net_bps"] = float(np.mean(values[k]))
        result[f"top_{k}_uplift_bps"] = float(np.mean(uplifts[k]))
    result["economic_screen_bps"] = float(sum(weight * result[f"top_{k}_uplift_bps"] for k, weight in SELECTION_KS))
    return result


def _robust_score(values: pd.Series) -> float:
    vals = values.to_numpy(float)
    return float(np.median(vals) - 0.5 * (np.quantile(vals, 0.75) - np.quantile(vals, 0.25)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    folds = sorted(path for path in root.iterdir() if path.is_dir() and (path / "run_manifest.json").exists())
    if not folds:
        raise FileNotFoundError(f"no completed fold directories in {root}")
    records: list[dict[str, float | str]] = []
    for fold in folds:
        manifest = json.loads((fold / "run_manifest.json").read_text())
        for spec in manifest["specs"]:
            name = spec["name"]
            records.append(_one_fold(fold / f"oos_predictions_{name}.parquet", name))
    detailed = pd.DataFrame(records).sort_values(["spec", "fold"], kind="stable")
    summary_rows: list[dict[str, float | str]] = []
    for spec, group in detailed.groupby("spec", sort=True):
        row: dict[str, float | str] = {"spec": spec, "folds": int(len(group))}
        for column in detailed.columns:
            if column in {"spec", "fold"}:
                continue
            row[f"median_{column}"] = float(group[column].median())
        row["robust_economic_screen_bps"] = _robust_score(group.economic_screen_bps)
        row["worst_fold_screen_bps"] = float(group.economic_screen_bps.min())
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values(
        ["robust_economic_screen_bps", "median_top_1_uplift_bps"], ascending=False, kind="stable"
    )
    detailed.to_parquet(root / "formulation_fold_metrics.parquet", index=False, compression="zstd")
    summary.to_parquet(root / "formulation_summary.parquet", index=False, compression="zstd")
    (root / "formulation_selection.json").write_text(json.dumps({
        "schema": "strict_r3_short_policy_conversion_formulation_selection_v1",
        "selection_period": "completed pre-October chronological development folds only",
        "economic_screen": "0.40 U1 + 0.25 U2 + 0.20 U4 + 0.10 U8 + 0.05 U16",
        "robust_score": "median fold screen minus 0.5 IQR",
        "completed_folds": [fold.name for fold in folds],
        "ordered_specs": summary.spec.tolist(),
    }, indent=2) + "\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
