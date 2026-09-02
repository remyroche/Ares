#!/usr/bin/env python3
"""Freeze only robust timestamp-level market-context targets for downstream use.

This is a selection receipt, not a model fit.  A target earns eligibility for
the later meta/MC1 combination stage only when its strict OOS timestamp-level
screen shows all of the following on the three declared portability folds:

* a positive top-20% temporal-policy uplift in every fold;
* positive target learnability (rank IC) in every fold;
* positive mean uplift and non-negative worst-fold uplift; and
* a positive mean context-to-policy rank IC.

These gates deliberately reject a label that merely happens to map to policy
outcomes while its own future market state is not causally learnable.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "strict_r3_o3v2_market_context_selection_v1"
PORTABILITY_FOLDS = ("oos_2026_q1", "oos_2026_q2", "oos_2026_jul")
SOURCES = {
    "trend": ROOT / "data_perp/artifacts/strict_r3_o3v2_market_context_trend_20260825_v1",
    "volatility": ROOT / "data_perp/artifacts/strict_r3_o3v2_market_context_volatility_20260825_v1",
    "breadth": ROOT / "data_perp/artifacts/strict_r3_o3v2_market_context_breadth_20260825_v1",
    "dispersion": ROOT / "data_perp/artifacts/strict_r3_o3v2_market_context_dispersion_20260825_v1",
    "flow": ROOT / "data_perp/artifacts/strict_r3_o3v2_market_context_flow_20260825_v1",
    "stress": ROOT / "data_perp/artifacts/strict_r3_o3v2_market_context_stress_20260825_v1",
    "stretch": ROOT / "data_perp/artifacts/strict_r3_o3v2_market_context_stretch_20260825_v2",
    "volatility_release": ROOT / "data_perp/artifacts/strict_r3_o3v2_market_context_volatility_release_20260825_v1",
    "dependence": ROOT / "data_perp/artifacts/strict_r3_o3v2_market_context_dependence_20260825_v1",
    "leadership": ROOT / "data_perp/artifacts/strict_r3_o3v2_market_context_leadership_20260825_v1",
    "structural": ROOT / "data_perp/artifacts/strict_r3_o3v2_market_context_structural_20260825_v1",
    "leverage": ROOT / "data_perp/artifacts/strict_r3_o3v2_market_context_leverage_20260825_v2",
}


def _write_exclusive(path: Path, value: object) -> None:
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(fd, "w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, default=str)


def run(out: Path) -> Path:
    if out.exists():
        raise FileExistsError(out)
    pieces: list[pd.DataFrame] = []
    for family, root in SOURCES.items():
        metrics = root / "market_context_metrics.parquet"
        manifest = root / "run_manifest.json"
        if not metrics.exists() or not manifest.exists():
            raise FileNotFoundError(f"missing finalized context source for {family}: {root}")
        source = pd.read_parquet(metrics)
        source["family"] = family
        source["source_artifact"] = str(root.resolve())
        pieces.append(source)
    metrics = pd.concat(pieces, ignore_index=True)
    selected = metrics.loc[
        metrics["fold"].isin(PORTABILITY_FOLDS)
        & metrics["band"].eq("top20")
    ].copy()
    rows: list[dict[str, object]] = []
    for (family, target, column, source), frame in selected.groupby(["family", "target", "target_column", "source_artifact"], sort=True):
        by_fold = frame.set_index("fold").reindex(PORTABILITY_FOLDS)
        delta = pd.to_numeric(by_fold["delta_vs_all_bps"], errors="coerce")
        target_ic = pd.to_numeric(by_fold["target_rank_ic"], errors="coerce")
        policy_ic = pd.to_numeric(by_fold["policy_time_rank_ic"], errors="coerce")
        complete = bool(len(frame) == len(PORTABILITY_FOLDS) and by_fold.index.notna().all())
        target_learnable = bool(target_ic.notna().all() and target_ic.gt(0).all())
        stable_uplift = bool(delta.notna().all() and delta.gt(0).all())
        policy_useful = bool(policy_ic.notna().all() and policy_ic.mean() > 0)
        retain = complete and target_learnable and stable_uplift and policy_useful
        rows.append({
            "family": family, "target": target, "target_column": column, "source_artifact": source,
            "portability_folds": int(len(frame)), "mean_delta_vs_all_bps": float(delta.mean()), "worst_delta_vs_all_bps": float(delta.min()),
            "mean_target_rank_ic": float(target_ic.mean()), "worst_target_rank_ic": float(target_ic.min()),
            "mean_policy_time_rank_ic": float(policy_ic.mean()), "retained": retain,
            "reason": "all strict temporal-context gates passed" if retain else "failed one or more learnability/stability gates",
        })
    summary = pd.DataFrame(rows).sort_values(["retained", "mean_delta_vs_all_bps"], ascending=[False, False], kind="stable")
    out.mkdir(parents=True, exist_ok=False)
    summary.to_parquet(out / "market_context_selection_summary.parquet", index=False, compression="zstd")
    retained = summary.loc[summary["retained"]].copy()
    _write_exclusive(out / "market_context_selection.json", {
        "schema": SCHEMA,
        "scope": "eligibility for later offline meta/MC1 combination only; not an inference or live promotion",
        "portability_folds": list(PORTABILITY_FOLDS),
        "gates": {
            "temporal_policy_uplift": "top20 uplift > 0 bps in every portability fold",
            "target_learnability": "target rank IC > 0 in every portability fold",
            "temporal_policy_rank": "policy time rank IC mean > 0 with all folds finite",
        },
        "retained": retained[["family", "target", "target_column", "source_artifact"]].to_dict(orient="records"),
        "rejected": summary.loc[~summary["retained"], ["family", "target", "reason"]].to_dict(orient="records"),
    })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(run(args.out.resolve()))


if __name__ == "__main__":
    main()
