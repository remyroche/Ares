#!/usr/bin/env python3
"""Finish a sealed router dual-MC1 replay without re-reading head panels.

The primary runner emits per-head diagnostics before its constrained portfolio.
On a restricted host those diagnostics can be more expensive than the actual
admission test.  This utility accepts only already sealed target-free score
panels and dual-MC1 predictions, then executes the unchanged portfolio and
causality audit.  It never fits or scores a model.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_strict_r3_router_downstream as downstream  # noqa: E402


def run(
    *, score_root: Path, mc1_root: Path, out: Path, thresholds: tuple[float, ...],
    target_free_root: Path | None = None,
) -> None:
    if out.exists():
        raise FileExistsError(out)
    folds = pd.read_parquet(score_root / "consensus_fold_audit.parquet")
    combined = pd.read_parquet(mc1_root / "dual_mc1_predictions.parquet")
    combined["__decision_ts__"] = pd.to_datetime(combined["__decision_ts__"], utc=True, errors="raise")
    start, end = min(downstream.EVALUATION_MONTHS), downstream._month_end(max(downstream.EVALUATION_MONTHS))
    combined = combined.loc[combined["__decision_ts__"].ge(start) & combined["__decision_ts__"].lt(end)].copy()
    if combined.empty:
        raise AssertionError("sealed MC1 receipt has no evaluation rows")
    out.mkdir(parents=True)
    metrics = downstream._portfolio_metrics(combined, out, thresholds)
    feature_root = target_free_root if target_free_root is not None else score_root
    audit = downstream._audit(out, feature_root / "target_free_monthly", score_root, folds, combined)
    (out / "run_manifest.json").write_text(json.dumps({
        "schema": "strict_r3_economic_router_downstream_finalize_v1",
        "scope": "offline research only; portfolio/audit from sealed score and MC1 receipts",
        "score_root": str(score_root), "mc1_root": str(mc1_root),
        "target_free_root": str(feature_root),
        "evaluation_months": [f"{month:%Y-%m}" for month in downstream.EVALUATION_MONTHS],
        "thresholds_bps": list(thresholds), "portfolio": metrics.to_dict(orient="records"),
        "correctness": audit,
    }, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-root", type=Path, required=True)
    parser.add_argument("--mc1-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target-free-root", type=Path)
    parser.add_argument("--thresholds", default="30,50")
    args = parser.parse_args()
    thresholds = tuple(float(value) for value in args.thresholds.split(",") if value)
    if not thresholds:
        parser.error("at least one threshold is required")
    run(score_root=args.score_root, mc1_root=args.mc1_root, out=args.out, thresholds=thresholds,
        target_free_root=args.target_free_root)


if __name__ == "__main__":
    main()
