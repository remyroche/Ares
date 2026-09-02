#!/usr/bin/env python3
"""Strict prequential C0 versus C1 portfolio replay for June--December 2025.

The causal S/R source must contain OOF entry snapshots for April onward.  The
MC1 map starts from February score history, while the April--May S/R outputs
warm C1 before the June evaluation.  Both arms retain the identical dual-MC1
admission, BCF-priority auction, rich parent-policy labels and portfolio
constraints.  This script is research-only and never imports live execution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_canonical_sr_e2_mc1_input_ablation as base


HISTORY_START = pd.Timestamp("2025-02-01T00:00:00Z")
EVAL_START = pd.Timestamp("2025-06-01T00:00:00Z")
EVAL_END = pd.Timestamp("2026-01-01T00:00:00Z")
CURRENT = base.CURRENT
BCF = base.BCF
DEFAULT_SR = ROOT / "data_perp/artifacts/causal_sr_heads_2025_c1_replay_20260831_v1"
DEFAULT_OUT = ROOT / "data_perp/artifacts/causal_sr_c1_mc1_2025_replay_20260831_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc(value: str) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _empty_e2(ids: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({
        "candidate_id": ids.astype(str),
        base.E2_OUTPUT: np.nan,
        base.E2_AVAILABLE: np.zeros(len(ids), dtype=np.int8),
    })


def _window(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    ts = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    return frame.loc[ts.ge(start) & ts.lt(end)].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current", type=Path, default=CURRENT)
    parser.add_argument("--bcf", type=Path, default=BCF)
    parser.add_argument("--sr-root", type=Path, default=DEFAULT_SR)
    parser.add_argument("--history-start", default=HISTORY_START.isoformat())
    parser.add_argument("--start", default=EVAL_START.isoformat())
    parser.add_argument("--end", default=EVAL_END.isoformat())
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    history_start, start, end = (_utc(args.history_start), _utc(args.start), _utc(args.end))
    if not (history_start < start < end):
        raise ValueError("require history-start < start < end")
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"immutable output already exists: {out}")
    out.mkdir(parents=True)
    current = base._load_family(args.current.resolve(), "current_v5")
    bcf = base._load_family(args.bcf.resolve(), "bcf")
    labels = base._candidate_labels(current, bcf)
    # Reuse the hardened family-local MC1 implementation with a period-local
    # start.  It filters every fit by both decision and resolved-label time.
    base.FEATURE_START = history_start
    e2 = _empty_e2(pd.concat([current.scores.candidate_id, bcf.scores.candidate_id], ignore_index=True).drop_duplicates())
    current_scores, current_full = base._augment_family(current, args.sr_root.resolve(), e2)
    bcf_scores, bcf_full = base._augment_family(bcf, args.sr_root.resolve(), e2)
    del current_scores, bcf_scores
    coverage = current_full.loc[:, ["candidate_id", "__decision_ts__", base.SR_AVAILABLE]].copy()
    coverage = _window(coverage, start, end)
    if coverage.empty:
        raise RuntimeError("C1 held window has no common score-family rows")
    arms = {
        "C0_refit_core": (),
        "C1_refit_core_plus_causal_sr": (*base.sr.SR_FEATURES, base.SR_AVAILABLE),
    }
    metrics, monthly, audits = [], [], []
    for arm, extras in arms.items():
        current_pred, current_audit = base._refit_family(current_full, family="current_v5", extras=extras, start=start, end=end)
        bcf_pred, bcf_audit = base._refit_family(bcf_full, family="bcf", extras=extras, start=start, end=end)
        current_pred.to_parquet(out / f"{arm}_current_mc1_predictions.parquet", index=False, compression="zstd")
        bcf_pred.to_parquet(out / f"{arm}_bcf_mc1_predictions.parquet", index=False, compression="zstd")
        target_free, outcome = base._combine_predictions(current_pred, bcf_pred, labels)
        target_free.to_parquet(out / f"{arm}_target_free_admission.parquet", index=False, compression="zstd")
        metric, month = base._replay(target_free, outcome, arm, out)
        metric["period"] = f"{start:%Y-%m}..{(end - pd.Timedelta(days=1)):%Y-%m}"
        metrics.append(metric)
        monthly.append(month)
        audit = pd.concat([current_audit, bcf_audit], ignore_index=True)
        audit.insert(0, "arm", arm)
        audits.append(audit)
    summary = base._append_delta(pd.DataFrame(metrics), "C0_refit_core")
    summary.to_parquet(out / "portfolio_summary.parquet", index=False, compression="zstd")
    summary.to_csv(out / "portfolio_summary.csv", index=False)
    pd.concat(monthly, ignore_index=True).to_parquet(out / "monthly_metrics.parquet", index=False, compression="zstd")
    pd.concat(audits, ignore_index=True).to_parquet(out / "mc1_fold_audit.parquet", index=False, compression="zstd")
    coverage.assign(decision_month=pd.to_datetime(coverage["__decision_ts__"], utc=True).dt.strftime("%Y-%m")).groupby("decision_month", as_index=False).agg(
        rows=("candidate_id", "size"), sr_available=(base.SR_AVAILABLE, "sum")
    ).to_parquet(out / "sr_coverage.parquet", index=False, compression="zstd")
    manifest = {
        "schema": "causal-sr-c1-2025-mc1-replay-v1",
        "scope": "offline research-only; strict prequential C0/C1 comparison; no live mutation or exchange I/O",
        "history_start": history_start.isoformat(),
        "held_window": [start.isoformat(), end.isoformat()],
        "warmup": "April--May 2025 causal S/R snapshots score C1 MC1 training rows; June--December are held portfolio months",
        "sources": {
            "current": {"path": str(args.current.resolve()), "sha256": _sha256(args.current.resolve())},
            "bcf": {"path": str(args.bcf.resolve()), "sha256": _sha256(args.bcf.resolve())},
            "sr": {"path": str(args.sr_root.resolve()), "manifest_sha256": _sha256(args.sr_root.resolve() / "run_manifest.json")},
        },
        "arms": {name: list(extra) for name, extra in arms.items()},
        "target": "family-local rich parent policy_net_bps, clipped by training fold; 21-day prior-resolved score-band shift",
        "admission": "BCF MC1 >= +50 AND current-v5 MC1 >= +50; priority BCF MC1 EV",
        "portfolio": "controlled global 7x/10%-slot, 2-new, 8-concurrent, 80%-wallet auction; invalid outcomes excluded before capacity",
        "causality": "S/R source heads are monthly strict OOF; MC1 fit rows have labels resolved before held boundary; structural missingness is a model input, never a candidate filter",
        "no_exchange_calls": True,
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
