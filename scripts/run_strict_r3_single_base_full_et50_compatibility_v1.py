#!/usr/bin/env python3
"""Replay one frozen Router50 Base through the original full ET50 downstream.

This is an offline compatibility harness, not a new score family:

    frozen Router50 single-Base target-free scores
      -> original cap80/cap120 consensus + generic correctness
      -> full strict-prequential Current and BCF MC1 maps
      -> dual admission and the existing chronological portfolio.

The point is to compare a one-Base challenger with the historical full ET50
control without substituting the weaker R/U mini-MC1 diagnostic.  The adapter
must already have retained every Router-selected row; it is never permitted to
add a post-Base cutoff here.  This script is research-only and has no live or
exchange authority.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT, ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import run_strict_r3_router_routed_base_stack as full  # noqa: E402


def _months(start: str, end: str) -> tuple[pd.Timestamp, ...]:
    """Return UTC calendar-month starts for either ``YYYY-MM`` or date input.

    The CLI originally advertised month-oriented arguments but accepted full
    dates as well.  Appending ``-01`` unconditionally made, for example,
    ``2025-11-01`` parse as 01:00 UTC.  The downstream reuse audit correctly
    rejected that non-midnight boundary.  Normalize an already parsed input
    instead, preserving the intended inclusive month range.
    """
    def month_start(value: str) -> pd.Timestamp:
        stamp = pd.Timestamp(value)
        stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
        return stamp.normalize().replace(day=1)

    first = month_start(start)
    last = month_start(end)
    if last < first:
        raise ValueError("end month precedes start month")
    return tuple(pd.date_range(first, last, freq="MS", tz="UTC"))


def _write_exclusive(path: Path, payload: object) -> None:
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _validate_target_free(root: Path, base_months: tuple[pd.Timestamp, ...]) -> dict[str, object]:
    manifest_path = root / "run_manifest.json"
    correctness_path = root / "correctness_report.json"
    if not manifest_path.exists() or not correctness_path.exists():
        raise FileNotFoundError("single-Base target-free source lacks its manifest or correctness receipt")
    manifest = json.loads(manifest_path.read_text())
    correctness = json.loads(correctness_path.read_text())
    if manifest.get("schema") != "strict_r3_single_head_downstream_source_v1":
        raise AssertionError("unexpected single-Base target-free schema")
    if not correctness.get("single_head_only"):
        raise AssertionError("compatibility source is not a single-head Base")
    if not correctness.get("no_post_router_base_cutoff"):
        raise AssertionError("single-Base source reintroduced a forbidden post-Router Base cutoff")
    for month in base_months:
        panel = root / f"month={month:%Y-%m}" / "scores_features.parquet"
        if not panel.exists():
            raise FileNotFoundError(f"missing frozen single-Base month: {panel}")
    return manifest


def run(
    *, target_free_root: Path, router_root: Path, source_root: Path,
    labels_root: Path, policy_path: Path, out: Path, base_start: str,
    consensus_start: str, evaluation_start: str, evaluation_end: str,
    n_jobs: int,
) -> Path:
    if out.exists():
        raise FileExistsError(f"immutable output exists: {out}")
    base_months = _months(base_start, evaluation_end)
    consensus_months = _months(consensus_start, evaluation_end)
    evaluation_months = _months(evaluation_start, evaluation_end)
    if min(consensus_months) <= min(base_months):
        raise ValueError("consensus must begin after the first Base month")
    if min(evaluation_months) <= min(consensus_months):
        raise ValueError("evaluation must begin after consensus warm-up")
    manifest = _validate_target_free(target_free_root, base_months)
    parent_contract = target_free_root.parent / "run_contract.json"
    _write_exclusive(parent_contract, {
        "schema": "strict_r3_single_base_full_et50_compatibility_source_v1",
        "scope": "offline target-free compatibility source; no live or exchange mutation",
        "base_contract": {
            "train_population": "router-selected rows only; labels resolved before same-model 28-day reserve",
            "base_components": "single_head",
            "r3_direct_ranking_authority": False,
            "base_bps_semantics": "frozen_single_head_score_adapter",
            "router_outputs_as_base_inputs": False,
            "router_outputs_persisted_for_downstream_ablation": False,
            "router_output_fields": [],
            "minimum_routed_train_rows": None,
            "reused_base_lineage": str(target_free_root / "run_manifest.json"),
        },
        "adapter_manifest": manifest,
        "base_months": [f"{value:%Y-%m}" for value in base_months],
        "consensus_months": [f"{value:%Y-%m}" for value in consensus_months],
        "evaluation_months": [f"{value:%Y-%m}" for value in evaluation_months],
    })
    original_base = full.ROUTED_BASE_MONTHS
    original_consensus = full.CONSENSUS_SCORE_MONTHS
    original_evaluation = full.EVALUATION_MONTHS
    try:
        full.ROUTED_BASE_MONTHS = base_months
        full.CONSENSUS_SCORE_MONTHS = consensus_months
        full.EVALUATION_MONTHS = evaluation_months
        full.run(
            router_root=router_root.resolve(), source_root=source_root.resolve(),
            labels_root=labels_root.resolve(), policy_path=policy_path.resolve(),
            out=out.resolve(), route_fraction=.50, thresholds=(50.0,), n_jobs=int(n_jobs),
            reuse_target_free=target_free_root.resolve(), reuse_score_root=None,
            base_components="et",
        )
    finally:
        full.ROUTED_BASE_MONTHS = original_base
        full.CONSENSUS_SCORE_MONTHS = original_consensus
        full.EVALUATION_MONTHS = original_evaluation
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-free-root", type=Path, required=True)
    parser.add_argument("--router-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base-start", default="2025-11")
    parser.add_argument("--consensus-start", default="2026-02")
    parser.add_argument("--evaluation-start", default="2026-05")
    parser.add_argument("--evaluation-end", default="2026-07")
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()
    print(run(
        target_free_root=args.target_free_root, router_root=args.router_root,
        source_root=args.source_root, labels_root=args.labels_root,
        policy_path=args.policy, out=args.out, base_start=args.base_start,
        consensus_start=args.consensus_start, evaluation_start=args.evaluation_start,
        evaluation_end=args.evaluation_end, n_jobs=args.n_jobs,
    ))


if __name__ == "__main__":
    main()
