#!/usr/bin/env python3
"""Evaluate router fractions from an immutable target-free score receipt.

This deliberately does not refit a router.  It reuses the exact score panel,
then joins policy outcomes only for retrospective timestamp-level metrics.
That makes the 30--50% route Pareto screen cheap without changing a single
candidate score or letting label completeness influence routing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
import run_strict_r3_economic_recall_router as router  # noqa: E402


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _fractions(value: str) -> tuple[float, ...]:
    result = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not result or tuple(sorted(set(result))) != result or any(not 0 < item <= 1 for item in result):
        raise argparse.ArgumentTypeError("fractions must be unique ascending values in (0, 1]")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--policy", type=Path, default=router.DEFAULT_POLICY)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fractions", type=_fractions, default=(.30, .35, .40, .45, .50))
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite route-Pareto artifact: {args.out}")
    contract_path = args.source / "run_contract.json"
    contract = json.loads(contract_path.read_text())
    months = tuple(contract["months"])
    policy = pd.read_parquet(args.policy, columns=["candidate_id", "policy_path_valid", "policy_net_bps"])
    if policy["candidate_id"].duplicated().any():
        raise AssertionError("policy ledger has duplicate candidate identities")
    rows: list[dict[str, object]] = []
    timestamps: list[pd.DataFrame] = []
    source_hashes: dict[str, str] = {}
    for month in months:
        score_path = args.source / "target_free_scores" / f"month={month}.parquet"
        score = pd.read_parquet(score_path)
        prohibited = [column for column in score if "policy_" in column or "path" in column or "label" in column]
        if prohibited:
            raise AssertionError(f"source score receipt is not target-free: {prohibited}")
        metric_rows, timestamp_rows = router._metric_rows(
            score, policy, pd.Timestamp(f"{month}-01", tz="UTC"), args.fractions,
        )
        rows.extend(metric_rows)
        timestamps.append(timestamp_rows)
        source_hashes[month] = _sha(score_path)
    args.out.mkdir(parents=True)
    metrics = pd.DataFrame(rows)
    metrics.to_parquet(args.out / "router_metrics.parquet", index=False, compression="zstd")
    pd.concat(timestamps, ignore_index=True).to_parquet(args.out / "router_timestamp_metrics.parquet", index=False, compression="zstd")
    router._aggregate_metrics(metrics).to_parquet(args.out / "router_metric_summary.parquet", index=False, compression="zstd")
    (args.out / "run_manifest.json").write_text(json.dumps({
        "schema": "strict_r3_router_route_pareto_v1", "scope": "research-only; no live mutation",
        "source": str(args.source), "source_contract_sha256": _sha(contract_path),
        "target_free_score_sha256_by_month": source_hashes, "policy": str(args.policy),
        "fractions": list(args.fractions), "months": months,
        "causality": "scores are reused unchanged; outcomes are joined only after timestamp-local routing",
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
