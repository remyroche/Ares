#!/usr/bin/env python3
"""Materialise schema-v2 target-free point-in-time candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.strict_r3_canonical_v2 import (  # noqa: E402
    CandidateSpec,
    SCHEMA,
    build_point_in_time_candidates,
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _list(path: Path, key: str) -> list[str]:
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        return [str(value) for value in payload]
    if key in payload:
        return [str(value) for value in payload[key]]
    if "fields" in payload:
        return [str(value) for value in payload["fields"]]
    raise ValueError(f"{path} has no {key!r} or 'fields' list")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market-panel", type=Path, required=True)
    parser.add_argument("--universe", type=Path, required=True)
    parser.add_argument("--feature-contract", type=Path, required=True)
    parser.add_argument("--cross-sectional-contract", type=Path, required=True)
    parser.add_argument("--spread-limit-bps", type=float, required=True)
    parser.add_argument("--required-feature-fraction", type=float, default=1.0)
    parser.add_argument("--sides", default="long,short")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output already exists: {args.out_dir}")
    features = _list(args.feature_contract, "fields")
    cross_sectional = _list(args.cross_sectional_contract, "sources")
    if args.universe.suffix.lower() == ".json":
        universe = _list(args.universe, "symbols")
    else:
        table = pd.read_csv(args.universe)
        column = "symbol" if "symbol" in table else table.columns[0]
        universe = table[column].dropna().astype(str).tolist()
    market = pd.read_parquet(args.market_panel)
    population, eligible, rejected = build_point_in_time_candidates(
        market,
        universe=universe,
        feature_fields=features,
        cross_sectional_sources=cross_sectional,
        spec=CandidateSpec(
            spread_limit_bps=float(args.spread_limit_bps),
            required_feature_fraction=float(args.required_feature_fraction),
            side_names=tuple(side.strip() for side in args.sides.split(",") if side.strip()),
        ),
    )
    args.out_dir.mkdir(parents=True)
    population.to_parquet(args.out_dir / "target_free_candidate_population.parquet", index=False, compression="zstd")
    eligible.to_parquet(args.out_dir / "eligible_candidates.parquet", index=False, compression="zstd")
    rejected.to_parquet(args.out_dir / "candidate_rejection_audit.parquet", index=False, compression="zstd")
    reason_audit = population.groupby(["side_name", "eligibility_reason"], as_index=False).agg(rows=("candidate_id", "size"))
    reason_audit.to_parquet(args.out_dir / "candidate_rejection_reason_summary.parquet", index=False)
    manifest = {
        "schema": f"{SCHEMA}_candidate_population",
        "market_panel": str(args.market_panel),
        "market_panel_sha256": _sha(args.market_panel),
        "universe_sha256": _sha(args.universe),
        "feature_contract_sha256": _sha(args.feature_contract),
        "cross_sectional_contract_sha256": _sha(args.cross_sectional_contract),
        "population_rows": len(population), "eligible_rows": len(eligible),
        "rejected_rows": len(rejected), "spread_limit_bps": args.spread_limit_bps,
        "target_free": True,
        "cross_sectional_order": "complete point-in-time universe before candidate filtering",
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"event": "complete", **manifest, "output": str(args.out_dir)}))


if __name__ == "__main__":
    main()
