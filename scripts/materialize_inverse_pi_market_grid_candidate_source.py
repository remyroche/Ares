#!/usr/bin/env python3
"""Materialize the separate Jan--Jul 2022 inverse-PI exact-1m request source.

This is an acquisition population, not a model handoff or a replay population.
It deliberately uses a fixed paired market grid so that the historical Kraken
PI contracts can be collected without applying a future-informed candidate
screen.  The bootstrap barrier exists solely to satisfy the request-stage path
contract; it is not current policy geometry and must not be used for economics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


SCHEMA = "inverse_pi_market_grid_candidate_source_v1"
EVIDENCE_SCOPE = "inverse_pi_market_grid_bootstrap_research"
POPULATION_LINEAGE = "jan_jul_2022_inverse_pi_market_grid_bootstrap_v1"
PRODUCT_LINEAGE = "kraken_inverse_pi_exact_product_binding_v1"
START = pd.Timestamp("2022-01-01T00:00:00Z")
END_EXCLUSIVE = pd.Timestamp("2022-08-01T00:00:00Z")
CADENCE = "1h"
# This is intentionally a bootstrap-only placeholder.  The exact-1m request
# stage carries it to define a download path, while later label materialization
# must replace it with an explicitly bound, side-parent geometry contract.
BOOTSTRAP_BARRIER_PCT = 0.02

INVERSE_PRODUCTS: tuple[tuple[str, str], ...] = (
    ("BTC/USD:BTC", "PI_XBTUSD"),
    ("ETH/USD:ETH", "PI_ETHUSD"),
    ("LTC/USD:LTC", "PI_LTCUSD"),
    ("XRP/USD:XRP", "PI_XRPUSD"),
    ("BCH/USD:BCH", "PI_BCHUSD"),
)
# The label-input contract constructs the signed geometry key as
# ``side_name + '__' + archetype_policy_key``.  With no causal archetype
# classifier in this acquisition grid, ``parent`` is the only valid key: it
# binds explicitly to ``long__parent`` and ``short__parent`` downstream.
SIDES: tuple[tuple[str, int, str], ...] = (
    ("long", 1, "parent"),
    ("short", -1, "parent"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_candidate_id(
    timestamp: pd.Timestamp, symbol: str, side_name: str, product_id: str
) -> str:
    value = "|".join(
        (
            POPULATION_LINEAGE,
            timestamp.isoformat(),
            symbol,
            side_name,
            product_id,
        )
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _month_chunks(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    chunks: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cursor = start
    while cursor < end:
        next_month = (
            cursor.tz_localize(None).to_period("M") + 1
        ).start_time.tz_localize("UTC")
        boundary = min(end, next_month)
        chunks.append((cursor, boundary))
        cursor = boundary
    return chunks


def build_population(start: pd.Timestamp = START, end_exclusive: pd.Timestamp = END_EXCLUSIVE) -> pd.DataFrame:
    """Build the paired, stable-order candidate population without I/O."""

    if start != START or end_exclusive != END_EXCLUSIVE:
        raise ValueError(
            "This versioned source has a fixed Jan--Jul 2022 interval; create a new "
            "population version rather than changing its dates."
        )
    timestamps = pd.date_range(start, end_exclusive, freq=CADENCE, inclusive="left")
    rows: list[dict[str, Any]] = []
    for timestamp in timestamps:
        for symbol, product_id in INVERSE_PRODUCTS:
            for side_name, side, policy_key in SIDES:
                rows.append(
                    {
                        "__ts__": timestamp,
                        "__symbol__": symbol,
                        "side_name": side_name,
                        "side": side,
                        "archetype_policy_key": policy_key,
                        "policy_archetype_assignment_source": "fixed_signed_side_parent_inverse_grid",
                        "__barrier_pct__": BOOTSTRAP_BARRIER_PCT,
                        "bootstrap_barrier_data_acquisition_only": True,
                        "selected_for_monitor": True,
                        "evidence_scope": EVIDENCE_SCOPE,
                        "candidate_population_lineage": POPULATION_LINEAGE,
                        "source_product_lineage": PRODUCT_LINEAGE,
                        "source_product_id": product_id,
                        "source_contract_family": "PI",
                        "source_candidate_id": _source_candidate_id(
                            timestamp, symbol, side_name, product_id
                        ),
                    }
                )
    frame = pd.DataFrame(rows)
    if frame["source_candidate_id"].duplicated().any():
        raise AssertionError("deterministic inverse-grid source identity collision")
    return frame


def run(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = output_dir / "candidate_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    population = build_population()
    source_records: list[dict[str, Any]] = []
    for start, end_exclusive in _month_chunks(START, END_EXCLUSIVE):
        shard = population.loc[
            (population["__ts__"] >= start) & (population["__ts__"] < end_exclusive)
        ].copy()
        path = shard_dir / f"candidates_{start.strftime('%Y%m')}.parquet"
        shard.to_parquet(path, index=False)
        source_records.append(
            {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "rows": int(len(shard)),
                "signal_start": start.isoformat(),
                "signal_end_exclusive": end_exclusive.isoformat(),
            }
        )
    expected_rows = len(pd.date_range(START, END_EXCLUSIVE, freq=CADENCE, inclusive="left")) * len(INVERSE_PRODUCTS) * len(SIDES)
    if len(population) != expected_rows:
        raise AssertionError("inverse-grid row count does not match its fixed contract")
    manifest = {
        "schema": SCHEMA,
        "status": "candidate_source_materialized_no_market_data_download",
        "evidence_scope": EVIDENCE_SCOPE,
        "promotion_eligible": False,
        "execution_parity_claim": False,
        "candidate_population_lineage": POPULATION_LINEAGE,
        "product_lineage": PRODUCT_LINEAGE,
        "population_contract": {
            "signal_start": START.isoformat(),
            "signal_end_exclusive": END_EXCLUSIVE.isoformat(),
            "cadence": CADENCE,
            "population": "paired_long_short_inverse_market_grid",
            "symbols": [symbol for symbol, _ in INVERSE_PRODUCTS],
            "products": [product for _, product in INVERSE_PRODUCTS],
            "sides": [side_name for side_name, _, _ in SIDES],
            "archetype_policy_key": "parent",
            "signed_side_parent_policy_keys": {
                side_name: f"{side_name}__{policy_key}"
                for side_name, _, policy_key in SIDES
            },
            "source_candidate_identity": "sha256(population_lineage|timestamp|symbol|side|product_id)",
        },
        "bootstrap_barrier": {
            "barrier_pct": BOOTSTRAP_BARRIER_PCT,
            "data_acquisition_only": True,
            "not_policy_geometry": True,
            "not_economic_label": True,
            "replacement_required_before_labels_or_replay": True,
        },
        "inverse_product_limitations": {
            "inverse_contracts": True,
            "collateral_varies_by_product": True,
            "notional_return_comparable_to_usd_linear_pf": False,
            "historical_spread_l2_available": False,
            "non_promotable_until_separate_economic_contract": True,
        },
        "rows": int(len(population)),
        "timestamps": int(population["__ts__"].nunique()),
        "distinct_symbols": int(population["__symbol__"].nunique()),
        "side_counts": {
            str(side): int(count)
            for side, count in population["side_name"].value_counts().sort_index().items()
        },
        "candidate_shards": source_records,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = run(args.output_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
