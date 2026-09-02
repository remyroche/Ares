#!/usr/bin/env python3
"""Rehydrate MC1 parent-score receipts from a sealed target-free history panel.

This is deliberately a storage-only adapter for historical O3-v2 research.
Some early target-free feature panels retain the exact current/BCF parent
coordinates used by their specialists, but not their original per-family
Parquet layout.  Reconstructing that layout here keeps a downstream MC1 test
on the *same score coordinate system* as the specialist inputs.

The adapter neither fits a model nor reads a policy/semantic outcome.  It
does not claim parity with later live score-family bundles: that comparison is
reported separately and must never be inferred from this history rehydration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT, ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import run_strict_r3_enhanced_base_live_stack_challenger as parent  # noqa: E402
import run_strict_r3_o3v2_target_funnel as target  # noqa: E402


SCHEMA = "strict_r3_o3v2_history_parent_adapter_v1"
FAMILIES = ("current", "bcf")
IDENTITY = ("candidate_id", "__decision_ts__", "side_name")
PARENT_COORDINATES = (
    "final_score",
    "base_rank42",
    "base_anchor_bps",
    "conditional_consensus_rank",
    "ordinary_shadow_consensus_rank",
    "upstream",
    "correctness_rank",
)
SOURCE_FIELDS = (
    "f1_enhanced_base_bps",
    *(f"f5_{family}_{field}" for family in FAMILIES for field in PARENT_COORDINATES),
)
PROHIBITED = frozenset(target.PROHIBITED_SCORE_COLUMNS) | frozenset({"__selection_target__"})


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _assert_target_free_schema(columns: Sequence[str]) -> None:
    leaked = sorted(PROHIBITED.intersection(columns))
    if leaked:
        raise AssertionError(f"history panel contains policy/semantic selection fields: {leaked}")
    missing = sorted(set((*IDENTITY, *SOURCE_FIELDS)) - set(columns))
    if missing:
        raise KeyError(f"history panel lacks parent-coordinate fields: {missing}")


def _route(frame: pd.DataFrame) -> pd.Series:
    """Apply the declared timestamp-local top-30% route deterministically."""
    return parent._exact_timestamp_top_fraction(frame, "enhanced_base_bps", parent.BASE_ROUTE)


def _family_receipt(frame: pd.DataFrame, family: str) -> pd.DataFrame:
    if family not in FAMILIES:
        raise ValueError(f"unknown parent family {family!r}")
    result = frame.loc[:, list(IDENTITY)].copy()
    result["enhanced_base_bps"] = pd.to_numeric(frame["f1_enhanced_base_bps"], errors="coerce")
    result["enhanced_base_routed"] = _route(result.assign(__decision_ts__=frame["__decision_ts__"]))
    for field in PARENT_COORDINATES:
        result[field] = pd.to_numeric(frame[f"f5_{family}_{field}"], errors="coerce")
    if result["candidate_id"].duplicated().any():
        raise AssertionError("duplicate candidate IDs in parent adapter receipt")
    if result.loc[:, ["enhanced_base_bps", *PARENT_COORDINATES]].isna().any().any():
        raise AssertionError(f"{family}: non-finite or missing parent coordinate in target-free history")
    leaked = sorted(PROHIBITED.intersection(result.columns))
    if leaked:
        raise AssertionError(f"{family}: adapter retained forbidden outcome fields: {leaked}")
    return result


def _write_json_exclusive(path: Path, payload: object) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def run(*, history_panel: Path, out: Path, months: Sequence[str]) -> None:
    if out.exists():
        raise FileExistsError(out)
    # Schema inspection must not materialise the 1.5m-row history panel.
    schema = set(pq.ParquetFile(history_panel).schema_arrow.names)
    _assert_target_free_schema(schema)
    parsed = tuple(pd.Timestamp(f"{month}-01", tz="UTC") for month in months)
    if not parsed:
        raise ValueError("at least one month is required")
    columns = [*IDENTITY, *SOURCE_FIELDS]
    panel = pd.read_parquet(history_panel, columns=columns)
    panel["__decision_ts__"] = pd.to_datetime(panel["__decision_ts__"], utc=True, errors="raise")
    if panel["candidate_id"].duplicated().any():
        raise AssertionError("history panel has duplicate candidate IDs")
    out.mkdir(parents=True)
    audits: list[dict[str, object]] = []
    for month in parsed:
        end = month + pd.offsets.MonthBegin(1)
        held = panel.loc[panel["__decision_ts__"].ge(month) & panel["__decision_ts__"].lt(end)].copy()
        if held.empty:
            raise AssertionError(f"{month:%Y-%m}: no target-free history rows")
        for family in FAMILIES:
            receipt = _family_receipt(held, family)
            path = out / "target_free_scores" / family / f"month={month:%Y-%m}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            receipt.to_parquet(path, index=False, compression="zstd")
            audits.append({
                "month": f"{month:%Y-%m}", "family": family, "rows": int(len(receipt)),
                "routed_rows": int(receipt["enhanced_base_routed"].sum()),
                "coordinate_complete_fraction": float(receipt.loc[:, ["enhanced_base_bps", *PARENT_COORDINATES]].notna().all(axis=1).mean()),
                "min_decision_ts": str(receipt["__decision_ts__"].min()),
                "max_decision_ts": str(receipt["__decision_ts__"].max()),
            })
    pd.DataFrame(audits).to_parquet(out / "adapter_audit.parquet", index=False, compression="zstd")
    _write_json_exclusive(out / "run_manifest.json", {
        "schema": SCHEMA,
        "scope": "target-free historical parent-coordinate adapter only; no fitting, policy join, admission, portfolio, live use, or score recalibration",
        "history_panel": str(history_panel), "history_panel_sha256": _hash(history_panel),
        "months": [f"{month:%Y-%m}" for month in parsed],
        "families": list(FAMILIES),
        "source_fields": list(SOURCE_FIELDS),
        "output_coordinates": list(PARENT_COORDINATES),
        "routing": "exact deterministic timestamp-local top 30 percent by sealed f1_enhanced_base_bps",
        "causality": {
            "source": "sealed target-free history only; schema rejects policy, semantic, and selection-target columns",
            "parent_coordinates": "rehydrated stored current/BCF coordinates used by the same historical specialist inputs",
            "limitation": "historical score-family coordinate only; not a live-bundle parity claim",
        },
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-panel", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--months", required=True, help="comma-separated YYYY-MM")
    args = parser.parse_args()
    run(history_panel=args.history_panel, out=args.out, months=tuple(token for token in args.months.split(",") if token))


if __name__ == "__main__":
    main()
