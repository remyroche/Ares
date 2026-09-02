#!/usr/bin/env python3
"""Hard-link a contiguous, target-free P8U Meta feature-owner bridge.

This is deliberately lineage-only.  It never opens labels, outcomes, MC1,
portfolio, live, or exchange state.  Each month must have exactly one source
feature owner and every owner must expose the same ordered parquet schema.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


SCHEMA = "strict_r3_p8u_meta_feature_bridge_v1"
IDENTITY = ("candidate_id", "__decision_ts__", "side_name")
FORBIDDEN_PREFIXES = ("policy_", "supportive_", "h12_", "label_", "outcome_")
FORBIDDEN_EXACT = {"mfe", "mae"}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _once(path: Path, payload: object) -> None:
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _months(text: str) -> tuple[pd.Timestamp, ...]:
    values = tuple(pd.Timestamp(f"{token.strip()}-01", tz="UTC") for token in text.split(",") if token.strip())
    if not values or len(values) != len(set(values)) or tuple(sorted(values)) != values:
        raise ValueError("--months must be ordered unique YYYY-MM values")
    if tuple(pd.date_range(values[0], values[-1], freq="MS", tz="UTC")) != values:
        raise ValueError("--months must be contiguous")
    return values


def _link(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--months", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    months = _months(args.months)
    roots = tuple(path.resolve() for path in args.source_roots)
    if len(roots) != len(set(roots)):
        raise ValueError("duplicate feature source root")
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)

    expected_schema = None
    receipts: list[dict[str, object]] = []
    for month in months:
        token = f"month={month:%Y-%m}"
        sources = [root / token / "causal_feature_universe.parquet" for root in roots]
        owners = [path for path in sources if path.is_file()]
        if len(owners) != 1:
            raise AssertionError(f"{month:%Y-%m}: expected one feature owner, found {len(owners)}")
        source = owners[0]
        schema = pq.ParquetFile(source).schema_arrow
        if tuple(schema.names[:3]) != IDENTITY:
            raise AssertionError(f"{source}: identity columns/order drift")
        bad = [name for name in schema.names if name in FORBIDDEN_EXACT or name.lower().startswith(FORBIDDEN_PREFIXES)]
        if bad:
            raise AssertionError(f"{source}: target/outcome fields in feature owner {bad[:8]}")
        if expected_schema is None:
            expected_schema = schema
        elif not expected_schema.equals(schema, check_metadata=False):
            raise AssertionError(f"{month:%Y-%m}: feature schema/order differs from earlier owner")
        target = out / token / "causal_feature_universe.parquet"
        mode = _link(source, target)
        receipts.append({
            "month": f"{month:%Y-%m}", "source": str(source), "source_sha256": _sha(source),
            "linked_path": str(target), "rows": int(pq.ParquetFile(source).metadata.num_rows),
            "mode": mode, "target_free": True,
        })

    _once(out / "run_manifest.json", {
        "schema": SCHEMA,
        "scope": "offline target-free P8U Meta feature-owner bridge only; no labels/outcomes/MC1/portfolio/live/exchange mutation",
        "months": [f"{month:%Y-%m}" for month in months], "source_roots": [str(root) for root in roots],
        "feature_count": int(len(expected_schema.names) - len(IDENTITY)) if expected_schema else 0,
        "source_receipts": receipts,
        "correctness": {
            "contiguous_single_owner_months": True,
            "feature_schema_and_order_exact_across_sources": True,
            "target_free_feature_sources_only": True,
            "hardlinks_preserve_source_bytes_when_available": True,
            "no_labels_outcomes_mc1_portfolio_live_or_exchange_mutation": True,
        },
    })
    print(out)


if __name__ == "__main__":
    main()
