#!/usr/bin/env python3
"""Audit captured P8U primitives against the exact canonical source builder.

This is an offline, target-free diagnostic.  It rebuilds the canonical raw
panel for the manifest's full universe, applies the same canonical OI,
funding, order-book and frozen-input adapters, then compares the primitives at
one decision timestamp with a captured append-only panel.  It never publishes
feature state or modifies inference/live contracts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_tp6_sl4_exact170_canonical_consensus import (  # noqa: E402
    _add_frozen_input_backfill,
    _add_oi_funding_panels,
    _add_orderbook_panels,
    _make_panel,
)


def _utc(raw: str) -> pd.Timestamp:
    value = pd.Timestamp(raw)
    return value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _metric(
    canonical: pd.DataFrame,
    captured: pd.DataFrame,
    *,
    stamp: pd.Timestamp,
    symbols: list[str],
) -> dict[str, object]:
    left = canonical.reindex(index=[stamp], columns=symbols).to_numpy(dtype=float).ravel()
    right = captured.reindex(index=[stamp], columns=symbols).to_numpy(dtype=float).ravel()
    finite = np.isfinite(left) & np.isfinite(right)
    exact = np.isclose(left, right, rtol=1e-6, atol=1e-6, equal_nan=True)
    nan_pair = np.isnan(left) & np.isnan(right)
    return {
        "exact_cells": int(exact.sum()),
        "mismatch_cells": int((~exact).sum()),
        "finite_mismatch_cells": int((finite & ~exact).sum()),
        "missing_mismatch_cells": int(((~finite) & ~nan_pair).sum()),
        "max_abs_delta": float(np.nanmax(np.abs(left[finite] - right[finite]))) if finite.any() else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument("--captured-panel", type=Path, required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end-exclusive", required=True)
    parser.add_argument("--signal-ts", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output exists: {args.out_dir}")
    manifest = json.loads(args.canonical_manifest.read_text())
    symbols = [str(value) for value in manifest.get("symbols", [])]
    if not symbols or len(set(symbols)) != len(symbols):
        raise ValueError("canonical manifest lacks a unique full symbols universe")
    start, end, stamp = (_utc(args.start), _utc(args.end_exclusive), _utc(args.signal_ts))
    if not start <= stamp < end:
        raise ValueError("signal timestamp must be inside canonical source interval")

    canonical, source_map = _make_panel(symbols, start, end, allow_minute_fallback=False, bar_phase_minutes=0)
    index = canonical["close"].index
    _add_orderbook_panels(canonical, symbols, index, start, end)
    _add_oi_funding_panels(canonical, symbols, index, start, end)
    _add_frozen_input_backfill(canonical, symbols, index, start, end)

    captured_state = joblib.load(args.captured_panel)
    captured = captured_state.get("panel")
    if not isinstance(captured, dict):
        raise ValueError("captured panel has no raw panel mapping")
    shared = sorted(set(canonical).intersection(captured))
    records = []
    for primitive in shared:
        left, right = canonical[primitive], captured[primitive]
        if not isinstance(left, pd.DataFrame) or not isinstance(right, pd.DataFrame):
            continue
        records.append({"primitive": primitive, **_metric(left, right, stamp=stamp, symbols=symbols)})
    audit = pd.DataFrame(records).sort_values(["mismatch_cells", "primitive"], ascending=[False, True])
    args.out_dir.mkdir(parents=True)
    audit.to_parquet(args.out_dir / "primitive_parity.parquet", index=False, compression="zstd")
    summary = {
        "schema": "strict_r3_p8u_canonical_raw_source_parity_v1",
        "status": "pass" if not int(audit.mismatch_cells.sum()) else "fail",
        "signal_ts": stamp.isoformat(),
        "canonical_symbols": len(symbols),
        "source_map_matches_manifest": source_map == manifest.get("source_map"),
        "source_map_matches_captured": source_map == captured_state.get("source_map"),
        "shared_primitives": int(len(audit)),
        "mismatch_cells": int(audit.mismatch_cells.sum()),
        "failing_primitives": audit.loc[audit.mismatch_cells.gt(0), "primitive"].tolist(),
        "max_abs_delta": float(audit.max_abs_delta.max()) if len(audit) else 0.0,
        "canonical_manifest_sha256": _sha256(args.canonical_manifest),
        "captured_panel_sha256": _sha256(args.captured_panel),
        "outcome_columns_consumed": [],
        "state_bundle_published": False,
    }
    with (args.out_dir / "summary.json").open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
