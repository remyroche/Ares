#!/usr/bin/env python3
"""Bootstrap an append-only P8U source state using the canonical builder.

This source-only bootstrap is the safe starting point for a long-lived P8U
feature process.  It intentionally invokes the exact canonical source
functions (coarse OHLCV precedence plus order-book/OI/funding/frozen-input
adapters) before persisting the full contemporaneous universe.  It is
offline, target-free, immutable, and does not activate an inference bundle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import joblib
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
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument("--history-start", required=True)
    parser.add_argument("--end-exclusive", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable canonical source output exists: {args.out_dir}")
    manifest = json.loads(args.canonical_manifest.read_text())
    symbols = [str(value) for value in manifest.get("symbols", [])]
    if not symbols or len(set(symbols)) != len(symbols):
        raise ValueError("canonical manifest lacks a unique full symbols universe")
    start, end = _utc(args.history_start), _utc(args.end_exclusive)
    if end <= start:
        raise ValueError("end-exclusive must follow history-start")

    panel, source_map = _make_panel(symbols, start, end, allow_minute_fallback=False, bar_phase_minutes=0)
    # A reconstructed successor manifest is permissible only when it carries
    # the predecessor's exact source mapping.  The generic canonical builder
    # derives its local archive mapping from available directories; verify
    # that derivation instead of silently accepting a historical alias change.
    expected_source_map = manifest.get("source_map")
    if expected_source_map is not None:
        expected = {str(key): (None if value is None else str(value)) for key, value in expected_source_map.items()}
        actual = {str(key): (None if value is None else str(value)) for key, value in source_map.items()}
        if expected != actual:
            mismatched = [
                symbol for symbol in symbols
                if expected.get(symbol) != actual.get(symbol)
            ]
            raise ValueError(
                "canonical source-map derivation differs from the frozen manifest: "
                f"{mismatched[:8]}"
            )
    index = panel["close"].index
    _add_orderbook_panels(panel, symbols, index, start, end)
    _add_oi_funding_panels(panel, symbols, index, start, end)
    _add_frozen_input_backfill(panel, symbols, index, start, end)

    args.out_dir.mkdir(parents=True)
    state = {
        "schema": "strict_r3_p8u_canonical_source_panel_state_v1",
        "history_start": start,
        "end_exclusive": end,
        "symbols": symbols,
        "source_map": source_map,
        "canonical_manifest_sha256": _sha256(args.canonical_manifest),
        "source_map_identity_verified": expected_source_map is not None,
        "panel": panel,
    }
    target = args.out_dir / "source_panel_state.joblib"
    temporary = args.out_dir / ".source_panel_state.tmp.joblib"
    joblib.dump(state, temporary, compress=3)
    os.replace(temporary, target)
    receipt = {
        "schema": "strict_r3_p8u_canonical_source_state_v1",
        "status": "source_bootstrapped_unactivated",
        "history_start": start.isoformat(),
        "end_exclusive": end.isoformat(),
        "symbols": len(symbols),
        "source_fields": sorted(panel),
        "source_panel_sha256": _sha256(target),
        "canonical_manifest_sha256": _sha256(args.canonical_manifest),
        "outcome_columns_consumed": [],
        "feature_state_published": False,
    }
    with (args.out_dir / "receipt.json").open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
