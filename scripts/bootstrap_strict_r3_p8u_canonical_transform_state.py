#!/usr/bin/env python3
"""Bootstrap a private, exact P8U canonical transform state.

The historical canonical source panel is read only.  We replay it strictly
chronologically through the same canonical adapter that will be used by the
future warm process.  The ordinary mode uses bounded overlapping raw tails.
``exact_full`` is a one-time cold-start mode that materialises the complete
already-retained target-free source panel, so persisted operators are seeded
from the canonical graph rather than from a truncated initial chunk.  Both
modes leave future hourly inference bounded and do not use outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.inference.p8u_canonical_feature_adapter import (  # noqa: E402
    canonical_features_from_saved_panel,
)
from extreme_price_movements.inference.p8u_warm_feature_state import (  # noqa: E402
    atomic_json,
    sha256_file,
)


def _utc(value: str) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _feature_fields(path: Path) -> list[str]:
    payload = json.loads(path.read_text())
    values = payload.get("full_union") if isinstance(payload, dict) else None
    if not isinstance(values, list) or not values:
        raise ValueError("feature plan has no full_union")
    output = [str(value) for value in values]
    if len(set(output)) != len(output):
        raise ValueError("feature plan has duplicate fields")
    return output


def _tail_panel(
    panel: dict[str, Any], *, start: pd.Timestamp, end: pd.Timestamp
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in panel.items():
        if isinstance(value, pd.DataFrame):
            output[key] = value.loc[(value.index >= start) & (value.index < end)].copy(deep=False)
        else:
            output[key] = value
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-plan", type=Path, required=True)
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument("--source-panel", type=Path, required=True)
    parser.add_argument("--history-start", required=True)
    parser.add_argument("--end-exclusive", required=True)
    parser.add_argument("--state-scope", required=True)
    parser.add_argument("--tail-hours", type=int, default=1536)
    parser.add_argument("--chunk-hours", type=int, default=720)
    parser.add_argument(
        "--initial-seed",
        choices=("bounded", "exact_full"),
        default="bounded",
        help=(
            "How to establish the private state before bounded append-only use. "
            "exact_full consumes the already-retained target-free source once; "
            "it does not change future runtime history or feature equations."
        ),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable state bootstrap exists: {args.out_dir}")
    if args.tail_hours < 1458:
        raise ValueError("tail-hours must cover the canonical 1458-bar fixed-FFD width")
    if args.chunk_hours < 1 or args.chunk_hours > args.tail_hours:
        raise ValueError("chunk-hours must be in [1, tail-hours]")
    start, end = _utc(args.history_start), _utc(args.end_exclusive)
    if end <= start:
        raise ValueError("end-exclusive must follow history-start")

    fields = _feature_fields(args.feature_plan)
    manifest = json.loads(args.canonical_manifest.read_text())
    symbols = tuple(sorted(dict.fromkeys(map(str, manifest.get("symbols", [])))))
    if not symbols:
        raise ValueError("canonical manifest has no full universe")
    loaded = joblib.load(args.source_panel)
    panel = loaded.get("panel")
    close = panel.get("close") if isinstance(panel, dict) else None
    if not isinstance(close, pd.DataFrame):
        raise ValueError("source state lacks a close panel")
    index = pd.DatetimeIndex(close.index)
    active_index = index[(index >= start) & (index < end)]
    expected = pd.date_range(start, end - pd.Timedelta(hours=1), freq="h", tz="UTC")
    if not active_index.equals(expected):
        raise ValueError("canonical source history is not a complete hourly range")

    args.out_dir.mkdir(parents=True)
    state_dir = args.out_dir / "state"
    rows: list[dict[str, object]] = []
    cursor = 0
    started = time.monotonic()
    if args.initial_seed == "exact_full":
        # Some nested producers feed the causal transform through histories
        # that cannot be recreated from a fresh short first chunk. Seed the
        # state from the complete retained source once; all later calls remain
        # bounded append-only and use the same feature equations.
        seed_started = time.monotonic()
        full = _tail_panel(panel, start=start, end=end)
        canonical_features_from_saved_panel(
            full,
            universe_symbols=symbols,
            requested_features=fields,
            # The persisted state must be seeded from the same full causal
            # graph that created the canonical reference.  A reduced output
            # projection can omit intermediate parents and produce a subtly
            # different history for a selected field, even when the selected
            # timestamp itself is available.  Normal inference remains a
            # bounded append against this compact state.
            full_config_causal_universe=True,
            state_dir=state_dir,
            state_scope=args.state_scope,
            state_contract_features=fields,
            # Seed every compatible stateful operator from the full canonical
            # graph. The selected pass below narrows staged parent histories
            # to its own active contract before future bounded appends.
            exact_causal_transform_state_seed=True,
        )
        # The broad graph seeds every transform field whose raw parents it
        # materialises. A few sealed selected fields are adapter aliases that
        # are absent from that broad output. Replay the same target-free
        # history through the selected graph to seed only those missing rows;
        # existing broad-graph state remains unchanged.
        canonical_features_from_saved_panel(
            full,
            universe_symbols=symbols,
            requested_features=fields,
            full_config_causal_universe=False,
            state_dir=state_dir,
            state_scope=args.state_scope,
            state_contract_features=fields,
            exact_causal_transform_state_seed=True,
        )
        rows.append({
            "chunk_start": start.isoformat(),
            "chunk_end_exclusive": end.isoformat(),
            "new_rows": int(len(active_index)),
            "input_rows": int(len(full["close"].index)),
            "runtime_seconds": time.monotonic() - seed_started,
            "kind": "exact_full_initial_seed",
        })
        cursor = len(active_index)
    while cursor < len(active_index):
        # The initial call starts at the canonical history boundary.  Later
        # calls retain only the declared raw tail; all older influence must
        # arrive through saved transform state.
        advance = min(args.chunk_hours if cursor else args.tail_hours, len(active_index) - cursor)
        next_cursor = cursor + advance
        chunk_end = active_index[next_cursor - 1] + pd.Timedelta(hours=1)
        chunk_start = max(start, chunk_end - pd.Timedelta(hours=args.tail_hours))
        chunk = _tail_panel(panel, start=chunk_start, end=chunk_end)
        row_started = time.monotonic()
        canonical_features_from_saved_panel(
            chunk,
            universe_symbols=symbols,
            requested_features=fields,
            full_config_causal_universe=False,
            state_dir=state_dir,
            state_scope=args.state_scope,
            state_contract_features=fields,
        )
        rows.append({
            "chunk_start": chunk_start.isoformat(),
            "chunk_end_exclusive": chunk_end.isoformat(),
            "new_rows": int(advance),
            "input_rows": int(len(chunk["close"].index)),
            "runtime_seconds": time.monotonic() - row_started,
        })
        cursor = next_cursor

    receipt = {
        "schema": "strict_r3_p8u_canonical_transform_state_bootstrap_v1",
        "status": "bootstrapped_unactivated",
        "history_start": start.isoformat(),
        "end_exclusive": end.isoformat(),
        "symbols": len(symbols),
        "features": len(fields),
        "tail_hours": int(args.tail_hours),
        "chunk_hours": int(args.chunk_hours),
        "initial_seed": str(args.initial_seed),
        "state_scope": str(args.state_scope),
        "chunks": rows,
        "runtime_seconds": time.monotonic() - started,
        "source_panel_sha256": sha256_file(args.source_panel),
        "feature_plan_sha256": sha256_file(args.feature_plan),
        "canonical_manifest_sha256": sha256_file(args.canonical_manifest),
        "outcome_columns_consumed": [],
        "state_bundle_published": False,
    }
    atomic_json(args.out_dir / "receipt.json", receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
