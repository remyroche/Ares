#!/usr/bin/env python3
"""Bootstrap P8U's direct price-memory and transform state target-free.

This is one prerequisite node of the single-timestamp executor.  It consumes
only the sealed append-only primitive source panel and persists sufficient
state for later one-hour updates; it neither calls the broad feature graph nor
emits a score or order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd

from extreme_price_movements.inference.p8u_single_timestamp_graph import (
    P8UPriceMemoryCausalState,
)


TRANSFORM_KEYS = (
    "prior_volatility",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-state", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = _args()
    source_path = Path(args.source_state).resolve()
    out_dir = Path(args.out_dir).resolve()
    if ROOT not in source_path.parents or ROOT not in out_dir.parents:
        raise ValueError("P8U paths must remain below repository root")
    if out_dir.exists():
        raise FileExistsError(f"immutable P8U price-memory bootstrap exists: {out_dir}")
    payload = joblib.load(source_path)
    if not isinstance(payload, dict) or not isinstance(payload.get("panel"), dict):
        raise ValueError("source state lacks a target-free panel")
    panel = payload["panel"]
    symbols = tuple(map(str, payload.get("symbols") or payload.get("universe_symbols") or ()))
    close = panel.get("close")
    if not isinstance(close, pd.DataFrame) or len(symbols) != 160:
        raise ValueError("source state lacks the sealed P8U 160-symbol close panel")
    for field in P8UPriceMemoryCausalState.SOURCE_FIELDS:
        frame = panel.get(field)
        if not isinstance(frame, pd.DataFrame) or tuple(map(str, frame.columns)) != symbols:
            raise ValueError(f"source state lacks aligned price-memory field {field!r}")
    started = time.perf_counter()
    state = P8UPriceMemoryCausalState(symbols=symbols, transform_keys=TRANSFORM_KEYS)
    outputs = state.bootstrap(panel)
    runtime = time.perf_counter() - started
    out_dir.mkdir(parents=True)
    state_root = out_dir / "state"
    state.save(state_root)
    latest = pd.DataFrame(
        {
            name: values[-1]
            for name, values in outputs.items()
            if name.startswith("feature__")
        },
        index=pd.Index(symbols, name="symbol"),
    )
    latest.to_parquet(out_dir / "latest_transformed_outputs.parquet", compression="zstd")
    receipt = {
        "schema": "strict_r3_p8u_one_timestamp_price_memory_bootstrap_v1",
        "status": "pass_target_free_partial_node",
        "source_state": str(source_path),
        "source_state_sha256": _sha256(source_path),
        "symbols": len(symbols),
        "source_rows": int(len(close)),
        "source_start": pd.Timestamp(close.index[0]).isoformat(),
        "source_end": pd.Timestamp(close.index[-1]).isoformat(),
        "latest_timestamp": state.last_timestamp,
        "state_contract_hash": state.contract_hash,
        "transform_keys": list(TRANSFORM_KEYS),
        "raw_outputs": list(P8UPriceMemoryCausalState.RAW_OUTPUTS),
        "state_files": sorted(path.relative_to(out_dir).as_posix() for path in state_root.rglob("*") if path.is_file()),
        "runtime_seconds": runtime,
        "outcome_columns_consumed": [],
        "model_features_emitted": False,
        "direct_node_calls_broad_feature_graph": False,
    }
    (out_dir / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
