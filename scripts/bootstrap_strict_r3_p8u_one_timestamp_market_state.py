#!/usr/bin/env python3
"""Bootstrap the direct P8U market/regime state from an immutable source panel.

This is deliberately narrow: it establishes the target-free common
market/regime node used by the future one-timestamp P8U graph.  It does not
compute model features, train a model, read outcomes, or submit an order.
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
    P8UMarketRegimeSnapshotState,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-state", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    source_path = Path(args.source_state).resolve()
    out_dir = Path(args.out_dir).resolve()
    if ROOT not in source_path.parents or ROOT not in out_dir.parents:
        raise ValueError("P8U paths must remain below repository root")
    if out_dir.exists():
        raise FileExistsError(f"immutable direct-state output exists: {out_dir}")
    payload = joblib.load(source_path)
    if not isinstance(payload, dict):
        raise TypeError("P8U source state is not a mapping")
    panel = payload.get("panel")
    # The immutable append-only source state calls this field ``symbols``;
    # model/feature states use ``universe_symbols``.  Accept only those two
    # explicit schema aliases, never infer a universe from an arbitrary panel.
    universe = tuple(map(str, payload.get("symbols") or payload.get("universe_symbols") or ()))
    if not isinstance(panel, dict) or len(universe) != 160:
        raise ValueError("source state lacks the sealed 160-symbol panel")
    close = panel.get("close")
    if not isinstance(close, pd.DataFrame) or tuple(map(str, close.columns)) != universe:
        raise ValueError("source state close panel does not match its sealed universe")
    source_sha = _sha256(source_path)
    started = time.perf_counter()
    state = P8UMarketRegimeSnapshotState(symbols=universe, market_basket=universe)
    output = state.bootstrap(panel)
    elapsed = time.perf_counter() - started
    out_dir.mkdir(parents=True)
    state_path = out_dir / "market_regime_state.npz"
    state.save(state_path)
    # Only the latest target-free row is retained as a bootstrap diagnostic;
    # model-feature values are intentionally not emitted by this narrow node.
    latest = pd.DataFrame(
        {name: values[-1] for name, values in output.items()},
        index=pd.Index(universe, name="symbol"),
    )
    latest.to_parquet(out_dir / "latest_market_regime_outputs.parquet", compression="zstd")
    receipt = {
        "schema": "strict_r3_p8u_one_timestamp_market_bootstrap_v1",
        "status": "pass_target_free_partial_node",
        "source_state": str(source_path),
        "source_state_sha256": source_sha,
        "symbols": len(universe),
        "source_rows": int(len(close)),
        "source_start": pd.Timestamp(close.index[0]).isoformat(),
        "source_end": pd.Timestamp(close.index[-1]).isoformat(),
        "state_path": str(state_path),
        "state_sha256": _sha256(state_path),
        "state_contract_hash": state.contract_hash,
        "latest_timestamp": state.last_timestamp,
        "outputs": list(P8UMarketRegimeSnapshotState.OUTPUTS),
        "runtime_seconds": elapsed,
        "outcome_columns_consumed": [],
        "model_features_emitted": False,
    }
    (out_dir / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
