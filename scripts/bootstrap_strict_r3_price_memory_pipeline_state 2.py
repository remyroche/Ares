#!/usr/bin/env python3
"""Bootstrap the exact raw-to-price-memory state from a sealed source panel."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.inference.price_memory_pipeline_state import (  # noqa: E402
    PriceMemoryPipelineState,
)
from extreme_price_movements.inference.causal_feature_output_state import (  # noqa: E402
    CausalFeatureOutputState,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-state", type=Path, required=True)
    parser.add_argument("--through", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable bootstrap output exists: {args.out_dir}")
    started = time.perf_counter()
    source = joblib.load(args.panel_state)
    panel = source.get("panel")
    symbols = tuple(map(str, source.get("symbols") or ()))
    if not symbols or not isinstance(panel, dict):
        raise ValueError("panel state lacks frozen symbols/panel")
    through = pd.Timestamp(args.through)
    if through.tzinfo is None:
        through = through.tz_localize("UTC")
    else:
        through = through.tz_convert("UTC")
    index = pd.DatetimeIndex(panel["close"].index)
    index = index[index <= through]
    if len(index) == 0 or index[-1] != through:
        raise ValueError("panel does not contain the requested terminal hour")
    if not index.equals(pd.date_range(index[0], index[-1], freq="h", tz="UTC")):
        raise ValueError("price-memory bootstrap panel is not hourly-contiguous")
    raw_keys = ["open", "high", "low", "close", "volume"]
    if isinstance(panel.get("quote_volume"), pd.DataFrame):
        raw_keys.append("quote_volume")
    arrays = {
        key: panel[key].reindex(index=index, columns=list(symbols)).to_numpy(
            dtype=np.float32, copy=False
        )
        for key in raw_keys
    }
    state_dir = args.out_dir / "states"
    pipeline = PriceMemoryPipelineState(cache_dir=state_dir, symbols=symbols)
    feature_keys = tuple(pipeline.features._OUTPUTS)
    raw_feature_history = {
        key: np.empty((len(index), len(symbols)), dtype=np.float32)
        for key in feature_keys
    }
    latest = None
    for position, timestamp in enumerate(index):
        latest = pipeline.update(
            {key: value[position] for key, value in arrays.items()},
            timestamp=timestamp,
        )
        for key in feature_keys:
            raw_feature_history[key][position, :] = latest[key]
    pipeline.snapshot()
    if latest is None:
        raise AssertionError("price-memory bootstrap emitted no state")
    output_state = CausalFeatureOutputState(
        feature_keys=feature_keys,
        symbols=symbols,
    )
    output_state.bootstrap(raw_feature_history, index=index)
    output_state_path = state_dir / "direct_causal_output_state.json"
    output_state.save(output_state_path)
    rows = []
    for name, values in latest.items():
        rows.extend(
            {
                "timestamp": through,
                "symbol": symbol,
                "feature": name,
                "value": float(values[position]),
            }
            for position, symbol in enumerate(symbols)
        )
    pd.DataFrame(rows).to_parquet(
        args.out_dir / "latest_pretransform_state_outputs.parquet", index=False
    )
    state_files = sorted(path for path in state_dir.iterdir() if path.is_file())
    manifest = {
        "schema": "strict_r3_price_memory_pipeline_bootstrap_v1",
        "panel_state": str(args.panel_state),
        "panel_state_sha256": _sha256(args.panel_state),
        "history_start": index[0].isoformat(),
        "through": through.isoformat(),
        "rows": int(len(index)),
        "symbols": int(len(symbols)),
        "runtime_seconds": float(time.perf_counter() - started),
        "state_sha256": {path.name: _sha256(path) for path in state_files},
        "direct_causal_output_state": str(output_state_path),
        "direct_causal_output_fields": list(feature_keys),
        "outcome_columns_consumed": [],
    }
    (args.out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
