#!/usr/bin/env python3
"""Bootstrap an immutable strict-R3 order-book state from a sealed panel."""

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

from extreme_price_movements.inference.orderbook_feature_state import (  # noqa: E402
    OrderbookFeatureState,
)
from extreme_price_movements.inference.causal_feature_output_state import (  # noqa: E402
    CausalFeatureOutputState,
)
from extreme_price_movements.config import CFG  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _panel_array(panel: dict, name: str, index: pd.DatetimeIndex, symbols: list[str]) -> np.ndarray:
    frame = panel.get(name)
    if not isinstance(frame, pd.DataFrame):
        raise ValueError(f"sealed panel lacks {name!r}")
    return frame.reindex(index=index, columns=symbols).to_numpy(dtype=np.float32, copy=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-state", type=Path, required=True)
    parser.add_argument("--through", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--microstructure-shift-bars", type=int, default=1)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable bootstrap output exists: {args.out_dir}")
    started = time.perf_counter()
    source = joblib.load(args.panel_state)
    panel = source.get("panel")
    symbols = list(map(str, source.get("symbols") or ()))
    if not isinstance(panel, dict) or not symbols:
        raise ValueError("panel state lacks frozen panel/symbol contract")
    through = pd.Timestamp(args.through)
    through = through.tz_localize("UTC") if through.tzinfo is None else through.tz_convert("UTC")
    close = panel.get("close")
    if not isinstance(close, pd.DataFrame):
        raise ValueError("sealed panel lacks close")
    index = pd.DatetimeIndex(close.index)
    index = index[index <= through]
    if len(index) == 0 or index[-1] != through:
        raise ValueError("sealed panel does not include requested terminal hour")
    if not index.equals(pd.date_range(index[0], index[-1], freq="h", tz="UTC")):
        raise ValueError("order-book bootstrap source must be hourly contiguous")
    field_map = {
        "best_bid": "orderbook_best_bid",
        "best_ask": "orderbook_best_ask",
        "mid": "orderbook_mid",
        "bid_qty_1": "orderbook_bid_qty_1",
        "ask_qty_1": "orderbook_ask_qty_1",
        "cum_bid_qty_l20": "orderbook_cum_bid_qty_l20",
        "cum_ask_qty_l20": "orderbook_cum_ask_qty_l20",
        "mean_trade_qty_1h": "orderbook_mean_trade_qty_1h",
        "close": "close",
        "volume": "volume",
    }
    arrays = {
        output: _panel_array(panel, source_name, index, symbols)
        for output, source_name in field_map.items()
    }
    # The panel source is already the canonical provider-validity/missingness
    # representation.  Explicit all-true does not weaken it: existing NaNs are
    # retained and become ffill inputs exactly as in the batch generator.
    arrays["source_valid"] = np.ones((len(index), len(symbols)), dtype=np.bool_)
    state = OrderbookFeatureState(
        symbols=symbols,
        market_basket=list(CFG.get("market_basket", ()) or ()),
        microstructure_shift_bars=args.microstructure_shift_bars,
    )
    outputs = state.update_batch(arrays, timestamps=index)
    state_dir = args.out_dir / "states"
    state_path = state_dir / "orderbook_feature_state.npz"
    state.save(state_path)
    transform_state = CausalFeatureOutputState(
        feature_keys=state.OUTPUTS,
        symbols=symbols,
    )
    transformed_outputs = transform_state.bootstrap(outputs, index=index)
    transform_state_path = state_dir / "direct_causal_output_state.json"
    transform_state.save(transform_state_path)
    raw_latest_rows = []
    transformed_latest_rows = []
    for field, matrix in outputs.items():
        raw_latest_rows.extend(
            {
                "timestamp": through,
                "symbol": symbol,
                "feature": field,
                "value": float(matrix[-1, position]),
            }
            for position, symbol in enumerate(symbols)
        )
    for field, matrix in transformed_outputs.items():
        transformed_latest_rows.extend(
            {
                "timestamp": through,
                "symbol": symbol,
                "feature": field,
                "value": float(matrix[-1, position]),
            }
            for position, symbol in enumerate(symbols)
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(raw_latest_rows).to_parquet(
        args.out_dir / "latest_precomposite_orderbook_outputs.parquet", index=False
    )
    pd.DataFrame(transformed_latest_rows).to_parquet(
        args.out_dir / "latest_transformed_orderbook_outputs.parquet", index=False
    )
    # Backward-compatible diagnostic alias; new consumers should use the
    # explicitly named pre-composite or transformed receipt above.
    pd.DataFrame(transformed_latest_rows).to_parquet(
        args.out_dir / "latest_pretransform_orderbook_outputs.parquet", index=False
    )
    manifest = {
        "schema": "strict_r3_orderbook_feature_state_bootstrap_v1",
        "panel_state": str(args.panel_state),
        "panel_state_sha256": _sha256(args.panel_state),
        "history_start": index[0].isoformat(),
        "through": through.isoformat(),
        "rows": int(len(index)),
        "symbols": int(len(symbols)),
        "microstructure_shift_bars": int(args.microstructure_shift_bars),
        "state_sha256": _sha256(state_path),
        "state_contract_hash": state.contract_hash,
        "transform_state_sha256": _sha256(transform_state_path),
        "transform_state_data_sha256": _sha256(
            transform_state_path.with_suffix(transform_state_path.suffix + ".state.npz")
        ),
        "outputs": list(state.OUTPUTS),
        "market_basket": list(state.market_basket),
        "precomposite_terminal_outputs": (
            "latest_precomposite_orderbook_outputs.parquet"
        ),
        "transformed_terminal_outputs": "latest_transformed_orderbook_outputs.parquet",
        "outcome_columns_consumed": [],
        "runtime_seconds": float(time.perf_counter() - started),
    }
    (args.out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
