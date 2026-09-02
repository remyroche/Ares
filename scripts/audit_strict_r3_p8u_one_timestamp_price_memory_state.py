#!/usr/bin/env python3
"""Offline canonical parity audit for direct P8U price-memory state.

The direct state is run chronologically from the frozen target-free source.
The historical batch graph is then invoked once as an *offline oracle* for
the two selected final outputs.  The audit has no scoring/execution path.
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

from extreme_price_movements.inference.p8u_canonical_feature_adapter import (
    canonical_features_from_saved_panel,
)
from extreme_price_movements.inference.p8u_single_timestamp_graph import (
    P8UPriceMemoryCausalState,
)


KEYS = ("prior_volatility", "bars_to_resistance_daily_donchian")


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
    parser.add_argument("--check-hours", type=int, default=4)
    parser.add_argument("--atol", type=float, default=2e-6)
    parser.add_argument("--rtol", type=float, default=2e-5)
    return parser.parse_args()


def main() -> None:
    args = _args()
    source_path = Path(args.source_state).resolve()
    out_dir = Path(args.out_dir).resolve()
    if ROOT not in source_path.parents or ROOT not in out_dir.parents:
        raise ValueError("P8U paths must remain below repository root")
    if out_dir.exists():
        raise FileExistsError(f"immutable price-memory audit exists: {out_dir}")
    if args.check_hours < 2:
        raise ValueError("multi-candle audit requires at least two hours")
    payload = joblib.load(source_path)
    if not isinstance(payload, dict) or not isinstance(payload.get("panel"), dict):
        raise ValueError("source state lacks target-free panel")
    panel = payload["panel"]
    symbols = tuple(map(str, payload.get("symbols") or payload.get("universe_symbols") or ()))
    close = panel.get("close")
    if not isinstance(close, pd.DataFrame) or len(symbols) != 160:
        raise ValueError("audit requires sealed 160-symbol source")
    if len(close) <= args.check_hours:
        raise ValueError("source history too short")
    started = time.perf_counter()
    direct = P8UPriceMemoryCausalState(symbols=symbols, transform_keys=KEYS)
    direct_history = direct.bootstrap(panel)
    # This is intentionally the broad historical oracle, never the direct
    # execution route. It supplies the only admissible comparison for final
    # transformed values at the same timestamps.
    reference = canonical_features_from_saved_panel(
        panel,
        universe_symbols=symbols,
        requested_features=KEYS,
        full_config_causal_universe=True,
    )
    index = pd.DatetimeIndex(pd.to_datetime(close.index, utc=True))
    positions = range(len(index) - args.check_hours, len(index))
    rows: list[dict[str, object]] = []
    for position in positions:
        timestamp = index[position]
        for key in KEYS:
            left = direct_history[f"feature__{key}"][position].astype(np.float32)
            right = reference[key].loc[timestamp, list(symbols)].to_numpy(np.float32)
            finite = np.isfinite(left) & np.isfinite(right)
            missing = int((np.isfinite(left) ^ np.isfinite(right)).sum())
            delta = np.abs(left[finite].astype(np.float64) - right[finite].astype(np.float64))
            passed = not missing and bool(np.isclose(left[finite], right[finite], atol=args.atol, rtol=args.rtol).all())
            rows.append(
                {
                    "timestamp": timestamp,
                    "feature": key,
                    "symbols": len(symbols),
                    "direct_finite": int(np.isfinite(left).sum()),
                    "canonical_finite": int(np.isfinite(right).sum()),
                    "finite_pairs": int(finite.sum()),
                    "missing_mismatch": missing,
                    "max_abs_delta": float(delta.max()) if len(delta) else 0.0,
                    "status": "pass" if passed else "fail",
                }
            )
    table = pd.DataFrame(rows)
    out_dir.mkdir(parents=True)
    table.to_parquet(out_dir / "per_feature_per_timestamp.parquet", compression="zstd", index=False)
    failures = table.loc[table["status"].ne("pass")]
    summary = {
        "schema": "strict_r3_p8u_one_timestamp_price_memory_state_parity_v1",
        "status": "pass" if failures.empty else "fail",
        "source_state": str(source_path),
        "source_state_sha256": _sha256(source_path),
        "symbols": len(symbols),
        "checked_timestamps": [index[position].isoformat() for position in positions],
        "features": list(KEYS),
        "comparisons": int(len(table)),
        "failed_comparisons": int(len(failures)),
        "max_abs_delta": float(table["max_abs_delta"].max()),
        "atol": args.atol,
        "rtol": args.rtol,
        "runtime_seconds": time.perf_counter() - started,
        "outcome_columns_consumed": [],
        "batch_oracle_role": "offline parity only",
        "direct_node_calls_broad_feature_graph": False,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not failures.empty:
        raise AssertionError("one-timestamp price-memory parity failed")


if __name__ == "__main__":
    main()
