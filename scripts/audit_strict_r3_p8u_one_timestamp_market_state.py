#!/usr/bin/env python3
"""Audit multi-candle parity for the direct P8U market/regime state node.

The canonical batch market/gate functions are used only as an offline oracle.
The state path itself is sequential, target-free and does not call the broad
feature graph.  This audit is intentionally narrower than full P8U scoring:
it proves a prerequisite DAG node before it is permitted to feed the sealed
175-feature executor.
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

from extreme_price_movements.features import add_regime_gates, compute_market_features
from extreme_price_movements.inference.p8u_single_timestamp_graph import (
    P8UMarketRegimeSnapshotState,
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
    parser.add_argument("--check-hours", type=int, default=4)
    parser.add_argument("--atol", type=float, default=2e-6)
    parser.add_argument("--rtol", type=float, default=2e-5)
    return parser.parse_args()


def _prefix(panel: dict[str, object], end: pd.Timestamp) -> dict[str, object]:
    return {
        name: frame.loc[:end] if isinstance(frame, pd.DataFrame) else frame
        for name, frame in panel.items()
    }


def main() -> None:
    args = _args()
    source_path = Path(args.source_state).resolve()
    out_dir = Path(args.out_dir).resolve()
    if ROOT not in source_path.parents or ROOT not in out_dir.parents:
        raise ValueError("all P8U audit paths must remain below repository root")
    if out_dir.exists():
        raise FileExistsError(f"immutable P8U market-node audit exists: {out_dir}")
    if args.check_hours < 2:
        raise ValueError("multi-candle audit requires at least two checks")
    payload = joblib.load(source_path)
    panel = payload.get("panel") if isinstance(payload, dict) else None
    symbols = tuple(map(str, (payload or {}).get("symbols") or ()))
    close = panel.get("close") if isinstance(panel, dict) else None
    if not isinstance(close, pd.DataFrame) or len(symbols) != 160:
        raise ValueError("audit source does not contain the sealed P8U 160-symbol panel")
    index = pd.DatetimeIndex(pd.to_datetime(close.index, utc=True))
    if len(index) <= args.check_hours:
        raise ValueError("source history is too short for requested parity checks")
    checked = index[-args.check_hours :]
    seed_end = index[-args.check_hours - 1]
    state = P8UMarketRegimeSnapshotState(symbols=symbols, market_basket=symbols)
    started = time.perf_counter()
    state.bootstrap(_prefix(panel, seed_end))
    rows: list[dict[str, object]] = []
    for timestamp in checked:
        actual = state.update(
            {
                field: panel[field].loc[timestamp].reindex(list(symbols)).to_numpy(np.float32)
                for field in P8UMarketRegimeSnapshotState.SOURCE_FIELDS
            },
            timestamp=timestamp,
        )
        reference_panel = _prefix(panel, timestamp)
        market = compute_market_features(reference_panel, list(symbols))
        reference = add_regime_gates(
            market,
            gate_vol_lookback_hours=24 * 7,
            gate_trend_thr=0.0,
        )
        for field in P8UMarketRegimeSnapshotState.OUTPUTS:
            left = actual[field]
            right = reference.loc[timestamp, field] if field in reference.columns else None
            if right is None:
                raise KeyError(f"canonical market/gate oracle lacks {field}")
            expected = np.full(len(symbols), np.float32(right), dtype=np.float32)
            finite = np.isfinite(left) & np.isfinite(expected)
            missing = int((np.isfinite(left) ^ np.isfinite(expected)).sum())
            delta = np.abs(left[finite].astype(np.float64) - expected[finite].astype(np.float64))
            passed = not missing and bool(
                np.isclose(left[finite], expected[finite], atol=args.atol, rtol=args.rtol).all()
            )
            rows.append(
                {
                    "timestamp": timestamp,
                    "feature": field,
                    "symbols": len(symbols),
                    "finite_pairs": int(finite.sum()),
                    "missing_mismatch": missing,
                    "max_abs_delta": float(delta.max()) if len(delta) else 0.0,
                    "status": "pass" if passed else "fail",
                }
            )
    table = pd.DataFrame(rows)
    out_dir.mkdir(parents=True)
    table.to_parquet(out_dir / "per_feature_per_timestamp.parquet", index=False, compression="zstd")
    failures = table.loc[table["status"].ne("pass")]
    summary = {
        "schema": "strict_r3_p8u_one_timestamp_market_state_parity_v1",
        "status": "pass" if failures.empty else "fail",
        "source_state": str(source_path),
        "source_state_sha256": _sha256(source_path),
        "symbols": len(symbols),
        "checked_timestamps": [timestamp.isoformat() for timestamp in checked],
        "features": len(P8UMarketRegimeSnapshotState.OUTPUTS),
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
    if failures.size:
        raise AssertionError("one-timestamp market/regime parity failed")


if __name__ == "__main__":
    main()
