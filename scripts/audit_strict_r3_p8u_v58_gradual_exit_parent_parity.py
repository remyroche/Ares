#!/usr/bin/env python3
"""Verify the gradual exact-one-minute adapter is bit-identical to its parent.

Research-only audit.  A null modulator must reproduce every parent-policy
outcome before a non-null gradual continuation controller can be evaluated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.exact_1m_h4_overlay_research import (
    replay_exact_1m_gradual_h4_overlay,
)
from scripts.run_strict_r3_p8u_v58_e2_50_exact_matched import (
    DEFAULT_PATH_ROOT,
    DEFAULT_POLICY,
    _load_policy,
)

DEFAULT_STATE = ROOT / "data_perp/artifacts/strict_r3_p8u_v58_exact_h4_states_20260830_v1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-root", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--path-root", type=Path, default=DEFAULT_PATH_ROOT)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"immutable output exists: {output}")
    output.mkdir(parents=True, exist_ok=False)

    route = pd.read_parquet(args.state_root / "target_free_route.parquet")
    route["candidate_id"] = route["candidate_id"].astype(str)
    route["entry_ts"] = pd.to_datetime(route["entry_ts"], utc=True, errors="raise")
    parent = pd.read_parquet(args.state_root / "exact_parent_outcomes.parquet")
    parent["candidate_id"] = parent["candidate_id"].astype(str)
    rows = pd.read_parquet(args.path_root / "valid_exact_paths_rows.parquet")
    index = pd.Series(np.arange(len(rows), dtype=np.int64), index=rows["candidate_id"].astype(str))
    route["path_index"] = route["candidate_id"].map(index)
    if route["path_index"].isna().any():
        raise AssertionError("route/path candidate identity mismatch")
    route = route.merge(parent, on="candidate_id", how="inner", validate="one_to_one")
    archive = np.load(args.path_root / "exact_paths.npz", allow_pickle=False)
    entry = np.asarray(archive["entry"], dtype=float)
    atr = np.asarray(archive["atr"], dtype=float)
    high = np.asarray(archive["high"], dtype=np.float32)
    low = np.asarray(archive["low"], dtype=np.float32)
    close = np.asarray(archive["close"], dtype=np.float32)
    params, median, _ = _load_policy(args.policy)

    observed: list[dict[str, object]] = []
    for row in route.itertuples(index=False):
        path = int(row.path_index)
        trace = replay_exact_1m_gradual_h4_overlay(
            entry_price=float(entry[path]), signal_atr=float(atr[path]), entry_ts=row.entry_ts,
            highs=high[path], lows=low[path], closes=close[path], params=params,
            median_atr_fraction=float(median), mc1_expected_bps=float(row.bcf_mc1_expected_bps),
            state_modulator=None, emit_states=False,
        )
        observed.append({
            "candidate_id": str(row.candidate_id), "generic_net_bps": float(trace["net_bps"]),
            "generic_gross_bps": float(trace["gross_bps"]), "generic_exit_ts": pd.Timestamp(trace["exit_timestamp"]),
            "generic_exit_price": float(trace["exit_price"]), "generic_exit_minute": int(trace["exit_minute"]),
            "generic_exit_reason": str(trace["exit_reason"]),
        })
    actual = pd.DataFrame(observed).merge(parent, on="candidate_id", how="inner", validate="one_to_one")
    comparisons = {
        "net_bps": np.abs(actual["generic_net_bps"] - actual["parent_exact_net_bps"]).to_numpy(float),
        "gross_bps": np.abs(actual["generic_gross_bps"] - actual["parent_exact_gross_bps"]).to_numpy(float),
        "exit_minute": np.abs(actual["generic_exit_minute"] - actual["parent_exit_minute"]).to_numpy(float),
        "exit_timestamp_seconds": np.abs((pd.to_datetime(actual["generic_exit_ts"], utc=True) - pd.to_datetime(actual["parent_exit_ts"], utc=True)).dt.total_seconds()).to_numpy(float),
    }
    receipt = {
        "candidate_count": int(len(actual)),
        "max_abs_net_bps_delta": float(comparisons["net_bps"].max(initial=0.0)),
        "max_abs_gross_bps_delta": float(comparisons["gross_bps"].max(initial=0.0)),
        "max_abs_exit_minute_delta": float(comparisons["exit_minute"].max(initial=0.0)),
        "max_abs_exit_timestamp_seconds_delta": float(comparisons["exit_timestamp_seconds"].max(initial=0.0)),
        "exit_reason_differences": int((actual["generic_exit_reason"] != actual["parent_exit_reason"].astype(str)).sum()),
        "result": "pass" if all(value.max(initial=0.0) == 0.0 for value in comparisons.values()) else "fail",
    }
    actual.to_parquet(output / "candidate_parity.parquet", index=False, compression="zstd")
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True))
    if receipt["result"] != "pass":
        raise SystemExit("generic adapter failed parent parity")


if __name__ == "__main__":
    main()
