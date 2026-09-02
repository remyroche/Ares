#!/usr/bin/env python3
"""Materialise conservative 25/50/75%-earlier-trailing continuation actions.

This research-only materialiser reuses the already sealed activation-50 target
as an exact control.  It adds the two adjacent actions, each beginning only on
the interval *after* its completed 15-minute state bar.  No live policy,
orders, or exchange calls are involved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.p8u_continuation_state import replay_open_long_policy_with_continuation_modulator
from scripts import materialize_strict_r3_p8u_15m_activation50_advantage as legacy
from scripts import run_strict_r3_p8u_15m_continuation_walkforward as base


DEFAULT_50 = ROOT / "data_perp/artifacts/strict_r3_p8u_15m_activation50_advantage_20260830_v1/activation50_advantage_states.parquet"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/strict_r3_p8u_15m_continuation_action_ladder_20260830_v1"
KEYS = ["candidate_id", "state_decision_ts", "state_bar_15m"]
ACTIONS = (0.25, 0.75)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _action_trace(row: pd.Series, high: np.ndarray, low: np.ndarray, close: np.ndarray, params, median: float, action: float) -> tuple[float, int, str]:
    action_bar = int(row["state_bar_15m"])

    def apply_next_interval(dynamic: dict[str, float]) -> float | None:
        # The policy calls this after a completed bar.  Zero means full
        # authority, while 2.0 is neutral; it cannot alter the same bar.
        return 0.0 if int(dynamic.pop("state_bar_15m")) == action_bar else 2.0

    trace = replay_open_long_policy_with_continuation_modulator(
        entry=float(row["entry_price"]), signal_atr=float(row["signal_atr"]),
        highs=high, lows=low, closes=close, params=params,
        median_atr_fraction=median, prediction_for_completed_bar=apply_next_interval,
        sl_tighten=0.0, giveback_tighten=0.0, activation_earlier=action,
    )
    return float(trace.terminal_gross_bps - 100.0), int(trace.terminal_exit_bar), str(trace.terminal_reason)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation50", type=Path, default=DEFAULT_50)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--symbol", action="append", help="optional repeatable smoke-test symbol")
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"immutable output already exists: {output}")
    source_path = args.activation50.resolve()
    panel = pd.read_parquet(source_path)
    required = set(KEYS) | {"__symbol__", "entry_decision_ts", "entry_price", "signal_atr", "parent_net_bps", "activation50_net_bps", "activation50_advantage_bps"}
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"activation-50 source misses {sorted(missing)}")
    panel = panel.copy()
    panel["candidate_id"] = panel.candidate_id.astype(str)
    panel["entry_decision_ts"] = pd.to_datetime(panel.entry_decision_ts, utc=True, errors="raise")
    panel["state_decision_ts"] = pd.to_datetime(panel.state_decision_ts, utc=True, errors="raise")
    if panel.duplicated(KEYS).any():
        raise AssertionError("activation-50 source has duplicate state identities")
    if args.symbol:
        requested = {str(value) for value in args.symbol}
        panel = panel.loc[panel.__symbol__.astype(str).isin(requested)].copy()
    if panel.empty:
        raise RuntimeError("no states selected")
    if not np.isclose(panel["activation50_net_bps"] - panel["parent_net_bps"], panel["activation50_advantage_bps"], rtol=0.0, atol=1e-8).all():
        raise AssertionError("activation-50 control no longer has an exact advantage identity")

    params, median, policy = base._load_policy()
    all_records: dict[float, list[dict[str, object]]] = {action: [] for action in ACTIONS}
    coverage: list[dict[str, object]] = []
    for symbol, rows in panel.groupby("__symbol__", sort=True):
        source = legacy._bars(str(symbol))
        if source is None:
            coverage.append({"symbol": str(symbol), "states": len(rows), "materialised": 0, "reason": "missing_15m_source"})
            continue
        count = 0
        for candidate_id, group in rows.groupby("candidate_id", sort=True):
            first = group.iloc[0]
            path = legacy._path(source, pd.Timestamp(first.entry_decision_ts))
            if path is None:
                continue
            high, low, close = path
            for _, row in group.sort_values("state_bar_15m", kind="stable").iterrows():
                identity = {
                    "candidate_id": str(candidate_id),
                    "state_decision_ts": pd.Timestamp(row.state_decision_ts),
                    "state_bar_15m": int(row.state_bar_15m),
                }
                for action in ACTIONS:
                    net, exit_bar, reason = _action_trace(row, high, low, close, params, median, action)
                    prefix = f"activation{int(round(action * 100)):02d}"
                    all_records[action].append({
                        **identity,
                        f"{prefix}_net_bps": net,
                        f"{prefix}_exit_bar": exit_bar,
                        f"{prefix}_exit_reason": reason,
                        f"{prefix}_advantage_bps": net - float(row.parent_net_bps),
                    })
                count += 1
        coverage.append({"symbol": str(symbol), "states": len(rows), "materialised": count, "reason": "ok" if count == len(rows) else "incomplete_path"})

    result = panel
    for action, records in all_records.items():
        detail = pd.DataFrame(records)
        if len(detail) != len(panel) or detail.duplicated(KEYS).any():
            raise AssertionError(f"activation-{int(action * 100)} target does not cover exact state identity")
        result = result.merge(detail, on=KEYS, how="inner", validate="one_to_one")
    if len(result) != len(panel):
        raise AssertionError("action-ladder merge changed the sealed state universe")

    output.mkdir(parents=True, exist_ok=False)
    result.to_parquet(output / "continuation_action_ladder_states.parquet", index=False, compression="zstd")
    pd.DataFrame(coverage).to_parquet(output / "source_coverage.parquet", index=False)
    summary_rows: list[dict[str, object]] = []
    for name in ("parent", "activation25", "activation50", "activation75"):
        net = "parent_net_bps" if name == "parent" else f"{name}_net_bps"
        advantage = None if name == "parent" else f"{name}_advantage_bps"
        grouped = result.assign(entry_month=result.entry_decision_ts.dt.strftime("%Y-%m")).groupby("entry_month", sort=True)
        for month, group in grouped:
            row = {"entry_month": month, "action": name, "states": len(group), "net_bps": float(group[net].mean())}
            if advantage:
                row["advantage_bps"] = float(group[advantage].mean())
                row["positive_advantage_fraction"] = float((group[advantage] > 0.0).mean())
            summary_rows.append(row)
    pd.DataFrame(summary_rows).to_parquet(output / "monthly_target_summary.parquet", index=False)
    manifest = {
        "schema": "strict-r3-p8u-15m-continuation-action-ladder-v1",
        "scope": "offline target materialisation only; no model fit, policy change, exchange IO, or order submission",
        "activation50_control": str(source_path), "activation50_control_sha256": _sha256(source_path),
        "actions": [0.25, 0.50, 0.75],
        "target": "each action's rich-policy net bps minus unchanged rich-parent net bps",
        "state_causality": "completed-state decision; action affects only the next 15m interval",
        "cost": "100 bps remains embedded exactly once in all policy outcomes",
        "rows": len(result), "candidates": int(result.candidate_id.nunique()),
        "policy": policy["params"],
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
