#!/usr/bin/env python3
"""Materialise causal activation-50 action advantages for continuation research.

For every already-open parent-policy state, this evaluates one feasible action:
starting on the *next* 15-minute interval, set the activation multiplier to
0.50.  The label is that counterfactual net outcome minus the unchanged rich
parent-policy net outcome.  It is a research target only, never an order or
live policy change.
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
from scripts import run_strict_r3_p8u_15m_continuation_walkforward as base


V2_PANEL = ROOT / "data_perp/artifacts/strict_r3_p8u_15m_continuation_v2_features_20260830_v1/continuation_v2_state_features.parquet"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/strict_r3_p8u_15m_activation50_advantage_20260830_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bars(symbol: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    path = base.BARS_ROOT / base._symbol_filename(symbol)
    if not path.is_file():
        return None
    frame = pd.read_parquet(path, columns=["high", "low", "close"])
    frame.index = pd.to_datetime(frame.index, utc=True, errors="coerce")
    frame = frame.loc[~frame.index.isna() & ~frame.index.duplicated(keep="last")].sort_index()
    array = frame.apply(pd.to_numeric, errors="coerce")
    return (
        array.index.asi8.copy(), array["high"].to_numpy(float), array["low"].to_numpy(float), array["close"].to_numpy(float),
    )


def _path(source: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], decision: pd.Timestamp) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    timestamps, high, low, close = source
    location = int(np.searchsorted(timestamps, decision.value))
    if location >= len(timestamps) or timestamps[location] != decision.value:
        return None
    positions = location + np.arange(base.HORIZON_BARS, dtype=np.int64)
    if positions[-1] >= len(timestamps):
        return None
    values = high[positions], low[positions], close[positions]
    return values if all(np.isfinite(item).all() for item in values) else None


def _action_trace(row: pd.Series, high: np.ndarray, low: np.ndarray, close: np.ndarray, params, median: float) -> tuple[float, int, str]:
    action_bar = int(row["state_bar_15m"])

    def action_after_completed_bar(dynamic: dict[str, float]) -> float | None:
        # The policy module calls this only after the completed bar's parent
        # checks and updates.  Returning zero applies the 50% activation
        # multiplier only from the following bar; 2.0 is exactly neutral.
        return 0.0 if int(dynamic.pop("state_bar_15m")) == action_bar else 2.0

    trace = replay_open_long_policy_with_continuation_modulator(
        entry=float(row["entry_price"]), signal_atr=float(row["signal_atr"]),
        highs=high, lows=low, closes=close, params=params, median_atr_fraction=median,
        prediction_for_completed_bar=action_after_completed_bar,
        sl_tighten=0.0, giveback_tighten=0.0, activation_earlier=0.50,
    )
    return float(trace.terminal_gross_bps - 100.0), int(trace.terminal_exit_bar), str(trace.terminal_reason)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-panel", type=Path, default=V2_PANEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--symbol", action="append", help="optional repeatable symbol subset for smoke testing")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"immutable output already exists: {output}")
    panel_path = args.v2_panel.resolve()
    panel = pd.read_parquet(panel_path)
    panel["candidate_id"] = panel.candidate_id.astype(str)
    panel["entry_decision_ts"] = pd.to_datetime(panel["entry_decision_ts"], utc=True, errors="raise")
    labels = base._labels()
    panel = panel.merge(labels, on="candidate_id", how="inner", validate="many_to_one")
    panel = panel.loc[panel["policy_path_valid"].fillna(False)].copy()
    if args.symbol:
        panel = panel.loc[panel["__symbol__"].astype(str).isin({str(item) for item in args.symbol})].copy()
    if panel.empty:
        raise RuntimeError("no policy-valid v2 continuation states")
    params, median, policy = base._load_policy()
    records: list[pd.DataFrame] = []
    coverage: list[dict[str, object]] = []
    for symbol, symbol_rows in panel.groupby("__symbol__", sort=True):
        source = _bars(str(symbol))
        if source is None:
            coverage.append({"symbol": symbol, "states": len(symbol_rows), "materialised": 0, "reason": "missing_15m_source"})
            continue
        rows: list[dict[str, object]] = []
        for candidate_id, group in symbol_rows.groupby("candidate_id", sort=True):
            first = group.iloc[0]
            path = _path(source, pd.Timestamp(first["entry_decision_ts"]))
            if path is None:
                continue
            high, low, close = path
            for _, row in group.sort_values("state_bar_15m", kind="stable").iterrows():
                net, exit_bar, reason = _action_trace(row, high, low, close, params, median)
                rows.append({
                    "candidate_id": str(candidate_id),
                    "state_decision_ts": pd.Timestamp(row["state_decision_ts"]),
                    "state_bar_15m": int(row["state_bar_15m"]),
                    "activation50_net_bps": net,
                    "activation50_exit_bar": exit_bar,
                    "activation50_exit_reason": reason,
                    "parent_net_bps": float(row["policy_net_bps"]),
                    "activation50_advantage_bps": net - float(row["policy_net_bps"]),
                })
        frame = pd.DataFrame(rows)
        if not frame.empty:
            records.append(frame)
        coverage.append({"symbol": symbol, "states": len(symbol_rows), "materialised": len(frame), "reason": "ok"})
    if not records:
        raise RuntimeError("no action advantages materialised")
    target = pd.concat(records, ignore_index=True)
    keys = ["candidate_id", "state_decision_ts", "state_bar_15m"]
    if target.duplicated(keys).any() or len(target) != len(panel):
        raise AssertionError("action target does not cover the exact parent-policy state identity")
    result = panel.merge(target, on=keys, how="inner", validate="one_to_one")
    # The policy target has a known H12 resolution boundary.  Persist it for
    # every subsequent fold and never infer availability from state timing.
    result["policy_label_available_ts"] = pd.to_datetime(result["policy_label_available_ts"], utc=True, errors="raise")
    if not np.isclose(result["policy_gross_bps"] - result["policy_net_bps"], 100.0, rtol=0.0, atol=1e-8).all():
        raise AssertionError("parent policy cost must be embedded once")
    output.mkdir(parents=True, exist_ok=False)
    result.to_parquet(output / "activation50_advantage_states.parquet", index=False, compression="zstd")
    pd.DataFrame(coverage).to_parquet(output / "source_coverage.parquet", index=False)
    result["entry_month"] = pd.to_datetime(result["entry_decision_ts"], utc=True).dt.strftime("%Y-%m")
    summary = result.groupby("entry_month", as_index=False).agg(
        states=("candidate_id", "size"), candidates=("candidate_id", "nunique"),
        parent_net_bps=("parent_net_bps", "mean"), activation50_net_bps=("activation50_net_bps", "mean"),
        advantage_bps=("activation50_advantage_bps", "mean"), positive_advantage_fraction=("activation50_advantage_bps", lambda value: float((value > 0.0).mean())),
    )
    summary.to_parquet(output / "monthly_target_summary.parquet", index=False)
    manifest = {
        "schema": "strict-r3-p8u-15m-activation50-direct-advantage-v1",
        "scope": "offline target materialisation only; no model fit, policy change, exchange IO, or order submission",
        "v2_panel": str(panel_path), "v2_panel_sha256": _sha256(panel_path),
        "parent_policy": str(base.POLICY), "parent_policy_sha256": _sha256(base.POLICY),
        "target": "activation50 rich-policy net bps minus unchanged rich-parent net bps, beginning only on the interval after the completed state bar",
        "state_causality": "state rows are produced after the completed 15m bar and action is applied only to the next bar",
        "cost": "100 bps embedded once in parent and activation-50 policy outcomes",
        "rows": len(result), "candidates": int(result.candidate_id.nunique()),
        "policy": policy["params"],
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
