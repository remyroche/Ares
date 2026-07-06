#!/usr/bin/env python3
"""Attach live trade lifecycle outcomes to a run-scoped prediction ledger.

The frozen contextual TP/SL gate needs post-freeze policy-action rows with
matured outcomes.  Live prediction ledgers contain the decision rows, while
`inference_trades.csv` contains the later lifecycle exits.  This script joins
them through the existing live replay helper and writes an enriched copy of the
prediction ledger without modifying the live ledger in place.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.inference.live_replay import build_live_candidate_replay_table  # noqa: E402

DEFAULT_PREDICTION_LEDGER = (
    ROOT
    / "data_perp/exchanges/krakenfutures/live_state/prediction_ledgers/20260629_050000_lgbm_mda/"
    "prediction_ledger.parquet"
)
DEFAULT_TRADE_LOG = ROOT / "inference_trades.csv"

REPLAY_COPY_COLUMNS = {
    "realized_trade_net_bps": "live_replay_realized_trade_net_bps",
    "realized_net": "live_replay_realized_net",
    "realized_exit_price": "live_replay_realized_exit_price",
    "lifecycle_exit_ts": "live_replay_lifecycle_exit_ts",
    "exit_reason": "live_replay_exit_reason",
    "is_unresolved_trade": "live_replay_is_unresolved_trade",
    "diagnostic_complete": "live_replay_diagnostic_complete",
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        val = float(value)
        return val if np.isfinite(val) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _truthy(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.fillna(False)
    text = values.astype(str).str.strip().str.lower()
    return text.isin({"1", "1.0", "true", "t", "yes", "y", "trade", "traded", "accepted", "filled"})


def _fill_or_create(frame: pd.DataFrame, column: str, values: pd.Series) -> None:
    if column not in frame.columns:
        frame[column] = values
        return
    existing = frame[column]
    if pd.api.types.is_numeric_dtype(values):
        existing_numeric = pd.to_numeric(existing, errors="coerce")
        frame[column] = existing_numeric.where(existing_numeric.notna(), values)
    else:
        existing_text = existing.astype("string")
        missing = existing.isna() | existing_text.str.strip().isin(["", "nan", "nat", "<NA>"])
        frame[column] = existing.where(~missing, values)


def _head_name(strategy_id: pd.Series) -> pd.Series:
    return strategy_id.astype(str).str.extract(
        r"^(short_bollinger|long_bars|long_dist|short_asset)",
        expand=False,
    )


def materialize(
    prediction_ledger_path: Path,
    trade_log_path: Path,
    output_path: Path,
    *,
    default_expected_fee_bps: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    prediction = pd.read_parquet(prediction_ledger_path)
    trades = pd.read_csv(trade_log_path, low_memory=False)
    prediction_ts = (
        pd.to_datetime(prediction["timestamp"], utc=True, errors="coerce")
        if "timestamp" in prediction.columns
        else pd.Series(pd.NaT, index=prediction.index)
    )
    trade_ts = (
        pd.to_datetime(trades["timestamp"], utc=True, errors="coerce")
        if "timestamp" in trades.columns
        else pd.Series(pd.NaT, index=trades.index)
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
        replay = build_live_candidate_replay_table(
            prediction,
            trade_log=trades,
            default_expected_fee_bps=float(default_expected_fee_bps),
        )
    if len(replay) != len(prediction):
        raise ValueError(
            f"Replay row parity failed: prediction rows={len(prediction)} replay rows={len(replay)}"
        )

    out = prediction.copy()
    for src, dst in REPLAY_COPY_COLUMNS.items():
        if src in replay.columns:
            out[dst] = replay[src].to_numpy()

    realized_bps = pd.to_numeric(replay.get("realized_trade_net_bps"), errors="coerce")
    live_replay_net_return = realized_bps / 10000.0
    out["live_replay_net_return"] = live_replay_net_return.astype("float32")
    _fill_or_create(out, "net_return", out["live_replay_net_return"])
    if "lifecycle_exit_ts" in replay.columns:
        lifecycle_exit = pd.to_datetime(replay["lifecycle_exit_ts"], utc=True, errors="coerce")
        out["live_replay_exit_timestamp"] = lifecycle_exit
        _fill_or_create(out, "exit_timestamp", lifecycle_exit)

    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    traded = _truthy(out.get("was_traded", pd.Series(False, index=out.index)))
    realized = pd.to_numeric(out["live_replay_net_return"], errors="coerce").notna()
    heads = _head_name(out["strategy_id"]) if "strategy_id" in out.columns else pd.Series(index=out.index, dtype=object)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)

    realized_rows = out.loc[realized].copy()
    prediction_ts_max = prediction_ts.max() if prediction_ts.notna().any() else pd.NaT
    trade_ts_max = trade_ts.max() if trade_ts.notna().any() else pd.NaT
    prediction_to_trade_lag_minutes = (
        float((trade_ts_max - prediction_ts_max).total_seconds() / 60.0)
        if pd.notna(prediction_ts_max) and pd.notna(trade_ts_max)
        else None
    )
    summary = {
        "generated_by": Path(__file__).name,
        "prediction_ledger": str(prediction_ledger_path),
        "trade_log": str(trade_log_path),
        "output": str(output_path),
        "row_parity_passed": True,
        "prediction_rows": int(len(prediction)),
        "trade_log_rows": int(len(trades)),
        "output_rows": int(len(out)),
        "traded_rows": int(traded.sum()),
        "realized_rows": int(realized.sum()),
        "realized_traded_rows": int((traded & realized).sum()),
        "unresolved_traded_rows": int((traded & ~realized).sum()),
        "realized_timestamps": int(out.loc[realized, "timestamp"].nunique()) if "timestamp" in out.columns else 0,
        "realized_active_heads": int(heads.loc[realized].dropna().nunique()),
        "timestamp_min": out["timestamp"].min().isoformat() if "timestamp" in out.columns and out["timestamp"].notna().any() else "",
        "timestamp_max": out["timestamp"].max().isoformat() if "timestamp" in out.columns and out["timestamp"].notna().any() else "",
        "trade_log_timestamp_min": trade_ts.min().isoformat() if trade_ts.notna().any() else "",
        "trade_log_timestamp_max": trade_ts_max.isoformat() if pd.notna(trade_ts_max) else "",
        "prediction_to_trade_log_lag_minutes": prediction_to_trade_lag_minutes,
        "prediction_ledger_stale_vs_trade_log": bool(
            prediction_to_trade_lag_minutes is not None and prediction_to_trade_lag_minutes > 30.0
        ),
        "realized_timestamp_min": realized_rows["timestamp"].min().isoformat()
        if "timestamp" in realized_rows.columns and realized_rows["timestamp"].notna().any()
        else "",
        "realized_timestamp_max": realized_rows["timestamp"].max().isoformat()
        if "timestamp" in realized_rows.columns and realized_rows["timestamp"].notna().any()
        else "",
        "default_expected_fee_bps": float(default_expected_fee_bps),
    }
    return out, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-ledger", type=Path, default=DEFAULT_PREDICTION_LEDGER)
    parser.add_argument("--trade-log", type=Path, default=DEFAULT_TRADE_LOG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--default-expected-fee-bps", type=float, default=0.0)
    args = parser.parse_args()

    args.report_dir.mkdir(parents=True, exist_ok=True)
    enriched, summary = materialize(
        args.prediction_ledger,
        args.trade_log,
        args.output,
        default_expected_fee_bps=float(args.default_expected_fee_bps),
    )
    (args.report_dir / "live_prediction_ledger_outcome_manifest.json").write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n"
    )
    columns = [
        "timestamp",
        "strategy_id",
        "symbol",
        "side",
        "was_traded",
        "portfolio_decision",
        "live_replay_net_return",
        "live_replay_realized_trade_net_bps",
        "live_replay_exit_timestamp",
        "live_replay_exit_reason",
    ]
    preview_cols = [col for col in columns if col in enriched.columns]
    realized = pd.to_numeric(enriched["live_replay_net_return"], errors="coerce").notna()
    enriched.loc[realized, preview_cols].tail(50).to_csv(
        args.report_dir / "live_prediction_ledger_realized_preview.csv",
        index=False,
    )
    lines = [
        "# Live Prediction Ledger Outcome Materialization",
        "",
        f"Prediction ledger: `{args.prediction_ledger}`",
        f"Trade log: `{args.trade_log}`",
        f"Output: `{args.output}`",
        "",
        "## Summary",
        "",
        pd.DataFrame([summary]).to_markdown(index=False),
    ]
    (args.report_dir / "live_prediction_ledger_outcome_report.md").write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
