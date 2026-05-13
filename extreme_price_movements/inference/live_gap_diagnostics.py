"""Build concrete live-vs-OOS diagnostic artifacts from local inference logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from extreme_price_movements.data_store import PartitionedOHLCVStore
from extreme_price_movements.inference.live_gap_report import (
    build_live_gap_report,
    classify_gap_rows,
    render_live_gap_report_markdown,
)
from extreme_price_movements.inference.live_replay import (
    _normalise_strategy_id,
    attach_forward_outcomes,
    build_live_candidate_replay_table,
)


def _read_table(path: Optional[str | Path]) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    if p.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(p)
    raise ValueError(f"Unsupported table format: {p}")


def _iter_strategy_rows(payload: Any):
    if isinstance(payload, list):
        yield from payload
    elif isinstance(payload, dict):
        rows = payload.get("strategies") or payload.get("selected_strategies")
        if isinstance(rows, list):
            yield from rows
        else:
            for key, value in payload.items():
                if isinstance(value, dict) and key != "__cross_strategy_diagnostics__":
                    row = dict(value)
                    row.setdefault("strategy_id", key)
                    yield row


def load_strategy_oos_expectations(path: str | Path) -> pd.DataFrame:
    """Load strategy-level OOS expectations used when row-level OOS rows are absent."""
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    payload = json.loads(p.read_text())
    rows = []
    for row in _iter_strategy_rows(payload):
        if not isinstance(row, dict) or row.get("selected") is False:
            continue
        strategy_id = _normalise_strategy_id(
            row.get("strategy_id") or row.get("strategy_for_inference")
        )
        if not strategy_id:
            continue
        avg_net = row.get("avg_net_pnl_per_trade")
        if avg_net is None:
            metrics = row.get("final_policy_deployment_metrics") or row.get("metrics") or {}
            if isinstance(metrics, dict):
                avg_net = metrics.get("avg_pnl_bankroll") or metrics.get("mean_net_trade")
        rows.append(
            {
                "strategy_id": strategy_id,
                "side": str(row.get("side") or "").lower(),
                "strategy_oos_expected_net": pd.to_numeric(
                    pd.Series([avg_net]), errors="coerce"
                ).iloc[0],
                "strategy_oos_expected_net_bps": pd.to_numeric(
                    pd.Series([avg_net]), errors="coerce"
                ).iloc[0]
                * 10000.0,
                "deployment_rank_threshold": row.get("deployment_rank_threshold"),
                "oos_expectation_source": "strategy_level_policy_artifact",
            }
        )
    return pd.DataFrame(rows)


def attach_strategy_oos_expectations(
    replay: pd.DataFrame,
    expectations: pd.DataFrame,
) -> pd.DataFrame:
    if replay is None or replay.empty or expectations is None or expectations.empty:
        return replay
    out = replay.copy()
    out["strategy_id"] = out["strategy_id"].map(_normalise_strategy_id)
    exp = expectations.copy()
    exp["strategy_id"] = exp["strategy_id"].map(_normalise_strategy_id)
    join_cols = [c for c in ("strategy_id", "side") if c in out.columns and c in exp.columns]
    merged = out.merge(exp, on=join_cols, how="left", suffixes=("", "_strategy_oos"))
    expected = pd.to_numeric(merged.get("oos_expected_net_bps"), errors="coerce")
    fallback = pd.to_numeric(
        merged.get("strategy_oos_expected_net_bps"), errors="coerce"
    )
    missing = expected.isna() & fallback.notna()
    merged.loc[missing, "oos_expected_net_bps"] = fallback[missing]
    merged.loc[missing, "oos_expected_net_for_same_policy"] = fallback[missing] / 10000.0
    existing_source = merged.get(
        "oos_expectation_source", pd.Series(index=merged.index, dtype=object)
    )
    merged["oos_expectation_source"] = np.where(
        missing,
        existing_source,
        np.where(expected.notna(), "row_level_oos_join", pd.NA),
    )
    return merged


def _needed_symbols(*frames: pd.DataFrame) -> list[str]:
    symbols: set[str] = set()
    for df in frames:
        if df is not None and not df.empty and "symbol" in df.columns:
            symbols.update(str(s) for s in df["symbol"].dropna().unique() if str(s))
    return sorted(symbols)


def load_forward_ohlcv_panels(
    *,
    data_root: str | Path,
    symbols: list[str],
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    store = PartitionedOHLCVStore(str(data_root))
    close_parts: dict[str, pd.Series] = {}
    high_parts: dict[str, pd.Series] = {}
    low_parts: dict[str, pd.Series] = {}
    for symbol in symbols:
        df = store.load(
            symbol,
            columns=["high", "low", "close"],
            start_ts=start_ts,
            end_ts=end_ts,
        )
        if df.empty:
            continue
        close_parts[symbol] = pd.to_numeric(df["close"], errors="coerce")
        high_parts[symbol] = pd.to_numeric(df["high"], errors="coerce")
        low_parts[symbol] = pd.to_numeric(df["low"], errors="coerce")
    return (
        pd.DataFrame(close_parts).sort_index(),
        pd.DataFrame(high_parts).sort_index(),
        pd.DataFrame(low_parts).sort_index(),
    )


def build_live_gap_diagnostics(
    *,
    prediction_ledger_path: str | Path = "data/live_state/prediction_ledger.parquet",
    trade_log_path: str | Path = "inference_trades.csv",
    policy_artifact_path: str | Path = "data/artifacts/20260321_140000/policy_params/strategy_for_inference.json",
    data_root: str | Path = "data",
    output_dir: str | Path = "data/artifacts/20260321_140000/live_gap_diagnostics",
    primary_horizon_hours: int = 24,
    default_expected_fee_bps: float = 0.0,
) -> dict[str, Any]:
    ledger = _read_table(prediction_ledger_path)
    trades = _read_table(trade_log_path)
    expectations = load_strategy_oos_expectations(policy_artifact_path)
    replay = build_live_candidate_replay_table(
        ledger,
        trade_log=trades,
        default_expected_fee_bps=default_expected_fee_bps,
    )
    replay = attach_strategy_oos_expectations(replay, expectations)

    symbols = _needed_symbols(replay, ledger, trades)
    ts = pd.to_datetime(
        replay.get("signal_bar_ts", replay.get("timestamp", pd.Series(dtype=object))),
        utc=True,
        errors="coerce",
    )
    if symbols and ts.notna().any():
        start = pd.Timestamp(ts.min()) - pd.Timedelta(hours=1)
        end = pd.Timestamp(ts.max()) + pd.Timedelta(hours=int(primary_horizon_hours) + 2)
        close, high, low = load_forward_ohlcv_panels(
            data_root=data_root,
            symbols=symbols,
            start_ts=start,
            end_ts=end,
        )
        if not close.empty:
            replay = attach_forward_outcomes(
                replay,
                close=close,
                high=high,
                low=low,
                horizons=(1, 4, int(primary_horizon_hours)),
                primary_horizon=int(primary_horizon_hours),
                bar_minutes=60,
            )

    classified = classify_gap_rows(replay)
    report = build_live_gap_report(classified)
    report["inputs"] = {
        "prediction_ledger_path": str(prediction_ledger_path),
        "trade_log_path": str(trade_log_path),
        "policy_artifact_path": str(policy_artifact_path),
        "data_root": str(data_root),
        "primary_horizon_hours": int(primary_horizon_hours),
        "oos_expectation_note": (
            "Row-level OOS joins are used when present; otherwise the report falls "
            "back to strategy-level avg_net_pnl_per_trade from strategy_for_inference."
        ),
    }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    replay.to_csv(out_dir / "live_replay.csv", index=False)
    classified.to_csv(out_dir / "live_replay_classified.csv", index=False)
    (out_dir / "live_gap_report.json").write_text(
        json.dumps(report, indent=2, default=str)
    )
    (out_dir / "live_gap_report.md").write_text(render_live_gap_report_markdown(report))
    return report


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-ledger", default="data/live_state/prediction_ledger.parquet")
    parser.add_argument("--trade-log", default="inference_trades.csv")
    parser.add_argument(
        "--policy-artifact",
        default="data/artifacts/20260321_140000/policy_params/strategy_for_inference.json",
    )
    parser.add_argument("--data-root", default="data")
    parser.add_argument(
        "--output-dir",
        default="data/artifacts/20260321_140000/live_gap_diagnostics",
    )
    parser.add_argument("--primary-horizon-hours", type=int, default=24)
    parser.add_argument("--default-expected-fee-bps", type=float, default=0.0)
    args = parser.parse_args(argv)
    report = build_live_gap_diagnostics(
        prediction_ledger_path=args.prediction_ledger,
        trade_log_path=args.trade_log,
        policy_artifact_path=args.policy_artifact,
        data_root=args.data_root,
        output_dir=args.output_dir,
        primary_horizon_hours=args.primary_horizon_hours,
        default_expected_fee_bps=args.default_expected_fee_bps,
    )
    print(render_live_gap_report_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
