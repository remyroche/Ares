#!/usr/bin/env python3
"""Trade funnel diagnostics for meta-labeling pipeline.

This script gives a quick view of where trades are being filtered out:
- raw consensus signals
- labeled events
- positive labels
- gated trades at several probability thresholds (if meta probabilities are provided)

Usage (example):
    python -m scripts.analyze_trade_funnel \
        --symbol ETHUSDT --exchange binance --timeframe 15m \
        --market-data-path historical_data/unified/binance/ethusdt_15m.parquet

It intentionally uses only the production labeling utilities and a simple
parquet loader, so you can point it at any cleaned OHLCV file.
"""

import argparse
from pathlib import Path
from typing import Sequence, List

import numpy as np
import pandas as pd

from src.utils.tprint import tprint_info, tprint_warning
from src.training.steps.labeling.feature_generation_meta_labeling_step import (
    generate_primary_signals,
    compute_realized_returns,
    DEFAULT_PROFIT_THRESHOLD,
    DEFAULT_STOP_THRESHOLD,
    DEFAULT_TRANSACTION_COST,
)


def load_market_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Market data file not found: {path}")
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix in {".csv", ".txt"}:
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported market data format: {path.suffix}")
    if not {"close", "high", "low"}.issubset(df.columns):
        raise ValueError("Market data must contain at least close/high/low columns")
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df = df.set_index("timestamp").sort_index()
        else:
            tprint_warning("No datetime index or timestamp column found; using RangeIndex.")
    return df


def summarize_funnel(df: pd.DataFrame, thresholds: Sequence[float]) -> None:
    tprint_info(f"Loaded market data: {len(df)} bars from {df.index.min()} to {df.index.max()}" if len(df) > 0 and isinstance(df.index, pd.DatetimeIndex) else f"Loaded market data: {len(df)} bars")

    signals = generate_primary_signals(df)
    consensus = signals.get("consensus", pd.Series(index=df.index, dtype=float))
    raw_signals = (consensus != 0).sum()

    realized_returns, binary_labels, exit_reasons, event_durations, mfe_series, mae_series, bin_long, bin_short = compute_realized_returns(
        df,
        signals,
        profit_threshold=DEFAULT_PROFIT_THRESHOLD,
        stop_threshold=DEFAULT_STOP_THRESHOLD,
        horizon=16,
        transaction_cost=DEFAULT_TRANSACTION_COST,
        min_event_spacing=2,
    )

    labeled_mask = ~realized_returns.isna()
    n_labeled = int(labeled_mask.sum())
    n_pos = int((binary_labels == 1.0).sum())
    n_neg = int((binary_labels == 0.0).sum())

    days = max(1.0, (len(df) / 96.0))  # rough 15m→days heuristic

    print("\n=== TRADE FUNNEL SUMMARY ===")
    print(f"Total bars: {len(df)}")
    print(f"Raw consensus signals: {raw_signals} ({raw_signals / days:.3f} per day)")
    print(f"Labeled events: {n_labeled} ({n_labeled / days:.3f} per day)")
    print(f"  Positive labels: {n_pos} ({(n_pos / max(1, n_labeled)) * 100:.2f}% of labeled)")
    print(f"  Negative labels: {n_neg}")

    print("\nNote: This script does not re-train the meta-model, so it cannot report gated\n"
          "trades at probability thresholds without external meta probabilities.\n"
          "Use the SNR/meta-labeling diagnostics or meta-gated backtest reports\n"
          "to inspect gating-level trade counts.")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Analyze trade funnel for meta-labeling pipeline.")
    parser.add_argument("--symbol", type=str, default="ETHUSDT", help="Symbol (for documentation only)")
    parser.add_argument("--exchange", type=str, default="binance", help="Exchange (for documentation only)")
    parser.add_argument("--timeframe", type=str, default="15m", help="Timeframe label (for documentation only)")
    parser.add_argument("--market-data-path", type=str, required=True, help="Path to OHLCV parquet/csv file")
    parser.add_argument("--thresholds", type=float, nargs="*", default=[0.50, 0.55, 0.60], help="Probability thresholds (currently informational only)")

    args = parser.parse_args(list(argv) if argv is not None else None)

    df = load_market_data(Path(args.market_data_path))
    summarize_funnel(df, args.thresholds)


if __name__ == "__main__":
    main()
