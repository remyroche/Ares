#!/usr/bin/env python3
"""Volume Force Threshold Sweep

This script performs a focused sweep over the ML Volume Force model configuration.
It optimizes the target definition (lookahead, ATR threshold) and feature normalization
windows to minimize OOF Log Loss across the 3 models (Breakout, Volatility, Trend).

It varies:
- `volume_force_target_threshold_atr`: Threshold for Breakout logic.
- `volume_force_lookahead`: Forecast horizon in bars (15m).
- `volume_force_normalization_window`: Rolling window size for feature z-scoring.

Usage (from project root):

    python3 scripts/volume_force_threshold_sweep.py \
        --symbol ETHUSDT \
        --exchange binance \
        --timeframe 15m \
        --outcomes-dir outcomes

"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple
import itertools

import pandas as pd

# Ensure project root is on sys.path so that `src.*` imports work
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.steps.market_analysis.ml_volume_force_step import MLVolumeForceStep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep Volume Force model thresholds and summarize quality metrics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--symbol", type=str, default="ETHUSDT", help="Trading symbol")
    parser.add_argument("--exchange", type=str, default="binance", help="Exchange name")
    parser.add_argument("--timeframe", type=str, default="15m", help="Base timeframe")
    parser.add_argument("--direction", type=str, default="long", help="Trading direction (mostly for context)")
    parser.add_argument("--execution-mode", type=str, default="light", help="Execution mode for the step")
    parser.add_argument("--outcomes-dir", type=str, default="outcomes", help="Directory to save sweep results")

    return parser.parse_args()


def build_base_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Construct a base configuration for the volume force step."""

    base_config: Dict[str, Any] = {
        "symbol": args.symbol,
        "exchange": args.exchange,
        "timeframe": args.timeframe,
        "direction": args.direction,
        "execution_mode": args.execution_mode,

        # Standard trainer defaults (can be overridden if needed, but keeping fixed for sweep)
        "xgb_model_use_gpu": False,
    }

    return base_config


def build_sweep_configs(base_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate configs for the sweep."""

    # Define sweep ranges
    # Focused ATR band around lower thresholds
    atr_thresholds = [0.8, 1.0, 1.2]
    lookaheads = [8, 12, 16]  # 2h, 3h, 4h
    # Feature norm window
    norm_windows = [96, 192] # approx 1 day, 2 days

    configs: List[Dict[str, Any]] = []

    # Generate Cartesian product
    for atr, lookahead, norm in itertools.product(atr_thresholds, lookaheads, norm_windows):
        cfg = dict(base_config)
        cfg["volume_force_target_threshold_atr"] = atr
        cfg["volume_force_lookahead"] = lookahead
        cfg["volume_force_normalization_window"] = norm

        # Tag for easy identification
        cfg["sweep_tag"] = f"atr{atr}_lah{lookahead}_norm{norm}"

        configs.append(cfg)

    return configs


def save_sweep_results(
    results_df: pd.DataFrame,
    analysis: Dict[str, Any],
    symbol: str,
    outcomes_dir: str,
) -> Tuple[Path, Path]:
    """Persist sweep results to CSV and JSON."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(outcomes_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"volume_force_sweep_{symbol}_{timestamp}.csv"
    analysis_path = out_dir / f"volume_force_sweep_{symbol}_{timestamp}_analysis.json"

    results_df.to_csv(csv_path, index=False)

    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2, default=str)

    print(f"\n💾 Saved sweep results to: {csv_path}")
    print(f"💾 Saved analysis summary to: {analysis_path}")

    return csv_path, analysis_path


async def main_async() -> None:
    args = parse_args()

    print("🚀 Volume Force Multi-Task Threshold Sweep")
    print("=" * 60)
    print(f"Symbol: {args.symbol}")
    print(f"Exchange: {args.exchange}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Outcomes dir: {args.outcomes_dir}")
    print("=" * 60)

    # Build configs
    base_config = build_base_config(args)
    sweep_configs = build_sweep_configs(base_config)

    print(f"\n🔧 Generated {len(sweep_configs)} sweep configurations")

    # Initialize step
    step = MLVolumeForceStep()

    # Run batch
    results = await step.run_config_batch(sweep_configs, args.symbol, args.exchange)

    # Analyze results
    results_df, analysis = step.analyze_and_rank_results(results)

    if results_df.empty:
        print("\n❌ No results to save; all configurations appear to have failed.")
        return

    # Persist outputs
    save_sweep_results(results_df, analysis, args.symbol, args.outcomes_dir)

    # Print summary
    successful = results_df[results_df.get("success", False) == True].copy()
    if successful.empty:
        print("\n⚠️ No successful configurations in sweep.")
        return

    # Sort by Average Log Loss (which MLVolumeForceStep reports as 'oof_log_loss')
    successful = successful.sort_values("oof_log_loss", ascending=True)

    cols = [
        "config_id",
        "oof_log_loss",  # This is the Average Log Loss across 3 models
        "breakout_log_loss",
        "volatility_log_loss",
        "trend_log_loss",
        "config_volume_force_target_threshold_atr",
        "config_volume_force_lookahead",
        "config_volume_force_normalization_window",
    ]

    available_cols = [c for c in cols if c in successful.columns]

    print("\n🏆 Top sweep configurations (by lowest Avg LogLoss):")
    print(successful[available_cols].head(10).to_string(index=False))


def main() -> None:
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\n⏹️ Sweep interrupted by user")
        sys.exit(1)


if __name__ == "__main__":
    main()
