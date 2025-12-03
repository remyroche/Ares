#!/usr/bin/env python3
"""XGB Meso Trend Sweep

This script performs a focused sweep over feature parameters for the
XGB Meso Trend step.

It generates variations for:
- Meso trend target volatility window
- EWMA short and long spans
- HTF feature lookbacks (RSI, ATR, MACD)

It runs `XGBMesoTrendStep.run_config_batch` to execute the sweep and
analyzes results based on OOF RMSE.

Usage:
    python3 scripts/xgb_meso_trend_sweep.py \
        --symbol ETHUSDT \
        --timeframe 15m \
        --max-configs 50
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.steps.market_analysis.xgb_meso_regime_step import XGBMesoTrendStep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep feature parameters for XGB Meso Trend",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--symbol", type=str, default="ETHUSDT", help="Trading symbol")
    parser.add_argument("--exchange", type=str, default="binance", help="Exchange name")
    parser.add_argument("--timeframe", type=str, default="15m", help="Regime timeframe (e.g. 15m)")
    parser.add_argument("--direction", type=str, default="long", help="Trading direction")
    parser.add_argument("--max-configs", type=int, default=30, help="Max number of configs to test")
    parser.add_argument("--outcomes-dir", type=str, default="outcomes", help="Directory to save sweep results")
    parser.add_argument("--execution-mode", type=str, default="light", choices=["full", "light", "blank"], help="Execution mode (full/light/blank)")
    return parser.parse_args()


def build_base_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "symbol": args.symbol,
        "exchange": args.exchange,
        "regime_timeframe": args.timeframe,
        "direction": args.direction,
        "execution_mode": args.execution_mode,
        "meso_sweep_max_configs": args.max_configs,
    }


def save_sweep_results(
    results_df: pd.DataFrame,
    analysis: Dict[str, Any],
    symbol: str,
    outcomes_dir: str,
) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(outcomes_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"meso_trend_sweep_{symbol}_{timestamp}.csv"
    analysis_path = out_dir / f"meso_trend_sweep_{symbol}_{timestamp}_analysis.json"

    results_df.to_csv(csv_path, index=False)

    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2, default=str)

    print(f"\n💾 Saved sweep results to: {csv_path}")
    print(f"💾 Saved analysis summary to: {analysis_path}")


async def main_async() -> None:
    args = parse_args()

    print("🚀 XGB Meso Trend Feature Sweep")
    print("=" * 60)
    print(f"Symbol: {args.symbol}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Max Configs: {args.max_configs}")
    print("=" * 60)

    base_config = build_base_config(args)

    step = XGBMesoTrendStep()

    # Generate variations
    configs = step.generate_config_variations(base_config)

    if not configs:
        print("❌ No configurations generated.")
        return

    # Run batch
    results = await step.run_config_batch(configs, args.symbol, args.exchange)

    # Analyze
    results_df, analysis = step.analyze_and_rank_results(results)

    if results_df.empty:
        print("\n❌ No results to save.")
        return

    save_sweep_results(results_df, analysis, args.symbol, args.outcomes_dir)

    # Print top 5
    successful = results_df[results_df.get("success", False) == True].copy()
    if not successful.empty:
        print("\n🏆 Top 5 Configurations (by RMSE):")
        cols = ["config_id", "rmse", "n_samples", "config_signature"]
        print(successful[cols].head(5).to_string(index=False))
    else:
        print("\n⚠️ No successful configurations.")


def main() -> None:
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\n⏹️ Sweep interrupted by user")
        sys.exit(1)


if __name__ == "__main__":
    main()
