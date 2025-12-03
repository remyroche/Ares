#!/usr/bin/env python3
"""ML Path Regime Sweep

This script performs a parameter sweep for the ML Path Regime Step.
It optimizes feature windows, thresholds, and model parameters to maximize
Regime Quality Score. XGBoost OOF Log Loss is recorded for informational purposes.

Parameters swept:
- path_ker_window_bars (24-96)
- path_trend_r2_window_bars (48-96)
- path_efficiency_high_threshold (0.55-0.65)
- risk_kde_bandwidth (0.03-0.08)
- xgb_quality_base_target_multiplier (2.0-3.0)

Usage:
    python3 scripts/ml_path_regime_sweep.py \
        --symbol ETHUSDT \
        --exchange binance \
        --timeframe 15m \
        --outcomes-dir outcomes \
        --variations 5
"""

import argparse
import sys
import itertools
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.steps.market_analysis.ml_path_regime_step import MLPathRegimeStep

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep ML Path Regime parameters")
    parser.add_argument("--symbol", type=str, default="ETHUSDT", help="Trading symbol")
    parser.add_argument("--exchange", type=str, default="binance", help="Exchange name")
    parser.add_argument("--timeframe", type=str, default="15m", help="Base timeframe")
    parser.add_argument("--direction", type=str, default="long", help="Trading direction")
    parser.add_argument("--execution-mode", type=str, default="light", help="Execution mode")
    parser.add_argument("--outcomes-dir", type=str, default="outcomes", help="Output directory")
    parser.add_argument("--variations", type=int, default=0, help="Limit number of variations (0=all)")
    return parser.parse_args()

def build_base_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "symbol": args.symbol,
        "exchange": args.exchange,
        "timeframe": args.timeframe,
        "direction": args.direction,
        "execution_mode": args.execution_mode,
        "risk_n_regimes": 4,
        "use_xgb_quality_scoring": True,
    }

def build_variations() -> List[Dict[str, Any]]:
    # Sweep ranges
    ker_windows = [24, 48, 72, 96]
    trend_r2_windows = [48, 72, 96]
    eff_thresholds = [0.55, 0.6, 0.65]
    kde_bandwidths = [0.03, 0.05, 0.08]
    xgb_multipliers = [2.0, 2.5, 3.0]

    variations = []

    for ker, r2, eff, bw, mult in itertools.product(
        ker_windows, trend_r2_windows, eff_thresholds, kde_bandwidths, xgb_multipliers
    ):
        var = {
            "path_ker_window_bars": ker,
            "path_trend_r2_window_bars": r2,
            "path_efficiency_high_threshold": eff,
            "risk_kde_bandwidth": bw,
            "xgb_quality_base_target_multiplier": mult,
            "sweep_tag": f"ker{ker}_r2{r2}_eff{eff}_bw{bw}_mult{mult}"
        }
        variations.append(var)

    return variations

def main():
    args = parse_args()

    print("🚀 ML Path Regime Sweep")
    print(f"Symbol: {args.symbol}, Timeframe: {args.timeframe}")

    base_config = build_base_config(args)
    variations = build_variations()

    if args.variations > 0:
        print(f"⚠️ limiting to first {args.variations} variations (of {len(variations)} total)")
        variations = variations[:args.variations]

    print(f"\n🔧 Generated {len(variations)} variations to sweep.")

    step = MLPathRegimeStep()

    # Run batch (Synchronous call to run_config_batch which handles asyncio.run internally)
    results_df = step.run_config_batch(
        base_config=base_config,
        variations=variations,
        output_dir=args.outcomes_dir
    )

    if results_df.empty:
        print("\n❌ No results generated.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ranking_path = f"{args.outcomes_dir}/ml_path_regime_sweep_ranking_{args.symbol}_{timestamp}.csv"

    # Analyze
    ranked_df = step.analyze_and_rank_results(
        results_df=results_df,
        output_path=ranking_path
    )

    print("\n✅ Sweep Complete.")

if __name__ == "__main__":
    main()
