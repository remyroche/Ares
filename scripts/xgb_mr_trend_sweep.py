#!/usr/bin/env python3
"""
Sweep script for XGBoost MR vs Trend Classifier.

Optimizes:
- mr_trend_horizon
- mr_trend_threshold
"""

import sys
import os
import asyncio
import logging
from typing import Dict, Any, List

# Add project root to path
sys.path.append(os.getcwd())

from src.utils.tprint import tprint, tprint_info, tprint_error, tprint_success
from src.training.steps.market_analysis.xgb_mr_trend_step import XGBMrTrendStep

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mr_trend_sweep")

async def run_sweep():
    """Run parameter sweep for MR vs Trend."""

    # 1. Base Config
    base_config = {
        "symbol": "ETHUSDT",
        "exchange": "binance",
        "timeframe": "15m",
        "sweep_max_configs": 20,
        "execution_mode": "full", # Use full data for sweep to get valid metrics
        "fast_sweep": True,
    }

    tprint(f"🚀 Starting MR vs Trend Sweep for {base_config['symbol']}...")

    # 2. Initialize Step
    step = XGBMrTrendStep()

    # 3. Generate Variations
    configs = step.generate_config_variations(base_config)
    tprint_info(f"Generated {len(configs)} configurations to test.")

    # 4. Run Batch
    results = await step.run_config_batch(configs, base_config["symbol"], base_config["exchange"])

    # 5. Analyze
    df, analysis = step.analyze_and_rank_results(results)

    if not df.empty:
        tprint("\n🏆 Sweep Results (Top 10):")
        cols = ["signature", "mr_trend_horizon", "mr_trend_threshold", "f1_trend", "f1_mr", "weighted_score"]
        print(df[cols].head(10).to_string(index=False))

        # Save results
        filename = f"outcomes/mr_trend_sweep_results_{int(asyncio.get_event_loop().time())}.csv"
        df.to_csv(filename, index=False)
        tprint_success(f"Sweep results saved to {filename}")

        if "best_config" in analysis:
            tprint_success(f"Best Configuration: {analysis['best_config']['signature']}")

            # Print best params details
            best = analysis['best_config']
            tprint_info(f"  Horizon: {best['mr_trend_horizon']}")
            tprint_info(f"  Threshold: {best['mr_trend_threshold']}")
            tprint_info(f"  Weighted Score: {best['weighted_score']:.4f}")
    else:
        tprint_error("No results generated.")

if __name__ == "__main__":
    try:
        asyncio.run(run_sweep())
    except KeyboardInterrupt:
        tprint_warning("Sweep interrupted by user.")
    except Exception as e:
        tprint_error(f"Sweep failed: {e}")
        import traceback
        traceback.print_exc()
