#!/usr/bin/env python3
"""ML Path Regime Sweep

This script performs a parameter sweep for the ML Path Regime Step.
It optimizations feature windows, thresholds, and model parameters to maximize
Regime Quality Score (primary) and minimize XGBoost OOF Log Loss (secondary).

It implements a hierarchical sweep strategy:
1.  **Stage 1:** Optimizes existing parameters (`ker`, `trend_r2`, `eff_threshold`, `kde_bw`, `xgb_mult`).
2.  **Stage 2:** Takes the best configuration from Stage 1 and sweeps `path_bad_return_threshold` and `path_bad_efficiency_threshold`.

Parameters swept:
Stage 1:
- path_ker_window_bars (24-96)
- path_trend_r2_window_bars (48-96)
- path_efficiency_high_threshold (0.55-0.65)
- risk_kde_bandwidth (0.03-0.08)
- xgb_quality_base_target_multiplier (2.0-3.0)

Stage 2:
- path_bad_return_threshold: [-0.005, -0.008, -0.010, -0.012, -0.015]
- path_bad_efficiency_threshold: [0.05, 0.10, 0.15, 0.20, 0.25]

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
    parser.add_argument("--execution-mode", type=str, default="blank", help="Execution mode")
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

def build_stage1_variations() -> List[Dict[str, Any]]:
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
            "sweep_tag": f"S1_ker{ker}_r2{r2}_eff{eff}_bw{bw}_mult{mult}"
        }
        variations.append(var)

    return variations

def build_stage2_variations(best_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    # Fixed parameters from Stage 1
    # We remove keys that are part of the sweep definition to avoid duplicates or confusion,
    # but strictly speaking, the best_config should be the base.

    # New sweep ranges for Stage 2
    bad_return_thresholds = [-0.005, -0.008, -0.010, -0.012, -0.015]
    bad_efficiency_thresholds = [0.05, 0.10, 0.15, 0.20, 0.25]

    variations = []

    for ret_thr, eff_thr in itertools.product(bad_return_thresholds, bad_efficiency_thresholds):
        # Start with the best configuration from Stage 1
        var = best_config.copy()

        # Remove metadata keys from Stage 1 if present
        for key in ["variation_id", "sweep_tag", "composite_rank", "rank_quality", "rank_logloss"]:
            var.pop(key, None)

        # Update with new sweep parameters
        var.update({
            "path_bad_return_threshold": ret_thr,
            "path_bad_efficiency_threshold": eff_thr,
            "sweep_tag": f"S2_ret{ret_thr}_eff{eff_thr}"
        })
        variations.append(var)

    return variations

def main():
    args = parse_args()

    print("🚀 ML Path Regime Hierarchical Sweep")
    print(f"Symbol: {args.symbol}, Timeframe: {args.timeframe}")

    base_config = build_base_config(args)
    step = MLPathRegimeStep()

    # --- Stage 1 ---
    print("\n--- Starting Stage 1: Optimizing Core Parameters ---")
    stage1_variations = build_stage1_variations()

    if args.variations > 0:
        print(f"⚠️ limiting Stage 1 to first {args.variations} variations (of {len(stage1_variations)} total)")
        stage1_variations = stage1_variations[:args.variations]

    print(f"🔧 Stage 1: {len(stage1_variations)} variations.")

    results_df_s1 = step.run_config_batch(
        base_config=base_config,
        variations=stage1_variations,
        output_dir=f"{args.outcomes_dir}/stage1"
    )

    if results_df_s1.empty:
        print("\n❌ Stage 1 produced no results. Aborting.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ranking_path_s1 = f"{args.outcomes_dir}/ml_path_regime_sweep_stage1_{args.symbol}_{timestamp}.csv"

    ranked_df_s1 = step.analyze_and_rank_results(
        results_df=results_df_s1,
        output_path=ranking_path_s1
    )

    # Select best config from Stage 1 (lowest composite_rank is best)
    best_row_s1 = ranked_df_s1.iloc[0]
    print(f"\n🏆 Best Stage 1 Config (Rank: {best_row_s1.get('composite_rank', 'N/A')}):")
    print(best_row_s1.to_dict())

    best_config_s1 = best_row_s1.to_dict()

    # --- Stage 2 ---
    print("\n--- Starting Stage 2: Sweeping Bad-Path Thresholds ---")

    # Prepare base config for Stage 2 by merging global base config with best Stage 1 params
    # We must exclude the result metrics columns to avoid passing them as config
    metric_cols = [
        'risk_cv_ratio', 'quality_score', 'wasserstein_distance', 'avg_duration_bars',
        'stability_score', 'xgb_stats_csv_path', 'xgb_clf_accuracy', 'xgb_clf_logloss',
        'xgb_reg_r2', 'xgb_reg_rmse', 'target_upper_hit_pct', 'target_noise_pct',
        'rank_quality', 'rank_logloss', 'composite_rank', 'variation_id', 'sweep_tag',
        'tail_risk_alignment_3h' # Exclude if present from Stage 1
    ]

    clean_best_config_s1 = {k: v for k, v in best_config_s1.items() if k not in metric_cols}
    # Ensure it includes the base config items if they were lost (though they should be in best_row_s1)
    clean_best_config_s1.update(base_config)

    stage2_variations = build_stage2_variations(clean_best_config_s1)

    if args.variations > 0:
         # For Stage 2, usually we want to run all (it's small, ~25), but respect the limit if very small
         limit_s2 = max(args.variations, 25) # Ensure we run at least a full small grid if possible
         if len(stage2_variations) > limit_s2:
             print(f"⚠️ limiting Stage 2 to first {limit_s2} variations")
             stage2_variations = stage2_variations[:limit_s2]

    print(f"🔧 Stage 2: {len(stage2_variations)} variations.")

    results_df_s2 = step.run_config_batch(
        base_config=clean_best_config_s1, # Pass the cleaned best config as base, although variations have it all
        variations=stage2_variations,
        output_dir=f"{args.outcomes_dir}/stage2"
    )

    if results_df_s2.empty:
        print("\n❌ Stage 2 produced no results.")
        return

    ranking_path_s2 = f"{args.outcomes_dir}/ml_path_regime_sweep_stage2_{args.symbol}_{timestamp}.csv"
    ranked_df_s2 = step.analyze_and_rank_results(
        results_df=results_df_s2,
        output_path=ranking_path_s2
    )

    print(f"\n✅ Hierarchical Sweep Complete. Final Ranking: {ranking_path_s2}")

if __name__ == "__main__":
    main()
