#!/usr/bin/env python3
"""Mean Reversion Threshold Sweep

This script performs a parameter sweep over the Mean Reversion regime model settings.
It optimizes both "Teacher" (regime definition) and "Student" (prediction features) parameters.

It sweeps:
- Teacher Thresholds: Hurst exponent, OU Half-life, ADF p-value, Variance Ratio.
- Student Features: Window sizes for Moving Averages, RSI, and VWAP.
- HPO Toggle: Baseline XGBoost vs Optimized XGBoost.

Usage (from project root):

    python3 scripts/mean_reversion_threshold_sweep.py \
        --symbol ETHUSDT \
        --exchange binance \
        --timeframe 15m \
        --direction long \
        --execution-mode light \
        --outcomes-dir outcomes

"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

# Ensure project root is on sys.path so that `src.*` imports work
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.launcher.ares_launcher import get_mode_lookback_days
from src.training.steps.market_analysis.ml_reversion_regime_step import MLMeanReversionRegimeStep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep Mean Reversion thresholds and student features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--symbol", type=str, default="ETHUSDT", help="Trading symbol")
    parser.add_argument("--exchange", type=str, default="binance", help="Exchange name")
    parser.add_argument("--timeframe", type=str, default="15m", help="Regime timeframe (e.g. 15m)")
    parser.add_argument("--direction", type=str, default="long", help="Trading direction")
    parser.add_argument("--execution-mode", type=str, default="blank", help="Execution mode for the step")
    parser.add_argument("--outcomes-dir", type=str, default="outcomes", help="Directory to save sweep results")
    parser.add_argument("--skip-hpo", action="store_true", help="Skip the student HPO variations to save time.")

    return parser.parse_args()


def build_base_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Construct a base configuration for the mean reversion step."""

    base_config: Dict[str, Any] = {
        "symbol": args.symbol,
        "exchange": args.exchange,
        "regime_timeframe": args.timeframe,
        "direction": args.direction,
        "execution_mode": args.execution_mode,
        # Ensure BaseStep honours the centralized blank/light lookback
        # window even when this script is run outside the launcher.
        "lookback_days": get_mode_lookback_days(args.execution_mode),
        # Default Teacher Thresholds (Baseline)
        "mr_hurst_threshold": 0.5,
        "mr_half_life_threshold": 12.0,
        "mr_adf_p_threshold": 0.15,
        "mr_vr_threshold": 1.2,
        # Default Student Features (Baseline)
        "mr_ma_fast_window": 20,
        "mr_ma_slow_window": 50,
        "mr_vwap_window": 30,
        "mr_rsi_window": 14,
        # Training
        "mr_enable_hpo": False, # Start with baseline default params
        "mr_oof_min_samples_for_training": 300,
    }

    return base_config


def build_sweep_configs(
    base_config: Dict[str, Any], skip_hpo: bool
) -> List[Dict[str, Any]]:
    """Generate a set of configs that sweep Teacher definitions and Student inputs."""

    configs: List[Dict[str, Any]] = []

    def add_variant(tag: str, updates: Dict[str, Any]) -> None:
        cfg = dict(base_config)
        cfg.update(updates)
        cfg["sweep_tag"] = tag
        configs.append(cfg)

    # 1. Baseline
    add_variant("baseline", {})

    # =========================================================================
    # TEACHER SWEEPS (Defining "What is Mean Reversion?")
    # =========================================================================

    # Stricter Mean Reversion (Higher confidence, fewer signals)
    add_variant(
        "teacher_strict",
        {
            "mr_hurst_threshold": 0.4,      # Lower Hurst = stronger reversion
            "mr_half_life_threshold": 8.0,  # Faster reversion required
            "mr_adf_p_threshold": 0.05,     # Stronger stationarity
        },
    )

    # Relaxed Mean Reversion (More signals, potentially noisier)
    add_variant(
        "teacher_relaxed",
        {
            "mr_hurst_threshold": 0.6,
            "mr_half_life_threshold": 16.0,
            "mr_adf_p_threshold": 0.20,
        },
    )

    # Variance Ratio Focused (Checks if variance grows slower than linear)
    add_variant(
        "teacher_vr_focused",
        {
            "mr_vr_threshold": 0.9, # Ratio < 1 indicates mean reversion
            "mr_hurst_threshold": 0.55, # Slightly relaxed Hurst
        }
    )

    # =========================================================================
    # STUDENT FEATURE SWEEPS (Input Information)
    # =========================================================================

    # "Fast" Student: Uses shorter lookback windows to react quickly
    add_variant(
        "student_fast",
        {
            "mr_ma_fast_window": 10,
            "mr_ma_slow_window": 30,
            "mr_vwap_window": 15,
            "mr_rsi_window": 9,
        },
    )

    # "Slow" Student: Uses longer lookback windows for trend awareness
    add_variant(
        "student_slow",
        {
            "mr_ma_fast_window": 30,
            "mr_ma_slow_window": 100,
            "mr_vwap_window": 60,
            "mr_rsi_window": 21,
        },
    )

    # =========================================================================
    # EVALUATION METRIC SWEEPS (F-beta weighting)
    # =========================================================================

    # F0.75 score (Slight emphasis on Precision)
    add_variant(
        "eval_f0.75",
        {
            "mr_eval_beta": 0.75,
        }
    )

    # F1 score (Balanced Precision/Recall)
    add_variant(
        "eval_f1",
        {
            "mr_eval_beta": 1.0,
        }
    )

    # F1.25 score (Slight emphasis on Recall)
    add_variant(
        "eval_f1.25",
        {
            "mr_eval_beta": 1.25,
        }
    )

    # F1.5 score (More emphasis on Recall than Precision compared to F1)
    add_variant(
        "eval_f1.5",
        {
            "mr_eval_beta": 1.5,
        }
    )

    # F2 score (Strong emphasis on Recall)
    add_variant(
        "eval_f2",
        {
            "mr_eval_beta": 2.0,
        }
    )

    # =========================================================================
    # STUDENT MODEL HPO SWEEPS
    # =========================================================================

    if not skip_hpo:
        # Run HPO on the Baseline configuration
        add_variant(
            "student_hpo_baseline",
            {
                "mr_enable_hpo": True,
            }
        )

        # Run HPO on the "Fast" feature set (Combination sweep)
        add_variant(
            "student_hpo_fast",
            {
                "mr_ma_fast_window": 10,
                "mr_ma_slow_window": 30,
                "mr_vwap_window": 15,
                "mr_rsi_window": 9,
                "mr_enable_hpo": True,
            }
        )

    return configs


def save_sweep_results(
    results_df: pd.DataFrame,
    analysis: Dict[str, Any],
    symbol: str,
    outcomes_dir: str,
) -> Tuple[Path, Path]:
    """Persist sweep results to CSV and JSON analysis files."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(outcomes_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"mr_threshold_sweep_{symbol}_{timestamp}.csv"
    analysis_path = out_dir / f"mr_threshold_sweep_{symbol}_{timestamp}_analysis.json"

    results_df.to_csv(csv_path, index=False)

    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2, default=str)

    print(f"\n💾 Saved sweep results to: {csv_path}")
    print(f"💾 Saved analysis summary to: {analysis_path}")

    return csv_path, analysis_path


async def main_async() -> None:
    args = parse_args()

    print("🚀 Mean Reversion Threshold Sweep")
    print("=" * 60)
    print(f"Symbol: {args.symbol}")
    print(f"Exchange: {args.exchange}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Direction: {args.direction}")
    print(f"Execution mode: {args.execution_mode}")
    print(f"Outcomes dir: {args.outcomes_dir}")
    print(f"Skip HPO: {args.skip_hpo}")
    print("=" * 60)

    # Build base config and sweep configs
    base_config = build_base_config(args)
    sweep_configs = build_sweep_configs(base_config, args.skip_hpo)

    print(f"\n🔧 Generated {len(sweep_configs)} sweep configurations")

    # Initialize step
    step = MLMeanReversionRegimeStep()

    # Run batch using the newly added batch runner
    results = await step.run_config_batch(sweep_configs, args.symbol, args.exchange)

    # Analyze and rank results
    results_df, analysis = step.analyze_and_rank_results(results)

    if results_df.empty:
        print("\n❌ No results to save; all configurations appear to have failed.")
        return

    # Persist sweep outputs
    save_sweep_results(results_df, analysis, args.symbol, args.outcomes_dir)


def main() -> None:
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\n⏹️ Sweep interrupted by user")
        sys.exit(1)


if __name__ == "__main__":
    main()
