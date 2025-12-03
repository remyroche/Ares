#!/usr/bin/env python3
"""SMC Regime Threshold Sweep

This script performs a focused sweep over the SMC regime parameters
in `MLSMCRegimeStep`.

It varies:
- Lookahead horizons
- Breakout/Breakdown thresholds
- Volume Profile lookbacks
- Feature selection (max features)
- Normalization windows

For each configuration, it runs `MLSMCRegimeStep.execute`,
collects OOF IC and Gated Sharpe metrics, and writes a compact CSV
and JSON analysis summarizing the results.

Usage (from project root):

    python3 scripts/ml_smc_regime_sweep.py \
        --symbol ETHUSDT \
        --exchange binance \
        --timeframe 15m \
        --direction long \
        --outcomes-dir outcomes
"""

import argparse
import asyncio
import json
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock

# Ensure project root is on sys.path so that `src.*` imports work
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.steps.market_analysis.ml_smc_regime_step import MLSMCRegimeStep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep SMC regime thresholds and summarize economic metrics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--symbol", type=str, default="ETHUSDT", help="Trading symbol")
    parser.add_argument("--exchange", type=str, default="binance", help="Exchange name")
    parser.add_argument("--timeframe", type=str, default="15m", help="Regime timeframe (e.g. 15m)")
    parser.add_argument("--direction", type=str, default="long", help="Trading direction")
    parser.add_argument("--outcomes-dir", type=str, default="outcomes", help="Directory to save sweep results")
    parser.add_argument("--execution-mode", type=str, default="blank", help="Execution mode (blank, light, full)")

    return parser.parse_args()


def generate_synthetic_data(symbol: str, timeframe: str, n_rows: int = 1000) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    print(f"⚠️ Generating {n_rows} rows of synthetic data for {symbol} {timeframe}")

    dates = pd.date_range(end=datetime.now(), periods=n_rows, freq="15min")

    # Random walk for close price
    np.random.seed(42)
    returns = np.random.normal(0, 0.002, n_rows)
    price = 1000 * np.cumprod(1 + returns)

    # Derive OHLC
    high = price * (1 + np.abs(np.random.normal(0, 0.001, n_rows)))
    low = price * (1 - np.abs(np.random.normal(0, 0.001, n_rows)))
    open_p = price * (1 + np.random.normal(0, 0.0005, n_rows))

    # Volume
    volume = np.abs(np.random.normal(100, 50, n_rows)) + 10

    df = pd.DataFrame({
        "open": open_p,
        "high": high,
        "low": low,
        "close": price,
        "volume": volume
    }, index=dates)

    # Add ATR (usually required by features)
    df["atr"] = (df["high"] - df["low"]).rolling(14).mean().bfill()

    return df


def build_base_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Construct a base configuration for the SMC step."""

    base_config: Dict[str, Any] = {
        "symbol": args.symbol,
        "exchange": args.exchange,
        "regime_timeframe": args.timeframe,
        "direction": args.direction,
        "execution_mode": args.execution_mode,
        # Default parameters
        "smc_lookahead": 16,
        "smc_breakout_threshold": 0.5,
        "smc_breakdown_threshold": -0.5,
        "smc_vp_lookback": 100,
        "smc_xgb_max_features": 48,
        "smc_normalization_window": 500,
        "smc_xgb_enable_training": True,
        "smc_xgb_min_samples": 500,  # Reduced for synthetic data compatibility
    }

    return base_config


def build_threshold_sweep_configs(base_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate a focused set of configs that sweep SMC thresholds."""

    configs: List[Dict[str, Any]] = []

    def add_variant(tag: str, updates: Dict[str, Any]) -> None:
        cfg = dict(base_config)
        cfg.update(updates)
        # Tag for easier identification in downstream analysis
        cfg["smc_sweep_tag"] = tag
        configs.append(cfg)

    # 0) Baseline: rely on defaults
    add_variant("baseline", {})

    # 1) Lookahead Sweep (Reaction Time vs Noise)
    # Shorter lookahead -> Faster
    add_variant("lookahead_short_12", {"smc_lookahead": 12})
    # Longer lookahead -> More stable trend
    add_variant("lookahead_long_24", {"smc_lookahead": 24})
    add_variant("lookahead_long_32", {"smc_lookahead": 32})

    # 2) Labeling Threshold Sweep (Sensitivity)
    # Lower threshold -> More breakout/breakdown labels
    add_variant("thresh_sensitive_0.4", {"smc_breakout_threshold": 0.4, "smc_breakdown_threshold": -0.4})
    # Higher threshold -> Fewer, stronger signals
    add_variant("thresh_robust_0.6", {"smc_breakout_threshold": 0.6, "smc_breakdown_threshold": -0.6})

    # 3) Volume Profile Lookback Sweep (Context)
    # Shorter context -> Adapts to recent volume faster
    add_variant("vp_short_50", {"smc_vp_lookback": 50})
    # Longer context -> More stable volume profile
    add_variant("vp_long_200", {"smc_vp_lookback": 200})

    # 4) Feature Count Sweep (Complexity vs Overfitting)
    add_variant("feats_lean_30", {"smc_xgb_max_features": 30})
    add_variant("feats_rich_64", {"smc_xgb_max_features": 64})

    # 5) Normalization Window Sweep (Adaptivity)
    add_variant("norm_fast_250", {"smc_normalization_window": 250})
    add_variant("norm_slow_1000", {"smc_normalization_window": 1000})

    # Combinations (Top candidates based on intuition)
    add_variant(
        "combo_fast_sensitive",
        {
            "smc_lookahead": 12,
            "smc_breakout_threshold": 0.4,
            "smc_breakdown_threshold": -0.4,
            "smc_normalization_window": 250
        }
    )
    add_variant(
        "combo_slow_robust",
        {
            "smc_lookahead": 32,
            "smc_breakout_threshold": 0.6,
            "smc_breakdown_threshold": -0.6,
            "smc_normalization_window": 1000
        }
    )

    return configs


def save_sweep_results(
    results_df: pd.DataFrame,
    analysis: Dict[str, Any],
    symbol: str,
    outcomes_dir: str,
) -> Tuple[Path, Path]:
    """Persist threshold-sweep results to CSV and JSON analysis files."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(outcomes_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"smc_threshold_sweep_{symbol}_{timestamp}.csv"
    analysis_path = out_dir / f"smc_threshold_sweep_{symbol}_{timestamp}_analysis.json"

    results_df.to_csv(csv_path, index=False)

    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2, default=str)

    print(f"\n💾 Saved sweep results to: {csv_path}")
    print(f"💾 Saved analysis summary to: {analysis_path}")

    return csv_path, analysis_path


async def main_async() -> None:
    args = parse_args()

    print("🚀 SMC Regime Threshold Sweep")
    print("=" * 60)
    print(f"Symbol: {args.symbol}")
    print(f"Exchange: {args.exchange}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Direction: {args.direction}")
    print(f"Outcomes dir: {args.outcomes_dir}")
    print(f"Execution mode: {args.execution_mode}")
    print("=" * 60)

    # Build base config and sweep configs
    base_config = build_base_config(args)
    sweep_configs = build_threshold_sweep_configs(base_config)

    print(f"\n🔧 Generated {len(sweep_configs)} threshold sweep configurations")

    # Initialize step
    step = MLSMCRegimeStep()

    # MOCK DATA INJECTION logic
    # Try to verify if real data exists; if not, patch load_market_data_or_fail
    try:
        # Quick check using the step's method (we expect it to fail if no data)
        # We don't actually want to load it here fully if it's heavy, but we want to catch the error
        # A lightweight check:
        step.load_market_data_or_fail(base_config, allow_config_override=False, light_mode_filter=True)
    except ValueError:
        print("\n⚠️  Real market data not found. Switching to SYNTHETIC DATA mode for validation.")

        # Create synthetic data
        synthetic_df = generate_synthetic_data(args.symbol, args.timeframe, n_rows=2000)

        # Patch the method on the instance
        def mock_load(*args, **kwargs):
            return synthetic_df.copy(), "synthetic_generator"

        step.load_market_data_or_fail = mock_load

        # Also patch _save_artifact to avoid cluttering disk with synthetic artifacts,
        # unless we really want to check them. For sweep, we might want to suppress large saves.
        # For now, we'll let it save to versioned_artifacts (it handles overwrites).

    # Run batch using the step's built-in HPO-style runner
    results = await step.run_config_batch(sweep_configs, args.symbol, args.exchange)

    # Analyze and rank results
    results_df, analysis = step.analyze_and_rank_results(results)

    if results_df.empty:
        print("\n❌ No results to save; all configurations appear to have failed.")
        return

    # Persist sweep outputs
    save_sweep_results(results_df, analysis, args.symbol, args.outcomes_dir)

    # Print a compact top-k summary to stdout
    successful = results_df[results_df.get("success", False) == True].copy()
    if successful.empty:
        print("\n⚠️ No successful configurations in sweep.")
        return

    # Sort by IC (primary) and Sharpe (secondary)
    successful = successful.sort_values(["smc_xgb_oof_ic", "smc_xgb_oof_sharpe_gated_25pct"], ascending=False)

    cols = [
        "config_id",
        "smc_sweep_tag",
        "smc_xgb_oof_ic",
        "smc_xgb_oof_sharpe_gated_25pct",
        "smc_xgb_oof_accuracy",
        "execution_time",
    ]

    available_cols = [c for c in cols if c in successful.columns]

    print("\n🏆 Top sweep configurations (by OOF IC):")
    print(successful[available_cols].head(10).to_string(index=False))


def main() -> None:
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\n⏹️ Sweep interrupted by user")
        sys.exit(1)


if __name__ == "__main__":
    main()
