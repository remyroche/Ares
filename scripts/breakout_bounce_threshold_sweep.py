#!/usr/bin/env python3
"""Breakout/Bounce Threshold Sweep

This script performs a focused sweep over the breakout/bounce regime parameters
in `MLBreakoutBounceRegimeStep`.

It varies:
- Horizon bars
- Cross/hold buffer percentages
- Bounce/trap percentages
- S/R lookback days

For each configuration, it runs `MLBreakoutBounceRegimeStep.execute`,
collects economic quality metrics (Sharpe), and writes a compact CSV
and JSON analysis summarizing the results.

Usage (from project root):

    python3 scripts/breakout_bounce_threshold_sweep.py \
        --symbol ETHUSDT \
        --exchange binance \
        --timeframe 1h \
        --direction long \
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

from src.training.steps.market_analysis.ml_breakout_bounce_regime_step import MLBreakoutBounceRegimeStep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep breakout/bounce thresholds and summarize economic metrics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--symbol", type=str, default="ETHUSDT", help="Trading symbol")
    parser.add_argument("--exchange", type=str, default="binance", help="Exchange name")
    parser.add_argument("--timeframe", type=str, default="15m", help="Regime timeframe (e.g. 15m)")
    parser.add_argument("--direction", type=str, default="long", help="Trading direction")
    parser.add_argument("--outcomes-dir", type=str, default="outcomes", help="Directory to save sweep results")

    return parser.parse_args()


def build_base_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Construct a base configuration for the breakout step."""

    base_config: Dict[str, Any] = {
        "symbol": args.symbol,
        "exchange": args.exchange,
        "regime_timeframe": args.timeframe,
        "direction": args.direction,
        "execution_mode": "blank",  # Explicitly use blank mode as requested
        # Default parameters
        "breakout_horizon_bars": 96,
        "breakout_lookback_days": 30,
        "breakout_cross_buffer_pct": 0.0040,
        "breakout_hold_buffer_pct": 0.0030,
        "breakout_bounce_move_pct": 0.0015,
        "breakout_trap_revert_pct": 0.0030,
        # Ensure outputs are enabled
        "breakout_meta_enable": True,
        "enable_sr_strength_features": True,
        "enable_trap_quality_features": True,
    }

    return base_config


def build_threshold_sweep_configs(base_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate a focused set of configs that sweep breakout thresholds."""

    configs: List[Dict[str, Any]] = []

    def add_variant(tag: str, updates: Dict[str, Any]) -> None:
        cfg = dict(base_config)
        cfg.update(updates)
        # Tag for easier identification in downstream analysis
        cfg["breakout_sweep_tag"] = tag
        configs.append(cfg)

    # 0) Baseline: rely on defaults
    add_variant("baseline", {})

    # 1) Horizon Sweep (Reaction Time vs Noise)
    # Shorter horizon (e.g. 48 bars / 12h on 15m) -> Faster, noisier
    add_variant(
        "horizon_short_48",
        {"breakout_horizon_bars": 48}
    )
    # Longer horizon (e.g. 192 bars / 48h on 15m) -> Slower, more stable
    add_variant(
        "horizon_long_192",
        {"breakout_horizon_bars": 192}
    )

    # 2) Cross Buffer Sweep (Breakout Sensitivity)
    # Tighter buffer -> Easier to trigger breakout (more signals, more traps)
    add_variant(
        "cross_buffer_tight_0.002",
        {"breakout_cross_buffer_pct": 0.0020}
    )
    # Looser buffer -> Harder to trigger breakout (fewer signals, cleaner breaks)
    add_variant(
        "cross_buffer_loose_0.006",
        {"breakout_cross_buffer_pct": 0.0060}
    )

    # 3) Hold Buffer Sweep (Confirmation Strength)
    # Tighter hold -> Less confirmation needed
    add_variant(
        "hold_buffer_tight_0.001",
        {"breakout_hold_buffer_pct": 0.0010}
    )
    # Looser hold -> More confirmation needed
    add_variant(
        "hold_buffer_loose_0.005",
        {"breakout_hold_buffer_pct": 0.0050}
    )

    # 4) Bounce/Trap Definition Sweep
    # Larger bounce move required (stronger rejection)
    add_variant(
        "bounce_move_large_0.003",
        {"breakout_bounce_move_pct": 0.0030}
    )
    # Deeper trap reversion required (stronger trap)
    add_variant(
        "trap_revert_deep_0.005",
        {"breakout_trap_revert_pct": 0.0050}
    )

    # 5) S/R Lookback Sweep
    # Shorter history -> More reactive levels
    add_variant(
        "sr_lookback_short_14d",
        {"breakout_lookback_days": 14}
    )
    # Longer history -> stronger, major levels
    add_variant(
        "sr_lookback_long_60d",
        {"breakout_lookback_days": 60}
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

    csv_path = out_dir / f"breakout_threshold_sweep_{symbol}_{timestamp}.csv"
    analysis_path = out_dir / f"breakout_threshold_sweep_{symbol}_{timestamp}_analysis.json"

    results_df.to_csv(csv_path, index=False)

    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2, default=str)

    print(f"\n💾 Saved sweep results to: {csv_path}")
    print(f"💾 Saved analysis summary to: {analysis_path}")

    return csv_path, analysis_path


async def main_async() -> None:
    args = parse_args()

    print("🚀 Breakout/Bounce Threshold Sweep")
    print("=" * 60)
    print(f"Symbol: {args.symbol}")
    print(f"Exchange: {args.exchange}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Direction: {args.direction}")
    print(f"Outcomes dir: {args.outcomes_dir}")
    print("=" * 60)

    # Build base config and sweep configs
    base_config = build_base_config(args)
    sweep_configs = build_threshold_sweep_configs(base_config)

    print(f"\n🔧 Generated {len(sweep_configs)} threshold sweep configurations")

    # Initialize step
    step = MLBreakoutBounceRegimeStep()

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

    # Sort by Val Sharpe
    successful = successful.sort_values("val_sharpe_gated_75pct", ascending=False)

    cols = [
        "config_id",
        "breakout_sweep_tag",
        "val_sharpe_gated_75pct",
        "val_log_loss",
        "val_accuracy",
        "execution_time",
    ]

    available_cols = [c for c in cols if c in successful.columns]

    print("\n🏆 Top sweep configurations (by Val Sharpe):")
    print(successful[available_cols].head(10).to_string(index=False))


def main() -> None:
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\n⏹️ Sweep interrupted by user")
        sys.exit(1)


if __name__ == "__main__":
    main()
