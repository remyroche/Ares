#!/usr/bin/env python3
"""Liquidity Tree Threshold Sweep

This script performs a focused sweep over the hierarchical liquidity
regime tree thresholds introduced in `MLLiquidityRegimeStep`.

It varies:
- Percentile bands used to discover the volume, range, delta, and Amihud
  thresholds.
- Optionally, direct overrides for volume and Amihud thresholds.

For each configuration, it runs `MLLiquidityRegimeStep.execute`,
collects liquidity-specific quality metrics, and writes a compact CSV
and JSON analysis summarizing the results.

Usage (from project root):

    python3 scripts/liquidity_tree_threshold_sweep.py \
        --symbol ETHUSDT \
        --exchange binance \
        --timeframe 1h \
        --direction long \
        --execution-mode light \
        --outcomes-dir outcomes

You can control whether override-based configs are included and set the
weight of returns-based CoV separation in the overall quality score via
CLI flags.
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

from src.training.steps.market_analysis.ml_liquidity_regime_step import MLLiquidityRegimeStep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep liquidity tree thresholds and summarize quality metrics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--symbol", type=str, default="ETHUSDT", help="Trading symbol")
    parser.add_argument("--exchange", type=str, default="binance", help="Exchange name")
    parser.add_argument("--timeframe", type=str, default="1h", help="Regime timeframe (e.g. 1h)")
    parser.add_argument("--direction", type=str, default="long", help="Trading direction")
    parser.add_argument("--execution-mode", type=str, default="light", help="Execution mode for the step")
    parser.add_argument("--outcomes-dir", type=str, default="outcomes", help="Directory to save sweep results")

    parser.add_argument(
        "--include-overrides",
        action="store_true",
        help="Include configs that use explicit volume/Amihud threshold overrides",
    )

    parser.add_argument(
        "--cov-returns-weight",
        type=float,
        default=0.15,
        help=(
            "Weight for returns-based CoV separation in the liquidity quality score. "
            "Microstructure metrics remain primary; this only adjusts the relative "
            "influence of returns separation."
        ),
    )

    return parser.parse_args()


def build_base_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Construct a base configuration for the liquidity step.

    This mirrors the defaults used by the automated optimizer but keeps
    the focus on liquidity-tree-related behavior.
    """

    base_config: Dict[str, Any] = {
        "symbol": args.symbol,
        "exchange": args.exchange,
        "regime_timeframe": args.timeframe,
        "direction": args.direction,
        "execution_mode": args.execution_mode,
        # Core liquidity regime settings
        "liquidity_n_regimes": 5,
        "liquidity_regime_detection_method": "wcv",
        "liquidity_min_regime_size": 0.03,
        "liquidity_max_regimes": 8,
        "liquidity_regime_stability_threshold": 0.6,
        "liquidity_enable_centroid_refinement": True,
        "liquidity_centroid_iterations": 2,
        "liquidity_use_ewm_features": True,
        "liquidity_ewm_periods": [2, 6, 10],
        "liquidity_enable_prob_calibration": True,
        # Relative-volume and range context
        "liquidity_rvol_lookback_24": 12,
        "liquidity_rvol_lookback_168": 96,
        "liquidity_range_std_lookback": 24,
        # Winsorization for stability
        "liquidity_winsor_lower": 0.005,
        "liquidity_winsor_upper": 0.975,
        # Confidence floor used when mapping ambiguous samples
        "liquidity_min_regime_confidence": 0.2,
        # Quality weighting: make returns-based separation slightly more influential
        "liquidity_quality_weight_cov_returns": float(args.cov_returns_weight),
    }

    return base_config


def build_threshold_sweep_configs(
    base_config: Dict[str, Any], include_overrides: bool
) -> List[Dict[str, Any]]:
    """Generate a focused set of configs that sweep tree thresholds.

    The sweep covers:
    - Baseline (defaults)
    - Adjusted percentile bands for volume, range, and Amihud splits
    - Optional explicit volume/Amihud threshold overrides
    """

    configs: List[Dict[str, Any]] = []

    def add_variant(tag: str, updates: Dict[str, Any]) -> None:
        cfg = dict(base_config)
        cfg.update(updates)
        # Tag for easier identification in downstream analysis
        cfg["liquidity_tree_sweep_tag"] = tag
        configs.append(cfg)

    # 0) Baseline: rely on default percentile grids
    add_variant("baseline_default_bands", {})

    # 1) More aggressive high-volume cut: push split slightly higher
    add_variant(
        "volume_pct_035_075",
        {
            "liquidity_tree_volume_pct_low": 0.35,
            "liquidity_tree_volume_pct_high": 0.75,
        },
    )

    # 2) More discriminative low-vol/high-range split: widen range band
    add_variant(
        "range_pct_025_075",
        {
            "liquidity_tree_range_pct_low": 0.25,
            "liquidity_tree_range_pct_high": 0.75,
        },
    )

    # 3) Sharper Ghost vs Steamroller split via Amihud band
    add_variant(
        "amihud_pct_045_080",
        {
            "liquidity_tree_amihud_pct_low": 0.45,
            "liquidity_tree_amihud_pct_high": 0.80,
        },
    )

    if include_overrides:
        # 4) Explicit volume threshold override (moderate)
        add_variant(
            "volume_thr_0p30",
            {
                "liquidity_tree_volume_threshold_override": 0.30,
            },
        )

        # 5) Explicit volume threshold override (lenient)
        add_variant(
            "volume_thr_0p10",
            {
                "liquidity_tree_volume_threshold_override": 0.10,
            },
        )

        # 6) Explicit Amihud threshold override for Ghost vs Steamroller
        add_variant(
            "amihud_thr_1p00",
            {
                "liquidity_tree_amihud_threshold_override": 1.00,
            },
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

    csv_path = out_dir / f"liquidity_tree_threshold_sweep_{symbol}_{timestamp}.csv"
    analysis_path = out_dir / f"liquidity_tree_threshold_sweep_{symbol}_{timestamp}_analysis.json"

    results_df.to_csv(csv_path, index=False)

    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2, default=str)

    print(f"\n💾 Saved sweep results to: {csv_path}")
    print(f"💾 Saved analysis summary to: {analysis_path}")

    return csv_path, analysis_path


async def main_async() -> None:
    args = parse_args()

    print("🚀 Liquidity Tree Threshold Sweep")
    print("=" * 60)
    print(f"Symbol: {args.symbol}")
    print(f"Exchange: {args.exchange}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Direction: {args.direction}")
    print(f"Execution mode: {args.execution_mode}")
    print(f"Outcomes dir: {args.outcomes_dir}")
    print(f"Include overrides: {args.include_overrides}")
    print(f"Cov-returns weight: {args.cov_returns_weight:.3f}")
    print("=" * 60)

    # Build base config and sweep configs
    base_config = build_base_config(args)
    sweep_configs = build_threshold_sweep_configs(base_config, args.include_overrides)

    print(f"\n🔧 Generated {len(sweep_configs)} threshold sweep configurations")

    # Initialize step
    step = MLLiquidityRegimeStep()

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

    successful = successful.sort_values("overall_quality_score", ascending=False)

    cols = [
        "config_id",
        "config_signature",
        "overall_quality_score",
        "effort_result_cov_separation_score",
        "returns_cov_separation_score",
        "class_balance_score",
        "n_regimes",
        "n_samples",
        "config_liquidity_tree_sweep_tag",
    ]

    available_cols = [c for c in cols if c in successful.columns]

    print("\n🏆 Top sweep configurations (by overall_quality_score):")
    print(successful[available_cols].head(10).to_string(index=False))


def main() -> None:
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\n⏹️ Sweep interrupted by user")
        sys.exit(1)


if __name__ == "__main__":
    main()
