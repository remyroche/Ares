#!/usr/bin/env python3
"""
Automated Liquidity Regime Configuration Optimizer

This script integrates with ml_liquidity_regime_step to automatically test
multiple configuration variations and identify the optimal settings using
the same scoring methodology as liquidity_regime_run_selector.py.

Usage:
    python3 scripts/automated_liquidity_optimizer.py \
        --symbol ETHUSDT \
        --exchange binance \
        --max-configs 30 \
        --outcomes-dir outcomes

The script will:
1. Generate systematic configuration variations
2. Run each configuration through ml_liquidity_regime_step
3. Analyze results using the same scoring as liquidity_regime_run_selector
4. Save the best configuration and detailed analysis
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, Any

# Ensure project root is on sys.path so that 'src.*' imports work
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.steps.market_analysis.ml_liquidity_regime_step import MLLiquidityRegimeStep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automated liquidity regime configuration optimization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--symbol", type=str, default="ETHUSDT", help="Trading symbol")
    parser.add_argument("--exchange", type=str, default="binance", help="Exchange name")
    parser.add_argument("--timeframe", type=str, default="1h", help="Timeframe for regimes")
    parser.add_argument("--direction", type=str, default="long", help="Trading direction")
    parser.add_argument("--execution-mode", type=str, default="light", help="Execution mode")
    
    parser.add_argument(
        "--max-configs",
        type=int,
        default=30,
        help="Maximum number of configurations to test"
    )
    parser.add_argument(
        "--outcomes-dir",
        type=str,
        default="outcomes",
        help="Directory to save results"
    )
    
    parser.add_argument(
        "--base-config",
        type=str,
        help="Path to base configuration JSON file"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate configurations but don't run them (for testing)"
    )

    return parser.parse_args()


def load_base_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Load base configuration."""
    
    # Default base configuration
    base_config = {
        "symbol": args.symbol,
        "exchange": args.exchange,
        "regime_timeframe": args.timeframe,
        "direction": args.direction,
        "execution_mode": args.execution_mode,
        
        # Liquidity-specific defaults
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
        "liquidity_rvol_lookback_24": 12,
        "liquidity_rvol_lookback_168": 96,
        "liquidity_range_std_lookback": 24,
        "liquidity_winsor_lower": 0.005,
        "liquidity_winsor_upper": 0.975,
        # For HPO runs, use a more permissive confidence threshold so that
        # we have enough samples for model training. Strict diagnostics can
        # still rely on the default 0.5 when this is unset.
        "liquidity_min_regime_confidence": 0.2,
        
        # Optimization settings
        "liquidity_max_config_combinations": args.max_configs,
    }
    
    # Override with base config file if provided
    if args.base_config:
        base_config_path = Path(args.base_config)
        if base_config_path.exists():
            with open(base_config_path, 'r') as f:
                user_config = json.load(f)
            base_config.update(user_config)
            print(f"✅ Loaded base config from: {base_config_path}")
        else:
            raise FileNotFoundError(f"Base config file not found: {base_config_path}")
    
    return base_config


async def main() -> None:
    args = parse_args()
    
    print("🔬 Automated Liquidity Regime Configuration Optimizer")
    print("=" * 60)
    print(f"Symbol: {args.symbol}")
    print(f"Exchange: {args.exchange}")
    print(f"Max Configurations: {args.max_configs}")
    print(f"Outcomes Directory: {args.outcomes_dir}")
    print("=" * 60)
    
    # Load base configuration
    base_config = load_base_config(args)
    
    # Initialize the ML Liquidity Regime Step
    step = MLLiquidityRegimeStep()
    
    if args.dry_run:
        print("🔍 DRY RUN MODE - Generating configurations only...")
        
        # Generate configurations to see what will be tested
        configs = step.generate_config_variations(base_config)
        
        print(f"\n📋 Generated {len(configs)} configurations:")
        for i, config in enumerate(configs[:10]):  # Show first 10
            signature = step.get_config_signature(config)
            print(f"  {i+1:2d}. {signature}")
        
        if len(configs) > 10:
            print(f"  ... and {len(configs) - 10} more")
        
        print(f"\n💡 Use --no-dry-run to execute these configurations")
        return
    
    try:
        # Run automated configuration optimization
        print("\n🚀 Starting automated configuration optimization...")
        
        results = await step.run_automated_config_optimization(base_config)
        
        # Display final summary
        print("\n" + "="*60)
        print("🎯 OPTIMIZATION COMPLETE")
        print("="*60)
        
        best_config = results.get("best_config")
        summary = results.get("optimization_summary", {})
        
        if best_config:
            print(f"\n🥇 BEST CONFIGURATION FOUND:")
            print(f"   Signature: {best_config.get('config_signature', 'N/A')}")
            print(f"   Composite Score: {best_config.get('composite_score', 0):.4f}")
            print(f"   Quality Score: {best_config.get('overall_quality_score', 0):.4f}")
            print(f"   Execution Time: {best_config.get('execution_time', 0):.1f}s")
            print(f"   N Regimes: {best_config.get('n_regimes', 'N/A')}")
        
        print(f"\n📊 OPTIMIZATION SUMMARY:")
        print(f"   Total Configurations Tested: {summary.get('total_configs_tested', 0)}")
        print(f"   Successful Runs: {summary.get('successful_runs', 0)}")
        print(f"   Success Rate: {summary.get('successful_runs', 0) / max(summary.get('total_configs_tested', 1), 1) * 100:.1f}%")
        print(f"   Best Composite Score: {summary.get('best_composite_score', 0):.4f}")
        
        print(f"\n📁 Results saved to: {args.outcomes_dir}/")
        
        # Show how to use the best config
        if best_config:
            print(f"\n💡 TO USE THE BEST CONFIGURATION:")
            print(f"   1. Check the YAML file: liquidity_best_config_{args.symbol}_*.yaml")
            print(f"   2. Load it in your pipeline:")
            print(f"      with open('liquidity_best_config_{args.symbol}_*.yaml') as f:")
            print(f"          config = yaml.safe_load(f)")
            print(f"   3. Run: step.execute(config)")
        
    except KeyboardInterrupt:
        print("\n⏹️ Optimization interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Optimization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
