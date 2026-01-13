#!/usr/bin/env python3
"""
Train Specialists with GMM Enhancement CLI

Trains all 11 specialist models and processes their outputs through 
GMM enhanced features for downstream use.

Usage:
    python3 scripts/train_specialists_with_gmm.py --symbol ETHUSDT --exchange binance --timeframe 15m --direction long
    python3 src/launcher/ares_launcher.py train_specialists_with_gmm --symbol ETHUSDT --execution-mode full
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.steps.labeling.train_specialists_with_gmm_step import TrainSpecialistsWithGMMStep


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Train Specialists with GMM Enhancement",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Specialists Trained:
- Momentum Persistence: Captures structural inertia and trend sustainability
- SMC Regime: Smart Money Concepts focused on order blocks and liquidity sweeps  
- Volatility Burst: Detects compression regimes and imminent expansion
- Volume Force: Binary breakout classifier focused on order-flow impulse
- Macro Regime: High-horizon trend and regime shift detection
- Meso Regime: Intermediate-horizon cyclical and trend patterns
- Liquidity Regime: Monitors market depth and capacity states
- Path Regime: Analyzes the "roughness" and risk of the price path
- Risk Regime: Focuses on tail-risk (VaR/CVaR) and volatility escalation
- Microstructure: Analyzes spread volatility, price efficiency, and imbalance
- Spectral Energy: Captures frequency-domain energy and dominant cycles

Examples:
  # Basic training
  python3 scripts/train_specialists_with_gmm.py --symbol ETHUSDT --exchange binance --timeframe 15m --direction long
  
  # Force retraining
  python3 scripts/train_specialists_with_gmm.py --symbol ETHUSDT --exchange binance --timeframe 15m --direction long --force-retrain
  
  # Train specific specialists
  python3 scripts/train_specialists_with_gmm.py --symbol ETHUSDT --specialists enhanced_ml_risk_regime_step enhanced_ml_spectral_step
        """
    )
    
    # Basic configuration
    parser.add_argument("--symbol", default="ETHUSDT", help="Trading symbol (default: ETHUSDT)")
    parser.add_argument("--exchange", default="binance", help="Exchange (default: binance)")
    parser.add_argument("--timeframe", default="15m", help="Timeframe (default: 15m)")
    parser.add_argument("--direction", default="long", choices=["long", "short", "both"], help="Direction (default: long)")
    
    # Training options
    parser.add_argument("--force-retrain", action="store_true", help="Force retraining of specialists")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--use-synthetic-data", action="store_true", help="Use synthetic data for testing in environments without historical data")
    
    # Specialist selection
    parser.add_argument("--specialists", nargs="+", help="Specific specialists to train (default: all)")
    
    # Data options
    parser.add_argument("--lookback-days", type=float, help="Restrict to last N days of data")
    
    args = parser.parse_args()
    
    # Create configuration
    config = {
        "symbol": args.symbol,
        "exchange": args.exchange,
        "timeframe": args.timeframe,
        "direction": args.direction,
        "force_retrain": args.force_retrain,
        "verbose": args.verbose,
        "lookback_days": args.lookback_days,
    }

    # Generate synthetic data for testing if requested
    if args.use_synthetic_data:
        import pandas as pd
        import numpy as np

        print("⚠️ Generating Synthetic Data for Testing...")
        # Create enough data for 15m timeframe (e.g. 60 days)
        dates = pd.date_range(start="2024-01-01", end="2024-03-01", freq="15min")
        n = len(dates)

        # Random walk price with trends
        returns = np.random.normal(0, 0.001, n)
        # Add some trendiness
        trend = np.sin(np.linspace(0, 10, n)) * 0.0005
        returns += trend

        price = 1000 * np.exp(np.cumsum(returns))

        market_data = pd.DataFrame({
            'open': price,
            'high': price * (1 + np.abs(np.random.normal(0, 0.002, n))),
            'low': price * (1 - np.abs(np.random.normal(0, 0.002, n))),
            'close': price * (1 + np.random.normal(0, 0.0005, n)),
            'volume': np.abs(np.random.normal(1000, 500, n)) + 100,
            'quote_volume': np.abs(np.random.normal(1000000, 500000, n)) + 100000
        }, index=dates)

        # Ensure high >= low
        market_data['high'] = np.maximum(market_data['high'], market_data[['open', 'close']].max(axis=1))
        market_data['low'] = np.minimum(market_data['low'], market_data[['open', 'close']].min(axis=1))

        config["market_data"] = market_data
        print(f"✅ Generated {len(market_data)} synthetic bars")
    
    # Filter specialists if specified
    if args.specialists:
        config["specialists"] = args.specialists
    
    print("🚀 Starting Specialist Training with GMM Enhancement")
    print(f"📊 Configuration: {args.symbol} {args.exchange} {args.timeframe} {args.direction}")
    print(f"🔄 Force retrain: {args.force_retrain}")
    if args.specialists:
        print(f"🎯 Training specialists: {args.specialists}")
    else:
        print("🎯 Training all 11 specialists")
    
    # Run the training
    try:
        step = TrainSpecialistsWithGMMStep()
        result = asyncio.run(step.execute(config))
        
        if result.get("success", False):
            print("\n🎉 Training completed successfully!")
            print(f"📊 Summary:")
            print(f"   - Specialists trained: {result.get('n_specialists', 0)}")
            print(f"   - Raw features shape: {result.get('raw_features_shape', 'N/A')}")
            print(f"   - Enhanced features shape: {result.get('enhanced_features_shape', 'N/A')}")
            print(f"   - Outputs saved to: outcomes/specialists_with_gmm_*")
            return 0
        else:
            print(f"\n❌ Training failed: {result.get('error', 'Unknown error')}")
            return 1
            
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
