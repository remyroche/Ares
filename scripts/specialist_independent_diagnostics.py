#!/usr/bin/env python3
"""Independent Specialist Diagnostics CLI.

This script runs diagnostics for individual specialist models without
dependencies on the meta-labeling pipeline or get_specialist_models_outputs.

Usage:
    python scripts/specialist_independent_diagnostics.py --specialist ml_momentum_persistence_step --symbol ETHUSDT --exchange binance --timeframe 15m --direction long
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.steps.base_step import step_registry


def run_specialist_diagnostics(specialist_name: str, symbol: str, exchange: str, 
                             timeframe: str, direction: str):
    """Run diagnostics for a specific specialist."""
    try:
        # Get specialist step from registry
        step_class = step_registry.get_step(specialist_name)
        if not step_class:
            print(f"❌ Specialist '{specialist_name}' not found in registry")
            print("Available specialists:")
            for name in sorted(step_registry.list_steps()):
                if any(keyword in name for keyword in ['ml_', 'regime', 'momentum', 'volatility', 'liquidity', 'breakout', 'path', 'reversion', 'smc', 'volume']):
                    print(f"  - {name}")
            return False
        
        # Instantiate and run diagnostics
        specialist = step_class()
        print(f"🔍 Running diagnostics for {specialist_name}...")
        
        result = specialist.run_diagnostics(symbol, exchange, timeframe, direction)
        
        if result.get('success'):
            print(f"✅ Diagnostics completed successfully!")
            print(f"📊 Report: {result.get('report_path')}")
            print(f"📈 Metrics: {result.get('csv_path')}")
            
            # Print key metrics
            metrics = result.get('metrics', {})
            if metrics:
                print("\n📊 Key Metrics:")
                for metric, value in metrics.items():
                    if isinstance(value, float):
                        print(f"  {metric}: {value:.4f}")
                    else:
                        print(f"  {metric}: {value}")
            
            return True
        else:
            print(f"❌ Diagnostics failed: {result.get('error')}")
            return False
            
    except Exception as e:
        print(f"❌ Error running diagnostics: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run independent diagnostics for specialist models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available specialists:
  ml_momentum_persistence_step    - Momentum persistence specialist
  ml_volatility_burst_step       - Volatility burst specialist
  ml_risk_regime_step            - Risk regime specialist
  ml_liquidity_regime_step       - Liquidity regime specialist
  ml_breakout_bounce_regime_step - Breakout/bounce specialist
  ml_path_regime_step            - Path regime specialist
  ml_reversion_regime_step       - Mean reversion specialist
  ml_smc_regime_step             - SMC regime specialist
  ml_volume_force_step          - Volume force specialist

Examples:
  python scripts/specialist_independent_diagnostics.py --specialist ml_momentum_persistence_step --symbol ETHUSDT --exchange binance --timeframe 15m --direction long
  
  python scripts/specialist_independent_diagnostics.py --specialist ml_volatility_burst_step --symbol ETHUSDT --exchange binance --timeframe 15m --direction long
        """
    )
    
    parser.add_argument(
        '--specialist',
        type=str,
        required=True,
        help='Name of the specialist to diagnose'
    )
    
    parser.add_argument(
        '--symbol',
        type=str,
        default='ETHUSDT',
        help='Trading symbol (default: ETHUSDT)'
    )
    
    parser.add_argument(
        '--exchange',
        type=str,
        default='binance',
        help='Exchange name (default: binance)'
    )
    
    parser.add_argument(
        '--timeframe',
        type=str,
        default='15m',
        help='Timeframe (default: 15m)'
    )
    
    parser.add_argument(
        '--direction',
        type=str,
        default='long',
        choices=['long', 'short'],
        help='Trading direction (default: long)'
    )
    
    args = parser.parse_args()
    
    print(f"🚀 Starting independent specialist diagnostics...")
    print(f"📈 Specialist: {args.specialist}")
    print(f"💱 Symbol: {args.symbol}/{args.exchange}")
    print(f"⏱️ Timeframe: {args.timeframe}")
    print(f"📊 Direction: {args.direction}")
    print()
    
    success = run_specialist_diagnostics(
        args.specialist,
        args.symbol,
        args.exchange,
        args.timeframe,
        args.direction
    )
    
    if success:
        print("\n✅ Specialist diagnostics completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Specialist diagnostics failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
