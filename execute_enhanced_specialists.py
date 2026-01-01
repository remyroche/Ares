#!/usr/bin/env python3
"""Execute all 14 enhanced specialists to generate features."""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from ares import ARES

def execute_all_enhanced_specialists():
    """Execute all 14 enhanced specialists."""
    
    # List of all 14 enhanced specialists
    enhanced_specialists = [
        "enhanced_ml_momentum_persistence_step",
        "enhanced_ml_smc_regime_step", 
        "enhanced_ml_volatility_burst_step",
        "enhanced_ml_volume_force_step",
        "enhanced_ml_breakout_bounce_regime_step",
        "enhanced_ml_reversion_regime_step",
        "enhanced_xgb_macro_regime_step",
        "enhanced_ml_liquidity_regime_step",
        "enhanced_ml_path_regime_step",
        "enhanced_ml_risk_regime_step",
        "enhanced_xgb_meso_regime_step",
        "enhanced_ml_microstructure_step",
        "enhanced_ml_candlestick_step",
        "enhanced_ml_spectral_step"
    ]
    
    print("🚀 Executing all 14 enhanced specialists...")
    
    # Initialize ARES
    ares = ARES()
    
    # Common configuration
    config = {
        "symbol": "ETHUSDT",
        "exchange": "binance", 
        "timeframe": "15m",
        "direction": "long",
        "model": "analyst"
    }
    
    results = {}
    
    for specialist in enhanced_specialists:
        try:
            print(f"\n📊 Executing {specialist}...")
            
            # Execute specialist
            result = ares.run_step(specialist, config)
            
            # Store result info
            results[specialist] = {
                "status": "success",
                "features": len(result) if hasattr(result, '__len__') else "unknown"
            }
            
            print(f"✅ {specialist} - SUCCESS")
            
        except Exception as e:
            print(f"❌ {specialist} - FAILED: {str(e)[:100]}...")
            results[specialist] = {
                "status": "failed",
                "error": str(e)[:100]
            }
    
    print("\n🎉 ENHANCED SPECIALISTS EXECUTION SUMMARY")
    print("=" * 60)
    
    success_count = sum(1 for r in results.values() if r["status"] == "success")
    failed_count = len(results) - success_count
    
    print(f"✅ Successful: {success_count}/14")
    print(f"❌ Failed: {failed_count}/14")
    
    if failed_count > 0:
        print("\n❌ Failed specialists:")
        for specialist, result in results.items():
            if result["status"] == "failed":
                print(f"  - {specialist}: {result['error']}")
    
    print(f"\n🚀 Ready for enhanced MI diagnostics!")
    return results

if __name__ == "__main__":
    execute_all_enhanced_specialists()
