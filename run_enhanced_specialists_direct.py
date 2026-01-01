#!/usr/bin/env python3
"""Direct execution of enhanced specialists to generate features."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

def run_enhanced_specialist(specialist_name: str, config: dict) -> pd.DataFrame:
    """Run a single enhanced specialist and return its predictions."""
    try:
        # Import the specialist class
        module_name = specialist_name.replace("enhanced_", "").replace("_step", "") + "_step_enhanced"
        class_name = "".join(word.capitalize() for word in module_name.split("_")) + "Step"
        
        # Handle special cases
        if "xgb" in specialist_name:
            class_name = "Enhanced" + class_name
        else:
            class_name = "Enhanced" + class_name.replace("Ml", "ML")
        
        module_path = f"src.training.steps.market_analysis.{module_name}_enhanced"
        
        exec(f"from {module_path} import {class_name}")
        
        # Initialize specialist
        specialist = eval(f"{class_name}()")
        
        # Load data (simplified - in reality this would load from artifacts)
        # For now, create dummy data to test execution
        dates = pd.date_range("2022-01-01", "2023-01-01", freq="15min")
        n_samples = len(dates)
        
        data = pd.DataFrame({
            "timestamp": dates,
            "open": np.random.randn(n_samples).cumsum() + 100,
            "high": np.random.randn(n_samples).cumsum() + 102,
            "low": np.random.randn(n_samples).cumsum() + 98,
            "close": np.random.randn(n_samples).cumsum() + 100,
            "volume": np.random.randint(1000, 10000, n_samples),
        })
        
        # Execute specialist
        result = specialist.execute(data, config)
        
        print(f"✅ {specialist_name} - Generated {len(result)} predictions")
        return result
        
    except Exception as e:
        print(f"❌ {specialist_name} - FAILED: {str(e)[:100]}...")
        return None

def main():
    """Run all 14 enhanced specialists."""
    
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
    
    config = {
        "symbol": "ETHUSDT",
        "exchange": "binance", 
        "timeframe": "15m",
        "direction": "long",
        "model": "analyst"
    }
    
    print("🚀 Running all 14 enhanced specialists...")
    
    results = {}
    for specialist in enhanced_specialists:
        result = run_enhanced_specialist(specialist, config)
        if result is not None:
            results[specialist] = result
    
    print(f"\n🎉 SUCCESS: {len(results)}/14 enhanced specialists executed")
    print("📈 Ready for enhanced MI analysis!")
    
    return results

if __name__ == "__main__":
    main()
