"""
Usage Example for Unified Regime Detector

This example demonstrates how to use the unified regime detector
for both TAS and NAS systems.
"""

import numpy as np
import pandas as pd
from src.utils.nas_tas.regime_detector import (
    create_tas_regime_detector,
    create_nas_regime_detector,
    create_hybrid_regime_detector,
    create_unified_regime_detector
)


def example_usage():
    """Example usage of the unified regime detector."""
    
    # Create sample market data
    np.random.seed(42)
    n_samples = 1000
    
    # Generate sample OHLCV data
    market_data = pd.DataFrame({
        'open': np.random.randn(n_samples).cumsum() + 100,
        'high': np.random.randn(n_samples).cumsum() + 105,
        'low': np.random.randn(n_samples).cumsum() + 95,
        'close': np.random.randn(n_samples).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, n_samples)
    })
    
    print("🚀 Unified Regime Detector Usage Example")
    print("=" * 50)
    
    # Example 1: TAS-specific regime detector
    print("\n1. TAS Regime Detector:")
    tas_detector = create_tas_regime_detector(n_regimes=5, primary_timeframe="1m")
    tas_result = tas_detector.detect_regimes(market_data)
    print(f"   Success: {tas_result.success}")
    print(f"   Regimes detected: {len(np.unique(tas_result.regime_predictions))}")
    print(f"   Economic significance: {np.mean(tas_result.economic_significance_scores):.3f}")
    
    # Example 2: NAS-specific regime detector
    print("\n2. NAS Regime Detector:")
    nas_detector = create_nas_regime_detector(n_regimes=5, primary_timeframe="1m")
    nas_result = nas_detector.detect_regimes(market_data)
    print(f"   Success: {nas_result.success}")
    print(f"   Regimes detected: {len(np.unique(nas_result.regime_predictions))}")
    print(f"   Trading viability: {np.mean(nas_result.trading_viability_scores):.3f}")
    
    # Example 3: Hybrid regime detector
    print("\n3. Hybrid TAS-NAS Regime Detector:")
    hybrid_detector = create_hybrid_regime_detector(n_regimes=5, primary_timeframe="1m")
    hybrid_result = hybrid_detector.detect_regimes(market_data)
    print(f"   Success: {hybrid_result.success}")
    print(f"   Regimes detected: {len(np.unique(hybrid_result.regime_predictions))}")
    print(f"   Regime stability: {np.mean(hybrid_result.regime_stability_scores):.3f}")
    
    # Example 4: Unified regime detector (recommended)
    print("\n4. Unified Regime Detector (Recommended):")
    unified_detector = create_unified_regime_detector(n_regimes=5, primary_timeframe="1m")
    unified_result = unified_detector.detect_regimes(market_data)
    print(f"   Success: {unified_result.success}")
    print(f"   Regimes detected: {len(np.unique(unified_result.regime_predictions))}")
    print(f"   Execution time: {unified_result.execution_time:.2f}s")
    print(f"   System type: {unified_result.system_type}")
    print(f"   Architecture: {unified_result.architecture_used}")
    
    # Example 5: Save and load results
    print("\n5. Save and Load Results:")
    save_path = "/tmp/unified_regime_result.pkl"
    success = unified_detector.save_results(unified_result, save_path)
    print(f"   Save successful: {success}")
    
    if success:
        loaded_result = unified_detector.load_results(save_path)
        print(f"   Load successful: {loaded_result is not None}")
        if loaded_result:
            print(f"   Loaded regimes: {len(np.unique(loaded_result.regime_predictions))}")
    
    print("\n✅ Example completed successfully!")
    print("\nKey Benefits:")
    print("- Single unified interface for both TAS and NAS systems")
    print("- Eliminates code duplication")
    print("- Consistent error handling and logging")
    print("- Easy to use with backward compatibility")
    print("- Hardware optimization support")
    print("- Economic significance validation")


if __name__ == "__main__":
    example_usage()