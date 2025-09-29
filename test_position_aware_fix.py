#!/usr/bin/env python3
"""
Test script to verify the position aware trading fix
"""

import numpy as np
import pandas as pd
from src.training.steps.market_analysis.hybrid_nas_tas_regime.shared_utils.position_aware_trading import PositionAwareTradingAnalyzer

# Create test data with the dimensions that were causing issues
np.random.seed(42)
n_samples = 960  # The expected length
market_data = pd.DataFrame({
    'close': np.random.randn(n_samples).cumsum() + 100
})

# Create regime labels
regime_labels = np.random.randint(0, 3, n_samples)

# Create position directions (should be n_samples - 1 after alignment)
position_directions = np.random.choice([-1, 0, 1], size=n_samples-1)

print(f"Market data length: {len(market_data)}")
print(f"Regime labels length: {len(regime_labels)}")
print(f"Position directions length: {len(position_directions)}")

# Test the analyzer
analyzer = PositionAwareTradingAnalyzer()

try:
    result = analyzer.analyze_regime_position_performance(
        market_data, regime_labels, position_directions
    )
    print("✅ Position-aware analysis completed successfully!")
    print(f"   Overall win rate: {result['overall_analysis']['overall_win_rate']:.3f}")
    print(f"   Number of regimes analyzed: {len(result['regime_analyses'])}")

    # Check if we have regime analyses
    for regime_id, regime_analysis in result['regime_analyses'].items():
        print(f"   {regime_id}: {regime_analysis['regime_duration']} samples, "
              f"win rate: {regime_analysis['overall_win_rate']:.3f}")

except Exception as e:
    print(f"❌ Error in position-aware analysis: {e}")
    import traceback
    traceback.print_exc()
