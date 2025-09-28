#!/usr/bin/env python3
"""
Test script to verify that the NAS/TAS contribution tracking fix works.
"""

import sys
import os
sys.path.append('/Users/remyroche/Documents/Ares/src')

from training.steps.market_analysis.hybrid_nas_tas_regime.hybrid_orchestrator import HybridOrchestrator, HybridOrchestratorConfig

def test_contribution_tracking():
    """Test that contribution tracking is working properly."""

    # Create a minimal config for testing
    config = HybridOrchestratorConfig(
        symbol="ETHUSDT",
        timeframe="15m",
        light_mode=True
    )

    # Create orchestrator
    orchestrator = HybridOrchestrator(config)

    # Mock some test data
    import numpy as np
    import pandas as pd

    # Create fake market data
    dates = pd.date_range('2023-01-01', periods=100, freq='15min')
    market_data = pd.DataFrame({
        'open': np.random.uniform(1000, 2000, 100),
        'high': np.random.uniform(1000, 2000, 100),
        'low': np.random.uniform(1000, 2000, 100),
        'close': np.random.uniform(1000, 2000, 100),
        'volume': np.random.uniform(1000, 10000, 100)
    }, index=dates)

    # Mock TAS and NAS results
    tas_result = {
        'success': True,
        'regime_predictions': np.array([0, 1, 0, 1, 0] * 20),  # 100 predictions
        'regime_count': 2,
        'features': np.random.random((100, 5)),
        'results': {'confidence': 0.8}
    }

    nas_result = {
        'success': True,
        'regime_predictions': np.array([1, 0, 1, 0, 1] * 20),  # 100 predictions (opposite of TAS)
        'regime_count': 2,
        'features': np.random.random((100, 5)),
        'results': {'confidence': 0.7}
    }

    # Test the contribution calculation
    hybrid_labels = np.array([0, 1, 0, 1, 0] * 20)  # Some hybrid result

    tas_contribution = orchestrator._calculate_system_contribution(tas_result, nas_result, hybrid_labels, 'tas')
    nas_contribution = orchestrator._calculate_system_contribution(nas_result, tas_result, hybrid_labels, 'nas')

    # Check that contributions are not empty
    print("TAS Contribution:", tas_contribution)
    print("NAS Contribution:", nas_contribution)

    # Verify key fields exist
    assert tas_contribution['system'] == 'TAS'
    assert nas_contribution['system'] == 'NAS'
    assert 'final_weight' in tas_contribution
    assert 'final_weight' in nas_contribution
    assert 'agreement_score' in tas_contribution
    assert 'feature_correlation' in tas_contribution

    print("✅ Contribution tracking test passed!")
    return True

if __name__ == "__main__":
    try:
        test_contribution_tracking()
        print("🎉 All tests passed!")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)
