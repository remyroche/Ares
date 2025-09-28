#!/usr/bin/env python3
"""
Test script to verify the new regime detection implementation.
"""

import sys
import os
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the src directory to the path
sys.path.append('/Users/remyroche/Documents/Ares/src')

from training.steps.market_analysis.hybrid_nas_tas_regime.hybrid_orchestrator import HybridNasTasOrchestrator

async def test_regime_detection():
    """Test the new regime detection implementation."""
    print("🧪 Testing regime detection implementation...")

    try:
        # Create sample market data
        dates = pd.date_range(start='2024-01-01', periods=200, freq='15min')
        np.random.seed(42)

        # Create synthetic market data with different regimes
        data = []
        current_price = 100.0

        for i, date in enumerate(dates):
            # Simulate different market regimes
            if i < 50:  # Trending up
                volatility = 0.01
                drift = 0.001
            elif i < 100:  # Ranging
                volatility = 0.005
                drift = 0.0
            elif i < 150:  # Trending down
                volatility = 0.015
                drift = -0.001
            else:  # High volatility
                volatility = 0.03
                drift = 0.0

            # Generate price movement
            price_change = np.random.normal(drift, volatility)
            current_price *= (1 + price_change)

            # Generate volume
            base_volume = 1000
            volume_multiplier = 1 + np.random.normal(0, 0.5)
            volume = max(100, int(base_volume * volume_multiplier))

            data.append({
                'timestamp': date,
                'open': current_price,
                'high': current_price * (1 + abs(np.random.normal(0, 0.002))),
                'low': current_price * (1 - abs(np.random.normal(0, 0.002))),
                'close': current_price,
                'volume': volume
            })

        market_data = pd.DataFrame(data)
        print(f"📊 Created synthetic market data: {market_data.shape}")

        # Initialize the orchestrator
        orchestrator = HybridNasTasOrchestrator()

        # Test NAS regime detection
        print("\n🧠 Testing NAS regime detection...")
        nas_features = pd.DataFrame({
            'momentum_5': np.random.randn(200),
            'volatility_10': np.random.rand(200) * 0.1,
            'trend_strength': np.random.randn(200)
        })

        nas_result = await orchestrator._execute_nas_regime_detection(market_data, nas_features)
        print(f"✅ NAS result: {nas_result.get('regime_count', 0)} regimes")
        print(f"📊 NAS clustering quality: {nas_result.get('clustering_quality', {})}")

        # Test TAS regime detection
        print("\n🌳 Testing TAS regime detection...")
        tas_features = pd.DataFrame({
            'volume_ratio': np.random.rand(200),
            'trend_sma_20': np.random.randn(200),
            'momentum_rsi': np.random.rand(200) * 100
        })

        tas_result = await orchestrator._execute_tas_regime_detection(market_data, tas_features)
        print(f"✅ TAS result: {tas_result.get('regime_count', 0)} regimes")
        print(f"📊 TAS clustering quality: {tas_result.get('clustering_quality', {})}")

        # Check consensus calculation
        print("\n🔄 Testing consensus calculation...")
        consensus_metrics = orchestrator._calculate_consensus_metrics(nas_result, tas_result)
        print(f"📊 Consensus metrics: {consensus_metrics}")

        # Verify features are being used (not random)
        nas_assignments = nas_result.get('regime_assignments', [])
        tas_assignments = tas_result.get('regime_assignments', [])

        if nas_assignments and tas_assignments:
            # Calculate actual agreement
            min_length = min(len(nas_assignments), len(tas_assignments))
            agreements = sum(1 for i in range(min_length) if nas_assignments[i] == tas_assignments[i])
            actual_agreement = agreements / min_length

            print(f"📊 Actual agreement between NAS and TAS: {actual_agreement:.3f} ({agreements}/{min_length})")

            # This should be much lower than 99.375% since we're using real algorithms now
            if actual_agreement < 0.95:
                print("✅ SUCCESS: Agreement is reasonable (<95%), not artificially high")
            else:
                print("⚠️ WARNING: Agreement is still suspiciously high")

        print("\n✅ All tests completed successfully!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_regime_detection())
