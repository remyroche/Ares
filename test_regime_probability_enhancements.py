#!/usr/bin/env python3
"""
Test script for regime detection ensemble ML model probability enhancements.

This script tests the enhanced regime detection and tagging functionality to ensure:
1. Ensemble models produce comprehensive probability information
2. Regime splitter tags data with all probability details
3. All probability metrics are calculated correctly
"""

import numpy as np
import pandas as pd
import sys
import os
from datetime import datetime, timedelta

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def create_test_market_data(n_samples=1000):
    """Create synthetic market data for testing."""
    np.random.seed(42)
    
    # Create timestamps
    start_time = datetime.now() - timedelta(days=30)
    timestamps = [start_time + timedelta(minutes=5*i) for i in range(n_samples)]
    
    # Create synthetic OHLCV data
    base_price = 100.0
    prices = [base_price]
    
    for i in range(1, n_samples):
        # Add some trend and volatility
        trend = 0.0001 * np.sin(i / 100)  # Slow trend
        volatility = 0.02 * np.random.normal(0, 1)  # Random volatility
        price_change = trend + volatility
        new_price = prices[-1] * (1 + price_change)
        prices.append(max(new_price, 1.0))  # Ensure positive prices
    
    # Create OHLCV data
    data = []
    for i, price in enumerate(prices):
        high = price * (1 + abs(np.random.normal(0, 0.01)))
        low = price * (1 - abs(np.random.normal(0, 0.01)))
        open_price = prices[i-1] if i > 0 else price
        close = price
        volume = np.random.uniform(1000, 10000)
        
        data.append({
            'timestamp': timestamps[i],
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    return pd.DataFrame(data)

def test_regime_probability_calculation():
    """Test the regime probability calculation functionality."""
    print("🧪 Testing regime probability calculation...")
    
    try:
        # Create test data
        market_data = create_test_market_data(100)
        print(f"✅ Created test market data: {market_data.shape}")
        
        # Create synthetic regime labels and probabilities
        n_regimes = 3
        n_samples = len(market_data)
        
        # Create realistic regime probabilities
        regime_labels = np.random.randint(0, n_regimes, n_samples)
        regime_probabilities = np.random.dirichlet(np.ones(n_regimes), n_samples)
        
        # Normalize to ensure they sum to 1
        regime_probabilities = regime_probabilities / np.sum(regime_probabilities, axis=1, keepdims=True)
        
        print(f"✅ Created synthetic regime data: {n_regimes} regimes, {n_samples} samples")
        print(f"📊 Probability shape: {regime_probabilities.shape}")
        print(f"📈 Probability range: [{regime_probabilities.min():.4f}, {regime_probabilities.max():.4f}]")
        
        # Test the tagging functionality
        from src.training.steps.market_analysis.regime_data_splitting.regime_data_splitting_main import RegimeDataSplittingStep
        
        # Create a minimal config for testing
        config = {
            'n_regimes': n_regimes,
            'enable_parquet_partitioning': False,
            'columnar_optimization': False
        }
        
        # Initialize the regime splitting step
        regime_splitter = RegimeDataSplittingStep(config)
        
        # Test the tagging method
        tagged_data = regime_splitter.tag_data_with_regime_probabilities(
            market_data, regime_labels, regime_probabilities
        )
        
        print(f"✅ Data tagged successfully: {tagged_data.shape}")
        
        # Verify all expected columns are present
        expected_columns = [
            'composite_cluster_id', 'regime_probabilities',
            'regime_0_probability', 'regime_1_probability', 'regime_2_probability',
            'regime_confidence', 'regime_stability', 'regime_entropy',
            'regime_dominance', 'regime_transition', 'regime_duration',
            'regime_quality_score', 'regime_uncertainty', 'regime_consistency'
        ]
        
        missing_columns = [col for col in expected_columns if col not in tagged_data.columns]
        if missing_columns:
            print(f"❌ Missing columns: {missing_columns}")
            return False
        
        print("✅ All expected probability columns present")
        
        # Verify probability calculations
        print("\n📊 Verifying probability calculations...")
        
        # Check that individual regime probabilities match the original
        for i in range(n_regimes):
            col_name = f'regime_{i}_probability'
            if col_name in tagged_data.columns:
                original_probs = regime_probabilities[:, i]
                tagged_probs = tagged_data[col_name].values
                if np.allclose(original_probs, tagged_probs, rtol=1e-10):
                    print(f"✅ {col_name} matches original probabilities")
                else:
                    print(f"❌ {col_name} does not match original probabilities")
                    return False
        
        # Check confidence scores
        expected_confidence = np.max(regime_probabilities, axis=1)
        actual_confidence = tagged_data['regime_confidence'].values
        if np.allclose(expected_confidence, actual_confidence, rtol=1e-10):
            print("✅ Confidence scores calculated correctly")
        else:
            print("❌ Confidence scores do not match expected values")
            return False
        
        # Check entropy calculation
        expected_entropy = -np.sum(regime_probabilities * np.log(regime_probabilities + 1e-10), axis=1)
        actual_entropy = tagged_data['regime_entropy'].values
        if np.allclose(expected_entropy, actual_entropy, rtol=1e-5):
            print("✅ Entropy calculated correctly")
        else:
            print("❌ Entropy does not match expected values")
            return False
        
        # Check regime transitions
        expected_transitions = np.concatenate([[False], regime_labels[1:] != regime_labels[:-1]])
        actual_transitions = tagged_data['regime_transition'].values
        if np.array_equal(expected_transitions, actual_transitions):
            print("✅ Regime transitions calculated correctly")
        else:
            print("❌ Regime transitions do not match expected values")
            return False
        
        print("\n🎉 All probability calculations verified successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_ensemble_model_probabilities():
    """Test the enhanced ensemble model probability prediction."""
    print("\n🧪 Testing ensemble model probability prediction...")
    
    try:
        # Create test market data
        market_data = create_test_market_data(50)
        
        # Create synthetic ensemble result
        n_regimes = 4
        n_samples = len(market_data)
        
        # Simulate ensemble model predictions
        regime_labels = np.random.randint(0, n_regimes, n_samples)
        regime_probabilities = np.random.dirichlet(np.ones(n_regimes), n_samples)
        
        # Create mock ensemble result
        ensemble_result = {
            'stacker_lgbm_calibrated': type('MockModel', (), {
                'predict': lambda x: regime_labels,
                'predict_proba': lambda x: regime_probabilities
            })(),
            'metadata': {
                'feature_names': ['close', 'volume', 'open', 'high', 'low']
            }
        }
        
        # Test the enhanced prediction method
        from src.training.steps.market_analysis.regime_data_splitting.regime_data_splitting_main import RegimeDataSplittingStep
        
        config = {'n_regimes': n_regimes}
        regime_splitter = RegimeDataSplittingStep(config)
        
        # Test the prediction method
        prediction_result = await regime_splitter._predict_regimes_with_ensemble_model(
            ensemble_result, market_data
        )
        
        if prediction_result is None:
            print("❌ Prediction result is None")
            return False
        
        # Verify the result structure
        expected_keys = ['labels', 'probabilities', 'probability_info']
        for key in expected_keys:
            if key not in prediction_result:
                print(f"❌ Missing key in prediction result: {key}")
                return False
        
        print("✅ Prediction result structure verified")
        
        # Verify probability info structure
        prob_info = prediction_result['probability_info']
        expected_prob_keys = [
            'raw_probabilities', 'regime_labels', 'confidence_scores',
            'n_regimes', 'regime_counts', 'regime_percentages',
            'avg_regime_probabilities', 'regime_stability', 'prediction_metadata'
        ]
        
        for key in expected_prob_keys:
            if key not in prob_info:
                print(f"❌ Missing key in probability info: {key}")
                return False
        
        print("✅ Probability info structure verified")
        
        # Verify calculations
        if prob_info['n_regimes'] == n_regimes:
            print("✅ Number of regimes correct")
        else:
            print(f"❌ Number of regimes incorrect: expected {n_regimes}, got {prob_info['n_regimes']}")
            return False
        
        if len(prob_info['regime_counts']) == n_regimes:
            print("✅ Regime counts length correct")
        else:
            print(f"❌ Regime counts length incorrect: expected {n_regimes}, got {len(prob_info['regime_counts'])}")
            return False
        
        print("🎉 Ensemble model probability prediction test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Ensemble model test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all tests."""
    print("🚀 Starting regime probability enhancement tests...")
    print("=" * 60)
    
    # Test 1: Regime probability calculation
    test1_passed = test_regime_probability_calculation()
    
    # Test 2: Ensemble model probabilities
    test2_passed = await test_ensemble_model_probabilities()
    
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    print(f"   Regime Probability Calculation: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"   Ensemble Model Probabilities: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! Regime probability enhancements are working correctly.")
        return True
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    import asyncio
    success = asyncio.run(main())
    sys.exit(0 if success else 1)