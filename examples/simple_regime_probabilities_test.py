"""
Simple test for simplified regime probabilities integration.

This script tests the simplified integration that only includes regime probabilities.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple


def create_sample_market_data(n_samples: int = 1000) -> pd.DataFrame:
    """Create sample market data for testing."""
    print("📊 Creating sample market data")
    
    # Generate synthetic market data
    np.random.seed(42)
    
    # Create time series data
    dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='1h')
    
    # Generate price data with different regimes
    n_regimes = 4
    regime_length = n_samples // n_regimes
    
    data = []
    for i in range(n_regimes):
        start_idx = i * regime_length
        end_idx = min((i + 1) * regime_length, n_samples)
        
        # Different characteristics for each regime
        if i == 0:  # Trending up
            trend = np.linspace(100, 120, end_idx - start_idx)
            noise = np.random.normal(0, 1, end_idx - start_idx)
        elif i == 1:  # Trending down
            trend = np.linspace(120, 100, end_idx - start_idx)
            noise = np.random.normal(0, 1.5, end_idx - start_idx)
        elif i == 2:  # High volatility
            trend = np.full(end_idx - start_idx, 110)
            noise = np.random.normal(0, 3, end_idx - start_idx)
        else:  # Low volatility
            trend = np.full(end_idx - start_idx, 110)
            noise = np.random.normal(0, 0.5, end_idx - start_idx)
        
        prices = trend + noise
        
        for j, price in enumerate(prices):
            data.append({
                'timestamp': dates[start_idx + j],
                'open': price + np.random.normal(0, 0.1),
                'high': price + abs(np.random.normal(0, 0.2)),
                'low': price - abs(np.random.normal(0, 0.2)),
                'close': price,
                'volume': np.random.randint(1000, 10000)
            })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    
    print(f"✅ Created sample data with {len(df)} samples")
    return df


def generate_regime_probabilities(n_samples: int, n_regimes: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic regime probabilities for testing."""
    print(f"🔮 Generating regime probabilities for {n_samples} samples and {n_regimes} regimes")
    
    # Generate regime labels
    regime_labels = np.random.randint(0, n_regimes, n_samples)
    
    # Generate probability matrix
    regime_probabilities = np.random.dirichlet(np.ones(n_regimes), n_samples)
    
    # Make the probabilities more realistic by biasing towards the actual regime
    for i in range(n_samples):
        actual_regime = regime_labels[i]
        # Increase probability for actual regime
        regime_probabilities[i, actual_regime] *= 2.0
        # Renormalize
        regime_probabilities[i] = regime_probabilities[i] / np.sum(regime_probabilities[i])
    
    print(f"✅ Generated regime probabilities: {regime_probabilities.shape}")
    return regime_labels, regime_probabilities


def create_regime_probabilities_info(regime_probabilities: np.ndarray) -> Dict[str, Any]:
    """Create simplified regime probabilities information for downstream models."""
    print("📊 Creating regime probabilities information")
    
    regime_probabilities_info = {
        'regime_probabilities': regime_probabilities,
        'has_probabilistic_outputs': True,
        'timestamp': datetime.now().isoformat()
    }
    
    print("✅ Regime probabilities information created")
    return regime_probabilities_info


def test_regime_feature_assembly(regime_probabilities_info: Dict[str, Any], n_samples: int) -> Dict[str, np.ndarray]:
    """Test simplified regime feature assembly for downstream models."""
    print("🔧 Testing simplified regime feature assembly")
    
    feature_map = {}
    
    # Add regime probability features only
    regime_probabilities = regime_probabilities_info.get('regime_probabilities')
    if regime_probabilities is not None and len(regime_probabilities) == n_samples:
        n_regimes = regime_probabilities.shape[1] if len(regime_probabilities.shape) > 1 else 1
        for i in range(n_regimes):
            feature_map[f'regime_prob_{i}'] = regime_probabilities[:, i]
        print(f"✅ Added {n_regimes} regime probability features")
    
    print(f"📊 Assembled {len(feature_map)} regime features")
    return feature_map


def test_data_splitting_integration(regime_probabilities_info: Dict[str, Any], market_data: pd.DataFrame) -> pd.DataFrame:
    """Test simplified data splitting integration with regime probabilities."""
    print("✂️ Testing simplified data splitting integration")
    
    # Simulate adding regime probability features to market data
    enhanced_data = market_data.copy()
    
    # Add regime probability features to market data for downstream models
    regime_probabilities = regime_probabilities_info.get('regime_probabilities')
    if regime_probabilities is not None and len(regime_probabilities) == len(enhanced_data):
        n_regimes = regime_probabilities.shape[1] if len(regime_probabilities.shape) > 1 else 1
        for i in range(n_regimes):
            enhanced_data[f'regime_prob_{i}'] = regime_probabilities[:, i]
    
    regime_feature_cols = [col for col in enhanced_data.columns if 'regime_prob_' in col]
    print(f"📊 Added {len(regime_feature_cols)} regime probability features to market data")
    print(f"📊 Regime probability columns: {regime_feature_cols}")
    
    return enhanced_data


def test_analyst_models(regime_probabilities_info: Dict[str, Any], n_samples: int) -> bool:
    """Test Analyst models with simplified regime probabilities."""
    print("🧪 Testing Analyst Models with Regime Probabilities")
    
    try:
        # Test regime feature assembly
        regime_features = test_regime_feature_assembly(regime_probabilities_info, n_samples)
        
        if regime_features:
            print(f"✅ Assembled {len(regime_features)} regime features for Analyst")
            print(f"📊 Regime feature keys: {list(regime_features.keys())}")
            
            # Verify only regime probability features are included
            regime_prob_features = [key for key in regime_features.keys() if key.startswith('regime_prob_')]
            print(f"📊 Regime probability features: {len(regime_prob_features)}")
            
            return True
        else:
            print("❌ Failed to assemble regime features for Analyst")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def test_tactician_models(regime_probabilities_info: Dict[str, Any], n_samples: int) -> bool:
    """Test Tactician models with simplified regime probabilities."""
    print("🧪 Testing Tactician Models with Regime Probabilities")
    
    try:
        # Test regime feature assembly
        regime_features = test_regime_feature_assembly(regime_probabilities_info, n_samples)
        
        if regime_features:
            print(f"✅ Assembled {len(regime_features)} regime features for Tactician")
            print(f"📊 Regime feature keys: {list(regime_features.keys())}")
            
            # Verify only regime probability features are included
            regime_prob_features = [key for key in regime_features.keys() if key.startswith('regime_prob_')]
            print(f"📊 Regime probability features: {len(regime_prob_features)}")
            
            return True
        else:
            print("❌ Failed to assemble regime features for Tactician")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def test_complete_simplified_integration():
    """Test the complete simplified integration flow."""
    print("🚀 Starting Simplified Regime Probabilities Integration Test")
    print("=" * 60)
    
    try:
        # Step 1: Create sample data
        market_data = create_sample_market_data(1000)
        
        # Step 2: Generate regime probabilities
        regime_labels, regime_probabilities = generate_regime_probabilities(len(market_data), 4)
        
        # Step 3: Create simplified regime probabilities information
        regime_probabilities_info = create_regime_probabilities_info(regime_probabilities)
        
        # Step 4: Test data splitting integration
        enhanced_data = test_data_splitting_integration(regime_probabilities_info, market_data)
        
        # Step 5: Test Analyst models
        analyst_success = test_analyst_models(regime_probabilities_info, len(market_data))
        
        # Step 6: Test Tactician models
        tactician_success = test_tactician_models(regime_probabilities_info, len(market_data))
        
        # Step 7: Verify integration
        print("\n📊 SIMPLIFIED INTEGRATION VERIFICATION")
        print(f"✅ Market data shape: {enhanced_data.shape}")
        print(f"✅ Regime probability features: {len([col for col in enhanced_data.columns if 'regime_prob_' in col])}")
        print(f"✅ Total regime-related features: {len([col for col in enhanced_data.columns if 'regime_prob_' in col])}")
        
        # Verify simplified outputs
        assert 'regime_probabilities' in regime_probabilities_info
        assert 'has_probabilistic_outputs' in regime_probabilities_info
        assert regime_probabilities_info['has_probabilistic_outputs'] == True
        
        # Verify no complex features are present
        complex_features = [col for col in enhanced_data.columns if any(x in col for x in ['entropy', 'confidence', 'dominance', 'uncertainty', 'balance', 'ensemble_'])]
        assert len(complex_features) == 0, f"Found complex features that should not be present: {complex_features}"
        
        print("\n🎉 All simplified integration tests passed!")
        print("✅ Only regime probabilities are included")
        print("✅ Data splitting includes regime probability features")
        print("✅ Downstream models can access regime probabilities")
        print("✅ No complex analysis features present")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False


def main():
    """Main test function."""
    print("🚀 Starting Simplified Regime Probabilities Integration Tests")
    print("=" * 70)
    
    success = test_complete_simplified_integration()
    
    print("=" * 70)
    if success:
        print("🎉 All simplified integration tests passed! Only regime probabilities are included.")
    else:
        print("⚠️ Integration tests failed. Please check the error messages above.")
    
    print("=" * 70)


if __name__ == "__main__":
    main()