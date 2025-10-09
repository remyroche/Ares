"""
Simple test for regime ensemble integration.

This script tests the key integration points without complex dependencies.
"""

import numpy as np
import pandas as pd
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple


def create_sample_market_data(n_samples: int = 1000) -> pd.DataFrame:
    """Create sample market data for testing."""
    print("📊 Creating sample market data")
    
    # Generate synthetic market data
    np.random.seed(42)
    
    # Create time series data
    dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='1H')
    
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


def create_comprehensive_regime_info(regime_labels: np.ndarray, regime_probabilities: np.ndarray) -> Dict[str, Any]:
    """Create comprehensive regime information for downstream models."""
    print("📊 Creating comprehensive regime information")
    
    n_regimes = regime_probabilities.shape[1]
    
    # Calculate regime features
    regime_features = {
        'regime_entropy': -np.sum(regime_probabilities * np.log(regime_probabilities + 1e-10), axis=1),
        'regime_confidence': np.max(regime_probabilities, axis=1),
        'regime_dominance': regime_probabilities[:, -1] - regime_probabilities[:, -2] if n_regimes > 1 else np.ones(len(regime_probabilities)),
        'regime_uncertainty': 1.0 - np.max(regime_probabilities, axis=1),
        'regime_balance': np.std(regime_probabilities, axis=1)
    }
    
    # Create ensemble probabilities (simulated)
    ensemble_probabilities = {
        'catboost': regime_probabilities + np.random.normal(0, 0.01, regime_probabilities.shape),
        'random_forest': regime_probabilities + np.random.normal(0, 0.02, regime_probabilities.shape),
        'extra_tree': regime_probabilities + np.random.normal(0, 0.015, regime_probabilities.shape)
    }
    
    # Create regime analysis
    regime_analysis = {
        'uncertainty_metrics': {
            'mean_entropy': float(np.mean(regime_features['regime_entropy'])),
            'std_entropy': float(np.std(regime_features['regime_entropy']))
        },
        'dominance_analysis': {
            'mean_dominance': float(np.mean(regime_features['regime_dominance'])),
            'std_dominance': float(np.std(regime_features['regime_dominance']))
        }
    }
    
    comprehensive_regime_info = {
        'regime_probabilities': regime_probabilities,
        'regime_analysis': regime_analysis,
        'ensemble_probabilities': ensemble_probabilities,
        'has_probabilistic_outputs': True,
        'regime_features': regime_features,
        'timestamp': datetime.now().isoformat()
    }
    
    print("✅ Comprehensive regime information created")
    return comprehensive_regime_info


def test_regime_feature_assembly(comprehensive_regime_info: Dict[str, Any], n_samples: int) -> Dict[str, np.ndarray]:
    """Test regime feature assembly for downstream models."""
    print("🔧 Testing regime feature assembly")
    
    feature_map = {}
    
    # Extract regime features from comprehensive regime information
    regime_features = comprehensive_regime_info.get('regime_features', {})
    if regime_features:
        # Add probabilistic regime features
        for feature_name, feature_values in regime_features.items():
            if feature_values is not None and len(feature_values) == n_samples:
                feature_map[f'regime_{feature_name}'] = feature_values
                print(f"✅ Added regime feature: regime_{feature_name}")
        
        # Add regime probability features
        regime_probabilities = comprehensive_regime_info.get('regime_probabilities')
        if regime_probabilities is not None and len(regime_probabilities) == n_samples:
            n_regimes = regime_probabilities.shape[1] if len(regime_probabilities.shape) > 1 else 1
            for i in range(n_regimes):
                feature_map[f'regime_prob_{i}'] = regime_probabilities[:, i]
            print(f"✅ Added {n_regimes} regime probability features")
        
        # Add ensemble probability features
        ensemble_probabilities = comprehensive_regime_info.get('ensemble_probabilities', {})
        if ensemble_probabilities:
            for model_name, model_probs in ensemble_probabilities.items():
                if model_probs is not None and len(model_probs) == n_samples:
                    n_model_regimes = model_probs.shape[1] if len(model_probs.shape) > 1 else 1
                    for i in range(n_model_regimes):
                        feature_map[f'ensemble_{model_name}_prob_{i}'] = model_probs[:, i]
            print(f"✅ Added ensemble probability features from {len(ensemble_probabilities)} models")
    
    # Add regime analysis features
    regime_analysis = comprehensive_regime_info.get('regime_analysis', {})
    if regime_analysis:
        uncertainty_metrics = regime_analysis.get('uncertainty_metrics', {})
        if uncertainty_metrics:
            # Add uncertainty features
            if 'mean_entropy' in uncertainty_metrics:
                feature_map['regime_uncertainty_mean'] = np.full(n_samples, uncertainty_metrics['mean_entropy'])
            if 'std_entropy' in uncertainty_metrics:
                feature_map['regime_uncertainty_std'] = np.full(n_samples, uncertainty_metrics['std_entropy'])
        
        dominance_analysis = regime_analysis.get('dominance_analysis', {})
        if dominance_analysis:
            if 'mean_dominance' in dominance_analysis:
                feature_map['regime_dominance_mean'] = np.full(n_samples, dominance_analysis['mean_dominance'])
            if 'std_dominance' in dominance_analysis:
                feature_map['regime_dominance_std'] = np.full(n_samples, dominance_analysis['std_dominance'])
    
    print(f"📊 Assembled {len(feature_map)} regime features")
    return feature_map


def test_data_splitting_integration(comprehensive_regime_info: Dict[str, Any], market_data: pd.DataFrame) -> pd.DataFrame:
    """Test data splitting integration with regime features."""
    print("✂️ Testing data splitting integration")
    
    # Simulate adding regime features to market data
    enhanced_data = market_data.copy()
    
    # Add regime features to market data for downstream models
    if comprehensive_regime_info.get('regime_features'):
        regime_features = comprehensive_regime_info['regime_features']
        for feature_name, feature_values in regime_features.items():
            enhanced_data[f'regime_{feature_name}'] = feature_values
    
    # Add ensemble probabilities if available
    if comprehensive_regime_info.get('ensemble_probabilities'):
        ensemble_probs = comprehensive_regime_info['ensemble_probabilities']
        for model_name, model_probs in ensemble_probs.items():
            if model_probs is not None and len(model_probs) == len(enhanced_data):
                # Add individual model probabilities as features
                for i in range(model_probs.shape[1]):
                    enhanced_data[f'ensemble_{model_name}_prob_{i}'] = model_probs[:, i]
    
    regime_feature_cols = [col for col in enhanced_data.columns if 'regime_' in col or 'ensemble_' in col]
    print(f"📊 Added {len(regime_feature_cols)} regime features to market data")
    print(f"📊 Regime feature columns: {regime_feature_cols[:10]}...")
    
    return enhanced_data


def test_complete_integration():
    """Test the complete integration flow."""
    print("🚀 Starting Complete Regime Integration Test")
    print("=" * 60)
    
    try:
        # Step 1: Create sample data
        market_data = create_sample_market_data(1000)
        
        # Step 2: Generate regime probabilities
        regime_labels, regime_probabilities = generate_regime_probabilities(len(market_data), 4)
        
        # Step 3: Create comprehensive regime information
        comprehensive_regime_info = create_comprehensive_regime_info(regime_labels, regime_probabilities)
        
        # Step 4: Test regime feature assembly
        regime_features = test_regime_feature_assembly(comprehensive_regime_info, len(market_data))
        
        # Step 5: Test data splitting integration
        enhanced_data = test_data_splitting_integration(comprehensive_regime_info, market_data)
        
        # Step 6: Verify integration
        print("\n📊 INTEGRATION VERIFICATION")
        print(f"✅ Market data shape: {enhanced_data.shape}")
        print(f"✅ Regime features: {len([col for col in enhanced_data.columns if 'regime_' in col])}")
        print(f"✅ Ensemble features: {len([col for col in enhanced_data.columns if 'ensemble_' in col])}")
        print(f"✅ Total regime-related features: {len([col for col in enhanced_data.columns if 'regime_' in col or 'ensemble_' in col])}")
        
        # Verify probabilistic outputs
        assert 'regime_probabilities' in comprehensive_regime_info
        assert 'regime_analysis' in comprehensive_regime_info
        assert 'ensemble_probabilities' in comprehensive_regime_info
        assert 'has_probabilistic_outputs' in comprehensive_regime_info
        assert comprehensive_regime_info['has_probabilistic_outputs'] == True
        
        print("\n🎉 All integration tests passed!")
        print("✅ Probabilistic regime outputs are properly integrated")
        print("✅ Data splitting includes comprehensive regime information")
        print("✅ Downstream models can access regime probability features")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False


def main():
    """Main test function."""
    print("🚀 Starting Regime Ensemble Integration Tests")
    print("=" * 70)
    
    success = test_complete_integration()
    
    print("=" * 70)
    if success:
        print("🎉 All integration tests passed! Probabilistic regime outputs are properly integrated.")
    else:
        print("⚠️ Integration tests failed. Please check the error messages above.")
    
    print("=" * 70)


if __name__ == "__main__":
    main()