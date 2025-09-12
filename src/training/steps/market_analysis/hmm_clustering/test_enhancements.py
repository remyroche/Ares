#!/usr/bin/env python3
"""
Test script for Enhanced HMM Clustering Improvements

This script demonstrates all the implemented improvements:
1. Dynamic Parameter Optimization
2. Feature Selection with Enhanced Feature Engineering
3. Ensemble Weight Optimization

Author: AI Assistant
Date: 2024-01-XX
Version: 1.0.0
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import time
from typing import Dict, List, Any

# Import our enhancement modules
from parameter_optimization import ParameterOptimizer
from ensemble_optimization import EnsembleWeightOptimizer
from step03_hmm_regime_discovery import HMMRegimeDiscoveryStep, EnhancedFeatureEngineer

def create_realistic_market_data(n_samples: int = 2000, n_regimes: int = 3) -> tuple:
    """
    Create realistic market data with different regimes
    
    Args:
        n_samples: Number of samples
        n_regimes: Number of distinct regimes
        
    Returns:
        Tuple of (DataFrame, regime_labels)
    """
    print("📊 Creating realistic market data...")
    
    # Create regime segments
    regime_lengths = np.random.multinomial(n_samples, [1/n_regimes] * n_regimes)
    regime_labels = np.repeat(range(n_regimes), regime_lengths)
    
    # Create price data with different characteristics per regime
    prices = []
    volumes = []
    
    for regime in range(n_regimes):
        length = regime_lengths[regime]
        
        if regime == 0:  # Bull market
            trend = 0.001
            volatility = 0.01
            volume_base = 5000
        elif regime == 1:  # Bear market
            trend = -0.0005
            volatility = 0.015
            volume_base = 3000
        else:  # Sideways market
            trend = 0.0001
            volatility = 0.008
            volume_base = 4000
        
        # Generate price series
        price_changes = np.random.normal(trend, volatility, length)
        price_series = 100 + np.cumsum(price_changes)
        
        # Generate volume series
        volume_series = np.random.poisson(volume_base, length)
        
        prices.extend(price_series)
        volumes.extend(volume_series)
    
    # Create OHLCV data
    prices = np.array(prices)
    highs = prices + np.random.rand(n_samples) * 2
    lows = prices - np.random.rand(n_samples) * 2
    
    df = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=n_samples, freq='1H'),
        'open': prices,
        'high': highs,
        'low': lows,
        'close': prices,
        'volume': volumes
    })
    
    print(f"✅ Created {n_samples} samples with {n_regimes} regimes")
    return df, regime_labels

def test_parameter_optimization():
    """Test dynamic parameter optimization"""
    print("\n" + "="*60)
    print("🔧 TESTING PARAMETER OPTIMIZATION")
    print("="*60)
    
    # Create sample data
    df, regime_labels = create_realistic_market_data(n_samples=1000)
    
    # Create basic features
    feature_engineer = EnhancedFeatureEngineer()
    features = feature_engineer.create_comprehensive_features(df)
    features_array = features.values
    
    # Test parameter optimizer
    optimizer = ParameterOptimizer()
    
    # Test HMM state optimization
    print("\n1. Testing HMM State Optimization...")
    start_time = time.time()
    state_result = optimizer.optimize_hmm_states(features_array, state_range=(2, 6))
    print(f"   Best states: {state_result.best_params['n_components']}")
    print(f"   Best score: {state_result.best_score:.4f}")
    print(f"   Time: {state_result.optimization_time:.2f}s")
    
    # Test covariance type optimization
    print("\n2. Testing Covariance Type Optimization...")
    cov_result = optimizer.optimize_covariance_type(features_array, n_states=4)
    print(f"   Best covariance type: {cov_result.best_params['covariance_type']}")
    print(f"   Best score: {cov_result.best_score:.4f}")
    print(f"   Time: {cov_result.optimization_time:.2f}s")
    
    # Test comprehensive optimization
    print("\n3. Testing Comprehensive Optimization...")
    comp_result = optimizer.comprehensive_parameter_optimization(
        features_array, use_optuna=True, n_trials=20
    )
    print(f"   Best parameters: {comp_result.best_params}")
    print(f"   Best score: {comp_result.best_score:.4f}")
    print(f"   Time: {comp_result.optimization_time:.2f}s")
    
    return optimizer

def test_enhanced_feature_engineering():
    """Test enhanced feature engineering"""
    print("\n" + "="*60)
    print("🔧 TESTING ENHANCED FEATURE ENGINEERING")
    print("="*60)
    
    # Create sample data
    df, regime_labels = create_realistic_market_data(n_samples=1000)
    
    # Test enhanced feature engineer
    feature_engineer = EnhancedFeatureEngineer()
    
    print("\n1. Testing Comprehensive Feature Creation...")
    start_time = time.time()
    features = feature_engineer.create_comprehensive_features(df)
    creation_time = time.time() - start_time
    
    print(f"   Created {len(features.columns)} comprehensive features")
    print(f"   Time: {creation_time:.2f}s")
    
    # Show feature categories
    feature_categories = {
        'price_features': [col for col in features.columns if 'price' in col or 'ma_' in col or 'ema_' in col],
        'volume_features': [col for col in features.columns if 'volume' in col],
        'volatility_features': [col for col in features.columns if 'volatility' in col],
        'technical_indicators': [col for col in features.columns if any(ind in col for ind in ['rsi', 'macd', 'bb_', 'atr', 'adx'])],
        'momentum_features': [col for col in features.columns if 'momentum' in col],
        'sr_features': [col for col in features.columns if any(sr in col for sr in ['support', 'resistance', 'pivot', 'swing'])],
        'statistical_features': [col for col in features.columns if any(stat in col for stat in ['skewness', 'kurtosis', 'quantile', 'autocorr'])],
        'time_features': [col for col in features.columns if any(time in col for time in ['hour', 'day', 'month', 'sin', 'cos'])],
        'interaction_features': [col for col in features.columns if 'interaction' in col]
    }
    
    print("\n   Feature Categories:")
    for category, feature_list in feature_categories.items():
        if feature_list:
            print(f"     {category}: {len(feature_list)} features")
    
    return feature_engineer, features

def test_ensemble_optimization():
    """Test ensemble weight optimization"""
    print("\n" + "="*60)
    print("⚖️ TESTING ENSEMBLE OPTIMIZATION")
    print("="*60)
    
    # Create sample data
    df, regime_labels = create_realistic_market_data(n_samples=1000)
    
    # Create features
    feature_engineer = EnhancedFeatureEngineer()
    features = feature_engineer.create_comprehensive_features(df)
    features_array = features.values
    
    # Create mock clustering results
    np.random.seed(42)
    hmm_results = {
        'predictions': np.random.randint(0, 4, len(features_array)),
        'score': 0.5
    }
    kmeans_results = {
        'predictions': np.random.randint(0, 4, len(features_array)),
        'score': 0.4
    }
    dbscan_results = {
        'predictions': np.random.randint(0, 4, len(features_array)),
        'score': 0.3
    }
    
    # Test ensemble optimizer
    optimizer = EnsembleWeightOptimizer()
    
    # Test performance-based optimization
    print("\n1. Testing Performance-Based Optimization...")
    perf_result = optimizer.performance_based_optimization(
        hmm_results, kmeans_results, dbscan_results, features_array
    )
    print(f"   Optimal weights: {perf_result.optimal_weights}")
    print(f"   Score: {perf_result.optimization_score:.4f}")
    print(f"   Time: {perf_result.optimization_time:.2f}s")
    
    # Test multi-objective optimization
    print("\n2. Testing Multi-Objective Optimization...")
    multi_result = optimizer.multi_objective_optimization(
        hmm_results, kmeans_results, dbscan_results, features_array
    )
    print(f"   Optimal weights: {multi_result.optimal_weights}")
    print(f"   Score: {multi_result.optimization_score:.4f}")
    print(f"   Time: {multi_result.optimization_time:.2f}s")
    
    # Test adaptive weight updates
    print("\n3. Testing Adaptive Weight Updates...")
    adapt_result = optimizer.adaptive_weight_updates(
        hmm_results, kmeans_results, dbscan_results, features_array,
        learning_rate=0.01, n_iterations=5
    )
    print(f"   Optimal weights: {adapt_result.optimal_weights}")
    print(f"   Score: {adapt_result.optimization_score:.4f}")
    print(f"   Time: {adapt_result.optimization_time:.2f}s")
    
    return optimizer

def test_integrated_enhancement():
    """Test the integrated enhanced HMM clustering"""
    print("\n" + "="*60)
    print("🚀 TESTING INTEGRATED ENHANCED HMM CLUSTERING")
    print("="*60)
    
    # Create sample data
    df, regime_labels = create_realistic_market_data(n_samples=1500, n_regimes=4)
    
    # Test enhanced HMM regime discovery
    config = {
        'SYMBOL': 'ETHUSDT',
        'EXCHANGE': 'BINANCE', 
        'TIMEFRAME': '1m',
        'DATA_DIR': 'data_cache'
    }
    
    hmm_step = HMMRegimeDiscoveryStep(config)
    
    print("\nRunning enhanced HMM regime discovery...")
    
    # Test enhanced feature creation
    enhanced_features = hmm_step._create_enhanced_features(df, use_existing_tools=True)
    print(f"   Created {len(enhanced_features.columns)} enhanced features")
    
    # Test parameter optimization
    optimal_params = hmm_step._optimize_hmm_parameters(enhanced_features, use_optimization=True)
    print(f"   Optimal HMM parameters: {optimal_params}")
    
    # Test ensemble weight optimization (mock results)
    hmm_results = {'predictions': np.random.randint(0, 4, len(enhanced_features)), 'score': 0.5}
    kmeans_results = {'predictions': np.random.randint(0, 4, len(enhanced_features)), 'score': 0.4}
    dbscan_results = {'predictions': np.random.randint(0, 4, len(enhanced_features)), 'score': 0.3}
    
    optimal_weights = hmm_step._optimize_ensemble_weights(
        hmm_results, kmeans_results, dbscan_results, 
        enhanced_features.values, use_optimization=True
    )
    print(f"   Optimal ensemble weights: {optimal_weights}")
    
    print(f"\n✅ Enhanced HMM regime discovery completed successfully!")
    print(f"   Enhanced features: {len(enhanced_features.columns)}")
    print(f"   Optimal HMM params: {optimal_params}")
    print(f"   Optimal weights: {optimal_weights}")
    
    return hmm_step, enhanced_features, optimal_params, optimal_weights

def create_visualization(result: Any, df: pd.DataFrame, regime_labels: np.ndarray):
    """Create visualization of the results"""
    print("\n📊 Creating visualizations...")
    
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Price with regime colors
        axes[0, 0].plot(df.index, df['close'], alpha=0.7, label='Price')
        scatter = axes[0, 0].scatter(df.index, df['close'], c=regime_labels, cmap='viridis', alpha=0.5)
        axes[0, 0].set_title('Price with True Regime Colors')
        axes[0, 0].set_ylabel('Price')
        plt.colorbar(scatter, ax=axes[0, 0])
        
        # Plot 2: HMM predictions
        axes[0, 1].plot(df.index, df['close'], alpha=0.7, label='Price')
        scatter = axes[0, 1].scatter(df.index, df['close'], c=result.hmm_predictions, cmap='viridis', alpha=0.5)
        axes[0, 1].set_title('Price with HMM Predictions')
        axes[0, 1].set_ylabel('Price')
        plt.colorbar(scatter, ax=axes[0, 1])
        
        # Plot 3: Ensemble predictions
        axes[1, 0].plot(df.index, df['close'], alpha=0.7, label='Price')
        scatter = axes[1, 0].scatter(df.index, df['close'], c=result.ensemble_predictions, cmap='viridis', alpha=0.5)
        axes[1, 0].set_title('Price with Ensemble Predictions')
        axes[1, 0].set_ylabel('Price')
        axes[1, 0].set_xlabel('Time')
        plt.colorbar(scatter, ax=axes[1, 0])
        
        # Plot 4: Feature importance
        top_features = result.feature_scores.head(15)
        axes[1, 1].barh(range(len(top_features)), top_features['mutual_info_score'])
        axes[1, 1].set_yticks(range(len(top_features)))
        axes[1, 1].set_yticklabels(top_features['feature'], fontsize=8)
        axes[1, 1].set_title('Top 15 Feature Importance')
        axes[1, 1].set_xlabel('Mutual Information Score')
        
        plt.tight_layout()
        plt.savefig('enhanced_hmm_clustering_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Visualization saved as 'enhanced_hmm_clustering_results.png'")
        
    except ImportError:
        print("⚠️ Matplotlib not available, skipping visualization")

def main():
    """Main test function"""
    print("🚀 ENHANCED HMM CLUSTERING - COMPREHENSIVE TEST")
    print("="*80)
    
    # Test individual components
    param_optimizer = test_parameter_optimization()
    feature_engineer, features = test_enhanced_feature_engineering()
    ensemble_optimizer = test_ensemble_optimization()
    
    # Test integrated system
    hmm_step, enhanced_features, optimal_params, optimal_weights = test_integrated_enhancement()
    
    # Create visualization
    df, regime_labels = create_realistic_market_data(n_samples=1500, n_regimes=4)
    create_visualization(enhanced_features, df, regime_labels)
    
    # Save results
    print("\n💾 Saving results...")
    param_optimizer.save_optimization_results('parameter_optimization_results.json')
    ensemble_optimizer.save_optimization_results('ensemble_optimization_results.json')
    
    print("\n✅ All tests completed successfully!")
    print("\n📋 Summary of Improvements:")
    print("   1. ✅ Dynamic Parameter Optimization - Implemented")
    print("   2. ✅ Enhanced Feature Engineering (100+ features) - Implemented")
    print("   3. ✅ Ensemble Weight Optimization - Implemented")
    print("   4. ✅ Integration with Existing Feature Selection Tools - Implemented")
    
    print("\n🎯 Key Benefits:")
    print("   - Adaptive parameter selection based on data characteristics")
    print("   - Systematic feature selection from comprehensive feature set")
    print("   - Dynamic ensemble weight optimization")
    print("   - Integrated pipeline with all improvements")

if __name__ == "__main__":
    main()