"""
Test script for VectorBT integration in feature selection.

This script demonstrates the usage of all three VectorBT enhancements:
1. Enhanced Financial Feature Importance
2. Time Series Aware Feature Selection  
3. Advanced Correlation Analysis
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Import VectorBT components
from .vectorbt_importance_analyzer import VectorBTImportanceAnalyzer, VectorBTImportanceConfig
from .vectorbt_directional_selector import VectorBTDirectionalSelector, VectorBTDirectionalConfig
from .vectorbt_correlation_analyzer import VectorBTCorrelationAnalyzer, VectorBTCorrelationConfig

def create_sample_data(n_samples=1000, n_features=50):
    """Create sample financial data for testing."""
    np.random.seed(42)
    
    # Generate price data with trend and volatility
    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
    
    # Generate price series with trend and volatility
    trend = np.linspace(100, 150, n_samples)
    noise = np.random.normal(0, 2, n_samples)
    prices = trend + noise + np.cumsum(np.random.normal(0, 0.5, n_samples))
    
    # Generate returns
    returns = np.diff(prices) / prices[:-1]
    returns = np.concatenate([[0], returns])  # Add initial return of 0
    
    # Generate feature matrix
    features = np.random.randn(n_samples, n_features)
    
    # Add some correlation structure
    for i in range(0, n_features, 5):
        if i + 1 < n_features:
            features[:, i+1] = 0.7 * features[:, i] + 0.3 * np.random.randn(n_samples)
    
    # Add some trend to some features
    for i in range(0, n_features, 10):
        features[:, i] += np.linspace(0, 2, n_samples)
    
    feature_names = [f"feature_{i}" for i in range(n_features)]
    
    return {
        'prices': pd.DataFrame({'price': prices}, index=dates),
        'returns': pd.Series(returns, index=dates),
        'features': pd.DataFrame(features, columns=feature_names, index=dates),
        'feature_names': feature_names
    }

def test_financial_importance_analyzer():
    """Test VectorBT Financial Importance Analyzer."""
    print("🔍 Testing VectorBT Financial Importance Analyzer...")
    
    # Create sample data
    data = create_sample_data()
    
    # Configure analyzer
    config = VectorBTImportanceConfig(
        include_technical_indicators=True,
        include_risk_metrics=True,
        include_performance_metrics=True,
        rsi_period=14,
        macd_fast=12,
        macd_slow=26
    )
    
    # Create analyzer
    analyzer = VectorBTImportanceAnalyzer(config)
    
    # Analyze financial importance
    result = analyzer.analyze_financial_importance(
        prices=data['prices'],
        returns=data['returns'],
        feature_names=data['feature_names']
    )
    
    print(f"✅ Market regime: {result.market_regime}")
    print(f"✅ Technical indicators: {len(result.technical_indicators)}")
    print(f"✅ Risk metrics: {len(result.risk_metrics)}")
    print(f"✅ Performance metrics: {len(result.performance_metrics)}")
    print(f"✅ Combined scores: {len(result.combined_scores)}")
    
    return result

def test_directional_selector():
    """Test VectorBT Directional Selector."""
    print("\n🔍 Testing VectorBT Directional Selector...")
    
    # Create sample data
    data = create_sample_data()
    
    # Configure selector
    config = VectorBTDirectionalConfig(
        enable_regime_detection=True,
        enable_temporal_analysis=True,
        enable_cross_asset=True,
        regime_window=50,
        max_features_per_regime=20
    )
    
    # Create selector
    selector = VectorBTDirectionalSelector(config)
    
    # Select features
    result = selector.select_features(
        features=data['features'],
        prices=data['prices'],
        returns=data['returns'],
        feature_names=data['feature_names']
    )
    
    print(f"✅ Selected features: {len(result.selected_features)}")
    print(f"✅ Regime: {result.regime_info.regime_type}")
    print(f"✅ Confidence: {result.regime_info.confidence:.3f}")
    print(f"✅ Temporal features: {len(result.temporal_features)}")
    print(f"✅ Cross-asset features: {len(result.cross_asset_features)}")
    
    return result

def test_correlation_analyzer():
    """Test VectorBT Correlation Analyzer."""
    print("\n🔍 Testing VectorBT Correlation Analyzer...")
    
    # Create sample data
    data = create_sample_data()
    
    # Configure analyzer
    config = VectorBTCorrelationConfig(
        enable_pearson=True,
        enable_spearman=True,
        enable_rolling_correlation=True,
        enable_lagged_correlation=True,
        enable_correlation_clustering=True,
        rolling_window=30,
        correlation_threshold=0.7
    )
    
    # Create analyzer
    analyzer = VectorBTCorrelationAnalyzer(config)
    
    # Analyze correlations
    result = analyzer.analyze_correlations(
        data=data['features'],
        feature_names=data['feature_names']
    )
    
    print(f"✅ Correlation matrix shape: {result.correlation_matrix.shape}")
    print(f"✅ Rolling correlations: {len(result.rolling_correlations)}")
    print(f"✅ Lagged correlations: {len(result.lagged_correlations)}")
    print(f"✅ Correlation clusters: {len(result.correlation_clusters)}")
    
    # Get high correlation pairs
    high_corr_pairs = analyzer.get_highly_correlated_pairs(
        result.correlation_matrix, threshold=0.8
    )
    print(f"✅ High correlation pairs: {len(high_corr_pairs)}")
    
    # Get summary
    summary = analyzer.get_correlation_summary(result)
    print(f"✅ Summary: {summary}")
    
    return result

def test_integrated_pipeline():
    """Test integrated VectorBT pipeline."""
    print("\n🔍 Testing Integrated VectorBT Pipeline...")
    
    # Create sample data
    data = create_sample_data()
    
    # Step 1: Correlation analysis
    corr_analyzer = VectorBTCorrelationAnalyzer()
    corr_result = corr_analyzer.analyze_correlations(
        data['features'], data['feature_names']
    )
    
    # Step 2: Financial importance analysis
    importance_analyzer = VectorBTImportanceAnalyzer()
    importance_result = importance_analyzer.analyze_financial_importance(
        data['prices'], data['returns'], data['feature_names']
    )
    
    # Step 3: Directional selection
    directional_selector = VectorBTDirectionalSelector()
    directional_result = directional_selector.select_features(
        data['features'], data['prices'], data['returns'], data['feature_names']
    )
    
    print(f"✅ Correlation clusters: {len(corr_result.correlation_clusters)}")
    print(f"✅ Market regime: {importance_result.market_regime}")
    print(f"✅ Selected features: {len(directional_result.selected_features)}")
    print(f"✅ Regime confidence: {directional_result.regime_info.confidence:.3f}")
    
    return {
        'correlation': corr_result,
        'importance': importance_result,
        'directional': directional_result
    }

def main():
    """Run all VectorBT integration tests."""
    print("🚀 Starting VectorBT Integration Tests")
    print("=" * 50)
    
    try:
        # Test individual components
        importance_result = test_financial_importance_analyzer()
        directional_result = test_directional_selector()
        correlation_result = test_correlation_analyzer()
        
        # Test integrated pipeline
        integrated_result = test_integrated_pipeline()
        
        print("\n" + "=" * 50)
        print("✅ All VectorBT integration tests completed successfully!")
        
        # Print performance stats
        print("\n📊 Performance Statistics:")
        print(f"Importance Analyzer: {importance_result.analysis_metadata['analysis_time']:.3f}s")
        print(f"Directional Selector: {directional_result.analysis_metadata['analysis_time']:.3f}s")
        print(f"Correlation Analyzer: {correlation_result.analysis_metadata['analysis_time']:.3f}s")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()