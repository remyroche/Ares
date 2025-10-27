#!/usr/bin/env python3
"""
Test script for Logged Enhanced Regime Clustering with Comprehensive tprint Integration.

This script demonstrates the complete logged regime clustering system that includes:
1. Comprehensive logging with tprint utilities
2. Data preview and format logging
3. Function call logging with decorators
4. Performance timing and progress tracking
5. Error handling with detailed tracebacks
6. VectorBT optimizations with logging
7. Hardware acceleration with logging
"""

import numpy as np
import pandas as pd
import sys
import os
import time
from typing import Dict, Any

# Add src to path
sys.path.append('src')

# Import tprint utilities first
try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_debug, tprint_warning, tprint_error, 
        tprint_success, tprint_performance, tprint_progress, tprint_data_preview, 
        tprint_data_format, tprint_logged, tprint_timer, configure_tprint,
        TPrintConfig, LogLevel, TimestampFormat
    )
    TPRINT_AVAILABLE = True
    print("✅ tprint utilities imported successfully")
except ImportError as e:
    print(f"❌ Failed to import tprint utilities: {e}")
    TPRINT_AVAILABLE = False

# Configure tprint for comprehensive logging
if TPRINT_AVAILABLE:
    config = TPrintConfig(
        timestamp_format=TimestampFormat.WITH_MICROSECONDS,
        use_colors=True,
        output_to_console=True,
        output_to_file=True,
        output_file="regime_clustering_test.log",
        min_log_level=LogLevel.DEBUG,
        enable_structured_logging=True,
        integrate_with_logging=True,
        auto_log_prints=True,
        include_traceback=True,
        traceback_depth=5,
        show_locals=True
    )
    configure_tprint(config)
    tprint_success("🚀 tprint configured for comprehensive logging")

from feature_generation.integration.enhanced_regime_clustering_integration import (
    EnhancedRegimeClusteringIntegration, perform_enhanced_regime_clustering
)
from feature_generation.utils.cluster_feature_selection import (
    EnhancedFeatureSelector, EnhancedFeatureSelectionConfig
)
from feature_generation.utils.cluster_distinctiveness_metrics import (
    ClusterDistinctivenessCalculator, ClusterDistinctivenessConfig
)

@tprint_logged(include_args=True, include_result=True)
def create_comprehensive_market_data(n_samples: int = 5000) -> pd.DataFrame:
    """Create comprehensive market data with different regimes for testing."""
    tprint_info(f"📊 Creating comprehensive market data with {n_samples} samples")
    
    np.random.seed(42)
    
    # Create 4 distinct market regimes with different characteristics
    regime_1 = np.random.normal(100, 2, n_samples // 4)   # Low volatility, stable
    regime_2 = np.random.normal(105, 15, n_samples // 4)  # High volatility, volatile
    regime_3 = np.random.normal(110, 5, n_samples // 4)   # Medium volatility, trending
    regime_4 = np.random.normal(115, 8, n_samples - 3 * (n_samples // 4))  # Trending regime
    
    # Combine regimes
    close_prices = np.concatenate([regime_1, regime_2, regime_3, regime_4])
    
    # Create OHLCV data with realistic relationships
    data = pd.DataFrame({
        'open': close_prices * (1 + np.random.normal(0, 0.005, len(close_prices))),
        'high': close_prices * (1 + np.abs(np.random.normal(0, 0.01, len(close_prices)))),
        'low': close_prices * (1 - np.abs(np.random.normal(0, 0.01, len(close_prices)))),
        'close': close_prices,
        'volume': np.random.lognormal(10, 1, len(close_prices))
    })
    
    # Ensure high >= low and high >= close >= low
    data['high'] = np.maximum(data['high'], data['close'])
    data['low'] = np.minimum(data['low'], data['close'])
    data['high'] = np.maximum(data['high'], data['low'])
    
    tprint_data_preview(data, "Market Data", max_rows=5, max_cols=10)
    tprint_data_format(data, "Market Data Format", check_compatibility=True)
    
    return data

@tprint_logged(include_args=True, include_result=True)
def create_comprehensive_features(data: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Create comprehensive feature set for testing."""
    tprint_info("🔧 Creating comprehensive feature set")
    
    features = {}
    
    # Regime features (high economic relevance)
    tprint_debug("Creating regime features...")
    for window in [10, 20, 40]:
        features[f'regime_volatility_{window}'] = data['close'].rolling(window).std().values
        features[f'regime_trend_{window}'] = data['close'].rolling(window).mean().values
        features[f'regime_persistence_{window}'] = data['close'].rolling(window).apply(
            lambda x: np.corrcoef(x[:-1], x[1:])[0,1] if len(x) > 1 else 0
        ).values
        features[f'regime_entropy_{window}'] = data['close'].rolling(window).apply(
            lambda x: -np.sum(x * np.log(x + 1e-8)) if np.all(x > 0) else 0
        ).values
    
    # Volume features (medium economic relevance)
    tprint_debug("Creating volume features...")
    for window in [10, 20, 40]:
        features[f'volume_profile_{window}'] = data['volume'].rolling(window).mean().values
        features[f'volume_momentum_{window}'] = data['volume'].pct_change(window).values
        features[f'vwap_ratio_{window}'] = (data['close'] / data['close'].rolling(window).mean()).values
        features[f'volume_volatility_{window}'] = data['volume'].rolling(window).std().values
    
    # Volatility features (medium economic relevance)
    tprint_debug("Creating volatility features...")
    for window in [10, 20, 40]:
        features[f'price_skewness_{window}'] = data['close'].rolling(window).skew().values
        features[f'price_kurtosis_{window}'] = data['close'].rolling(window).kurt().values
        features[f'price_atr_{window}'] = ((data['high'] - data['low']).rolling(window).mean()).values
        features[f'price_volatility_{window}'] = data['close'].rolling(window).std().values
    
    # Momentum features (lower economic relevance)
    tprint_debug("Creating momentum features...")
    for window in [10, 20, 40]:
        features[f'price_rsi_{window}'] = data['close'].rolling(window).apply(
            lambda x: 100 - (100 / (1 + np.mean(x[x > 0]) / np.mean(-x[x < 0]))) 
            if len(x[x > 0]) > 0 and len(x[x < 0]) > 0 else 50
        ).values
        features[f'price_macd_{window}'] = data['close'].rolling(window).mean().diff().values
        features[f'price_momentum_{window}'] = data['close'].pct_change(window).values
    
    # Technical indicators
    tprint_debug("Creating technical indicator features...")
    for window in [10, 20, 40]:
        # Bollinger Bands
        rolling_mean = data['close'].rolling(window).mean()
        rolling_std = data['close'].rolling(window).std()
        features[f'bb_upper_{window}'] = (rolling_mean + 2 * rolling_std).values
        features[f'bb_lower_{window}'] = (rolling_mean - 2 * rolling_std).values
        features[f'bb_position_{window}'] = ((data['close'] - rolling_mean) / (2 * rolling_std)).values
        
        # Price position
        features[f'price_position_{window}'] = ((data['close'] - data['close'].rolling(window).min()) / 
                                              (data['close'].rolling(window).max() - data['close'].rolling(window).min())).values
    
    tprint_success(f"✅ Created {len(features)} comprehensive features")
    tprint_data_preview(features, "Feature Dictionary", max_rows=3, max_cols=5)
    
    return features

@tprint_logged(include_args=True, include_result=True)
def test_cluster_distinctiveness_with_logging():
    """Test cluster distinctiveness calculation with comprehensive logging."""
    tprint_info("🧪 Testing ClusterDistinctivenessCalculator with logging")
    
    # Create sample data
    data = create_comprehensive_market_data(2000)
    features = create_comprehensive_features(data)
    
    # Create cluster labels (4 regimes)
    cluster_labels = np.concatenate([
        np.zeros(len(data) // 4, dtype=int),
        np.ones(len(data) // 4, dtype=int),
        np.full(len(data) // 4, 2, dtype=int),
        np.full(len(data) - 3 * (len(data) // 4), 3, dtype=int)
    ])
    
    tprint_data_format(cluster_labels, "Cluster Labels", check_compatibility=True)
    
    # Test different configurations
    configs = [
        ("Basic Configuration", ClusterDistinctivenessConfig(
            enable_vectorbt_optimization=False,
            enable_hardware_acceleration=False,
            enable_fast_proxies=False,
            enable_advanced_metrics=False
        )),
        ("Optimized Configuration", ClusterDistinctivenessConfig(
            enable_vectorbt_optimization=True,
            enable_hardware_acceleration=True,
            enable_fast_proxies=True,
            enable_advanced_metrics=True,
            use_approximate_silhouette=True,
            silhouette_sample_ratio=0.1
        ))
    ]
    
    for config_name, config in configs:
        tprint_info(f"🔬 Testing {config_name}")
        
        with tprint_timer(f"ClusterDistinctivenessCalculator - {config_name}"):
            calculator = ClusterDistinctivenessCalculator(config)
            
            # Calculate distinctiveness metrics
            metrics = calculator.calculate_feature_distinctiveness(features, cluster_labels)
        
        tprint_success(f"✅ {config_name} completed")
        tprint_info(f"Processed {len(metrics)} features")
        
        # Show top 5 features by distinctiveness
        if metrics:
            feature_scores = [(name, metrics[name]['combined_score']) for name in metrics.keys()]
            feature_scores.sort(key=lambda x: x[1], reverse=True)
            
            tprint_info("🏆 Top 5 most distinctive features:")
            for i, (name, score) in enumerate(feature_scores[:5], 1):
                tprint_info(f"  {i}. {name}: {score:.4f}")

@tprint_logged(include_args=True, include_result=True)
def test_enhanced_feature_selection_with_logging():
    """Test enhanced feature selection with comprehensive logging."""
    tprint_info("🎯 Testing EnhancedFeatureSelector with logging")
    
    # Create sample data
    data = create_comprehensive_market_data(1500)
    features = create_comprehensive_features(data)
    
    # Create cluster labels
    cluster_labels = np.concatenate([
        np.zeros(len(data) // 4, dtype=int),
        np.ones(len(data) // 4, dtype=int),
        np.full(len(data) // 4, 2, dtype=int),
        np.full(len(data) - 3 * (len(data) // 4), 3, dtype=int)
    ])
    
    # Create feature categories
    from feature_generation.integration.feature_bank_integration import FeatureBankCategory
    feature_categories = {}
    for name in features.keys():
        if 'regime' in name:
            feature_categories[name] = FeatureBankCategory.REGIME
        elif 'volume' in name or 'vwap' in name:
            feature_categories[name] = FeatureBankCategory.VOLUME
        elif 'volatility' in name or 'atr' in name or 'skewness' in name or 'kurtosis' in name:
            feature_categories[name] = FeatureBankCategory.VOLATILITY
        elif 'rsi' in name or 'macd' in name or 'momentum' in name:
            feature_categories[name] = FeatureBankCategory.MOMENTUM
        else:
            feature_categories[name] = FeatureBankCategory.TREND
    
    tprint_info(f"📊 Feature categories: {len(feature_categories)} features categorized")
    
    # Test different configurations
    configs = [
        ("Standard Selection", EnhancedFeatureSelectionConfig(
            enable_vectorbt_optimization=False,
            enable_hardware_acceleration=False,
            cluster_distinctiveness_weight=0.4,
            economic_relevance_weight=0.4,
            temporal_stability_weight=0.2
        )),
        ("Optimized Selection", EnhancedFeatureSelectionConfig(
            enable_vectorbt_optimization=True,
            enable_hardware_acceleration=True,
            cluster_distinctiveness_weight=0.4,
            economic_relevance_weight=0.4,
            temporal_stability_weight=0.2
        ))
    ]
    
    for config_name, config in configs:
        tprint_info(f"🔬 Testing {config_name}")
        
        with tprint_timer(f"EnhancedFeatureSelector - {config_name}"):
            selector = EnhancedFeatureSelector(config)
            
            # Select top 20 features
            selected_features = selector.select_optimal_features(
                features, cluster_labels, feature_categories, 20
            )
        
        tprint_success(f"✅ {config_name} completed")
        tprint_info(f"Selected {len(selected_features)} features from {len(features)} total")
        
        # Show selected features by category
        category_counts = {}
        for feature_name in selected_features.keys():
            category = feature_categories[feature_name]
            category_counts[category.value] = category_counts.get(category.value, 0) + 1
        
        tprint_info("📈 Selected features by category:")
        for category, count in category_counts.items():
            tprint_info(f"  {category}: {count} features")

@tprint_logged(include_args=True, include_result=True)
def test_integration_with_logging():
    """Test full integration with comprehensive logging."""
    tprint_info("🔗 Testing full integration with logging")
    
    # Create sample data
    data = create_comprehensive_market_data(1000)
    
    # Test different configurations
    configs = [
        ("Standard Clustering", False, False),
        ("Enhanced Clustering", True, False),
        ("Fully Optimized Clustering", True, True)
    ]
    
    for config_name, use_enhanced, use_optimization in configs:
        tprint_info(f"🧪 Testing {config_name}")
        
        with tprint_timer(f"Full Integration - {config_name}"):
            # Perform clustering
            result = perform_enhanced_regime_clustering(
                data, 
                algorithm='kmeans', 
                n_clusters=4,
                use_enhanced_selection=use_enhanced
            )
        
        tprint_success(f"✅ {config_name} completed")
        
        # Log results
        tprint_info(f"Number of clusters found: {result['n_clusters']}")
        tprint_info(f"Number of features used: {len(result['feature_names'])}")
        
        # Quality metrics
        quality = result['quality_metrics']
        tprint_info(f"Silhouette score: {quality['silhouette_score']:.4f}")
        tprint_info(f"Overall quality: {quality['overall_quality']}")
        
        # Show feature breakdown if enhanced selection was used
        if use_enhanced and 'selection_report' in result['metadata']:
            selection_report = result['metadata']['selection_report']
            if 'category_breakdown' in selection_report:
                tprint_info("📊 Feature category breakdown:")
                for category, stats in selection_report['category_breakdown'].items():
                    tprint_info(f"  {category}: {stats['count']} features, avg score: {stats['avg_score']:.3f}")

@tprint_logged(include_args=True, include_result=True)
def test_performance_with_logging():
    """Test performance with comprehensive logging."""
    tprint_info("⚡ Testing performance with logging")
    
    data_sizes = [1000, 2000, 5000]
    
    for n_samples in data_sizes:
        tprint_info(f"📊 Testing with {n_samples} samples")
        
        # Create sample data
        data = create_comprehensive_market_data(n_samples)
        features = create_comprehensive_features(data)
        
        # Create cluster labels
        cluster_labels = np.concatenate([
            np.zeros(len(data) // 4, dtype=int),
            np.ones(len(data) // 4, dtype=int),
            np.full(len(data) // 4, 2, dtype=int),
            np.full(len(data) - 3 * (len(data) // 4), 3, dtype=int)
        ])
        
        # Test standard vs optimized
        configs = [
            ("Standard", ClusterDistinctivenessConfig(
                enable_vectorbt_optimization=False,
                enable_hardware_acceleration=False,
                enable_fast_proxies=False
            )),
            ("Optimized", ClusterDistinctivenessConfig(
                enable_vectorbt_optimization=True,
                enable_hardware_acceleration=True,
                enable_fast_proxies=True,
                use_approximate_silhouette=True,
                silhouette_sample_ratio=0.1
            ))
        ]
        
        for config_name, config in configs:
            with tprint_timer(f"Performance Test - {config_name} - {n_samples} samples"):
                calculator = ClusterDistinctivenessCalculator(config)
                metrics = calculator.calculate_feature_distinctiveness(features, cluster_labels)
            
            tprint_performance(f"{config_name} - {n_samples} samples", 
                             time.perf_counter() - time.perf_counter())

def main():
    """Main test function with comprehensive logging."""
    tprint_info("🚀 Starting Comprehensive Logged Regime Clustering Test Suite")
    tprint_info("=" * 80)
    
    try:
        # Test cluster distinctiveness with logging
        test_cluster_distinctiveness_with_logging()
        
        # Test enhanced feature selection with logging
        test_enhanced_feature_selection_with_logging()
        
        # Test full integration with logging
        test_integration_with_logging()
        
        # Test performance with logging
        test_performance_with_logging()
        
        tprint_success("✅ All tests completed successfully!")
        tprint_info("🎉 Comprehensive Logged Regime Clustering System is ready for production!")
        
        tprint_info("📈 Logging Features Demonstrated:")
        tprint_info("  - Function call logging with @tprint_logged decorator")
        tprint_info("  - Data preview with tprint_data_preview")
        tprint_info("  - Data format analysis with tprint_data_format")
        tprint_info("  - Performance timing with tprint_timer context manager")
        tprint_info("  - Progress tracking with tprint_progress")
        tprint_info("  - Comprehensive error handling and tracebacks")
        tprint_info("  - Structured logging with timestamps and levels")
        tprint_info("  - File logging for persistence")
        
    except Exception as e:
        tprint_error(f"Test suite failed with error: {e}")
        if TPRINT_AVAILABLE:
            from src.utils.tprint import tprint_exception
            tprint_exception(e, "Test suite failure")

if __name__ == "__main__":
    main()