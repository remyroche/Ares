#!/usr/bin/env python3
"""
Test MS-DR Clustering with Real Market Data and Proper Regime Features

This script tests the MS-DR clustering functionality using:
1. Real market data from the artifact manager
2. Proper regime features from the feature generation system
3. Both basic clustering and auto-tuning functionality

Features used are from:
- src/feature_generation/categories/regime_feature_categorization.py
- src/feature_generation/categories/regime_feature_integration.py
"""

import sys
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import traceback

# Add src to path
sys.path.insert(0, 'src')

# Import MS-DR clustering components
from src.training.steps.market_analysis.ms_dr_clustering.ms_dr_clusterer import MSDRClusterer, MSDRConfig
from src.training.steps.market_analysis.ms_dr_clustering.ms_dr_auto_tuner import MSDRAutoTuner, MSDRTuningConfig

# Import feature generation components
from src.feature_generation.categories.regime_feature_categorization import (
    RegimeFeatureCategorizer, FeatureUseCase, get_regime_clustering_features
)
from src.feature_generation.categories.regime_feature_integration import (
    RegimeFeatureIntegration, RegimeFeatureConfig
)

# Import data management
try:
    from src.data_management.artifact_manager import ArtifactManager
    ARTIFACT_MANAGER_AVAILABLE = True
except ImportError:
    ARTIFACT_MANAGER_AVAILABLE = False
    print("⚠️ ArtifactManager not available")

try:
    from src.data_management.klines_parquet_manager import KlinesParquetManager
    KLINES_MANAGER_AVAILABLE = True
except ImportError:
    KLINES_MANAGER_AVAILABLE = False
    print("⚠️ KlinesParquetManager not available")

# Import logging
try:
    from src.utils.tprint import tprint, tprint_info, tprint_success, tprint_error, tprint_warning
except ImportError:
    def tprint(*args, **kwargs):
        print(*args)
    tprint_info = tprint_success = tprint_error = tprint_warning = tprint


def load_market_data(symbol: str = "ETHUSDT", timeframe: str = "1h", limit: int = 1000) -> Optional[pd.DataFrame]:
    """Load market data using available data managers."""
    tprint_info(f"📊 Loading market data for {symbol} {timeframe}...")
    
    # Try ArtifactManager first
    if ARTIFACT_MANAGER_AVAILABLE:
        try:
            artifact_manager = ArtifactManager()
            data = artifact_manager.get_klines(symbol, timeframe, limit=limit)
            if data is not None and not data.empty:
                tprint_success(f"✅ Loaded {len(data)} records from ArtifactManager")
                return data
        except Exception as e:
            tprint_warning(f"⚠️ ArtifactManager failed: {e}")
    
    # Try KlinesParquetManager
    if KLINES_MANAGER_AVAILABLE:
        try:
            klines_manager = KlinesParquetManager()
            data = klines_manager.get_klines(symbol, timeframe, limit=limit)
            if data is not None and not data.empty:
                tprint_success(f"✅ Loaded {len(data)} records from KlinesParquetManager")
                return data
        except Exception as e:
            tprint_warning(f"⚠️ KlinesParquetManager failed: {e}")
    
    # Fallback: create sample data
    tprint_warning("⚠️ No data managers available, creating sample data")
    return create_sample_market_data(limit)


def create_sample_market_data(n_samples: int = 1000) -> pd.DataFrame:
    """Create sample market data for testing."""
    np.random.seed(42)
    
    # Create realistic price data with regime switching
    base_price = 3000.0
    prices = [base_price]
    
    # Create 3 distinct regimes
    regime_lengths = [n_samples // 3] * 3
    regime_lengths[0] += n_samples % 3  # Add remainder to first regime
    
    regime_params = [
        {"drift": 0.001, "volatility": 0.02, "trend": 1.0},  # Bull market
        {"drift": -0.0005, "volatility": 0.05, "trend": -1.0},  # Bear market
        {"drift": 0.0, "volatility": 0.01, "trend": 0.0}  # Sideways
    ]
    
    current_idx = 0
    for regime_idx, length in enumerate(regime_lengths):
        params = regime_params[regime_idx]
        for i in range(length):
            if current_idx == 0:
                price = base_price
            else:
                # Add regime-specific drift and volatility
                drift = params["drift"] + params["trend"] * np.sin(current_idx * 0.1) * 0.001
                volatility = params["volatility"]
                price = prices[-1] * (1 + np.random.normal(drift, volatility))
            
            prices.append(price)
            current_idx += 1
    
    # Create OHLCV data
    data = []
    for i, price in enumerate(prices[1:]):  # Skip first price
        # Generate realistic OHLCV from price
        volatility = abs(np.random.normal(0, 0.01))
        high = price * (1 + volatility)
        low = price * (1 - volatility)
        open_price = prices[i] if i > 0 else price
        close = price
        volume = np.random.uniform(1000, 10000)
        
        data.append({
            'timestamp': pd.Timestamp.now() - pd.Timedelta(hours=len(prices)-i-1),
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    return df


def generate_regime_features(data: pd.DataFrame) -> pd.DataFrame:
    """Generate regime features using the feature generation system."""
    tprint_info("🔧 Generating regime features...")
    
    try:
        # Get regime clustering features
        categorizer = RegimeFeatureCategorizer()
        regime_features = categorizer.get_priority_features(FeatureUseCase.REGIME_CLUSTERING, 50)
        tprint_info(f"📋 Using {len(regime_features)} regime features for clustering")
        
        # Initialize regime feature generator
        config = RegimeFeatureConfig(
            enable_regime_detection=True,
            enable_adaptive_features=True,
            enable_regime_transitions=True,
            lookback_period=20
        )
        generator = RegimeFeatureIntegration(config)
        
        # Generate features for each data point
        feature_data = []
        for i in range(20, len(data)):  # Start after lookback period
            window_data = data.iloc[i-20:i+1]  # Include current point
            features = generator._generate_regime_features(window_data)
            
            # Add to feature data
            feature_row = {'timestamp': data.index[i]}
            feature_row.update(features)
            feature_data.append(feature_row)
        
        feature_df = pd.DataFrame(feature_data)
        feature_df.set_index('timestamp', inplace=True)
        
        # Select only numeric features for clustering
        numeric_features = feature_df.select_dtypes(include=[np.number])
        
        tprint_success(f"✅ Generated {numeric_features.shape[1]} numeric features from {len(feature_data)} samples")
        tprint_info(f"📊 Feature columns: {list(numeric_features.columns)}")
        
        return numeric_features
        
    except Exception as e:
        tprint_error(f"❌ Error generating regime features: {e}")
        traceback.print_exc()
        return None


def test_basic_clustering(feature_data: pd.DataFrame) -> Dict[str, Any]:
    """Test basic MS-DR clustering."""
    tprint_info("🧪 Testing basic MS-DR clustering...")
    
    try:
        # Create MS-DR configuration
        config = MSDRConfig(
            n_regimes=3,
            model_type='autoregression',
            order=1,
            switching_variance=True,
            random_state=42
        )
        
        # Initialize clusterer
        clusterer = MSDRClusterer(config)
        
        # Run clustering
        result = clusterer.fit_predict(feature_data.values)
        
        tprint_success("✅ Basic clustering completed successfully!")
        
        return {
            'success': True,
            'n_clusters': result.n_clusters,
            'aic': result.aic,
            'bic': result.bic,
            'transition_persistence': result.transition_persistence,
            'quality_metrics': result.quality_metrics,
            'regime_params': result.regime_params,
            'labels': result.labels
        }
        
    except Exception as e:
        tprint_error(f"❌ Basic clustering failed: {e}")
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


def test_auto_tuning(feature_data: pd.DataFrame) -> Dict[str, Any]:
    """Test MS-DR auto-tuning."""
    tprint_info("🔧 Testing MS-DR auto-tuning...")
    
    try:
        # Create tuning configuration
        tuning_config = MSDRTuningConfig(
            n_trials=10,  # Reduced for testing
            timeout_minutes=2.0,
            random_state=42
        )
        
        # Initialize auto-tuner
        tuner = MSDRAutoTuner(tuning_config)
        
        # Run auto-tuning
        result = tuner.auto_tune(feature_data.values)
        
        if hasattr(result, 'success') and result.success:
            tprint_success("✅ Auto-tuning completed successfully!")
            return {
                'success': True,
                'best_score': result.best_score,
                'best_params': result.best_params,
                'optimization_time': result.optimization_time
            }
        else:
            tprint_error(f"❌ Auto-tuning failed: {result}")
            return {'success': False, 'error': str(result)}
        
    except Exception as e:
        tprint_error(f"❌ Auto-tuning failed: {e}")
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


def generate_detailed_report(basic_result: Dict[str, Any], auto_tune_result: Dict[str, Any], 
                           feature_data: pd.DataFrame, market_data: pd.DataFrame) -> None:
    """Generate detailed report on MS-DR clustering results."""
    tprint_info("📊 Generating detailed report...")
    
    print("\n" + "="*80)
    print("🎯 MS-DR CLUSTERING DETAILED REPORT")
    print("="*80)
    
    # Data Information
    print(f"\n📈 DATA INFORMATION:")
    print(f"   • Market data samples: {len(market_data)}")
    print(f"   • Feature data samples: {len(feature_data)}")
    print(f"   • Feature dimensions: {feature_data.shape[1]}")
    print(f"   • Feature columns: {list(feature_data.columns)}")
    
    # Basic Clustering Results
    print(f"\n🧪 BASIC CLUSTERING RESULTS:")
    if basic_result['success']:
        print(f"   ✅ Status: SUCCESS")
        print(f"   • Number of regimes discovered: {basic_result['n_clusters']}")
        print(f"   • AIC: {basic_result['aic']:.4f}")
        print(f"   • BIC: {basic_result['bic']:.4f}")
        print(f"   • Transition persistence: {basic_result['transition_persistence']:.4f}")
        
        if 'quality_metrics' in basic_result and basic_result['quality_metrics']:
            metrics = basic_result['quality_metrics']
            print(f"   • Silhouette score: {metrics.get('silhouette_score', 'N/A'):.4f}")
            print(f"   • Calinski-Harabasz score: {metrics.get('calinski_harabasz_score', 'N/A'):.4f}")
            print(f"   • Davies-Bouldin score: {metrics.get('davies_bouldin_score', 'N/A'):.4f}")
            print(f"   • Composite quality score: {metrics.get('composite_quality_score', 'N/A'):.4f}")
        
        # Regime parameters
        if 'regime_params' in basic_result and basic_result['regime_params']:
            print(f"\n   📊 REGIME PARAMETERS:")
            for i, params in enumerate(basic_result['regime_params']):
                print(f"      Regime {i+1}: {params}")
    else:
        print(f"   ❌ Status: FAILED")
        print(f"   • Error: {basic_result.get('error', 'Unknown error')}")
    
    # Auto-tuning Results
    print(f"\n🔧 AUTO-TUNING RESULTS:")
    if auto_tune_result['success']:
        print(f"   ✅ Status: SUCCESS")
        print(f"   • Best score: {auto_tune_result['best_score']:.4f}")
        print(f"   • Optimization time: {auto_tune_result['optimization_time']:.2f} seconds")
        print(f"   • Best parameters:")
        for param, value in auto_tune_result['best_params'].items():
            print(f"      - {param}: {value}")
    else:
        print(f"   ❌ Status: FAILED")
        print(f"   • Error: {auto_tune_result.get('error', 'Unknown error')}")
    
    # Feature Analysis
    print(f"\n🔍 FEATURE ANALYSIS:")
    print(f"   • Feature statistics:")
    print(f"     - Mean: {feature_data.mean().mean():.4f}")
    print(f"     - Std: {feature_data.std().mean():.4f}")
    print(f"     - Min: {feature_data.min().min():.4f}")
    print(f"     - Max: {feature_data.max().max():.4f}")
    
    # Feature correlation analysis
    if len(feature_data.columns) > 1:
        corr_matrix = feature_data.corr()
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.8:
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
        
        if high_corr_pairs:
            print(f"   • High correlation pairs (|r| > 0.8): {len(high_corr_pairs)}")
            for feat1, feat2, corr in high_corr_pairs[:5]:  # Show first 5
                print(f"     - {feat1} vs {feat2}: {corr:.3f}")
        else:
            print(f"   • No high correlation pairs found")
    
    print("\n" + "="*80)
    print("✅ REPORT COMPLETE")
    print("="*80)


def main():
    """Main test function."""
    tprint_info("🚀 Starting MS-DR Clustering Test with Regime Features")
    
    try:
        # Step 1: Load market data
        market_data = load_market_data("ETHUSDT", "1h", 1000)
        if market_data is None or market_data.empty:
            tprint_error("❌ Failed to load market data")
            return
        
        tprint_success(f"✅ Loaded market data: {market_data.shape}")
        
        # Step 2: Generate regime features
        feature_data = generate_regime_features(market_data)
        if feature_data is None or feature_data.empty:
            tprint_error("❌ Failed to generate regime features")
            return
        
        tprint_success(f"✅ Generated regime features: {feature_data.shape}")
        
        # Step 3: Test basic clustering
        basic_result = test_basic_clustering(feature_data)
        
        # Step 4: Test auto-tuning
        auto_tune_result = test_auto_tuning(feature_data)
        
        # Step 5: Generate detailed report
        generate_detailed_report(basic_result, auto_tune_result, feature_data, market_data)
        
        tprint_success("🎉 MS-DR Clustering Test Completed Successfully!")
        
    except Exception as e:
        tprint_error(f"❌ Test failed with error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
