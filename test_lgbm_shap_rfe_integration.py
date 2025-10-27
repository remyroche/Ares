#!/usr/bin/env python3
"""
Test script for LGBM-SHAP RFE Integration

This script demonstrates the LGBM-SHAP RFE feature selection integration
with comprehensive logging and detailed reporting.
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add src to path
sys.path.append('/workspace/src')

# Import the enhanced models training integration
from feature_generation.integration.enhanced_models_training_integration import (
    EnhancedModelsTrainingIntegration
)

def create_realistic_ohlcv_data(n_samples: int = 2000, 
                               start_price: float = 100.0,
                               volatility: float = 0.02,
                               trend: float = 0.0001) -> pd.DataFrame:
    """Create realistic OHLCV data for testing."""
    print("📊 Creating realistic OHLCV data...")
    
    # Generate price series with trend and volatility
    np.random.seed(42)
    dt = 1/24/4  # 15-minute intervals
    t = np.arange(n_samples) * dt
    
    # Generate returns with trend and volatility
    returns = np.random.normal(trend * dt, volatility * np.sqrt(dt), n_samples)
    log_prices = np.cumsum(returns)
    prices = start_price * np.exp(log_prices)
    
    # Generate OHLCV data
    data = pd.DataFrame(index=pd.date_range(start='2024-01-01', periods=n_samples, freq='15T'))
    
    # Close prices
    data['close'] = prices
    
    # Open prices (previous close + small gap)
    data['open'] = np.roll(prices, 1) * (1 + np.random.normal(0, 0.001, n_samples))
    data['open'].iloc[0] = start_price
    
    # High and low prices
    intraday_volatility = np.abs(np.random.normal(0, volatility/4, n_samples))
    data['high'] = np.maximum(data['open'], data['close']) * (1 + intraday_volatility)
    data['low'] = np.minimum(data['open'], data['close']) * (1 - intraday_volatility)
    
    # Volume (log-normal distribution)
    data['volume'] = np.random.lognormal(10, 1, n_samples)
    
    # Ensure OHLC consistency
    data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
    data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
    
    print(f"✅ Created OHLCV data: {data.shape[0]} samples, {data.shape[1]} columns")
    print(f"📈 Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
    print(f"📊 Volume range: {data['volume'].min():.0f} - {data['volume'].max():.0f}")
    
    return data

def test_lgbm_shap_rfe_integration():
    """Test the LGBM-SHAP RFE integration."""
    print("🚀 Testing LGBM-SHAP RFE Integration")
    print("=" * 60)
    
    # Create test data
    data = create_realistic_ohlcv_data(n_samples=1500)
    
    # Create integration with custom parameters
    print("\n🔧 Creating Enhanced Models Training Integration with LGBM-SHAP RFE...")
    integration = EnhancedModelsTrainingIntegration(
        target_features=60,  # Strictly select 60 features
        enable_comprehensive_features=True,
        enable_lgbm_shap_rfe=True,
        removal_percentage=0.25,  # Remove 25% of features per iteration
        lgbm_params={
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42
        },
        enable_detailed_logging=True
    )
    
    # Run feature selection
    print("\n🔍 Running LGBM-SHAP RFE feature selection...")
    print("=" * 60)
    
    try:
        result = integration.select_features_for_regime_training(data)
        
        # Display results
        print("\n🎉 Feature Selection Completed Successfully!")
        print("=" * 60)
        
        # Summary statistics
        print(f"📊 Selected Features: {result['selected_features']['count']}")
        print(f"🗑️ Removed Features: {result['removed_features']['count']}")
        print(f"🔄 Total Iterations: {result['selection_process']['total_iterations']}")
        print(f"📈 Target Features: {result['target_features']}")
        print(f"📉 Removal Percentage: {result['removal_percentage']:.1%}")
        
        # Performance summary
        perf_summary = result['selection_process']['performance_summary']
        if perf_summary:
            print(f"\n📈 Performance Summary:")
            print(f"   Mean Performance: {perf_summary['mean_performance']:.6f}")
            print(f"   Final Performance: {perf_summary['final_performance']:.6f}")
            print(f"   Performance Trend: {perf_summary['performance_trend']}")
        
        # Importance summary
        imp_summary = result['selection_process']['importance_summary']
        if imp_summary:
            print(f"\n🎯 Importance Summary:")
            print(f"   Mean Importance: {imp_summary['mean_importance']:.6f}")
            print(f"   Max Importance: {imp_summary['max_importance']:.6f}")
            print(f"   Min Importance: {imp_summary['min_importance']:.6f}")
        
        # Selected features (first 20)
        print(f"\n✅ Selected Features (first 20):")
        for i, feature_name in enumerate(result['selected_features']['names'][:20], 1):
            print(f"   {i:2d}. {feature_name}")
        
        if len(result['selected_features']['names']) > 20:
            print(f"   ... and {len(result['selected_features']['names']) - 20} more")
        
        # Removed features (first 20)
        print(f"\n🗑️ Removed Features (first 20):")
        for i, feature_name in enumerate(result['removed_features']['names'][:20], 1):
            print(f"   {i:2d}. {feature_name}")
        
        if len(result['removed_features']['names']) > 20:
            print(f"   ... and {len(result['removed_features']['names']) - 20} more")
        
        # Iteration details
        print(f"\n🔄 Iteration Details:")
        for i, iteration in enumerate(result['selection_process']['history'], 1):
            print(f"   Iteration {i}: Removed {len(iteration['features_removed'])} features, "
                  f"Remaining: {iteration['features_remaining']}, "
                  f"Performance: {iteration['performance']:.6f}")
        
        # Report information
        if 'detailed_report' in result:
            report = result['detailed_report']
            print(f"\n📋 Detailed Report Generated:")
            print(f"   Timestamp: {report['timestamp']}")
            print(f"   Global Metrics: {len(report['global_metrics'])} metrics")
            print(f"   Per-Feature Metrics: {len(report['per_feature_metrics'])} features")
            print(f"   Report saved to: outcomes/ directory")
        
        return result
        
    except Exception as e:
        print(f"❌ Error during feature selection: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function to run the test."""
    print("🧪 LGBM-SHAP RFE Integration Test")
    print("=" * 60)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run the test
    result = test_lgbm_shap_rfe_integration()
    
    print(f"\n⏰ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if result:
        print("✅ Test completed successfully!")
        return 0
    else:
        print("❌ Test failed!")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)