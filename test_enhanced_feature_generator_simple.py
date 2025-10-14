#!/usr/bin/env python3
"""
Simple test for Enhanced Feature Generator without complex dependencies.

This script tests the enhanced feature generator directly without importing
the full pipeline to avoid dependency issues.
"""

import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def create_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """Create sample OHLCV data for testing."""
    print(f"📊 Creating sample data with {n_samples} samples")
    
    # Generate realistic price data
    np.random.seed(42)
    
    # Create time index
    start_time = datetime.now() - timedelta(minutes=n_samples * 15)
    time_index = [start_time + timedelta(minutes=i * 15) for i in range(n_samples)]
    
    # Generate price data with trend and volatility
    base_price = 100.0
    returns = np.random.normal(0.0001, 0.02, n_samples)  # 0.01% mean return, 2% volatility
    prices = [base_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    prices = np.array(prices)
    
    # Generate OHLCV data
    data = pd.DataFrame(index=time_index)
    data['open'] = prices * (1 + np.random.normal(0, 0.001, n_samples))
    data['high'] = np.maximum(data['open'], prices) * (1 + np.abs(np.random.normal(0, 0.005, n_samples)))
    data['low'] = np.minimum(data['open'], prices) * (1 - np.abs(np.random.normal(0, 0.005, n_samples)))
    data['close'] = prices
    data['volume'] = np.random.lognormal(10, 1, n_samples)
    
    # Ensure high >= low
    data['high'] = np.maximum(data['high'], data['low'])
    
    print(f"✅ Sample data created: {data.shape}")
    print(f"   Price range: {data['close'].min():.2f} - {data['close'].max():.2f}")
    print(f"   Volume range: {data['volume'].min():.0f} - {data['volume'].max():.0f}")
    
    return data

def create_sample_targets(data: pd.DataFrame) -> pd.Series:
    """Create sample targets for supervised learning."""
    print("🎯 Creating sample targets")
    
    # Create forward returns as targets
    targets = data['close'].pct_change(5).shift(-5)  # 5-period forward returns
    targets = targets.dropna()
    
    print(f"✅ Targets created: {len(targets)} samples")
    print(f"   Target range: {targets.min():.4f} - {targets.max():.4f}")
    print(f"   Target mean: {targets.mean():.4f}, std: {targets.std():.4f}")
    
    return targets

def test_enhanced_feature_generator():
    """Test the enhanced feature generator directly."""
    print("\n" + "="*80)
    print("🚀 TESTING ENHANCED FEATURE GENERATOR")
    print("="*80)
    
    try:
        # Import the enhanced feature generator directly
        from src.training.steps.pre_training.unified_data_driven_pipeline.enhanced_components.enhanced_feature_generator import (
            EnhancedFeatureGenerator, FeatureGenerationConfig, create_enhanced_feature_generator
        )
        
        print("✅ Enhanced feature generator imported successfully")
        
        # Create sample data
        data = create_sample_data(500)  # Smaller dataset for faster testing
        targets = create_sample_targets(data)
        
        # Create base features for interaction testing
        base_features = pd.DataFrame()
        base_features['price_change'] = data['close'].pct_change()
        base_features['volume_change'] = data['volume'].pct_change()
        base_features['high_low_ratio'] = data['high'] / data['low']
        base_features['close_open_ratio'] = data['close'] / data['open']
        
        # Create enhanced feature generator
        print("\n🔧 Creating enhanced feature generator")
        feature_generator = create_enhanced_feature_generator()
        
        # Generate enhanced features
        print("\n⚡ Generating enhanced features")
        start_time = time.time()
        
        feature_result = feature_generator.generate_features(
            data, targets, base_features
        )
        
        generation_time = time.time() - start_time
        
        if feature_result.success:
            print(f"✅ Enhanced feature generation completed in {generation_time:.3f}s")
            print(f"   Cross-timeframe features: {len(feature_result.cross_timeframe_features)}")
            print(f"   Interaction features: {len(feature_result.interaction_features)}")
            print(f"   No features: {len(feature_result.no_features)}")
            print(f"   Total features: {len(feature_result.all_features)}")
            
            # Display sample features
            print("\n📋 Sample Cross-Timeframe Features:")
            for i, feature in enumerate(feature_result.cross_timeframe_features[:5]):
                print(f"   {i+1}. {feature.name}")
                print(f"      Formula: {feature.formula}")
                print(f"      Utility: {feature.utility_score:.4f}")
                print(f"      Lookback: {feature.lookback_period}")
                print(f"      Method: {feature.creation_method}")
            
            print("\n📋 Sample Interaction Features:")
            for i, feature in enumerate(feature_result.interaction_features[:5]):
                print(f"   {i+1}. {feature.name}")
                print(f"      Formula: {feature.formula}")
                print(f"      Parents: {feature.parent_features}")
                print(f"      Utility: {feature.utility_score:.4f}")
                print(f"      Method: {feature.creation_method}")
            
            print("\n📋 Sample No Features:")
            for i, feature in enumerate(feature_result.no_features[:5]):
                print(f"   {i+1}. {feature.name}")
                print(f"      Formula: {feature.formula}")
                print(f"      Utility: {feature.utility_score:.4f}")
                print(f"      Method: {feature.creation_method}")
            
            # Test feature quality
            print("\n📊 Feature Quality Analysis:")
            all_features = feature_result.all_features
            if all_features:
                utilities = [f.utility_score for f in all_features]
                print(f"   Average utility: {np.mean(utilities):.4f}")
                print(f"   Max utility: {np.max(utilities):.4f}")
                print(f"   Min utility: {np.min(utilities):.4f}")
                print(f"   Features with utility > 0.1: {sum(1 for u in utilities if u > 0.1)}")
                
                # Check for different creation methods
                methods = [f.creation_method for f in all_features if f.creation_method]
                if methods:
                    method_counts = pd.Series(methods).value_counts()
                    print(f"   Creation methods used: {dict(method_counts)}")
                
                # Check for different feature types
                types = [f.feature_type for f in all_features]
                type_counts = pd.Series(types).value_counts()
                print(f"   Feature types: {dict(type_counts)}")
            
            # Test specific feature types
            print("\n🧪 Testing Specific Feature Types:")
            
            # Test cross-timeframe features
            cross_timeframe_features = feature_result.cross_timeframe_features
            if cross_timeframe_features:
                print(f"   ✅ Cross-timeframe features: {len(cross_timeframe_features)}")
                lookback_periods = [f.lookback_period for f in cross_timeframe_features if f.lookback_period]
                if lookback_periods:
                    print(f"      Lookback periods: {sorted(set(lookback_periods))}")
            
            # Test interaction features
            interaction_features = feature_result.interaction_features
            if interaction_features:
                print(f"   ✅ Interaction features: {len(interaction_features)}")
                interaction_orders = [f.metadata.get('interaction_order', 'unknown') for f in interaction_features]
                if interaction_orders:
                    order_counts = pd.Series(interaction_orders).value_counts()
                    print(f"      Interaction orders: {dict(order_counts)}")
            
            # Test no features
            no_features = feature_result.no_features
            if no_features:
                print(f"   ✅ No features: {len(no_features)}")
                no_methods = [f.creation_method for f in no_features if f.creation_method]
                if no_methods:
                    no_method_counts = pd.Series(no_methods).value_counts()
                    print(f"      Creation methods: {dict(no_method_counts)}")
            
            print("\n🎉 Enhanced feature generation test completed successfully!")
            
        else:
            print(f"❌ Enhanced feature generation failed: {feature_result.error_message}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main test function."""
    print("🧪 ENHANCED FEATURE GENERATOR TEST")
    print("="*80)
    print("Testing comprehensive feature generation including:")
    print("✅ Cross timeframe features with optimized lookback period")
    print("✅ Interaction (2-3) features with optimized lookback period")
    print("✅ Feature creation in multiple ways (addition, subtraction, log, multiplication, division)")
    print("✅ No features with optimized lookback period")
    print("="*80)
    
    # Test enhanced feature generator
    test_enhanced_feature_generator()
    
    print("\n" + "="*80)
    print("🎉 TEST COMPLETED")
    print("="*80)

if __name__ == "__main__":
    main()