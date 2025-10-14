#!/usr/bin/env python3
"""
Test script for Enhanced UnifiedDataDrivenPipeline with comprehensive feature generation.

This script demonstrates the enhanced pipeline capabilities including:
- Cross timeframe features with optimized lookback period
- Interaction (2-3) features with optimized lookback period  
- Feature creation in multiple ways (addition, subtraction, log, multiplication, division)
- No features with optimized lookback period
"""

import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta

# Import the enhanced pipeline
try:
    from src.training.steps.pre_training.unified_data_driven_pipeline.enhanced_unified_pipeline import (
        EnhancedUnifiedDataDrivenPipeline, create_enhanced_unified_pipeline
    )
    from src.training.steps.pre_training.unified_data_driven_pipeline.enhanced_components.enhanced_feature_generator import (
        FeatureGenerationConfig
    )
    PIPELINE_AVAILABLE = True
except ImportError as e:
    print(f"❌ Failed to import enhanced pipeline: {e}")
    PIPELINE_AVAILABLE = False

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

def test_enhanced_feature_generation():
    """Test the enhanced feature generation capabilities."""
    print("\n" + "="*80)
    print("🚀 TESTING ENHANCED FEATURE GENERATION")
    print("="*80)
    
    if not PIPELINE_AVAILABLE:
        print("❌ Enhanced pipeline not available")
        return
    
    try:
        # Create sample data
        data = create_sample_data(500)  # Smaller dataset for faster testing
        targets = create_sample_targets(data)
        
        # Create enhanced pipeline
        print("\n🔧 Creating enhanced pipeline")
        pipeline = create_enhanced_unified_pipeline()
        
        # Test enhanced feature generator directly
        print("\n🧪 Testing enhanced feature generator directly")
        from src.training.steps.pre_training.unified_data_driven_pipeline.enhanced_components.enhanced_feature_generator import (
            create_enhanced_feature_generator
        )
        
        feature_generator = create_enhanced_feature_generator()
        
        # Generate base features for interaction testing
        base_features = pd.DataFrame()
        base_features['price_change'] = data['close'].pct_change()
        base_features['volume_change'] = data['volume'].pct_change()
        base_features['high_low_ratio'] = data['high'] / data['low']
        base_features['close_open_ratio'] = data['close'] / data['open']
        
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
                method_counts = pd.Series(methods).value_counts()
                print(f"   Creation methods used: {dict(method_counts)}")
            
        else:
            print(f"❌ Enhanced feature generation failed: {feature_result.error_message}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

def test_full_pipeline():
    """Test the full enhanced pipeline."""
    print("\n" + "="*80)
    print("🚀 TESTING FULL ENHANCED PIPELINE")
    print("="*80)
    
    if not PIPELINE_AVAILABLE:
        print("❌ Enhanced pipeline not available")
        return
    
    try:
        # Create sample data
        data = create_sample_data(1000)
        targets = create_sample_targets(data)
        
        # Create enhanced pipeline
        print("\n🔧 Creating enhanced pipeline")
        pipeline = create_enhanced_unified_pipeline()
        
        # Run full pipeline
        print("\n⚡ Running full enhanced pipeline")
        start_time = time.time()
        
        result = pipeline.process(data, targets, "15m")
        
        execution_time = time.time() - start_time
        
        if result.success:
            print(f"✅ Full pipeline completed in {execution_time:.3f}s")
            print(f"   Optimal periods: {len(result.optimal_periods)}")
            print(f"   Selected features: {len(result.selected_features)}")
            print(f"   Generated interactions: {len(result.generated_interactions)}")
            print(f"   HTF interactions: {len(result.htf_interactions)}")
            print(f"   Cross-timeframe features: {len(result.cross_timeframe_features)}")
            print(f"   Interaction features: {len(result.interaction_features)}")
            print(f"   No features: {len(result.no_features)}")
            
            # Display performance stats
            print("\n📊 Performance Statistics:")
            stats = pipeline.get_performance_stats()
            print(f"   VectorBT operations: {stats['vectorbt_operations']}")
            print(f"   Economic evaluations: {stats['economic_evaluations']}")
            print(f"   Feature selections: {stats['feature_selections']}")
            print(f"   Interaction generations: {stats['interaction_generations']}")
            print(f"   HTF generations: {stats['htf_generations']}")
            print(f"   Lookback optimizations: {stats['lookback_optimizations']}")
            print(f"   Enhanced feature generations: {stats['enhanced_feature_generations']}")
            
            # Display enhanced feature metrics
            if result.enhanced_feature_metrics:
                print("\n📈 Enhanced Feature Metrics:")
                for key, value in result.enhanced_feature_metrics.items():
                    print(f"   {key}: {value}")
            
        else:
            print(f"❌ Full pipeline failed: {result.error_message}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main test function."""
    print("🧪 ENHANCED UNIFIED PIPELINE TEST SUITE")
    print("="*80)
    print("Testing comprehensive feature generation including:")
    print("✅ Cross timeframe features with optimized lookback period")
    print("✅ Interaction (2-3) features with optimized lookback period")
    print("✅ Feature creation in multiple ways (addition, subtraction, log, multiplication, division)")
    print("✅ No features with optimized lookback period")
    print("="*80)
    
    # Test enhanced feature generation
    test_enhanced_feature_generation()
    
    # Test full pipeline
    test_full_pipeline()
    
    print("\n" + "="*80)
    print("🎉 TEST SUITE COMPLETED")
    print("="*80)

if __name__ == "__main__":
    main()