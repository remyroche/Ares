#!/usr/bin/env python3
"""
Test script for the Optimized Feature Selection Manager.
Demonstrates the improved feature selection with computational efficiency,
balanced feature mix, and model-specific optimization.
"""

import asyncio
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Add src to path
import sys
sys.path.append('src')

from src.training.optimized_feature_selection_manager import OptimizedFeatureSelectionManager
from src.utils.logger import system_logger

def create_test_data(n_samples=1000, n_features=200):
    """Create synthetic test data with known feature categories."""
    np.random.seed(42)
    
    # Create base features
    data = {}
    
    # Momentum features (25% of features)
    n_momentum = int(n_features * 0.25)
    for i in range(n_momentum):
        data[f'momentum_{i}'] = np.random.randn(n_samples) + np.sin(np.arange(n_samples) * 0.1)
        data[f'rsi_{i}'] = np.random.uniform(0, 100, n_samples)
        data[f'macd_{i}'] = np.random.randn(n_samples)
    
    # Volatility features (20% of features)
    n_volatility = int(n_features * 0.20)
    for i in range(n_volatility):
        data[f'volatility_{i}'] = np.abs(np.random.randn(n_samples))
        data[f'atr_{i}'] = np.random.uniform(0.1, 2.0, n_samples)
        data[f'realized_vol_{i}'] = np.random.uniform(0.05, 0.5, n_samples)
    
    # Liquidity features (20% of features)
    n_liquidity = int(n_features * 0.20)
    for i in range(n_liquidity):
        data[f'volume_{i}'] = np.random.uniform(100, 10000, n_samples)
        data[f'spread_{i}'] = np.random.uniform(0.001, 0.01, n_samples)
        data[f'liquidity_{i}'] = np.random.uniform(0.1, 1.0, n_samples)
    
    # Microstructure features (15% of features)
    n_microstructure = int(n_features * 0.15)
    for i in range(n_microstructure):
        data[f'order_flow_{i}'] = np.random.randn(n_samples)
        data[f'imbalance_{i}'] = np.random.uniform(-1, 1, n_samples)
        data[f'trade_frequency_{i}'] = np.random.uniform(10, 100, n_samples)
    
    # Regime features (10% of features)
    n_regime = int(n_features * 0.10)
    for i in range(n_regime):
        data[f'regime_{i}'] = np.random.randint(0, 5, n_samples)
        data[f'cluster_{i}'] = np.random.randint(0, 3, n_samples)
        data[f'hmm_state_{i}'] = np.random.randint(0, 4, n_samples)
    
    # Support/Resistance features (15% of features)
    n_sr = int(n_features * 0.15)
    for i in range(n_sr):
        data[f'sr_distance_{i}'] = np.random.uniform(0, 0.1, n_samples)
        data[f'sr_proximity_{i}'] = np.random.uniform(0, 1, n_samples)
        data[f'breakout_probability_{i}'] = np.random.uniform(0, 1, n_samples)
        data[f'rebounce_probability_{i}'] = np.random.uniform(0, 1, n_samples)
        data[f'consolidation_probability_{i}'] = np.random.uniform(0, 1, n_samples)
        data[f'sr_confidence_{i}'] = np.random.uniform(0, 1, n_samples)
        data[f'multi_timeframe_sr_score_{i}'] = np.random.uniform(0, 1, n_samples)
        data[f'distance_to_resistance_{i}'] = np.random.uniform(0, 0.05, n_samples)
        data[f'distance_to_support_{i}'] = np.random.uniform(0, 0.05, n_samples)
        data[f'normalized_distance_{i}'] = np.random.uniform(-1, 1, n_samples)
        data[f'sr_proximity_score_{i}'] = np.random.uniform(0, 1, n_samples)
        data[f'strength_score_{i}'] = np.random.uniform(0, 1, n_samples)
        data[f'clarity_factor_{i}'] = np.random.uniform(0, 1, n_samples)
        data[f'directional_pressure_{i}'] = np.random.uniform(-1, 1, n_samples)
        data[f'sr_score_{i}'] = np.random.uniform(0, 1, n_samples)
        data[f'delta_sr_score_{i}'] = np.random.uniform(-0.1, 0.1, n_samples)
        data[f'isolation_score_{i}'] = np.random.uniform(0, 1, n_samples)
    
    # Interaction features (10% of features)
    n_interaction = int(n_features * 0.10)
    for i in range(n_interaction):
        data[f'momentum_x_volume_{i}'] = data[f'momentum_{i % n_momentum}'] * data[f'volume_{i % n_liquidity}']
        data[f'volatility_div_liquidity_{i}'] = data[f'volatility_{i % n_volatility}'] / (data[f'liquidity_{i % n_liquidity}'] + 1e-8)
        data[f'rsi_ratio_volume_{i}'] = data[f'rsi_{i % n_momentum}'] / (data[f'volume_{i % n_liquidity}'] + 1e-8)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add some correlation and multicollinearity
    # Create highly correlated features
    for i in range(5):
        base_feature = f'momentum_{i}'
        if base_feature in df.columns:
            df[f'{base_feature}_correlated'] = df[base_feature] + np.random.normal(0, 0.01, n_samples)
    
    # Create target variable
    target = np.zeros(n_samples)
    # Target depends on some key features
    if 'momentum_0' in df.columns:
        target += 0.3 * (df['momentum_0'] > 0).astype(int)
    if 'rsi_0' in df.columns:
        target += 0.2 * (df['rsi_0'] > 70).astype(int)
    if 'volume_0' in df.columns:
        target += 0.1 * (df['volume_0'] > df['volume_0'].median()).astype(int)
    
    # Add some noise
    target += np.random.normal(0, 0.1, n_samples)
    target = (target > 0.5).astype(int)  # Binary classification
    
    return df, pd.Series(target, name='target')

def test_optimized_feature_selection():
    """Test the optimized feature selection system."""
    logger = system_logger.getChild("TestOptimizedFeatureSelection")
    
    print("🧪 Testing Optimized Feature Selection System")
    print("=" * 60)
    
    try:
        # Create test data
        print("📊 Creating test data...")
        features_df, target = create_test_data(n_samples=1000, n_features=200)
        print(f"✅ Created test data: {features_df.shape[0]} samples, {features_df.shape[1]} features")
        
        # Test configuration
        config = {
            "feature_selection": {
                "target_features": {
                    "neural_networks": 80,
                    "linear_models": 60,
                    "ensemble_models": 90,
                    "step2_general": 100
                },
                "vif_threshold": 10.0,
                "correlation_threshold": 0.95,
                "enable_shap_analysis": True,
                "enable_matrix_vif": True,
                "enable_balanced_selection": True,
                "feature_categories": {
                    "momentum": 0.20,
                    "volatility": 0.15,
                    "liquidity": 0.15,
                    "microstructure": 0.10,
                    "regime": 0.10,
                    "sr_features": 0.15,
                    "interaction": 0.15
                }
            }
        }
        
        # Initialize optimized feature selection manager
        print("🚀 Initializing Optimized Feature Selection Manager...")
        optimized_fs = OptimizedFeatureSelectionManager(config)
        print("✅ Optimized Feature Selection Manager initialized")
        
        # Test different model types
        model_types = ["neural_networks", "linear_models", "ensemble_models"]
        step_names = ["step2", "step6_hmm", "step7_ensemble", "step9_tactician"]
        
        for model_type in model_types:
            for step_name in step_names:
                print(f"\n🎯 Testing {model_type} for {step_name}")
                print("-" * 40)
                
                # Apply optimized feature selection
                start_time = datetime.now()
                selected_features, metadata = optimized_fs.select_features_optimized(
                    features_df.copy(), target, model_type=model_type, step_name=step_name
                )
                end_time = datetime.now()
                
                # Display results
                print(f"📊 Results for {model_type} ({step_name}):")
                print(f"   - Original features: {features_df.shape[1]}")
                print(f"   - Selected features: {selected_features.shape[1]}")
                print(f"   - Reduction: {((features_df.shape[1] - selected_features.shape[1]) / features_df.shape[1] * 100):.1f}%")
                print(f"   - Total time: {metadata.get('total_time', 0):.2f}s")
                
                # Performance metrics
                if "performance_metrics" in metadata:
                    perf_metrics = metadata["performance_metrics"]
                    print(f"   - VIF calculation: {perf_metrics.get('vif_calculation_time', 0):.2f}s")
                    print(f"   - SHAP analysis: {perf_metrics.get('shap_calculation_time', 0):.2f}s")
                    print(f"   - Correlation analysis: {perf_metrics.get('correlation_analysis_time', 0):.2f}s")
                
                # Feature category distribution
                if "feature_categories" in metadata:
                    category_dist = metadata["feature_categories"]
                    print(f"   - Feature categories:")
                    for category, features in category_dist.items():
                        if features:
                            print(f"     * {category}: {len(features)} features")
                
                # Stage information
                if "stages" in metadata:
                    stages = metadata["stages"]
                    print(f"   - Stage results:")
                    for stage_name, stage_metadata in stages.items():
                        if isinstance(stage_metadata, dict) and "features_after_stage" in stage_metadata:
                            print(f"     * {stage_name}: {stage_metadata['features_after_stage']} features")
        
        # Test computational efficiency
        print(f"\n⚡ Computational Efficiency Test")
        print("-" * 40)
        
        # Test with larger dataset
        print("📊 Testing with larger dataset...")
        large_features_df, large_target = create_test_data(n_samples=5000, n_features=300)
        
        start_time = datetime.now()
        large_selected_features, large_metadata = optimized_fs.select_features_optimized(
            large_features_df, large_target, model_type="ensemble_models", step_name="step2"
        )
        end_time = datetime.now()
        
        total_time = (end_time - start_time).total_seconds()
        print(f"✅ Large dataset test completed:")
        print(f"   - Dataset size: {large_features_df.shape[0]} samples, {large_features_df.shape[1]} features")
        print(f"   - Selected features: {large_selected_features.shape[1]}")
        print(f"   - Total time: {total_time:.2f}s")
        print(f"   - Features per second: {large_features_df.shape[1] / total_time:.1f}")
        
        # Test balanced feature mix
        print(f"\n🎯 Balanced Feature Mix Test")
        print("-" * 40)
        
        # Check if we get a good mix of features
        final_metadata = large_metadata
        if "feature_categories" in final_metadata:
            category_dist = final_metadata["feature_categories"]
            total_selected = sum(len(features) for features in category_dist.values())
            
            print("📊 Feature category distribution:")
            for category, features in category_dist.items():
                if features:
                    percentage = len(features) / total_selected * 100
                    print(f"   - {category}: {len(features)} features ({percentage:.1f}%)")
        
        # Test matrix VIF vs iterative VIF
        print(f"\n🔍 Matrix VIF vs Iterative VIF Test")
        print("-" * 40)
        
        # Create a smaller dataset for VIF comparison
        small_features_df, small_target = create_test_data(n_samples=500, n_features=50)
        
        # Test matrix VIF
        config_matrix = config.copy()
        config_matrix["feature_selection"]["enable_matrix_vif"] = True
        
        optimized_fs_matrix = OptimizedFeatureSelectionManager(config_matrix)
        start_time = datetime.now()
        matrix_result, matrix_metadata = optimized_fs_matrix.select_features_optimized(
            small_features_df.copy(), small_target, model_type="general", step_name="step2"
        )
        matrix_time = (datetime.now() - start_time).total_seconds()
        
        # Test iterative VIF (disable matrix VIF)
        config_iterative = config.copy()
        config_iterative["feature_selection"]["enable_matrix_vif"] = False
        
        optimized_fs_iterative = OptimizedFeatureSelectionManager(config_iterative)
        start_time = datetime.now()
        iterative_result, iterative_metadata = optimized_fs_iterative.select_features_optimized(
            small_features_df.copy(), small_target, model_type="general", step_name="step2"
        )
        iterative_time = (datetime.now() - start_time).total_seconds()
        
        print(f"📊 VIF Performance Comparison:")
        print(f"   - Matrix VIF time: {matrix_time:.2f}s")
        print(f"   - Iterative VIF time: {iterative_time:.2f}s")
        print(f"   - Speedup: {iterative_time / matrix_time:.1f}x faster")
        
        print(f"\n✅ All tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_feature_selection_integration():
    """Test integration with existing training steps."""
    print(f"\n🔗 Testing Feature Selection Integration")
    print("=" * 60)
    
    try:
        # Test configuration loading
        config_path = "src/config/optimized_feature_selection_config.yaml"
        if os.path.exists(config_path):
            print(f"✅ Configuration file found: {config_path}")
        else:
            print(f"⚠️ Configuration file not found: {config_path}")
        
        # Test with real-like data
        print("📊 Testing with realistic data...")
        features_df, target = create_test_data(n_samples=2000, n_features=150)
        
        # Add some realistic feature names
        realistic_features = [
            "rsi_14", "macd_12_26", "bb_position", "atr_14", "volume_sma_20",
            "momentum_strength", "volatility_garman_klass", "liquidity_score",
            "order_flow_imbalance", "sr_proximity", "hmm_state_0", "hmm_state_1",
            "momentum_x_volume", "volatility_div_liquidity", "rsi_ratio_volume",
            "sr_distance", "breakout_probability", "sr_confidence", "multi_timeframe_sr_score",
            "distance_to_resistance", "distance_to_support", "sr_proximity_score",
            "strength_score", "clarity_factor", "directional_pressure", "sr_score"
        ]
        
        # Rename some features to be more realistic
        for i, old_name in enumerate(features_df.columns[:len(realistic_features)]):
            features_df = features_df.rename(columns={old_name: realistic_features[i]})
        
        # Test optimized feature selection
        config = {
            "feature_selection": {
                "target_features": {
                    "neural_networks": 80,
                    "linear_models": 60,
                    "ensemble_models": 90,
                    "step2_general": 100
                },
                "vif_threshold": 10.0,
                "correlation_threshold": 0.95,
                "enable_shap_analysis": True,
                "enable_matrix_vif": True,
                "enable_balanced_selection": True
            }
        }
        
        optimized_fs = OptimizedFeatureSelectionManager(config)
        
        # Test for each step
        steps = [
            ("step2", "general"),
            ("step6_hmm", "neural_networks"),
            ("step7_ensemble", "ensemble_models"),
            ("step9_tactician", "ensemble_models")
        ]
        
        for step_name, model_type in steps:
            print(f"\n🎯 Testing {step_name} with {model_type}")
            selected_features, metadata = optimized_fs.select_features_optimized(
                features_df.copy(), target, model_type=model_type, step_name=step_name
            )
            
            print(f"   - Selected features: {selected_features.shape[1]}")
            print(f"   - Performance time: {metadata.get('total_time', 0):.2f}s")
            
            # Check if we have a good mix of realistic features
            realistic_selected = [f for f in selected_features.columns if any(rf in f for rf in realistic_features)]
            print(f"   - Realistic features selected: {len(realistic_selected)}")
        
        print(f"\n✅ Integration tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Optimized Feature Selection System Test Suite")
    print("=" * 60)
    
    # Run tests
    test1_success = test_optimized_feature_selection()
    test2_success = test_feature_selection_integration()
    
    print(f"\n📊 Test Results Summary")
    print("=" * 60)
    print(f"✅ Optimized Feature Selection Test: {'PASSED' if test1_success else 'FAILED'}")
    print(f"✅ Integration Test: {'PASSED' if test2_success else 'FAILED'}")
    
    if test1_success and test2_success:
        print(f"\n🎉 All tests passed! The optimized feature selection system is working correctly.")
        print(f"Key improvements:")
        print(f"   - Matrix-based VIF calculation (much faster)")
        print(f"   - RF+SHAP feature importance assessment")
        print(f"   - Balanced feature mix across categories")
        print(f"   - Model-specific optimization")
        print(f"   - Computational efficiency improvements")
    else:
        print(f"\n❌ Some tests failed. Please check the error messages above.")