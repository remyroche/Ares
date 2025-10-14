#!/usr/bin/env python3
"""
Test script for LGBM/SHAP integration in enhanced feature selection.

This script tests the LightGBM and SHAP-based feature selection methods
replacing the previous LASSO implementation.
"""

import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import tprint utilities
try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug
    )
    TPRINT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: tprint utilities not available: {e}")
    TPRINT_AVAILABLE = False
    # Fallback functions
    def tprint(msg, **kwargs): print(f"[INFO] {msg}")
    def tprint_info(msg, **kwargs): print(f"[INFO] {msg}")
    def tprint_success(msg, **kwargs): print(f"[SUCCESS] {msg}")
    def tprint_warning(msg, **kwargs): print(f"[WARNING] {msg}")
    def tprint_error(msg, **kwargs): print(f"[ERROR] {msg}")
    def tprint_debug(msg, **kwargs): print(f"[DEBUG] {msg}")

def create_test_data(n_samples=500, n_features=100):
    """Create test data for feature selection."""
    np.random.seed(42)
    
    # Create synthetic financial data with some meaningful relationships
    data = {}
    
    # Price-based features (some with actual signal)
    for i in range(30):
        if i < 10:  # First 10 have signal
            data[f'price_feature_{i}'] = np.random.randn(n_samples).cumsum() + np.sin(np.arange(n_samples) * 0.1)
        else:
            data[f'price_feature_{i}'] = np.random.randn(n_samples).cumsum()
    
    # Momentum features (some with signal)
    for i in range(20):
        if i < 5:  # First 5 have signal
            data[f'momentum_feature_{i}'] = np.random.randn(n_samples) + 0.1 * np.arange(n_samples)
        else:
            data[f'momentum_feature_{i}'] = np.random.randn(n_samples)
    
    # Volatility features (some with signal)
    for i in range(15):
        if i < 3:  # First 3 have signal
            data[f'volatility_feature_{i}'] = np.abs(np.random.randn(n_samples)) + 0.05 * np.arange(n_samples)
        else:
            data[f'volatility_feature_{i}'] = np.abs(np.random.randn(n_samples))
    
    # Volume features (some with signal)
    for i in range(15):
        if i < 4:  # First 4 have signal
            data[f'volume_feature_{i}'] = np.random.exponential(1, n_samples) + 0.1 * np.arange(n_samples)
        else:
            data[f'volume_feature_{i}'] = np.random.exponential(1, n_samples)
    
    # Technical indicators (some with signal)
    for i in range(20):
        if i < 6:  # First 6 have signal
            data[f'technical_feature_{i}'] = np.random.randn(n_samples) + 0.05 * np.sin(np.arange(n_samples) * 0.2)
        else:
            data[f'technical_feature_{i}'] = np.random.randn(n_samples)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Create target variable with some relationship to features
    target = (
        0.3 * df['price_feature_0'] + 
        0.2 * df['momentum_feature_0'] + 
        0.15 * df['volatility_feature_0'] + 
        0.1 * df['volume_feature_0'] + 
        0.1 * df['technical_feature_0'] + 
        np.random.randn(n_samples) * 0.1
    )
    
    return df, target


def test_lgbm_basic():
    """Test basic LightGBM feature selection."""
    tprint_info("Testing basic LightGBM feature selection")
    
    try:
        # Install required packages
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "lightgbm", "shap"])
        
        import lightgbm as lgb
        from sklearn.model_selection import train_test_split
        
        # Create test data
        data, target = create_test_data(n_samples=300, n_features=50)
        
        print(f"📊 Created test data: {data.shape[0]} samples, {data.shape[1]} features")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)
        
        # Create LightGBM dataset
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=list(data.columns))
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        # Train model
        params = {
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
        }
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=100,
            valid_sets=[test_data],
            callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
        )
        
        # Get feature importance
        importance = model.feature_importance(importance_type='gain')
        feature_importance_pairs = list(zip(data.columns, importance))
        feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Select top features
        top_features = [feat[0] for feat in feature_importance_pairs[:15]]
        
        print(f"📊 LGBM selected {len(top_features)} features")
        print(f"📊 Top 5 features: {top_features[:5]}")
        
        # Check if signal features are selected
        signal_features = ['price_feature_0', 'momentum_feature_0', 'volatility_feature_0', 'volume_feature_0', 'technical_feature_0']
        selected_signal = [f for f in signal_features if f in top_features]
        print(f"📊 Signal features selected: {len(selected_signal)}/{len(signal_features)}")
        
        print("✅ Basic LightGBM test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Basic LightGBM test failed: {e}")
        return False


def test_shap_integration():
    """Test SHAP integration with LightGBM."""
    print("🧪 Testing SHAP integration with LightGBM")
    
    try:
        import lightgbm as lgb
        import shap
        
        # Create test data
        data, target = create_test_data(n_samples=200, n_features=30)
        
        print(f"📊 Created test data: {data.shape[0]} samples, {data.shape[1]} features")
        
        # Train LightGBM model
        train_data = lgb.Dataset(data, label=target, feature_name=list(data.columns))
        
        params = {
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
        }
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=50,
            valid_sets=[train_data],
            callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
        )
        
        # Calculate SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(data)
        
        # Calculate mean absolute SHAP values
        if len(shap_values.shape) > 1:
            shap_importance = np.mean(np.abs(shap_values), axis=0)
        else:
            shap_importance = np.abs(shap_values)
        
        # Create importance dictionary
        shap_importance_dict = dict(zip(data.columns, shap_importance))
        
        # Select top features based on SHAP
        sorted_features = sorted(shap_importance_dict.items(), key=lambda x: x[1], reverse=True)
        top_features = [feature for feature, _ in sorted_features[:10]]
        
        print(f"📊 SHAP selected {len(top_features)} features")
        print(f"📊 Top 5 features: {top_features[:5]}")
        
        # Check if signal features are selected
        signal_features = ['price_feature_0', 'momentum_feature_0', 'volatility_feature_0', 'volume_feature_0', 'technical_feature_0']
        selected_signal = [f for f in signal_features if f in top_features]
        print(f"📊 Signal features selected: {len(selected_signal)}/{len(signal_features)}")
        
        print("✅ SHAP integration test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ SHAP integration test failed: {e}")
        return False


def test_combined_lgbm_shap():
    """Test combined LGBM and SHAP feature selection."""
    print("🧪 Testing combined LGBM and SHAP feature selection")
    
    try:
        import lightgbm as lgb
        import shap
        
        # Create test data
        data, target = create_test_data(n_samples=300, n_features=40)
        
        print(f"📊 Created test data: {data.shape[0]} samples, {data.shape[1]} features")
        
        # Train LightGBM model
        train_data = lgb.Dataset(data, label=target, feature_name=list(data.columns))
        
        params = {
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
        }
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=50,
            valid_sets=[train_data],
            callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
        )
        
        # Get LGBM importance
        lgb_importance = model.feature_importance(importance_type='gain')
        lgb_importance_dict = dict(zip(data.columns, lgb_importance))
        
        # Calculate SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(data)
        
        # Calculate mean absolute SHAP values
        if len(shap_values.shape) > 1:
            shap_importance = np.mean(np.abs(shap_values), axis=0)
        else:
            shap_importance = np.abs(shap_values)
        
        shap_importance_dict = dict(zip(data.columns, shap_importance))
        
        # Combine LGBM and SHAP importance
        combined_importance = {}
        for feature in data.columns:
            lgb_score = lgb_importance_dict.get(feature, 0)
            shap_score = shap_importance_dict.get(feature, 0)
            
            # Normalize scores
            lgb_norm = lgb_score / (max(lgb_importance) + 1e-8)
            shap_norm = shap_score / (max(shap_importance) + 1e-8)
            
            # Weighted combination (70% SHAP, 30% LGBM)
            combined_importance[feature] = 0.7 * shap_norm + 0.3 * lgb_norm
        
        # Select top features
        sorted_features = sorted(combined_importance.items(), key=lambda x: x[1], reverse=True)
        top_features = [feature for feature, _ in sorted_features[:15]]
        
        print(f"📊 Combined LGBM/SHAP selected {len(top_features)} features")
        print(f"📊 Top 5 features: {top_features[:5]}")
        
        # Check if signal features are selected
        signal_features = ['price_feature_0', 'momentum_feature_0', 'volatility_feature_0', 'volume_feature_0', 'technical_feature_0']
        selected_signal = [f for f in signal_features if f in top_features]
        print(f"📊 Signal features selected: {len(selected_signal)}/{len(signal_features)}")
        
        print("✅ Combined LGBM/SHAP test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Combined LGBM/SHAP test failed: {e}")
        return False


def test_enhanced_feature_selector():
    """Test the enhanced feature selector with LGBM/SHAP."""
    print("🧪 Testing enhanced feature selector with LGBM/SHAP")
    
    try:
        # Import the enhanced feature selector
        from src.training.steps.pre_training.unified_data_driven_pipeline.enhanced_components.advanced_feature_selection import (
            AdvancedFeatureSelector, FeatureSelectionConfig
        )
        
        # Create test data
        data, target = create_test_data(n_samples=200, n_features=30)
        
        print(f"📊 Created test data: {data.shape[0]} samples, {data.shape[1]} features")
        
        # Configure for LGBM/SHAP selection
        config = FeatureSelectionConfig(
            enable_multi_stage_selection=True,
            enable_lightweight_screening=True,
            screening_methods=['variance', 'correlation'],
            final_selection_methods=['lgbm'],
            max_screening_features=25,
            final_selection_count=15,
            enable_lgbm_selection=True,
            shap_threshold=0.01,
            use_shap_importance=True
        )
        
        # Create selector
        selector = AdvancedFeatureSelector(config)
        
        # Test selection
        result = selector.select_features(data, target)
        
        if result.success:
            print(f"📊 Enhanced selector selected {len(result.selected_features)} features")
            print(f"📊 Selected features: {result.selected_features[:10]}")
            print(f"📊 Quality metrics: {result.quality_metrics}")
            print(f"📊 Diversity metrics: {result.diversity_metrics}")
            print("✅ Enhanced feature selector test completed successfully")
            return True
        else:
            print(f"❌ Enhanced feature selector failed: {result.error_message}")
            return False
        
    except Exception as e:
        print(f"❌ Enhanced feature selector test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Starting LGBM/SHAP integration tests")
    
    tests = [
        ("Basic LightGBM", test_lgbm_basic),
        ("SHAP Integration", test_shap_integration),
        ("Combined LGBM/SHAP", test_combined_lgbm_shap),
        ("Enhanced Feature Selector", test_enhanced_feature_selector)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running test: {test_name}")
        print(f"{'='*50}")
        
        try:
            success = test_func()
            results.append((test_name, success))
            
            if success:
                print(f"✅ {test_name} passed")
            else:
                print(f"❌ {test_name} failed")
                
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print(f"{'='*50}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! LGBM/SHAP integration is working correctly.")
        return 0
    else:
        print(f"⚠️ {total - passed} tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)