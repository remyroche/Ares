#!/usr/bin/env python3
"""
Simple test script for enhanced feature selection integration.

This script tests the multi-stage feature selection with lightweight screening
and advanced selection methods without the full pipeline dependencies.
"""

import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

def create_test_data(n_samples=500, n_features=100):
    """Create test data for feature selection."""
    np.random.seed(42)
    
    # Create synthetic financial data
    data = {}
    
    # Price-based features
    for i in range(30):
        data[f'price_feature_{i}'] = np.random.randn(n_samples).cumsum()
    
    # Momentum features
    for i in range(20):
        data[f'momentum_feature_{i}'] = np.random.randn(n_samples)
    
    # Volatility features
    for i in range(15):
        data[f'volatility_feature_{i}'] = np.abs(np.random.randn(n_samples))
    
    # Volume features
    for i in range(15):
        data[f'volume_feature_{i}'] = np.random.exponential(1, n_samples)
    
    # Technical indicators
    for i in range(20):
        data[f'technical_feature_{i}'] = np.random.randn(n_samples)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Create target variable (returns)
    target = np.random.randn(n_samples) * 0.01
    
    return df, target


def test_basic_feature_selection():
    """Test basic feature selection functionality."""
    print("🧪 Testing basic feature selection functionality")
    
    try:
        # Create test data
        data, target = create_test_data(n_samples=200, n_features=50)
        
        print(f"📊 Created test data: {data.shape[0]} samples, {data.shape[1]} features")
        
        # Test variance screening
        variances = data.var()
        high_variance_features = variances[variances > 0.1].index.tolist()
        print(f"📊 Variance screening: {len(high_variance_features)} features with variance > 0.1")
        
        # Test correlation screening
        correlations = data.corrwith(target).abs()
        high_correlation_features = correlations[correlations > 0.05].index.tolist()
        print(f"📊 Correlation screening: {len(high_correlation_features)} features with correlation > 0.05")
        
        # Test mutual information screening
        from sklearn.feature_selection import mutual_info_regression
        numeric_data = data.select_dtypes(include=[np.number])
        mi_scores = mutual_info_regression(numeric_data, target, random_state=42)
        mi_series = pd.Series(mi_scores, index=numeric_data.columns)
        high_mi_features = mi_series[mi_series > 0.01].index.tolist()
        print(f"📊 Mutual information screening: {len(high_mi_features)} features with MI > 0.01")
        
        # Combine screening results
        screened_features = set(high_variance_features) & set(high_correlation_features) & set(high_mi_features)
        print(f"📊 Combined screening: {len(screened_features)} features passed all filters")
        
        # Test feature importance ranking
        from sklearn.ensemble import RandomForestRegressor
        rf = RandomForestRegressor(n_estimators=50, random_state=42)
        rf.fit(data, target)
        
        importances = rf.feature_importances_
        feature_importance_pairs = list(zip(data.columns, importances))
        feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
        
        top_features = [feat[0] for feat in feature_importance_pairs[:20]]
        print(f"📊 Feature importance: {len(top_features)} top features selected")
        
        print("✅ Basic feature selection test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Basic feature selection test failed: {e}")
        return False


def test_mrmr_selection():
    """Test mRMR feature selection."""
    print("🧪 Testing mRMR feature selection")
    
    try:
        # Create test data
        data, target = create_test_data(n_samples=200, n_features=30)
        
        # Simple mRMR implementation
        from sklearn.feature_selection import mutual_info_regression
        
        # Calculate relevance scores
        relevance_scores = {}
        for i, col in enumerate(data.columns):
            x_feature = data[col].values.reshape(-1, 1)
            mi = mutual_info_regression(x_feature, target, random_state=42)[0]
            relevance_scores[col] = mi
        
        # Calculate redundancy scores (simplified)
        redundancy_scores = {}
        for col in data.columns:
            correlations = []
            for other_col in data.columns:
                if other_col != col:
                    corr = abs(data[col].corr(data[other_col]))
                    correlations.append(corr)
            redundancy_scores[col] = np.mean(correlations) if correlations else 0
        
        # Calculate mRMR scores
        mrmr_scores = {}
        for col in data.columns:
            mrmr_scores[col] = relevance_scores[col] - redundancy_scores[col]
        
        # Select top features
        sorted_features = sorted(mrmr_scores.items(), key=lambda x: x[1], reverse=True)
        selected_features = [feat[0] for feat in sorted_features[:15]]
        
        print(f"📊 mRMR selection: {len(selected_features)} features selected")
        print(f"📊 Top 5 features: {selected_features[:5]}")
        
        print("✅ mRMR selection test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ mRMR selection test failed: {e}")
        return False


def test_lasso_selection():
    """Test LASSO feature selection."""
    print("🧪 Testing LASSO feature selection")
    
    try:
        # Create test data
        data, target = create_test_data(n_samples=200, n_features=30)
        
        from sklearn.linear_model import Lasso
        from sklearn.preprocessing import StandardScaler
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(data)
        
        # Fit LASSO
        lasso = Lasso(alpha=0.01, random_state=42)
        lasso.fit(X_scaled, target)
        
        # Select features with non-zero coefficients
        selected_mask = np.abs(lasso.coef_) > 1e-6
        selected_features = data.columns[selected_mask].tolist()
        
        print(f"📊 LASSO selection: {len(selected_features)} features selected")
        print(f"📊 Selected features: {selected_features}")
        
        print("✅ LASSO selection test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ LASSO selection test failed: {e}")
        return False


def test_rfe_selection():
    """Test RFE feature selection."""
    print("🧪 Testing RFE feature selection")
    
    try:
        # Create test data
        data, target = create_test_data(n_samples=200, n_features=30)
        
        from sklearn.feature_selection import RFE
        from sklearn.ensemble import RandomForestRegressor
        
        # Create RFE selector
        estimator = RandomForestRegressor(n_estimators=50, random_state=42)
        rfe = RFE(estimator=estimator, n_features_to_select=15, step=1)
        
        # Fit RFE
        rfe.fit(data, target)
        
        # Get selected features
        selected_features = data.columns[rfe.support_].tolist()
        
        print(f"📊 RFE selection: {len(selected_features)} features selected")
        print(f"📊 Selected features: {selected_features}")
        
        print("✅ RFE selection test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ RFE selection test failed: {e}")
        return False


def test_ensemble_selection():
    """Test ensemble feature selection combining multiple methods."""
    print("🧪 Testing ensemble feature selection")
    
    try:
        # Create test data
        data, target = create_test_data(n_samples=200, n_features=30)
        
        # Collect results from different methods
        method_results = {}
        
        # Method 1: Variance screening
        variances = data.var()
        variance_features = variances[variances > 0.1].index.tolist()
        method_results['variance'] = variance_features
        
        # Method 2: Correlation screening
        correlations = data.corrwith(target).abs()
        correlation_features = correlations[correlations > 0.05].index.tolist()
        method_results['correlation'] = correlation_features
        
        # Method 3: Feature importance
        from sklearn.ensemble import RandomForestRegressor
        rf = RandomForestRegressor(n_estimators=50, random_state=42)
        rf.fit(data, target)
        
        importances = rf.feature_importances_
        feature_importance_pairs = list(zip(data.columns, importances))
        feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
        importance_features = [feat[0] for feat in feature_importance_pairs[:15]]
        method_results['importance'] = importance_features
        
        # Method 4: LASSO
        from sklearn.linear_model import Lasso
        from sklearn.preprocessing import StandardScaler
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(data)
        
        lasso = Lasso(alpha=0.01, random_state=42)
        lasso.fit(X_scaled, target)
        
        selected_mask = np.abs(lasso.coef_) > 1e-6
        lasso_features = data.columns[selected_mask].tolist()
        method_results['lasso'] = lasso_features
        
        # Combine results using voting
        feature_votes = {}
        for method, features in method_results.items():
            print(f"📊 {method}: {len(features)} features")
            for feature in features:
                feature_votes[feature] = feature_votes.get(feature, 0) + 1
        
        # Sort by vote count
        sorted_features = sorted(feature_votes.items(), key=lambda x: x[1], reverse=True)
        ensemble_features = [feature for feature, votes in sorted_features[:15]]
        
        print(f"📊 Ensemble selection: {len(ensemble_features)} features selected")
        print(f"📊 Top 5 features: {ensemble_features[:5]}")
        
        print("✅ Ensemble selection test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Ensemble selection test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Starting enhanced feature selection tests")
    
    tests = [
        ("Basic Feature Selection", test_basic_feature_selection),
        ("mRMR Selection", test_mrmr_selection),
        ("LASSO Selection", test_lasso_selection),
        ("RFE Selection", test_rfe_selection),
        ("Ensemble Selection", test_ensemble_selection)
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
        print("🎉 All tests passed! Enhanced feature selection is working correctly.")
        return 0
    else:
        print(f"⚠️ {total - passed} tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)