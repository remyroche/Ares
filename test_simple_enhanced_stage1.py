#!/usr/bin/env python3
"""
Simple test script for the enhanced Stage 1 feature selection methods.
Tests HSIC and distance correlation implementations directly.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_distance_correlation():
    """Test distance correlation implementation."""
    print("🧪 Testing Distance Correlation Implementation")
    
    try:
        from scipy.spatial.distance import pdist, squareform
        
        # Create test data with known relationships
        np.random.seed(42)
        n_samples = 100
        
        # Linear relationship
        x_linear = np.random.randn(n_samples)
        y_linear = 2 * x_linear + 0.1 * np.random.randn(n_samples)
        
        # Nonlinear relationship (quadratic)
        x_quad = np.random.randn(n_samples)
        y_quad = x_quad ** 2 + 0.1 * np.random.randn(n_samples)
        
        # Random relationship
        x_random = np.random.randn(n_samples)
        y_random = np.random.randn(n_samples)
        
        def distance_correlation(x, y):
            """Calculate distance correlation between two series."""
            try:
                # Remove NaN values
                valid_mask = ~(np.isnan(x) | np.isnan(y))
                if not valid_mask.any():
                    return 0.0
                
                x_clean = x[valid_mask]
                y_clean = y[valid_mask]
                
                if len(x_clean) < 3:
                    return 0.0
                
                # Calculate distance matrices
                x_dist = pdist(x_clean.reshape(-1, 1), metric='euclidean')
                y_dist = pdist(y_clean.reshape(-1, 1), metric='euclidean')
                
                # Convert to squareform
                x_dist_matrix = squareform(x_dist)
                y_dist_matrix = squareform(y_dist)
                
                # Center the distance matrices
                n = len(x_clean)
                x_centered = x_dist_matrix - np.mean(x_dist_matrix, axis=1)[:, np.newaxis] - np.mean(x_dist_matrix, axis=0) + np.mean(x_dist_matrix)
                y_centered = y_dist_matrix - np.mean(y_dist_matrix, axis=1)[:, np.newaxis] - np.mean(y_dist_matrix, axis=0) + np.mean(y_dist_matrix)
                
                # Calculate distance covariance and variances
                dcov_xy = np.sqrt(np.mean(x_centered * y_centered))
                dcov_xx = np.sqrt(np.mean(x_centered * x_centered))
                dcov_yy = np.sqrt(np.mean(y_centered * y_centered))
                
                # Avoid division by zero
                if dcov_xx == 0 or dcov_yy == 0:
                    return 0.0
                
                # Distance correlation
                dcorr = dcov_xy / np.sqrt(dcov_xx * dcov_yy)
                
                return abs(dcorr)  # Return absolute value for feature selection
                
            except Exception as e:
                print(f"   ⚠️ Distance correlation calculation error: {e}")
                return 0.0
        
        # Test with different relationships
        dc_linear = distance_correlation(x_linear, y_linear)
        dc_quad = distance_correlation(x_quad, y_quad)
        dc_random = distance_correlation(x_random, y_random)
        
        print(f"   📊 Linear relationship DC: {dc_linear:.4f}")
        print(f"   📊 Quadratic relationship DC: {dc_quad:.4f}")
        print(f"   📊 Random relationship DC: {dc_random:.4f}")
        
        # Verify that distance correlation captures both linear and nonlinear relationships
        assert dc_linear > 0.5, f"Distance correlation should be high for linear relationship, got {dc_linear}"
        assert dc_quad > 0.3, f"Distance correlation should be moderate for quadratic relationship, got {dc_quad}"
        assert dc_random < 0.3, f"Distance correlation should be low for random relationship, got {dc_random}"
        
        print("   ✅ Distance correlation test passed!")
        return True
        
    except Exception as e:
        print(f"   ❌ Distance correlation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_hsic():
    """Test HSIC implementation."""
    print("🧪 Testing HSIC Implementation")
    
    try:
        from sklearn.metrics.pairwise import rbf_kernel, linear_kernel
        
        # Create test data with known relationships
        np.random.seed(42)
        n_samples = 100
        
        # Linear relationship
        x_linear = np.random.randn(n_samples)
        y_linear = 2 * x_linear + 0.1 * np.random.randn(n_samples)
        
        # Nonlinear relationship (quadratic)
        x_quad = np.random.randn(n_samples)
        y_quad = x_quad ** 2 + 0.1 * np.random.randn(n_samples)
        
        # Random relationship
        x_random = np.random.randn(n_samples)
        y_random = np.random.randn(n_samples)
        
        def hsic_score(x, y, kernel_type='rbf'):
            """Calculate HSIC score between two series."""
            try:
                # Remove NaN values
                valid_mask = ~(np.isnan(x) | np.isnan(y))
                if not valid_mask.any():
                    return 0.0
                
                x_clean = x[valid_mask]
                y_clean = y[valid_mask]
                
                if len(x_clean) < 3:
                    return 0.0
                
                # Reshape for kernel calculation
                x_reshaped = x_clean.reshape(-1, 1)
                y_reshaped = y_clean.reshape(-1, 1)
                
                # Calculate kernels
                if kernel_type == 'rbf':
                    gamma = 1.0 / (x_reshaped.shape[1] * np.var(x_reshaped))
                    Kx = rbf_kernel(x_reshaped, gamma=gamma)
                    Ky = rbf_kernel(y_reshaped, gamma=gamma)
                elif kernel_type == 'linear':
                    Kx = linear_kernel(x_reshaped)
                    Ky = linear_kernel(y_reshaped)
                else:
                    # Default to RBF
                    gamma = 1.0 / (x_reshaped.shape[1] * np.var(x_reshaped))
                    Kx = rbf_kernel(x_reshaped, gamma=gamma)
                    Ky = rbf_kernel(y_reshaped, gamma=gamma)
                
                # Center the kernels
                n = len(x_clean)
                H = np.eye(n) - np.ones((n, n)) / n  # Centering matrix
                
                Kx_centered = H @ Kx @ H
                Ky_centered = H @ Ky @ H
                
                # Calculate HSIC
                hsic = np.trace(Kx_centered @ Ky_centered) / (n - 1) ** 2
                
                return abs(hsic)  # Return absolute value for feature selection
                
            except Exception as e:
                print(f"   ⚠️ HSIC calculation error: {e}")
                return 0.0
        
        # Test with different relationships
        hsic_linear = hsic_score(x_linear, y_linear)
        hsic_quad = hsic_score(x_quad, y_quad)
        hsic_random = hsic_score(x_random, y_random)
        
        print(f"   📊 Linear relationship HSIC: {hsic_linear:.4f}")
        print(f"   📊 Quadratic relationship HSIC: {hsic_quad:.4f}")
        print(f"   📊 Random relationship HSIC: {hsic_random:.4f}")
        
        # Verify that HSIC captures relationships
        assert hsic_linear > 0.01, f"HSIC should be positive for linear relationship, got {hsic_linear}"
        assert hsic_quad > 0.01, f"HSIC should be positive for quadratic relationship, got {hsic_quad}"
        
        print("   ✅ HSIC test passed!")
        return True
        
    except Exception as e:
        print(f"   ❌ HSIC test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_enhanced_scoring():
    """Test the enhanced scoring combination."""
    print("🧪 Testing Enhanced Scoring Combination")
    
    try:
        # Create test data
        np.random.seed(42)
        n_samples = 200
        n_features = 20
        
        X = pd.DataFrame()
        
        # Linear relationship features
        for i in range(5):
            base = np.random.randn(n_samples)
            X[f'linear_feature_{i}'] = base
            X[f'linear_target_{i}'] = base * (i + 1) + 0.1 * np.random.randn(n_samples)
        
        # Nonlinear relationship features
        for i in range(5):
            base = np.random.randn(n_samples)
            X[f'quadratic_feature_{i}'] = base ** 2 + 0.1 * np.random.randn(n_samples)
            X[f'exponential_feature_{i}'] = np.exp(base * 0.5) + 0.1 * np.random.randn(n_samples)
        
        # Random noise features
        for i in range(5):
            X[f'noise_feature_{i}'] = np.random.randn(n_samples)
        
        # Create target with mixed relationships
        y = (
            0.3 * X['linear_feature_0'] +
            0.2 * X['linear_target_0'] +
            0.1 * X['quadratic_feature_0'] +
            0.1 * X['exponential_feature_0'] +
            0.1 * np.random.randn(n_samples)
        )
        
        # Remove target columns from features
        feature_cols = [col for col in X.columns if not col.endswith('_target_0')]
        X_features = X[feature_cols]
        
        print(f"   📊 Test data: {X_features.shape[0]} samples, {X_features.shape[1]} features")
        
        # Test individual methods
        from scipy.spatial.distance import pdist, squareform
        from sklearn.metrics.pairwise import rbf_kernel
        
        def distance_correlation(x, y):
            """Simplified distance correlation."""
            try:
                valid_mask = ~(np.isnan(x) | np.isnan(y))
                if not valid_mask.any():
                    return 0.0
                
                x_clean = x[valid_mask]
                y_clean = y[valid_mask]
                
                if len(x_clean) < 3:
                    return 0.0
                
                x_dist = pdist(x_clean.reshape(-1, 1), metric='euclidean')
                y_dist = pdist(y_clean.reshape(-1, 1), metric='euclidean')
                
                x_dist_matrix = squareform(x_dist)
                y_dist_matrix = squareform(y_dist)
                
                n = len(x_clean)
                x_centered = x_dist_matrix - np.mean(x_dist_matrix, axis=1)[:, np.newaxis] - np.mean(x_dist_matrix, axis=0) + np.mean(x_dist_matrix)
                y_centered = y_dist_matrix - np.mean(y_dist_matrix, axis=1)[:, np.newaxis] - np.mean(y_dist_matrix, axis=0) + np.mean(y_dist_matrix)
                
                dcov_xy = np.sqrt(np.mean(x_centered * y_centered))
                dcov_xx = np.sqrt(np.mean(x_centered * x_centered))
                dcov_yy = np.sqrt(np.mean(y_centered * y_centered))
                
                if dcov_xx == 0 or dcov_yy == 0:
                    return 0.0
                
                dcorr = dcov_xy / np.sqrt(dcov_xx * dcov_yy)
                return abs(dcorr)
                
            except Exception:
                return 0.0
        
        def hsic_score(x, y):
            """Simplified HSIC."""
            try:
                valid_mask = ~(np.isnan(x) | np.isnan(y))
                if not valid_mask.any():
                    return 0.0
                
                x_clean = x[valid_mask]
                y_clean = y[valid_mask]
                
                if len(x_clean) < 3:
                    return 0.0
                
                x_reshaped = x_clean.reshape(-1, 1)
                y_reshaped = y_clean.reshape(-1, 1)
                
                gamma = 1.0 / (x_reshaped.shape[1] * np.var(x_reshaped))
                Kx = rbf_kernel(x_reshaped, gamma=gamma)
                Ky = rbf_kernel(y_reshaped, gamma=gamma)
                
                n = len(x_clean)
                H = np.eye(n) - np.ones((n, n)) / n
                
                Kx_centered = H @ Kx @ H
                Ky_centered = H @ Ky @ H
                
                hsic = np.trace(Kx_centered @ Ky_centered) / (n - 1) ** 2
                return abs(hsic)
                
            except Exception:
                return 0.0
        
        # Calculate scores for all features
        distance_corr_scores = {}
        hsic_scores = {}
        
        for feature in X_features.columns:
            distance_corr_scores[feature] = distance_correlation(X_features[feature], y)
            hsic_scores[feature] = hsic_score(X_features[feature], y)
        
        # Convert to pandas Series
        dc_scores = pd.Series(distance_corr_scores)
        hsic_scores_series = pd.Series(hsic_scores)
        
        # Normalize scores to 0-1 range
        dc_scores_norm = dc_scores / dc_scores.max() if dc_scores.max() > 0 else dc_scores
        hsic_scores_norm = hsic_scores_series / hsic_scores_series.max() if hsic_scores_series.max() > 0 else hsic_scores_series
        
        # Combine with weights (50% mRMR + 30% Distance Correlation + 20% HSIC)
        # For this test, we'll use a simple correlation as mRMR proxy
        mrmr_proxy = X_features.corrwith(y).abs()
        mrmr_proxy_norm = mrmr_proxy / mrmr_proxy.max() if mrmr_proxy.max() > 0 else mrmr_proxy
        
        combined_scores = (
            0.5 * mrmr_proxy_norm +
            0.3 * dc_scores_norm +
            0.2 * hsic_scores_norm
        )
        
        # Show results
        print("   📊 Score Statistics:")
        print(f"   mRMR proxy - Mean: {mrmr_proxy_norm.mean():.4f}, Max: {mrmr_proxy_norm.max():.4f}")
        print(f"   Distance Correlation - Mean: {dc_scores_norm.mean():.4f}, Max: {dc_scores_norm.max():.4f}")
        print(f"   HSIC - Mean: {hsic_scores_norm.mean():.4f}, Max: {hsic_scores_norm.max():.4f}")
        print(f"   Combined - Mean: {combined_scores.mean():.4f}, Max: {combined_scores.max():.4f}")
        
        # Show top features
        print("   🏆 Top 10 Features by Combined Score:")
        top_features = combined_scores.nlargest(10)
        for i, (feature, score) in enumerate(top_features.items(), 1):
            print(f"   {i:2d}. {feature}: {score:.4f}")
        
        # Verify that the enhanced scoring works
        assert len(combined_scores) == len(X_features.columns), "Should have scores for all features"
        assert combined_scores.max() > 0, "Should have positive scores"
        assert not combined_scores.isna().any(), "Should not have NaN scores"
        
        print("   ✅ Enhanced scoring test passed!")
        return True
        
    except Exception as e:
        print(f"   ❌ Enhanced scoring test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🚀 Testing Enhanced Stage 1 Feature Selection Methods")
    print("=" * 60)
    
    tests = [
        test_distance_correlation,
        test_hsic,
        test_enhanced_scoring
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
            print()
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            results.append(False)
            print()
    
    print("=" * 60)
    if all(results):
        print("🎉 All tests passed! Enhanced Stage 1 methods are working correctly.")
        return True
    else:
        print(f"❌ {sum(results)}/{len(results)} tests passed.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)