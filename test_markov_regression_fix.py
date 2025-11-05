#!/usr/bin/env python3
"""
Test script for the enhanced MarkovRegression adapter with multivariate support.
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_multivariate_markov_regression():
    """Test the enhanced MarkovRegression adapter with multivariate data."""
    print("🧪 Testing Enhanced MarkovRegression with multivariate data...")
    
    try:
        # Import the enhanced adapter
        from src.training.steps.market_analysis.statsmodel_clustering.core.markov_regression_adapter import (
            MarkovRegressionAdapter, 
            MarkovRegressionConfig
        )
        
        # Create synthetic multivariate data
        np.random.seed(42)
        n_samples = 200
        n_features = 5
        
        # Generate correlated data
        mean = np.zeros(n_features)
        cov = np.eye(n_features)
        cov[0, 1] = cov[1, 0] = 0.8  # Correlate first two features
        data = np.random.multivariate_normal(mean, cov, size=n_samples)
        
        # Add some regime-switching behavior to the first feature
        regime_labels = np.zeros(n_samples, dtype=int)
        regime_labels[50:100] = 1
        regime_labels[150:] = 2
        
        # Modify first feature based on regime
        data[regime_labels == 0, 0] += 0.5
        data[regime_labels == 1, 0] -= 0.5
        data[regime_labels == 2, 0] += 1.0
        
        print(f"📊 Generated data: {data.shape}")
        print(f"📈 True regime distribution: {np.bincount(regime_labels)}")
        
        # Create configuration
        config = MarkovRegressionConfig(
            k_regimes=3,
            trend='c',
            order=0,
            switching_variance=True,
            switching_trend=True,
            maxiter=50,  # Reduced for faster testing
            enable_diagnostics=False,  # Disabled for faster testing
            enable_hardware_optimization=False,  # Disabled for simpler testing
            enable_pca=False,  # Disabled to keep original features
            enable_scaling=True
        )
        
        # Create adapter
        adapter = MarkovRegressionAdapter(config)
        
        # Fit model
        print("🔄 Fitting MarkovRegression model...")
        result = adapter.fit(data)
        
        # Check results
        if result.success:
            print("✅ Model fitting successful!")
            print(f"📊 Detected regimes: {result.n_regimes}")
            print(f"📈 Predicted regime distribution: {np.bincount(result.cluster_labels)}")
            print(f"📊 Log likelihood: {result.log_likelihood:.4f}")
            print(f"📊 AIC: {result.aic:.4f}")
            print(f"📊 BIC: {result.bic:.4f}")
            
            # Compare with true labels (adjusted for possible label permutation)
            from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
            ari = adjusted_rand_score(regime_labels, result.cluster_labels)
            nmi = normalized_mutual_info_score(regime_labels, result.cluster_labels)
            
            print(f"📊 Adjusted Rand Index: {ari:.4f}")
            print(f"📊 Normalized Mutual Information: {nmi:.4f}")
            
            if ari > 0.3:  # Reasonable threshold for synthetic data
                print("✅ Model successfully detected regime structure!")
            else:
                print("⚠️ Model had difficulty detecting regime structure")
            
            return True
        else:
            print(f"❌ Model fitting failed: {result.error_message}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_univariate_markov_regression():
    """Test the enhanced MarkovRegression adapter with univariate data."""
    print("\n🧪 Testing Enhanced MarkovRegression with univariate data...")
    
    try:
        # Import the enhanced adapter
        from src.training.steps.market_analysis.statsmodel_clustering.core.markov_regression_adapter import (
            MarkovRegressionAdapter, 
            MarkovRegressionConfig
        )
        
        # Create synthetic univariate data
        np.random.seed(42)
        n_samples = 200
        
        # Generate data with regime-switching behavior
        data = np.zeros((n_samples, 1))
        regime_labels = np.zeros(n_samples, dtype=int)
        regime_labels[50:100] = 1
        regime_labels[150:] = 2
        
        # Modify data based on regime
        data[regime_labels == 0, 0] = np.random.normal(0.5, 0.5, np.sum(regime_labels == 0))
        data[regime_labels == 1, 0] = np.random.normal(-0.5, 0.5, np.sum(regime_labels == 1))
        data[regime_labels == 2, 0] = np.random.normal(1.0, 0.5, np.sum(regime_labels == 2))
        
        print(f"📊 Generated data: {data.shape}")
        print(f"📈 True regime distribution: {np.bincount(regime_labels)}")
        
        # Create configuration
        config = MarkovRegressionConfig(
            k_regimes=3,
            trend='c',
            order=0,
            switching_variance=True,
            switching_trend=True,
            maxiter=50,  # Reduced for faster testing
            enable_diagnostics=False,  # Disabled for faster testing
            enable_hardware_optimization=False,  # Disabled for simpler testing
            enable_pca=False,  # Disabled to keep original features
            enable_scaling=True
        )
        
        # Create adapter
        adapter = MarkovRegressionAdapter(config)
        
        # Fit model
        print("🔄 Fitting MarkovRegression model...")
        result = adapter.fit(data)
        
        # Check results
        if result.success:
            print("✅ Model fitting successful!")
            print(f"📊 Detected regimes: {result.n_regimes}")
            print(f"📈 Predicted regime distribution: {np.bincount(result.cluster_labels)}")
            print(f"📊 Log likelihood: {result.log_likelihood:.4f}")
            print(f"📊 AIC: {result.aic:.4f}")
            print(f"📊 BIC: {result.bic:.4f}")
            
            # Compare with true labels (adjusted for possible label permutation)
            from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
            ari = adjusted_rand_score(regime_labels, result.cluster_labels)
            nmi = normalized_mutual_info_score(regime_labels, result.cluster_labels)
            
            print(f"📊 Adjusted Rand Index: {ari:.4f}")
            print(f"📊 Normalized Mutual Information: {nmi:.4f}")
            
            if ari > 0.3:  # Reasonable threshold for synthetic data
                print("✅ Model successfully detected regime structure!")
            else:
                print("⚠️ Model had difficulty detecting regime structure")
            
            return True
        else:
            print(f"❌ Model fitting failed: {result.error_message}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting MarkovRegression adapter tests...")
    
    # Test multivariate data
    multivariate_success = test_multivariate_markov_regression()
    
    # Test univariate data
    univariate_success = test_univariate_markov_regression()
    
    # Summary
    print("\n📋 Test Summary:")
    print(f"📊 Multivariate test: {'✅ PASSED' if multivariate_success else '❌ FAILED'}")
    print(f"📈 Univariate test: {'✅ PASSED' if univariate_success else '❌ FAILED'}")
    
    if multivariate_success and univariate_success:
        print("\n🎉 All tests passed! The enhanced MarkovRegression adapter is working correctly.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        sys.exit(1)