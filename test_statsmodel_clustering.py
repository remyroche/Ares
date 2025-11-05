#!/usr/bin/env python3
"""
Simple test script for statsmodel clustering components
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_basic_imports():
    """Test basic imports."""
    print("Testing basic imports...")
    
    try:
        from src.training.steps.market_analysis.statsmodel_clustering.core import (
            MarkovRegressionAdapter,
            MarkovRegressionConfig,
            MarkovRegressionResult,
            ParameterMapper,
            MarkovRegressionDiagnostics,
            create_enhanced_markov_regression_adapter
        )
        print("✅ Core imports successful")
        return True
    except ImportError as e:
        print(f"❌ Core imports failed: {e}")
        return False

def test_utils_imports():
    """Test utils imports."""
    print("Testing utils imports...")
    
    try:
        from src.training.steps.market_analysis.statsmodel_clustering.utils import (
            ResultConverter,
            ConversionConfig,
            ConversionResult,
            convert_statsmodels_to_pyro,
            convert_pyro_to_statsmodels,
            create_unified_result,
            save_result_to_file,
            
            ModelValidator,
            ValidationConfig,
            ValidationResult,
            validate_input_data,
            validate_model_fit,
            cross_validate_regime_model,
            
            ModelDiagnostics,
            DiagnosticsConfig,
            DiagnosticsResult,
            analyze_model_fit,
            analyze_regime_stability,
            create_diagnostics_report
        )
        print("✅ Utils imports successful")
        return True
    except ImportError as e:
        print(f"❌ Utils imports failed: {e}")
        return False

def test_data_downloader():
    """Test data downloader."""
    print("Testing data downloader...")
    
    try:
        from src.training.steps.market_analysis.statsmodel_clustering.core import (
            BaseDataDownloader,
            StandardDataDownloader,
            create_data_downloader,
            download_clustering_data
        )
        print("✅ Data downloader imports successful")
        return True
    except ImportError as e:
        print(f"❌ Data downloader imports failed: {e}")
        return False

def test_markov_regression():
    """Test Markov regression with sample data."""
    print("Testing Markov regression...")
    
    try:
        from src.training.steps.market_analysis.statsmodel_clustering.core import (
            create_enhanced_markov_regression_adapter
        )
        
        # Create sample data
        np.random.seed(42)
        n_samples = 500
        data = pd.DataFrame({
            'returns': np.random.normal(0, 0.02, n_samples),
            'volatility': np.random.gamma(2, 0.01, n_samples)
        })
        
        # Create adapter
        adapter = create_enhanced_markov_regression_adapter(
            k_regimes=3,
            enable_pca=False,
            enable_diagnostics=True,
            enable_hardware_optimization=False
        )
        
        print("✅ Markov regression adapter created successfully")
        return True
    except Exception as e:
        print(f"❌ Markov regression test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Running statsmodel clustering tests...\n")
    
    tests = [
        test_basic_imports,
        test_utils_imports,
        test_data_downloader,
        test_markov_regression
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
            print()
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)
            print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    print(f"📊 Test Summary: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)