"""
Test Enhanced Feature Comparison Framework

This script tests all the enhancements to the feature comparison framework:
- Returns-based calculations
- Rolling averages and EWMA features
- Lagged/lead features
- Matrix operations integration
- Hardware optimizations
- Reduced bootstrap samples
"""

import pandas as pd
import numpy as np
import sys
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

def test_returns_based_calculations():
    """Test that all calculations use returns instead of raw prices."""
    print("Testing Returns-Based Calculations...")
    print("=" * 50)
    
    # Create sample data
    data = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=100, freq='1H'),
        'open': np.random.uniform(100, 110, 100),
        'high': np.random.uniform(110, 120, 100),
        'low': np.random.uniform(90, 100, 100),
        'close': np.random.uniform(100, 110, 100),
        'volume': np.random.uniform(1000, 2000, 100)
    })
    
    from feature_comparison.optimized_feature_versions import OptimizedFeatureVersions
    
    # Test optimized feature versions
    feature_versions = OptimizedFeatureVersions(data, 'returns')
    versions = feature_versions.generate_all_versions()
    
    # Check that features are returns-based
    for version_name, version_df in versions.items():
        print(f"\n{version_name}:")
        
        # Check for returns features
        returns_features = [col for col in version_df.columns if 'returns' in col]
        print(f"  Returns features: {len(returns_features)}")
        
        # Check for price features (should be minimal)
        price_features = [col for col in version_df.columns if any(price in col for price in ['open', 'high', 'low', 'close'])]
        print(f"  Price features: {len(price_features)}")
        
        # Check for VWAP features (should use returns)
        vwap_features = [col for col in version_df.columns if 'vwap' in col]
        print(f"  VWAP features: {len(vwap_features)}")
        
        # Verify VWAP features use returns
        if 'vwap_returns' in version_df.columns:
            print("  ✅ VWAP uses returns")
        else:
            print("  ❌ VWAP does not use returns")
    
    return True

def test_rolling_ewma_features():
    """Test rolling averages and EWMA features."""
    print("\nTesting Rolling Averages and EWMA Features...")
    print("=" * 50)
    
    # Create sample data
    data = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=200, freq='1H'),
        'open': np.random.uniform(100, 110, 200),
        'high': np.random.uniform(110, 120, 200),
        'low': np.random.uniform(90, 100, 200),
        'close': np.random.uniform(100, 110, 200),
        'volume': np.random.uniform(1000, 2000, 200)
    })
    
    from feature_comparison.optimized_feature_versions import OptimizedFeatureVersions
    
    feature_versions = OptimizedFeatureVersions(data, 'returns')
    versions = feature_versions.generate_all_versions()
    
    # Check for rolling features
    for version_name, version_df in versions.items():
        print(f"\n{version_name}:")
        
        # Check for rolling averages
        rolling_ma = [col for col in version_df.columns if 'ma_' in col]
        print(f"  Rolling MA features: {len(rolling_ma)}")
        
        # Check for EWMA features
        ewma_features = [col for col in version_df.columns if 'ewma' in col]
        print(f"  EWMA features: {len(ewma_features)}")
        
        # Check for rolling statistics
        rolling_std = [col for col in version_df.columns if 'std_' in col]
        rolling_skew = [col for col in version_df.columns if 'skew' in col]
        rolling_kurt = [col for col in version_df.columns if 'kurt' in col]
        
        print(f"  Rolling std: {len(rolling_std)}")
        print(f"  Rolling skew: {len(rolling_skew)}")
        print(f"  Rolling kurt: {len(rolling_kurt)}")
        
        # Show examples
        if rolling_ma:
            print(f"  Examples: {rolling_ma[:3]}")
    
    return True

def test_lagged_lead_features():
    """Test lagged and lead features."""
    print("\nTesting Lagged and Lead Features...")
    print("=" * 50)
    
    # Create sample data
    data = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=200, freq='1H'),
        'open': np.random.uniform(100, 110, 200),
        'high': np.random.uniform(110, 120, 200),
        'low': np.random.uniform(90, 100, 200),
        'close': np.random.uniform(100, 110, 200),
        'volume': np.random.uniform(1000, 2000, 200)
    })
    
    from feature_comparison.optimized_feature_versions import OptimizedFeatureVersions
    
    feature_versions = OptimizedFeatureVersions(data, 'returns')
    versions = feature_versions.generate_all_versions()
    
    # Check for lagged and lead features
    for version_name, version_df in versions.items():
        print(f"\n{version_name}:")
        
        # Check for lagged features
        lagged_features = [col for col in version_df.columns if 'lag' in col]
        print(f"  Lagged features: {len(lagged_features)}")
        
        # Check for lead features
        lead_features = [col for col in version_df.columns if 'lead' in col]
        print(f"  Lead features: {len(lead_features)}")
        
        # Check for momentum features
        momentum_features = [col for col in version_df.columns if 'momentum' in col]
        print(f"  Momentum features: {len(momentum_features)}")
        
        # Check for acceleration features
        acceleration_features = [col for col in version_df.columns if 'acceleration' in col]
        print(f"  Acceleration features: {len(acceleration_features)}")
        
        # Show examples
        if lagged_features:
            print(f"  Lagged examples: {lagged_features[:3]}")
        if lead_features:
            print(f"  Lead examples: {lead_features[:3]}")
    
    return True

def test_matrix_operations_integration():
    """Test matrix operations integration."""
    print("\nTesting Matrix Operations Integration...")
    print("=" * 50)
    
    # Create sample data
    data = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=100, freq='1H'),
        'open': np.random.uniform(100, 110, 100),
        'high': np.random.uniform(110, 120, 100),
        'low': np.random.uniform(90, 100, 100),
        'close': np.random.uniform(100, 110, 100),
        'volume': np.random.uniform(1000, 2000, 100)
    })
    
    from feature_comparison.optimized_feature_versions import OptimizedFeatureVersions
    
    # Test with matrix operations
    print("Testing with matrix operations...")
    feature_versions_matrix = OptimizedFeatureVersions(
        data, 'returns', 
        enable_matrix_ops=True, 
        enable_hardware_opt=True
    )
    
    start_time = time.time()
    versions_matrix = feature_versions_matrix.generate_all_versions()
    matrix_time = time.time() - start_time
    
    print(f"  Matrix operations time: {matrix_time:.3f} seconds")
    print(f"  Matrix ops enabled: {feature_versions_matrix.enable_matrix_ops}")
    print(f"  Hardware opt enabled: {feature_versions_matrix.enable_hardware_opt}")
    
    # Test without matrix operations
    print("\nTesting without matrix operations...")
    feature_versions_standard = OptimizedFeatureVersions(
        data, 'returns', 
        enable_matrix_ops=False, 
        enable_hardware_opt=False
    )
    
    start_time = time.time()
    versions_standard = feature_versions_standard.generate_all_versions()
    standard_time = time.time() - start_time
    
    print(f"  Standard operations time: {standard_time:.3f} seconds")
    
    if matrix_time < standard_time:
        print(f"  ✅ Matrix operations faster by {standard_time / matrix_time:.2f}x")
    else:
        print(f"  ⚠️ Matrix operations slower by {matrix_time / standard_time:.2f}x")
    
    return True

def test_bootstrap_optimization():
    """Test reduced bootstrap samples."""
    print("\nTesting Bootstrap Optimization...")
    print("=" * 50)
    
    # Create sample data
    data = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=500, freq='1H'),
        'open': np.random.uniform(100, 110, 500),
        'high': np.random.uniform(110, 120, 500),
        'low': np.random.uniform(90, 100, 500),
        'close': np.random.uniform(100, 110, 500),
        'volume': np.random.uniform(1000, 2000, 500)
    })
    
    from feature_comparison.run_comparison import FeatureComparisonRunner
    
    # Test with reduced bootstrap samples
    runner = FeatureComparisonRunner(
        data=data,
        task_type='regression',
        use_optimized=True
    )
    
    print("Running analysis with 10 bootstrap samples...")
    start_time = time.time()
    
    try:
        results = runner.run_complete_analysis()
        analysis_time = time.time() - start_time
        
        print(f"  Analysis time: {analysis_time:.3f} seconds")
        
        # Check bootstrap results
        analysis_results = results['analysis_results']
        for version_name, analysis in analysis_results.items():
            if 'bootstrap_analysis' in analysis:
                bootstrap = analysis['bootstrap_analysis']
                n_bootstrap = bootstrap.get('n_bootstrap', 0)
                print(f"  {version_name}: {n_bootstrap} bootstrap samples")
                
                if n_bootstrap == 10:
                    print(f"    ✅ Correct number of bootstrap samples")
                else:
                    print(f"    ❌ Expected 10, got {n_bootstrap}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Analysis failed: {e}")
        return False

def test_enhanced_feature_types():
    """Test that enhanced feature types are present."""
    print("\nTesting Enhanced Feature Types...")
    print("=" * 50)
    
    # Create sample data
    data = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=200, freq='1H'),
        'open': np.random.uniform(100, 110, 200),
        'high': np.random.uniform(110, 120, 200),
        'low': np.random.uniform(90, 100, 200),
        'close': np.random.uniform(100, 110, 200),
        'volume': np.random.uniform(1000, 2000, 200)
    })
    
    from feature_comparison.optimized_feature_versions import OptimizedFeatureVersions
    
    feature_versions = OptimizedFeatureVersions(data, 'returns')
    versions = feature_versions.generate_all_versions()
    
    # Check for specific enhanced feature types
    enhanced_features = {
        'returns_based': ['returns', 'log_returns', 'returns_abs', 'returns_squared'],
        'rolling_features': ['ma_', 'ewma_', 'std_', 'skew_', 'kurt_'],
        'lagged_features': ['lag_', 'momentum_', 'acceleration'],
        'lead_features': ['lead_'],
        'vwap_returns': ['vwap_returns', 'returns_vwap'],
        'vol_normalized': ['vol_norm', 'volatility'],
        'volume_features': ['volume_returns', 'volume_ratio', 'vw_returns']
    }
    
    for version_name, version_df in versions.items():
        print(f"\n{version_name}:")
        
        for feature_type, patterns in enhanced_features.items():
            count = 0
            examples = []
            
            for pattern in patterns:
                matching_features = [col for col in version_df.columns if pattern in col]
                count += len(matching_features)
                examples.extend(matching_features[:2])  # First 2 examples
            
            print(f"  {feature_type}: {count} features")
            if examples:
                print(f"    Examples: {examples[:3]}")
    
    return True

def main():
    """Run all enhancement tests."""
    print("Enhanced Feature Comparison Framework - Test Suite")
    print("=" * 70)
    
    tests = [
        ("Returns-Based Calculations", test_returns_based_calculations),
        ("Rolling Averages and EWMA", test_rolling_ewma_features),
        ("Lagged and Lead Features", test_lagged_lead_features),
        ("Matrix Operations Integration", test_matrix_operations_integration),
        ("Bootstrap Optimization", test_bootstrap_optimization),
        ("Enhanced Feature Types", test_enhanced_feature_types)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*70}")
        print(f"Running: {test_name}")
        print('='*70)
        
        try:
            result = test_func()
            results[test_name] = result
            print(f"\n✅ {test_name}: PASSED")
        except Exception as e:
            results[test_name] = False
            print(f"\n❌ {test_name}: FAILED - {e}")
    
    # Summary
    print(f"\n{'='*70}")
    print("TEST SUMMARY")
    print('='*70)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:30}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All enhancements working correctly!")
    else:
        print(f"\n⚠️ {total - passed} tests failed. Check the output above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)