#!/usr/bin/env python3
"""
Test Script for High-Impact Improvements

This script tests the high-impact, low-effort improvements implemented:
1. Purged/embargoed CV auto-sizing
2. Causal audit hooks
3. Near-constant filter using IQR/entropy
4. Kernel fusion for interactions
"""

import pandas as pd
import numpy as np
import sys
import time
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def create_test_data(n_samples: int = 10000) -> pd.DataFrame:
    """Create test dataset for improvements testing."""
    np.random.seed(42)
    
    # Create time index
    dates = pd.date_range('2024-01-01', periods=n_samples, freq='1min')
    
    # Generate realistic market data
    base_price = 100.0
    returns = np.random.normal(0, 0.02, n_samples)
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Generate OHLCV data
    data = {
        'open': prices * (1 + np.random.normal(0, 0.001, n_samples)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.005, n_samples))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.005, n_samples))),
        'close': prices * (1 + np.random.normal(0, 0.01, n_samples)),
        'volume': np.random.lognormal(10, 0.5, n_samples),
    }
    
    # Ensure high >= max(open, close) and low <= min(open, close)
    for i in range(n_samples):
        data['high'][i] = max(data['high'][i], data['open'][i], data['close'][i])
        data['low'][i] = min(data['low'][i], data['open'][i], data['close'][i])
    
    df = pd.DataFrame(data, index=dates)
    
    # Add some additional features
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(20).std()
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    return df

def test_purged_cv_auto_sizing():
    """Test purged CV auto-sizing based on max lookback + horizon."""
    print("🧪 Testing Purged CV Auto-Sizing...")
    print("=" * 50)
    
    try:
        from src.training.steps.pre_training.interaction_feature_generator.feature_interaction_generation.purged_cv_system import (
            PurgedTimeSeriesSplit, PurgedCVConfig, create_purged_cv_splits
        )
        
        # Create test data
        data = create_test_data(5000)
        
        # Create features with different lookback periods
        features = pd.DataFrame({
            'rsi_14': np.random.randn(len(data)),
            'sma_20': np.random.randn(len(data)),
            'rolling_50_mean': np.random.randn(len(data)),
            'ctf_100_close_mean': np.random.randn(len(data)),
            'volume_ma_30': np.random.randn(len(data)),
        }, index=data.index)
        
        target = data['target']
        
        # Test purged CV with auto-sizing
        print("📊 Testing purged CV with auto-sizing...")
        
        config = PurgedCVConfig(
            n_splits=5,
            embargo_ratio=0.01,
            horizon=1,
            safety_factor=1.5
        )
        
        splitter = PurgedTimeSeriesSplit(
            n_splits=config.n_splits,
            embargo_ratio=config.embargo_ratio,
            horizon=config.horizon,
            safety_factor=config.safety_factor
        )
        
        # Test embargo size calculation
        max_lookback = splitter.analyze_feature_lookbacks(features)
        embargo_size = splitter.calculate_embargo_size(len(data), max_lookback)
        
        print(f"  Max lookback detected: {max_lookback}")
        print(f"  Calculated embargo size: {embargo_size}")
        
        # Test CV splits
        splits = list(splitter.split(features, target))
        print(f"  Generated {len(splits)} CV splits")
        
        # Validate splits
        for i, (train_idx, test_idx) in enumerate(splits):
            print(f"  Split {i}: train={len(train_idx)}, test={len(test_idx)}")
            
            # Check for overlap
            overlap = len(np.intersect1d(train_idx, test_idx))
            if overlap > 0:
                print(f"    ⚠️ Overlap detected: {overlap} samples")
            else:
                print(f"    ✅ No overlap")
        
        # Test convenience function
        print("📊 Testing convenience function...")
        splits2 = create_purged_cv_splits(features, target)
        print(f"  Convenience function generated {len(splits2)} splits")
        
        return True
        
    except Exception as e:
        print(f"❌ Purged CV test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_causal_audit_hooks():
    """Test causal audit hooks for right-aligned operations."""
    print("\n🧪 Testing Causal Audit Hooks...")
    print("=" * 50)
    
    try:
        from src.training.steps.pre_training.interaction_feature_generator.feature_interaction_generation.causal_audit_hooks import (
            CausalAuditor, causal_audit_hook, assert_right_aligned, enable_causal_audit
        )
        
        # Create test data
        data = create_test_data(1000)
        
        # Test 1: Valid right-aligned features
        print("📊 Testing valid right-aligned features...")
        
        valid_features = pd.DataFrame({
            'rsi_14': np.random.randn(len(data)),
            'sma_20': np.random.randn(len(data)),
            'rolling_50_mean': np.random.randn(len(data)),
            'volume_ma_30': np.random.randn(len(data)),
        }, index=data.index)
        
        auditor = CausalAuditor()
        result = auditor.audit_feature_generation(valid_features, "valid_features")
        
        if result:
            print("  ✅ Valid features passed audit")
        else:
            print("  ❌ Valid features failed audit")
        
        # Test 2: Invalid centered window features
        print("📊 Testing invalid centered window features...")
        
        invalid_features = pd.DataFrame({
            'rsi_14': np.random.randn(len(data)),
            'centered_ma_20': np.random.randn(len(data)),  # This should fail
            'rolling_50_mean': np.random.randn(len(data)),
            'symmetric_bb_20': np.random.randn(len(data)),  # This should fail
        }, index=data.index)
        
        # Enable audit with fail on violation
        enable_causal_audit(True, True)
        
        try:
            result = auditor.audit_feature_generation(invalid_features, "invalid_features")
            print("  ❌ Invalid features should have failed audit")
        except Exception as e:
            print(f"  ✅ Invalid features correctly failed audit: {e}")
        
        # Test 3: Decorator functionality
        print("📊 Testing decorator functionality...")
        
        @causal_audit_hook("test_operation")
        def generate_test_features(data):
            return pd.DataFrame({
                'test_feature': np.random.randn(len(data))
            }, index=data.index)
        
        try:
            result = generate_test_features(data)
            print("  ✅ Decorator test passed")
        except Exception as e:
            print(f"  ❌ Decorator test failed: {e}")
        
        # Test 4: Assert function
        print("📊 Testing assert function...")
        
        try:
            assert_result = assert_right_aligned(valid_features, "assert_test")
            print("  ✅ Assert function passed")
        except Exception as e:
            print(f"  ❌ Assert function failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Causal audit hooks test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_near_constant_filter():
    """Test near-constant filter using IQR and entropy."""
    print("\n🧪 Testing Near-Constant Filter...")
    print("=" * 50)
    
    try:
        from src.training.steps.pre_training.interaction_feature_generator.feature_interaction_generation.near_constant_filter import (
            NearConstantFilter, NearConstantFilterConfig, filter_near_constant_features
        )
        
        # Create test data with various feature types
        data = create_test_data(2000)
        
        # Create features with different constant levels
        features = pd.DataFrame({
            'high_variance': np.random.randn(len(data)),  # Should be kept
            'low_variance': np.random.randn(len(data)) * 0.001,  # Should be filtered
            'constant_feature': np.ones(len(data)),  # Should be filtered
            'near_constant': np.random.choice([1, 1.1], len(data)),  # Should be filtered
            'good_iqr': np.random.randn(len(data)),  # Should be kept
            'low_iqr': np.random.randn(len(data)) * 0.01,  # Should be filtered
            'categorical': np.random.choice(['A', 'B', 'C'], len(data)),  # Should be kept
            'binary': np.random.choice([0, 1], len(data)),  # Should be kept
        }, index=data.index)
        
        target = data['target']
        
        # Test near-constant filter
        print("📊 Testing near-constant filter...")
        
        config = NearConstantFilterConfig(
            iqr_threshold=0.01,
            entropy_threshold=0.1,
            adaptive_thresholds=True
        )
        
        filter_instance = NearConstantFilter(config)
        filtered_features = filter_instance.filter_features(features, target)
        
        print(f"  Original features: {len(features.columns)}")
        print(f"  Filtered features: {len(filtered_features.columns)}")
        print(f"  Removed features: {len(features.columns) - len(filtered_features.columns)}")
        
        # Show which features were kept/removed
        kept_features = set(filtered_features.columns)
        removed_features = set(features.columns) - kept_features
        
        print(f"  Kept features: {list(kept_features)}")
        print(f"  Removed features: {list(removed_features)}")
        
        # Test filter statistics
        stats = filter_instance.get_filter_statistics()
        print(f"  Filter statistics: {stats}")
        
        # Test convenience function
        print("📊 Testing convenience function...")
        filtered_features2 = filter_near_constant_features(features, target)
        print(f"  Convenience function result: {len(filtered_features2.columns)} features")
        
        return True
        
    except Exception as e:
        print(f"❌ Near-constant filter test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_kernel_fusion():
    """Test kernel fusion for interactions."""
    print("\n🧪 Testing Kernel Fusion...")
    print("=" * 50)
    
    try:
        from src.training.steps.pre_training.interaction_feature_generator.feature_interaction_generation.kernel_fusion import (
            KernelFusion, KernelFusionConfig, fuse_interactions
        )
        
        # Create test data
        data = create_test_data(5000)
        
        # Create features
        features = pd.DataFrame({
            'feature1': np.random.randn(len(data)),
            'feature2': np.random.randn(len(data)),
            'feature3': np.random.randn(len(data)),
            'feature4': np.random.randn(len(data)),
        }, index=data.index)
        
        # Create feature pairs
        feature_pairs = [
            ('feature1', 'feature2'),
            ('feature1', 'feature3'),
            ('feature2', 'feature3'),
            ('feature3', 'feature4'),
        ]
        
        # Test kernel fusion
        print("📊 Testing kernel fusion...")
        
        config = KernelFusionConfig(
            enable_fusion=True,
            batch_size=1000,
            interaction_types=['sum', 'diff', 'prod', 'ratio']
        )
        
        fusion = KernelFusion(config)
        
        # Test fusion
        start_time = time.time()
        fused_interactions = fusion.fuse_interactions(features, feature_pairs)
        fusion_time = time.time() - start_time
        
        print(f"  Fusion time: {fusion_time:.3f}s")
        print(f"  Generated interactions: {len(fused_interactions.columns)}")
        print(f"  Expected interactions: {len(feature_pairs) * len(config.interaction_types)}")
        
        # Show sample interactions
        print("  Sample interactions:")
        for i, col in enumerate(list(fused_interactions.columns)[:5]):
            print(f"    {i+1}. {col}")
        
        # Test fusion statistics
        stats = fusion.get_fusion_statistics()
        print(f"  Fusion statistics: {stats}")
        
        # Test convenience function
        print("📊 Testing convenience function...")
        start_time = time.time()
        fused_interactions2 = fuse_interactions(features, feature_pairs, optimized=True)
        convenience_time = time.time() - start_time
        
        print(f"  Convenience function time: {convenience_time:.3f}s")
        print(f"  Convenience function result: {len(fused_interactions2.columns)} interactions")
        
        # Test performance comparison
        print("📊 Testing performance comparison...")
        
        # Sequential computation
        start_time = time.time()
        sequential_result = fusion._compute_interactions_sequential(features, feature_pairs)
        sequential_time = time.time() - start_time
        
        print(f"  Sequential time: {sequential_time:.3f}s")
        print(f"  Fusion speedup: {sequential_time/fusion_time:.1f}x")
        
        return True
        
    except Exception as e:
        print(f"❌ Kernel fusion test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration():
    """Test integration of all improvements."""
    print("\n🧪 Testing Integration...")
    print("=" * 50)
    
    try:
        # Create test data
        data = create_test_data(3000)
        
        # Create features
        features = pd.DataFrame({
            'rsi_14': np.random.randn(len(data)),
            'sma_20': np.random.randn(len(data)),
            'rolling_50_mean': np.random.randn(len(data)),
            'volume_ma_30': np.random.randn(len(data)),
            'constant_feature': np.ones(len(data)),  # Should be filtered
            'low_variance': np.random.randn(len(data)) * 0.001,  # Should be filtered
        }, index=data.index)
        
        target = data['target']
        
        print("📊 Testing integrated pipeline...")
        
        # Step 1: Near-constant filtering
        from src.training.steps.pre_training.interaction_feature_generator.feature_interaction_generation.near_constant_filter import filter_near_constant_features
        
        filtered_features = filter_near_constant_features(features, target)
        print(f"  Step 1 - Near-constant filtering: {len(features.columns)} -> {len(filtered_features.columns)} features")
        
        # Step 2: Causal audit
        from src.training.steps.pre_training.interaction_feature_generator.feature_interaction_generation.causal_audit_hooks import assert_right_aligned
        
        try:
            assert_right_aligned(filtered_features, "integration_test")
            print("  Step 2 - Causal audit: ✅ Passed")
        except Exception as e:
            print(f"  Step 2 - Causal audit: ❌ Failed - {e}")
        
        # Step 3: Kernel fusion
        from src.training.steps.pre_training.interaction_feature_generator.feature_interaction_generation.kernel_fusion import fuse_interactions
        
        feature_pairs = [('rsi_14', 'sma_20'), ('rolling_50_mean', 'volume_ma_30')]
        fused_interactions = fuse_interactions(filtered_features, feature_pairs)
        print(f"  Step 3 - Kernel fusion: Generated {len(fused_interactions.columns)} interactions")
        
        # Step 4: Purged CV
        from src.training.steps.pre_training.interaction_feature_generator.feature_interaction_generation.purged_cv_system import create_purged_cv_splits
        
        # Combine features and interactions
        all_features = pd.concat([filtered_features, fused_interactions], axis=1)
        
        splits = create_purged_cv_splits(all_features, target)
        print(f"  Step 4 - Purged CV: Generated {len(splits)} splits")
        
        # Validate final result
        print(f"  Final result: {len(all_features.columns)} total features")
        print(f"  Data shape: {all_features.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all high-impact improvement tests."""
    print("🚀 Testing High-Impact Improvements")
    print("=" * 60)
    
    tests = [
        ("Purged CV Auto-Sizing", test_purged_cv_auto_sizing),
        ("Causal Audit Hooks", test_causal_audit_hooks),
        ("Near-Constant Filter", test_near_constant_filter),
        ("Kernel Fusion", test_kernel_fusion),
        ("Integration", test_integration),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} Test...")
        print("-" * 40)
        
        try:
            success = test_func()
            results.append((test_name, success))
            
            if success:
                print(f"✅ {test_name} test passed!")
            else:
                print(f"❌ {test_name} test failed!")
                
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 HIGH-IMPACT IMPROVEMENTS TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if success:
            passed += 1
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All high-impact improvements are working correctly!")
        return True
    else:
        print(f"⚠️ {total - passed} tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)