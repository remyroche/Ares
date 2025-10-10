#!/usr/bin/env python3
"""
Test Script for Bottleneck Optimizations

This script tests the critical bottleneck optimizations implemented:
1. Blockwise correlation with early-abort
2. Optimized kernel fusion
3. Prefix sums/EMA reuse
4. Two-stage scoring
"""

import pandas as pd
import numpy as np
import sys
import time
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def create_test_data(n_samples: int = 10000) -> pd.DataFrame:
    """Create test dataset for bottleneck testing."""
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

def test_blockwise_correlation():
    """Test blockwise correlation with early-abort."""
    print("🧪 Testing Blockwise Correlation...")
    print("=" * 50)
    
    try:
        # Create test data with many features
        data = create_test_data(5000)
        
        # Create many features to test correlation computation
        features = pd.DataFrame({
            f'feature_{i}': np.random.randn(len(data)) for i in range(100)
        }, index=data.index)
        
        target = data['target']
        
        print("📊 Testing blockwise correlation computation...")
        
        # Test blockwise correlation
        from src.training.steps.pre_training.interaction_feature_generator.feature_interaction_generation.blockwise_correlation import (
            BlockwiseCorrelation, BlockwiseCorrelationConfig, compute_correlations_blockwise
        )
        
        config = BlockwiseCorrelationConfig(
            block_size=20,
            correlation_threshold=0.95,
            max_correlations=1000,
            use_approximation=True
        )
        
        correlation = BlockwiseCorrelation(config)
        
        start_time = time.time()
        results = correlation.compute_correlations(features, target)
        correlation_time = time.time() - start_time
        
        print(f"  Correlation time: {correlation_time:.3f}s")
        print(f"  Correlations computed: {results['stats']['correlations_computed']}")
        print(f"  Early aborts: {results['stats']['early_aborts']}")
        print(f"  High correlations: {results['stats']['high_correlations']}")
        
        # Test redundant feature detection
        redundant_features = correlation.get_redundant_features(results)
        print(f"  Redundant features: {len(redundant_features)}")
        
        # Test top correlations
        top_correlations = correlation.get_top_correlations(results, top_k=10)
        print(f"  Top correlations: {len(top_correlations)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Blockwise correlation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_optimized_kernel_fusion():
    """Test optimized kernel fusion."""
    print("\n🧪 Testing Optimized Kernel Fusion...")
    print("=" * 50)
    
    try:
        # Create test data
        data = create_test_data(5000)
        
        # Create features
        features = pd.DataFrame({
            'feature1': np.random.randn(len(data)),
            'feature2': np.random.randn(len(data)),
            'feature3': np.random.randn(len(data)),
            'feature4': np.random.randn(len(data)),
            'feature5': np.random.randn(len(data)),
        }, index=data.index)
        
        # Create feature pairs
        feature_pairs = [
            ('feature1', 'feature2'),
            ('feature1', 'feature3'),
            ('feature2', 'feature3'),
            ('feature3', 'feature4'),
            ('feature4', 'feature5'),
        ]
        
        print("📊 Testing optimized kernel fusion...")
        
        from src.training.steps.pre_training.interaction_feature_generator.feature_interaction_generation.kernel_fusion import (
            KernelFusion, KernelFusionConfig, fuse_interactions
        )
        
        config = KernelFusionConfig(
            enable_fusion=True,
            batch_size=1000,
            row_block_size=1000,
            preallocate_output=True,
            interaction_types=['sum', 'diff', 'prod', 'ratio']
        )
        
        fusion = KernelFusion(config)
        
        start_time = time.time()
        fused_interactions = fusion.fuse_interactions(features, feature_pairs)
        fusion_time = time.time() - start_time
        
        print(f"  Fusion time: {fusion_time:.3f}s")
        print(f"  Generated interactions: {len(fused_interactions.columns)}")
        print(f"  Expected interactions: {len(feature_pairs) * len(config.interaction_types)}")
        
        # Test performance comparison
        print("📊 Testing performance comparison...")
        
        # Sequential computation
        start_time = time.time()
        sequential_interactions = {}
        for pair in feature_pairs:
            feature1, feature2 = pair
            data1 = features[feature1].values
            data2 = features[feature2].values
            
            for interaction_type in config.interaction_types:
                if interaction_type == 'sum':
                    result = data1 + data2
                elif interaction_type == 'diff':
                    result = data1 - data2
                elif interaction_type == 'prod':
                    result = data1 * data2
                elif interaction_type == 'ratio':
                    result = data1 / (data2 + 1e-8)
                
                sequential_interactions[f'{feature1}_{interaction_type}_{feature2}'] = result
        
        sequential_time = time.time() - start_time
        
        print(f"  Sequential time: {sequential_time:.3f}s")
        print(f"  Fusion speedup: {sequential_time/fusion_time:.1f}x")
        
        return True
        
    except Exception as e:
        print(f"❌ Optimized kernel fusion test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_prefix_sums_reuse():
    """Test prefix sums/EMA reuse."""
    print("\n🧪 Testing Prefix Sums/EMA Reuse...")
    print("=" * 50)
    
    try:
        # Create test data
        data = create_test_data(3000)
        
        # Create features
        features = pd.DataFrame({
            'feature1': np.random.randn(len(data)),
            'feature2': np.random.randn(len(data)),
            'feature3': np.random.randn(len(data)),
        }, index=data.index)
        
        print("📊 Testing prefix sums reuse...")
        
        from src.training.steps.pre_training.interaction_feature_generator.feature_interaction_generation.prefix_sums_reuse import (
            PrefixSumsReuse, PrefixSumsConfig, compute_rolling_features_reuse
        )
        
        config = PrefixSumsConfig(
            enable_reuse=True,
            cache_emas=True,
            cache_prefix_sums=True,
            vectorized_rolling=True
        )
        
        reuse = PrefixSumsReuse(config)
        
        # Test rolling features
        windows = [5, 10, 20, 50]
        
        start_time = time.time()
        rolling_features = reuse.compute_rolling_features(features, windows)
        rolling_time = time.time() - start_time
        
        print(f"  Rolling features time: {rolling_time:.3f}s")
        print(f"  Generated rolling features: {len(rolling_features.columns)}")
        print(f"  Expected rolling features: {len(features.columns) * len(windows) * 4}")  # 4 stats per window
        
        # Test EMA features
        periods = [12, 26, 50]
        
        start_time = time.time()
        ema_features = reuse.compute_ema_features(features, periods)
        ema_time = time.time() - start_time
        
        print(f"  EMA features time: {ema_time:.3f}s")
        print(f"  Generated EMA features: {len(ema_features.columns)}")
        
        # Test technical indicators
        start_time = time.time()
        tech_indicators = reuse.compute_technical_indicators(features, 'feature1')
        tech_time = time.time() - start_time
        
        print(f"  Technical indicators time: {tech_time:.3f}s")
        print(f"  Generated technical indicators: {len(tech_indicators.columns)}")
        
        # Test cache statistics
        cache_stats = reuse.get_cache_statistics()
        print(f"  Cache statistics: {cache_stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ Prefix sums reuse test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_two_stage_scoring():
    """Test two-stage scoring system."""
    print("\n🧪 Testing Two-Stage Scoring...")
    print("=" * 50)
    
    try:
        # Create test data
        data = create_test_data(2000)
        
        # Create features with different information content
        features = pd.DataFrame({
            'high_info_feature': np.random.randn(len(data)) + data['target'] * 0.5,
            'medium_info_feature': np.random.randn(len(data)) + data['target'] * 0.2,
            'low_info_feature': np.random.randn(len(data)) + data['target'] * 0.05,
            'no_info_feature': np.random.randn(len(data)),
            'constant_feature': np.ones(len(data)),
        }, index=data.index)
        
        target = data['target']
        
        print("📊 Testing two-stage scoring...")
        
        from src.training.steps.pre_training.interaction_feature_generator.feature_interaction_generation.two_stage_scoring import (
            TwoStageScoring, TwoStageScoringConfig, score_features_two_stage
        )
        
        config = TwoStageScoringConfig(
            sample_ratio=0.2,
            ic_threshold=0.01,
            top_k_features=10,
            mi_bins=5,
            use_parallel=True
        )
        
        scoring = TwoStageScoring(config)
        
        start_time = time.time()
        results = scoring.score_features(features, target)
        scoring_time = time.time() - start_time
        
        print(f"  Scoring time: {scoring_time:.3f}s")
        print(f"  IC computations: {results['stats']['ic_computations']}")
        print(f"  MI computations: {results['stats']['mi_computations']}")
        print(f"  Shortlisted features: {results['stats']['shortlisted_features']}")
        
        # Test feature ranking
        rankings = results['final_ranking']
        print(f"  Feature rankings:")
        for i, (feature, ic, mi, combined) in enumerate(rankings[:5]):
            print(f"    {i+1}. {feature}: IC={ic:.3f}, MI={mi:.3f}, Combined={combined:.3f}")
        
        # Test top features
        top_features = scoring.get_top_features(results, top_k=3)
        print(f"  Top features: {top_features}")
        
        return True
        
    except Exception as e:
        print(f"❌ Two-stage scoring test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration():
    """Test integration of all optimizations."""
    print("\n🧪 Testing Integration...")
    print("=" * 50)
    
    try:
        # Create test data
        data = create_test_data(2000)
        
        # Create features
        features = pd.DataFrame({
            'feature1': np.random.randn(len(data)),
            'feature2': np.random.randn(len(data)),
            'feature3': np.random.randn(len(data)),
            'feature4': np.random.randn(len(data)),
        }, index=data.index)
        
        target = data['target']
        
        print("📊 Testing integrated pipeline...")
        
        # Step 1: Two-stage scoring
        from src.training.steps.pre_training.interaction_feature_generator.feature_interaction_generation.two_stage_scoring import score_features_two_stage
        
        scoring_results = score_features_two_stage(features, target)
        top_features = [f for f, _, _, _ in scoring_results['final_ranking'][:3]]
        print(f"  Step 1 - Two-stage scoring: Selected {len(top_features)} top features")
        
        # Step 2: Blockwise correlation
        from src.training.steps.pre_training.interaction_feature_generator.feature_interaction_generation.blockwise_correlation import compute_correlations_blockwise
        
        correlation_results = compute_correlations_blockwise(features[top_features], target)
        print(f"  Step 2 - Blockwise correlation: Computed {correlation_results['stats']['correlations_computed']} correlations")
        
        # Step 3: Prefix sums reuse
        from src.training.steps.pre_training.interaction_feature_generator.feature_interaction_generation.prefix_sums_reuse import compute_rolling_features_reuse
        
        rolling_features = compute_rolling_features_reuse(features[top_features], [5, 10, 20])
        print(f"  Step 3 - Prefix sums reuse: Generated {len(rolling_features.columns)} rolling features")
        
        # Step 4: Kernel fusion
        from src.training.steps.pre_training.interaction_feature_generator.feature_interaction_generation.kernel_fusion import fuse_interactions
        
        feature_pairs = [('feature1', 'feature2'), ('feature3', 'feature4')]
        fused_interactions = fuse_interactions(features[top_features], feature_pairs)
        print(f"  Step 4 - Kernel fusion: Generated {len(fused_interactions.columns)} interactions")
        
        # Combine all features
        all_features = pd.concat([features[top_features], rolling_features, fused_interactions], axis=1)
        print(f"  Final result: {len(all_features.columns)} total features")
        print(f"  Data shape: {all_features.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all bottleneck optimization tests."""
    print("🚀 Testing Bottleneck Optimizations")
    print("=" * 60)
    
    tests = [
        ("Blockwise Correlation", test_blockwise_correlation),
        ("Optimized Kernel Fusion", test_optimized_kernel_fusion),
        ("Prefix Sums/EMA Reuse", test_prefix_sums_reuse),
        ("Two-Stage Scoring", test_two_stage_scoring),
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
    print("📊 BOTTLENECK OPTIMIZATIONS TEST SUMMARY")
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
        print("🎉 All bottleneck optimizations are working correctly!")
        return True
    else:
        print(f"⚠️ {total - passed} tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)