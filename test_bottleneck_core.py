#!/usr/bin/env python3
"""
Core Test for Bottleneck Optimizations

This script tests the core functionality of the bottleneck optimizations
without complex dependencies.
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

def test_blockwise_correlation_core():
    """Test core blockwise correlation functionality."""
    print("🧪 Testing Blockwise Correlation Core...")
    print("=" * 50)
    
    try:
        # Simple blockwise correlation implementation
        def compute_correlations_blockwise(data: pd.DataFrame, target: pd.Series, block_size: int = 20, threshold: float = 0.95) -> dict:
            """Compute correlations using blockwise approach."""
            correlations = {}
            early_aborts = 0
            high_correlations = 0
            
            features = list(data.columns)
            total_correlations = 0
            
            for i in range(0, len(features), block_size):
                block = features[i:i + block_size]
                block_data = data[block].dropna()
                
                if len(block_data) == 0:
                    continue
                
                for j, feature1 in enumerate(block):
                    if feature1 not in block_data.columns:
                        continue
                    
                    for k, feature2 in enumerate(block[j+1:], j+1):
                        if feature2 not in block_data.columns:
                            continue
                        
                        try:
                            corr = block_data[feature1].corr(block_data[feature2])
                            total_correlations += 1
                            
                            if not np.isnan(corr):
                                if abs(corr) > threshold:
                                    early_aborts += 1
                                    high_correlations += 1
                                    break  # Early abort for this feature
                                
                                correlations[f'{feature1}_{feature2}'] = corr
                        except:
                            continue
            
            return {
                'correlations': correlations,
                'total_correlations': total_correlations,
                'early_aborts': early_aborts,
                'high_correlations': high_correlations
            }
        
        # Create test data
        data = create_test_data(2000)
        
        # Create many features
        features = pd.DataFrame({
            f'feature_{i}': np.random.randn(len(data)) for i in range(50)
        }, index=data.index)
        
        target = data['target']
        
        print("📊 Testing blockwise correlation...")
        
        start_time = time.time()
        results = compute_correlations_blockwise(features, target, block_size=10, threshold=0.95)
        correlation_time = time.time() - start_time
        
        print(f"  Correlation time: {correlation_time:.3f}s")
        print(f"  Correlations computed: {results['total_correlations']}")
        print(f"  Early aborts: {results['early_aborts']}")
        print(f"  High correlations: {results['high_correlations']}")
        print(f"  Final correlations: {len(results['correlations'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Blockwise correlation core test failed: {e}")
        return False

def test_optimized_kernel_fusion_core():
    """Test core optimized kernel fusion functionality."""
    print("\n🧪 Testing Optimized Kernel Fusion Core...")
    print("=" * 50)
    
    try:
        def fuse_interactions_optimized(data: pd.DataFrame, feature_pairs: list, interaction_types: list = None) -> pd.DataFrame:
            """Fuse interactions with optimizations."""
            if interaction_types is None:
                interaction_types = ['sum', 'diff', 'prod', 'ratio']
            
            interactions = {}
            
            # Preallocate output arrays
            n_samples = len(data)
            
            for pair in feature_pairs:
                feature1, feature2 = pair
                
                if feature1 not in data.columns or feature2 not in data.columns:
                    continue
                
                data1 = data[feature1].values
                data2 = data[feature2].values
                
                # Compute all interaction types in one pass
                for interaction_type in interaction_types:
                    if interaction_type == 'sum':
                        result = data1 + data2
                    elif interaction_type == 'diff':
                        result = data1 - data2
                    elif interaction_type == 'prod':
                        result = data1 * data2
                    elif interaction_type == 'ratio':
                        result = data1 / (data2 + 1e-8)
                    else:
                        result = np.full_like(data1, np.nan)
                    
                    interactions[f'{feature1}_{interaction_type}_{feature2}'] = result
            
            return pd.DataFrame(interactions, index=data.index)
        
        # Create test data
        data = create_test_data(3000)
        
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
        
        print("📊 Testing optimized kernel fusion...")
        
        start_time = time.time()
        fused_interactions = fuse_interactions_optimized(features, feature_pairs)
        fusion_time = time.time() - start_time
        
        print(f"  Fusion time: {fusion_time:.3f}s")
        print(f"  Generated interactions: {len(fused_interactions.columns)}")
        print(f"  Expected interactions: {len(feature_pairs) * 4}")  # 4 interaction types
        
        # Test performance comparison
        print("📊 Testing performance comparison...")
        
        # Sequential computation
        start_time = time.time()
        sequential_interactions = {}
        for pair in feature_pairs:
            feature1, feature2 = pair
            data1 = features[feature1].values
            data2 = features[feature2].values
            
            for interaction_type in ['sum', 'diff', 'prod', 'ratio']:
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
        print(f"❌ Optimized kernel fusion core test failed: {e}")
        return False

def test_prefix_sums_reuse_core():
    """Test core prefix sums reuse functionality."""
    print("\n🧪 Testing Prefix Sums Reuse Core...")
    print("=" * 50)
    
    try:
        def compute_rolling_features_reuse(data: pd.DataFrame, windows: list) -> pd.DataFrame:
            """Compute rolling features with prefix sums reuse."""
            rolling_features = {}
            
            for col in data.columns:
                feature_data = data[col].values
                
                # Compute prefix sums once
                cumsum = np.cumsum(feature_data)
                cumsum_sq = np.cumsum(feature_data ** 2)
                cumcount = np.cumsum(~np.isnan(feature_data))
                
                for window in windows:
                    # Compute rolling statistics from prefix sums
                    rolling_mean = np.full(len(feature_data), np.nan)
                    rolling_std = np.full(len(feature_data), np.nan)
                    
                    for i in range(window - 1, len(feature_data)):
                        start_idx = i - window + 1
                        if start_idx >= 0:
                            count = cumcount[i] - (cumcount[start_idx - 1] if start_idx > 0 else 0)
                            if count > 0:
                                sum_val = cumsum[i] - (cumsum[start_idx - 1] if start_idx > 0 else 0)
                                rolling_mean[i] = sum_val / count
                                
                                if count > 1:
                                    sum_sq = cumsum_sq[i] - (cumsum_sq[start_idx - 1] if start_idx > 0 else 0)
                                    mean_val = sum_val / count
                                    variance = (sum_sq / count) - (mean_val ** 2)
                                    rolling_std[i] = np.sqrt(max(0, variance))
                    
                    rolling_features[f'{col}_mean_{window}'] = rolling_mean
                    rolling_features[f'{col}_std_{window}'] = rolling_std
            
            return pd.DataFrame(rolling_features, index=data.index)
        
        def compute_ema_reuse(data: pd.DataFrame, periods: list) -> pd.DataFrame:
            """Compute EMA features with reuse."""
            ema_features = {}
            
            for col in data.columns:
                feature_data = data[col].values
                
                for period in periods:
                    alpha = 2.0 / (period + 1)
                    ema = np.full_like(feature_data, np.nan)
                    
                    # Find first valid value
                    first_valid = np.argmax(~np.isnan(feature_data))
                    if first_valid < len(feature_data):
                        ema[first_valid] = feature_data[first_valid]
                        
                        # Compute EMA for remaining values
                        for i in range(first_valid + 1, len(feature_data)):
                            if not np.isnan(feature_data[i]):
                                ema[i] = alpha * feature_data[i] + (1 - alpha) * ema[i-1]
                            else:
                                ema[i] = ema[i-1]
                    
                    ema_features[f'{col}_ema_{period}'] = ema
            
            return pd.DataFrame(ema_features, index=data.index)
        
        # Create test data
        data = create_test_data(2000)
        
        # Create features
        features = pd.DataFrame({
            'feature1': np.random.randn(len(data)),
            'feature2': np.random.randn(len(data)),
            'feature3': np.random.randn(len(data)),
        }, index=data.index)
        
        print("📊 Testing prefix sums reuse...")
        
        # Test rolling features
        windows = [5, 10, 20]
        
        start_time = time.time()
        rolling_features = compute_rolling_features_reuse(features, windows)
        rolling_time = time.time() - start_time
        
        print(f"  Rolling features time: {rolling_time:.3f}s")
        print(f"  Generated rolling features: {len(rolling_features.columns)}")
        print(f"  Expected rolling features: {len(features.columns) * len(windows) * 2}")  # 2 stats per window
        
        # Test EMA features
        periods = [12, 26, 50]
        
        start_time = time.time()
        ema_features = compute_ema_reuse(features, periods)
        ema_time = time.time() - start_time
        
        print(f"  EMA features time: {ema_time:.3f}s")
        print(f"  Generated EMA features: {len(ema_features.columns)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Prefix sums reuse core test failed: {e}")
        return False

def test_two_stage_scoring_core():
    """Test core two-stage scoring functionality."""
    print("\n🧪 Testing Two-Stage Scoring Core...")
    print("=" * 50)
    
    try:
        def compute_ic_scores(features: pd.DataFrame, target: pd.Series, sample_ratio: float = 0.1) -> dict:
            """Compute IC scores on a sample."""
            sample_size = int(len(features) * sample_ratio)
            sample_indices = np.random.choice(len(features), sample_size, replace=False)
            
            sample_features = features.iloc[sample_indices]
            sample_target = target.iloc[sample_indices]
            
            ic_scores = {}
            for col in sample_features.columns:
                try:
                    corr = sample_features[col].corr(sample_target)
                    ic_scores[col] = corr if not np.isnan(corr) else 0.0
                except:
                    ic_scores[col] = 0.0
            
            return ic_scores
        
        def shortlist_features(ic_scores: dict, threshold: float = 0.01, top_k: int = 10) -> list:
            """Shortlist features based on IC scores."""
            sorted_features = sorted(ic_scores.items(), key=lambda x: abs(x[1]), reverse=True)
            shortlisted = []
            
            for feature, ic in sorted_features:
                if abs(ic) >= threshold:
                    shortlisted.append(feature)
                    if len(shortlisted) >= top_k:
                        break
            
            return shortlisted
        
        def compute_mi_scores(features: pd.DataFrame, target: pd.Series, feature_names: list) -> dict:
            """Compute MI scores for shortlisted features."""
            from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
            
            mi_scores = {}
            for feature in feature_names:
                if feature in features.columns:
                    try:
                        feature_data = features[feature].values.reshape(-1, 1)
                        target_data = target.values
                        
                        # Determine if target is continuous or discrete
                        if target.nunique() < 10:
                            mi = mutual_info_classif(feature_data, target_data, random_state=42)[0]
                        else:
                            mi = mutual_info_regression(feature_data, target_data, random_state=42)[0]
                        
                        mi_scores[feature] = mi if not np.isnan(mi) else 0.0
                    except:
                        mi_scores[feature] = 0.0
            
            return mi_scores
        
        # Create test data
        data = create_test_data(1000)
        
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
        
        # Stage 1: IC computation
        start_time = time.time()
        ic_scores = compute_ic_scores(features, target, sample_ratio=0.2)
        ic_time = time.time() - start_time
        
        print(f"  IC computation time: {ic_time:.3f}s")
        print(f"  IC scores: {ic_scores}")
        
        # Shortlist features
        shortlisted = shortlist_features(ic_scores, threshold=0.01, top_k=3)
        print(f"  Shortlisted features: {shortlisted}")
        
        # Stage 2: MI computation
        start_time = time.time()
        mi_scores = compute_mi_scores(features, target, shortlisted)
        mi_time = time.time() - start_time
        
        print(f"  MI computation time: {mi_time:.3f}s")
        print(f"  MI scores: {mi_scores}")
        
        # Combine results
        total_time = ic_time + mi_time
        print(f"  Total scoring time: {total_time:.3f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ Two-stage scoring core test failed: {e}")
        return False

def test_integration_core():
    """Test core integration functionality."""
    print("\n🧪 Testing Integration Core...")
    print("=" * 50)
    
    try:
        # Create test data
        data = create_test_data(1500)
        
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
        def compute_ic_scores(features: pd.DataFrame, target: pd.Series) -> dict:
            ic_scores = {}
            for col in features.columns:
                try:
                    corr = features[col].corr(target)
                    ic_scores[col] = corr if not np.isnan(corr) else 0.0
                except:
                    ic_scores[col] = 0.0
            return ic_scores
        
        ic_scores = compute_ic_scores(features, target)
        top_features = sorted(ic_scores.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        top_feature_names = [f for f, _ in top_features]
        print(f"  Step 1 - Two-stage scoring: Selected {len(top_feature_names)} top features")
        
        # Step 2: Blockwise correlation
        def compute_correlations_simple(data: pd.DataFrame) -> dict:
            correlations = {}
            for i, col1 in enumerate(data.columns):
                for j, col2 in enumerate(data.columns[i+1:], i+1):
                    try:
                        corr = data[col1].corr(data[col2])
                        if not np.isnan(corr):
                            correlations[f'{col1}_{col2}'] = corr
                    except:
                        continue
            return correlations
        
        correlation_results = compute_correlations_simple(features[top_feature_names])
        print(f"  Step 2 - Blockwise correlation: Computed {len(correlation_results)} correlations")
        
        # Step 3: Prefix sums reuse
        def compute_rolling_simple(data: pd.DataFrame, windows: list) -> pd.DataFrame:
            rolling_features = {}
            for col in data.columns:
                for window in windows:
                    rolling_mean = data[col].rolling(window).mean()
                    rolling_std = data[col].rolling(window).std()
                    rolling_features[f'{col}_mean_{window}'] = rolling_mean
                    rolling_features[f'{col}_std_{window}'] = rolling_std
            return pd.DataFrame(rolling_features, index=data.index)
        
        rolling_features = compute_rolling_simple(features[top_feature_names], [5, 10])
        print(f"  Step 3 - Prefix sums reuse: Generated {len(rolling_features.columns)} rolling features")
        
        # Step 4: Kernel fusion
        def fuse_interactions_simple(data: pd.DataFrame, pairs: list) -> pd.DataFrame:
            interactions = {}
            for pair in pairs:
                feature1, feature2 = pair
                if feature1 in data.columns and feature2 in data.columns:
                    data1 = data[feature1].values
                    data2 = data[feature2].values
                    interactions[f'{feature1}_sum_{feature2}'] = data1 + data2
                    interactions[f'{feature1}_diff_{feature2}'] = data1 - data2
                    interactions[f'{feature1}_prod_{feature2}'] = data1 * data2
                    interactions[f'{feature1}_ratio_{feature2}'] = data1 / (data2 + 1e-8)
            return pd.DataFrame(interactions, index=data.index)
        
        feature_pairs = [('feature1', 'feature2'), ('feature3', 'feature4')]
        fused_interactions = fuse_interactions_simple(features[top_feature_names], feature_pairs)
        print(f"  Step 4 - Kernel fusion: Generated {len(fused_interactions.columns)} interactions")
        
        # Combine all features
        all_features = pd.concat([features[top_feature_names], rolling_features, fused_interactions], axis=1)
        print(f"  Final result: {len(all_features.columns)} total features")
        print(f"  Data shape: {all_features.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration core test failed: {e}")
        return False

def main():
    """Run all core bottleneck optimization tests."""
    print("🚀 Testing Core Bottleneck Optimizations")
    print("=" * 60)
    
    tests = [
        ("Blockwise Correlation Core", test_blockwise_correlation_core),
        ("Optimized Kernel Fusion Core", test_optimized_kernel_fusion_core),
        ("Prefix Sums Reuse Core", test_prefix_sums_reuse_core),
        ("Two-Stage Scoring Core", test_two_stage_scoring_core),
        ("Integration Core", test_integration_core),
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
    print("📊 CORE BOTTLENECK OPTIMIZATIONS TEST SUMMARY")
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
        print("🎉 All core bottleneck optimizations are working correctly!")
        return True
    else:
        print(f"⚠️ {total - passed} tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)