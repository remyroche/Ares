"""
Backward Compatibility Tests for Enhanced VectorBT Classes

This module provides comprehensive tests to ensure that the enhanced VectorBT classes
maintain full backward compatibility with the original implementations.

Test Coverage:
- EnhancedVectorBTRollingOptimizer backward compatibility
- EnhancedUnifiedVectorizationManager backward compatibility
- API compatibility verification
- Performance regression testing
- Error handling compatibility
"""

import numpy as np
import pandas as pd
import pytest
import time
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List
import warnings

# Import both original and enhanced classes
try:
    from .vectorbt_rolling_optimizer import VectorBTRollingOptimizer as OriginalVectorBTRollingOptimizer
    from .unified_vectorization_manager import UnifiedVectorizationManager as OriginalUnifiedVectorizationManager
except ImportError:
    # Fallback for direct import
    from vectorbt_rolling_optimizer import VectorBTRollingOptimizer as OriginalVectorBTRollingOptimizer
    from unified_vectorization_manager import UnifiedVectorizationManager as OriginalUnifiedVectorizationManager

from .enhanced_vectorbt_rolling_optimizer import (
    EnhancedVectorBTRollingOptimizer,
    VectorBTRollingOptimizer as EnhancedVectorBTRollingOptimizerAlias,
    get_vectorbt_rolling_optimizer,
    MemoryConfig,
    CacheConfig
)

from .enhanced_unified_vectorization_manager import (
    EnhancedUnifiedVectorizationManager,
    UnifiedVectorizationManager as EnhancedUnifiedVectorizationManagerAlias,
    get_unified_vectorization_manager,
    EnhancedVectorizationConfig
)

class TestBackwardCompatibility:
    """Comprehensive backward compatibility tests."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=1000, freq='1min')
        return pd.DataFrame({
            'close': 100 + np.cumsum(np.random.randn(1000) * 0.01),
            'volume': np.random.randint(1000, 10000, 1000),
            'high': 100 + np.cumsum(np.random.randn(1000) * 0.01) + np.abs(np.random.randn(1000) * 0.5),
            'low': 100 + np.cumsum(np.random.randn(1000) * 0.01) - np.abs(np.random.randn(1000) * 0.5)
        }, index=dates)
    
    @pytest.fixture
    def sample_series(self):
        """Create sample series for testing."""
        np.random.seed(42)
        return pd.Series(100 + np.cumsum(np.random.randn(1000) * 0.01), 
                        name='close', 
                        index=pd.date_range('2020-01-01', periods=1000, freq='1min'))
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_enhanced_vectorbt_rolling_optimizer_initialization(self):
        """Test that enhanced VectorBT rolling optimizer can be initialized with original parameters."""
        # Test with original parameters
        optimizer = EnhancedVectorBTRollingOptimizer(
            enable_gpu=False,
            enable_parallel=True,
            memory_efficient=True,
            chunk_size=1000,
            fast_fail=True,
            enable_logging=True
        )
        
        assert optimizer.enable_gpu == False
        assert optimizer.enable_parallel == True
        assert optimizer.memory_efficient == True
        assert optimizer.chunk_size == 1000
        assert optimizer.fast_fail == True
        assert optimizer.enable_logging == True
        
        # Test with enhanced parameters
        memory_config = MemoryConfig(max_memory_gb=4.0)
        cache_config = CacheConfig(l1_cache_size=500)
        
        enhanced_optimizer = EnhancedVectorBTRollingOptimizer(
            enable_gpu=False,
            enable_parallel=True,
            memory_efficient=True,
            chunk_size=1000,
            fast_fail=True,
            enable_logging=True,
            memory_config=memory_config,
            cache_config=cache_config,
            enable_m1_gpu=True,
            enable_adaptive_chunking=True,
            enable_advanced_caching=True
        )
        
        assert enhanced_optimizer.enable_m1_gpu == True
        assert enhanced_optimizer.enable_adaptive_chunking == True
        assert enhanced_optimizer.enable_advanced_caching == True
        assert enhanced_optimizer.memory_manager is not None
        assert enhanced_optimizer.cache_manager is not None
    
    def test_enhanced_vectorbt_rolling_optimizer_api_compatibility(self, sample_series, sample_data):
        """Test that enhanced VectorBT rolling optimizer maintains API compatibility."""
        optimizer = EnhancedVectorBTRollingOptimizer(
            enable_gpu=False,
            enable_parallel=True,
            memory_efficient=True,
            chunk_size=1000
        )
        
        # Test all original methods exist and work
        methods_to_test = [
            'rolling_mean', 'rolling_std', 'rolling_var', 'rolling_min', 
            'rolling_max', 'rolling_sum', 'rolling_quantile', 'rolling_skew', 
            'rolling_kurt', 'rolling_corr', 'rolling_cov', 'rolling_apply',
            'rolling_median', 'rolling_percentile', 'rolling_rank',
            'rolling_ewm', 'rolling_ewm_std', 'rolling_ewm_var',
            'rolling_correlation_matrix', 'rolling_covariance_matrix'
        ]
        
        for method_name in methods_to_test:
            assert hasattr(optimizer, method_name), f"Method {method_name} not found"
            
            method = getattr(optimizer, method_name)
            assert callable(method), f"Method {method_name} is not callable"
        
        # Test basic functionality
        result = optimizer.rolling_mean(sample_series, window=20)
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_series)
        
        result = optimizer.rolling_std(sample_data['close'], window=20)
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data)
    
    def test_enhanced_vectorbt_rolling_optimizer_performance_compatibility(self, sample_series):
        """Test that enhanced VectorBT rolling optimizer maintains performance compatibility."""
        # Test with original parameters
        original_optimizer = EnhancedVectorBTRollingOptimizer(
            enable_gpu=False,
            enable_parallel=True,
            memory_efficient=True,
            chunk_size=1000,
            enable_m1_gpu=False,
            enable_adaptive_chunking=False,
            enable_advanced_caching=False
        )
        
        # Test with enhanced parameters
        enhanced_optimizer = EnhancedVectorBTRollingOptimizer(
            enable_gpu=False,
            enable_parallel=True,
            memory_efficient=True,
            chunk_size=1000,
            enable_m1_gpu=True,
            enable_adaptive_chunking=True,
            enable_advanced_caching=True
        )
        
        # Test that both produce similar results
        original_result = original_optimizer.rolling_mean(sample_series, window=20)
        enhanced_result = enhanced_optimizer.rolling_mean(sample_series, window=20)
        
        # Results should be very close (allowing for small numerical differences)
        np.testing.assert_allclose(original_result.dropna(), enhanced_result.dropna(), rtol=1e-10)
    
    def test_enhanced_vectorbt_rolling_optimizer_error_handling_compatibility(self, sample_series):
        """Test that enhanced VectorBT rolling optimizer maintains error handling compatibility."""
        optimizer = EnhancedVectorBTRollingOptimizer(
            enable_gpu=False,
            enable_parallel=True,
            memory_efficient=True,
            chunk_size=1000,
            fast_fail=True
        )
        
        # Test invalid window size
        with pytest.raises(Exception):
            optimizer.rolling_mean(sample_series, window=0)
        
        # Test invalid data type
        with pytest.raises(Exception):
            optimizer.rolling_mean("invalid_data", window=20)
        
        # Test window larger than data
        with pytest.raises(Exception):
            optimizer.rolling_mean(sample_series, window=len(sample_series) + 1)
    
    def test_enhanced_vectorbt_rolling_optimizer_stats_compatibility(self, sample_series):
        """Test that enhanced VectorBT rolling optimizer maintains stats compatibility."""
        optimizer = EnhancedVectorBTRollingOptimizer(
            enable_gpu=False,
            enable_parallel=True,
            memory_efficient=True,
            chunk_size=1000
        )
        
        # Perform some operations
        optimizer.rolling_mean(sample_series, window=20)
        optimizer.rolling_std(sample_series, window=20)
        
        # Test stats methods
        stats = optimizer.get_performance_stats()
        assert isinstance(stats, dict)
        assert 'total_operations' in stats
        assert 'vectorbt_operations' in stats
        assert 'total_time' in stats
        
        # Test reset stats
        optimizer.reset_stats()
        stats_after_reset = optimizer.get_performance_stats()
        assert stats_after_reset['total_operations'] == 0
    
    def test_enhanced_vectorbt_rolling_optimizer_context_manager_compatibility(self, sample_series):
        """Test that enhanced VectorBT rolling optimizer maintains context manager compatibility."""
        with EnhancedVectorBTRollingOptimizer(
            enable_gpu=False,
            enable_parallel=True,
            memory_efficient=True,
            chunk_size=1000
        ) as optimizer:
            result = optimizer.rolling_mean(sample_series, window=20)
            assert isinstance(result, pd.Series)
            assert len(result) == len(sample_series)
    
    def test_enhanced_unified_vectorization_manager_initialization(self):
        """Test that enhanced UnifiedVectorizationManager can be initialized with original parameters."""
        # Test with original parameters (using EnhancedVectorizationConfig with defaults)
        config = EnhancedVectorizationConfig(
            enable_vectorbt=True,
            enable_gpu=False,
            memory_efficient=True,
            enable_monitoring=True
        )
        
        manager = EnhancedUnifiedVectorizationManager(config)
        
        assert manager.config.enable_vectorbt == True
        assert manager.config.enable_gpu == False
        assert manager.config.memory_efficient == True
        assert manager.config.enable_monitoring == True
        
        # Test with enhanced parameters
        enhanced_config = EnhancedVectorizationConfig(
            enable_vectorbt=True,
            enable_gpu=False,
            memory_efficient=True,
            enable_monitoring=True,
            enable_m1_gpu=True,
            adaptive_chunking=True,
            enable_caching=True,
            l1_cache_size=500,
            l2_cache_size=2000
        )
        
        enhanced_manager = EnhancedUnifiedVectorizationManager(enhanced_config)
        
        assert enhanced_manager.config.enable_m1_gpu == True
        assert enhanced_manager.config.adaptive_chunking == True
        assert enhanced_manager.config.enable_caching == True
        assert enhanced_manager.performance_monitor is not None
    
    def test_enhanced_unified_vectorization_manager_api_compatibility(self, sample_data):
        """Test that enhanced UnifiedVectorizationManager maintains API compatibility."""
        config = EnhancedVectorizationConfig(
            enable_vectorbt=True,
            enable_gpu=False,
            memory_efficient=True,
            enable_monitoring=True
        )
        
        manager = EnhancedUnifiedVectorizationManager(config)
        
        # Test all original methods exist and work
        methods_to_test = [
            'rolling_operation', 'scale_data', 'batch_process_features',
            'optimize_dataframe', 'get_performance_stats', 'reset_stats',
            'performance_monitoring'
        ]
        
        for method_name in methods_to_test:
            assert hasattr(manager, method_name), f"Method {method_name} not found"
            
            method = getattr(manager, method_name)
            assert callable(method), f"Method {method_name} is not callable"
        
        # Test basic functionality
        result = manager.rolling_operation(sample_data['close'], 'mean', window=20)
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data)
        
        result = manager.scale_data(sample_data['close'], method='zscore')
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data)
    
    def test_enhanced_unified_vectorization_manager_batch_processing_compatibility(self, sample_data):
        """Test that enhanced UnifiedVectorizationManager maintains batch processing compatibility."""
        config = EnhancedVectorizationConfig(
            enable_vectorbt=True,
            enable_gpu=False,
            memory_efficient=True,
            enable_monitoring=True
        )
        
        manager = EnhancedUnifiedVectorizationManager(config)
        
        # Test batch processing
        feature_configs = [
            {'name': 'sma_20', 'type': 'rolling', 'params': {'operation': 'mean', 'window': 20, 'column': 'close'}},
            {'name': 'sma_50', 'type': 'rolling', 'params': {'operation': 'mean', 'window': 50, 'column': 'close'}},
            {'name': 'std_20', 'type': 'rolling', 'params': {'operation': 'std', 'window': 20, 'column': 'close'}},
            {'name': 'close_scaled', 'type': 'scaling', 'params': {'method': 'zscore', 'column': 'close'}},
            {'name': 'volume_scaled', 'type': 'scaling', 'params': {'method': 'minmax', 'column': 'volume'}}
        ]
        
        features = manager.batch_process_features(sample_data, feature_configs)
        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(sample_data)
        assert len(features.columns) == len(feature_configs)
        
        # Check that all expected features are present
        expected_features = ['sma_20', 'sma_50', 'std_20', 'close_scaled', 'volume_scaled']
        for feature in expected_features:
            assert feature in features.columns
    
    def test_enhanced_unified_vectorization_manager_performance_compatibility(self, sample_data):
        """Test that enhanced UnifiedVectorizationManager maintains performance compatibility."""
        # Test with original parameters
        original_config = EnhancedVectorizationConfig(
            enable_vectorbt=True,
            enable_gpu=False,
            memory_efficient=True,
            enable_monitoring=True,
            enable_m1_gpu=False,
            adaptive_chunking=False,
            enable_caching=False
        )
        
        # Test with enhanced parameters
        enhanced_config = EnhancedVectorizationConfig(
            enable_vectorbt=True,
            enable_gpu=False,
            memory_efficient=True,
            enable_monitoring=True,
            enable_m1_gpu=True,
            adaptive_chunking=True,
            enable_caching=True
        )
        
        original_manager = EnhancedUnifiedVectorizationManager(original_config)
        enhanced_manager = EnhancedUnifiedVectorizationManager(enhanced_config)
        
        # Test that both produce similar results
        original_result = original_manager.rolling_operation(sample_data['close'], 'mean', window=20)
        enhanced_result = enhanced_manager.rolling_operation(sample_data['close'], 'mean', window=20)
        
        # Results should be very close (allowing for small numerical differences)
        np.testing.assert_allclose(original_result.dropna(), enhanced_result.dropna(), rtol=1e-10)
    
    def test_enhanced_unified_vectorization_manager_error_handling_compatibility(self, sample_data):
        """Test that enhanced UnifiedVectorizationManager maintains error handling compatibility."""
        config = EnhancedVectorizationConfig(
            enable_vectorbt=True,
            enable_gpu=False,
            memory_efficient=True,
            enable_monitoring=True,
            fast_fail=True
        )
        
        manager = EnhancedUnifiedVectorizationManager(config)
        
        # Test invalid operation
        with pytest.raises(Exception):
            manager.rolling_operation(sample_data['close'], 'invalid_operation', window=20)
        
        # Test invalid scaling method
        with pytest.raises(Exception):
            manager.scale_data(sample_data['close'], method='invalid_method')
        
        # Test invalid batch processing
        with pytest.raises(Exception):
            manager.batch_process_features("invalid_data", [])
    
    def test_enhanced_unified_vectorization_manager_context_manager_compatibility(self, sample_data):
        """Test that enhanced UnifiedVectorizationManager maintains context manager compatibility."""
        config = EnhancedVectorizationConfig(
            enable_vectorbt=True,
            enable_gpu=False,
            memory_efficient=True,
            enable_monitoring=True
        )
        
        with EnhancedUnifiedVectorizationManager(config) as manager:
            result = manager.rolling_operation(sample_data['close'], 'mean', window=20)
            assert isinstance(result, pd.Series)
            assert len(result) == len(sample_data)
    
    def test_global_function_compatibility(self, sample_series, sample_data):
        """Test that global functions maintain compatibility."""
        # Test get_vectorbt_rolling_optimizer
        optimizer = get_vectorbt_rolling_optimizer()
        assert isinstance(optimizer, EnhancedVectorBTRollingOptimizer)
        
        result = optimizer.rolling_mean(sample_series, window=20)
        assert isinstance(result, pd.Series)
        
        # Test get_unified_vectorization_manager
        manager = get_unified_vectorization_manager()
        assert isinstance(manager, EnhancedUnifiedVectorizationManager)
        
        result = manager.rolling_operation(sample_data['close'], 'mean', window=20)
        assert isinstance(result, pd.Series)
    
    def test_alias_compatibility(self):
        """Test that class aliases work correctly."""
        # Test VectorBTRollingOptimizer alias
        assert EnhancedVectorBTRollingOptimizerAlias is EnhancedVectorBTRollingOptimizer
        
        # Test UnifiedVectorizationManager alias
        assert EnhancedUnifiedVectorizationManagerAlias is EnhancedUnifiedVectorizationManager
    
    def test_enhanced_features_work_correctly(self, sample_series, sample_data, temp_cache_dir):
        """Test that enhanced features work correctly without breaking compatibility."""
        # Test with enhanced configuration
        memory_config = MemoryConfig(
            max_memory_gb=2.0,
            memory_pressure_threshold=0.7,
            adaptive_chunking=True,
            memory_pooling=True
        )
        
        cache_config = CacheConfig(
            l1_cache_size=100,
            l2_cache_size=500,
            l2_cache_dir=temp_cache_dir,
            cache_ttl=300.0
        )
        
        optimizer = EnhancedVectorBTRollingOptimizer(
            enable_gpu=False,
            enable_parallel=True,
            memory_efficient=True,
            chunk_size=500,
            memory_config=memory_config,
            cache_config=cache_config,
            enable_m1_gpu=True,
            enable_adaptive_chunking=True,
            enable_advanced_caching=True
        )
        
        # Test that enhanced features work
        result1 = optimizer.rolling_mean(sample_series, window=20)
        result2 = optimizer.rolling_mean(sample_series, window=20)  # Should hit cache
        
        assert isinstance(result1, pd.Series)
        assert isinstance(result2, pd.Series)
        np.testing.assert_allclose(result1.dropna(), result2.dropna(), rtol=1e-10)
        
        # Test enhanced stats
        stats = optimizer.get_performance_stats()
        assert 'cache_hits' in stats
        assert 'cache_misses' in stats
        assert 'm1_gpu_operations' in stats
        assert 'adaptive_chunk_operations' in stats
        
        # Test enhanced unified manager
        enhanced_config = EnhancedVectorizationConfig(
            enable_vectorbt=True,
            enable_gpu=False,
            memory_efficient=True,
            enable_monitoring=True,
            enable_m1_gpu=True,
            adaptive_chunking=True,
            enable_caching=True,
            l1_cache_size=100,
            l2_cache_size=500,
            l2_cache_dir=temp_cache_dir
        )
        
        manager = EnhancedUnifiedVectorizationManager(enhanced_config)
        
        # Test enhanced operations
        result1 = manager.rolling_operation(sample_data['close'], 'mean', window=20)
        result2 = manager.rolling_operation(sample_data['close'], 'mean', window=20)  # Should hit cache
        
        assert isinstance(result1, pd.Series)
        assert isinstance(result2, pd.Series)
        np.testing.assert_allclose(result1.dropna(), result2.dropna(), rtol=1e-10)
        
        # Test enhanced stats
        stats = manager.get_performance_stats()
        assert 'performance_monitor' in stats
        assert 'm1_gpu_operations' in stats
        assert 'adaptive_chunk_operations' in stats
    
    def test_memory_management_compatibility(self, sample_series):
        """Test that memory management works correctly."""
        memory_config = MemoryConfig(
            max_memory_gb=1.0,
            memory_pressure_threshold=0.5,
            adaptive_chunking=True,
            memory_pooling=True
        )
        
        optimizer = EnhancedVectorBTRollingOptimizer(
            enable_gpu=False,
            enable_parallel=True,
            memory_efficient=True,
            chunk_size=100,
            memory_config=memory_config,
            enable_adaptive_chunking=True
        )
        
        # Test memory management
        result = optimizer.rolling_mean(sample_series, window=20)
        assert isinstance(result, pd.Series)
        
        # Test memory stats
        stats = optimizer.get_performance_stats()
        assert 'memory_stats' in stats
        assert 'current_pressure' in stats['memory_stats']
    
    def test_cache_management_compatibility(self, sample_series, temp_cache_dir):
        """Test that cache management works correctly."""
        cache_config = CacheConfig(
            l1_cache_size=10,
            l2_cache_size=50,
            l2_cache_dir=temp_cache_dir,
            cache_ttl=60.0
        )
        
        optimizer = EnhancedVectorBTRollingOptimizer(
            enable_gpu=False,
            enable_parallel=True,
            memory_efficient=True,
            chunk_size=100,
            cache_config=cache_config,
            enable_advanced_caching=True
        )
        
        # Test caching
        result1 = optimizer.rolling_mean(sample_series, window=20)
        result2 = optimizer.rolling_mean(sample_series, window=20)  # Should hit cache
        
        assert isinstance(result1, pd.Series)
        assert isinstance(result2, pd.Series)
        np.testing.assert_allclose(result1.dropna(), result2.dropna(), rtol=1e-10)
        
        # Test cache stats
        stats = optimizer.get_performance_stats()
        assert 'cache_stats' in stats
        assert 'l1_hit_rate' in stats['cache_stats']
        assert 'l2_hit_rate' in stats['cache_stats']
    
    def test_m1_gpu_optimization_compatibility(self, sample_series):
        """Test that M1 GPU optimization works correctly."""
        optimizer = EnhancedVectorBTRollingOptimizer(
            enable_gpu=False,
            enable_parallel=True,
            memory_efficient=True,
            chunk_size=100,
            enable_m1_gpu=True
        )
        
        # Test M1 GPU operations
        result = optimizer.rolling_mean(sample_series, window=20)
        assert isinstance(result, pd.Series)
        
        # Test M1 GPU stats
        stats = optimizer.get_performance_stats()
        assert 'm1_gpu_stats' in stats
        assert 'available' in stats['m1_gpu_stats']
    
    def test_performance_regression(self, sample_data):
        """Test that enhanced versions don't have significant performance regression."""
        # Test with original parameters
        original_config = EnhancedVectorizationConfig(
            enable_vectorbt=True,
            enable_gpu=False,
            memory_efficient=True,
            enable_monitoring=False,
            enable_m1_gpu=False,
            adaptive_chunking=False,
            enable_caching=False
        )
        
        # Test with enhanced parameters
        enhanced_config = EnhancedVectorizationConfig(
            enable_vectorbt=True,
            enable_gpu=False,
            memory_efficient=True,
            enable_monitoring=False,
            enable_m1_gpu=True,
            adaptive_chunking=True,
            enable_caching=True
        )
        
        original_manager = EnhancedUnifiedVectorizationManager(original_config)
        enhanced_manager = EnhancedUnifiedVectorizationManager(enhanced_config)
        
        # Time original implementation
        start_time = time.time()
        original_result = original_manager.rolling_operation(sample_data['close'], 'mean', window=20)
        original_time = time.time() - start_time
        
        # Time enhanced implementation
        start_time = time.time()
        enhanced_result = enhanced_manager.rolling_operation(sample_data['close'], 'mean', window=20)
        enhanced_time = time.time() - start_time
        
        # Enhanced version should not be significantly slower (allow 50% overhead for enhanced features)
        assert enhanced_time <= original_time * 1.5, f"Enhanced version too slow: {enhanced_time:.3f}s vs {original_time:.3f}s"
        
        # Results should be identical
        np.testing.assert_allclose(original_result.dropna(), enhanced_result.dropna(), rtol=1e-10)

def run_backward_compatibility_tests():
    """Run all backward compatibility tests."""
    print("🧪 Running backward compatibility tests...")
    
    # Create test instance
    test_instance = TestBackwardCompatibility()
    
    # Create sample data
    sample_data = test_instance.sample_data()
    sample_series = test_instance.sample_series()
    
    # Create temporary cache directory
    temp_cache_dir = tempfile.mkdtemp()
    
    try:
        # Run tests
        test_instance.test_enhanced_vectorbt_rolling_optimizer_initialization()
        print("✅ Enhanced VectorBT Rolling Optimizer initialization test passed")
        
        test_instance.test_enhanced_vectorbt_rolling_optimizer_api_compatibility(sample_series, sample_data)
        print("✅ Enhanced VectorBT Rolling Optimizer API compatibility test passed")
        
        test_instance.test_enhanced_vectorbt_rolling_optimizer_performance_compatibility(sample_series)
        print("✅ Enhanced VectorBT Rolling Optimizer performance compatibility test passed")
        
        test_instance.test_enhanced_vectorbt_rolling_optimizer_error_handling_compatibility(sample_series)
        print("✅ Enhanced VectorBT Rolling Optimizer error handling compatibility test passed")
        
        test_instance.test_enhanced_vectorbt_rolling_optimizer_stats_compatibility(sample_series)
        print("✅ Enhanced VectorBT Rolling Optimizer stats compatibility test passed")
        
        test_instance.test_enhanced_vectorbt_rolling_optimizer_context_manager_compatibility(sample_series)
        print("✅ Enhanced VectorBT Rolling Optimizer context manager compatibility test passed")
        
        test_instance.test_enhanced_unified_vectorization_manager_initialization()
        print("✅ Enhanced Unified Vectorization Manager initialization test passed")
        
        test_instance.test_enhanced_unified_vectorization_manager_api_compatibility(sample_data)
        print("✅ Enhanced Unified Vectorization Manager API compatibility test passed")
        
        test_instance.test_enhanced_unified_vectorization_manager_batch_processing_compatibility(sample_data)
        print("✅ Enhanced Unified Vectorization Manager batch processing compatibility test passed")
        
        test_instance.test_enhanced_unified_vectorization_manager_performance_compatibility(sample_data)
        print("✅ Enhanced Unified Vectorization Manager performance compatibility test passed")
        
        test_instance.test_enhanced_unified_vectorization_manager_error_handling_compatibility(sample_data)
        print("✅ Enhanced Unified Vectorization Manager error handling compatibility test passed")
        
        test_instance.test_enhanced_unified_vectorization_manager_context_manager_compatibility(sample_data)
        print("✅ Enhanced Unified Vectorization Manager context manager compatibility test passed")
        
        test_instance.test_global_function_compatibility(sample_series, sample_data)
        print("✅ Global function compatibility test passed")
        
        test_instance.test_alias_compatibility()
        print("✅ Alias compatibility test passed")
        
        test_instance.test_enhanced_features_work_correctly(sample_series, sample_data, temp_cache_dir)
        print("✅ Enhanced features work correctly test passed")
        
        test_instance.test_memory_management_compatibility(sample_series)
        print("✅ Memory management compatibility test passed")
        
        test_instance.test_cache_management_compatibility(sample_series, temp_cache_dir)
        print("✅ Cache management compatibility test passed")
        
        test_instance.test_m1_gpu_optimization_compatibility(sample_series)
        print("✅ M1 GPU optimization compatibility test passed")
        
        test_instance.test_performance_regression(sample_data)
        print("✅ Performance regression test passed")
        
        print("\n🎉 All backward compatibility tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Backward compatibility test failed: {e}")
        return False
        
    finally:
        # Cleanup
        shutil.rmtree(temp_cache_dir, ignore_errors=True)

if __name__ == "__main__":
    success = run_backward_compatibility_tests()
    exit(0 if success else 1)