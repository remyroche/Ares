"""
Hardware Optimization Integration Examples

This module provides comprehensive examples of how to use the hardware optimization
system for various demanding processes in the codebase.
"""

import pandas as pd
import numpy as np
import time
from typing import List, Dict, Any, Callable
import logging

# Hardware optimization imports
from .hardware.integrated_hardware_manager import (
    get_integrated_hardware_manager, WorkloadType, OptimizationLevel
)
from .hardware.adaptive_optimization_engine import (
    get_adaptive_optimization_engine, OptimizationTarget
)
from .hardware.advanced_memory_optimizer import (
    get_advanced_memory_manager, MemoryStrategy
)
from .hardware.enhanced_gpu_manager import (
    get_enhanced_gpu_manager, GPUOperationType
)

# Parallel processing imports
from .parallel_processing_optimizer import (
    MacM1ParallelOptimizer, hardware_optimized, memory_efficient_processing,
    gpu_accelerated, adaptive_workload_optimization
)

# ML common imports
from .ml_common.hardware_optimized_parallel_processor import (
    HardwareOptimizedMLProcessor, get_hardware_optimized_ml_processor,
    ml_training_optimized, feature_engineering_optimized, hpo_optimized
)
from .ml_common.gpu_acceleration_utils import (
    GPUAccelerationUtils, get_gpu_acceleration_utils,
    gpu_accelerated as ml_gpu_accelerated, adaptive_gpu_acceleration
)

logger = logging.getLogger(__name__)

class HardwareOptimizationExamples:
    """Comprehensive examples of hardware optimization usage."""

    def __init__(self):
        """Initialize the examples with hardware managers."""
        self.hardware_manager = get_integrated_hardware_manager()
        self.adaptive_engine = get_adaptive_optimization_engine()
        self.memory_manager = get_advanced_memory_manager()
        self.gpu_manager = get_enhanced_gpu_manager()
        self.parallel_optimizer = MacM1ParallelOptimizer(enable_hardware_optimization=True)
        self.ml_processor = get_hardware_optimized_ml_processor()
        self.gpu_utils = get_gpu_acceleration_utils()

    def example_1_basic_parallel_processing(self):
        """Example 1: Basic parallel processing with hardware optimization."""
        print("=== Example 1: Basic Parallel Processing ===")
        
        # Create sample data
        df = pd.DataFrame({
            'feature_1': np.random.randn(10000),
            'feature_2': np.random.randn(10000),
            'feature_3': np.random.randn(10000)
        })
        
        def process_chunk(chunk_df):
            """Process a chunk of data."""
            return chunk_df.apply(lambda x: x ** 2 + np.sin(x))
        
        # Method 1: Using hardware-optimized parallel processing
        start_time = time.time()
        result1 = self.parallel_optimizer.parallel_apply(df, process_chunk)
        time1 = time.time() - start_time
        
        # Method 2: Using decorators
        @hardware_optimized(WorkloadType.DATA_PROCESSING, OptimizationLevel.AGGRESSIVE)
        def process_data_optimized(data):
            return self.parallel_optimizer.parallel_apply(data, process_chunk)
        
        start_time = time.time()
        result2 = process_data_optimized(df)
        time2 = time.time() - start_time
        
        print(f"Hardware-optimized processing time: {time1:.2f}s")
        print(f"Decorator-based processing time: {time2:.2f}s")
        print(f"Results match: {np.allclose(result1, result2)}")
        
        return result1

    def example_2_memory_optimization(self):
        """Example 2: Memory optimization for large datasets."""
        print("\n=== Example 2: Memory Optimization ===")
        
        # Create large dataset
        large_df = pd.DataFrame({
            'int_col': np.random.randint(0, 1000000, 100000),
            'float_col': np.random.randn(100000),
            'string_col': [f'string_{i}' for i in range(100000)]
        })
        
        print(f"Original memory usage: {large_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Method 1: Using memory manager directly
        with self.memory_manager.memory_context(MemoryStrategy.ADAPTIVE):
            optimized_df1, optimization_info = self.memory_manager.optimize_dataframe(large_df)
        
        print(f"Optimized memory usage: {optimized_df1.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print(f"Memory saved: {optimization_info.get('memory_saved_mb', 0):.2f} MB")
        
        # Method 2: Using decorators
        @memory_efficient_processing(memory_threshold_mb=50.0)
        def process_large_dataframe(df):
            return df.apply(lambda x: x * 2 if x.dtype in ['int64', 'float64'] else x)
        
        result = process_large_dataframe(large_df)
        print(f"Decorator-based processing completed")
        
        return optimized_df1

    def example_3_gpu_acceleration(self):
        """Example 3: GPU acceleration for suitable operations."""
        print("\n=== Example 3: GPU Acceleration ===")
        
        # Create matrices for multiplication
        A = np.random.randn(2000, 2000).astype(np.float32)
        B = np.random.randn(2000, 2000).astype(np.float32)
        
        def matrix_multiply(a, b):
            return np.dot(a, b)
        
        # Method 1: Using GPU acceleration decorator
        @gpu_accelerated(GPUOperationType.MATRIX_MULTIPLICATION)
        def gpu_matrix_multiply(a, b):
            return matrix_multiply(a, b)
        
        start_time = time.time()
        result_gpu = gpu_matrix_multiply(A, B)
        gpu_time = time.time() - start_time
        
        # Method 2: CPU fallback
        start_time = time.time()
        result_cpu = matrix_multiply(A, B)
        cpu_time = time.time() - start_time
        
        print(f"GPU processing time: {gpu_time:.2f}s")
        print(f"CPU processing time: {cpu_time:.2f}s")
        print(f"Speedup: {cpu_time / gpu_time:.2f}x")
        print(f"Results match: {np.allclose(result_gpu, result_cpu)}")
        
        return result_gpu

    def example_4_ml_training_optimization(self):
        """Example 4: ML training with hardware optimization."""
        print("\n=== Example 4: ML Training Optimization ===")
        
        # Create sample ML data
        from sklearn.datasets import make_classification
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        
        X, y = make_classification(n_samples=10000, n_features=20, n_classes=2, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Method 1: Using hardware-optimized ML processor
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        start_time = time.time()
        trained_model = self.ml_processor.process_ml_training(model, X_train, y_train)
        ml_time = time.time() - start_time
        
        # Method 2: Using decorators
        @ml_training_optimized(enable_gpu=True)
        def train_model_decorated(model, X, y):
            return model.fit(X, y)
        
        model2 = RandomForestClassifier(n_estimators=100, random_state=42)
        start_time = time.time()
        trained_model2 = train_model_decorated(model2, X_train, y_train)
        decorator_time = time.time() - start_time
        
        print(f"Hardware-optimized training time: {ml_time:.2f}s")
        print(f"Decorator-based training time: {decorator_time:.2f}s")
        
        # Evaluate performance
        score1 = trained_model.score(X_test, y_test)
        score2 = trained_model2.score(X_test, y_test)
        print(f"Model 1 accuracy: {score1:.4f}")
        print(f"Model 2 accuracy: {score2:.4f}")
        
        return trained_model

    def example_5_feature_engineering_optimization(self):
        """Example 5: Feature engineering with hardware optimization."""
        print("\n=== Example 5: Feature Engineering Optimization ===")
        
        # Create sample data
        df = pd.DataFrame({
            'price': np.random.randn(10000) * 100 + 1000,
            'volume': np.random.randint(1000, 10000, 10000),
            'sector': np.random.choice(['tech', 'finance', 'healthcare'], 10000)
        })
        
        def create_technical_indicators(df):
            """Create technical indicators."""
            result = df.copy()
            result['sma_20'] = df['price'].rolling(20).mean()
            result['rsi'] = self._calculate_rsi(df['price'])
            result['bollinger_upper'] = df['price'].rolling(20).mean() + 2 * df['price'].rolling(20).std()
            return result[['sma_20', 'rsi', 'bollinger_upper']]
        
        def create_volume_features(df):
            """Create volume-based features."""
            result = df.copy()
            result['volume_sma'] = df['volume'].rolling(10).mean()
            result['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
            return result[['volume_sma', 'volume_ratio']]
        
        feature_funcs = [create_technical_indicators, create_volume_features]
        
        # Method 1: Using hardware-optimized ML processor
        start_time = time.time()
        features1 = self.ml_processor.process_feature_engineering(df, feature_funcs)
        processor_time = time.time() - start_time
        
        # Method 2: Using decorators
        @feature_engineering_optimized(enable_gpu=True)
        def create_features_decorated(df):
            return self.ml_processor.process_feature_engineering(df, feature_funcs)
        
        start_time = time.time()
        features2 = create_features_decorated(df)
        decorator_time = time.time() - start_time
        
        print(f"Hardware-optimized feature engineering time: {processor_time:.2f}s")
        print(f"Decorator-based feature engineering time: {decorator_time:.2f}s")
        print(f"Features created: {features1.shape[1]}")
        print(f"Results match: {np.allclose(features1.select_dtypes(include=[np.number]), features2.select_dtypes(include=[np.number]))}")
        
        return features1

    def example_6_hyperparameter_optimization(self):
        """Example 6: Hyperparameter optimization with hardware acceleration."""
        print("\n=== Example 6: Hyperparameter Optimization ===")
        
        # Create sample data
        from sklearn.datasets import make_classification
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        
        X, y = make_classification(n_samples=5000, n_features=15, n_classes=2, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10]
        }
        
        # Method 1: Using hardware-optimized ML processor
        model = RandomForestClassifier(random_state=42)
        
        start_time = time.time()
        best_model1 = self.ml_processor.process_hyperparameter_optimization(
            model, X_train, y_train, param_grid, cv=3
        )
        processor_time = time.time() - start_time
        
        # Method 2: Using decorators
        @hpo_optimized(enable_gpu=True)
        def optimize_hyperparameters_decorated(model, X, y, param_grid):
            return self.ml_processor.process_hyperparameter_optimization(
                model, X, y, param_grid, cv=3
            )
        
        start_time = time.time()
        best_model2 = optimize_hyperparameters_decorated(model, X_train, y_train, param_grid)
        decorator_time = time.time() - start_time
        
        print(f"Hardware-optimized HPO time: {processor_time:.2f}s")
        print(f"Decorator-based HPO time: {decorator_time:.2f}s")
        
        # Evaluate best models
        score1 = best_model1.score(X_test, y_test)
        score2 = best_model2.score(X_test, y_test)
        print(f"Best model 1 accuracy: {score1:.4f}")
        print(f"Best model 2 accuracy: {score2:.4f}")
        
        return best_model1

    def example_7_adaptive_optimization(self):
        """Example 7: Adaptive optimization based on workload characteristics."""
        print("\n=== Example 7: Adaptive Optimization ===")
        
        # Create different types of workloads
        workloads = {
            'small_data': pd.DataFrame(np.random.randn(1000, 10)),
            'medium_data': pd.DataFrame(np.random.randn(10000, 20)),
            'large_data': pd.DataFrame(np.random.randn(100000, 50))
        }
        
        def process_workload(df):
            """Process workload with different characteristics."""
            return df.apply(lambda x: x ** 2 + np.sin(x))
        
        # Use adaptive optimization
        @adaptive_workload_optimization()
        def adaptive_process(data):
            return self.parallel_optimizer.parallel_apply(data, process_workload)
        
        results = {}
        for name, data in workloads.items():
            start_time = time.time()
            result = adaptive_process(data)
            processing_time = time.time() - start_time
            
            results[name] = {
                'size': data.shape,
                'time': processing_time,
                'throughput': data.shape[0] / processing_time
            }
            
            print(f"{name}: {data.shape} -> {processing_time:.2f}s ({results[name]['throughput']:.0f} rows/s)")
        
        return results

    def example_8_performance_monitoring(self):
        """Example 8: Performance monitoring and optimization reporting."""
        print("\n=== Example 8: Performance Monitoring ===")
        
        # Get system status
        system_status = self.hardware_manager.get_system_status()
        print("System Status:")
        print(f"  Initialized: {system_status.get('initialized', False)}")
        print(f"  Current workload: {system_status.get('current_workload', 'None')}")
        
        # Get optimization report
        optimization_report = self.hardware_manager.get_optimization_report()
        print("\nOptimization Report:")
        print(f"  Cache hit rate: {optimization_report['cache_statistics']['hit_rate']:.2%}")
        print(f"  Total operations: {optimization_report['performance_metrics']['total_operations']}")
        print(f"  Optimizations applied: {optimization_report['performance_metrics']['optimizations_applied']}")
        
        # Get memory report
        memory_report = self.hardware_manager.get_memory_report()
        print(f"\nMemory Report:")
        print(f"  Total memory usage: {memory_report['total_memory_usage_mb']:.2f} MB")
        
        # Get learning report
        learning_report = self.adaptive_engine.get_learning_report()
        print(f"\nLearning Report:")
        print(f"  Learning enabled: {learning_report['learning_enabled']}")
        print(f"  Auto-tuning enabled: {learning_report['auto_tuning_enabled']}")
        print(f"  Total performance records: {learning_report['learning_statistics']['total_performance_records']}")
        
        return {
            'system_status': system_status,
            'optimization_report': optimization_report,
            'memory_report': memory_report,
            'learning_report': learning_report
        }

    def _calculate_rsi(self, prices, window=14):
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def run_all_examples(self):
        """Run all examples and return results."""
        print("🚀 Running Hardware Optimization Examples")
        print("=" * 50)
        
        results = {}
        
        try:
            results['parallel_processing'] = self.example_1_basic_parallel_processing()
        except Exception as e:
            print(f"Example 1 failed: {e}")
        
        try:
            results['memory_optimization'] = self.example_2_memory_optimization()
        except Exception as e:
            print(f"Example 2 failed: {e}")
        
        try:
            results['gpu_acceleration'] = self.example_3_gpu_acceleration()
        except Exception as e:
            print(f"Example 3 failed: {e}")
        
        try:
            results['ml_training'] = self.example_4_ml_training_optimization()
        except Exception as e:
            print(f"Example 4 failed: {e}")
        
        try:
            results['feature_engineering'] = self.example_5_feature_engineering_optimization()
        except Exception as e:
            print(f"Example 5 failed: {e}")
        
        try:
            results['hyperparameter_optimization'] = self.example_6_hyperparameter_optimization()
        except Exception as e:
            print(f"Example 6 failed: {e}")
        
        try:
            results['adaptive_optimization'] = self.example_7_adaptive_optimization()
        except Exception as e:
            print(f"Example 7 failed: {e}")
        
        try:
            results['performance_monitoring'] = self.example_8_performance_monitoring()
        except Exception as e:
            print(f"Example 8 failed: {e}")
        
        print("\n✅ All examples completed!")
        return results

def main():
    """Main function to run examples."""
    examples = HardwareOptimizationExamples()
    results = examples.run_all_examples()
    return results

if __name__ == "__main__":
    main()