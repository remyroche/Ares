"""
Comprehensive Examples for M1 Hardware Enhancements.

This module provides comprehensive examples demonstrating all M1/M2/M3/M4
hardware optimizations including unified memory, CPU optimization, GPU acceleration,
and Neural Engine integration.
"""

import logging
import time
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Import all M1 optimization modules
from .m1_comprehensive_optimizer import (
    get_comprehensive_optimizer, ComprehensiveConfig, OptimizationStrategy,
    WorkloadCategory, m1_optimized
)
from .m1_unified_memory_manager import (
    get_unified_memory_manager, optimize_for_unified_memory, unified_memory_optimized
)
from .m1_advanced_cpu_optimizer import (
    get_advanced_cpu_optimizer, optimize_cpu_execution, parallel_cpu_execution
)
from .m1_enhanced_gpu_manager import (
    get_enhanced_gpu_manager, gpu_accelerated, GPUOperationType
)
from .m1_neural_engine_manager import (
    get_neural_engine_manager, neural_engine_optimized, NeuralEngineOperation
)

from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_success, tprint_warning, tprint_error,
    tprint_performance, LogLevel
)

logger = logging.getLogger(__name__)

class M1ComprehensiveExamples:
    """Comprehensive examples for M1 hardware optimizations."""
    
    def __init__(self):
        self.logger = logger.getChild('M1ComprehensiveExamples')
        
        # Initialize optimizers
        self.comprehensive_optimizer = get_comprehensive_optimizer()
        self.unified_memory_manager = get_unified_memory_manager()
        self.cpu_optimizer = get_advanced_cpu_optimizer()
        self.gpu_manager = get_enhanced_gpu_manager()
        self.neural_engine_manager = get_neural_engine_manager()
        
        self.logger.info("🚀 M1 Comprehensive Examples initialized")
    
    def run_all_examples(self):
        """Run all comprehensive examples."""
        tprint_success("🚀 Starting M1 Comprehensive Examples")
        
        try:
            # Example 1: Unified Memory Management
            self.example_unified_memory_management()
            
            # Example 2: Advanced CPU Optimization
            self.example_advanced_cpu_optimization()
            
            # Example 3: Enhanced GPU Acceleration
            self.example_enhanced_gpu_acceleration()
            
            # Example 4: Neural Engine Integration
            self.example_neural_engine_integration()
            
            # Example 5: Comprehensive Optimization
            self.example_comprehensive_optimization()
            
            # Example 6: Real-world Financial Modeling
            self.example_financial_modeling()
            
            # Example 7: Machine Learning Pipeline
            self.example_ml_pipeline()
            
            # Example 8: Performance Monitoring
            self.example_performance_monitoring()
            
            tprint_success("✅ All M1 Comprehensive Examples completed successfully")
            
        except Exception as e:
            tprint_error(f"❌ Examples failed: {e}")
            self.logger.error(f"Examples failed: {e}")
    
    def example_unified_memory_management(self):
        """Example: Unified Memory Management."""
        tprint_info("🧠 Example 1: Unified Memory Management")
        
        # Create large datasets
        large_matrix = np.random.random((1000, 1000)).astype(np.float32)
        large_dataframe = pd.DataFrame(np.random.random((10000, 100)))
        
        # Optimize for unified memory
        optimized_matrix = optimize_for_unified_memory(large_matrix, 'matrix_operations', 'gpu')
        optimized_dataframe = optimize_for_unified_memory(large_dataframe, 'data_processing', 'cpu')
        
        # Demonstrate memory allocation
        allocation_id = self.unified_memory_manager.allocate_for_operation(
            'matrix_operations', 100.0, 'example'
        )
        
        # Get memory statistics
        memory_stats = self.unified_memory_manager.get_comprehensive_stats()
        
        tprint_success(f"✅ Unified Memory: {memory_stats['allocations']} allocations, "
                      f"{memory_stats['current_usage_mb']:.1f}MB used")
        
        # Cleanup
        self.unified_memory_manager.memory_pool.free_memory(allocation_id)
    
    def example_advanced_cpu_optimization(self):
        """Example: Advanced CPU Optimization."""
        tprint_info("⚡ Example 2: Advanced CPU Optimization")
        
        # CPU-intensive computation
        @optimize_cpu_execution(WorkloadType.CPU_INTENSIVE)
        def cpu_intensive_task(data):
            """CPU-intensive matrix operations."""
            result = np.zeros_like(data)
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    result[i, j] = np.sum(data[i, :] * data[:, j])
            return result
        
        # Memory-intensive computation
        @optimize_cpu_execution(WorkloadType.MEMORY_INTENSIVE)
        def memory_intensive_task(data):
            """Memory-intensive operations."""
            return np.fft.fft(data, axis=0)
        
        # Test data
        test_data = np.random.random((500, 500)).astype(np.float32)
        
        # Execute with optimization
        start_time = time.time()
        cpu_result = cpu_intensive_task(test_data)
        cpu_time = time.time() - start_time
        
        start_time = time.time()
        memory_result = memory_intensive_task(test_data)
        memory_time = time.time() - start_time
        
        # Get CPU metrics
        cpu_metrics = self.cpu_optimizer.get_performance_metrics()
        
        tprint_success(f"✅ CPU Optimization: {cpu_time:.3f}s CPU, {memory_time:.3f}s memory, "
                      f"{cpu_metrics['cpu_metrics']['total_operations']} operations")
    
    def example_enhanced_gpu_acceleration(self):
        """Example: Enhanced GPU Acceleration."""
        tprint_info("🎮 Example 3: Enhanced GPU Acceleration")
        
        if not self.gpu_manager.is_available():
            tprint_warning("⚠️ GPU not available - skipping GPU examples")
            return
        
        # Matrix multiplication with GPU acceleration
        @gpu_accelerated(GPUOperationType.MATRIX_MULTIPLICATION)
        def gpu_matrix_multiply(A, B):
            """GPU-accelerated matrix multiplication."""
            return np.dot(A, B)
        
        # Tensor operations with GPU acceleration
        @gpu_accelerated(GPUOperationType.TENSOR_OPERATIONS)
        def gpu_tensor_ops(data):
            """GPU-accelerated tensor operations."""
            return data * 2 + 1
        
        # Test data
        A = np.random.random((1000, 1000)).astype(np.float32)
        B = np.random.random((1000, 1000)).astype(np.float32)
        tensor_data = np.random.random((500, 500)).astype(np.float32)
        
        # Execute with GPU acceleration
        start_time = time.time()
        gpu_result = gpu_matrix_multiply(A, B)
        gpu_time = time.time() - start_time
        
        start_time = time.time()
        tensor_result = gpu_tensor_ops(tensor_data)
        tensor_time = time.time() - start_time
        
        # Get GPU metrics
        gpu_metrics = self.gpu_manager.get_performance_metrics()
        
        tprint_success(f"✅ GPU Acceleration: {gpu_time:.3f}s matrix, {tensor_time:.3f}s tensor, "
                      f"{gpu_metrics['gpu_metrics']['total_operations']} operations")
    
    def example_neural_engine_integration(self):
        """Example: Neural Engine Integration."""
        tprint_info("🧠 Example 4: Neural Engine Integration")
        
        if not self.neural_engine_manager.is_available():
            tprint_warning("⚠️ Neural Engine not available - skipping Neural Engine examples")
            return
        
        # Neural network inference with Neural Engine
        @neural_engine_optimized(NeuralEngineOperation.INFERENCE)
        def neural_inference(model, data):
            """Neural Engine optimized inference."""
            # Simulate neural network inference
            return np.random.random(data.shape[0])
        
        # Test data
        test_data = np.random.random((100, 10)).astype(np.float32)
        dummy_model = "dummy_model"  # In real implementation, would be actual model
        
        # Execute with Neural Engine optimization
        start_time = time.time()
        inference_result = neural_inference(dummy_model, test_data)
        inference_time = time.time() - start_time
        
        # Get Neural Engine metrics
        neural_metrics = self.neural_engine_manager.get_performance_metrics()
        
        tprint_success(f"✅ Neural Engine: {inference_time:.3f}s inference, "
                      f"{neural_metrics['executor_metrics']['total_inferences']} inferences")
    
    def example_comprehensive_optimization(self):
        """Example: Comprehensive Optimization."""
        tprint_info("🚀 Example 5: Comprehensive Optimization")
        
        # Comprehensive optimization decorator
        @m1_optimized("matrix_operations", WorkloadCategory.MACHINE_LEARNING)
        def comprehensive_matrix_operations(data):
            """Comprehensively optimized matrix operations."""
            # This would be automatically optimized by the comprehensive optimizer
            return np.dot(data, data.T)
        
        # Test data
        test_data = np.random.random((800, 800)).astype(np.float32)
        
        # Execute with comprehensive optimization
        start_time = time.time()
        result = comprehensive_matrix_operations(test_data)
        execution_time = time.time() - start_time
        
        # Get comprehensive metrics
        comprehensive_metrics = self.comprehensive_optimizer.get_comprehensive_metrics()
        
        tprint_success(f"✅ Comprehensive: {execution_time:.3f}s execution, "
                      f"{comprehensive_metrics['overall_metrics']['total_operations']} operations")
    
    def example_financial_modeling(self):
        """Example: Real-world Financial Modeling."""
        tprint_info("💰 Example 6: Financial Modeling")
        
        # Monte Carlo simulation with M1 optimization
        @m1_optimized("monte_carlo", WorkloadCategory.FINANCIAL_MODELING)
        def monte_carlo_simulation(returns, num_simulations=10000):
            """Monte Carlo simulation for portfolio optimization."""
            # Generate random scenarios
            scenarios = np.random.normal(0, 0.02, (num_simulations, len(returns)))
            
            # Calculate portfolio values
            portfolio_values = np.zeros(num_simulations)
            for i in range(num_simulations):
                portfolio_values[i] = np.sum(returns * (1 + scenarios[i]))
            
            return {
                'mean': np.mean(portfolio_values),
                'std': np.std(portfolio_values),
                'var_95': np.percentile(portfolio_values, 5),
                'var_99': np.percentile(portfolio_values, 1)
            }
        
        # Backtesting with M1 optimization
        @m1_optimized("backtesting", WorkloadCategory.BACKTESTING)
        def backtest_strategy(prices, signals):
            """Backtest trading strategy with M1 optimization."""
            returns = np.diff(prices) / prices[:-1]
            strategy_returns = returns * signals[:-1]
            
            return {
                'total_return': np.prod(1 + strategy_returns) - 1,
                'sharpe_ratio': np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252),
                'max_drawdown': np.min(np.cumprod(1 + strategy_returns)) - 1
            }
        
        # Test data
        returns = np.random.normal(0.001, 0.02, 1000)
        prices = np.cumprod(1 + returns) * 100
        signals = np.random.choice([-1, 0, 1], 1000, p=[0.3, 0.4, 0.3])
        
        # Execute financial modeling
        start_time = time.time()
        mc_result = monte_carlo_simulation(returns)
        mc_time = time.time() - start_time
        
        start_time = time.time()
        bt_result = backtest_strategy(prices, signals)
        bt_time = time.time() - start_time
        
        tprint_success(f"✅ Financial Modeling: {mc_time:.3f}s Monte Carlo, {bt_time:.3f}s backtesting")
        tprint_info(f"   Monte Carlo: Mean={mc_result['mean']:.4f}, VaR95={mc_result['var_95']:.4f}")
        tprint_info(f"   Backtesting: Return={bt_result['total_return']:.4f}, Sharpe={bt_result['sharpe_ratio']:.4f}")
    
    def example_ml_pipeline(self):
        """Example: Machine Learning Pipeline."""
        tprint_info("🤖 Example 7: Machine Learning Pipeline")
        
        # Feature engineering with M1 optimization
        @m1_optimized("feature_engineering", WorkloadCategory.MACHINE_LEARNING)
        def feature_engineering(data):
            """Feature engineering with M1 optimization."""
            features = {}
            
            # Technical indicators
            features['sma_20'] = data.rolling(20).mean()
            features['rsi'] = self._calculate_rsi(data)
            features['bollinger_upper'] = features['sma_20'] + 2 * data.rolling(20).std()
            features['bollinger_lower'] = features['sma_20'] - 2 * data.rolling(20).std()
            
            # Volatility features
            features['volatility'] = data.rolling(20).std()
            features['volatility_ratio'] = features['volatility'] / features['sma_20']
            
            return pd.DataFrame(features)
        
        # Model training with M1 optimization
        @m1_optimized("model_training", WorkloadCategory.MACHINE_LEARNING)
        def train_model(X, y):
            """Train model with M1 optimization."""
            # Simulate model training
            from sklearn.linear_model import LinearRegression
            from sklearn.model_selection import train_test_split
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            score = model.score(X_test, y_test)
            return model, score
        
        # Test data
        np.random.seed(42)
        data = pd.Series(np.cumsum(np.random.randn(1000)) + 100)
        y = np.random.randn(1000)
        
        # Execute ML pipeline
        start_time = time.time()
        features = feature_engineering(data)
        feature_time = time.time() - start_time
        
        start_time = time.time()
        model, score = train_model(features.fillna(0), y)
        training_time = time.time() - start_time
        
        tprint_success(f"✅ ML Pipeline: {feature_time:.3f}s features, {training_time:.3f}s training")
        tprint_info(f"   Features: {features.shape[1]} features, {features.shape[0]} samples")
        tprint_info(f"   Model Score: {score:.4f}")
    
    def example_performance_monitoring(self):
        """Example: Performance Monitoring."""
        tprint_info("📊 Example 8: Performance Monitoring")
        
        # Get comprehensive metrics
        comprehensive_metrics = self.comprehensive_optimizer.get_comprehensive_metrics()
        
        # Display metrics
        tprint_info("📈 Comprehensive Performance Metrics:")
        tprint_info(f"   Total Operations: {comprehensive_metrics['overall_metrics']['total_operations']}")
        tprint_info(f"   Successful Operations: {comprehensive_metrics['overall_metrics']['successful_operations']}")
        tprint_info(f"   Average Execution Time: {comprehensive_metrics['overall_metrics']['average_execution_time']:.4f}s")
        
        # Memory metrics
        memory_metrics = comprehensive_metrics['unified_memory']
        tprint_info(f"   Memory Allocations: {memory_metrics['allocations']}")
        tprint_info(f"   Current Memory Usage: {memory_metrics['current_usage_mb']:.1f}MB")
        tprint_info(f"   Peak Memory Usage: {memory_metrics['peak_usage_mb']:.1f}MB")
        
        # CPU metrics
        cpu_metrics = comprehensive_metrics['cpu_optimizer']
        tprint_info(f"   CPU Operations: {cpu_metrics['cpu_metrics']['total_operations']}")
        tprint_info(f"   Thermal Throttling Events: {cpu_metrics['cpu_metrics']['thermal_throttling_events']}")
        
        # GPU metrics
        gpu_metrics = comprehensive_metrics['gpu_manager']
        tprint_info(f"   GPU Operations: {gpu_metrics['gpu_metrics']['total_operations']}")
        tprint_info(f"   GPU Available: {gpu_metrics['mps_available']}")
        
        # Neural Engine metrics
        neural_metrics = comprehensive_metrics['neural_engine']
        tprint_info(f"   Neural Engine Available: {neural_metrics['neural_engine_available']}")
        tprint_info(f"   Neural Engine Inferences: {neural_metrics['executor_metrics']['total_inferences']}")
        
        tprint_success("✅ Performance monitoring completed")
    
    def _calculate_rsi(self, data, window=14):
        """Calculate RSI indicator."""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

def run_comprehensive_examples():
    """Run all comprehensive examples."""
    examples = M1ComprehensiveExamples()
    examples.run_all_examples()

def run_benchmark_comparison():
    """Run benchmark comparison between optimized and non-optimized code."""
    tprint_info("🏁 Running Benchmark Comparison")
    
    # Test data
    large_matrix = np.random.random((2000, 2000)).astype(np.float32)
    
    # Non-optimized matrix multiplication
    start_time = time.time()
    result_unoptimized = np.dot(large_matrix, large_matrix.T)
    unoptimized_time = time.time() - start_time
    
    # Optimized matrix multiplication
    @m1_optimized("matrix_operations", WorkloadCategory.MACHINE_LEARNING)
    def optimized_matrix_multiply(A, B):
        return np.dot(A, B)
    
    start_time = time.time()
    result_optimized = optimized_matrix_multiply(large_matrix, large_matrix.T)
    optimized_time = time.time() - start_time
    
    # Calculate improvement
    improvement = (unoptimized_time - optimized_time) / unoptimized_time * 100
    
    tprint_success(f"✅ Benchmark Results:")
    tprint_info(f"   Unoptimized: {unoptimized_time:.3f}s")
    tprint_info(f"   Optimized: {optimized_time:.3f}s")
    tprint_info(f"   Improvement: {improvement:.1f}%")
    
    return {
        'unoptimized_time': unoptimized_time,
        'optimized_time': optimized_time,
        'improvement_percent': improvement
    }

if __name__ == "__main__":
    # Run examples
    run_comprehensive_examples()
    
    # Run benchmark
    run_benchmark_comparison()