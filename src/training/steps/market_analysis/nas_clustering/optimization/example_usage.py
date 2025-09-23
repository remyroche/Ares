"""
Example Usage of NAS Optimization with Grid Utils and Hardware Integration.

This module demonstrates how to use the NAS optimization framework with:
- Grid utilities for coarse-to-fine optimization
- Matrix operations for efficient computations
- Hardware optimization for performance
- Multi-objective optimization for regime detection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path

# Import NAS optimization components
from .nas_optimization_integration import NASOptimizationIntegration
from .nas_optimization_config import NASOptimizationConfig, OptimizationStrategy
from src.utils.hardware.unified_hardware_manager import WorkloadType, OptimizationLevel

# Import NAS clustering components
from ..core.nas_config import NASClusteringConfig, NASArchitectureType
from ..core.nas_clusterer import NASClusterer

logger = logging.getLogger(__name__)


def example_short_term_trading_optimization():
    """Example: Optimize NAS for short-term trading."""
    print("🚀 Example: Short-term Trading NAS Optimization")
    
    # Create configuration for short-term trading
    config = NASOptimizationConfig.create_short_term_trading_config()
    
    # Create optimization integration
    optimizer = NASOptimizationIntegration(config)
    
    # Generate sample market data
    market_data = generate_sample_market_data()
    features = generate_sample_features()
    timestamps = generate_sample_timestamps()
    
    # Create base NAS configuration
    nas_config = NASClusteringConfig.create_short_term_trading_config()
    
    # Run optimization
    results = optimizer.run_optimization(
        market_data=market_data,
        features=features,
        timestamps=timestamps,
        nas_config=nas_config,
        save_path="nas_optimization_results/short_term_trading"
    )
    
    # Print results
    print(f"✅ Optimization completed in {results['execution_time']:.2f}s")
    print(f"📊 Best score: {results['optimization_results']['best_overall_score']:.4f}")
    print(f"🎯 Best phase: {results['optimization_results']['best_phase']}")
    print(f"📈 Recommendations: {results['recommendations']}")
    
    return results


def example_high_performance_optimization():
    """Example: Optimize NAS for high performance."""
    print("🚀 Example: High Performance NAS Optimization")
    
    # Create configuration for high performance
    config = NASOptimizationConfig.create_high_performance_config()
    
    # Create optimization integration
    optimizer = NASOptimizationIntegration(config)
    
    # Generate sample market data
    market_data = generate_sample_market_data()
    features = generate_sample_features()
    timestamps = generate_sample_timestamps()
    
    # Create base NAS configuration
    nas_config = NASClusteringConfig.create_short_term_trading_config()
    
    # Run optimization
    results = optimizer.run_optimization(
        market_data=market_data,
        features=features,
        timestamps=timestamps,
        nas_config=nas_config,
        save_path="nas_optimization_results/high_performance"
    )
    
    # Print results
    print(f"✅ Optimization completed in {results['execution_time']:.2f}s")
    print(f"📊 Best score: {results['optimization_results']['best_overall_score']:.4f}")
    print(f"🎯 Best phase: {results['optimization_results']['best_phase']}")
    print(f"📈 Recommendations: {results['recommendations']}")
    
    return results


def example_quick_test_optimization():
    """Example: Quick test optimization."""
    print("🚀 Example: Quick Test NAS Optimization")
    
    # Create configuration for quick testing
    config = NASOptimizationConfig.create_quick_test_config()
    
    # Create optimization integration
    optimizer = NASOptimizationIntegration(config)
    
    # Generate sample market data
    market_data = generate_sample_market_data()
    features = generate_sample_features()
    timestamps = generate_sample_timestamps()
    
    # Create base NAS configuration
    nas_config = NASClusteringConfig.create_short_term_trading_config()
    
    # Run optimization
    results = optimizer.run_optimization(
        market_data=market_data,
        features=features,
        timestamps=timestamps,
        nas_config=nas_config,
        save_path="nas_optimization_results/quick_test"
    )
    
    # Print results
    print(f"✅ Optimization completed in {results['execution_time']:.2f}s")
    print(f"📊 Best score: {results['optimization_results']['best_overall_score']:.4f}")
    print(f"🎯 Best phase: {results['optimization_results']['best_phase']}")
    print(f"📈 Recommendations: {results['recommendations']}")
    
    return results


def example_custom_optimization():
    """Example: Custom optimization configuration."""
    print("🚀 Example: Custom NAS Optimization")
    
    # Create custom configuration
    config = NASOptimizationConfig(
        optimization_strategy=OptimizationStrategy.HYBRID,
        # Custom grid configuration
        grid_config=config.grid_config.__class__(
            enable_coarse_grid=True,
            enable_fine_grid=True,
            coarse_grid_points=10,
            fine_grid_points=6,
            grid_phase_trials=40
        ),
        # Custom hardware configuration
        hardware_config=config.hardware_config.__class__(
            enable_hardware_optimization=True,
            workload_type=WorkloadType.ML_TRAINING,
            optimization_level=OptimizationLevel.AGGRESSIVE,
            memory_limit_gb=16.0
        ),
        # Custom Bayesian configuration
        bayesian_config=config.bayesian_config.__class__(
            enable_tpe_optimization=True,
            n_trials=150,
            n_startup_trials=25,
            objectives=['regime_stability', 'economic_significance', 'trading_viability'],
            objective_weights=[0.4, 0.3, 0.3]
        )
    )
    
    # Create optimization integration
    optimizer = NASOptimizationIntegration(config)
    
    # Generate sample market data
    market_data = generate_sample_market_data()
    features = generate_sample_features()
    timestamps = generate_sample_timestamps()
    
    # Create base NAS configuration
    nas_config = NASClusteringConfig.create_short_term_trading_config()
    
    # Run optimization
    results = optimizer.run_optimization(
        market_data=market_data,
        features=features,
        timestamps=timestamps,
        nas_config=nas_config,
        save_path="nas_optimization_results/custom"
    )
    
    # Print results
    print(f"✅ Optimization completed in {results['execution_time']:.2f}s")
    print(f"📊 Best score: {results['optimization_results']['best_overall_score']:.4f}")
    print(f"🎯 Best phase: {results['optimization_results']['best_phase']}")
    print(f"📈 Recommendations: {results['recommendations']}")
    
    return results


def example_grid_utils_integration():
    """Example: Demonstrate grid utils integration."""
    print("📊 Example: Grid Utils Integration")
    
    # Import grid utilities
    from src.utils.ml_common.optimization.grid_utils import (
        build_coarse_grid_from_search_space,
        build_fine_grid_around_best
    )
    
    # Define search space
    search_space = {
        'architecture_depth': {
            'type': 'int',
            'low': 3,
            'high': 9
        },
        'hidden_units': {
            'type': 'int',
            'low': 32,
            'high': 256
        },
        'learning_rate': {
            'type': 'float',
            'low': 0.001,
            'high': 0.1,
            'log': True
        }
    }
    
    # Build coarse grid
    coarse_grid = build_coarse_grid_from_search_space(search_space, 8)
    print(f"📊 Coarse grid generated: {len(coarse_grid)} parameter combinations")
    
    # Simulate best parameters
    best_params = {
        'architecture_depth': 5,
        'hidden_units': 128,
        'learning_rate': 0.01
    }
    
    # Build fine grid around best parameters
    fine_grid = build_fine_grid_around_best(search_space, best_params, 5)
    print(f"🔍 Fine grid generated: {len(fine_grid)} parameter combinations")
    
    # Print sample parameters
    print("📋 Sample coarse grid parameters:")
    for i, params in enumerate(coarse_grid[:3]):
        print(f"  {i+1}: {params}")
    
    print("📋 Sample fine grid parameters:")
    for i, params in enumerate(fine_grid[:3]):
        print(f"  {i+1}: {params}")
    
    return coarse_grid, fine_grid


def example_matrix_operations_integration():
    """Example: Demonstrate matrix operations integration."""
    print("🔢 Example: Matrix Operations Integration")
    
    # Import matrix operations
    from src.utils.matrix_operations import UnifiedMatrixOperations
    
    # Create matrix operations instance
    matrix_ops = UnifiedMatrixOperations()
    
    # Generate sample matrices
    A = np.random.rand(100, 50)
    B = np.random.rand(50, 80)
    
    # Test matrix multiplication
    print("🧮 Testing matrix multiplication...")
    start_time = time.time()
    C = matrix_ops.matrix_multiply(A, B)
    multiplication_time = time.time() - start_time
    print(f"✅ Matrix multiplication completed in {multiplication_time:.4f}s")
    print(f"📊 Result shape: {C.shape}")
    
    # Test matrix inverse
    print("🧮 Testing matrix inverse...")
    square_matrix = np.random.rand(20, 20)
    start_time = time.time()
    inverse_matrix = matrix_ops.matrix_inverse(square_matrix)
    inverse_time = time.time() - start_time
    print(f"✅ Matrix inverse completed in {inverse_time:.4f}s")
    print(f"📊 Result shape: {inverse_matrix.shape}")
    
    # Test matrix decomposition
    print("🧮 Testing matrix decomposition...")
    start_time = time.time()
    U, S, V = matrix_ops.matrix_decomposition(square_matrix, method='svd')
    decomposition_time = time.time() - start_time
    print(f"✅ Matrix decomposition completed in {decomposition_time:.4f}s")
    print(f"📊 U shape: {U.shape}, S shape: {S.shape}, V shape: {V.shape}")
    
    return {
        'multiplication_time': multiplication_time,
        'inverse_time': inverse_time,
        'decomposition_time': decomposition_time
    }


def example_hardware_optimization_integration():
    """Example: Demonstrate hardware optimization integration."""
    print("🖥️ Example: Hardware Optimization Integration")
    
    # Import hardware optimization
    from src.utils.hardware.unified_hardware_manager import (
        UnifiedHardwareManager, HardwareConfig, WorkloadType, OptimizationLevel
    )
    
    # Create hardware configuration
    hardware_config = HardwareConfig(
        cpu_optimization_level=OptimizationLevel.BALANCED,
        gpu_optimization_level=OptimizationLevel.BALANCED,
        memory_optimization_level=OptimizationLevel.BALANCED,
        memory_limit_gb=8.0,
        enable_adaptive_optimization=True,
        learning_enabled=True,
        auto_tuning_enabled=True
    )
    
    # Create hardware manager
    hardware_manager = UnifiedHardwareManager(hardware_config)
    
    # Start optimization
    print("🚀 Starting hardware optimization...")
    hardware_manager.start_optimization(
        workload_type=WorkloadType.ML_TRAINING,
        optimization_level=OptimizationLevel.BALANCED
    )
    
    # Simulate some work
    print("⚙️ Simulating ML training workload...")
    time.sleep(2)  # Simulate work
    
    # Get performance metrics
    metrics = hardware_manager.get_performance_metrics()
    print(f"📊 Hardware metrics: {metrics}")
    
    # Stop optimization
    hardware_manager.stop_optimization()
    print("✅ Hardware optimization stopped")
    
    return metrics


def generate_sample_market_data() -> pd.DataFrame:
    """Generate sample market data for testing."""
    np.random.seed(42)
    n_samples = 1000
    
    # Generate OHLCV data
    data = {
        'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='15T'),
        'open': 100 + np.cumsum(np.random.randn(n_samples) * 0.1),
        'high': 100 + np.cumsum(np.random.randn(n_samples) * 0.1) + np.random.rand(n_samples) * 2,
        'low': 100 + np.cumsum(np.random.randn(n_samples) * 0.1) - np.random.rand(n_samples) * 2,
        'close': 100 + np.cumsum(np.random.randn(n_samples) * 0.1),
        'volume': np.random.randint(1000, 10000, n_samples)
    }
    
    # Ensure high >= max(open, close) and low <= min(open, close)
    for i in range(n_samples):
        data['high'][i] = max(data['open'][i], data['close'][i], data['high'][i])
        data['low'][i] = min(data['open'][i], data['close'][i], data['low'][i])
    
    return pd.DataFrame(data)


def generate_sample_features() -> np.ndarray:
    """Generate sample features for testing."""
    np.random.seed(42)
    n_samples = 1000
    n_features = 50
    
    # Generate random features
    features = np.random.randn(n_samples, n_features)
    
    # Add some structure
    features[:, 0] = np.sin(np.linspace(0, 4*np.pi, n_samples))  # Sine wave
    features[:, 1] = np.cos(np.linspace(0, 4*np.pi, n_samples))  # Cosine wave
    features[:, 2] = np.linspace(0, 1, n_samples)  # Linear trend
    
    return features


def generate_sample_timestamps() -> np.ndarray:
    """Generate sample timestamps for testing."""
    return pd.date_range('2024-01-01', periods=1000, freq='15T').values


def run_all_examples():
    """Run all optimization examples."""
    print("🚀 Running All NAS Optimization Examples")
    print("=" * 60)
    
    # Example 1: Short-term trading optimization
    print("\n1. Short-term Trading Optimization")
    print("-" * 40)
    example_short_term_trading_optimization()
    
    # Example 2: High performance optimization
    print("\n2. High Performance Optimization")
    print("-" * 40)
    example_high_performance_optimization()
    
    # Example 3: Quick test optimization
    print("\n3. Quick Test Optimization")
    print("-" * 40)
    example_quick_test_optimization()
    
    # Example 4: Custom optimization
    print("\n4. Custom Optimization")
    print("-" * 40)
    example_custom_optimization()
    
    # Example 5: Grid utils integration
    print("\n5. Grid Utils Integration")
    print("-" * 40)
    example_grid_utils_integration()
    
    # Example 6: Matrix operations integration
    print("\n6. Matrix Operations Integration")
    print("-" * 40)
    example_matrix_operations_integration()
    
    # Example 7: Hardware optimization integration
    print("\n7. Hardware Optimization Integration")
    print("-" * 40)
    example_hardware_optimization_integration()
    
    print("\n✅ All examples completed successfully!")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Run all examples
    run_all_examples()