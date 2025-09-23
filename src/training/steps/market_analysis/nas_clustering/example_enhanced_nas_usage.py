"""
Enhanced NAS Clustering Example Usage

This module demonstrates how to use the enhanced NAS clustering system with true
Neural Architecture Search capabilities for regime detection.
"""

import numpy as np
import pandas as pd
import time
import logging
from typing import Dict, Any, Optional

# Import enhanced NAS components
from .core.enhanced_nas_clusterer import EnhancedNASClusterer, EnhancedNASClusteringResult
from .core.nas_config import NASClusteringConfig, NASArchitectureType
from .core.nas_search.search_space import (
    get_volatility_regime_search_space,
    get_trend_regime_search_space,
    get_volume_regime_search_space,
    get_hybrid_regime_search_space
)

# Import existing utilities
from src.utils.matrix_operations import UnifiedMatrixOperations
from src.utils.hardware.unified_hardware_manager import (
    UnifiedHardwareManager, HardwareConfig, WorkloadType, OptimizationLevel
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_market_data(n_samples: int = 1000, n_features: int = 20) -> tuple:
    """Create sample market data for testing."""
    np.random.seed(42)
    
    # Generate synthetic market data with different regimes
    timestamps = pd.date_range('2024-01-01', periods=n_samples, freq='15min')
    
    # Create different market regimes
    regime_length = n_samples // 4
    regimes = []
    
    for i in range(4):
        if i == 0:  # High volatility regime
            regime_data = np.random.randn(regime_length, n_features) * 2.0
        elif i == 1:  # Low volatility regime
            regime_data = np.random.randn(regime_length, n_features) * 0.5
        elif i == 2:  # Trending regime
            trend = np.linspace(0, 5, regime_length).reshape(-1, 1)
            regime_data = np.random.randn(regime_length, n_features) * 0.8 + trend
        else:  # Mean-reverting regime
            regime_data = np.random.randn(regime_length, n_features) * 1.2
    
        regimes.append(regime_data)
    
    # Combine regimes
    market_data = np.vstack(regimes)
    
    return market_data, timestamps


def example_basic_enhanced_nas():
    """Example of basic enhanced NAS clustering."""
    logger.info("🚀 Example 1: Basic Enhanced NAS Clustering")
    
    # Create sample data
    market_data, timestamps = create_sample_market_data(n_samples=500, n_features=15)
    
    # Configure enhanced NAS
    config = NASClusteringConfig(
        timeframe="15m",
        n_regimes=8,
        nas_architecture_type=NASArchitectureType.HYBRID,
        enable_true_nas=True,
        nas_generations=20,
        nas_population_size=25,
        enable_multi_objective=True,
        enable_hardware_acceleration=True,
        economic_significance_threshold=0.7,
        trading_viability_threshold=0.6
    )
    
    # Initialize enhanced NAS clusterer
    clusterer = EnhancedNASClusterer(config)
    
    # Perform clustering
    start_time = time.time()
    result = clusterer.cluster(
        data=market_data,
        timestamps=timestamps.values,
        optimize_parameters=True,
        generate_report=True
    )
    execution_time = time.time() - start_time
    
    # Display results
    logger.info(f"✅ Basic Enhanced NAS completed in {execution_time:.2f}s")
    logger.info(f"📊 Regimes detected: {len(np.unique(result.labels))}")
    logger.info(f"🎯 Quality metrics: {result.quality_metrics}")
    logger.info(f"💰 Economic significance: {np.mean(result.economic_significance_scores):.3f}")
    logger.info(f"📈 Trading viability: {np.mean(result.trading_viability_scores):.3f}")
    
    if result.best_architecture:
        logger.info(f"🧠 Best architecture layers: {len(result.best_architecture.layers)}")
        logger.info(f"🏆 Best architecture fitness: {result.best_architecture.fitness_score:.4f}")
    
    return result


def example_volatility_focused_nas():
    """Example of volatility-focused NAS clustering."""
    logger.info("🚀 Example 2: Volatility-Focused NAS Clustering")
    
    # Create high-volatility sample data
    market_data, timestamps = create_sample_market_data(n_samples=800, n_features=12)
    
    # Configure volatility-focused NAS
    config = NASClusteringConfig(
        timeframe="15m",
        n_regimes=6,
        nas_architecture_type=NASArchitectureType.VOLATILITY_FOCUSED,
        enable_true_nas=True,
        nas_generations=30,
        nas_population_size=30,
        enable_multi_objective=True,
        economic_significance_threshold=0.8,
        trading_viability_threshold=0.7
    )
    
    # Initialize enhanced NAS clusterer
    clusterer = EnhancedNASClusterer(config)
    
    # Perform clustering
    result = clusterer.cluster(
        data=market_data,
        timestamps=timestamps.values,
        optimize_parameters=True,
        generate_report=True
    )
    
    # Display results
    logger.info(f"✅ Volatility-Focused NAS completed")
    logger.info(f"📊 Volatility regimes detected: {len(np.unique(result.labels))}")
    logger.info(f"🎯 Volatility quality metrics: {result.quality_metrics}")
    
    if result.neural_network_performance:
        perf = result.neural_network_performance
        logger.info(f"🧠 Neural network accuracy: {perf.get('final_accuracy', 0.0):.4f}")
        logger.info(f"⏱️ Training time: {perf.get('training_time', 0.0):.2f}s")
    
    return result


def example_multi_objective_optimization():
    """Example of multi-objective NAS optimization."""
    logger.info("🚀 Example 3: Multi-Objective NAS Optimization")
    
    # Create comprehensive sample data
    market_data, timestamps = create_sample_market_data(n_samples=1200, n_features=18)
    
    # Configure multi-objective NAS
    config = NASClusteringConfig(
        timeframe="15m",
        n_regimes=10,
        nas_architecture_type=NASArchitectureType.HYBRID,
        enable_true_nas=True,
        nas_generations=40,
        nas_population_size=40,
        enable_multi_objective=True,
        economic_significance_threshold=0.75,
        trading_viability_threshold=0.65
    )
    
    # Initialize enhanced NAS clusterer
    clusterer = EnhancedNASClusterer(config)
    
    # Perform clustering with multi-objective optimization
    result = clusterer.cluster(
        data=market_data,
        timestamps=timestamps.values,
        optimize_parameters=True,
        generate_report=True
    )
    
    # Display multi-objective results
    logger.info(f"✅ Multi-Objective NAS completed")
    
    if result.multi_objective_results:
        multi_obj = result.multi_objective_results
        logger.info(f"🎯 Pareto solutions: {multi_obj.get('total_solutions', 0)}")
        logger.info(f"📊 Pareto fronts: {multi_obj.get('num_fronts', 0)}")
        
        if 'objective_ranges' in multi_obj:
            logger.info("📈 Objective ranges:")
            for obj_name, obj_range in multi_obj['objective_ranges'].items():
                logger.info(f"   {obj_name}: {obj_range['min']:.3f} - {obj_range['max']:.3f}")
    
    if result.pareto_frontier:
        best_solutions = result.pareto_frontier.get_best_solutions(3)
        logger.info(f"🏆 Top 3 Pareto solutions:")
        for i, solution in enumerate(best_solutions):
            logger.info(f"   Solution {i+1}: {solution.objectives}")
    
    return result


def example_comparison_traditional_vs_nas():
    """Example comparing traditional clustering vs enhanced NAS."""
    logger.info("🚀 Example 4: Traditional vs Enhanced NAS Comparison")
    
    # Create sample data
    market_data, timestamps = create_sample_market_data(n_samples=600, n_features=16)
    
    # Test traditional clustering
    logger.info("📊 Running traditional clustering...")
    config_traditional = NASClusteringConfig(
        timeframe="15m",
        n_regimes=6,
        enable_true_nas=False,  # Disable NAS
        enable_hardware_acceleration=False
    )
    
    clusterer_traditional = EnhancedNASClusterer(config_traditional)
    start_time = time.time()
    result_traditional = clusterer_traditional.cluster(market_data, timestamps.values)
    traditional_time = time.time() - start_time
    
    # Test enhanced NAS
    logger.info("🧠 Running enhanced NAS clustering...")
    config_nas = NASClusteringConfig(
        timeframe="15m",
        n_regimes=6,
        enable_true_nas=True,
        nas_generations=15,
        nas_population_size=20,
        enable_multi_objective=True,
        enable_hardware_acceleration=False
    )
    
    clusterer_nas = EnhancedNASClusterer(config_nas)
    start_time = time.time()
    result_nas = clusterer_nas.cluster(market_data, timestamps.values)
    nas_time = time.time() - start_time
    
    # Compare results
    logger.info("📊 Comparison Results:")
    logger.info(f"   Traditional clustering:")
    logger.info(f"     Execution time: {traditional_time:.2f}s")
    logger.info(f"     Silhouette score: {result_traditional.quality_metrics.get('silhouette_score', 0.0):.4f}")
    logger.info(f"     Regimes detected: {len(np.unique(result_traditional.labels))}")
    
    logger.info(f"   Enhanced NAS clustering:")
    logger.info(f"     Execution time: {nas_time:.2f}s")
    logger.info(f"     Silhouette score: {result_nas.quality_metrics.get('silhouette_score', 0.0):.4f}")
    logger.info(f"     Regimes detected: {len(np.unique(result_nas.labels))}")
    logger.info(f"     Economic significance: {np.mean(result_nas.economic_significance_scores):.3f}")
    logger.info(f"     Trading viability: {np.mean(result_nas.trading_viability_scores):.3f}")
    
    if result_nas.best_architecture:
        logger.info(f"     Best architecture fitness: {result_nas.best_architecture.fitness_score:.4f}")
    
    return result_traditional, result_nas


def example_custom_search_space():
    """Example of using custom search space."""
    logger.info("🚀 Example 5: Custom Search Space")
    
    # Create sample data
    market_data, timestamps = create_sample_market_data(n_samples=700, n_features=14)
    
    # Configure with custom search space
    config = NASClusteringConfig(
        timeframe="15m",
        n_regimes=7,
        nas_architecture_type=NASArchitectureType.TREND_FOCUSED,
        enable_true_nas=True,
        nas_generations=25,
        nas_population_size=35,
        enable_multi_objective=True
    )
    
    # Initialize clusterer
    clusterer = EnhancedNASClusterer(config)
    
    # Customize search space
    search_space = clusterer.search_space
    logger.info(f"📊 Original search space layer types: {len(search_space.available_layer_types)}")
    
    # Modify search space constraints
    search_space.constraints.max_layers = 8
    search_space.constraints.max_conv_layers = 2
    search_space.constraints.require_skip_connections = True
    
    logger.info("🔧 Modified search space constraints:")
    logger.info(f"   Max layers: {search_space.constraints.max_layers}")
    logger.info(f"   Max conv layers: {search_space.constraints.max_conv_layers}")
    logger.info(f"   Require skip connections: {search_space.constraints.require_skip_connections}")
    
    # Perform clustering with custom search space
    result = clusterer.cluster(
        data=market_data,
        timestamps=timestamps.values,
        optimize_parameters=True,
        generate_report=True
    )
    
    logger.info(f"✅ Custom search space NAS completed")
    logger.info(f"📊 Regimes detected: {len(np.unique(result.labels))}")
    
    if result.best_architecture:
        logger.info(f"🧠 Best architecture:")
        logger.info(f"   Layers: {len(result.best_architecture.layers)}")
        logger.info(f"   Connections: {len(result.best_architecture.connections)}")
        logger.info(f"   Parameters: {result.best_architecture.parameters_count}")
        logger.info(f"   Fitness: {result.best_architecture.fitness_score:.4f}")
    
    return result


def example_performance_benchmarking():
    """Example of performance benchmarking."""
    logger.info("🚀 Example 6: Performance Benchmarking")
    
    # Test different data sizes
    data_sizes = [200, 500, 1000]
    results = {}
    
    for size in data_sizes:
        logger.info(f"📊 Testing with {size} samples...")
        
        # Create data
        market_data, timestamps = create_sample_market_data(n_samples=size, n_features=12)
        
        # Configure NAS
        config = NASClusteringConfig(
            timeframe="15m",
            n_regimes=5,
            enable_true_nas=True,
            nas_generations=10,
            nas_population_size=15,
            enable_multi_objective=False,  # Disable for faster benchmarking
            enable_hardware_acceleration=False
        )
        
        # Run clustering
        clusterer = EnhancedNASClusterer(config)
        start_time = time.time()
        result = clusterer.cluster(market_data, timestamps.values)
        execution_time = time.time() - start_time
        
        # Store results
        results[size] = {
            'execution_time': execution_time,
            'silhouette_score': result.quality_metrics.get('silhouette_score', 0.0),
            'regimes_detected': len(np.unique(result.labels)),
            'success': result.success
        }
        
        logger.info(f"   Execution time: {execution_time:.2f}s")
        logger.info(f"   Silhouette score: {results[size]['silhouette_score']:.4f}")
        logger.info(f"   Regimes detected: {results[size]['regimes_detected']}")
    
    # Display benchmark summary
    logger.info("📊 Performance Benchmark Summary:")
    for size, metrics in results.items():
        logger.info(f"   {size} samples: {metrics['execution_time']:.2f}s, "
                   f"silhouette={metrics['silhouette_score']:.4f}, "
                   f"regimes={metrics['regimes_detected']}")
    
    return results


def main():
    """Run all examples."""
    logger.info("🚀 Enhanced NAS Clustering Examples")
    logger.info("=" * 50)
    
    try:
        # Run examples
        example_basic_enhanced_nas()
        logger.info("\n" + "=" * 50)
        
        example_volatility_focused_nas()
        logger.info("\n" + "=" * 50)
        
        example_multi_objective_optimization()
        logger.info("\n" + "=" * 50)
        
        example_comparison_traditional_vs_nas()
        logger.info("\n" + "=" * 50)
        
        example_custom_search_space()
        logger.info("\n" + "=" * 50)
        
        example_performance_benchmarking()
        
        logger.info("\n✅ All examples completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Example execution failed: {e}")
        raise


if __name__ == "__main__":
    main()