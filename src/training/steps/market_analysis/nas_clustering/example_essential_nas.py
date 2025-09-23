"""
Essential NAS Example Usage

This module demonstrates how to use the essential NAS clustering system
for true Neural Architecture Search without unnecessary complexity.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any

# Import essential NAS components
from .core.essential_nas_clusterer import EssentialNASClusterer, EssentialNASResult
from .core.nas_search.search_space import get_default_search_space

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_data(n_samples: int = 500, n_features: int = 10) -> tuple:
    """Create sample data for NAS testing."""
    np.random.seed(42)
    
    # Generate synthetic data with different patterns
    timestamps = pd.date_range('2024-01-01', periods=n_samples, freq='15min')
    
    # Create different regimes
    regime_length = n_samples // 3
    regimes = []
    
    for i in range(3):
        if i == 0:  # High volatility regime
            regime_data = np.random.randn(regime_length, n_features) * 2.0
        elif i == 1:  # Low volatility regime
            regime_data = np.random.randn(regime_length, n_features) * 0.5
        else:  # Trending regime
            trend = np.linspace(0, 3, regime_length).reshape(-1, 1)
            regime_data = np.random.randn(regime_length, n_features) * 0.8 + trend
    
        regimes.append(regime_data)
    
    # Combine regimes
    data = np.vstack(regimes)
    
    # Generate labels based on regimes
    labels = np.concatenate([
        np.full(regime_length, 0),  # High volatility
        np.full(regime_length, 1),  # Low volatility
        np.full(regime_length, 2)   # Trending
    ])
    
    return data, labels, timestamps


def example_basic_nas():
    """Example of basic essential NAS."""
    logger.info("🚀 Example 1: Basic Essential NAS")
    
    # Create sample data
    data, labels, timestamps = create_sample_data(n_samples=400, n_features=8)
    
    # Initialize essential NAS clusterer
    clusterer = EssentialNASClusterer(
        population_size=20,
        generations=30,
        enable_multi_objective=True
    )
    
    # Perform NAS search
    result = clusterer.search(data, labels)
    
    # Print results
    clusterer.print_search_results(result)
    
    return result


def example_custom_search_space():
    """Example with custom search space."""
    logger.info("🚀 Example 2: Custom Search Space")
    
    # Create sample data
    data, labels, timestamps = create_sample_data(n_samples=600, n_features=12)
    
    # Get default search space and customize it
    search_space = get_default_search_space()
    
    # Modify constraints
    search_space.constraints.max_layers = 6
    search_space.constraints.max_conv_layers = 2
    search_space.constraints.max_rnn_layers = 2
    search_space.constraints.max_total_parameters = 300000
    
    logger.info("🔧 Custom search space constraints:")
    logger.info(f"   Max layers: {search_space.constraints.max_layers}")
    logger.info(f"   Max conv layers: {search_space.constraints.max_conv_layers}")
    logger.info(f"   Max RNN layers: {search_space.constraints.max_rnn_layers}")
    logger.info(f"   Max parameters: {search_space.constraints.max_total_parameters}")
    
    # Initialize clusterer with custom search space
    clusterer = EssentialNASClusterer(
        search_space=search_space,
        population_size=25,
        generations=40,
        enable_multi_objective=True
    )
    
    # Perform NAS search
    result = clusterer.search(data, labels)
    
    # Print results
    clusterer.print_search_results(result)
    
    return result


def example_single_objective_vs_multi_objective():
    """Example comparing single vs multi-objective optimization."""
    logger.info("🚀 Example 3: Single vs Multi-Objective Comparison")
    
    # Create sample data
    data, labels, timestamps = create_sample_data(n_samples=500, n_features=10)
    
    # Test single objective (no multi-objective optimization)
    logger.info("📊 Running single-objective NAS...")
    clusterer_single = EssentialNASClusterer(
        population_size=20,
        generations=25,
        enable_multi_objective=False
    )
    
    result_single = clusterer_single.search(data, labels)
    
    # Test multi-objective optimization
    logger.info("🎯 Running multi-objective NAS...")
    clusterer_multi = EssentialNASClusterer(
        population_size=20,
        generations=25,
        enable_multi_objective=True
    )
    
    result_multi = clusterer_multi.search(data, labels)
    
    # Compare results
    logger.info("📊 Comparison Results:")
    logger.info(f"   Single-objective:")
    logger.info(f"     Execution time: {result_single.execution_time:.2f}s")
    logger.info(f"     Best fitness: {result_single.best_architecture.fitness_score:.4f}")
    logger.info(f"     Success: {result_single.success}")
    
    logger.info(f"   Multi-objective:")
    logger.info(f"     Execution time: {result_multi.execution_time:.2f}s")
    logger.info(f"     Best fitness: {result_multi.best_architecture.fitness_score:.4f}")
    logger.info(f"     Pareto solutions: {len(result_multi.pareto_frontier.solutions) if result_multi.pareto_frontier else 0}")
    logger.info(f"     Success: {result_multi.success}")
    
    return result_single, result_multi


def example_architecture_analysis():
    """Example of detailed architecture analysis."""
    logger.info("🚀 Example 4: Architecture Analysis")
    
    # Create sample data
    data, labels, timestamps = create_sample_data(n_samples=700, n_features=15)
    
    # Initialize clusterer
    clusterer = EssentialNASClusterer(
        population_size=30,
        generations=35,
        enable_multi_objective=True
    )
    
    # Perform NAS search
    result = clusterer.search(data, labels)
    
    if result.success:
        # Get detailed architecture information
        arch_info = clusterer.get_best_architecture_info(result)
        
        logger.info("🏗️ Best Architecture Analysis:")
        logger.info(f"   Fitness score: {arch_info['fitness_score']:.4f}")
        logger.info(f"   Parameters: {arch_info['parameters_count']:,}")
        logger.info(f"   Layers: {len(arch_info['layers'])}")
        logger.info(f"   Connections: {len(arch_info['connections'])}")
        
        # Analyze layer types
        layer_types = [layer['type'] for layer in arch_info['layers']]
        layer_type_counts = {}
        for layer_type in layer_types:
            layer_type_counts[layer_type] = layer_type_counts.get(layer_type, 0) + 1
        
        logger.info("   Layer type distribution:")
        for layer_type, count in layer_type_counts.items():
            logger.info(f"     {layer_type}: {count}")
        
        # Analyze connections
        connection_types = [conn['type'] for conn in arch_info['connections']]
        connection_type_counts = {}
        for conn_type in connection_types:
            connection_type_counts[conn_type] = connection_type_counts.get(conn_type, 0) + 1
        
        logger.info("   Connection type distribution:")
        for conn_type, count in connection_type_counts.items():
            logger.info(f"     {conn_type}: {count}")
        
        # Multi-objective analysis
        if result.pareto_frontier:
            pareto_summary = clusterer.get_pareto_summary(result)
            
            logger.info("🎯 Multi-Objective Analysis:")
            logger.info(f"   Total Pareto solutions: {pareto_summary.get('total_solutions', 0)}")
            logger.info(f"   Number of fronts: {pareto_summary.get('num_fronts', 0)}")
            
            if 'objective_ranges' in pareto_summary:
                logger.info("   Objective ranges:")
                for obj_name, obj_range in pareto_summary['objective_ranges'].items():
                    logger.info(f"     {obj_name}: {obj_range['min']:.3f} - {obj_range['max']:.3f}")
    
    return result


def example_performance_benchmark():
    """Example of performance benchmarking."""
    logger.info("🚀 Example 5: Performance Benchmark")
    
    # Test different population sizes
    population_sizes = [10, 20, 30]
    results = {}
    
    for pop_size in population_sizes:
        logger.info(f"📊 Testing population size: {pop_size}")
        
        # Create data
        data, labels, timestamps = create_sample_data(n_samples=300, n_features=8)
        
        # Initialize clusterer
        clusterer = EssentialNASClusterer(
            population_size=pop_size,
            generations=20,
            enable_multi_objective=False  # Disable for faster benchmarking
        )
        
        # Run search
        result = clusterer.search(data, labels)
        
        # Store results
        results[pop_size] = {
            'execution_time': result.execution_time,
            'best_fitness': result.best_architecture.fitness_score if result.best_architecture else 0.0,
            'success': result.success,
            'parameters_count': result.best_architecture.parameters_count if result.best_architecture else 0
        }
        
        logger.info(f"   Execution time: {result.execution_time:.2f}s")
        logger.info(f"   Best fitness: {result.best_architecture.fitness_score:.4f}")
        logger.info(f"   Parameters: {result.best_architecture.parameters_count:,}")
    
    # Display benchmark summary
    logger.info("📊 Performance Benchmark Summary:")
    for pop_size, metrics in results.items():
        logger.info(f"   Population {pop_size}: "
                   f"{metrics['execution_time']:.2f}s, "
                   f"fitness={metrics['best_fitness']:.4f}, "
                   f"params={metrics['parameters_count']:,}")
    
    return results


def main():
    """Run all examples."""
    logger.info("🚀 Essential NAS Examples")
    logger.info("=" * 60)
    
    try:
        # Run examples
        example_basic_nas()
        logger.info("\n" + "=" * 60)
        
        example_custom_search_space()
        logger.info("\n" + "=" * 60)
        
        example_single_objective_vs_multi_objective()
        logger.info("\n" + "=" * 60)
        
        example_architecture_analysis()
        logger.info("\n" + "=" * 60)
        
        example_performance_benchmark()
        
        logger.info("\n✅ All essential NAS examples completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Example execution failed: {e}")
        raise


if __name__ == "__main__":
    main()