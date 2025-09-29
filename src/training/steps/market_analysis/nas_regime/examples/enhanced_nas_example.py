"""
Enhanced NAS Example

This example demonstrates how to use the Enhanced Neural Architecture Search system
with advanced architectures and search strategies for regime detection.
"""

import numpy as np
import torch
import logging
from pathlib import Path
import sys

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.append(str(project_root))

from src.training.steps.market_analysis.nas_regime.core.enhanced_nas_integration import (
    EnhancedNASConfig, EnhancedNASSystem, create_enhanced_nas_system,
    quick_enhanced_nas_search, compare_all_strategies, benchmark_all_architectures
)

from src.training.steps.market_analysis.nas_regime.core.advanced_neural_architectures import (
    AdvancedArchitectureConfig, ArchitectureType
)

from src.training.steps.market_analysis.nas_regime.core.enhanced_search_strategies import (
    SearchStrategyConfig, SearchStrategyType
)

from src.utils.tprint import tprint, tprint_success, tprint_info, tprint_warning

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_1_basic_enhanced_nas():
    """Example 1: Basic Enhanced NAS with Transformer Architecture and RL Search."""
    tprint("🚀 Example 1: Basic Enhanced NAS", color="cyan", bold=True)
    
    # Create configuration
    config = EnhancedNASConfig()
    
    # Configure architecture
    config.architecture_config.architecture_type = ArchitectureType.TRANSFORMER_REGIME
    config.architecture_config.input_dim = 64
    config.architecture_config.hidden_dim = 256
    config.architecture_config.num_heads = 8
    config.architecture_config.num_layers = 6
    config.architecture_config.num_regimes = 8
    
    # Configure search strategy
    config.search_config.strategy_type = SearchStrategyType.REINFORCEMENT_LEARNING
    config.search_config.rl_learning_rate = 0.001
    config.search_config.rl_gamma = 0.99
    
    # Set search parameters
    config.max_search_iterations = 100
    config.output_dir = "examples/enhanced_nas_results"
    
    tprint("📊 Configuration:", color="blue")
    tprint(f"   Architecture: {config.architecture_config.architecture_type.value}")
    tprint(f"   Search Strategy: {config.search_config.strategy_type.value}")
    tprint(f"   Max Iterations: {config.max_search_iterations}")
    
    # Create and run Enhanced NAS system
    nas_system = create_enhanced_nas_system(config)
    result = nas_system.search()
    
    # Display results
    if result.success:
        tprint_success(f"✅ Enhanced NAS completed successfully!")
        tprint(f"   Best Performance: {result.best_performance:.4f}")
        tprint(f"   Execution Time: {result.execution_time:.2f}s")
        tprint(f"   Strategy Used: {result.search_strategy_used}")
        tprint(f"   Architecture Info: {result.architecture_info}")
    else:
        tprint_warning(f"❌ Enhanced NAS failed: {result.error_message}")
    
    return result


def example_2_hybrid_search():
    """Example 2: Hybrid Search Strategy combining multiple approaches."""
    tprint("🚀 Example 2: Hybrid Search Strategy", color="cyan", bold=True)
    
    # Create configuration for hybrid search
    config = EnhancedNASConfig()
    config.architecture_config.architecture_type = ArchitectureType.HYBRID_TRANSFORMER_GNN
    config.search_config.strategy_type = SearchStrategyType.HYBRID_SEARCH
    config.max_search_iterations = 50
    config.output_dir = "examples/hybrid_search_results"
    
    tprint("📊 Configuration:", color="blue")
    tprint(f"   Architecture: {config.architecture_config.architecture_type.value}")
    tprint(f"   Search Strategy: Hybrid (Multiple Strategies)")
    
    # Create and run system
    nas_system = create_enhanced_nas_system(config)
    result = nas_system.search()
    
    # Display results
    if result.success:
        tprint_success(f"✅ Hybrid search completed successfully!")
        tprint(f"   Best Performance: {result.best_performance:.4f}")
        tprint(f"   Execution Time: {result.execution_time:.2f}s")
    else:
        tprint_warning(f"❌ Hybrid search failed: {result.error_message}")
    
    return result


def example_3_strategy_comparison():
    """Example 3: Compare multiple search strategies."""
    tprint("🚀 Example 3: Strategy Comparison", color="cyan", bold=True)
    
    # Quick comparison of all strategies
    tprint("🔍 Comparing all search strategies...", color="yellow")
    
    results = compare_all_strategies(max_iterations=30)
    
    tprint("📊 Strategy Comparison Results:", color="blue")
    for strategy_name, result in results.items():
        if result.success:
            tprint(f"   {strategy_name}: {result.best_performance:.4f} (Time: {result.execution_time:.2f}s)")
        else:
            tprint(f"   {strategy_name}: Failed - {result.error_message}")
    
    # Find best strategy
    best_strategy = max(results.items(), key=lambda x: x[1].best_performance if x[1].success else 0)
    tprint_success(f"🏆 Best Strategy: {best_strategy[0]} with performance {best_strategy[1].best_performance:.4f}")
    
    return results


def example_4_architecture_benchmark():
    """Example 4: Benchmark different architecture types."""
    tprint("🚀 Example 4: Architecture Benchmark", color="cyan", bold=True)
    
    # Benchmark all architecture types
    tprint("🔍 Benchmarking all architecture types...", color="yellow")
    
    results = benchmark_all_architectures()
    
    tprint("📊 Architecture Benchmark Results:", color="blue")
    for arch_name, performance in results.items():
        tprint(f"   {arch_name}: {performance:.4f}")
    
    # Find best architecture
    best_architecture = max(results.items(), key=lambda x: x[1])
    tprint_success(f"🏆 Best Architecture: {best_architecture[0]} with performance {best_architecture[1]:.4f}")
    
    return results


def example_5_progressive_search():
    """Example 5: Progressive Architecture Search."""
    tprint("🚀 Example 5: Progressive Architecture Search", color="cyan", bold=True)
    
    # Create configuration for progressive search
    config = EnhancedNASConfig()
    config.architecture_config.architecture_type = ArchitectureType.TRANSFORMER_REGIME
    config.search_config.strategy_type = SearchStrategyType.PROGRESSIVE_SEARCH
    config.search_config.progressive_initial_ops = 2
    config.search_config.progressive_growth_rate = 1.5
    config.search_config.progressive_max_ops = 8
    config.max_search_iterations = 100
    config.output_dir = "examples/progressive_search_results"
    
    tprint("📊 Configuration:", color="blue")
    tprint(f"   Architecture: {config.architecture_config.architecture_type.value}")
    tprint(f"   Search Strategy: {config.search_config.strategy_type.value}")
    tprint(f"   Initial Ops: {config.search_config.progressive_initial_ops}")
    tprint(f"   Growth Rate: {config.search_config.progressive_growth_rate}")
    tprint(f"   Max Ops: {config.search_config.progressive_max_ops}")
    
    # Create and run system
    nas_system = create_enhanced_nas_system(config)
    result = nas_system.search()
    
    # Display results
    if result.success:
        tprint_success(f"✅ Progressive search completed successfully!")
        tprint(f"   Best Performance: {result.best_performance:.4f}")
        tprint(f"   Execution Time: {result.execution_time:.2f}s")
        tprint(f"   Search History Length: {len(result.search_history)}")
    else:
        tprint_warning(f"❌ Progressive search failed: {result.error_message}")
    
    return result


def example_6_multi_objective_search():
    """Example 6: Multi-Objective Evolutionary Search."""
    tprint("🚀 Example 6: Multi-Objective Evolutionary Search", color="cyan", bold=True)
    
    # Create configuration for multi-objective search
    config = EnhancedNASConfig()
    config.architecture_config.architecture_type = ArchitectureType.GRAPH_NEURAL_NETWORK
    config.search_config.strategy_type = SearchStrategyType.MULTI_OBJECTIVE_EVOLUTIONARY
    config.search_config.mo_population_size = 30
    config.search_config.mo_generations = 50
    config.search_config.mo_crossover_rate = 0.8
    config.search_config.mo_mutation_rate = 0.1
    config.max_search_iterations = 100
    config.output_dir = "examples/multi_objective_results"
    
    tprint("📊 Configuration:", color="blue")
    tprint(f"   Architecture: {config.architecture_config.architecture_type.value}")
    tprint(f"   Search Strategy: {config.search_config.strategy_type.value}")
    tprint(f"   Population Size: {config.search_config.mo_population_size}")
    tprint(f"   Generations: {config.search_config.mo_generations}")
    
    # Create and run system
    nas_system = create_enhanced_nas_system(config)
    result = nas_system.search()
    
    # Display results
    if result.success:
        tprint_success(f"✅ Multi-objective search completed successfully!")
        tprint(f"   Best Performance: {result.best_performance:.4f}")
        tprint(f"   Execution Time: {result.execution_time:.2f}s")
        tprint(f"   Search History Length: {len(result.search_history)}")
    else:
        tprint_warning(f"❌ Multi-objective search failed: {result.error_message}")
    
    return result


def example_7_quick_search():
    """Example 7: Quick Enhanced NAS search with default settings."""
    tprint("🚀 Example 7: Quick Enhanced NAS Search", color="cyan", bold=True)
    
    # Quick search with default settings
    tprint("🔍 Running quick search with default settings...", color="yellow")
    
    result = quick_enhanced_nas_search(
        architecture_type=ArchitectureType.TRANSFORMER_REGIME,
        search_strategy=SearchStrategyType.REINFORCEMENT_LEARNING,
        max_iterations=50
    )
    
    # Display results
    if result.success:
        tprint_success(f"✅ Quick search completed successfully!")
        tprint(f"   Best Performance: {result.best_performance:.4f}")
        tprint(f"   Execution Time: {result.execution_time:.2f}s")
        tprint(f"   Strategy Used: {result.search_strategy_used}")
    else:
        tprint_warning(f"❌ Quick search failed: {result.error_message}")
    
    return result


def run_all_examples():
    """Run all examples."""
    tprint("🎯 Running All Enhanced NAS Examples", color="green", bold=True)
    tprint("=" * 60, color="green")
    
    examples = [
        ("Basic Enhanced NAS", example_1_basic_enhanced_nas),
        ("Hybrid Search", example_2_hybrid_search),
        ("Strategy Comparison", example_3_strategy_comparison),
        ("Architecture Benchmark", example_4_architecture_benchmark),
        ("Progressive Search", example_5_progressive_search),
        ("Multi-Objective Search", example_6_multi_objective_search),
        ("Quick Search", example_7_quick_search)
    ]
    
    results = {}
    
    for example_name, example_func in examples:
        try:
            tprint(f"\n{'='*20} {example_name} {'='*20}", color="cyan")
            result = example_func()
            results[example_name] = result
            tprint(f"✅ {example_name} completed successfully", color="green")
        except Exception as e:
            tprint_warning(f"❌ {example_name} failed: {e}")
            results[example_name] = None
    
    # Summary
    tprint("\n" + "="*60, color="green")
    tprint("📊 SUMMARY", color="green", bold=True)
    tprint("="*60, color="green")
    
    successful_examples = sum(1 for r in results.values() if r is not None)
    total_examples = len(examples)
    
    tprint(f"Successful Examples: {successful_examples}/{total_examples}", color="blue")
    
    for example_name, result in results.items():
        if result is not None and hasattr(result, 'best_performance'):
            tprint(f"   {example_name}: {result.best_performance:.4f}", color="blue")
        else:
            tprint(f"   {example_name}: Failed", color="red")
    
    return results


if __name__ == "__main__":
    # Create output directories
    output_dirs = [
        "examples/enhanced_nas_results",
        "examples/hybrid_search_results", 
        "examples/progressive_search_results",
        "examples/multi_objective_results"
    ]
    
    for output_dir in output_dirs:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Run examples
    try:
        results = run_all_examples()
        tprint_success("🎉 All examples completed!")
    except KeyboardInterrupt:
        tprint_warning("⚠️ Examples interrupted by user")
    except Exception as e:
        tprint_warning(f"❌ Examples failed with error: {e}")
        logger.exception("Full error details:")