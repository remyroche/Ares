#!/usr/bin/env python3
"""
Example script demonstrating the Unified NAS-TAS System

This script shows how to use the unified system for both neural and tree
architecture search with comprehensive evaluation and regime detection.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Callable

# Import unified system components
from src.utils.nas_tas.shared_utils import (
    create_unified_system, UnifiedConfig, ArchitectureType, SearchStrategy,
    OptimizationAlgorithm, EvaluationType, RegimeDetectionMethod,
    ConfigFormat, create_default_config
)

def create_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """Create sample market data for demonstration."""
    np.random.seed(42)
    
    # Generate synthetic market data
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
    
    # Generate price data with trends and volatility
    price_base = 100
    returns = np.random.normal(0.001, 0.02, n_samples)  # Daily returns
    prices = price_base * np.exp(np.cumsum(returns))
    
    # Add some regime-like behavior
    regime_1 = prices[:n_samples//3] * 1.1  # Bull market
    regime_2 = prices[n_samples//3:2*n_samples//3] * 0.9  # Bear market
    regime_3 = prices[2*n_samples//3:] * 1.05  # Recovery
    
    prices = np.concatenate([regime_1, regime_2, regime_3])
    
    # Create OHLCV data
    data = pd.DataFrame({
        'date': dates,
        'open': prices * (1 + np.random.normal(0, 0.005, n_samples)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.01, n_samples))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.01, n_samples))),
        'close': prices,
        'volume': np.random.lognormal(10, 0.5, n_samples)
    })
    
    return data

def create_sample_search_space() -> Dict[str, Any]:
    """Create sample search space for architecture search."""
    return {
        'hidden_layers': [1, 2, 3, 4, 5],
        'hidden_size': [32, 64, 128, 256],
        'activation': ['relu', 'tanh', 'sigmoid'],
        'dropout_rate': (0.0, 0.5),
        'learning_rate': (0.001, 0.1),
        'batch_size': [16, 32, 64, 128]
    }

def create_sample_objective_function() -> Callable:
    """Create sample objective function for architecture evaluation."""
    def objective_function(architecture: Dict[str, Any]) -> float:
        # Simulate architecture evaluation
        # In practice, this would train and evaluate the actual architecture
        
        # Base score from architecture complexity
        complexity_score = 0.0
        
        # Hidden layers contribution
        complexity_score += architecture.get('hidden_layers', 1) * 0.1
        
        # Hidden size contribution
        complexity_score += architecture.get('hidden_size', 64) / 1000.0
        
        # Activation function contribution
        activation_scores = {'relu': 0.8, 'tanh': 0.7, 'sigmoid': 0.6}
        complexity_score += activation_scores.get(architecture.get('activation', 'relu'), 0.5)
        
        # Dropout contribution (higher dropout = better regularization)
        complexity_score += architecture.get('dropout_rate', 0.0) * 0.5
        
        # Learning rate contribution (optimal range)
        lr = architecture.get('learning_rate', 0.01)
        if 0.01 <= lr <= 0.1:
            complexity_score += 0.3
        else:
            complexity_score += 0.1
        
        # Batch size contribution
        batch_size = architecture.get('batch_size', 32)
        if batch_size in [32, 64]:
            complexity_score += 0.2
        else:
            complexity_score += 0.1
        
        # Add some randomness to simulate real evaluation
        complexity_score += np.random.normal(0, 0.05)
        
        return min(max(complexity_score, 0.0), 1.0)
    
    return objective_function

def create_sample_objective_functions() -> Dict[str, Callable]:
    """Create sample objective functions for multi-objective optimization."""
    def accuracy_objective(architecture: Dict[str, Any]) -> float:
        # Simulate accuracy evaluation
        base_accuracy = 0.7
        complexity_bonus = architecture.get('hidden_layers', 1) * 0.05
        size_bonus = min(architecture.get('hidden_size', 64) / 500.0, 0.2)
        return min(base_accuracy + complexity_bonus + size_bonus + np.random.normal(0, 0.02), 1.0)
    
    def efficiency_objective(architecture: Dict[str, Any]) -> float:
        # Simulate efficiency evaluation (inverse of complexity)
        complexity = architecture.get('hidden_layers', 1) * architecture.get('hidden_size', 64)
        efficiency = 1.0 / (1.0 + complexity / 1000.0)
        return efficiency + np.random.normal(0, 0.01)
    
    def profitability_objective(architecture: Dict[str, Any]) -> float:
        # Simulate profitability evaluation
        base_profitability = 0.6
        accuracy_bonus = accuracy_objective(architecture) * 0.3
        efficiency_bonus = efficiency_objective(architecture) * 0.2
        return min(base_profitability + accuracy_bonus + efficiency_bonus + np.random.normal(0, 0.03), 1.0)
    
    return {
        'accuracy': accuracy_objective,
        'efficiency': efficiency_objective,
        'profitability': profitability_objective
    }

def demonstrate_unified_system():
    """Demonstrate the unified NAS-TAS system."""
    print("🚀 Unified NAS-TAS System Demonstration")
    print("=" * 50)
    
    # Create sample data
    print("\n📊 Creating sample market data...")
    market_data = create_sample_data(1000)
    returns = market_data['close'].pct_change().dropna().values
    
    # Create sample search space and objective functions
    search_space = create_sample_search_space()
    objective_function = create_sample_objective_function()
    objective_functions = create_sample_objective_functions()
    
    print(f"   Market data shape: {market_data.shape}")
    print(f"   Returns shape: {returns.shape}")
    print(f"   Search space parameters: {len(search_space)}")
    
    # Create unified system
    print("\n🔧 Creating unified system...")
    system = create_unified_system()
    
    # Demonstrate search engine
    print("\n🔍 Demonstrating unified search engine...")
    search_result = system['search_engine'].search(search_space, objective_function)
    
    print(f"   Search completed: {search_result.success}")
    print(f"   Execution time: {search_result.execution_time:.2f}s")
    print(f"   Best architecture: {search_result.best_architecture}")
    print(f"   Best score: {max(search_result.best_scores.values()):.4f}")
    
    # Demonstrate multi-objective optimization
    print("\n🎯 Demonstrating multi-objective optimization...")
    parameter_bounds = {
        'hidden_layers': (1, 5),
        'hidden_size': (32, 256),
        'learning_rate': (0.001, 0.1),
        'dropout_rate': (0.0, 0.5)
    }
    
    optimization_result = system['optimizer'].optimize(objective_functions, parameter_bounds)
    
    print(f"   Optimization completed: {optimization_result.success}")
    print(f"   Execution time: {optimization_result.execution_time:.2f}s")
    print(f"   Best parameters: {optimization_result.best_parameters}")
    print(f"   Best scores: {optimization_result.best_scores}")
    print(f"   Pareto frontier size: {len(optimization_result.pareto_frontier)}")
    
    # Demonstrate regime detection
    print("\n🔍 Demonstrating regime detection...")
    
    # Create sample predictions (regime labels)
    n_regimes = 3
    regime_labels = np.random.randint(0, n_regimes, len(returns))
    
    regime_result = system['regime_detector'].detect_regimes(market_data)
    
    print(f"   Regime detection completed: {regime_result.success}")
    print(f"   Execution time: {regime_result.detection_time:.2f}s")
    print(f"   Detected regimes: {regime_result.n_regimes}")
    print(f"   Regime quality score: {regime_result.regime_quality_score:.4f}")
    
    # Demonstrate economic evaluation
    print("\n💰 Demonstrating economic evaluation...")
    
    evaluation_result = system['evaluator'].evaluate(regime_labels, market_data, returns)
    
    print(f"   Evaluation completed: {evaluation_result.success}")
    print(f"   Execution time: {evaluation_result.total_evaluation_time:.2f}s")
    print(f"   Economic significance: {evaluation_result.economic_metrics.economic_significance:.4f}")
    print(f"   Trading viability: {evaluation_result.economic_metrics.trading_viability:.4f}")
    print(f"   Overall score: {evaluation_result.economic_metrics.overall_score:.4f}")
    print(f"   Sharpe ratio: {evaluation_result.economic_metrics.sharpe_ratio:.4f}")
    print(f"   Max drawdown: {evaluation_result.economic_metrics.max_drawdown:.4f}")
    
    # Demonstrate utilities
    print("\n🛠️ Demonstrating utilities...")
    
    validation_result = system['utilities'].validate_data(
        market_data, 
        system['utilities'].__class__.__module__.split('.')[-1],  # Get data type
        ArchitectureType.HYBRID
    )
    
    print(f"   Data validation completed: {validation_result['is_valid']}")
    print(f"   Data quality score: {validation_result['data_quality_score']:.4f}")
    print(f"   Warnings: {len(validation_result['warnings'])}")
    print(f"   Errors: {len(validation_result['errors'])}")
    
    # Demonstrate configuration management
    print("\n⚙️ Demonstrating configuration management...")
    
    config = system['config']
    print(f"   System name: {config.system_name}")
    print(f"   Version: {config.version}")
    print(f"   Environment: {config.environment}")
    
    # Validate configuration
    validation_result = config.validate()
    print(f"   Configuration valid: {validation_result['is_valid']}")
    print(f"   Configuration warnings: {len(validation_result['warnings'])}")
    print(f"   Configuration errors: {len(validation_result['errors'])}")
    
    # Save configuration
    config.save_to_file('example_config.yaml', ConfigFormat.YAML)
    print("   Configuration saved to example_config.yaml")
    
    # Demonstrate performance monitoring
    print("\n📈 Performance Summary:")
    
    search_summary = system['search_engine'].get_performance_summary()
    print(f"   Search engine - Total searches: {search_summary['total_searches']}")
    
    optimization_summary = system['optimizer'].get_performance_summary()
    print(f"   Optimizer - Total optimizations: {optimization_summary['total_optimizations']}")
    
    evaluation_summary = system['evaluator'].get_performance_summary()
    print(f"   Evaluator - Total evaluations: {evaluation_summary['total_evaluations']}")
    
    regime_summary = system['regime_detector'].get_detection_summary()
    print(f"   Regime detector - Total detections: {regime_summary['total_detections']}")
    
    utilities_summary = system['utilities'].get_performance_summary()
    print(f"   Utilities - Total operations: {utilities_summary['total_operations']}")
    
    print("\n✅ Unified system demonstration completed successfully!")
    print("\n📋 Summary:")
    print(f"   - Search engine: {'✅' if search_result.success else '❌'}")
    print(f"   - Multi-objective optimization: {'✅' if optimization_result.success else '❌'}")
    print(f"   - Regime detection: {'✅' if regime_result.success else '❌'}")
    print(f"   - Economic evaluation: {'✅' if evaluation_result.success else '❌'}")
    print(f"   - Data validation: {'✅' if validation_result['is_valid'] else '❌'}")
    print(f"   - Configuration: {'✅' if config.validate()['is_valid'] else '❌'}")

def demonstrate_different_architectures():
    """Demonstrate the system with different architecture types."""
    print("\n🏗️ Architecture-Specific Demonstrations")
    print("=" * 50)
    
    # Create sample data
    market_data = create_sample_data(500)
    returns = market_data['close'].pct_change().dropna().values
    search_space = create_sample_search_space()
    objective_function = create_sample_objective_function()
    
    architectures = [
        (ArchitectureType.NEURAL, "Neural Architecture Search"),
        (ArchitectureType.TREE, "Tree Architecture Search"),
        (ArchitectureType.HYBRID, "Hybrid Architecture Search")
    ]
    
    for arch_type, arch_name in architectures:
        print(f"\n🔧 {arch_name}")
        print("-" * 30)
        
        # Create configuration for specific architecture
        config = create_default_config()
        config.search_config.architecture_type = arch_type
        config.evaluation_config.architecture_type = arch_type
        
        # Create system with architecture-specific configuration
        system = create_unified_system(config)
        
        # Perform search
        search_result = system['search_engine'].search(search_space, objective_function)
        
        print(f"   Architecture type: {arch_type.value}")
        print(f"   Search completed: {search_result.success}")
        print(f"   Execution time: {search_result.execution_time:.2f}s")
        print(f"   Best score: {max(search_result.best_scores.values()):.4f}")
        
        # Perform evaluation
        regime_labels = np.random.randint(0, 3, len(returns))
        evaluation_result = system['evaluator'].evaluate(regime_labels, market_data, returns)
        
        print(f"   Evaluation completed: {evaluation_result.success}")
        print(f"   Economic significance: {evaluation_result.economic_metrics.economic_significance:.4f}")
        print(f"   Trading viability: {evaluation_result.economic_metrics.trading_viability:.4f}")

if __name__ == "__main__":
    try:
        # Main demonstration
        demonstrate_unified_system()
        
        # Architecture-specific demonstrations
        demonstrate_different_architectures()
        
        print("\n🎉 All demonstrations completed successfully!")
        print("\n📚 For more information, see UNIFIED_SYSTEM_GUIDE.md")
        
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()