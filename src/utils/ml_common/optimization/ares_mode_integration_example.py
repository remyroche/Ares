"""
Ares Mode Integration Example

This example demonstrates how to use the enhanced HPO system with Ares launcher
execution modes for adaptive optimization intensity.

Key Features Demonstrated:
- Automatic mode detection from Ares launcher
- Mode-specific intensity scaling (light=10%, blank=25%, full=100%)
- Enhanced pruner with adaptive early stopping
- Integration with existing HPO workflows
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Callable

# Import the enhanced HPO system
from .consolidated_hpo import (
    ConsolidatedHPO, HPOConfig,
    create_ares_mode_hpo, create_auto_mode_hpo,
    create_bayesian_hpo, create_bohb_hpo
)

# Import enhanced pruner system
from .enhanced_pruner_system import (
    create_enhanced_pruner, get_ares_mode_from_context,
    AresExecutionMode, PrunerStrategy
)


def example_automatic_mode_detection():
    """Example: Automatic Ares mode detection and HPO configuration."""
    print("🔍 Example: Automatic Mode Detection")
    print("=" * 50)
    
    # This will automatically detect the Ares execution mode
    # from environment variables or launcher context
    hpo = create_auto_mode_hpo(
        strategy='bayesian',
        n_trials=100,  # Will be scaled based on detected mode
        enable_monitoring=True
    )
    
    print(f"Detected mode: {hpo.config.ares_execution_mode}")
    print(f"Scaled trials: {hpo.config.n_trials}")
    print(f"Startup trials: {hpo.config.n_startup_trials}")
    print()


def example_manual_mode_specification():
    """Example: Manual specification of Ares execution modes."""
    print("🎯 Example: Manual Mode Specification")
    print("=" * 50)
    
    modes = ['light', 'blank', 'full']
    
    for mode in modes:
        hpo = create_ares_mode_hpo(
            ares_mode=mode,
            strategy='bayesian',
            n_trials=100,
            enable_monitoring=True
        )
        
        print(f"Mode: {mode}")
        print(f"  Trials: {hpo.config.n_trials}")
        print(f"  Startup trials: {hpo.config.n_startup_trials}")
        print(f"  Timeout: {hpo.config.timeout}")
        print()


def example_enhanced_pruner_usage():
    """Example: Using enhanced pruner directly."""
    print("⚡ Example: Enhanced Pruner Usage")
    print("=" * 50)
    
    # Create enhanced pruner for different modes
    for mode in ['light', 'blank', 'full']:
        pruner = create_enhanced_pruner(
            ares_mode=mode,
            strategy='adaptive',
            base_patience=10,
            improvement_threshold=0.001
        )
        
        print(f"Mode: {mode}")
        print(f"  Strategy: {pruner.config.strategy.value}")
        print(f"  Patience: {pruner.config.base_patience}")
        print(f"  Threshold: {pruner.config.improvement_threshold:.6f}")
        print()


def example_optimization_with_mode_scaling():
    """Example: Complete optimization workflow with mode scaling."""
    print("🚀 Example: Complete Optimization Workflow")
    print("=" * 50)
    
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(1000, 20)
    y = np.random.randn(1000)
    
    # Define a simple model factory
    def model_factory(**params):
        from sklearn.linear_model import Ridge
        return Ridge(**params)
    
    # Define search space
    search_space = {
        'alpha': {'type': 'float', 'low': 0.001, 'high': 10.0, 'log': True},
        'fit_intercept': {'type': 'categorical', 'choices': [True, False]},
        'normalize': {'type': 'categorical', 'choices': [True, False]}
    }
    
    # Test different modes
    for mode in ['light', 'blank', 'full']:
        print(f"\n--- Testing {mode.upper()} mode ---")
        
        # Create HPO for specific mode
        hpo = create_ares_mode_hpo(
            ares_mode=mode,
            strategy='bayesian',
            n_trials=50,  # Will be scaled
            enable_monitoring=True,
            timeout=60  # Will be scaled
        )
        
        print(f"Configured trials: {hpo.config.n_trials}")
        print(f"Configured timeout: {hpo.config.timeout}s")
        
        # Run optimization (in real usage, this would be the actual optimization)
        print(f"Would run {hpo.config.n_trials} trials with {hpo.config.ares_execution_mode} mode scaling")
        print()


def example_ares_launcher_integration():
    """Example: Integration with Ares launcher workflow."""
    print("🔗 Example: Ares Launcher Integration")
    print("=" * 50)
    
    # Simulate Ares launcher context
    import os
    
    # Set environment variable to simulate launcher context
    os.environ['ARES_EXECUTION_MODE'] = 'light'
    
    # This would be called from within an Ares step
    def run_hpo_in_step():
        # Auto-detect mode from launcher context
        hpo = create_auto_mode_hpo(
            strategy='bayesian',
            n_trials=100,
            enable_monitoring=True
        )
        
        print(f"Step detected mode: {hpo.config.ares_execution_mode}")
        print(f"Step will use {hpo.config.n_trials} trials")
        
        return hpo
    
    # Simulate step execution
    hpo = run_hpo_in_step()
    
    # Clean up
    if 'ARES_EXECUTION_MODE' in os.environ:
        del os.environ['ARES_EXECUTION_MODE']
    
    print()


def example_pruning_statistics():
    """Example: Accessing pruning statistics."""
    print("📊 Example: Pruning Statistics")
    print("=" * 50)
    
    # Create HPO with enhanced pruner
    hpo = create_ares_mode_hpo(
        ares_mode='full',
        strategy='bayesian',
        n_trials=50,
        enable_monitoring=True
    )
    
    # In a real optimization, you would run:
    # result = hpo.optimize(model_factory, X, y, search_space, "test_model")
    
    # The result would contain pruning statistics:
    print("After optimization, you can access:")
    print("- result.convergence_info['pruning_rate']")
    print("- result.convergence_info['total_trials']")
    print("- result.convergence_info['pruned_trials']")
    print("- result.convergence_info['strategy']")
    print("- result.convergence_info['ares_mode']")
    print()


def example_migration_from_old_system():
    """Example: Migrating from old HPO system."""
    print("🔄 Example: Migration from Old System")
    print("=" * 50)
    
    # Old way (still works)
    old_hpo = create_bayesian_hpo(n_trials=100)
    print("Old system:")
    print(f"  Trials: {old_hpo.config.n_trials}")
    print(f"  Pruner: MedianPruner (basic)")
    
    # New way with mode integration
    new_hpo = create_auto_mode_hpo(strategy='bayesian', n_trials=100)
    print("\nNew system:")
    print(f"  Trials: {new_hpo.config.n_trials} (scaled by mode)")
    print(f"  Mode: {new_hpo.config.ares_execution_mode}")
    print(f"  Pruner: EnhancedPruner (adaptive)")
    print()


if __name__ == "__main__":
    """Run all examples."""
    print("🎯 Ares Mode Integration Examples")
    print("=" * 60)
    print()
    
    example_automatic_mode_detection()
    example_manual_mode_specification()
    example_enhanced_pruner_usage()
    example_optimization_with_mode_scaling()
    example_ares_launcher_integration()
    example_pruning_statistics()
    example_migration_from_old_system()
    
    print("✅ All examples completed!")
    print("\nKey Benefits:")
    print("• Automatic intensity scaling based on Ares execution mode")
    print("• Enhanced early stopping with adaptive pruning")
    print("• Better resource utilization for different use cases")
    print("• Seamless integration with existing Ares launcher workflow")
    print("• Detailed pruning statistics and convergence tracking")
    print("• Light mode: 5% intensity (5 trials, 5 patience, 2x threshold, 20% timeout)")