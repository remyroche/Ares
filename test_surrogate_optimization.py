#!/usr/bin/env python3
"""
Simple test script for surrogate optimization implementation.
"""

import asyncio
import numpy as np
from typing import Dict, Any

# Import the surrogate optimizer
from src.training.optimization.computational_optimization_manager import (
    SurrogateOptimizer,
    ComputationalOptimizationConfig,
)

# Import utility functions
from src.utils.logger import system_logger


def simple_test_objective(params: Dict[str, Any]) -> float:
    """Simple test objective function."""
    x = params.get('x', 0)
    y = params.get('y', 0)

    # Simple quadratic function with global maximum at (0, 0)
    return -(x**2 + y**2)


def create_simple_parameter_space() -> Dict[str, Any]:
    """Create simple parameter space for testing."""
    return {
        'x': {'type': 'float', 'min': -5, 'max': 5},
        'y': {'type': 'float', 'min': -5, 'max': 5}
    }


async def test_surrogate_optimization():
    """Test the surrogate optimization implementation."""
    print("🧪 Testing Surrogate Optimization Implementation")
    print("="*60)

    # Create configuration
    config = ComputationalOptimizationConfig(
        enable_surrogate_models=True,
        expensive_trials=10,
        update_frequency=3,
        surrogate_model_type="gaussian_process",
        expensive_evaluation_ratio=0.3,
        enable_surrogate_models_multi=False
    )

    # Create optimizer
    optimizer = SurrogateOptimizer(config)

    # Create parameter space
    parameter_space = create_simple_parameter_space()

    print("🚀 Starting surrogate optimization test...")

    try:
        # Run optimization
        result = optimizer.optimize_with_surrogates(
            objective_func=simple_test_objective,
            n_trials=30,
            parameter_space=parameter_space
        )

        print("✅ Surrogate optimization completed successfully!")
        print(f"📊 Best score: {result.get('best_score', 0):.4f}")
        print(f"🎯 Best parameters: {result.get('best_params', {})}")

        # Print detailed statistics
        if 'surrogate_accuracy' in result:
            accuracy = result['surrogate_accuracy']
            print(f"📈 Surrogate accuracy - R²: {accuracy.get('r2', 0):.4f}")
            print(f"📈 Surrogate accuracy - MAE: {accuracy.get('mae', 0):.4f}")

        if 'optimization_efficiency' in result:
            efficiency = result['optimization_efficiency']
            print(f"⚡ Expensive evaluation ratio: {efficiency.get('expensive_evaluation_ratio', 0):.2f}")
            print(f"⚡ Time saved: {efficiency.get('total_time_saved', 0):.2f}")

        # Get surrogate statistics
        stats = optimizer.get_surrogate_statistics()
        print(f"🧠 Model type: {stats.get('model_type', 'unknown')}")
        print(f"🔬 Expensive evaluations: {stats.get('expensive_evaluations', 0)}")

        return True

    except Exception as e:
        print(f"❌ Surrogate optimization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_multiple_surrogate_types():
    """Test different surrogate model types."""
    print("\n🔧 Testing Multiple Surrogate Model Types")
    print("="*60)

    surrogate_types = ["gaussian_process", "random_forest", "xgboost"]
    results = {}

    for surrogate_type in surrogate_types:
        print(f"\n🧪 Testing {surrogate_type}...")

        config = ComputationalOptimizationConfig(
            enable_surrogate_models=True,
            expensive_trials=8,
            update_frequency=2,
            surrogate_model_type=surrogate_type,
            expensive_evaluation_ratio=0.4,
            enable_surrogate_models_multi=False
        )

        optimizer = SurrogateOptimizer(config)
        parameter_space = create_simple_parameter_space()

        try:
            result = optimizer.optimize_with_surrogates(
                objective_func=simple_test_objective,
                n_trials=20,
                parameter_space=parameter_space
            )

            results[surrogate_type] = {
                'best_score': result.get('best_score', 0),
                'surrogate_accuracy': result.get('surrogate_accuracy', {}).get('r2', 0),
                'expensive_evaluations': result.get('expensive_evaluations', 0)
            }

            print(f"  ✅ {surrogate_type}: Best score = {result.get('best_score', 0):.4f}")

        except Exception as e:
            print(f"  ❌ {surrogate_type} failed: {e}")
            results[surrogate_type] = {'error': str(e)}

    # Compare results
    print("\n📊 Comparison of Surrogate Model Types:")
    print("-" * 50)
    for model_type, result in results.items():
        if 'error' not in result:
            print(f"{model_type:15} | Score: {result['best_score']:8.4f} | R²: {result['surrogate_accuracy']:6.4f} | Expensive: {result['expensive_evaluations']:2d}")
        else:
            print(f"{model_type:15} | Error: {result['error']}")

    return results


async def main():
    """Main test function."""
    print("🚀 Starting Surrogate Optimization Tests")
    print("="*80)

    # Test basic functionality
    basic_test_passed = await test_surrogate_optimization()

    # Test multiple surrogate types
    multi_test_results = await test_multiple_surrogate_types()

    # Summary
    print("\n" + "="*80)
    print("📋 TEST SUMMARY")
    print("="*80)
    print(f"Basic functionality test: {'✅ PASSED' if basic_test_passed else '❌ FAILED'}")

    successful_models = sum(1 for result in multi_test_results.values() if 'error' not in result)
    total_models = len(multi_test_results)
    print(f"Surrogate model types: {successful_models}/{total_models} successful")

    if successful_models > 0:
        best_model = max(
            (k, v) for k, v in multi_test_results.items() if 'error' not in v,
            key=lambda x: x[1]['best_score']
        )
        print(f"Best performing model: {best_model[0]} (score: {best_model[1]['best_score']:.4f})")

    print("\n✅ All tests completed!")


if __name__ == "__main__":
    asyncio.run(main())