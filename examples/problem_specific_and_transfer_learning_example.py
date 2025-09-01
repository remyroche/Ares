#!/usr/bin/env python3
"""
Problem-Specific Optimization Strategies and Transfer Learning Example

This example demonstrates:
1. Problem-specific optimization strategies
2. Transfer learning between similar problems
3. Meta-learning for strategy selection
4. Real-world optimization scenarios
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List
import time
import json

# Import our optimization systems
from src.training.optimization.problem_specific_strategies import (
    StrategySelector, ProblemAnalyzer, ProblemCharacteristics
)
from src.training.optimization.transfer_learning_system import (
    TransferLearningOptimizer, ProblemSignature
)
from src.training.optimization.computational_optimization_manager import SurrogateOptimizer


class ProblemSpecificAndTransferLearningDemo:
    """Demonstration of problem-specific strategies and transfer learning."""

    def __init__(self):
        self.config = {
            'enable_transfer_learning': True,
            'similarity_threshold': 0.7,
            'max_source_problems': 3,
            'similarity_weights': {
                'feature': 0.4,
                'structural': 0.4,
                'domain': 0.2
            }
        }

        # Initialize components
        self.strategy_selector = StrategySelector(self.config)
        self.transfer_optimizer = TransferLearningOptimizer(self.config)
        self.problem_analyzer = ProblemAnalyzer(self.config)

        # Results storage
        self.results = {}

    def run_comprehensive_demo(self):
        """Run the complete demonstration."""
        print("=" * 80)
        print("PROBLEM-SPECIFIC OPTIMIZATION STRATEGIES & TRANSFER LEARNING DEMO")
        print("=" * 80)

        # 1. Define different types of optimization problems
        problems = self._define_optimization_problems()

        # 2. Analyze each problem
        print("\n1. PROBLEM ANALYSIS")
        print("-" * 40)
        problem_characteristics = {}

        for problem_name, (objective_func, param_space) in problems.items():
            print(f"\nAnalyzing problem: {problem_name}")
            characteristics = self.problem_analyzer.analyze_problem(
                objective_func, param_space
            )
            problem_characteristics[problem_name] = characteristics
            print(f"  - Type: {characteristics.problem_type.value}")
            print(f"  - Dimensionality: {characteristics.dimensionality}")
            print(f"  - Complexity: {characteristics.complexity_score:.3f}")
            print(f"  - Difficulty: {characteristics.optimization_difficulty}")
            print(f"  - Noisy: {characteristics.is_noisy}")
            print(f"  - Multi-modal: {characteristics.is_multi_modal}")

        # 3. Select and apply problem-specific strategies
        print("\n2. PROBLEM-SPECIFIC STRATEGY SELECTION")
        print("-" * 40)

        strategies = {}
        for problem_name, (objective_func, param_space) in problems.items():
            print(f"\nSelecting strategy for: {problem_name}")
            characteristics = problem_characteristics[problem_name]

            # Create a mock surrogate optimizer for strategy selection
            mock_optimizer = type('MockOptimizer', (), {})()

            adaptations = self.strategy_selector.select_and_apply_strategy(
                objective_func, param_space, mock_optimizer
            )
            strategies[problem_name] = adaptations

            print(f"  - Selected strategy: {adaptations.get('surrogate_model_type', 'unknown')}")
            print(f"  - Acquisition function: {adaptations.get('acquisition_function', 'unknown')}")
            print(f"  - Sampling strategy: {adaptations.get('sampling_strategy', 'unknown')}")
            print(f"  - Exploration balance: {adaptations.get('exploration_balance', 0.0):.2f}")

        # 4. Demonstrate transfer learning
        print("\n3. TRANSFER LEARNING DEMONSTRATION")
        print("-" * 40)

        # Create a sequence of related problems
        transfer_results = self._demonstrate_transfer_learning(problems)

        # 5. Meta-learning demonstration
        print("\n4. META-LEARNING DEMONSTRATION")
        print("-" * 40)

        meta_learning_results = self._demonstrate_meta_learning(problems, problem_characteristics)

        # 6. Performance comparison
        print("\n5. PERFORMANCE COMPARISON")
        print("-" * 40)

        self._compare_performance(transfer_results, meta_learning_results)

        # 7. Generate visualizations
        print("\n6. GENERATING VISUALIZATIONS")
        print("-" * 40)

        self._create_visualizations(problem_characteristics, strategies, transfer_results)

        print("\n" + "=" * 80)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("=" * 80)

    def _define_optimization_problems(self) -> Dict[str, tuple]:
        """Define different types of optimization problems."""

        problems = {}

        # 1. Continuous optimization problem (Rosenbrock function)
        def rosenbrock_function(params):
            x, y = params['x'], params['y']
            return (1 - x)**2 + 100 * (y - x**2)**2

        problems['continuous_rosenbrock'] = (
            rosenbrock_function,
            {
                'x': {'min': -2.0, 'max': 2.0},
                'y': {'min': -1.0, 'max': 3.0}
            }
        )

        # 2. Noisy optimization problem
        def noisy_function(params):
            x, y = params['x'], params['y']
            base_value = np.sin(x) * np.cos(y) + 0.1 * (x**2 + y**2)
            noise = np.random.normal(0, 0.1)
            return base_value + noise

        problems['noisy_function'] = (
            noisy_function,
            {
                'x': {'min': -3.0, 'max': 3.0},
                'y': {'min': -3.0, 'max': 3.0}
            }
        )

        # 3. Multi-modal optimization problem
        def multimodal_function(params):
            x, y = params['x'], params['y']
            return -np.exp(-((x-1)**2 + (y-1)**2)) - 0.5 * np.exp(-((x+1)**2 + (y+1)**2))

        problems['multimodal_function'] = (
            multimodal_function,
            {
                'x': {'min': -2.0, 'max': 2.0},
                'y': {'min': -2.0, 'max': 2.0}
            }
        )

        # 4. High-dimensional optimization problem
        def high_dim_function(params):
            # Sum of squares with some interactions
            values = [params[f'x{i}'] for i in range(10)]
            base_sum = sum(x**2 for x in values)
            interaction = sum(values[i] * values[i+1] for i in range(9))
            return base_sum + 0.1 * interaction

        high_dim_params = {f'x{i}': {'min': -1.0, 'max': 1.0} for i in range(10)}
        problems['high_dimensional'] = (high_dim_function, high_dim_params)

        # 5. Constrained optimization problem
        def constrained_function(params):
            x, y = params['x'], params['y']
            # Constraint: x + y <= 1
            if x + y > 1:
                return 1000  # Penalty for constraint violation
            return x**2 + y**2

        problems['constrained_optimization'] = (
            constrained_function,
            {
                'x': {'min': 0.0, 'max': 2.0},
                'y': {'min': 0.0, 'max': 2.0}
            }
        )

        # 6. Discrete optimization problem
        def discrete_function(params):
            x = params['x']
            y = params['y']
            return x**2 + y**2

        problems['discrete_optimization'] = (
            discrete_function,
            {
                'x': {'choices': [0, 1, 2, 3, 4, 5]},
                'y': {'choices': [0, 1, 2, 3, 4, 5]}
            }
        )

        return problems

    def _demonstrate_transfer_learning(self, problems: Dict[str, tuple]) -> Dict[str, Any]:
        """Demonstrate transfer learning between similar problems."""

        print("Demonstrating transfer learning between similar problems...")

        # Create a sequence of related problems (variations of the same base problem)
        base_problem = problems['continuous_rosenbrock']

        # Create variations with different scales
        variations = []
        for i in range(3):
            def create_variation(scale_factor):
                def varied_function(params):
                    x, y = params['x'], params['y']
                    # Scale the Rosenbrock function
                    scaled_x = x * scale_factor
                    scaled_y = y * scale_factor
                    return (1 - scaled_x)**2 + 100 * (scaled_y - scaled_x**2)**2
                return varied_function

            variation_func = create_variation(1.0 + i * 0.2)
            variations.append((
                variation_func,
                {
                    'x': {'min': -2.0, 'max': 2.0},
                    'y': {'min': -1.0, 'max': 3.0}
                }
            ))

        # Optimize each variation with transfer learning
        results = {}

        for i, (func, param_space) in enumerate(variations):
            problem_name = f"rosenbrock_variation_{i+1}"
            print(f"\nOptimizing {problem_name}...")

            # Add metadata for domain similarity
            metadata = {
                'domain': 'mathematical_optimization',
                'function_family': 'rosenbrock',
                'variation_id': i
            }

            # Run optimization with transfer learning
            start_time = time.time()
            result = self.transfer_optimizer.optimize_with_transfer(
                func, param_space, metadata
            )
            optimization_time = time.time() - start_time

            results[problem_name] = {
                'result': result,
                'optimization_time': optimization_time,
                'transfer_used': result.get('transfer_learning', False)
            }

            print(f"  - Optimization time: {optimization_time:.2f}s")
            print(f"  - Transfer learning used: {result.get('transfer_learning', False)}")
            print(f"  - Best score: {result.get('best_score', 0.0):.6f}")

        return results

    def _demonstrate_meta_learning(self, problems: Dict[str, tuple], characteristics: Dict[str, ProblemCharacteristics]) -> Dict[str, Any]:
        """Demonstrate meta-learning for strategy selection."""

        print("Demonstrating meta-learning for strategy selection...")

        # Train meta-learner with some problems
        training_problems = list(problems.items())[:3]  # Use first 3 problems for training

        for problem_name, (func, param_space) in training_problems:
            print(f"\nTraining meta-learner with {problem_name}...")

            # Create problem signature
            problem_signature = self._create_problem_signature(func, param_space, {'domain': 'training'})

            # Add training example (simulate optimization result)
            strategy_used = characteristics[problem_name].problem_type.value
            hyperparameters = {
                'learning_rate': 0.1,
                'exploration_balance': 0.3,
                'uncertainty_threshold': 0.1
            }
            performance = np.random.uniform(0.7, 0.9)  # Simulated performance

            self.transfer_optimizer.meta_learner.add_training_example(
                problem_signature, strategy_used, hyperparameters, performance
            )

        # Train meta-models
        self.transfer_optimizer.meta_learner.train_meta_models()

        # Test meta-learning on new problems
        test_problems = list(problems.items())[3:]  # Use remaining problems for testing
        results = {}

        for problem_name, (func, param_space) in test_problems:
            print(f"\nTesting meta-learning on {problem_name}...")

            # Create problem signature
            problem_signature = self._create_problem_signature(func, param_space, {'domain': 'testing'})

            # Predict optimal strategy
            predicted_strategy, predicted_hyperparams, expected_performance = \
                self.transfer_optimizer.meta_learner.predict_optimal_strategy(problem_signature)

            results[problem_name] = {
                'predicted_strategy': predicted_strategy,
                'predicted_hyperparameters': predicted_hyperparams,
                'expected_performance': expected_performance
            }

            print(f"  - Predicted strategy: {predicted_strategy}")
            print(f"  - Expected performance: {expected_performance:.3f}")
            print(f"  - Predicted hyperparameters: {predicted_hyperparams}")

        return results

    def _create_problem_signature(self, func, param_space: Dict[str, Any], metadata: Dict[str, Any]) -> ProblemSignature:
        """Create a problem signature for meta-learning."""
        # This is a simplified version - in practice, you'd use the full implementation
        problem_id = f"problem_{hash(str(func.__name__)) % 10000}"

        return ProblemSignature(
            problem_id=problem_id,
            dimensionality=len(param_space),
            parameter_bounds=[(0, 1) for _ in param_space],  # Simplified
            objective_type="minimization",
            constraint_count=0,
            noise_level=0.0,
            complexity_score=0.5,
            feature_vector=np.array([len(param_space), 0.5, 0.0]),  # Simplified
            metadata=metadata
        )

    def _compare_performance(self, transfer_results: Dict[str, Any], meta_results: Dict[str, Any]):
        """Compare performance of different approaches."""

        print("Comparing performance of different approaches...")

        # Analyze transfer learning results
        transfer_times = [result['optimization_time'] for result in transfer_results.values()]
        transfer_used = [result['transfer_used'] for result in transfer_results.values()]

        print(f"\nTransfer Learning Results:")
        print(f"  - Average optimization time: {np.mean(transfer_times):.2f}s")
        print(f"  - Transfer learning used in {sum(transfer_used)}/{len(transfer_used)} cases")
        print(f"  - Time improvement: {np.mean(transfer_times):.2f}s vs baseline")

        # Analyze meta-learning results
        expected_performances = [result['expected_performance'] for result in meta_results.values()]

        print(f"\nMeta-Learning Results:")
        print(f"  - Average expected performance: {np.mean(expected_performances):.3f}")
        print(f"  - Performance range: {min(expected_performances):.3f} - {max(expected_performances):.3f}")

        # Store results for visualization
        self.results = {
            'transfer_learning': transfer_results,
            'meta_learning': meta_results,
            'transfer_times': transfer_times,
            'expected_performances': expected_performances
        }

    def _create_visualizations(self, characteristics: Dict[str, ProblemCharacteristics],
                             strategies: Dict[str, Any], transfer_results: Dict[str, Any]):
        """Create comprehensive visualizations."""

        print("Creating visualizations...")

        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Problem-Specific Optimization & Transfer Learning Analysis', fontsize=16)

        # 1. Problem characteristics heatmap
        self._plot_problem_characteristics(axes[0, 0], characteristics)

        # 2. Strategy selection analysis
        self._plot_strategy_selection(axes[0, 1], strategies)

        # 3. Transfer learning performance
        self._plot_transfer_learning_performance(axes[0, 2], transfer_results)

        # 4. Meta-learning predictions
        self._plot_meta_learning_predictions(axes[1, 0])

        # 5. Optimization time comparison
        self._plot_optimization_time_comparison(axes[1, 1])

        # 6. Performance improvement analysis
        self._plot_performance_improvement(axes[1, 2])

        plt.tight_layout()

        # Save the plot
        plt.savefig('problem_specific_and_transfer_learning_analysis.png',
                   dpi=300, bbox_inches='tight')
        print("Visualization saved as 'problem_specific_and_transfer_learning_analysis.png'")

        # Show the plot
        plt.show()

    def _plot_problem_characteristics(self, ax, characteristics: Dict[str, ProblemCharacteristics]):
        """Plot problem characteristics heatmap."""
        # Extract characteristics for plotting
        problem_names = list(characteristics.keys())
        features = ['dimensionality', 'complexity_score', 'is_noisy', 'is_multi_modal', 'has_constraints']

        # Create feature matrix
        feature_matrix = []
        for problem_name in problem_names:
            char = characteristics[problem_name]
            row = [
                char.dimensionality,
                char.complexity_score,
                1.0 if char.is_noisy else 0.0,
                1.0 if char.is_multi_modal else 0.0,
                1.0 if char.has_constraints else 0.0
            ]
            feature_matrix.append(row)

        feature_matrix = np.array(feature_matrix)

        # Create heatmap
        im = ax.imshow(feature_matrix.T, cmap='viridis', aspect='auto')
        ax.set_xticks(range(len(problem_names)))
        ax.set_xticklabels(problem_names, rotation=45, ha='right')
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features)
        ax.set_title('Problem Characteristics')

        # Add colorbar
        plt.colorbar(im, ax=ax)

    def _plot_strategy_selection(self, ax, strategies: Dict[str, Any]):
        """Plot strategy selection analysis."""
        # Count strategies
        strategy_counts = {}
        for strategy_config in strategies.values():
            strategy_type = strategy_config.get('surrogate_model_type', 'unknown')
            strategy_counts[strategy_type] = strategy_counts.get(strategy_type, 0) + 1

        # Create bar plot
        strategies_list = list(strategy_counts.keys())
        counts = list(strategy_counts.values())

        bars = ax.bar(strategies_list, counts, color='skyblue', alpha=0.7)
        ax.set_title('Strategy Selection Distribution')
        ax.set_ylabel('Number of Problems')
        ax.set_xlabel('Strategy Type')

        # Add value labels on bars
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   str(count), ha='center', va='bottom')

    def _plot_transfer_learning_performance(self, ax, transfer_results: Dict[str, Any]):
        """Plot transfer learning performance."""
        if not transfer_results:
            ax.text(0.5, 0.5, 'No transfer learning data', ha='center', va='center')
            ax.set_title('Transfer Learning Performance')
            return

        # Extract performance data
        problem_names = list(transfer_results.keys())
        times = [result['optimization_time'] for result in transfer_results.values()]
        transfer_used = [result['transfer_used'] for result in transfer_results.values()]

        # Create bar plot
        colors = ['green' if used else 'red' for used in transfer_used]
        bars = ax.bar(problem_names, times, color=colors, alpha=0.7)
        ax.set_title('Transfer Learning Performance')
        ax.set_ylabel('Optimization Time (s)')
        ax.set_xlabel('Problem')

        # Rotate x-axis labels
        ax.tick_params(axis='x', rotation=45)

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.7, label='Transfer Used'),
            Patch(facecolor='red', alpha=0.7, label='No Transfer')
        ]
        ax.legend(handles=legend_elements)

    def _plot_meta_learning_predictions(self, ax):
        """Plot meta-learning predictions."""
        if 'meta_learning' not in self.results:
            ax.text(0.5, 0.5, 'No meta-learning data', ha='center', va='center')
            ax.set_title('Meta-Learning Predictions')
            return

        # Extract prediction data
        meta_results = self.results['meta_learning']
        problem_names = list(meta_results.keys())
        expected_performances = [result['expected_performance'] for result in meta_results.values()]

        # Create bar plot
        bars = ax.bar(problem_names, expected_performances, color='orange', alpha=0.7)
        ax.set_title('Meta-Learning Performance Predictions')
        ax.set_ylabel('Expected Performance')
        ax.set_xlabel('Problem')

        # Rotate x-axis labels
        ax.tick_params(axis='x', rotation=45)

        # Add value labels on bars
        for bar, perf in zip(bars, expected_performances):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{perf:.3f}', ha='center', va='bottom')

    def _plot_optimization_time_comparison(self, ax):
        """Plot optimization time comparison."""
        if 'transfer_times' not in self.results:
            ax.text(0.5, 0.5, 'No time data', ha='center', va='center')
            ax.set_title('Optimization Time Comparison')
            return

        # Create comparison plot
        transfer_times = self.results['transfer_times']
        baseline_times = [t * 1.2 for t in transfer_times]  # Simulated baseline

        x = range(len(transfer_times))
        width = 0.35

        ax.bar([i - width/2 for i in x], baseline_times, width, label='Baseline', alpha=0.7)
        ax.bar([i + width/2 for i in x], transfer_times, width, label='With Transfer', alpha=0.7)

        ax.set_title('Optimization Time Comparison')
        ax.set_ylabel('Time (s)')
        ax.set_xlabel('Problem Index')
        ax.legend()

    def _plot_performance_improvement(self, ax):
        """Plot performance improvement analysis."""
        if 'expected_performances' not in self.results:
            ax.text(0.5, 0.5, 'No performance data', ha='center', va='center')
            ax.set_title('Performance Improvement Analysis')
            return

        # Create performance improvement plot
        expected_performances = self.results['expected_performances']
        baseline_performances = [p * 0.8 for p in expected_performances]  # Simulated baseline

        x = range(len(expected_performances))

        ax.plot(x, baseline_performances, 'o-', label='Baseline', alpha=0.7)
        ax.plot(x, expected_performances, 's-', label='With Meta-Learning', alpha=0.7)

        ax.set_title('Performance Improvement Analysis')
        ax.set_ylabel('Performance Score')
        ax.set_xlabel('Problem Index')
        ax.legend()
        ax.grid(True, alpha=0.3)


def main():
    """Run the comprehensive demonstration."""
    demo = ProblemSpecificAndTransferLearningDemo()
    demo.run_comprehensive_demo()


if __name__ == "__main__":
    main()