#!/usr/bin/env python3
"""
Comprehensive Surrogate Optimization Example

This example demonstrates the full surrogate optimization system with:
    self.logger.info("Implementation placeholder - needs specific logic")
- Multiple surrogate model types (Gaussian Process, Random Forest, XGBoost, Neural Network)
- Advanced acquisition functions (Expected Improvement, UCB, Probability of Improvement)
- Multi-objective optimization
- Adaptive sampling and exploration-exploitation balance
- Comprehensive analysis and visualization
"""

import asyncio
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import json

# Import the surrogate optimizer
from src.training.optimization.computational_optimization_manager import (
    SurrogateOptimizer,
    ComputationalOptimizationConfig,
)

# Import utility functions
from src.utils.logger import system_logger
from src.utils.decorators import handle_errors


class SurrogateOptimizationDemo:
    pass"""Comprehensive demonstration of surrogate optimization capabilities."""

    def __init__(...):
    passself.logger = system_logger.getChild("SurrogateOptimizationDemo")
        self.results = {}
        self.visualizations = {}

    @handle_errors(default_return=None, context="surrogate_optimization_demo_initialization")
    async def initialize(...) -> ...:
    """..."""
    passself.logger.info("🚀 Initializing Surrogate Optimization Demo")

        # Create different configurations for testing
        self.configs = {
            'gaussian_process': self._create_gp_config(),
            'random_forest': self._create_rf_config(),
            'xgboost': self._create_xgb_config(),
            'neural_network': self._create_nn_config(),
            'multi_objective': self._create_multi_objective_config(),
        }

        self.logger.info("✅ Surrogate Optimization Demo initialized")
        return True

    def _create_gp_config(...) -> ...:
    """..."""
    passreturn ComputationalOptimizationConfig(
            enable_surrogate_models=True,
            expensive_trials=20,
            update_frequency=5,
            surrogate_model_type="gaussian_process",
            expensive_evaluation_ratio=0.3,
            enable_surrogate_models_multi=False
        )

    def _create_rf_config(...) -> ...:
    """..."""
    passreturn ComputationalOptimizationConfig(
            enable_surrogate_models=True,
            expensive_trials=15,
            update_frequency=3,
            surrogate_model_type="random_forest",
            expensive_evaluation_ratio=0.4,
            enable_surrogate_models_multi=True
        )

    def _create_xgb_config(...) -> ...:
    """..."""
    passreturn ComputationalOptimizationConfig(
            enable_surrogate_models=True,
            expensive_trials=25,
            update_frequency=7,
            surrogate_model_type="xgboost",
            expensive_evaluation_ratio=0.25,
            enable_surrogate_models_multi=True
        )

    def _create_nn_config(...) -> ...:
    """..."""
    passreturn ComputationalOptimizationConfig(
            enable_surrogate_models=True,
            expensive_trials=30,
            update_frequency=10,
            surrogate_model_type="neural_network",
            expensive_evaluation_ratio=0.2,
            enable_surrogate_models_multi=False
        )

    def _create_multi_objective_config(...) -> ...:
    """..."""
    passreturn ComputationalOptimizationConfig(
            enable_surrogate_models=True,
            expensive_trials=20,
            update_frequency=5,
            surrogate_model_type="gaussian_process",
            expensive_evaluation_ratio=0.3,
            enable_surrogate_models_multi=True
        )

    async def run_comprehensive_demo(...) -> ...:
    """..."""
    passself.logger.info("🎯 Starting Comprehensive Surrogate Optimization Demo")

        # Test different objective functions
        objective_functions = {
            'simple_quadratic': self._simple_quadratic_objective,
            'complex_multi_modal': self._complex_multi_modal_objective,
            'noisy_function': self._noisy_function_objective,
            'multi_objective': self._multi_objective_function,
        }

        # Test different parameter spaces
        parameter_spaces = {
            'trading_strategy': self._create_trading_strategy_space(),
            'ml_hyperparameters': self._create_ml_hyperparameter_space(),
            'feature_engineering': self._create_feature_engineering_space(),
        }

        all_results = {}

        # Run optimization with different configurations
        for config_name, config in self.configs.items():
    passpassself.logger.info(f"🔧 Testing {config_name} configuration...")

            config_results = {}
            optimizer = SurrogateOptimizer(config)

            for obj_name, objective_func in objective_functions.items():
    passfor space_name, parameter_space in parameter_spaces.items():
    passtest_name = f"{config_name}_{obj_name}_{space_name}"

                    self.logger.info(f"  Running {test_name}...")

                    try:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
                        result = optimizer.optimize_with_surrogates(
                            objective_func=objective_func,
                            n_trials=50,
                            parameter_space=parameter_space,
                            constraints=self._create_constraints()
                        )

                        config_results[test_name] = result

                        self.logger.info(f"    ✅ {test_name} completed. Best score: {result.get('best_score', 0):.4f}")

                    except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"    ❌ {test_name} failed: {e}")
                        config_results[test_name] = {'error': str(e)}

            all_results[config_name] = config_results

        # Analyze and compare results
        analysis = self._analyze_all_results(all_results)

        # Generate visualizations
        self._generate_comprehensive_visualizations(all_results)

        # Save results
        self._save_results(all_results, analysis)

        self.logger.info("✅ Comprehensive Surrogate Optimization Demo completed")
        return {
            'results': all_results,
            'analysis': analysis,
            'visualizations': self.visualizations
        }

    def _simple_quadratic_objective(...) -> ...:
    """..."""
    passx = params.get('x', 0)
        y = params.get('y', 0)

        # Simple quadratic function with global minimum at (0, 0)
        return -(x**2 + y**2)

    def _complex_multi_modal_objective(...) -> ...:
    pass"""..."""
    passx = params.get('x', 0)
        y = params.get('y', 0)

        # Multi-modal function with multiple local optima
        return -(
            np.sin(x) * np.cos(y) +
            0.5 * np.sin(2*x) * np.cos(2*y) +
            0.25 * np.sin(3*x) * np.cos(3*y)
        )

    def _noisy_function_objective(...) -> ...:
    pass"""..."""
    passx = params.get('x', 0)
        y = params.get('y', 0)

        # Add noise to the objective
        noise = np.random.normal(0, 0.1)
        return -(x**2 + y**2) + noise

    def _multi_objective_function(...) -> ...:
    """..."""
    passx = params.get('x', 0)
        y = params.get('y', 0)

        # Performance objective (maximize)
        performance = -(x**2 + y**2)

        # Risk objective (minimize, so we return negative)
        risk = -(abs(x) + abs(y))

        # Cost objective (minimize, so we return negative)
        cost = -(abs(x) + abs(y)) * 0.1

        return {
            'performance': performance,
            'risk': risk,
            'cost': cost
        }

    def _create_trading_strategy_space(...) -> ...:
    """..."""
    passreturn {
            'sma_short': {'type': 'int', 'min': 5, 'max': 50},
            'sma_long': {'type': 'int', 'min': 20, 'max': 200},
            'rsi_threshold': {'type': 'float', 'min': 20, 'max': 80},
            'volatility_window': {'type': 'int', 'min': 10, 'max': 100},
            'momentum_period': {'type': 'int', 'min': 5, 'max': 50},
            'stop_loss': {'type': 'float', 'min': 0.01, 'max': 0.1},
            'take_profit': {'type': 'float', 'min': 0.02, 'max': 0.2}
        }

    def _create_ml_hyperparameter_space(...) -> ...:
    """..."""
    passreturn {
            'learning_rate': {'type': 'float', 'min': 0.001, 'max': 0.3},
            'n_estimators': {'type': 'int', 'min': 50, 'max': 500},
            'max_depth': {'type': 'int', 'min': 3, 'max': 15},
            'min_samples_split': {'type': 'int', 'min': 2, 'max': 20},
            'min_samples_leaf': {'type': 'int', 'min': 1, 'max': 10},
            'subsample': {'type': 'float', 'min': 0.5, 'max': 1.0},
            'colsample_bytree': {'type': 'float', 'min': 0.5, 'max': 1.0}
        }

    def _create_feature_engineering_space(...) -> ...:
    """..."""
    passreturn {
            'window_size': {'type': 'int', 'min': 5, 'max': 100},
            'feature_threshold': {'type': 'float', 'min': 0.01, 'max': 0.5},
            'correlation_threshold': {'type': 'float', 'min': 0.5, 'max': 0.95},
            'vif_threshold': {'type': 'float', 'min': 1.0, 'max': 10.0},
            'pca_components': {'type': 'int', 'min': 5, 'max': 50},
            'feature_selection_method': {
                'type': 'categorical',
                'choices': ['mutual_info', 'f_regression', 'chi2', 'anova']
            }
        }

    def _create_constraints(...) -> ...:
    """..."""
    passreturn {
            'sma_constraint': lambda params: params.get('sma_long', 0) > params.get('sma_short', 0),
            'positive_values': lambda params: all(v > 0 for v in params.values() if isinstance(v, (int, float))),
            'reasonable_ratios': lambda params: params.get('take_profit', 1) > params.get('stop_loss', 0)
        }

    def _analyze_all_results(...) -> ...:
    """..."""
    passself.logger.info("📊 Analyzing all optimization results...")

        analysis = {
            'performance_comparison': {},
            'efficiency_analysis': {},
            'model_accuracy': {},
            'convergence_analysis': {},
            'recommendations': []
        }

        # Performance comparison
        for config_name, config_results in all_results.items():
    passbest_scores = []
            convergence_rates = []
            time_savings = []

            for test_name, result in config_results.items():
    passif 'error' not in result:
    passbest_scores.append(result.get('best_score', 0))

                    convergence = result.get('convergence_metrics', {})
                    convergence_rates.append(convergence.get('convergence_rate', 0))

                    efficiency = result.get('optimization_efficiency', {})
                    time_savings.append(efficiency.get('total_time_saved', 0))

            if best_scores:
    passanalysis['performance_comparison'][config_name] = {
                    'mean_best_score': np.mean(best_scores),
                    'std_best_score': np.std(best_scores),
                    'max_best_score': np.max(best_scores),
                    'min_best_score': np.min(best_scores)
                }

                analysis['efficiency_analysis'][config_name] = {
                    'mean_time_savings': np.mean(time_savings),
                    'mean_convergence_rate': np.mean(convergence_rates)
                }

        # Model accuracy analysis
        for config_name, config_results in all_results.items():
    passaccuracies = []

            for test_name, result in config_results.items():
    passif 'error' not in result:
    passsurrogate_accuracy = result.get('surrogate_accuracy', {})
                    accuracies.append(surrogate_accuracy.get('r2', 0))

            if accuracies:
    passanalysis['model_accuracy'][config_name] = {
                    'mean_r2': np.mean(accuracies),
                    'std_r2': np.std(accuracies),
                    'min_r2': np.min(accuracies),
                    'max_r2': np.max(accuracies)
                }

        # Generate recommendations
        analysis['recommendations'] = self._generate_recommendations(analysis)

        return analysis

    def _generate_recommendations(...) -> ...:
    """..."""
    passrecommendations = []

        # Find best performing configuration
        if analysis['performance_comparison']:
    passbest_config = max(
                analysis['performance_comparison'].items(),
                key=lambda x: x[1]['mean_best_score']
            )
            recommendations.append(f"Best overall performance: {best_config[0]} configuration")

        # Find most efficient configuration
        if analysis['efficiency_analysis']:
    passmost_efficient = max(
                analysis['efficiency_analysis'].items(),
                key=lambda x: x[1]['mean_time_savings']
            )
            recommendations.append(f"Most time-efficient: {most_efficient[0]} configuration")

        # Find most accurate surrogate model
        if analysis['model_accuracy']:
    passmost_accurate = max(
                analysis['model_accuracy'].items(),
                key=lambda x: x[1]['mean_r2']
            )
            recommendations.append(f"Most accurate surrogate: {most_accurate[0]} configuration")

        # General recommendations
        recommendations.extend([
            "Use ensemble models for better uncertainty quantification",
            "Adjust exploration-exploitation balance based on problem complexity",
            "Monitor surrogate accuracy and retrain when necessary",
            "Consider multi-objective optimization for complex problems"
        ])

        return recommendations

    def _generate_comprehensive_visualizations(...) -> ...:
    pass"""..."""
    passself.logger.info("📈 Generating comprehensive visualizations...")

        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

        # 1. Performance comparison across configurations
        self._plot_performance_comparison(all_results)

        # 2. Convergence analysis
        self._plot_convergence_analysis(all_results)

        # 3. Surrogate accuracy comparison
        self._plot_surrogate_accuracy(all_results)

        # 4. Efficiency analysis
        self._plot_efficiency_analysis(all_results)

        # 5. Uncertainty analysis
        self._plot_uncertainty_analysis(all_results)

        # Save all plots
        self._save_visualizations()

    def _plot_performance_comparison(...) -> ...:
    """..."""
    passfig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Surrogate Optimization Performance Comparison', fontsize=16)

        # Extract data for plotting
        configs = []
        best_scores = []
        convergence_rates = []
        time_savings = []
        accuracies = []

        for config_name, config_results in all_results.items():
    passconfig_scores = []
            config_convergence = []
            config_time_savings = []
            config_accuracies = []

            for test_name, result in config_results.items():
    passif 'error' not in result:
    passconfig_scores.append(result.get('best_score', 0))

                    convergence = result.get('convergence_metrics', {})
                    config_convergence.append(convergence.get('convergence_rate', 0))

                    efficiency = result.get('optimization_efficiency', {})
                    config_time_savings.append(efficiency.get('total_time_saved', 0))

                    surrogate_accuracy = result.get('surrogate_accuracy', {})
                    config_accuracies.append(surrogate_accuracy.get('r2', 0))

            if config_scores:
    passconfigs.append(config_name)
                best_scores.append(np.mean(config_scores))
                convergence_rates.append(np.mean(config_convergence))
                time_savings.append(np.mean(config_time_savings))
                accuracies.append(np.mean(config_accuracies))

        # Plot 1: Best scores
        axes[0, 0].bar(configs, best_scores, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Mean Best Scores')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # Plot 2: Convergence rates
        axes[0, 1].bar(configs, convergence_rates, color='lightgreen', alpha=0.7)
        axes[0, 1].set_title('Mean Convergence Rates')
        axes[0, 1].set_ylabel('Convergence Rate')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # Plot 3: Time savings
        axes[1, 0].bar(configs, time_savings, color='salmon', alpha=0.7)
        axes[1, 0].set_title('Mean Time Savings')
        axes[1, 0].set_ylabel('Time Saved')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # Plot 4: Surrogate accuracy
        axes[1, 1].bar(configs, accuracies, color='gold', alpha=0.7)
        axes[1, 1].set_title('Mean Surrogate Accuracy (R²)')
        axes[1, 1].set_ylabel('R² Score')
        axes[1, 1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        self.visualizations['performance_comparison'] = fig

    def _plot_convergence_analysis(...) -> ...:
    """..."""
    passfig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Convergence Analysis', fontsize=16)

        plot_idx = 0
        for config_name, config_results in all_results.items():
    passif plot_idx >= 4:
    passbreak

            row, col = plot_idx // 2, plot_idx % 2

            # Find a good example for this configuration
            best_example = None
            best_score = float('-inf')

            for test_name, result in config_results.items():
    passif 'error' not in result:
    passscore = result.get('best_score', 0)
                    if score > best_score:
    passbest_score = score
                        best_example = result

            if best_example:
    passconvergence = best_example.get('convergence_metrics', {})
                progression = convergence.get('best_score_progression', [])

                if progression:
    passaxes[row, col].plot(progression, linewidth=2, alpha=0.8)
                    axes[row, col].set_title(f'{config_name} - Best Score Progression')
                    axes[row, col].set_xlabel('Trial')
                    axes[row, col].set_ylabel('Best Score')
                    axes[row, col].grid(True, alpha=0.3)

            plot_idx += 1

        plt.tight_layout()
        self.visualizations['convergence_analysis'] = fig

    def _plot_surrogate_accuracy(...) -> ...:
    """..."""
    passfig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Surrogate Model Accuracy Analysis', fontsize=16)

        plot_idx = 0
        for config_name, config_results in all_results.items():
    passif plot_idx >= 4:
    passbreak

            row, col = plot_idx // 2, plot_idx % 2

            # Collect accuracy metrics
            r2_scores = []
            mae_scores = []
            rmse_scores = []

            for test_name, result in config_results.items():
    passif 'error' not in result:
    passaccuracy = result.get('surrogate_accuracy', {})
                    r2_scores.append(accuracy.get('r2', 0))
                    mae_scores.append(accuracy.get('mae', 0))
                    rmse_scores.append(accuracy.get('rmse', 0))

            if r2_scores:
    pass# Create box plot
                data = [r2_scores, mae_scores, rmse_scores]
                labels = ['R²', 'MAE', 'RMSE']

                bp = axes[row, col].boxplot(data, labels=labels, patch_artist=True)
                colors = ['lightblue', 'lightgreen', 'lightcoral']
                for patch, color in zip(bp['boxes'], colors):
    passpatch.set_facecolor(color)

                axes[row, col].set_title(f'{config_name} - Accuracy Metrics')
                axes[row, col].set_ylabel('Score')
                axes[row, col].grid(True, alpha=0.3)

            plot_idx += 1

        plt.tight_layout()
        self.visualizations['surrogate_accuracy'] = fig

    def _plot_efficiency_analysis(...) -> ...:
    """..."""
    passfig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Optimization Efficiency Analysis', fontsize=16)

        # Extract efficiency data
        configs = []
        expensive_ratios = []
        surrogate_utilizations = []
        time_savings = []

        for config_name, config_results in all_results.items():
    passconfig_expensive_ratios = []
            config_surrogate_utilizations = []
            config_time_savings = []

            for test_name, result in config_results.items():
    passif 'error' not in result:
    passefficiency = result.get('optimization_efficiency', {})
                    config_expensive_ratios.append(efficiency.get('expensive_evaluation_ratio', 0))
                    config_surrogate_utilizations.append(efficiency.get('surrogate_utilization', 0))
                    config_time_savings.append(efficiency.get('total_time_saved', 0))

            if config_expensive_ratios:
    passconfigs.append(config_name)
                expensive_ratios.append(np.mean(config_expensive_ratios))
                surrogate_utilizations.append(np.mean(config_surrogate_utilizations))
                time_savings.append(np.mean(config_time_savings))

        # Plot 1: Evaluation ratios
        x = np.arange(len(configs))
        width = 0.35

        axes[0].bar(x - width/2, expensive_ratios, width, label='Expensive Evaluations', alpha=0.7)
        axes[0].bar(x + width/2, surrogate_utilizations, width, label='Surrogate Evaluations', alpha=0.7)
        axes[0].set_xlabel('Configuration')
        axes[0].set_ylabel('Ratio')
        axes[0].set_title('Evaluation Type Distribution')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(configs, rotation=45)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Time savings
        axes[1].bar(configs, time_savings, color='orange', alpha=0.7)
        axes[1].set_xlabel('Configuration')
        axes[1].set_ylabel('Time Saved')
        axes[1].set_title('Time Savings')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        self.visualizations['efficiency_analysis'] = fig

    def _plot_uncertainty_analysis(...) -> ...:
    """..."""
    passfig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Uncertainty Analysis', fontsize=16)

        plot_idx = 0
        for config_name, config_results in all_results.items():
    passif plot_idx >= 4:
    passbreak

            row, col = plot_idx // 2, plot_idx % 2

            # Collect uncertainty data
            uncertainties = []

            for test_name, result in config_results.items():
    passif 'error' not in result:
    passuncertainty_analysis = result.get('uncertainty_analysis', {})
                    uncertainty_trend = uncertainty_analysis.get('uncertainty_trend', [])
                    if uncertainty_trend:
    passuncertainties.extend(uncertainty_trend)

            if uncertainties:
    passaxes[row, col].hist(uncertainties, bins=20, alpha=0.7, color='purple')
                axes[row, col].set_title(f'{config_name} - Uncertainty Distribution')
                axes[row, col].set_xlabel('Uncertainty')
                axes[row, col].set_ylabel('Frequency')
                axes[row, col].grid(True, alpha=0.3)

            plot_idx += 1

        plt.tight_layout()
        self.visualizations['uncertainty_analysis'] = fig

    def _save_visualizations(...) -> ...:
    """..."""
    passfor name, fig in self.visualizations.items():
    passfilename = f"surrogate_optimization_{name}.png"
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            self.logger.info(f"📊 Saved visualization: {filename}")
            plt.close(fig)

    def _save_results(...) -> ...:
    """..."""
    passoutput = {
            'timestamp': time.time(),
            'results': all_results,
            'analysis': analysis,
            'summary': {
                'total_configurations': len(all_results),
                'total_tests': sum(len(config_results) for config_results in all_results.values()),
                'successful_tests': sum(
                    sum(1 for result in config_results.values() if 'error' not in result)
                    for config_results in all_results.values()
                )
            }
        }

        filename = f"surrogate_optimization_results_{int(time.time())}.json"
        with open(filename, 'w') as f:
    passpasspassjson.dump(output, f, indent=2, default=str)

        self.logger.info(f"💾 Saved results: {filename}")

    def print_summary(...) -> ...:
    """..."""
    passprint("\n" + "="*80)
        print("🎯 SURROGATE OPTIMIZATION DEMO SUMMARY")
        print("="*80)

        analysis = results.get('analysis', {})

        # Performance summary
        print("\n📊 PERFORMANCE COMPARISON:")
        if analysis.get('performance_comparison'):
    passfor config, metrics in analysis['performance_comparison'].items():
    passprint(f"  {config}:")
                print(f"    Mean Best Score: {metrics['mean_best_score']:.4f}")
                print(f"    Std Best Score: {metrics['std_best_score']:.4f}")
                print(f"    Max Best Score: {metrics['max_best_score']:.4f}")

        # Efficiency summary
        print("\n⚡ EFFICIENCY ANALYSIS:")
        if analysis.get('efficiency_analysis'):
    passfor config, metrics in analysis['efficiency_analysis'].items():
    passprint(f"  {config}:")
                print(f"    Mean Time Savings: {metrics['mean_time_savings']:.2f}")
                print(f"    Mean Convergence Rate: {metrics['mean_convergence_rate']:.4f}")

        # Accuracy summary
        print("\n🎯 MODEL ACCURACY:")
        if analysis.get('model_accuracy'):
    passfor config, metrics in analysis['model_accuracy'].items():
    passprint(f"  {config}:")
                print(f"    Mean R²: {metrics['mean_r2']:.4f}")
                print(f"    Std R²: {metrics['std_r2']:.4f}")

        # Recommendations
        print("\n💡 RECOMMENDATIONS:")
        for i, recommendation in enumerate(analysis.get('recommendations', []), 1):
    passprint(f"  {i}. {recommendation}")

        print("\n" + "="*80)


async def main(...):
    pass"""Main function to run the surrogate optimization demo."""
    print("🚀 Starting Comprehensive Surrogate Optimization Demo")
    print("="*80)

    # Initialize demo
    demo = SurrogateOptimizationDemo()
    success = await demo.initialize()

    if not success:
    passprint("❌ Failed to initialize demo")
        return

    # Run comprehensive demo
    results = await demo.run_comprehensive_demo()

    # Print summary
    demo.print_summary(results)

    print("\n✅ Demo completed successfully!")
    print("📊 Check the generated visualizations and results files for detailed analysis.")


if __name__ == "__main__":
    passpassasyncio.run(main())