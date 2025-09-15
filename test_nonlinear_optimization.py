"""
Non-Linear Optimization Testing Framework

This script provides a comprehensive testing environment for non-linear optimizations
including logarithmic, fractional power, and other transformations.

Usage:
    python test_nonlinear_optimization.py --test_type all
    python test_nonlinear_optimization.py --test_type logs --n_trials 100
"""

import numpy as np
import pandas as pd
import optuna
import time
import logging
from typing import Dict, Any, List, Callable, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OptimizationResult:
    """Container for optimization results."""
    method: str
    best_params: Dict[str, Any]
    best_value: float
    n_trials: int
    execution_time: float
    convergence_history: List[float]

class NonLinearOptimizationTester:
    """Comprehensive testing framework for non-linear optimizations."""
    
    def __init__(self):
        self.results = []
        self.test_functions = self._initialize_test_functions()
        
    def _initialize_test_functions(self) -> Dict[str, Callable]:
        """Initialize test objective functions."""
        return {
            'rosenbrock': self._rosenbrock_function,
            'rastrigin': self._rastrigin_function,
            'ackley': self._ackley_function,
            'financial_metric': self._financial_metric_function,
            'multi_modal': self._multi_modal_function
        }
    
    def _rosenbrock_function(self, x: float, y: float) -> float:
        """Rosenbrock function - classic optimization test."""
        return 100 * (y - x**2)**2 + (1 - x)**2
    
    def _rastrigin_function(self, x: float, y: float) -> float:
        """Rastrigin function - highly multimodal."""
        return 20 + x**2 + y**2 - 10 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))
    
    def _ackley_function(self, x: float, y: float) -> float:
        """Ackley function - many local minima."""
        return -20 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2))) - \
               np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))) + np.e + 20
    
    def _financial_metric_function(self, confidence: float, position_size: float, 
                                 leverage: float) -> float:
        """Simulated financial optimization function."""
        # Non-linear relationship between parameters and performance
        base_return = confidence * position_size * leverage
        risk_penalty = (position_size * leverage) ** 2.5  # Non-linear risk scaling
        confidence_bonus = np.log(1 + confidence * 10)  # Log transformation
        
        return base_return + confidence_bonus - risk_penalty
    
    def _multi_modal_function(self, x: float, y: float) -> float:
        """Multi-modal function with multiple peaks."""
        return np.sin(x) * np.cos(y) + 0.1 * (x**2 + y**2) - 0.5 * np.exp(-(x**2 + y**2))
    
    def test_linear_optimization(self, objective_func: Callable, 
                               param_ranges: Dict[str, Tuple[float, float]], 
                               n_trials: int = 50) -> OptimizationResult:
        """Test standard linear parameter optimization."""
        start_time = time.time()
        convergence_history = []
        
        def objective(trial):
            params = {}
            for param_name, (min_val, max_val) in param_ranges.items():
                params[param_name] = trial.suggest_float(param_name, min_val, max_val)
            
            value = objective_func(**params)
            convergence_history.append(value)
            return value
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        execution_time = time.time() - start_time
        
        return OptimizationResult(
            method='linear',
            best_params=study.best_params,
            best_value=study.best_value,
            n_trials=n_trials,
            execution_time=execution_time,
            convergence_history=convergence_history
        )
    
    def test_log_optimization(self, objective_func: Callable, 
                            param_ranges: Dict[str, Tuple[float, float]], 
                            n_trials: int = 50) -> OptimizationResult:
        """Test logarithmic parameter optimization."""
        start_time = time.time()
        convergence_history = []
        
        def objective(trial):
            params = {}
            for param_name, (min_val, max_val) in param_ranges.items():
                # Use log-space sampling
                log_min = np.log(max(min_val, 1e-10))  # Avoid log(0)
                log_max = np.log(max_val)
                log_param = trial.suggest_float(f"log_{param_name}", log_min, log_max)
                params[param_name] = np.exp(log_param)
            
            value = objective_func(**params)
            convergence_history.append(value)
            return value
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        execution_time = time.time() - start_time
        
        return OptimizationResult(
            method='logarithmic',
            best_params=study.best_params,
            best_value=study.best_value,
            n_trials=n_trials,
            execution_time=execution_time,
            convergence_history=convergence_history
        )
    
    def test_fractional_power_optimization(self, objective_func: Callable, 
                                        param_ranges: Dict[str, Tuple[float, float]], 
                                        power: float = 0.5, n_trials: int = 50) -> OptimizationResult:
        """Test fractional power parameter optimization."""
        start_time = time.time()
        convergence_history = []
        
        def objective(trial):
            params = {}
            for param_name, (min_val, max_val) in param_ranges.items():
                # Use fractional power transformation
                raw_param = trial.suggest_float(param_name, 0, 1)
                # Transform to [min_val, max_val] using power function
                normalized = raw_param ** power
                params[param_name] = min_val + normalized * (max_val - min_val)
            
            value = objective_func(**params)
            convergence_history.append(value)
            return value
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        execution_time = time.time() - start_time
        
        return OptimizationResult(
            method=f'fractional_power_{power}',
            best_params=study.best_params,
            best_value=study.best_value,
            n_trials=n_trials,
            execution_time=execution_time,
            convergence_history=convergence_history
        )
    
    def test_sigmoid_optimization(self, objective_func: Callable, 
                                param_ranges: Dict[str, Tuple[float, float]], 
                                n_trials: int = 50) -> OptimizationResult:
        """Test sigmoid parameter optimization."""
        start_time = time.time()
        convergence_history = []
        
        def objective(trial):
            params = {}
            for param_name, (min_val, max_val) in param_ranges.items():
                # Use sigmoid transformation
                raw_param = trial.suggest_float(param_name, -6, 6)  # Wide range for sigmoid
                sigmoid_param = 1 / (1 + np.exp(-raw_param))  # Sigmoid function
                params[param_name] = min_val + sigmoid_param * (max_val - min_val)
            
            value = objective_func(**params)
            convergence_history.append(value)
            return value
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        execution_time = time.time() - start_time
        
        return OptimizationResult(
            method='sigmoid',
            best_params=study.best_params,
            best_value=study.best_value,
            n_trials=n_trials,
            execution_time=execution_time,
            convergence_history=convergence_history
        )
    
    def test_adaptive_optimization(self, objective_func: Callable, 
                                 param_ranges: Dict[str, Tuple[float, float]], 
                                 n_trials: int = 50) -> OptimizationResult:
        """Test adaptive non-linear optimization."""
        start_time = time.time()
        convergence_history = []
        
        def objective(trial):
            params = {}
            for param_name, (min_val, max_val) in param_ranges.items():
                # Adaptive transformation based on parameter range
                range_size = max_val - min_val
                if range_size > 10:  # Large range - use log
                    log_min = np.log(max(min_val, 1e-10))
                    log_max = np.log(max_val)
                    log_param = trial.suggest_float(f"log_{param_name}", log_min, log_max)
                    params[param_name] = np.exp(log_param)
                elif range_size < 1:  # Small range - use sigmoid
                    raw_param = trial.suggest_float(param_name, -6, 6)
                    sigmoid_param = 1 / (1 + np.exp(-raw_param))
                    params[param_name] = min_val + sigmoid_param * (max_val - min_val)
                else:  # Medium range - use fractional power
                    raw_param = trial.suggest_float(param_name, 0, 1)
                    power_param = raw_param ** 0.7
                    params[param_name] = min_val + power_param * (max_val - min_val)
            
            value = objective_func(**params)
            convergence_history.append(value)
            return value
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        execution_time = time.time() - start_time
        
        return OptimizationResult(
            method='adaptive',
            best_params=study.best_params,
            best_value=study.best_value,
            n_trials=n_trials,
            execution_time=execution_time,
            convergence_history=convergence_history
        )
    
    def run_comprehensive_test(self, test_function: str = 'rosenbrock', 
                             n_trials: int = 100) -> Dict[str, OptimizationResult]:
        """Run comprehensive test comparing all optimization methods."""
        logger.info(f"🧪 Running comprehensive test for {test_function} function")
        
        if test_function not in self.test_functions:
            raise ValueError(f"Unknown test function: {test_function}")
        
        objective_func = self.test_functions[test_function]
        
        # Define parameter ranges based on test function
        if test_function in ['rosenbrock', 'rastrigin', 'ackley', 'multi_modal']:
            param_ranges = {'x': (-5, 5), 'y': (-5, 5)}
        elif test_function == 'financial_metric':
            param_ranges = {
                'confidence': (0.1, 0.9),
                'position_size': (0.01, 0.2),
                'leverage': (0.5, 2.0)
            }
        else:
            param_ranges = {'x': (-2, 2), 'y': (-2, 2)}
        
        results = {}
        
        # Test all optimization methods
        methods = [
            ('linear', self.test_linear_optimization),
            ('logarithmic', self.test_log_optimization),
            ('fractional_power_0.3', lambda obj, ranges, n: self.test_fractional_power_optimization(obj, ranges, 0.3, n)),
            ('fractional_power_0.5', lambda obj, ranges, n: self.test_fractional_power_optimization(obj, ranges, 0.5, n)),
            ('fractional_power_0.7', lambda obj, ranges, n: self.test_fractional_power_optimization(obj, ranges, 0.7, n)),
            ('sigmoid', self.test_sigmoid_optimization),
            ('adaptive', self.test_adaptive_optimization)
        ]
        
        for method_name, method_func in methods:
            logger.info(f"🔄 Testing {method_name} optimization...")
            try:
                result = method_func(objective_func, param_ranges, n_trials)
                results[method_name] = result
                logger.info(f"✅ {method_name}: best_value={result.best_value:.6f}, time={result.execution_time:.2f}s")
            except Exception as e:
                logger.error(f"❌ {method_name} failed: {e}")
        
        return results
    
    def analyze_results(self, results: Dict[str, OptimizationResult]) -> pd.DataFrame:
        """Analyze and compare optimization results."""
        analysis_data = []
        
        for method, result in results.items():
            analysis_data.append({
                'method': method,
                'best_value': result.best_value,
                'execution_time': result.execution_time,
                'n_trials': result.n_trials,
                'convergence_rate': self._calculate_convergence_rate(result.convergence_history),
                'final_improvement': self._calculate_final_improvement(result.convergence_history)
            })
        
        df = pd.DataFrame(analysis_data)
        df = df.sort_values('best_value')
        
        return df
    
    def _calculate_convergence_rate(self, history: List[float]) -> float:
        """Calculate convergence rate from optimization history."""
        if len(history) < 10:
            return 0.0
        
        # Calculate improvement in last 20% of trials
        last_portion = int(len(history) * 0.2)
        if last_portion < 2:
            return 0.0
        
        recent_history = history[-last_portion:]
        initial_value = recent_history[0]
        final_value = recent_history[-1]
        
        if initial_value == 0:
            return 0.0
        
        return abs(final_value - initial_value) / abs(initial_value)
    
    def _calculate_final_improvement(self, history: List[float]) -> float:
        """Calculate final improvement from optimization history."""
        if len(history) < 2:
            return 0.0
        
        initial_value = history[0]
        final_value = history[-1]
        
        if initial_value == 0:
            return 0.0
        
        return (initial_value - final_value) / abs(initial_value)
    
    def plot_convergence_comparison(self, results: Dict[str, OptimizationResult], 
                                  save_path: str = None):
        """Plot convergence comparison for all methods."""
        plt.figure(figsize=(12, 8))
        
        for method, result in results.items():
            plt.plot(result.convergence_history, label=method, alpha=0.7)
        
        plt.xlabel('Trial Number')
        plt.ylabel('Objective Value')
        plt.title('Convergence Comparison: Non-Linear Optimization Methods')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"📊 Convergence plot saved to {save_path}")
        
        plt.show()
    
    def plot_parameter_distribution(self, results: Dict[str, OptimizationResult], 
                                  param_name: str, save_path: str = None):
        """Plot parameter distribution for different methods."""
        plt.figure(figsize=(12, 6))
        
        param_values = []
        methods = []
        
        for method, result in results.items():
            if param_name in result.best_params:
                param_values.append(result.best_params[param_name])
                methods.append(method)
        
        if param_values:
            plt.bar(methods, param_values)
            plt.xlabel('Optimization Method')
            plt.ylabel(f'{param_name} Value')
            plt.title(f'Best {param_name} Values by Optimization Method')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"📊 Parameter distribution plot saved to {save_path}")
            
            plt.show()

def main():
    """Main testing function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test non-linear optimization methods')
    parser.add_argument('--test_type', choices=['all', 'logs', 'powers', 'sigmoid', 'adaptive'], 
                       default='all', help='Type of test to run')
    parser.add_argument('--function', choices=['rosenbrock', 'rastrigin', 'ackley', 'financial_metric', 'multi_modal'],
                       default='rosenbrock', help='Test function to use')
    parser.add_argument('--n_trials', type=int, default=100, help='Number of optimization trials')
    parser.add_argument('--save_plots', action='store_true', help='Save plots to files')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path('nonlinear_optimization_results')
    output_dir.mkdir(exist_ok=True)
    
    # Initialize tester
    tester = NonLinearOptimizationTester()
    
    # Run tests based on type
    if args.test_type == 'all':
        logger.info("🚀 Running comprehensive non-linear optimization test")
        results = tester.run_comprehensive_test(args.function, args.n_trials)
        
        # Analyze results
        analysis_df = tester.analyze_results(results)
        logger.info("\n📊 Optimization Results Analysis:")
        logger.info(analysis_df.to_string(index=False))
        
        # Save results
        results_file = output_dir / f'{args.function}_optimization_results.csv'
        analysis_df.to_csv(results_file, index=False)
        logger.info(f"💾 Results saved to {results_file}")
        
        # Create plots
        if args.save_plots:
            convergence_plot = output_dir / f'{args.function}_convergence_comparison.png'
            tester.plot_convergence_comparison(results, str(convergence_plot))
            
            # Plot parameter distributions if applicable
            if args.function in ['rosenbrock', 'rastrigin', 'ackley', 'multi_modal']:
                param_plot = output_dir / f'{args.function}_parameter_distribution.png'
                tester.plot_parameter_distribution(results, 'x', str(param_plot))
    
    else:
        logger.info(f"🧪 Running {args.test_type} optimization test")
        # Run specific test type
        objective_func = tester.test_functions[args.function]
        
        if args.function in ['rosenbrock', 'rastrigin', 'ackley', 'multi_modal']:
            param_ranges = {'x': (-5, 5), 'y': (-5, 5)}
        else:
            param_ranges = {
                'confidence': (0.1, 0.9),
                'position_size': (0.01, 0.2),
                'leverage': (0.5, 2.0)
            }
        
        if args.test_type == 'logs':
            result = tester.test_log_optimization(objective_func, param_ranges, args.n_trials)
        elif args.test_type == 'powers':
            result = tester.test_fractional_power_optimization(objective_func, param_ranges, 0.5, args.n_trials)
        elif args.test_type == 'sigmoid':
            result = tester.test_sigmoid_optimization(objective_func, param_ranges, args.n_trials)
        elif args.test_type == 'adaptive':
            result = tester.test_adaptive_optimization(objective_func, param_ranges, args.n_trials)
        
        logger.info(f"✅ {args.test_type} optimization completed:")
        logger.info(f"   Best value: {result.best_value:.6f}")
        logger.info(f"   Best params: {result.best_params}")
        logger.info(f"   Execution time: {result.execution_time:.2f}s")

if __name__ == "__main__":
    main()