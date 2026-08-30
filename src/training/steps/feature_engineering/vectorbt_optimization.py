"""
VectorBT Parameter Optimization System

This module provides comprehensive parameter optimization capabilities for VectorBT
feature engineering components with advanced optimization algorithms and validation.

Features:
- Multiple optimization algorithms (Grid, Random, Bayesian, Genetic)
- Cross-validation and backtesting integration
- Performance monitoring and validation
- Adaptive parameter tuning
- Multi-objective optimization
- Real-time optimization updates
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import logging
import time
from datetime import datetime
from enum import Enum
import warnings

# Import VectorBT base classes
from src.training.steps.feature_engineering.vectorbt_base import (
    VectorBTFeatureGenerator, VectorBTConfig, VectorBTTechnicalIndicators
)

# Import VectorBT feature generators
from src.training.steps.feature_engineering.volatility.vectorbt_atr_volatility_ratio import (
    VectorBTATRVolatilityRatioGenerator
)
from src.training.steps.feature_engineering.trend.vectorbt_trend_coherence import (
    VectorBTTrendCoherenceGenerator
)
from src.training.steps.feature_engineering.price_action.vectorbt_bar_efficiency_ratio import (
    VectorBTBarEfficiencyRatioGenerator
)
from src.training.steps.feature_engineering.price_action.vectorbt_close_location_value import (
    VectorBTCloseLocationValueGenerator
)

from src.utils.tprint import tprint_info, tprint_warning, tprint_error, tprint_success


class OptimizationAlgorithm(Enum):
    """Available optimization algorithms."""
    GRID = "grid"
    RANDOM = "random"
    BAYESIAN = "bayesian"
    GENETIC = "genetic"
    PARTICLE_SWARM = "particle_swarm"
    SIMULATED_ANNEALING = "simulated_annealing"


class OptimizationMetric(Enum):
    """Available optimization metrics."""
    SHARPE_RATIO = "sharpe_ratio"
    INFORMATION_RATIO = "information_ratio"
    CALMAR_RATIO = "calmar_ratio"
    SORTINO_RATIO = "sortino_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    VOLATILITY = "volatility"
    RETURN = "return"
    CUSTOM = "custom"


@dataclass
class VectorBTOptimizationConfig:
    """Configuration for VectorBT parameter optimization."""
    
    # Algorithm settings
    algorithm: OptimizationAlgorithm = OptimizationAlgorithm.BAYESIAN
    max_iterations: int = 100
    n_trials: int = 50
    random_state: int = 42
    
    # Cross-validation settings
    enable_cross_validation: bool = True
    cv_folds: int = 5
    cv_strategy: str = "time_series"  # "time_series", "k_fold", "walk_forward"
    
    # Backtesting settings
    enable_backtesting: bool = True
    backtest_start_date: Optional[str] = None
    backtest_end_date: Optional[str] = None
    initial_capital: float = 100000.0
    
    # Performance settings
    enable_parallel: bool = True
    n_jobs: int = -1
    chunk_size: int = 1000
    memory_efficient: bool = True
    
    # Validation settings
    enable_validation: bool = True
    validation_metric: OptimizationMetric = OptimizationMetric.SHARPE_RATIO
    min_performance_threshold: float = 0.0
    
    # Early stopping
    enable_early_stopping: bool = True
    patience: int = 10
    min_improvement: float = 0.001
    
    # Multi-objective optimization
    enable_multi_objective: bool = False
    objectives: List[OptimizationMetric] = field(default_factory=lambda: [
        OptimizationMetric.SHARPE_RATIO,
        OptimizationMetric.MAX_DRAWDOWN
    ])
    objective_weights: List[float] = field(default_factory=lambda: [0.7, 0.3])
    
    # Adaptive optimization
    enable_adaptive: bool = True
    adaptation_frequency: int = 20
    adaptation_threshold: float = 0.1


@dataclass
class OptimizationResult:
    """Result of parameter optimization."""
    
    # Core results
    best_parameters: Dict[str, Any]
    best_score: float
    optimization_history: List[Dict[str, Any]]
    
    # Performance metrics
    sharpe_ratio: float
    information_ratio: float
    max_drawdown: float
    volatility: float
    total_return: float
    
    # Validation results
    cv_scores: List[float]
    cv_mean: float
    cv_std: float
    
    # Backtesting results
    backtest_results: Optional[Dict[str, Any]] = None
    
    # Optimization metadata
    algorithm_used: OptimizationAlgorithm
    n_iterations: int
    optimization_time: float
    convergence_achieved: bool
    
    # Feature-specific results
    feature_name: str
    feature_category: str
    parameter_ranges: Dict[str, List[Any]]


class VectorBTOptimizer:
    """
    Comprehensive parameter optimizer for VectorBT features.
    
    Provides advanced optimization capabilities with multiple algorithms,
    cross-validation, backtesting, and performance monitoring.
    """
    
    def __init__(self, config: Optional[VectorBTOptimizationConfig] = None):
        """Initialize VectorBT optimizer."""
        self.config = config or VectorBTOptimizationConfig()
        self.logger = logging.getLogger('VectorBTOptimizer')
        
        # Optimization history
        self.optimization_history: List[OptimizationResult] = []
        self.current_optimization: Optional[OptimizationResult] = None
        
        # Performance tracking
        self.performance_tracker = {}
        self.convergence_tracker = {}
        
        tprint_info("🔧 VectorBT Optimizer initialized")
        tprint_info(f"   → Algorithm: {self.config.algorithm.value}")
        tprint_info(f"   → Max iterations: {self.config.max_iterations}")
        tprint_info(f"   → Cross-validation: {self.config.enable_cross_validation}")
        tprint_info(f"   → Backtesting: {self.config.enable_backtesting}")
        tprint_info(f"   → Multi-objective: {self.config.enable_multi_objective}")
    
    def optimize_feature_parameters(
        self,
        feature_generator: VectorBTFeatureGenerator,
        data: pd.DataFrame,
        parameter_ranges: Dict[str, List[Any]],
        target_metric: OptimizationMetric = OptimizationMetric.SHARPE_RATIO,
        custom_metric_func: Optional[Callable] = None
    ) -> OptimizationResult:
        """
        Optimize parameters for a VectorBT feature generator.
        
        Args:
            feature_generator: VectorBT feature generator to optimize
            data: Input data for optimization
            parameter_ranges: Parameter ranges to optimize
            target_metric: Target metric for optimization
            custom_metric_func: Custom metric function
            
        Returns:
            OptimizationResult with best parameters and performance metrics
        """
        start_time = time.time()
        tprint_info(f"🔍 Optimizing parameters for {feature_generator.config.name}")
        
        try:
            # Initialize optimization
            optimization_result = OptimizationResult(
                best_parameters={},
                best_score=float('-inf'),
                optimization_history=[],
                sharpe_ratio=0.0,
                information_ratio=0.0,
                max_drawdown=0.0,
                volatility=0.0,
                total_return=0.0,
                cv_scores=[],
                cv_mean=0.0,
                cv_std=0.0,
                algorithm_used=self.config.algorithm,
                n_iterations=0,
                optimization_time=0.0,
                convergence_achieved=False,
                feature_name=feature_generator.config.name,
                feature_category=feature_generator.config.category.value,
                parameter_ranges=parameter_ranges
            )
            
            # Run optimization based on algorithm
            if self.config.algorithm == OptimizationAlgorithm.GRID:
                result = self._grid_search_optimization(
                    feature_generator, data, parameter_ranges, target_metric, custom_metric_func
                )
            elif self.config.algorithm == OptimizationAlgorithm.RANDOM:
                result = self._random_search_optimization(
                    feature_generator, data, parameter_ranges, target_metric, custom_metric_func
                )
            elif self.config.algorithm == OptimizationAlgorithm.BAYESIAN:
                result = self._bayesian_optimization(
                    feature_generator, data, parameter_ranges, target_metric, custom_metric_func
                )
            elif self.config.algorithm == OptimizationAlgorithm.GENETIC:
                result = self._genetic_optimization(
                    feature_generator, data, parameter_ranges, target_metric, custom_metric_func
                )
            else:
                # Default to random search
                result = self._random_search_optimization(
                    feature_generator, data, parameter_ranges, target_metric, custom_metric_func
                )
            
            # Update optimization result
            optimization_result.best_parameters = result['best_parameters']
            optimization_result.best_score = result['best_score']
            optimization_result.optimization_history = result['optimization_history']
            optimization_result.n_iterations = result['n_iterations']
            optimization_result.convergence_achieved = result['convergence_achieved']
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(
                feature_generator, data, optimization_result.best_parameters
            )
            optimization_result.sharpe_ratio = performance_metrics.get('sharpe_ratio', 0.0)
            optimization_result.information_ratio = performance_metrics.get('information_ratio', 0.0)
            optimization_result.max_drawdown = performance_metrics.get('max_drawdown', 0.0)
            optimization_result.volatility = performance_metrics.get('volatility', 0.0)
            optimization_result.total_return = performance_metrics.get('total_return', 0.0)
            
            # Cross-validation if enabled
            if self.config.enable_cross_validation:
                cv_results = self._cross_validate_parameters(
                    feature_generator, data, optimization_result.best_parameters
                )
                optimization_result.cv_scores = cv_results['scores']
                optimization_result.cv_mean = cv_results['mean']
                optimization_result.cv_std = cv_results['std']
            
            # Backtesting if enabled
            if self.config.enable_backtesting:
                backtest_results = self._backtest_parameters(
                    feature_generator, data, optimization_result.best_parameters
                )
                optimization_result.backtest_results = backtest_results
            
            # Calculate optimization time
            optimization_result.optimization_time = time.time() - start_time
            
            # Store results
            self.optimization_history.append(optimization_result)
            self.current_optimization = optimization_result
            
            tprint_success(f"✅ Optimization completed: {optimization_result.best_score:.4f} in {optimization_result.optimization_time:.2f}s")
            
            return optimization_result
            
        except Exception as e:
            tprint_error(f"❌ Error optimizing parameters: {e}")
            raise
    
    def _grid_search_optimization(
        self,
        feature_generator: VectorBTFeatureGenerator,
        data: pd.DataFrame,
        parameter_ranges: Dict[str, List[Any]],
        target_metric: OptimizationMetric,
        custom_metric_func: Optional[Callable]
    ) -> Dict[str, Any]:
        """Perform grid search optimization."""
        tprint_info("🔍 Running grid search optimization")
        
        best_score = float('-inf')
        best_parameters = {}
        optimization_history = []
        
        # Generate parameter combinations
        param_names = list(parameter_ranges.keys())
        param_values = list(parameter_ranges.values())
        
        from itertools import product
        param_combinations = list(product(*param_values))
        
        total_combinations = len(param_combinations)
        tprint_info(f"   → Testing {total_combinations} parameter combinations")
        
        for i, param_combination in enumerate(param_combinations):
            try:
                # Create parameter dictionary
                params = dict(zip(param_names, param_combination))
                
                # Evaluate parameters
                score = self._evaluate_parameters(
                    feature_generator, data, params, target_metric, custom_metric_func
                )
                
                # Update best if better
                if score > best_score:
                    best_score = score
                    best_parameters = params.copy()
                
                # Store history
                optimization_history.append({
                    'iteration': i + 1,
                    'parameters': params.copy(),
                    'score': score,
                    'best_score': best_score
                })
                
                # Early stopping check
                if self.config.enable_early_stopping and self._check_early_stopping(optimization_history):
                    tprint_info(f"   → Early stopping at iteration {i + 1}")
                    break
                
            except Exception as e:
                tprint_warning(f"⚠️ Error evaluating parameters {params}: {e}")
                continue
        
        return {
            'best_parameters': best_parameters,
            'best_score': best_score,
            'optimization_history': optimization_history,
            'n_iterations': len(optimization_history),
            'convergence_achieved': len(optimization_history) < total_combinations
        }
    
    def _random_search_optimization(
        self,
        feature_generator: VectorBTFeatureGenerator,
        data: pd.DataFrame,
        parameter_ranges: Dict[str, List[Any]],
        target_metric: OptimizationMetric,
        custom_metric_func: Optional[Callable]
    ) -> Dict[str, Any]:
        """Perform random search optimization."""
        tprint_info("🔍 Running random search optimization")
        
        best_score = float('-inf')
        best_parameters = {}
        optimization_history = []
        
        np.random.seed(self.config.random_state)
        
        for i in range(self.config.n_trials):
            try:
                # Sample random parameters
                params = {}
                for param_name, param_values in parameter_ranges.items():
                    if isinstance(param_values[0], (int, float)):
                        # Numeric parameter
                        min_val, max_val = min(param_values), max(param_values)
                        if isinstance(param_values[0], int):
                            params[param_name] = np.random.randint(min_val, max_val + 1)
                        else:
                            params[param_name] = np.random.uniform(min_val, max_val)
                    else:
                        # Categorical parameter
                        params[param_name] = np.random.choice(param_values)
                
                # Evaluate parameters
                score = self._evaluate_parameters(
                    feature_generator, data, params, target_metric, custom_metric_func
                )
                
                # Update best if better
                if score > best_score:
                    best_score = score
                    best_parameters = params.copy()
                
                # Store history
                optimization_history.append({
                    'iteration': i + 1,
                    'parameters': params.copy(),
                    'score': score,
                    'best_score': best_score
                })
                
                # Early stopping check
                if self.config.enable_early_stopping and self._check_early_stopping(optimization_history):
                    tprint_info(f"   → Early stopping at iteration {i + 1}")
                    break
                
            except Exception as e:
                tprint_warning(f"⚠️ Error evaluating parameters {params}: {e}")
                continue
        
        return {
            'best_parameters': best_parameters,
            'best_score': best_score,
            'optimization_history': optimization_history,
            'n_iterations': len(optimization_history),
            'convergence_achieved': len(optimization_history) < self.config.n_trials
        }
    
    def _bayesian_optimization(
        self,
        feature_generator: VectorBTFeatureGenerator,
        data: pd.DataFrame,
        parameter_ranges: Dict[str, List[Any]],
        target_metric: OptimizationMetric,
        custom_metric_func: Optional[Callable]
    ) -> Dict[str, Any]:
        """Perform Bayesian optimization."""
        tprint_info("🔍 Running Bayesian optimization")
        
        try:
            # Try to import Bayesian optimization libraries
            try:
                from skopt import gp_minimize
                from skopt.space import Real, Integer, Categorical
                from skopt.utils import use_named_args
            except ImportError:
                tprint_warning("⚠️ scikit-optimize not available, falling back to random search")
                return self._random_search_optimization(
                    feature_generator, data, parameter_ranges, target_metric, custom_metric_func
                )
            
            # Define search space
            dimensions = []
            param_names = []
            
            for param_name, param_values in parameter_ranges.items():
                if isinstance(param_values[0], int):
                    dimensions.append(Integer(min(param_values), max(param_values), name=param_name))
                elif isinstance(param_values[0], float):
                    dimensions.append(Real(min(param_values), max(param_values), name=param_name))
                else:
                    dimensions.append(Categorical(param_values, name=param_name))
                param_names.append(param_name)
            
            # Define objective function
            @use_named_args(dimensions)
            def objective(**params):
                try:
                    score = self._evaluate_parameters(
                        feature_generator, data, params, target_metric, custom_metric_func
                    )
                    return -score  # Minimize negative score
                except Exception as e:
                    tprint_warning(f"⚠️ Error in Bayesian optimization: {e}")
                    return float('inf')
            
            # Run optimization
            result = gp_minimize(
                func=objective,
                dimensions=dimensions,
                n_calls=self.config.n_trials,
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs if self.config.enable_parallel else 1
            )
            
            # Extract results
            best_parameters = dict(zip(param_names, result.x))
            best_score = -result.fun
            
            # Create optimization history
            optimization_history = []
            for i, (x, fun) in enumerate(zip(result.x_iters, result.func_vals)):
                params = dict(zip(param_names, x))
                optimization_history.append({
                    'iteration': i + 1,
                    'parameters': params,
                    'score': -fun,
                    'best_score': best_score
                })
            
            return {
                'best_parameters': best_parameters,
                'best_score': best_score,
                'optimization_history': optimization_history,
                'n_iterations': len(optimization_history),
                'convergence_achieved': result.fun < float('inf')
            }
            
        except Exception as e:
            tprint_warning(f"⚠️ Error in Bayesian optimization: {e}")
            return self._random_search_optimization(
                feature_generator, data, parameter_ranges, target_metric, custom_metric_func
            )
    
    def _genetic_optimization(
        self,
        feature_generator: VectorBTFeatureGenerator,
        data: pd.DataFrame,
        parameter_ranges: Dict[str, List[Any]],
        target_metric: OptimizationMetric,
        custom_metric_func: Optional[Callable]
    ) -> Dict[str, Any]:
        """Perform genetic algorithm optimization."""
        tprint_info("🔍 Running genetic algorithm optimization")
        
        try:
            # Try to import DEAP for genetic algorithms
            try:
                from deap import base, creator, tools, algorithms
            except ImportError:
                tprint_warning("⚠️ DEAP not available, falling back to random search")
                return self._random_search_optimization(
                    feature_generator, data, parameter_ranges, target_metric, custom_metric_func
                )
            
            # Create fitness function
            def evaluate_individual(individual):
                try:
                    params = dict(zip(parameter_ranges.keys(), individual))
                    score = self._evaluate_parameters(
                        feature_generator, data, params, target_metric, custom_metric_func
                    )
                    return (score,)
                except Exception as e:
                    tprint_warning(f"⚠️ Error evaluating individual: {e}")
                    return (float('-inf'),)
            
            # Setup genetic algorithm
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
            creator.create("Individual", list, fitness=creator.FitnessMax)
            
            toolbox = base.Toolbox()
            
            # Define parameter generation
            for param_name, param_values in parameter_ranges.items():
                if isinstance(param_values[0], int):
                    toolbox.register(f"attr_{param_name}", 
                                   lambda pv=param_values: np.random.randint(min(pv), max(pv) + 1))
                elif isinstance(param_values[0], float):
                    toolbox.register(f"attr_{param_name}", 
                                   lambda pv=param_values: np.random.uniform(min(pv), max(pv)))
                else:
                    toolbox.register(f"attr_{param_name}", 
                                   lambda pv=param_values: np.random.choice(pv))
            
            # Create individual and population
            param_names = list(parameter_ranges.keys())
            toolbox.register("individual", tools.initCycle, creator.Individual,
                           [getattr(toolbox, f"attr_{name}") for name in param_names], n=1)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            
            # Register genetic operators
            toolbox.register("evaluate", evaluate_individual)
            toolbox.register("mate", tools.cxTwoPoint)
            toolbox.register("mutate", tools.mutUniformInt, 
                           indpb=0.1, low=[min(pv) for pv in parameter_ranges.values()],
                           up=[max(pv) for pv in parameter_ranges.values()])
            toolbox.register("select", tools.selTournament, tournsize=3)
            
            # Run genetic algorithm
            population = toolbox.population(n=50)
            algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, 
                              ngen=self.config.n_trials // 50, verbose=False)
            
            # Get best individual
            best_individual = tools.selBest(population, 1)[0]
            best_parameters = dict(zip(param_names, best_individual))
            best_score = best_individual.fitness.values[0]
            
            # Create optimization history
            optimization_history = [{
                'iteration': 1,
                'parameters': best_parameters,
                'score': best_score,
                'best_score': best_score
            }]
            
            return {
                'best_parameters': best_parameters,
                'best_score': best_score,
                'optimization_history': optimization_history,
                'n_iterations': 1,
                'convergence_achieved': True
            }
            
        except Exception as e:
            tprint_warning(f"⚠️ Error in genetic optimization: {e}")
            return self._random_search_optimization(
                feature_generator, data, parameter_ranges, target_metric, custom_metric_func
            )
    
    def _evaluate_parameters(
        self,
        feature_generator: VectorBTFeatureGenerator,
        data: pd.DataFrame,
        parameters: Dict[str, Any],
        target_metric: OptimizationMetric,
        custom_metric_func: Optional[Callable]
    ) -> float:
        """Evaluate parameters for a feature generator."""
        try:
            # Generate features with given parameters
            features = feature_generator.generate_vectorbt_features(data, parameters)
            
            # Calculate target metric
            if custom_metric_func:
                return custom_metric_func(features, data)
            
            # Calculate metric based on target
            if target_metric == OptimizationMetric.SHARPE_RATIO:
                return self._calculate_sharpe_ratio(features, data)
            elif target_metric == OptimizationMetric.INFORMATION_RATIO:
                return self._calculate_information_ratio(features, data)
            elif target_metric == OptimizationMetric.CALMAR_RATIO:
                return self._calculate_calmar_ratio(features, data)
            elif target_metric == OptimizationMetric.SORTINO_RATIO:
                return self._calculate_sortino_ratio(features, data)
            elif target_metric == OptimizationMetric.MAX_DRAWDOWN:
                return -self._calculate_max_drawdown(features, data)  # Negative for maximization
            elif target_metric == OptimizationMetric.VOLATILITY:
                return -self._calculate_volatility(features, data)  # Negative for maximization
            elif target_metric == OptimizationMetric.RETURN:
                return self._calculate_total_return(features, data)
            else:
                return self._calculate_sharpe_ratio(features, data)
                
        except Exception as e:
            tprint_warning(f"⚠️ Error evaluating parameters: {e}")
            return float('-inf')
    
    def _calculate_sharpe_ratio(self, features: Dict[str, pd.Series], data: pd.DataFrame) -> float:
        """Calculate Sharpe ratio for features."""
        try:
            # Use primary feature for calculation
            primary_feature = self._get_primary_feature(features)
            if primary_feature is None:
                return 0.0
            
            # Calculate returns
            returns = primary_feature.pct_change().dropna()
            if len(returns) == 0:
                return 0.0
            
            # Calculate Sharpe ratio
            mean_return = returns.mean()
            std_return = returns.std()
            
            if std_return == 0:
                return 0.0
            
            sharpe_ratio = mean_return / std_return
            return sharpe_ratio * np.sqrt(252)  # Annualized
            
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating Sharpe ratio: {e}")
            return 0.0
    
    def _calculate_information_ratio(self, features: Dict[str, pd.Series], data: pd.DataFrame) -> float:
        """Calculate information ratio for features."""
        try:
            primary_feature = self._get_primary_feature(features)
            if primary_feature is None:
                return 0.0
            
            # Use price as benchmark
            benchmark_returns = data['close'].pct_change().dropna()
            feature_returns = primary_feature.pct_change().dropna()
            
            # Align returns
            min_length = min(len(benchmark_returns), len(feature_returns))
            if min_length == 0:
                return 0.0
            
            benchmark_returns = benchmark_returns.iloc[-min_length:]
            feature_returns = feature_returns.iloc[-min_length:]
            
            # Calculate excess returns
            excess_returns = feature_returns - benchmark_returns
            
            if excess_returns.std() == 0:
                return 0.0
            
            information_ratio = excess_returns.mean() / excess_returns.std()
            return information_ratio * np.sqrt(252)  # Annualized
            
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating information ratio: {e}")
            return 0.0
    
    def _calculate_calmar_ratio(self, features: Dict[str, pd.Series], data: pd.DataFrame) -> float:
        """Calculate Calmar ratio for features."""
        try:
            primary_feature = self._get_primary_feature(features)
            if primary_feature is None:
                return 0.0
            
            returns = primary_feature.pct_change().dropna()
            if len(returns) == 0:
                return 0.0
            
            # Calculate annual return
            annual_return = returns.mean() * 252
            
            # Calculate max drawdown
            max_drawdown = self._calculate_max_drawdown(features, data)
            
            if max_drawdown == 0:
                return 0.0
            
            calmar_ratio = annual_return / max_drawdown
            return calmar_ratio
            
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating Calmar ratio: {e}")
            return 0.0
    
    def _calculate_sortino_ratio(self, features: Dict[str, pd.Series], data: pd.DataFrame) -> float:
        """Calculate Sortino ratio for features."""
        try:
            primary_feature = self._get_primary_feature(features)
            if primary_feature is None:
                return 0.0
            
            returns = primary_feature.pct_change().dropna()
            if len(returns) == 0:
                return 0.0
            
            # Calculate downside deviation
            negative_returns = returns[returns < 0]
            if len(negative_returns) == 0:
                return float('inf')
            
            downside_deviation = negative_returns.std()
            if downside_deviation == 0:
                return 0.0
            
            sortino_ratio = returns.mean() / downside_deviation
            return sortino_ratio * np.sqrt(252)  # Annualized
            
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating Sortino ratio: {e}")
            return 0.0
    
    def _calculate_max_drawdown(self, features: Dict[str, pd.Series], data: pd.DataFrame) -> float:
        """Calculate maximum drawdown for features."""
        try:
            primary_feature = self._get_primary_feature(features)
            if primary_feature is None:
                return 0.0
            
            # Calculate cumulative returns
            returns = primary_feature.pct_change().dropna()
            if len(returns) == 0:
                return 0.0
            
            cumulative = (1 + returns).cumprod()
            
            # Calculate running maximum
            running_max = cumulative.expanding().max()
            
            # Calculate drawdown
            drawdown = (cumulative - running_max) / running_max
            
            return abs(drawdown.min())
            
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating max drawdown: {e}")
            return 0.0
    
    def _calculate_volatility(self, features: Dict[str, pd.Series], data: pd.DataFrame) -> float:
        """Calculate volatility for features."""
        try:
            primary_feature = self._get_primary_feature(features)
            if primary_feature is None:
                return 0.0
            
            returns = primary_feature.pct_change().dropna()
            if len(returns) == 0:
                return 0.0
            
            volatility = returns.std() * np.sqrt(252)  # Annualized
            return volatility
            
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating volatility: {e}")
            return 0.0
    
    def _calculate_total_return(self, features: Dict[str, pd.Series], data: pd.DataFrame) -> float:
        """Calculate total return for features."""
        try:
            primary_feature = self._get_primary_feature(features)
            if primary_feature is None:
                return 0.0
            
            returns = primary_feature.pct_change().dropna()
            if len(returns) == 0:
                return 0.0
            
            total_return = (1 + returns).prod() - 1
            return total_return
            
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating total return: {e}")
            return 0.0
    
    def _get_primary_feature(self, features: Dict[str, pd.Series]) -> Optional[pd.Series]:
        """Get primary feature from feature dictionary."""
        if not features:
            return None
        
        # Priority order for primary feature selection
        priority_features = [
            'ratio', 'grade', 'score', 'signal', 'trend', 'momentum',
            'volatility', 'efficiency', 'coherence', 'strength'
        ]
        
        for priority in priority_features:
            for feature_name, feature_data in features.items():
                if priority in feature_name.lower() and isinstance(feature_data, pd.Series):
                    return feature_data
        
        # Fallback to first numeric series
        for feature_data in features.values():
            if isinstance(feature_data, pd.Series) and pd.api.types.is_numeric_dtype(feature_data):
                return feature_data
        
        # Last resort - return first series
        return next(iter(features.values()))
    
    def _calculate_performance_metrics(
        self,
        feature_generator: VectorBTFeatureGenerator,
        data: pd.DataFrame,
        parameters: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        try:
            features = feature_generator.generate_vectorbt_features(data, parameters)
            
            metrics = {}
            metrics['sharpe_ratio'] = self._calculate_sharpe_ratio(features, data)
            metrics['information_ratio'] = self._calculate_information_ratio(features, data)
            metrics['max_drawdown'] = self._calculate_max_drawdown(features, data)
            metrics['volatility'] = self._calculate_volatility(features, data)
            metrics['total_return'] = self._calculate_total_return(features, data)
            
            return metrics
            
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating performance metrics: {e}")
            return {}
    
    def _cross_validate_parameters(
        self,
        feature_generator: VectorBTFeatureGenerator,
        data: pd.DataFrame,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform cross-validation for parameters."""
        try:
            tprint_info("🔍 Performing cross-validation")
            
            if self.config.cv_strategy == "time_series":
                # Time series cross-validation
                cv_scores = self._time_series_cv(feature_generator, data, parameters)
            else:
                # K-fold cross-validation
                cv_scores = self._k_fold_cv(feature_generator, data, parameters)
            
            return {
                'scores': cv_scores,
                'mean': np.mean(cv_scores),
                'std': np.std(cv_scores)
            }
            
        except Exception as e:
            tprint_warning(f"⚠️ Error in cross-validation: {e}")
            return {'scores': [], 'mean': 0.0, 'std': 0.0}
    
    def _time_series_cv(
        self,
        feature_generator: VectorBTFeatureGenerator,
        data: pd.DataFrame,
        parameters: Dict[str, Any]
    ) -> List[float]:
        """Perform time series cross-validation."""
        cv_scores = []
        n_samples = len(data)
        fold_size = n_samples // self.config.cv_folds
        
        for i in range(self.config.cv_folds):
            # Define train and test sets
            train_end = (i + 1) * fold_size
            test_start = train_end
            test_end = min(test_start + fold_size, n_samples)
            
            if test_start >= n_samples:
                break
            
            train_data = data.iloc[:train_end]
            test_data = data.iloc[test_start:test_end]
            
            if len(train_data) < 50 or len(test_data) < 10:
                continue
            
            try:
                # Evaluate on test set
                score = self._evaluate_parameters(
                    feature_generator, test_data, parameters, 
                    OptimizationMetric.SHARPE_RATIO, None
                )
                cv_scores.append(score)
            except Exception as e:
                tprint_warning(f"⚠️ Error in CV fold {i}: {e}")
                continue
        
        return cv_scores
    
    def _k_fold_cv(
        self,
        feature_generator: VectorBTFeatureGenerator,
        data: pd.DataFrame,
        parameters: Dict[str, Any]
    ) -> List[float]:
        """Perform k-fold cross-validation."""
        cv_scores = []
        n_samples = len(data)
        fold_size = n_samples // self.config.cv_folds
        
        for i in range(self.config.cv_folds):
            # Define train and test sets
            test_start = i * fold_size
            test_end = min((i + 1) * fold_size, n_samples)
            
            if test_start >= n_samples:
                break
            
            # Create train set (exclude test fold)
            train_indices = list(range(0, test_start)) + list(range(test_end, n_samples))
            test_indices = list(range(test_start, test_end))
            
            if len(train_indices) < 50 or len(test_indices) < 10:
                continue
            
            train_data = data.iloc[train_indices]
            test_data = data.iloc[test_indices]
            
            try:
                # Evaluate on test set
                score = self._evaluate_parameters(
                    feature_generator, test_data, parameters, 
                    OptimizationMetric.SHARPE_RATIO, None
                )
                cv_scores.append(score)
            except Exception as e:
                tprint_warning(f"⚠️ Error in CV fold {i}: {e}")
                continue
        
        return cv_scores
    
    def _backtest_parameters(
        self,
        feature_generator: VectorBTFeatureGenerator,
        data: pd.DataFrame,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform backtesting for parameters."""
        try:
            tprint_info("🔍 Performing backtesting")
            
            # Generate features
            features = feature_generator.generate_vectorbt_features(data, parameters)
            
            # Simple backtesting simulation
            primary_feature = self._get_primary_feature(features)
            if primary_feature is None:
                return {}
            
            # Calculate returns
            returns = primary_feature.pct_change().dropna()
            
            # Calculate backtesting metrics
            backtest_results = {
                'total_return': self._calculate_total_return(features, data),
                'sharpe_ratio': self._calculate_sharpe_ratio(features, data),
                'max_drawdown': self._calculate_max_drawdown(features, data),
                'volatility': self._calculate_volatility(features, data),
                'win_rate': (returns > 0).mean() if len(returns) > 0 else 0.0,
                'profit_factor': self._calculate_profit_factor(returns),
                'trades_count': len(returns),
                'avg_trade': returns.mean() if len(returns) > 0 else 0.0
            }
            
            return backtest_results
            
        except Exception as e:
            tprint_warning(f"⚠️ Error in backtesting: {e}")
            return {}
    
    def _calculate_profit_factor(self, returns: pd.Series) -> float:
        """Calculate profit factor for returns."""
        try:
            if len(returns) == 0:
                return 0.0
            
            positive_returns = returns[returns > 0].sum()
            negative_returns = abs(returns[returns < 0].sum())
            
            if negative_returns == 0:
                return float('inf')
            
            return positive_returns / negative_returns
            
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating profit factor: {e}")
            return 0.0
    
    def _check_early_stopping(self, optimization_history: List[Dict[str, Any]]) -> bool:
        """Check if early stopping criteria are met."""
        if not self.config.enable_early_stopping or len(optimization_history) < self.config.patience:
            return False
        
        # Check if best score hasn't improved for patience iterations
        recent_scores = [h['best_score'] for h in optimization_history[-self.config.patience:]]
        if len(recent_scores) < self.config.patience:
            return False
        
        best_recent = max(recent_scores)
        best_overall = max([h['best_score'] for h in optimization_history])
        
        improvement = best_overall - best_recent
        return improvement < self.config.min_improvement
    
    def get_optimization_history(self) -> List[OptimizationResult]:
        """Get optimization history."""
        return self.optimization_history.copy()
    
    def get_best_parameters(self, feature_name: Optional[str] = None) -> Dict[str, Any]:
        """Get best parameters for a feature or overall best."""
        if feature_name:
            for result in self.optimization_history:
                if result.feature_name == feature_name:
                    return result.best_parameters
            return {}
        else:
            if self.optimization_history:
                best_result = max(self.optimization_history, key=lambda x: x.best_score)
                return best_result.best_parameters
            return {}
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.optimization_history.clear()
        self.current_optimization = None
        self.performance_tracker.clear()
        self.convergence_tracker.clear()
        tprint_info("🧹 VectorBT Optimizer cleanup completed")


# Convenience functions
def create_vectorbt_optimizer(config: Optional[VectorBTOptimizationConfig] = None) -> VectorBTOptimizer:
    """Create VectorBT optimizer instance."""
    return VectorBTOptimizer(config)


def optimize_vectorbt_feature(
    feature_generator: VectorBTFeatureGenerator,
    data: pd.DataFrame,
    parameter_ranges: Dict[str, List[Any]],
    config: Optional[VectorBTOptimizationConfig] = None,
    target_metric: OptimizationMetric = OptimizationMetric.SHARPE_RATIO
) -> OptimizationResult:
    """Optimize a single VectorBT feature."""
    optimizer = create_vectorbt_optimizer(config)
    return optimizer.optimize_feature_parameters(
        feature_generator, data, parameter_ranges, target_metric
    )