"""
Parameter Optimization Engine for SR Level Detection

This module focuses on optimizing the core parameters for SR level detection and quality assessment,
rather than training ML models. It optimizes parameters like:
- Volume thresholds for SR confirmation
- Minimum touches required
- Bounce strength requirements
- Touch tolerance levels
- Quality scoring weights

The goal is to find the optimal parameters that best identify high-quality SR levels.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from itertools import product
import warnings
from scipy.optimize import minimize, differential_evolution
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class ParameterOptimizationConfig:
    """Configuration for parameter optimization."""
    # Optimization method
    optimization_method: str = 'grid_search'  # 'grid_search', 'bayesian', 'genetic', 'scipy'
    
    # Parameter ranges to optimize
    touch_tolerance_range: Tuple[float, float] = (0.001, 0.01)  # 0.1% to 1%
    min_bounce_strength_range: Tuple[float, float] = (0.0005, 0.005)  # 0.05% to 0.5%
    volume_threshold_range: Tuple[float, float] = (1.0, 3.0)  # 1x to 3x average volume
    min_touches_range: Tuple[int, int] = (2, 8)  # 2 to 8 minimum touches
    max_hold_time_range: Tuple[int, int] = (1, 48)  # 1 to 48 hours
    
    # Quality scoring weight ranges
    success_rate_weight_range: Tuple[float, float] = (0.1, 0.5)
    bounce_strength_weight_range: Tuple[float, float] = (0.1, 0.4)
    volume_confirmation_weight_range: Tuple[float, float] = (0.1, 0.3)
    time_persistence_weight_range: Tuple[float, float] = (0.1, 0.3)
    touch_frequency_weight_range: Tuple[float, float] = (0.05, 0.2)
    
    # Optimization settings
    n_trials: int = 100  # Number of parameter combinations to try
    cv_folds: int = 3  # Cross-validation folds
    objective_metric: str = 'quality_score_correlation'  # 'quality_score_correlation', 'success_rate', 'composite'
    
    # Small sample handling
    min_samples_for_optimization: int = 10
    adaptive_optimization: bool = True  # Adapt optimization based on sample size
    
    # Grid search settings
    grid_search_steps: int = 5  # Steps per parameter for grid search
    
    # Genetic algorithm settings
    population_size: int = 20
    generations: int = 10
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8

@dataclass
class ParameterOptimizationResult:
    """Result of parameter optimization."""
    best_parameters: Dict[str, Any]
    best_score: float
    optimization_method: str
    n_trials: int
    optimization_success: bool
    parameter_scores: List[Tuple[Dict[str, Any], float]] = field(default_factory=list)
    optimization_details: Dict[str, Any] = field(default_factory=dict)

class ParameterOptimizationEngine:
    """Engine for optimizing SR level detection parameters."""
    
    def __init__(self, config: Optional[ParameterOptimizationConfig] = None):
        self.config = config or ParameterOptimizationConfig()
        self.logger = logger.getChild('ParameterOptimizationEngine')
        
        self.logger.info("Initializing ParameterOptimizationEngine")
        self.logger.info(f"Optimization method: {self.config.optimization_method}")
        self.logger.info(f"Objective metric: {self.config.objective_metric}")
    
    def optimize_parameters(self, backtest_results: List[Any], 
                          market_data: pd.DataFrame) -> ParameterOptimizationResult:
        """
        Optimize SR level detection parameters based on backtesting results.
        
        Args:
            backtest_results: List of BacktestResult objects
            market_data: Market data used for backtesting
            
        Returns:
            ParameterOptimizationResult with optimized parameters
        """
        try:
            self.logger.info(f"Starting parameter optimization with {len(backtest_results)} results")
            
            if len(backtest_results) < self.config.min_samples_for_optimization:
                self.logger.warning(f"Insufficient samples: {len(backtest_results)} < {self.config.min_samples_for_optimization}")
                return self._create_fallback_result(backtest_results)
            
            # Determine optimization strategy based on sample size
            if self.config.adaptive_optimization:
                strategy = self._determine_optimization_strategy(len(backtest_results))
                self.logger.info(f"Using {strategy} optimization strategy")
            else:
                strategy = 'standard'
            
            # Run optimization based on method
            if self.config.optimization_method == 'grid_search':
                return self._grid_search_optimization(backtest_results, market_data, strategy)
            elif self.config.optimization_method == 'genetic':
                return self._genetic_algorithm_optimization(backtest_results, market_data, strategy)
            elif self.config.optimization_method == 'scipy':
                return self._scipy_optimization(backtest_results, market_data, strategy)
            else:
                self.logger.error(f"Unknown optimization method: {self.config.optimization_method}")
                return self._create_fallback_result(backtest_results)
                
        except Exception as e:
            self.logger.error(f"Parameter optimization failed: {e}")
            return self._create_fallback_result(backtest_results)
    
    def _determine_optimization_strategy(self, n_samples: int) -> str:
        """Determine optimization strategy based on sample size."""
        if n_samples < 20:
            return 'minimal'
        elif n_samples < 50:
            return 'conservative'
        else:
            return 'standard'
    
    def _grid_search_optimization(self, backtest_results: List[Any], 
                                 market_data: pd.DataFrame, 
                                 strategy: str) -> ParameterOptimizationResult:
        """Grid search optimization for parameters."""
        self.logger.info("Starting grid search optimization")
        
        # Define parameter grid based on strategy
        if strategy == 'minimal':
            param_grid = self._create_minimal_parameter_grid()
        elif strategy == 'conservative':
            param_grid = self._create_conservative_parameter_grid()
        else:
            param_grid = self._create_standard_parameter_grid()
        
        self.logger.info(f"Parameter grid size: {len(param_grid)} combinations")
        
        # Evaluate each parameter combination
        best_score = -np.inf
        best_parameters = {}
        parameter_scores = []
        
        for i, params in enumerate(param_grid):
            try:
                score = self._evaluate_parameters(params, backtest_results, market_data)
                parameter_scores.append((params, score))
                
                if score > best_score:
                    best_score = score
                    best_parameters = params
                
                if i % 10 == 0:
                    self.logger.info(f"Evaluated {i+1}/{len(param_grid)} parameter combinations")
                    
            except Exception as e:
                self.logger.warning(f"Failed to evaluate parameters {params}: {e}")
                continue
        
        return ParameterOptimizationResult(
            best_parameters=best_parameters,
            best_score=best_score,
            optimization_method='grid_search',
            n_trials=len(param_grid),
            optimization_success=len(parameter_scores) > 0,
            parameter_scores=parameter_scores,
            optimization_details={'strategy': strategy}
        )
    
    def _genetic_algorithm_optimization(self, backtest_results: List[Any], 
                                      market_data: pd.DataFrame, 
                                      strategy: str) -> ParameterOptimizationResult:
        """Genetic algorithm optimization for parameters."""
        self.logger.info("Starting genetic algorithm optimization")
        
        # Define parameter bounds
        bounds = self._get_parameter_bounds(strategy)
        
        def objective_function(params):
            """Objective function for genetic algorithm."""
            try:
                param_dict = self._params_array_to_dict(params, bounds)
                score = self._evaluate_parameters(param_dict, backtest_results, market_data)
                return -score  # Minimize negative score
            except Exception as e:
                self.logger.warning(f"Objective function failed: {e}")
                return 1.0  # Return high penalty for failed evaluation
        
        # Run genetic algorithm
        result = differential_evolution(
            objective_function,
            bounds,
            maxiter=self.config.generations,
            popsize=self.config.population_size,
            mutation=self.config.mutation_rate,
            recombination=self.config.crossover_rate,
            seed=42
        )
        
        if result.success:
            best_params = self._params_array_to_dict(result.x, bounds)
            best_score = -result.fun
            
            return ParameterOptimizationResult(
                best_parameters=best_params,
                best_score=best_score,
                optimization_method='genetic_algorithm',
                n_trials=result.nfev,
                optimization_success=True,
                optimization_details={
                    'strategy': strategy,
                    'generations': result.nit,
                    'function_evaluations': result.nfev
                }
            )
        else:
            self.logger.warning("Genetic algorithm optimization failed")
            return self._create_fallback_result(backtest_results)
    
    def _scipy_optimization(self, backtest_results: List[Any], 
                           market_data: pd.DataFrame, 
                           strategy: str) -> ParameterOptimizationResult:
        """Scipy optimization for parameters."""
        self.logger.info("Starting scipy optimization")
        
        # Define parameter bounds
        bounds = self._get_parameter_bounds(strategy)
        
        def objective_function(params):
            """Objective function for scipy optimization."""
            try:
                param_dict = self._params_array_to_dict(params, bounds)
                score = self._evaluate_parameters(param_dict, backtest_results, market_data)
                return -score  # Minimize negative score
            except Exception as e:
                self.logger.warning(f"Objective function failed: {e}")
                return 1.0
        
        # Initial guess (middle of bounds)
        x0 = [(bounds[i][0] + bounds[i][1]) / 2 for i in range(len(bounds))]
        
        # Run optimization
        result = minimize(
            objective_function,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100}
        )
        
        if result.success:
            best_params = self._params_array_to_dict(result.x, bounds)
            best_score = -result.fun
            
            return ParameterOptimizationResult(
                best_parameters=best_params,
                best_score=best_score,
                optimization_method='scipy_optimization',
                n_trials=result.nfev,
                optimization_success=True,
                optimization_details={
                    'strategy': strategy,
                    'iterations': result.nit,
                    'function_evaluations': result.nfev
                }
            )
        else:
            self.logger.warning("Scipy optimization failed")
            return self._create_fallback_result(backtest_results)
    
    def _create_minimal_parameter_grid(self) -> List[Dict[str, Any]]:
        """Create minimal parameter grid for small samples."""
        # Use fewer parameter combinations for small samples
        touch_tolerance_values = np.linspace(*self.config.touch_tolerance_range, 3)
        min_bounce_strength_values = np.linspace(*self.config.min_bounce_strength_range, 3)
        volume_threshold_values = np.linspace(*self.config.volume_threshold_range, 3)
        min_touches_values = [2, 3, 4]
        
        param_grid = []
        for tt, mbs, vt, mt in product(touch_tolerance_values, min_bounce_strength_values, 
                                      volume_threshold_values, min_touches_values):
            params = {
                'touch_tolerance': tt,
                'min_bounce_strength': mbs,
                'volume_threshold_multiplier': vt,
                'min_touches_required': mt,
                'max_hold_time': 24,  # Fixed for small samples
                'success_rate_weight': 0.3,
                'bounce_strength_weight': 0.25,
                'volume_confirmation_weight': 0.2,
                'time_persistence_weight': 0.15,
                'touch_frequency_weight': 0.1
            }
            param_grid.append(params)
        
        return param_grid
    
    def _create_conservative_parameter_grid(self) -> List[Dict[str, Any]]:
        """Create conservative parameter grid for medium samples."""
        # Use moderate number of parameter combinations
        touch_tolerance_values = np.linspace(*self.config.touch_tolerance_range, 4)
        min_bounce_strength_values = np.linspace(*self.config.min_bounce_strength_range, 4)
        volume_threshold_values = np.linspace(*self.config.volume_threshold_range, 4)
        min_touches_values = [2, 3, 4, 5]
        max_hold_time_values = [12, 24, 36]
        
        param_grid = []
        for tt, mbs, vt, mt, mht in product(touch_tolerance_values, min_bounce_strength_values, 
                                           volume_threshold_values, min_touches_values, max_hold_time_values):
            params = {
                'touch_tolerance': tt,
                'min_bounce_strength': mbs,
                'volume_threshold_multiplier': vt,
                'min_touches_required': mt,
                'max_hold_time': mht,
                'success_rate_weight': 0.3,
                'bounce_strength_weight': 0.25,
                'volume_confirmation_weight': 0.2,
                'time_persistence_weight': 0.15,
                'touch_frequency_weight': 0.1
            }
            param_grid.append(params)
        
        return param_grid
    
    def _create_standard_parameter_grid(self) -> List[Dict[str, Any]]:
        """Create standard parameter grid for large samples."""
        # Use full parameter grid
        touch_tolerance_values = np.linspace(*self.config.touch_tolerance_range, self.config.grid_search_steps)
        min_bounce_strength_values = np.linspace(*self.config.min_bounce_strength_range, self.config.grid_search_steps)
        volume_threshold_values = np.linspace(*self.config.volume_threshold_range, self.config.grid_search_steps)
        min_touches_values = list(range(*self.config.min_touches_range))
        max_hold_time_values = [6, 12, 24, 36, 48]
        
        # Weight combinations
        weight_combinations = [
            [0.3, 0.25, 0.2, 0.15, 0.1],  # Default
            [0.4, 0.2, 0.2, 0.1, 0.1],   # Focus on success rate
            [0.2, 0.4, 0.2, 0.1, 0.1],   # Focus on bounce strength
            [0.25, 0.25, 0.3, 0.1, 0.1], # Focus on volume
            [0.2, 0.2, 0.2, 0.3, 0.1],   # Focus on time persistence
        ]
        
        param_grid = []
        for tt, mbs, vt, mt, mht, weights in product(touch_tolerance_values, min_bounce_strength_values, 
                                                    volume_threshold_values, min_touches_values, 
                                                    max_hold_time_values, weight_combinations):
            params = {
                'touch_tolerance': tt,
                'min_bounce_strength': mbs,
                'volume_threshold_multiplier': vt,
                'min_touches_required': mt,
                'max_hold_time': mht,
                'success_rate_weight': weights[0],
                'bounce_strength_weight': weights[1],
                'volume_confirmation_weight': weights[2],
                'time_persistence_weight': weights[3],
                'touch_frequency_weight': weights[4]
            }
            param_grid.append(params)
        
        return param_grid
    
    def _get_parameter_bounds(self, strategy: str) -> List[Tuple[float, float]]:
        """Get parameter bounds for optimization."""
        bounds = [
            self.config.touch_tolerance_range,
            self.config.min_bounce_strength_range,
            self.config.volume_threshold_range,
            (float(self.config.min_touches_range[0]), float(self.config.min_touches_range[1])),
            (float(self.config.max_hold_time_range[0]), float(self.config.max_hold_time_range[1])),
            self.config.success_rate_weight_range,
            self.config.bounce_strength_weight_range,
            self.config.volume_confirmation_weight_range,
            self.config.time_persistence_weight_range,
            self.config.touch_frequency_weight_range
        ]
        
        return bounds
    
    def _params_array_to_dict(self, params: np.ndarray, bounds: List[Tuple[float, float]]) -> Dict[str, Any]:
        """Convert parameter array to dictionary."""
        param_names = [
            'touch_tolerance', 'min_bounce_strength', 'volume_threshold_multiplier',
            'min_touches_required', 'max_hold_time', 'success_rate_weight',
            'bounce_strength_weight', 'volume_confirmation_weight',
            'time_persistence_weight', 'touch_frequency_weight'
        ]
        
        param_dict = {}
        for i, name in enumerate(param_names):
            if name in ['min_touches_required', 'max_hold_time']:
                param_dict[name] = int(round(params[i]))
            else:
                param_dict[name] = params[i]
        
        return param_dict
    
    def _evaluate_parameters(self, params: Dict[str, Any], 
                           backtest_results: List[Any], 
                           market_data: pd.DataFrame) -> float:
        """Evaluate parameter set by recalculating quality scores."""
        try:
            # Recalculate quality scores with new parameters
            recalculated_scores = []
            original_scores = []
            
            for result in backtest_results:
                # Store original score
                original_scores.append(result.quality_score)
                
                # Recalculate with new parameters
                new_score = self._calculate_quality_score_with_params(result, params)
                recalculated_scores.append(new_score)
            
            # Calculate objective metric
            if self.config.objective_metric == 'quality_score_correlation':
                # Maximize correlation between original and recalculated scores
                correlation, _ = pearsonr(original_scores, recalculated_scores)
                return correlation if not np.isnan(correlation) else 0.0
            
            elif self.config.objective_metric == 'success_rate':
                # Maximize average success rate
                success_rates = [r.success_rate for r in backtest_results]
                return np.mean(success_rates)
            
            elif self.config.objective_metric == 'composite':
                # Composite metric combining multiple factors
                correlation, _ = pearsonr(original_scores, recalculated_scores)
                correlation = correlation if not np.isnan(correlation) else 0.0
                
                success_rates = [r.success_rate for r in backtest_results]
                avg_success_rate = np.mean(success_rates)
                
                # Combine correlation and success rate
                composite_score = 0.6 * correlation + 0.4 * avg_success_rate
                return composite_score
            
            else:
                # Default to correlation
                correlation, _ = pearsonr(original_scores, recalculated_scores)
                return correlation if not np.isnan(correlation) else 0.0
                
        except Exception as e:
            self.logger.warning(f"Parameter evaluation failed: {e}")
            return 0.0
    
    def _calculate_quality_score_with_params(self, result: Any, params: Dict[str, Any]) -> float:
        """Calculate quality score with given parameters."""
        try:
            # Extract features
            success_rate = result.success_rate
            bounce_strength = result.avg_bounce_strength
            volume_confirmation = min(result.total_volume_at_level / 10000, 1.0)  # Normalize volume
            time_persistence = result.time_persistence
            touch_frequency = min(result.total_touches / 10, 1.0)  # Normalize touches
            
            # Apply volume threshold filter
            if result.total_volume_at_level < params['volume_threshold_multiplier'] * 1000:  # Assume 1000 is avg volume
                volume_confirmation = 0.0
            
            # Apply touch count filter
            if result.total_touches < params['min_touches_required']:
                touch_frequency = 0.0
            
            # Calculate weighted score
            quality_score = (
                success_rate * params['success_rate_weight'] +
                bounce_strength * 100 * params['bounce_strength_weight'] +  # Scale bounce strength
                volume_confirmation * params['volume_confirmation_weight'] +
                time_persistence * params['time_persistence_weight'] +
                touch_frequency * params['touch_frequency_weight']
            )
            
            return min(max(quality_score, 0.0), 1.0)  # Clamp to [0, 1]
            
        except Exception as e:
            self.logger.warning(f"Quality score calculation failed: {e}")
            return 0.0
    
    def _create_fallback_result(self, backtest_results: List[Any]) -> ParameterOptimizationResult:
        """Create fallback result when optimization fails."""
        # Use default parameters
        default_params = {
            'touch_tolerance': 0.002,
            'min_bounce_strength': 0.001,
            'volume_threshold_multiplier': 1.5,
            'min_touches_required': 3,
            'max_hold_time': 24,
            'success_rate_weight': 0.3,
            'bounce_strength_weight': 0.25,
            'volume_confirmation_weight': 0.2,
            'time_persistence_weight': 0.15,
            'touch_frequency_weight': 0.1
        }
        
        return ParameterOptimizationResult(
            best_parameters=default_params,
            best_score=0.0,
            optimization_method='fallback',
            n_trials=0,
            optimization_success=False,
            optimization_details={'reason': 'insufficient_samples_or_optimization_failed'}
        )

def get_parameter_optimization_engine(config: Optional[ParameterOptimizationConfig] = None) -> ParameterOptimizationEngine:
    """Get a parameter optimization engine instance."""
    return ParameterOptimizationEngine(config)