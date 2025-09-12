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
    optimization_method: str = 'adaptive_grid_search'  # 'grid_search', 'adaptive_grid_search', 'genetic', 'scipy'
    
    # Parameter ranges to optimize
    touch_tolerance_range: Tuple[float, float] = (0.001, 0.01)  # 0.1% to 1%
    min_bounce_strength_range: Tuple[float, float] = (0.0005, 0.005)  # 0.05% to 0.5%
    volume_threshold_range: Tuple[float, float] = (1.0, 3.0)  # 1x to 3x average volume
    min_touches_range: Tuple[int, int] = (1, 8)  # 1 to 8 minimum touches (changed from 2-8)
    max_hold_time_range: Tuple[int, int] = (1, 48)  # 1 to 48 hours
    
    # Quality scoring multiplier ranges (more intuitive than weights)
    success_rate_multiplier_range: Tuple[float, float] = (0.5, 2.0)  # 0.5x to 2.0x emphasis
    bounce_strength_multiplier_range: Tuple[float, float] = (0.5, 2.0)
    volume_confirmation_multiplier_range: Tuple[float, float] = (0.5, 2.0)
    time_persistence_multiplier_range: Tuple[float, float] = (0.5, 2.0)
    touch_frequency_multiplier_range: Tuple[float, float] = (0.5, 2.0)
    
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
            elif self.config.optimization_method == 'adaptive_grid_search':
                return self._adaptive_grid_search_optimization(backtest_results, market_data, strategy)
            elif self.config.optimization_method == 'genetic':
                return self._genetic_algorithm_optimization(backtest_results, market_data, strategy)
            elif self.config.optimization_method == 'scipy':
                return self._scipy_optimization(backtest_results, market_data, strategy)
            else:
                self.logger.error(f"Unknown optimization method: {self.config.optimization_method}")
                raise ValueError(f"Unknown optimization method: {self.config.optimization_method}")
                
        except Exception as e:
            self.logger.error(f"Parameter optimization failed: {e}")
            raise RuntimeError(f"Parameter optimization failed: {e}")
    
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
    
    def _adaptive_grid_search_optimization(self, backtest_results: List[Any], 
                                         market_data: pd.DataFrame, 
                                         strategy: str) -> ParameterOptimizationResult:
        """Adaptive grid search optimization with coarse-to-fine approach."""
        self.logger.info("Starting adaptive grid search optimization")
        
        # Stage 1: Coarse grid search
        self.logger.info("Stage 1: Coarse grid search")
        coarse_result = self._coarse_grid_search(backtest_results, market_data, strategy)
        
        if not coarse_result.optimization_success:
            self.logger.warning("Coarse grid search failed, using data-driven parameters")
            return self._create_data_driven_result(backtest_results, market_data)
        
        # Stage 2: Fine grid search around best parameters
        self.logger.info("Stage 2: Fine grid search around best parameters")
        fine_result = self._fine_grid_search(backtest_results, market_data, coarse_result.best_parameters)
        
        if fine_result.optimization_success and fine_result.best_score > coarse_result.best_score:
            self.logger.info(f"Fine grid search improved score: {coarse_result.best_score:.4f} -> {fine_result.best_score:.4f}")
            return fine_result
        else:
            self.logger.info("Fine grid search did not improve results, using coarse results")
            return coarse_result
    
    def _coarse_grid_search(self, backtest_results: List[Any], 
                           market_data: pd.DataFrame, 
                           strategy: str) -> ParameterOptimizationResult:
        """Coarse grid search with fewer parameter combinations."""
        self.logger.info("Running coarse grid search")
        
        # Use fewer parameter combinations for coarse search
        if strategy == 'minimal':
            param_grid = self._create_minimal_parameter_grid()
        elif strategy == 'conservative':
            param_grid = self._create_conservative_parameter_grid()
        else:
            # Create coarse grid for standard strategy
            param_grid = self._create_coarse_parameter_grid()
        
        self.logger.info(f"Coarse parameter grid size: {len(param_grid)} combinations")
        
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
                
                if i % 5 == 0:
                    self.logger.info(f"Coarse search: evaluated {i+1}/{len(param_grid)} combinations")
                    
            except Exception as e:
                self.logger.warning(f"Failed to evaluate parameters {params}: {e}")
                continue
        
        return ParameterOptimizationResult(
            best_parameters=best_parameters,
            best_score=best_score,
            optimization_method='coarse_grid_search',
            n_trials=len(param_grid),
            optimization_success=len(parameter_scores) > 0,
            parameter_scores=parameter_scores,
            optimization_details={'strategy': strategy, 'stage': 'coarse'}
        )
    
    def _fine_grid_search(self, backtest_results: List[Any], 
                         market_data: pd.DataFrame, 
                         best_parameters: Dict[str, Any]) -> ParameterOptimizationResult:
        """Fine grid search around the best parameters from coarse search."""
        self.logger.info("Running fine grid search around best parameters")
        
        # Create fine grid around best parameters
        param_grid = self._create_fine_parameter_grid(best_parameters)
        
        self.logger.info(f"Fine parameter grid size: {len(param_grid)} combinations")
        
        # Evaluate each parameter combination
        best_score = -np.inf
        best_parameters_fine = {}
        parameter_scores = []
        
        for i, params in enumerate(param_grid):
            try:
                score = self._evaluate_parameters(params, backtest_results, market_data)
                parameter_scores.append((params, score))
                
                if score > best_score:
                    best_score = score
                    best_parameters_fine = params
                
                if i % 10 == 0:
                    self.logger.info(f"Fine search: evaluated {i+1}/{len(param_grid)} combinations")
                    
            except Exception as e:
                self.logger.warning(f"Failed to evaluate parameters {params}: {e}")
                continue
        
        return ParameterOptimizationResult(
            best_parameters=best_parameters_fine,
            best_score=best_score,
            optimization_method='fine_grid_search',
            n_trials=len(param_grid),
            optimization_success=len(parameter_scores) > 0,
            parameter_scores=parameter_scores,
            optimization_details={'stage': 'fine', 'coarse_best_score': best_score}
        )
    
    def _create_coarse_parameter_grid(self) -> List[Dict[str, Any]]:
        """Create coarse parameter grid for initial search."""
        # Use fewer parameter combinations for coarse search
        touch_tolerance_values = np.linspace(*self.config.touch_tolerance_range, 3)
        min_bounce_strength_values = np.linspace(*self.config.min_bounce_strength_range, 3)
        volume_threshold_values = np.linspace(*self.config.volume_threshold_range, 3)
        min_touches_values = [1, 3, 5, 7]  # Fewer touch values
        max_hold_time_values = [6, 24, 48]  # Fewer time values
        
        # Use only default weight combination for coarse search
        weight_combination = [0.3, 0.25, 0.2, 0.15, 0.1]  # Default weights
        
        param_grid = []
        for tt, mbs, vt, mt, mht in product(touch_tolerance_values, min_bounce_strength_values, 
                                           volume_threshold_values, min_touches_values, max_hold_time_values):
            params = {
                'touch_tolerance': tt,
                'min_bounce_strength': mbs,
                'volume_threshold_multiplier': vt,
                'min_touches_required': mt,
                'max_hold_time': mht,
                'success_rate_multiplier': weight_combination[0],
                'bounce_strength_multiplier': weight_combination[1],
                'volume_confirmation_multiplier': weight_combination[2],
                'time_persistence_multiplier': weight_combination[3],
                'touch_frequency_multiplier': weight_combination[4]
            }
            param_grid.append(params)
        
        return param_grid
    
    def _create_fine_parameter_grid(self, best_parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create fine parameter grid around best parameters using adaptive search.
        
        Algorithm: Adaptive Local Search with Multi-Dimensional Refinement
        1. Use smaller step sizes around the best parameters
        2. Apply different refinement strategies for different parameter types
        3. Use golden ratio search for continuous parameters
        4. Use discrete neighborhood search for integer parameters
        5. Apply parameter-specific sensitivity analysis
        """
        param_grid = []
        
        # Define adaptive fine search ranges based on parameter sensitivity
        fine_ranges = {
            'touch_tolerance': 0.0005,  # ±0.05% around best (smaller range for precision)
            'min_bounce_strength': 0.0002,  # ±0.02% around best (very sensitive parameter)
            'volume_threshold_multiplier': 0.1,  # ±0.1 around best (moderate sensitivity)
            'min_touches_required': 1,  # ±1 around best (discrete parameter)
            'max_hold_time': 3,  # ±3 hours around best (time sensitivity)
            # Multiplier parameters use percentage-based ranges
            'success_rate_multiplier': 0.2,  # ±20% around best
            'bounce_strength_multiplier': 0.2,
            'volume_confirmation_multiplier': 0.2,
            'time_persistence_multiplier': 0.2,
            'touch_frequency_multiplier': 0.2,
        }
        
        # Create fine grid using adaptive search strategy
        for param, range_size in fine_ranges.items():
            best_value = best_parameters.get(param, 0)
            
            if param in ['min_touches_required', 'max_hold_time']:
                # Integer parameters: discrete neighborhood search
                min_val = max(1, int(best_value - range_size))
                max_val = int(best_value + range_size)
                values = list(range(min_val, max_val + 1))
                
            elif param.endswith('_multiplier'):
                # Multiplier parameters: percentage-based search
                min_val = max(0.1, best_value * (1 - range_size))
                max_val = best_value * (1 + range_size)
                values = np.linspace(min_val, max_val, 5)
                
            else:
                # Continuous parameters: golden ratio search for efficiency
                min_val = max(0.0001, best_value - range_size)
                max_val = best_value + range_size
                # Use golden ratio for more efficient search
                phi = (1 + np.sqrt(5)) / 2  # Golden ratio
                values = []
                for i in range(5):
                    if i == 0:
                        values.append(min_val)
                    elif i == 4:
                        values.append(max_val)
                    else:
                        # Golden ratio spacing
                        ratio = (phi - 1) ** i
                        values.append(min_val + (max_val - min_val) * ratio)
                values = sorted(values)
            
            # Create parameter combinations with the refined values
            for value in values:
                params = best_parameters.copy()
                params[param] = value
                param_grid.append(params)
        
        # Add multi-parameter combinations for interaction effects
        # This helps capture parameter interactions that single-parameter search might miss
        interaction_combinations = self._create_parameter_interaction_combinations(best_parameters)
        param_grid.extend(interaction_combinations)
        
        return param_grid
    
    def _create_parameter_interaction_combinations(self, best_parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create parameter combinations to test interaction effects."""
        interaction_grid = []
        
        # Test key parameter interactions
        interaction_pairs = [
            ('touch_tolerance', 'min_bounce_strength'),
            ('volume_threshold_multiplier', 'min_touches_required'),
            ('success_rate_multiplier', 'bounce_strength_multiplier'),
        ]
        
        for param1, param2 in interaction_pairs:
            if param1 in best_parameters and param2 in best_parameters:
                val1 = best_parameters[param1]
                val2 = best_parameters[param2]
                
                # Create 2x2 grid around the best values
                if param1 in ['min_touches_required', 'max_hold_time']:
                    vals1 = [max(1, val1 - 1), val1, val1 + 1]
                else:
                    vals1 = [val1 * 0.9, val1, val1 * 1.1]
                
                if param2 in ['min_touches_required', 'max_hold_time']:
                    vals2 = [max(1, val2 - 1), val2, val2 + 1]
                else:
                    vals2 = [val2 * 0.9, val2, val2 * 1.1]
                
                # Create combinations
                for v1 in vals1:
                    for v2 in vals2:
                        params = best_parameters.copy()
                        params[param1] = v1
                        params[param2] = v2
                        interaction_grid.append(params)
        
        return interaction_grid
    
    def _create_data_driven_result(self, backtest_results: List[Any], 
                                 market_data: pd.DataFrame) -> ParameterOptimizationResult:
        """Create data-driven parameters without optimization."""
        self.logger.info("Creating data-driven parameters")
        
        # Calculate data-driven parameters
        data_driven_params = self._calculate_data_driven_parameters(backtest_results, market_data)
        
        return ParameterOptimizationResult(
            best_parameters=data_driven_params,
            best_score=0.0,
            optimization_method='data_driven',
            n_trials=0,
            optimization_success=True,
            parameter_scores=[],
            optimization_details={'method': 'data_driven_calculation'}
        )
    
    def _calculate_data_driven_parameters(self, backtest_results: List[Any], 
                                        market_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate data-driven parameters from market data and backtest results."""
        try:
            # Calculate touch tolerance from price volatility
            returns = market_data['close'].pct_change().dropna()
            price_volatility = returns.rolling(20).std().mean()
            touch_tolerance = max(0.001, min(0.01, price_volatility * 2))
            
            # Calculate min bounce strength from historical bounces
            high_low_returns = (market_data['high'] - market_data['low']) / market_data['close']
            min_bounce_strength = max(0.0005, high_low_returns.quantile(0.25))
            
            # Calculate volume threshold from volume distribution
            if 'volume' in market_data.columns:
                avg_volume = market_data['volume'].rolling(20).mean().mean()
                volume_volatility = market_data['volume'].pct_change().rolling(20).std().mean()
                volume_threshold_multiplier = 1.5 + volume_volatility
            else:
                volume_threshold_multiplier = 1.5
            
            # Calculate optimal min touches from backtest results
            if backtest_results:
                touch_counts = [r.total_touches for r in backtest_results]
                success_rates = [r.success_rate for r in backtest_results]
                
                # Find touch count that maximizes success rate
                touch_success_data = {}
                for touches, success_rate in zip(touch_counts, success_rates):
                    if touches not in touch_success_data:
                        touch_success_data[touches] = []
                    touch_success_data[touches].append(success_rate)
                
                avg_success_by_touches = {}
                for touches, success_rates in touch_success_data.items():
                    if len(success_rates) >= 2:
                        avg_success_by_touches[touches] = np.mean(success_rates)
                
                if avg_success_by_touches:
                    best_touches = max(avg_success_by_touches.items(), key=lambda x: x[1])[0]
                    min_touches_required = max(1, min(best_touches, 6))
                else:
                    min_touches_required = 3
            else:
                min_touches_required = 3
            
            # Calculate max hold time from market characteristics
            if 'timestamp' in market_data.columns:
                time_diffs = market_data['timestamp'].diff().dt.total_seconds() / 3600
                avg_time_diff = time_diffs.mean()
                max_hold_time = max(1, min(48, int(avg_time_diff * 10)))
            else:
                max_hold_time = 24
            
            return {
                'touch_tolerance': touch_tolerance,
                'min_bounce_strength': min_bounce_strength,
                'volume_threshold_multiplier': volume_threshold_multiplier,
                'min_touches_required': min_touches_required,
                'max_hold_time': max_hold_time,
                'success_rate_multiplier': 1.0,
                'bounce_strength_multiplier': 1.0,
                'volume_confirmation_multiplier': 1.0,
                'time_persistence_multiplier': 1.0,
                'touch_frequency_multiplier': 1.0
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate data-driven parameters: {e}")
            # Return conservative defaults
            return {
                'touch_tolerance': 0.002,
                'min_bounce_strength': 0.001,
                'volume_threshold_multiplier': 1.5,
                'min_touches_required': 3,
                'max_hold_time': 24,
                'success_rate_multiplier': 1.0,
                'bounce_strength_multiplier': 1.0,
                'volume_confirmation_multiplier': 1.0,
                'time_persistence_multiplier': 1.0,
                'touch_frequency_multiplier': 1.0
            }
    
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
        min_touches_values = [1, 2, 3, 4]
        
        param_grid = []
        for tt, mbs, vt, mt in product(touch_tolerance_values, min_bounce_strength_values, 
                                      volume_threshold_values, min_touches_values):
            params = {
                'touch_tolerance': tt,
                'min_bounce_strength': mbs,
                'volume_threshold_multiplier': vt,
                'min_touches_required': mt,
                'max_hold_time': 24,  # Fixed for small samples
                'success_rate_multiplier': 1.0,
                'bounce_strength_multiplier': 1.0,
                'volume_confirmation_multiplier': 1.0,
                'time_persistence_multiplier': 1.0,
                'touch_frequency_multiplier': 1.0
            }
            param_grid.append(params)
        
        return param_grid
    
    def _create_conservative_parameter_grid(self) -> List[Dict[str, Any]]:
        """Create conservative parameter grid for medium samples."""
        # Use moderate number of parameter combinations
        touch_tolerance_values = np.linspace(*self.config.touch_tolerance_range, 4)
        min_bounce_strength_values = np.linspace(*self.config.min_bounce_strength_range, 4)
        volume_threshold_values = np.linspace(*self.config.volume_threshold_range, 4)
        min_touches_values = [1, 2, 3, 4, 5]
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
                'success_rate_multiplier': 1.0,
                'bounce_strength_multiplier': 1.0,
                'volume_confirmation_multiplier': 1.0,
                'time_persistence_multiplier': 1.0,
                'touch_frequency_multiplier': 1.0
            }
            param_grid.append(params)
        
        return param_grid
    
    def _create_standard_parameter_grid(self) -> List[Dict[str, Any]]:
        """Create standard parameter grid for large samples."""
        # Use full parameter grid
        touch_tolerance_values = np.linspace(*self.config.touch_tolerance_range, self.config.grid_search_steps)
        min_bounce_strength_values = np.linspace(*self.config.min_bounce_strength_range, self.config.grid_search_steps)
        volume_threshold_values = np.linspace(*self.config.volume_threshold_range, self.config.grid_search_steps)
        min_touches_values = list(range(self.config.min_touches_range[0], self.config.min_touches_range[1] + 1))
        max_hold_time_values = [6, 12, 24, 36, 48]
        
        # Multiplier combinations (more intuitive than weights)
        multiplier_combinations = [
            [1.0, 1.0, 1.0, 1.0, 1.0],   # Default (equal emphasis)
            [1.5, 0.8, 0.8, 0.8, 0.8],   # Focus on success rate
            [0.8, 1.5, 0.8, 0.8, 0.8],   # Focus on bounce strength
            [0.8, 0.8, 1.5, 0.8, 0.8],   # Focus on volume
            [0.8, 0.8, 0.8, 1.5, 0.8],   # Focus on time persistence
        ]
        
        param_grid = []
        for tt, mbs, vt, mt, mht, multipliers in product(touch_tolerance_values, min_bounce_strength_values, 
                                                        volume_threshold_values, min_touches_values, 
                                                        max_hold_time_values, multiplier_combinations):
            params = {
                'touch_tolerance': tt,
                'min_bounce_strength': mbs,
                'volume_threshold_multiplier': vt,
                'min_touches_required': mt,
                'max_hold_time': mht,
                'success_rate_multiplier': multipliers[0],
                'bounce_strength_multiplier': multipliers[1],
                'volume_confirmation_multiplier': multipliers[2],
                'time_persistence_multiplier': multipliers[3],
                'touch_frequency_multiplier': multipliers[4]
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
            self.config.success_rate_multiplier_range,
            self.config.bounce_strength_multiplier_range,
            self.config.volume_confirmation_multiplier_range,
            self.config.time_persistence_multiplier_range,
            self.config.touch_frequency_multiplier_range
        ]
        
        return bounds
    
    def _params_array_to_dict(self, params: np.ndarray, bounds: List[Tuple[float, float]]) -> Dict[str, Any]:
        """Convert parameter array to dictionary."""
        param_names = [
            'touch_tolerance', 'min_bounce_strength', 'volume_threshold_multiplier',
            'min_touches_required', 'max_hold_time', 'success_rate_multiplier',
            'bounce_strength_multiplier', 'volume_confirmation_multiplier',
            'time_persistence_multiplier', 'touch_frequency_multiplier'
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
            
            # Calculate quality score using multipliers
            quality_score = (
                success_rate * params['success_rate_multiplier'] +
                bounce_strength * 100 * params['bounce_strength_multiplier'] +  # Scale bounce strength
                volume_confirmation * params['volume_confirmation_multiplier'] +
                time_persistence * params['time_persistence_multiplier'] +
                touch_frequency * params['touch_frequency_multiplier']
            )
            
            # Normalize by total multiplier sum to keep score in [0, 1] range
            total_multiplier = (
                params['success_rate_multiplier'] +
                params['bounce_strength_multiplier'] +
                params['volume_confirmation_multiplier'] +
                params['time_persistence_multiplier'] +
                params['touch_frequency_multiplier']
            )
            
            if total_multiplier > 0:
                quality_score = quality_score / total_multiplier
            
            return min(max(quality_score, 0.0), 1.0)  # Clamp to [0, 1]
            
        except Exception as e:
            self.logger.warning(f"Quality score calculation failed: {e}")
            return 0.0
    

def get_parameter_optimization_engine(config: Optional[ParameterOptimizationConfig] = None) -> ParameterOptimizationEngine:
    """Get a parameter optimization engine instance."""
    return ParameterOptimizationEngine(config)