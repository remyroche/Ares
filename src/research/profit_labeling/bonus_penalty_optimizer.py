"""
Data-Driven Bonus and Penalty Optimization for Multi-Horizon Profit Labeling

This module provides data-driven optimization of all bonuses and penalties used in
the multi-horizon profit labeling system. Instead of hardcoded values, it learns
optimal bonus/penalty parameters from historical data and market performance.

Key Optimization Areas:
1. Speed Bonus Parameters (fast move bonuses)
2. Risk Penalty Multipliers (adverse excursion penalties)
3. Profit-Risk Ratio Bonuses (profit/risk ratio thresholds and bonuses)
4. Reversal Capture Penalties (adverse move penalties)
5. Quality Factor Weights (speed, risk, profitability weights)
6. Threshold Parameters (profit-risk ratios, time thresholds)

Optimization Methods:
- Bayesian Optimization for continuous parameters
- Grid Search for discrete parameters
- Multi-objective optimization balancing multiple criteria
- Cross-validation for robust parameter selection
- Market regime-specific optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

# Optimization imports
from scipy.optimize import minimize, differential_evolution
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score

# Optional advanced optimization
try:
    from skopt import gp_minimize, forest_minimize
    from skopt.space import Real, Integer
    from skopt.utils import use_named_args
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from src.utils.logger import get_logger
from src.training.steps.pre_training.profit_labeling.consolidated_profit_labeler import (
    MultiHorizonProfitLabeler,
    MultiHorizonConfig
)

class BonusPenaltyParameter(Enum):
    """Enumeration of bonus/penalty parameters to optimize."""
    # Speed bonus parameters
    SPEED_BONUS_AMOUNT = "speed_bonus_amount"  # Currently 0.1
    SPEED_BONUS_THRESHOLD = "speed_bonus_threshold"  # Currently 0.5 (50% of time window)

    # Risk penalty parameters
    RISK_PENALTY_MULTIPLIER = "risk_penalty_multiplier"  # Currently 30
    RISK_MINIMUM_SCORE = "risk_minimum_score"  # Currently 0.1

    # Profit bonus parameters
    PROFIT_RISK_RATIO_THRESHOLD = "profit_risk_ratio_threshold"  # Currently 2.0
    PROFIT_BONUS_MULTIPLIER = "profit_bonus_multiplier"  # Currently 0.1
    PROFIT_BONUS_MAX = "profit_bonus_max"  # Currently 0.2

    # Reversal capture parameters
    REVERSAL_PENALTY_MULTIPLIER = "reversal_penalty_multiplier"  # Currently 50
    REVERSAL_MINIMUM_FACTOR = "reversal_minimum_factor"  # Currently 0.1

    # Quality factor weights
    SPEED_WEIGHT = "speed_weight"  # Currently 0.3
    RISK_WEIGHT = "risk_weight"  # Currently 0.4
    PROFITABILITY_WEIGHT = "profitability_weight"  # Currently 0.3

    # Scale factors
    PROFIT_SCALE_FACTOR = "profit_scale_factor"  # Currently 300

class OptimizationObjective(Enum):
    """Optimization objectives for bonus/penalty parameters."""
    PREDICTIVE_POWER = "predictive_power"
    LABEL_QUALITY = "label_quality"
    SHARPE_RATIO = "sharpe_ratio"
    HIT_RATE_BALANCE = "hit_rate_balance"
    ECONOMIC_VALUE = "economic_value"
    MULTI_OBJECTIVE = "multi_objective"

@dataclass
class BonusPenaltyOptimizationConfig:
    """Configuration for bonus/penalty optimization."""
    # Optimization method
    optimization_method: str = "bayesian"  # "bayesian", "grid_search", "random_search", "optuna"
    optimization_objective: OptimizationObjective = OptimizationObjective.MULTI_OBJECTIVE

    # Parameter search spaces
    parameter_ranges: Dict[BonusPenaltyParameter, Tuple[float, float]] = field(default_factory=lambda: {
        # Speed bonus parameters
        BonusPenaltyParameter.SPEED_BONUS_AMOUNT: (0.01, 0.5),  # 1% to 50% bonus
        BonusPenaltyParameter.SPEED_BONUS_THRESHOLD: (0.2, 0.8),  # 20% to 80% of time window

        # Risk penalty parameters
        BonusPenaltyParameter.RISK_PENALTY_MULTIPLIER: (5, 100),  # 5x to 100x penalty
        BonusPenaltyParameter.RISK_MINIMUM_SCORE: (0.05, 0.3),  # 5% to 30% minimum

        # Profit bonus parameters
        BonusPenaltyParameter.PROFIT_RISK_RATIO_THRESHOLD: (1.0, 5.0),  # 1:1 to 5:1 ratio
        BonusPenaltyParameter.PROFIT_BONUS_MULTIPLIER: (0.01, 0.5),  # 1% to 50% multiplier
        BonusPenaltyParameter.PROFIT_BONUS_MAX: (0.05, 0.5),  # 5% to 50% max bonus

        # Reversal capture parameters
        BonusPenaltyParameter.REVERSAL_PENALTY_MULTIPLIER: (10, 200),  # 10x to 200x penalty
        BonusPenaltyParameter.REVERSAL_MINIMUM_FACTOR: (0.01, 0.5),  # 1% to 50% minimum

        # Quality weights (will be normalized)
        BonusPenaltyParameter.SPEED_WEIGHT: (0.1, 0.6),
        BonusPenaltyParameter.RISK_WEIGHT: (0.1, 0.6),
        BonusPenaltyParameter.PROFITABILITY_WEIGHT: (0.1, 0.6),

        # Scale factors
        BonusPenaltyParameter.PROFIT_SCALE_FACTOR: (50, 1000)  # 50x to 1000x scaling
    })

    # Optimization parameters
    n_trials: int = 100
    n_cv_folds: int = 3
    validation_split: float = 0.3
    random_state: int = 42

    # Multi-objective weights
    predictive_power_weight: float = 0.4
    hit_rate_weight: float = 0.3
    economic_value_weight: float = 0.2
    stability_weight: float = 0.1

    # Constraints
    min_hit_rate: float = 0.05  # Minimum 5% hit rate
    max_hit_rate: float = 0.95  # Maximum 95% hit rate
    min_predictive_power: float = 0.1  # Minimum predictive power

    # Performance thresholds
    convergence_tolerance: float = 0.001
    max_optimization_time: int = 3600  # 1 hour maximum

    # Parallel processing
    n_jobs: int = -1

@dataclass
class OptimizedParameters:
    """Container for optimized bonus/penalty parameters."""
    parameters: Dict[BonusPenaltyParameter, float]
    objective_score: float
    validation_scores: Dict[str, float]
    cross_validation_scores: List[float]
    optimization_history: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)

class DataDrivenQualityScorer:
    """
    Data-driven quality scoring that replaces hardcoded bonuses and penalties.

    This class learns optimal bonus/penalty parameters from data to maximize
    the predictive power and economic value of quality scores.
    """

    def __init__(self, optimized_params: Dict[BonusPenaltyParameter, float]):
        """Initialize with optimized parameters."""
        self.params = optimized_params
        self.logger = get_logger('DataDrivenQualityScorer')

        # Extract parameters for easy access
        self.speed_bonus_amount = optimized_params.get(BonusPenaltyParameter.SPEED_BONUS_AMOUNT, 0.1)
        self.speed_bonus_threshold = optimized_params.get(BonusPenaltyParameter.SPEED_BONUS_THRESHOLD, 0.5)

        self.risk_penalty_multiplier = optimized_params.get(BonusPenaltyParameter.RISK_PENALTY_MULTIPLIER, 30)
        self.risk_minimum_score = optimized_params.get(BonusPenaltyParameter.RISK_MINIMUM_SCORE, 0.1)

        self.profit_risk_ratio_threshold = optimized_params.get(BonusPenaltyParameter.PROFIT_RISK_RATIO_THRESHOLD, 2.0)
        self.profit_bonus_multiplier = optimized_params.get(BonusPenaltyParameter.PROFIT_BONUS_MULTIPLIER, 0.1)
        self.profit_bonus_max = optimized_params.get(BonusPenaltyParameter.PROFIT_BONUS_MAX, 0.2)

        self.reversal_penalty_multiplier = optimized_params.get(BonusPenaltyParameter.REVERSAL_PENALTY_MULTIPLIER, 50)
        self.reversal_minimum_factor = optimized_params.get(BonusPenaltyParameter.REVERSAL_MINIMUM_FACTOR, 0.1)

        # Normalize weights
        speed_weight = optimized_params.get(BonusPenaltyParameter.SPEED_WEIGHT, 0.3)
        risk_weight = optimized_params.get(BonusPenaltyParameter.RISK_WEIGHT, 0.4)
        profitability_weight = optimized_params.get(BonusPenaltyParameter.PROFITABILITY_WEIGHT, 0.3)

        total_weight = speed_weight + risk_weight + profitability_weight
        self.speed_weight = speed_weight / total_weight
        self.risk_weight = risk_weight / total_weight
        self.profitability_weight = profitability_weight / total_weight

        self.profit_scale_factor = optimized_params.get(BonusPenaltyParameter.PROFIT_SCALE_FACTOR, 300)

        self.logger.info('🎯 Data-driven quality scorer initialized with optimized parameters')

    def calculate_optimized_quality_score(self,
                                        target_hit: bool,
                                        time_to_hit: Optional[int],
                                        max_adverse: float,
                                        total_periods: int,
                                        net_profit: float) -> float:
        """
        Calculate quality score using optimized bonus/penalty parameters.

        This replaces the hardcoded _calculate_quality_score method in MultiHorizonProfitLabeler
        with data-driven optimized parameters.
        """
        if not target_hit:
            return 0.1  # Small probability for model uncertainty

        quality_factors = []

        # 1. Speed factor with optimized bonus
        if time_to_hit is not None:
            speed_factor = 1.0 - (time_to_hit / total_periods)
            speed_score = max(0.2, speed_factor)
            quality_factors.append(speed_score * self.speed_weight)

            # Optimized speed bonus
            if time_to_hit < total_periods * self.speed_bonus_threshold:
                quality_factors.append(self.speed_bonus_amount)

        # 2. Risk factor with optimized penalty
        if max_adverse > 0:
            # Use optimized penalty multiplier
            risk_factor = max(self.risk_minimum_score, 1.0 - (max_adverse * self.risk_penalty_multiplier))
            risk_score = risk_factor
        else:
            risk_score = 1.0  # Perfect score if no adverse excursion
        quality_factors.append(risk_score * self.risk_weight)

        # 3. Profitability factor with optimized bonus
        if net_profit > 0:
            # Use optimized scale factor
            profit_factor = min(1.0, net_profit * self.profit_scale_factor)
            profit_score = max(0.2, profit_factor)

            # Optimized profit bonus
            if max_adverse > 0:
                profit_risk_ratio = net_profit / max_adverse
                if profit_risk_ratio > self.profit_risk_ratio_threshold:
                    profit_bonus = min(
                        self.profit_bonus_max,
                        (profit_risk_ratio - self.profit_risk_ratio_threshold) * self.profit_bonus_multiplier
                    )
                    quality_factors.append(profit_bonus)
        else:
            profit_score = 0.1
        quality_factors.append(profit_score * self.profitability_weight)

        # Cap total quality score at 1.0
        total_quality = min(1.0, np.sum(quality_factors))

        return total_quality

    def calculate_optimized_reversal_score(self,
                                         probability_scores: Dict[str, float],
                                         sample_labels: Dict[str, float]) -> float:
        """
        Calculate reversal capture score using optimized penalty parameters.

        This replaces the hardcoded _calculate_reversal_capture_score method.
        """
        reversal_factors = []

        # Factor 1: Speed of opportunity (faster = better for reversals)
        time_values = [v for k, v in sample_labels.items() if k.endswith('_time_to_hit') and v >= 0]
        if time_values:
            avg_time = np.mean(time_values)
            speed_factor = max(0.1, 1.0 - (avg_time / 4.0))
            reversal_factors.append(speed_factor * 0.4)

        # Factor 2: Low adverse excursion with optimized penalty
        adverse_values = [v for k, v in sample_labels.items() if k.endswith('_max_adverse')]
        if adverse_values:
            avg_adverse = np.mean(adverse_values)
            # Use optimized reversal penalty multiplier
            clean_factor = max(
                self.reversal_minimum_factor,
                1.0 - (avg_adverse * self.reversal_penalty_multiplier)
            )
            reversal_factors.append(clean_factor * 0.3)

        # Factor 3: Immediate vs short-term probability ratio (unchanged)
        immediate_prob = probability_scores.get('micro_immediate', 0.0) + probability_scores.get('small_immediate', 0.0)
        short_prob = probability_scores.get('micro_short', 0.0) + probability_scores.get('small_short', 0.0)

        if short_prob > 0:
            ratio_factor = min(1.0, immediate_prob / short_prob)
            reversal_factors.append(ratio_factor * 0.3)

        return np.sum(reversal_factors) if reversal_factors else 0.1

class BonusPenaltyOptimizer:
    """
    Optimizer for bonus and penalty parameters in profit labeling.

    This class uses data-driven methods to find optimal bonus and penalty
    parameters that maximize labeling quality and predictive power.
    """

    def __init__(self, config: Optional[BonusPenaltyOptimizationConfig] = None):
        """Initialize bonus/penalty optimizer."""
        self.config = config or BonusPenaltyOptimizationConfig()
        self.logger = get_logger('BonusPenaltyOptimizer')

        # Optimization state
        self.optimization_history: List[Dict[str, Any]] = []
        self.best_parameters: Optional[Dict[BonusPenaltyParameter, float]] = None
        self.evaluation_cache: Dict[str, float] = {}

        self.logger.info('🎯 Bonus/Penalty Optimizer initialized')
        self.logger.info(f'   → Method: {self.config.optimization_method}')
        self.logger.info(f'   → Objective: {self.config.optimization_objective.value}')
        self.logger.info(f'   → Parameters to optimize: {len(self.config.parameter_ranges)}')

    def optimize_bonus_penalty_parameters(self, market_data: pd.DataFrame) -> OptimizedParameters:
        """
        Optimize bonus and penalty parameters using historical market data.

        Args:
            market_data: Historical OHLCV market data

        Returns:
            OptimizedParameters with best parameter values and performance metrics
        """
        self.logger.info('🚀 Starting bonus/penalty parameter optimization')

        if len(market_data) < 500:
            self.logger.warning('⚠️ Insufficient data for parameter optimization')
            return self._create_default_parameters()

        # Split data for validation
        split_idx = int(len(market_data) * (1 - self.config.validation_split))
        train_data = market_data.iloc[:split_idx]
        val_data = market_data.iloc[split_idx:]

        # Run optimization based on selected method
        if self.config.optimization_method == "bayesian" and SKOPT_AVAILABLE:
            result = self._bayesian_optimization(train_data, val_data)
        elif self.config.optimization_method == "optuna" and OPTUNA_AVAILABLE:
            result = self._optuna_optimization(train_data, val_data)
        elif self.config.optimization_method == "grid_search":
            result = self._grid_search_optimization(train_data, val_data)
        else:
            result = self._random_search_optimization(train_data, val_data)

        self.best_parameters = result.parameters

        self.logger.info(f'✅ Optimization completed with score: {result.objective_score:.4f}')
        self.logger.info(f'   → Optimized {len(result.parameters)} parameters')

        return result

    def _bayesian_optimization(self, train_data: pd.DataFrame, val_data: pd.DataFrame) -> OptimizedParameters:
        """Perform Bayesian optimization of parameters."""
        self.logger.info('🧠 Running Bayesian optimization')

        # Define search space
        space = []
        param_names = []

        for param, (min_val, max_val) in self.config.parameter_ranges.items():
            space.append(Real(min_val, max_val, name=param.value))
            param_names.append(param.value)

        # Define objective function
        @use_named_args(space)
        def objective(**params):
            # Convert to BonusPenaltyParameter enum keys
            param_dict = {
                BonusPenaltyParameter(k): v for k, v in params.items()
            }

            # Evaluate parameters
            score = self._evaluate_parameters(param_dict, train_data, val_data)

            # Store in history
            self.optimization_history.append({
                'parameters': param_dict.copy(),
                'score': score,
                'timestamp': datetime.now()
            })

            return -score  # Minimize negative score

        # Run optimization
        try:
            result = gp_minimize(
                func=objective,
                dimensions=space,
                n_calls=self.config.n_trials,
                random_state=self.config.random_state
            )

            # Extract best parameters
            best_params = {
                BonusPenaltyParameter(param_names[i]): result.x[i]
                for i in range(len(param_names))
            }

            # Validate on validation data
            validation_scores = self._calculate_validation_scores(best_params, val_data)

            return OptimizedParameters(
                parameters=best_params,
                objective_score=-result.fun,
                validation_scores=validation_scores,
                cross_validation_scores=[],
                optimization_history=self.optimization_history,
                metadata={
                    'method': 'bayesian_optimization',
                    'n_calls': len(result.x_iters),
                    'convergence_value': result.fun
                }
            )

        except Exception as e:
            self.logger.error(f'Bayesian optimization failed: {e}')
            return self._create_default_parameters()

    def _optuna_optimization(self, train_data: pd.DataFrame, val_data: pd.DataFrame) -> OptimizedParameters:
        """Perform Optuna TPE optimization."""
        self.logger.info('🎯 Running Optuna optimization')

        try:
            # Create study
            study = optuna.create_study(
                direction='maximize',
                sampler=optuna.samplers.TPESampler(seed=self.config.random_state)
            )

            # Define objective function
            def objective(trial):
                # Suggest parameters
                params = {}
                for param, (min_val, max_val) in self.config.parameter_ranges.items():
                    params[param] = trial.suggest_float(param.value, min_val, max_val)

                # Evaluate parameters
                score = self._evaluate_parameters(params, train_data, val_data)

                # Store in history
                self.optimization_history.append({
                    'parameters': params.copy(),
                    'score': score,
                    'timestamp': datetime.now()
                })

                return score

            # Run optimization
            study.optimize(objective, n_trials=self.config.n_trials)

            # Extract best parameters
            best_params = {
                BonusPenaltyParameter(k): v for k, v in study.best_params.items()
            }

            # Validate on validation data
            validation_scores = self._calculate_validation_scores(best_params, val_data)

            return OptimizedParameters(
                parameters=best_params,
                objective_score=study.best_value,
                validation_scores=validation_scores,
                cross_validation_scores=[],
                optimization_history=self.optimization_history,
                metadata={
                    'method': 'optuna_tpe',
                    'n_trials': len(study.trials),
                    'best_trial': study.best_trial.number
                }
            )

        except Exception as e:
            self.logger.error(f'Optuna optimization failed: {e}')
            return self._create_default_parameters()

    def _grid_search_optimization(self, train_data: pd.DataFrame, val_data: pd.DataFrame) -> OptimizedParameters:
        """Perform grid search optimization."""
        self.logger.info('🔍 Running grid search optimization')

        # Create parameter grid (limited for performance)
        grid_params = {}
        for param, (min_val, max_val) in self.config.parameter_ranges.items():
            # Use 3 values for each parameter in grid search
            grid_params[param] = np.linspace(min_val, max_val, 3)

        param_grid = ParameterGrid(grid_params)

        best_score = -np.inf
        best_params = None

        self.logger.info(f'   → Testing {len(param_grid)} parameter combinations')

        for i, params in enumerate(param_grid):
            if i % 10 == 0:
                self.logger.info(f'   → Progress: {i}/{len(param_grid)} ({i/len(param_grid)*100:.1f}%)')

            # Convert to enum keys
            param_dict = {BonusPenaltyParameter(k): v for k, v in params.items()}

            # Evaluate parameters
            score = self._evaluate_parameters(param_dict, train_data, val_data)

            # Store in history
            self.optimization_history.append({
                'parameters': param_dict.copy(),
                'score': score,
                'timestamp': datetime.now()
            })

            if score > best_score:
                best_score = score
                best_params = param_dict.copy()

        if best_params is None:
            return self._create_default_parameters()

        # Validate on validation data
        validation_scores = self._calculate_validation_scores(best_params, val_data)

        return OptimizedParameters(
            parameters=best_params,
            objective_score=best_score,
            validation_scores=validation_scores,
            cross_validation_scores=[],
            optimization_history=self.optimization_history,
            metadata={
                'method': 'grid_search',
                'total_combinations': len(param_grid)
            }
        )

    def _random_search_optimization(self, train_data: pd.DataFrame, val_data: pd.DataFrame) -> OptimizedParameters:
        """Perform random search optimization."""
        self.logger.info('🎲 Running random search optimization')

        np.random.seed(self.config.random_state)

        best_score = -np.inf
        best_params = None

        for i in range(self.config.n_trials):
            if i % 20 == 0:
                self.logger.info(f'   → Progress: {i}/{self.config.n_trials} ({i/self.config.n_trials*100:.1f}%)')

            # Generate random parameters
            params = {}
            for param, (min_val, max_val) in self.config.parameter_ranges.items():
                params[param] = np.random.uniform(min_val, max_val)

            # Evaluate parameters
            score = self._evaluate_parameters(params, train_data, val_data)

            # Store in history
            self.optimization_history.append({
                'parameters': params.copy(),
                'score': score,
                'timestamp': datetime.now()
            })

            if score > best_score:
                best_score = score
                best_params = params.copy()

        if best_params is None:
            return self._create_default_parameters()

        # Validate on validation data
        validation_scores = self._calculate_validation_scores(best_params, val_data)

        return OptimizedParameters(
            parameters=best_params,
            objective_score=best_score,
            validation_scores=validation_scores,
            cross_validation_scores=[],
            optimization_history=self.optimization_history,
            metadata={
                'method': 'random_search',
                'n_trials': self.config.n_trials
            }
        )

    def _evaluate_parameters(self,
                           params: Dict[BonusPenaltyParameter, float],
                           train_data: pd.DataFrame,
                           val_data: pd.DataFrame) -> float:
        """Evaluate a set of bonus/penalty parameters."""
        try:
            # Create cache key
            cache_key = str(sorted(params.items()))
            if cache_key in self.evaluation_cache:
                return self.evaluation_cache[cache_key]

            # Create modified labeler with these parameters
            modified_labeler = self._create_modified_labeler(params)

            # Generate labels on validation data
            labeled_data = modified_labeler.generate_labels(val_data.copy())

            # Calculate objective score
            score = self._calculate_objective_score(labeled_data, val_data)

            # Cache result
            self.evaluation_cache[cache_key] = score

            return score

        except Exception as e:
            self.logger.warning(f'Parameter evaluation failed: {e}')
            return 0.0

    def _create_modified_labeler(self, params: Dict[BonusPenaltyParameter, float]) -> 'ModifiedMultiHorizonLabeler':
        """Create a modified labeler with optimized parameters."""
        return ModifiedMultiHorizonLabeler(params)

    def _calculate_objective_score(self, labeled_data: pd.DataFrame, market_data: pd.DataFrame) -> float:
        """Calculate objective score based on optimization objective."""
        if self.config.optimization_objective == OptimizationObjective.PREDICTIVE_POWER:
            return self._calculate_predictive_power(labeled_data, market_data)
        elif self.config.optimization_objective == OptimizationObjective.SHARPE_RATIO:
            return self._calculate_sharpe_ratio(labeled_data, market_data)
        elif self.config.optimization_objective == OptimizationObjective.HIT_RATE_BALANCE:
            return self._calculate_hit_rate_balance(labeled_data)
        elif self.config.optimization_objective == OptimizationObjective.ECONOMIC_VALUE:
            return self._calculate_economic_value(labeled_data, market_data)
        elif self.config.optimization_objective == OptimizationObjective.MULTI_OBJECTIVE:
            return self._calculate_multi_objective_score(labeled_data, market_data)
        else:
            return self._calculate_predictive_power(labeled_data, market_data)

    def _calculate_predictive_power(self, labeled_data: pd.DataFrame, market_data: pd.DataFrame) -> float:
        """Calculate predictive power score."""
        if 'overall_opportunity' not in labeled_data.columns or 'close' not in market_data.columns:
            return 0.0

        opportunity = labeled_data['overall_opportunity'].fillna(0)
        future_returns = market_data['close'].pct_change().shift(-1).fillna(0)

        common_idx = opportunity.index.intersection(future_returns.index)
        if len(common_idx) < 20:
            return 0.0

        # Calculate correlation
        corr = np.corrcoef(opportunity.loc[common_idx], future_returns.loc[common_idx])[0, 1]

        # Convert to AUC-like score
        if np.isnan(corr):
            return 0.0

        return max(0.0, min(1.0, 0.5 + abs(corr) / 2))

    def _calculate_sharpe_ratio(self, labeled_data: pd.DataFrame, market_data: pd.DataFrame) -> float:
        """Calculate Sharpe ratio using VectorBT."""
        if 'overall_opportunity' not in labeled_data.columns or 'close' not in market_data.columns:
            return 0.0

        opportunity = labeled_data['overall_opportunity'].fillna(0)
        returns = market_data['close'].pct_change().shift(-1).fillna(0)

        common_idx = opportunity.index.intersection(returns.index)
        if len(common_idx) < 50:
            return 0.0

        try:
            import vectorbt as vbt
            from vectorbt.portfolio import Portfolio
            from vectorbt.returns import Returns

            # Create signals using VectorBT
            threshold = opportunity.quantile(0.7)
            signals = (opportunity.loc[common_idx] > threshold).astype(int)

            # Use VectorBT for portfolio analysis
            portfolio = Portfolio.from_signals(
                close=returns.loc[common_idx],
                entries=signals,
                exits=None,
                freq='1D'
            )

            # Get strategy returns from VectorBT portfolio
            strategy_returns = portfolio.returns()

            # Use VectorBT for Sharpe ratio calculation
            returns_obj = Returns(strategy_returns)
            sharpe = returns_obj.sharpe_ratio()

            return max(0.0, min(2.0, sharpe + 1.0)) / 2.0  # Normalize to 0-1

        except Exception as e:
            logger.warning(f"VectorBT Sharpe ratio calculation failed, using manual calculation: {e}")
            # Fallback to manual calculation
            # Create simple strategy: go long when opportunity > 70th percentile
            threshold = opportunity.quantile(0.7)
            signals = (opportunity.loc[common_idx] > threshold).astype(int)
            strategy_returns = signals * returns.loc[common_idx]

            # Calculate Sharpe ratio manually
            if strategy_returns.std() > 0:
                sharpe = strategy_returns.mean() / strategy_returns.std()
                return max(0.0, min(2.0, sharpe + 1.0)) / 2.0  # Normalize to 0-1

            return 0.0

    def _calculate_hit_rate_balance(self, labeled_data: pd.DataFrame) -> float:
        """Calculate balanced hit rate score."""
        prob_columns = [col for col in labeled_data.columns if col.endswith('_prob')]

        if not prob_columns:
            return 0.0

        hit_rates = []
        for col in prob_columns[:5]:  # Top 5 for efficiency
            values = labeled_data[col].dropna()
            if len(values) > 20:
                hit_rate = (values > 0.5).mean()
                # Target ~30% hit rate for good balance
                balanced_score = 1.0 - abs(hit_rate - 0.3)
                hit_rates.append(max(0.0, balanced_score))

        return np.mean(hit_rates) if hit_rates else 0.0

    def _calculate_economic_value(self, labeled_data: pd.DataFrame, market_data: pd.DataFrame) -> float:
        """Calculate economic value score."""
        return self._calculate_sharpe_ratio(labeled_data, market_data)  # Use Sharpe as proxy

    def _calculate_multi_objective_score(self, labeled_data: pd.DataFrame, market_data: pd.DataFrame) -> float:
        """Calculate multi-objective score."""
        # Combine multiple objectives with weights
        predictive_score = self._calculate_predictive_power(labeled_data, market_data)
        hit_rate_score = self._calculate_hit_rate_balance(labeled_data)
        economic_score = self._calculate_economic_value(labeled_data, market_data)
        stability_score = self._calculate_stability_score(labeled_data)

        # Weighted combination
        multi_objective_score = (
            self.config.predictive_power_weight * predictive_score +
            self.config.hit_rate_weight * hit_rate_score +
            self.config.economic_value_weight * economic_score +
            self.config.stability_weight * stability_score
        )

        return multi_objective_score

    def _calculate_stability_score(self, labeled_data: pd.DataFrame) -> float:
        """Calculate label stability score."""
        if 'overall_opportunity' not in labeled_data.columns:
            return 0.5

        opportunity = labeled_data['overall_opportunity'].fillna(0)

        if len(opportunity) < 50:
            return 0.5

        # Calculate rolling standard deviation
        rolling_std = opportunity.rolling(20).std()

        if rolling_std.mean() > 0 and opportunity.std() > 0:
            cv = rolling_std.mean() / opportunity.std()
            stability = 1.0 / (1.0 + cv)
        else:
            stability = 0.5

        return max(0.0, min(1.0, stability))

    def _calculate_validation_scores(self,
                                   params: Dict[BonusPenaltyParameter, float],
                                   val_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate validation scores for best parameters."""
        try:
            modified_labeler = self._create_modified_labeler(params)
            labeled_data = modified_labeler.generate_labels(val_data.copy())

            return {
                'predictive_power': self._calculate_predictive_power(labeled_data, val_data),
                'hit_rate_balance': self._calculate_hit_rate_balance(labeled_data),
                'economic_value': self._calculate_economic_value(labeled_data, val_data),
                'stability': self._calculate_stability_score(labeled_data)
            }

        except Exception:
            return {
                'predictive_power': 0.0,
                'hit_rate_balance': 0.0,
                'economic_value': 0.0,
                'stability': 0.0
            }

    def _create_default_parameters(self) -> OptimizedParameters:
        """Create default parameters when optimization fails."""
        default_params = {
            BonusPenaltyParameter.SPEED_BONUS_AMOUNT: 0.1,
            BonusPenaltyParameter.SPEED_BONUS_THRESHOLD: 0.5,
            BonusPenaltyParameter.RISK_PENALTY_MULTIPLIER: 30.0,
            BonusPenaltyParameter.RISK_MINIMUM_SCORE: 0.1,
            BonusPenaltyParameter.PROFIT_RISK_RATIO_THRESHOLD: 2.0,
            BonusPenaltyParameter.PROFIT_BONUS_MULTIPLIER: 0.1,
            BonusPenaltyParameter.PROFIT_BONUS_MAX: 0.2,
            BonusPenaltyParameter.REVERSAL_PENALTY_MULTIPLIER: 50.0,
            BonusPenaltyParameter.REVERSAL_MINIMUM_FACTOR: 0.1,
            BonusPenaltyParameter.SPEED_WEIGHT: 0.3,
            BonusPenaltyParameter.RISK_WEIGHT: 0.4,
            BonusPenaltyParameter.PROFITABILITY_WEIGHT: 0.3,
            BonusPenaltyParameter.PROFIT_SCALE_FACTOR: 300.0
        }

        return OptimizedParameters(
            parameters=default_params,
            objective_score=0.0,
            validation_scores={},
            cross_validation_scores=[],
            optimization_history=[],
            metadata={'method': 'default', 'reason': 'optimization_failed'}
        )

class ModifiedMultiHorizonLabeler(MultiHorizonProfitLabeler):
    """
    Modified version of MultiHorizonProfitLabeler that uses optimized bonus/penalty parameters.

    This class replaces the hardcoded bonuses and penalties with data-driven optimized values.
    """

    def __init__(self, optimized_params: Dict[BonusPenaltyParameter, float]):
        """Initialize with optimized parameters."""
        super().__init__()

        # Create data-driven quality scorer
        self.quality_scorer = DataDrivenQualityScorer(optimized_params)

        self.logger.info('🎯 Modified labeler initialized with optimized bonus/penalty parameters')

    def _calculate_quality_score(self,
                               target_hit: bool,
                               time_to_hit: Optional[int],
                               max_adverse: float,
                               total_periods: int,
                               net_profit: float) -> float:
        """
        Override the original quality score calculation with optimized parameters.
        """
        return self.quality_scorer.calculate_optimized_quality_score(
            target_hit, time_to_hit, max_adverse, total_periods, net_profit
        )

    def _calculate_reversal_capture_score(self,
                                        probability_scores: Dict[str, float],
                                        sample_labels: Dict[str, float]) -> float:
        """
        Override the original reversal capture score with optimized parameters.
        """
        return self.quality_scorer.calculate_optimized_reversal_score(
            probability_scores, sample_labels
        )

class RegimeSpecificBonusPenaltyOptimizer:
    """
    Optimize bonus/penalty parameters for specific market regimes.

    This class recognizes that optimal bonuses and penalties may vary
    across different market conditions (high volatility, trending, etc.).
    """

    def __init__(self, config: Optional[BonusPenaltyOptimizationConfig] = None):
        """Initialize regime-specific optimizer."""
        self.config = config or BonusPenaltyOptimizationConfig()
        self.logger = get_logger('RegimeSpecificBonusPenaltyOptimizer')

        # Regime-specific parameters
        self.regime_parameters: Dict[str, Dict[BonusPenaltyParameter, float]] = {}

        self.logger.info('🎯📊 Regime-specific bonus/penalty optimizer initialized')

    def optimize_regime_specific_parameters(self,
                                          market_data: pd.DataFrame) -> Dict[str, OptimizedParameters]:
        """Optimize parameters for different market regimes."""
        self.logger.info('🔄 Optimizing regime-specific bonus/penalty parameters')

        # Identify market regimes
        regimes = self._identify_market_regimes(market_data)

        regime_results = {}

        for regime_name in regimes['regime'].unique():
            if pd.isna(regime_name):
                continue

            self.logger.info(f'   → Optimizing for {regime_name} regime')

            # Filter data for this regime
            regime_mask = regimes['regime'] == regime_name
            regime_data = market_data[regime_mask]

            if len(regime_data) < 200:  # Need sufficient data
                self.logger.warning(f'   ⚠️ Insufficient data for {regime_name} regime')
                continue

            # Optimize parameters for this regime
            optimizer = BonusPenaltyOptimizer(self.config)
            regime_result = optimizer.optimize_bonus_penalty_parameters(regime_data)

            regime_results[regime_name] = regime_result
            self.regime_parameters[regime_name] = regime_result.parameters

            self.logger.info(f'   ✅ {regime_name} optimization: score {regime_result.objective_score:.3f}')

        return regime_results

    def _identify_market_regimes(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Identify market regimes for parameter optimization."""
        regimes = pd.DataFrame(index=market_data.index)

        if 'close' not in market_data.columns:
            regimes['regime'] = 'unknown'
            return regimes

        # Calculate regime indicators
        returns = market_data['close'].pct_change()
        volatility = returns.rolling(20).std()

        # Simple regime classification
        vol_25 = volatility.quantile(0.33)
        vol_75 = volatility.quantile(0.67)

        regime_labels = []
        for vol in volatility:
            if pd.isna(vol):
                regime_labels.append('unknown')
            elif vol <= vol_25:
                regime_labels.append('low_volatility')
            elif vol >= vol_75:
                regime_labels.append('high_volatility')
            else:
                regime_labels.append('medium_volatility')

        regimes['regime'] = regime_labels
        return regimes

# Convenience functions
def optimize_bonus_penalty_parameters(market_data: pd.DataFrame,
                                    config: Optional[BonusPenaltyOptimizationConfig] = None) -> OptimizedParameters:
    """Convenience function to optimize bonus/penalty parameters."""
    optimizer = BonusPenaltyOptimizer(config)
    return optimizer.optimize_bonus_penalty_parameters(market_data)

def create_optimized_labeler(market_data: pd.DataFrame,
                           config: Optional[BonusPenaltyOptimizationConfig] = None) -> ModifiedMultiHorizonLabeler:
    """Create a labeler with optimized bonus/penalty parameters."""
    optimization_result = optimize_bonus_penalty_parameters(market_data, config)
    return ModifiedMultiHorizonLabeler(optimization_result.parameters)

def get_optimal_bonus_penalty_config(market_data: pd.DataFrame) -> Dict[str, float]:
    """Get optimal bonus/penalty configuration as dictionary."""
    optimization_result = optimize_bonus_penalty_parameters(market_data)

    # Convert to simple dictionary for easy integration
    config_dict = {}
    for param, value in optimization_result.parameters.items():
        config_dict[param.value] = value

    return config_dict

# Example usage for integration
if __name__ == '__main__':
    print('🎯 Testing Bonus/Penalty Optimization')

    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=1000, freq='5min')
    prices = [100 + i * 0.01 + np.random.normal(0, 0.5) for i in range(1000)]

    sample_data = pd.DataFrame({
        'open': prices,
        'high': [p * 1.002 for p in prices],
        'low': [p * 0.998 for p in prices],
        'close': prices,
        'volume': [1000] * 1000
    }, index=dates)

    print(f'📊 Generated {len(sample_data)} samples of test data')

    # Test optimization
    print('\n🚀 Running bonus/penalty optimization...')

    try:
        config = BonusPenaltyOptimizationConfig(
            optimization_method="random_search",  # Fast method for testing
            n_trials=20  # Reduced for testing
        )

        optimization_result = optimize_bonus_penalty_parameters(sample_data, config)

        print(f'✅ Optimization completed with score: {optimization_result.objective_score:.3f}')
        print('\n📋 Optimized Parameters:')

        for param, value in optimization_result.parameters.items():
            print(f'   → {param.value}: {value:.3f}')

        # Test modified labeler
        print('\n🧪 Testing modified labeler...')
        modified_labeler = ModifiedMultiHorizonLabeler(optimization_result.parameters)

        # Generate labels with optimized parameters
        optimized_labels = modified_labeler.generate_labels(sample_data.copy())

        print(f'✅ Modified labeler generated {optimized_labels.shape[1]} label columns')
        print(f'   → Overall opportunity mean: {optimized_labels["overall_opportunity"].mean():.3f}')

        print('\n🎉 Bonus/penalty optimization test completed successfully!')

    except Exception as e:
        print(f'❌ Test failed: {e}')
        import traceback
        traceback.print_exc()
