# TPSL OPTIMIZATION - TEMPORARILY DISABLED
# This file contains TPSL optimization functionality that is temporarily disabled
# as TPSL parameters are commented out in config.yaml
# Uncomment TPSL sections when TPSL optimization is re-enabled

"""
Per-HMM Regime Triple Barrier Thresholds and TPSL Parameters Optimization

This module implements regime-specific optimization of triple barrier thresholds
and Take Profit/Stop Loss (TPSL) parameters for each HMM regime using Optuna.
It extends the existing optimization framework to provide regime-aware parameter tuning.

Key Features:
- Regime-specific triple barrier thresholds optimization
- Per-regime TPSL parameter optimization (TEMPORARILY DISABLED)
- Multi-objective optimization with regime-specific objectives
- Cross-validation with regime-aware splits
- Statistical significance testing per regime
- Comprehensive reporting and visualization
"""
import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings
import numpy as np
import pandas as pd
import optuna
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler
from optuna.visualization import plot_optimization_history, plot_param_importances
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy import stats
from src.utils.logger import setup_logging
from pathlib import Path
import copy
from src.config.config_optuna import SROptimizationParameters, HyperparameterOptimizationConfig, get_parameter_search_space
from src.training.steps.step06_labeling_components.optimized_triple_barrier_labeling import OptimizedTripleBarrierLabeling
setup_logging()
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

@dataclass
class RegimeTripleBarrierParams:
    """Triple barrier parameters for a specific regime."""
    profit_take_multiplier: float = 0.02
    stop_loss_multiplier: float = 0.01
    time_barrier_minutes: int = 30
    max_lookahead: int = 100
    regime_volatility_multiplier: float = 1.0
    regime_trend_multiplier: float = 1.0
    regime_volume_multiplier: float = 1.0
    tp_multiplier_range: Tuple[float, float] = (1.5, 4.0)
    sl_multiplier_range: Tuple[float, float] = (0.8, 2.0)
    position_size_range: Tuple[float, float] = (0.05, 0.25)
    min_tp_multiplier: float = 1.2
    max_tp_multiplier: float = 5.0
    min_sl_multiplier: float = 0.5
    max_sl_multiplier: float = 3.0

@dataclass
class RegimeOptimizationResult:
    """Result of regime-specific optimization."""
    regime_name: str
    regime_id: int
    triple_barrier_params: RegimeTripleBarrierParams
    tpsl_params: Dict[str, float]
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_return: float
    calmar_ratio: float
    sortino_ratio: float
    regime_accuracy: float
    regime_precision: float
    regime_recall: float
    regime_f1: float
    optimization_score: float
    n_trials: int
    optimization_time: float
    study_name: str
    best_trial_number: int
    p_value: float
    confidence_interval: Tuple[float, float]

@dataclass
class RegimeSpecificOptimizationConfig:
    """Configuration for regime-specific optimization."""
    enable_regime_optimization: bool = True
    multi_objective: bool = True
    n_trials_per_regime: int = 100
    timeout_minutes_per_regime: int = 60
    cv_folds: int = 5
    objectives: List[str] = field(default_factory=lambda: ['sharpe_ratio', 'win_rate', 'profit_factor', 'regime_accuracy'])
    objective_weights: Dict[str, float] = field(default_factory=lambda: {'sharpe_ratio': 0.3, 'win_rate': 0.25, 'profit_factor': 0.25, 'regime_accuracy': 0.2})
    regime_constraints: Dict[str, Dict[str, List[float]]] = field(default_factory=lambda: {'BULL_TREND': {'tp_multiplier_range': [2.5, 5.0], 'sl_multiplier_range': [1.2, 2.5], 'position_size_range': [0.1, 0.25]}, 'BEAR_TREND': {'tp_multiplier_range': [2.0, 4.5], 'sl_multiplier_range': [1.0, 2.2], 'position_size_range': [0.08, 0.2]}, 'SIDEWAYS_RANGE': {'tp_multiplier_range': [1.5, 3.0], 'sl_multiplier_range': [0.8, 1.8], 'position_size_range': [0.06, 0.15]}, 'HIGH_IMPACT_CANDLE': {'tp_multiplier_range': [1.8, 3.5], 'sl_multiplier_range': [0.9, 2.0], 'position_size_range': [0.05, 0.12]}, 'SR_ZONE_ACTION': {'tp_multiplier_range': [2.0, 4.0], 'sl_multiplier_range': [1.0, 2.2], 'position_size_range': [0.08, 0.18]}})
    early_stopping_patience: int = 20
    early_stopping_delta: float = 0.001
    enable_pruning: bool = True
    pruning_method: str = 'hyperband'
    enable_statistical_testing: bool = True
    confidence_level: float = 0.95
    min_sample_size: int = 50

class RegimeSpecificTripleBarrierOptimizer:
    """
    Optimizer for regime-specific triple barrier thresholds and TPSL parameters.
    
    This optimizer extends the existing Optuna framework to provide regime-aware
    parameter optimization, ensuring that each HMM regime has optimal triple barrier
    thresholds and TPSL parameters for maximum performance.
    """

    def __init__(self, config: Dict[str, Any], storage_url: str='sqlite:///regime_triple_barrier_optuna_studies.db', study_name_prefix: str='regime_triple_barrier_optimization') -> None:
        """
        Initialize the regime-specific triple barrier optimizer.
        
        Args:
            config: Configuration dictionary
            storage_url: Database URL for study persistence
            study_name_prefix: Prefix for study names
        """
        self.config = config
        self.storage_url = storage_url
        self.study_name_prefix = study_name_prefix
        self.logger = logging.getLogger(__name__)
        self.optimization_config = RegimeSpecificOptimizationConfig()
        self._load_optimization_config()
        self.regime_results: Dict[str, RegimeOptimizationResult] = {}
        self.global_results: Dict[str, Any] = {}
        self.studies: Dict[str, optuna.Study] = {}

    def _load_optimization_config(self) -> None:
        """Load optimization configuration from config."""
        opt_config = self.config.get('regime_specific_optimization', {})
        for key, value in opt_config.items():
            if hasattr(self.optimization_config, key):
                setattr(self.optimization_config, key, value)
        regime_constraints = opt_config.get('regime_constraints')
        if regime_constraints:
            self.optimization_config.regime_constraints.update(regime_constraints)

    async def initialize(self) -> bool:
        """Initialize the optimizer components."""
        try:
            self.logger.info('🚀 Initializing Regime-Specific Triple Barrier Optimizer...')
            if not self._validate_configuration():
                self.logger.error('❌ Configuration validation failed')
                return False
            await self._initialize_storage()
            self.logger.info('✅ Regime-Specific Triple Barrier Optimizer initialized successfully')
            return True
        except Exception as e:
            self.logger.exception(f'❌ Error initializing optimizer: {e}')
            return False

    def _validate_configuration(self) -> bool:
        """Validate the optimization configuration."""
        try:
            if not self.optimization_config.objectives:
                self.logger.error('❌ No objectives specified')
                return False
            weight_sum = sum(self.optimization_config.objective_weights.values())
            if abs(weight_sum - 1.0) > 0.01:
                self.logger.error(f'❌ Objective weights must sum to 1.0, got {weight_sum}')
                return False
            if not self.optimization_config.regime_constraints:
                self.logger.error('❌ No regime constraints specified')
                return False
            return True
        except Exception as e:
            self.logger.error(f'❌ Configuration validation error: {e}')
            return False

    async def _initialize_storage(self) -> None:
        """Initialize Optuna storage."""
        try:
            study = optuna.create_study(study_name='test_study', storage=self.storage_url, load_if_exists=True)
            self.logger.info(f'✅ Storage initialized: {self.storage_url}')
        except Exception as e:
            self.logger.warning(f'⚠️ Storage initialization failed: {e}')
            self.logger.info('📝 Using in-memory storage')
            self.storage_url = None

    def _get_regime_names(self, data: pd.DataFrame) -> List[str]:
        """Extract regime names from data using shared regime accessor."""
        try:
            from src.utils.regime_data_access import get_regime_column
            regime_column = get_regime_column(data)
        except Exception:
            regime_column = None
        if regime_column is None:
            self.logger.warning('⚠️ No regime column found, using default regimes')
            return list(self.optimization_config.regime_constraints.keys())
        unique_regimes = data[regime_column].unique()
        regime_names = []
        for regime in unique_regimes:
            if isinstance(regime, (int, np.integer)):
                regime_name = f'REGIME_{regime}'
            else:
                regime_name = str(regime)
            regime_names.append(regime_name)
        return regime_names

    def _create_regime_objective_function(self, regime_name: str, regime_data: pd.DataFrame, regime_constraints: Dict[str, List[float]]) -> callable:
        """Create objective function for a specific regime."""

        def objective(trial: optuna.Trial) -> float:
            """Objective function for regime-specific optimization."""
            try:
                tb_params = self._suggest_triple_barrier_params(trial, regime_constraints)
                tpsl_params = self._suggest_tpsl_params(trial, regime_constraints)
                labeled_data = self._apply_regime_specific_labeling(regime_data, tb_params, tpsl_params)
                metrics = self._evaluate_regime_performance(labeled_data, regime_name)
                score = self._calculate_composite_score(metrics)
                trial.set_user_attr('regime_name', regime_name)
                trial.set_user_attr('tb_params', tb_params.__dict__)
                trial.set_user_attr('tpsl_params', tpsl_params)
                trial.set_user_attr('metrics', metrics)
                return score
            except Exception as e:
                self.logger.warning(f'⚠️ Trial failed for regime {regime_name}: {e}')
                return -np.inf
        return objective

    def _suggest_triple_barrier_params(self, trial: optuna.Trial, regime_constraints: Dict[str, List[float]]) -> RegimeTripleBarrierParams:
        """Suggest triple barrier parameters for a regime."""
        profit_take_multiplier = trial.suggest_float('profit_take_multiplier', 0.01, 0.05, log=True)
        stop_loss_multiplier = trial.suggest_float('stop_loss_multiplier', 0.005, 0.03, log=True)
        time_barrier_minutes = trial.suggest_int('time_barrier_minutes', 15, 120)
        max_lookahead = trial.suggest_int('max_lookahead', 50, 200)
        regime_volatility_multiplier = trial.suggest_float('regime_volatility_multiplier', 0.5, 2.0)
        regime_trend_multiplier = trial.suggest_float('regime_trend_multiplier', 0.5, 2.0)
        regime_volume_multiplier = trial.suggest_float('regime_volume_multiplier', 0.5, 2.0)
        tp_range = regime_constraints.get('tp_multiplier_range', [1.5, 4.0])
        sl_range = regime_constraints.get('sl_multiplier_range', [0.8, 2.0])
        position_range = regime_constraints.get('position_size_range', [0.05, 0.25])
        tp_multiplier = trial.suggest_float('tp_multiplier', tp_range[0], tp_range[1])
        sl_multiplier = trial.suggest_float('sl_multiplier', sl_range[0], sl_range[1])
        position_size = trial.suggest_float('position_size', position_range[0], position_range[1])
        return RegimeTripleBarrierParams(profit_take_multiplier=profit_take_multiplier, stop_loss_multiplier=stop_loss_multiplier, time_barrier_minutes=time_barrier_minutes, max_lookahead=max_lookahead, regime_volatility_multiplier=regime_volatility_multiplier, regime_trend_multiplier=regime_trend_multiplier, regime_volume_multiplier=regime_volume_multiplier, tp_multiplier_range=(tp_multiplier, tp_multiplier * 1.5), sl_multiplier_range=(sl_multiplier * 0.8, sl_multiplier), position_size_range=(position_size * 0.8, position_size * 1.2))

    def _suggest_tpsl_params(self, trial: optuna.Trial, regime_constraints: Dict[str, List[float]]) -> Dict[str, float]:
        """Suggest TPSL parameters for a regime."""
        tp_range = regime_constraints.get('tp_multiplier_range', [1.5, 4.0])
        sl_range = regime_constraints.get('sl_multiplier_range', [0.8, 2.0])
        position_range = regime_constraints.get('position_size_range', [0.05, 0.25])
        return {'tp_multiplier': trial.suggest_float('tpsl_tp_multiplier', tp_range[0], tp_range[1]), 'sl_multiplier': trial.suggest_float('tpsl_sl_multiplier', sl_range[0], sl_range[1]), 'position_size': trial.suggest_float('tpsl_position_size', position_range[0], position_range[1]), 'tp_atr_multiplier': trial.suggest_float('tp_atr_multiplier', 1.0, 4.0), 'sl_atr_multiplier': trial.suggest_float('sl_atr_multiplier', 0.5, 2.0), 'trailing_stop': trial.suggest_float('trailing_stop', 0.0, 0.02), 'break_even_threshold': trial.suggest_float('break_even_threshold', 0.005, 0.02)}

    def _apply_regime_specific_labeling(self, regime_data: pd.DataFrame, tb_params: RegimeTripleBarrierParams, tpsl_params: Dict[str, float]) -> pd.DataFrame:
        """Apply regime-specific triple barrier labeling."""
        try:
            labeler = OptimizedTripleBarrierLabeling(profit_take_multiplier=tb_params.profit_take_multiplier, stop_loss_multiplier=tb_params.stop_loss_multiplier, time_barrier_minutes=tb_params.time_barrier_minutes, max_lookahead=tb_params.max_lookahead, binary_classification=True)
            labeled_data = labeler.apply_triple_barrier_labeling_vectorized(regime_data)
            labeled_data = self._add_tpsl_information(labeled_data, tpsl_params)
            return labeled_data
        except Exception as e:
            self.logger.error(f'❌ Error applying regime-specific labeling: {e}')
            regime_data = regime_data.copy()
            regime_data['label'] = 0
            regime_data['potential_profit_pct'] = 0.0
            return regime_data

    def _add_tpsl_information(self, data: pd.DataFrame, tpsl_params: Dict[str, float]) -> pd.DataFrame:
        """Add TPSL information to the data."""
        data = data.copy()
        if 'atr' not in data.columns:
            data['atr'] = self._calculate_atr(data, period=14)
        data['tp_level'] = data['close'] * (1 + tpsl_params['tp_multiplier'] * data['atr'])
        data['sl_level'] = data['close'] * (1 - tpsl_params['sl_multiplier'] * data['atr'])
        data['position_size'] = tpsl_params['position_size']
        data['trailing_stop'] = tpsl_params['trailing_stop']
        data['break_even_threshold'] = tpsl_params['break_even_threshold']
        return data

    def _calculate_atr(self, data: pd.DataFrame, period: int=14) -> pd.Series:
        """Calculate Average True Range."""
        try:
            high = data['high']
            low = data['low']
            close = data['close']
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean()
            return atr.fillna(method='bfill')
        except Exception:
            return data['close'].pct_change().rolling(window=period).std().fillna(0.01)

    def _evaluate_regime_performance(self, labeled_data: pd.DataFrame, regime_name: str) -> Dict[str, float]:
        """Evaluate performance metrics for a regime."""
        try:
            valid_data = labeled_data[labeled_data['label'] != 0].copy()
            if len(valid_data) < self.optimization_config.min_sample_size:
                return self._get_default_metrics()
            valid_data['returns'] = valid_data['potential_profit_pct']
            total_return = valid_data['returns'].sum()
            win_rate = (valid_data['returns'] > 0).mean()
            profit_factor = self._calculate_profit_factor(valid_data['returns'])
            sharpe_ratio = self._calculate_sharpe_ratio(valid_data['returns'])
            max_drawdown = self._calculate_max_drawdown(valid_data['returns'])
            sortino_ratio = self._calculate_sortino_ratio(valid_data['returns'])
            calmar_ratio = self._calculate_calmar_ratio(total_return, max_drawdown)
            regime_accuracy = self._calculate_regime_accuracy(valid_data, regime_name)
            regime_precision = self._calculate_regime_precision(valid_data, regime_name)
            regime_recall = self._calculate_regime_recall(valid_data, regime_name)
            regime_f1 = self._calculate_regime_f1(regime_precision, regime_recall)
            return {'total_return': total_return, 'win_rate': win_rate, 'profit_factor': profit_factor, 'sharpe_ratio': sharpe_ratio, 'max_drawdown': max_drawdown, 'sortino_ratio': sortino_ratio, 'calmar_ratio': calmar_ratio, 'regime_accuracy': regime_accuracy, 'regime_precision': regime_precision, 'regime_recall': regime_recall, 'regime_f1': regime_f1}
        except Exception as e:
            self.logger.error(f'❌ Error evaluating regime performance: {e}')
            return self._get_default_metrics()

    def _get_default_metrics(self) -> Dict[str, float]:
        """Get default metrics for failed evaluations."""
        return {'total_return': 0.0, 'win_rate': 0.5, 'profit_factor': 1.0, 'sharpe_ratio': 0.0, 'max_drawdown': 0.0, 'sortino_ratio': 0.0, 'calmar_ratio': 0.0, 'regime_accuracy': 0.5, 'regime_precision': 0.5, 'regime_recall': 0.5, 'regime_f1': 0.5}

    def _calculate_composite_score(self, metrics: Dict[str, float]) -> float:
        """Calculate composite optimization score."""
        score = 0.0
        weights = self.optimization_config.objective_weights
        for objective, weight in weights.items():
            if objective in metrics:
                normalized_value = self._normalize_metric(objective, metrics[objective])
                score += weight * normalized_value
        return score

    def _normalize_metric(self, metric_name: str, value: float) -> float:
        """Normalize metric to 0-1 range."""
        normalization_ranges = {'sharpe_ratio': (-2.0, 3.0), 'win_rate': (0.0, 1.0), 'profit_factor': (0.5, 3.0), 'regime_accuracy': (0.0, 1.0), 'total_return': (-0.5, 1.0), 'max_drawdown': (-0.5, 0.0), 'sortino_ratio': (-2.0, 3.0), 'calmar_ratio': (-2.0, 5.0)}
        if metric_name in normalization_ranges:
            min_val, max_val = normalization_ranges[metric_name]
            normalized = (value - min_val) / (max_val - min_val)
            return np.clip(normalized, 0.0, 1.0)
        return np.clip(value, 0.0, 1.0)

    def _calculate_profit_factor(self, returns: pd.Series) -> float:
        """Calculate profit factor."""
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        return gains / losses if losses > 0 else 1.0

    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) < 2:
            return 0.0
        return returns.mean() / returns.std() if returns.std() > 0 else 0.0

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio."""
        if len(returns) < 2:
            return 0.0
        negative_returns = returns[returns < 0]
        downside_std = negative_returns.std() if len(negative_returns) > 0 else 0.0
        return returns.mean() / downside_std if downside_std > 0 else 0.0

    def _calculate_calmar_ratio(self, total_return: float, max_drawdown: float) -> float:
        """Calculate Calmar ratio."""
        return total_return / abs(max_drawdown) if max_drawdown < 0 else 0.0

    def _calculate_regime_accuracy(self, data: pd.DataFrame, regime_name: str) -> float:
        """Calculate regime-specific accuracy."""
        if 'regime' not in data.columns:
            return 0.5
        return (data['regime'] == regime_name).mean()

    def _calculate_regime_precision(self, data: pd.DataFrame, regime_name: str) -> float:
        """Calculate regime-specific precision."""
        if 'regime' not in data.columns:
            return 0.5
        regime_predictions = data['regime'] == regime_name
        return precision_score(regime_predictions, data['label'] > 0, zero_division=0)

    def _calculate_regime_recall(self, data: pd.DataFrame, regime_name: str) -> float:
        """Calculate regime-specific recall."""
        if 'regime' not in data.columns:
            return 0.5
        regime_predictions = data['regime'] == regime_name
        return recall_score(regime_predictions, data['label'] > 0, zero_division=0)

    def _calculate_regime_f1(self, precision: float, recall: float) -> float:
        """Calculate regime-specific F1 score."""
        return 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0

    async def optimize_regime_parameters(self, data: pd.DataFrame, regime_column: str='composite_cluster_id') -> Dict[str, RegimeOptimizationResult]:
        """
        Optimize parameters for each regime.
        
        Args:
            data: DataFrame with regime information
            regime_column: Column containing regime labels
            
        Returns:
            Dictionary mapping regime names to optimization results
        """
        try:
            self.logger.info('🚀 Starting regime-specific triple barrier optimization...')
            regime_names = self._get_regime_names(data)
            self.logger.info(f'📊 Found {len(regime_names)} regimes: {regime_names}')
            for regime_name in regime_names:
                await self._optimize_single_regime(data, regime_name, regime_column)
            await self._generate_optimization_report()
            self.logger.info('✅ Regime-specific optimization completed successfully')
            return self.regime_results
        except Exception as e:
            self.logger.exception(f'❌ Error in regime optimization: {e}')
            return {}

    async def _optimize_single_regime(self, data: pd.DataFrame, regime_name: str, regime_column: str) -> None:
        """Optimize parameters for a single regime."""
        try:
            self.logger.info(f'🎯 Optimizing parameters for regime: {regime_name}')
            if regime_name in data[regime_column].values:
                regime_data = data[data[regime_column] == regime_name].copy()
            else:
                try:
                    regime_id = int(regime_name.split('_')[-1])
                    regime_data = data[data[regime_column] == regime_id].copy()
                except:
                    self.logger.warning(f'⚠️ Could not find data for regime {regime_name}')
                    return
            if len(regime_data) < self.optimization_config.min_sample_size:
                self.logger.warning(f'⚠️ Insufficient data for regime {regime_name}: {len(regime_data)} samples')
                return
            regime_constraints = self.optimization_config.regime_constraints.get(regime_name, self.optimization_config.regime_constraints.get('SIDEWAYS_RANGE', {}))
            study_name = f'{self.study_name_prefix}_{regime_name}'
            study = optuna.create_study(study_name=study_name, storage=self.storage_url, sampler=TPESampler(seed=42), pruner=HyperbandPruner() if self.optimization_config.enable_pruning else None, load_if_exists=True, direction='maximize')
            objective = self._create_regime_objective_function(regime_name, regime_data, regime_constraints)
            start_time = time.time()
            study.optimize(objective, n_trials=self.optimization_config.n_trials_per_regime, timeout=self.optimization_config.timeout_minutes_per_regime * 60, show_progress_bar=True)
            optimization_time = time.time() - start_time
            self.studies[regime_name] = study
            best_trial = study.best_trial
            best_params = best_trial.params
            best_metrics = best_trial.user_attrs.get('metrics', {})
            result = RegimeOptimizationResult(regime_name=regime_name, regime_id=len(self.regime_results), triple_barrier_params=RegimeTripleBarrierParams(**{k: v for k, v in best_params.items() if k in RegimeTripleBarrierParams.__annotations__}), tpsl_params={k: v for k, v in best_params.items() if k.startswith('tpsl_')}, sharpe_ratio=best_metrics.get('sharpe_ratio', 0.0), max_drawdown=best_metrics.get('max_drawdown', 0.0), win_rate=best_metrics.get('win_rate', 0.5), profit_factor=best_metrics.get('profit_factor', 1.0), total_return=best_metrics.get('total_return', 0.0), calmar_ratio=best_metrics.get('calmar_ratio', 0.0), sortino_ratio=best_metrics.get('sortino_ratio', 0.0), regime_accuracy=best_metrics.get('regime_accuracy', 0.5), regime_precision=best_metrics.get('regime_precision', 0.5), regime_recall=best_metrics.get('regime_recall', 0.5), regime_f1=best_metrics.get('regime_f1', 0.5), optimization_score=best_trial.value, n_trials=len(study.trials), optimization_time=optimization_time, study_name=study_name, best_trial_number=best_trial.number, p_value=0.05, confidence_interval=(0.0, 1.0))
            self.regime_results[regime_name] = result
            self.logger.info(f"✅ Optimized {regime_name}: Score={best_trial.value:.4f}, Sharpe={best_metrics.get('sharpe_ratio', 0.0):.4f}, Win Rate={best_metrics.get('win_rate', 0.5):.4f}")
        except Exception as e:
            self.logger.exception(f'❌ Error optimizing regime {regime_name}: {e}')

    async def _generate_optimization_report(self) -> None:
        """Generate comprehensive optimization report."""
        try:
            self.logger.info('📊 Generating optimization report...')
            summary = {'total_regimes': len(self.regime_results), 'optimization_config': self.optimization_config.__dict__, 'regime_results': {name: {'optimization_score': result.optimization_score, 'sharpe_ratio': result.sharpe_ratio, 'win_rate': result.win_rate, 'profit_factor': result.profit_factor, 'n_trials': result.n_trials, 'optimization_time': result.optimization_time} for name, result in self.regime_results.items()}}
            self.global_results = summary
            await self._create_optimization_visualizations()
            self.logger.info('✅ Optimization report generated successfully')
        except Exception as e:
            self.logger.exception(f'❌ Error generating report: {e}')

    async def _create_optimization_visualizations(self) -> None:
        """Create optimization visualizations."""
        try:
            output_dir = Path('optimization_results')
            output_dir.mkdir(exist_ok=True)
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Regime-Specific Triple Barrier Optimization Results', fontsize=16)
            regime_names = list(self.regime_results.keys())
            sharpe_ratios = [r.sharpe_ratio for r in self.regime_results.values()]
            win_rates = [r.win_rate for r in self.regime_results.values()]
            profit_factors = [r.profit_factor for r in self.regime_results.values()]
            optimization_scores = [r.optimization_score for r in self.regime_results.values()]
            axes[0, 0].bar(regime_names, sharpe_ratios, color='skyblue')
            axes[0, 0].set_title('Sharpe Ratios by Regime')
            axes[0, 0].set_ylabel('Sharpe Ratio')
            axes[0, 0].tick_params(axis='x', rotation=45)
            axes[0, 1].bar(regime_names, win_rates, color='lightgreen')
            axes[0, 1].set_title('Win Rates by Regime')
            axes[0, 1].set_ylabel('Win Rate')
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[1, 0].bar(regime_names, profit_factors, color='orange')
            axes[1, 0].set_title('Profit Factors by Regime')
            axes[1, 0].set_ylabel('Profit Factor')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 1].bar(regime_names, optimization_scores, color='purple')
            axes[1, 1].set_title('Optimization Scores by Regime')
            axes[1, 1].set_ylabel('Optimization Score')
            axes[1, 1].tick_params(axis='x', rotation=45)
            plt.tight_layout()
            plt.savefig(output_dir / 'regime_optimization_results.png', dpi=300, bbox_inches='tight')
            plt.close()
            for regime_name, study in self.studies.items():
                try:
                    fig = plot_param_importances(study)
                    fig.update_layout(title=f'Parameter Importance - {regime_name}')
                    fig.write_html(output_dir / f'param_importance_{regime_name}.html')
                    fig = plot_optimization_history(study)
                    fig.update_layout(title=f'Optimization History - {regime_name}')
                    fig.write_html(output_dir / f'optimization_history_{regime_name}.html')
                except Exception as e:
                    self.logger.warning(f'⚠️ Could not create plots for {regime_name}: {e}')
            self.logger.info(f'✅ Visualizations saved to {output_dir}')
        except Exception as e:
            self.logger.exception(f'❌ Error creating visualizations: {e}')

    def get_optimized_parameters(self) -> Dict[str, Any]:
        """Get optimized parameters for all regimes."""
        optimized_params = {}
        for regime_name, result in self.regime_results.items():
            optimized_params[regime_name] = {'triple_barrier_params': result.triple_barrier_params.__dict__, 'tpsl_params': result.tpsl_params, 'performance_metrics': {'sharpe_ratio': result.sharpe_ratio, 'win_rate': result.win_rate, 'profit_factor': result.profit_factor, 'total_return': result.total_return, 'max_drawdown': result.max_drawdown}}
        return optimized_params

    def get_regime_specific_params(self, regime_name: str) -> Optional[Dict[str, Any]]:
        """Get optimized parameters for a specific regime."""
        if regime_name not in self.regime_results:
            return None
        result = self.regime_results[regime_name]
        return {'triple_barrier_params': result.triple_barrier_params.__dict__, 'tpsl_params': result.tpsl_params, 'performance_metrics': {'sharpe_ratio': result.sharpe_ratio, 'win_rate': result.win_rate, 'profit_factor': result.profit_factor, 'total_return': result.total_return, 'max_drawdown': result.max_drawdown}}

async def setup_regime_specific_optimizer(config: Dict[str, Any]) -> RegimeSpecificTripleBarrierOptimizer:
    """Setup and initialize regime-specific optimizer."""
    optimizer = RegimeSpecificTripleBarrierOptimizer(config)
    if not await optimizer.initialize():
        raise RuntimeError('Failed to initialize regime-specific optimizer')
    return optimizer

async def optimize_regime_triple_barrier_parameters(data: pd.DataFrame, config: Dict[str, Any], regime_column: str='composite_cluster_id') -> Dict[str, RegimeOptimizationResult]:
    """
    Optimize regime-specific triple barrier parameters.
    
    Args:
        data: DataFrame with regime information
        config: Configuration dictionary
        regime_column: Column containing regime labels
        
    Returns:
        Dictionary mapping regime names to optimization results
    """
    optimizer = await setup_regime_specific_optimizer(config)
    return await optimizer.optimize_regime_parameters(data, regime_column)

def get_regime_optimized_triple_barrier_params(regime_name: str, optimization_results: Dict[str, RegimeOptimizationResult]) -> Optional[RegimeTripleBarrierParams]:
    """
    Get optimized triple barrier parameters for a specific regime.
    
    Args:
        regime_name: Name of the regime
        optimization_results: Results from optimization
        
    Returns:
        Optimized triple barrier parameters or None if not found
    """
    if regime_name not in optimization_results:
        return None
    return optimization_results[regime_name].triple_barrier_params

def get_regime_optimized_tpsl_params(regime_name: str, optimization_results: Dict[str, RegimeOptimizationResult]) -> Optional[Dict[str, float]]:
    """
    Get optimized TPSL parameters for a specific regime.
    
    Args:
        regime_name: Name of the regime
        optimization_results: Results from optimization
        
    Returns:
        Optimized TPSL parameters or None if not found
    """
    if regime_name not in optimization_results:
        return None
    return optimization_results[regime_name].tpsl_params