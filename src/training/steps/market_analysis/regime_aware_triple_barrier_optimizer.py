"""
MARKET_ANALYSIS Regime-Aware Triple Barrier Optimizer

This module provides regime-aware triple barrier optimization that integrates with HMM regime detection.
It optimizes triple barrier parameters for each market regime to improve labeling accuracy and profitability.

Key Features:
- Regime-specific parameter optimization
- HMM regime integration
- Performance-based barrier adjustment
- Comprehensive regime analysis
- Integration with market analysis pipeline
"""

from src.utils.tprint import tprint
from src.utils.logger import get_logger
from src.core.decorators import handles_errors, traced, validates, log_execution_time, cached
from src.utils.math_validation import safe_divide, validate_positive, MathValidationError

import pandas as pd
import numpy as np
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
import contextlib
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from scipy.optimize import minimize
from src.training.steps.pre_training.multi_horizon_profit_labeler import MultiHorizonConfig, MultiHorizonProfitLabeler
import warnings

# Import the triple barrier labeling module
from ..pre_training.multi_horizon_profit_labeler import (
    MultiHorizonProfitLabeler,
    MultiHorizonConfig
)

@dataclass
class RegimeBarrierParams:
    """Parameters for regime-specific triple barrier optimization."""
    profit_take_multiplier: float = 0.002
    stop_loss_multiplier: float = 0.001
    time_barrier_minutes: int = 30
    max_lookahead: int = 100
    transaction_cost: float = 0.001

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'profit_take_multiplier': self.profit_take_multiplier,
            'stop_loss_multiplier': self.stop_loss_multiplier,
            'time_barrier_minutes': self.time_barrier_minutes,
            'max_lookahead': self.max_lookahead,
            'transaction_cost': self.transaction_cost
        }

    @classmethod
    def from_dict(cls, params_dict: Dict[str, Any]) -> 'RegimeBarrierParams':
        """Create from dictionary."""
        return cls(**params_dict)

@dataclass
class RegimePerformanceMetrics:
    """Performance metrics for a specific regime."""
    regime_id: Union[int, str]
    regime_name: str
    total_samples: int
    labeled_samples: int
    win_rate: float
    avg_profit: float
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    profit_factor: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'regime_id': self.regime_id,
            'regime_name': self.regime_name,
            'total_samples': self.total_samples,
            'labeled_samples': self.labeled_samples,
            'win_rate': self.win_rate,
            'avg_profit': self.avg_profit,
            'total_return': self.total_return,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'profit_factor': self.profit_factor,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score
        }

class RegimeAwareTripleBarrierOptimizer:
    """
    Regime-aware triple barrier optimizer for market analysis.

    This class optimizes triple barrier parameters for each market regime
    detected by HMM analysis to improve labeling accuracy and profitability.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the regime-aware triple barrier optimizer.

        Args:
            config: Configuration dictionary with optimization parameters
        """
        self.config = config or {}
        self.logger = get_logger('RegimeAwareTripleBarrierOptimizer')

        # Default optimization parameters
        self.optimization_params = {
            'profit_take_range': (0.0005, 0.01),  # 0.05% to 1%
            'stop_loss_range': (0.0005, 0.005),   # 0.05% to 0.5%
            'time_barrier_range': (15, 60),       # 15 to 60 minutes
            'max_lookahead_range': (50, 200),     # 50 to 200 points
            'transaction_cost': 0.001,            # 0.10%
            'optimization_method': 'minimize',
            'objective_function': 'sharpe_ratio',
            'max_iterations': 100,
            'convergence_tolerance': 1e-6
        }

        # Update with provided config
        self.optimization_params.update(self.config)

        # Initialize regime parameters storage
        self.regime_parameters: Dict[Union[int, str], RegimeBarrierParams] = {}
        self.regime_performance: Dict[Union[int, str], RegimePerformanceMetrics] = {}
        self.optimization_history: List[Dict[str, Any]] = []

        self._log_initialization()

    def _log_initialization(self):
        """Log initialization parameters."""
        self.logger.info('🚀 Initializing Regime-Aware Triple Barrier Optimizer')
        self.logger.info(f'📋 Optimization parameters:')
        self.logger.info(f'   → Profit take range: {self.optimization_params["profit_take_range"]}')
        self.logger.info(f'   → Stop loss range: {self.optimization_params["stop_loss_range"]}')
        self.logger.info(f'   → Time barrier range: {self.optimization_params["time_barrier_range"]}')
        self.logger.info(f'   → Max lookahead range: {self.optimization_params["max_lookahead_range"]}')
        self.logger.info(f'   → Optimization method: {self.optimization_params["optimization_method"]}')
        self.logger.info(f'   → Objective function: {self.optimization_params["objective_function"]}')

    @traced(span_name='optimize_regime_parameters')
    @validates()
    @handles_errors(exceptions=(Exception,), default_return={})
    @log_execution_time()
    def optimize_regime_parameters(
        self,
        data: pd.DataFrame,
        regime_column: str = 'hmm_regime',
        validation_split: float = 0.2,
        random_state: int = 42
    ) -> Dict[Union[int, str], RegimeBarrierParams]:
        """Optimize triple barrier parameters for each regime.

        Args:
            data: DataFrame with OHLCV data and regime information
            regime_column: Column containing regime labels
            validation_split: Fraction of data to use for validation
            random_state: Random state for reproducibility

        Returns:
            Dictionary mapping regime IDs to optimized parameters
        """
        self.logger.info(f'🎯 Starting regime parameter optimization')
        self.logger.info(f'   Data shape: {data.shape}')
        self.logger.info(f'   Regime column: {regime_column}')

        if regime_column not in data.columns:
            self.logger.error(f'❌ Regime column "{regime_column}" not found in data')
            return {}

        # Get unique regimes
        regimes = data[regime_column].unique()
        self.logger.info(f'📊 Found {len(regimes)} unique regimes: {regimes}')

        # Split data for validation
        np.random.seed(random_state)
        validation_mask = np.random.random(len(data)) < validation_split
        train_data = data[~validation_mask].copy()
        val_data = data[validation_mask].copy()

        self.logger.info(f'📊 Data split: {len(train_data)} training, {len(val_data)} validation')

        # Optimize parameters for each regime
        for regime in regimes:
            self.logger.info(f'🔄 Optimizing parameters for regime {regime}')

            try:
                # Get regime-specific data
                regime_train_data = train_data[train_data[regime_column] == regime].copy()
                regime_val_data = val_data[val_data[regime_column] == regime].copy()

                if len(regime_train_data) < 50:  # Minimum samples for optimization
                    self.logger.warning(f'⚠️ Insufficient data for regime {regime} ({len(regime_train_data)} samples)')
                    # Use default parameters
                    self.regime_parameters[regime] = RegimeBarrierParams()
                    continue

                # Optimize parameters for this regime
                optimized_params = self._optimize_single_regime(
                    regime_train_data, regime_val_data, regime, regime_column
                )

                self.regime_parameters[regime] = optimized_params

                # Calculate performance metrics
                performance = self._calculate_regime_performance(
                    regime_val_data, optimized_params, regime, regime_column
                )
                self.regime_performance[regime] = performance

                self.logger.info(f'✅ Regime {regime} optimization completed')
                self.logger.info(f'   → Optimized profit take: {optimized_params.profit_take_multiplier:.4f}')
                self.logger.info(f'   → Optimized stop loss: {optimized_params.stop_loss_multiplier:.4f}')
                self.logger.info(f'   → Win rate: {performance.win_rate:.3f}')
                self.logger.info(f'   → Sharpe ratio: {performance.sharpe_ratio:.3f}')

            except Exception as e:
                self.logger.error(f'❌ Error optimizing regime {regime}: {e}')
                # Use default parameters as fallback
                self.regime_parameters[regime] = RegimeBarrierParams()

        self.logger.info(f'✅ Regime parameter optimization completed for {len(self.regime_parameters)} regimes')
        return self.regime_parameters

    def _optimize_single_regime(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        regime: Union[int, str],
        regime_column: str
    ) -> RegimeBarrierParams:
        """Optimize parameters for a single regime.

        Args:
            train_data: Training data for this regime
            val_data: Validation data for this regime
            regime: Regime identifier
            regime_column: Column containing regime labels

        Returns:
            Optimized parameters for this regime
        """
        # Define objective function
        def objective(params):
            try:
                # Extract parameters
                profit_take = max(0.0005, min(0.01, params[0]))
                stop_loss = max(0.0005, min(0.005, params[1]))
                time_barrier = max(15, min(60, int(params[2])))
                max_lookahead = max(50, min(200, int(params[3])))

                # Create regime-specific config
                regime_config = MultiHorizonConfig(
                    profit_take_multiplier=profit_take,
                    stop_loss_multiplier=stop_loss,
                    time_barrier_minutes=time_barrier,
                    max_lookahead=max_lookahead,
                    transaction_cost=self.optimization_params['transaction_cost'],
                    binary_classification=True,
                    regime_aware=False  # We're optimizing for a single regime
                )

                # Create labeler with regime-specific config
                labeler = MultiHorizonProfitLabeler(regime_config)

                # Apply labeling to validation data
                result = labeler.apply_labeling(val_data)
                labeled_data = result.labeled_data if result.success else pd.DataFrame()

                if len(labeled_data) == 0:
                    return -1.0  # Poor performance if no labels generated

                # Calculate performance metric
                performance = self._calculate_objective_metric(labeled_data)

                return -performance  # Minimize negative performance (maximize performance)

            except Exception as e:
                self.logger.warning(f'⚠️ Error in objective function: {e}')
                return -1.0

        # Define parameter bounds
        bounds = [
            self.optimization_params['profit_take_range'],
            self.optimization_params['stop_loss_range'],
            self.optimization_params['time_barrier_range'],
            self.optimization_params['max_lookahead_range']
        ]

        # Initial parameters (middle of ranges)
        initial_params = [
            np.mean(self.optimization_params['profit_take_range']),
            np.mean(self.optimization_params['stop_loss_range']),
            np.mean(self.optimization_params['time_barrier_range']),
            np.mean(self.optimization_params['max_lookahead_range'])
        ]

        # Optimize using scipy
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            result = minimize(
                objective,
                initial_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={
                    'maxiter': self.optimization_params['max_iterations'],
                    'ftol': self.optimization_params['convergence_tolerance']
                }
            )

        # Extract optimized parameters
        if result.success:
            optimized_profit_take = max(0.0005, min(0.01, result.x[0]))
            optimized_stop_loss = max(0.0005, min(0.005, result.x[1]))
            optimized_time_barrier = max(15, min(60, int(result.x[2])))
            optimized_max_lookahead = max(50, min(200, int(result.x[3])))

            self.logger.info(f'🎯 Optimization successful for regime {regime}')
            self.logger.info(f'   → Final objective: {-result.fun:.4f}')
            self.logger.info(f'   → Iterations: {result.nit}')

        else:
            self.logger.warning(f'⚠️ Optimization failed for regime {regime}, using default parameters')
            optimized_profit_take = 0.002
            optimized_stop_loss = 0.001
            optimized_time_barrier = 30
            optimized_max_lookahead = 100

        return RegimeBarrierParams(
            profit_take_multiplier=optimized_profit_take,
            stop_loss_multiplier=optimized_stop_loss,
            time_barrier_minutes=optimized_time_barrier,
            max_lookahead=optimized_max_lookahead,
            transaction_cost=self.optimization_params['transaction_cost']
        )

    def _calculate_objective_metric(self, labeled_data: pd.DataFrame) -> float:
        """Calculate the objective metric for optimization.

        Args:
            labeled_data: DataFrame with labels and profit information

        Returns:
            Objective metric value (higher is better)
        """
        if len(labeled_data) == 0:
            return 0.0

        # Get labels and profits
        labels = labeled_data['label'].values
        profits = labeled_data.get('net_profit_pct', labeled_data.get('potential_profit_pct', np.zeros(len(labels)))).values

        # Calculate different metrics based on objective function
        objective_function = self.optimization_params['objective_function']

        if objective_function == 'sharpe_ratio':
            return self._calculate_sharpe_ratio(profits)
        elif objective_function == 'win_rate':
            return self._calculate_win_rate(labels)
        elif objective_function == 'profit_factor':
            return self._calculate_profit_factor(profits)
        elif objective_function == 'total_return':
            return np.sum(profits)
        else:
            # Default to Sharpe ratio
            return self._calculate_sharpe_ratio(profits)

    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        return np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized

    def _calculate_win_rate(self, labels: np.ndarray) -> float:
        """Calculate win rate."""
        if len(labels) == 0:
            return 0.0
        return np.mean(labels > 0)

    def _calculate_profit_factor(self, profits: np.ndarray) -> float:
        """Calculate profit factor."""
        if len(profits) == 0:
            return 0.0

        gross_profit = np.sum(profits[profits > 0])
        gross_loss = abs(np.sum(profits[profits < 0]))

        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0

        return gross_profit / gross_loss

    def _calculate_regime_performance(
        self,
        data: pd.DataFrame,
        params: RegimeBarrierParams,
        regime: Union[int, str],
        regime_column: str
    ) -> RegimePerformanceMetrics:
        """Calculate comprehensive performance metrics for a regime.

        Args:
            data: Validation data for this regime
            params: Optimized parameters for this regime
            regime: Regime identifier
            regime_column: Column containing regime labels

        Returns:
            Comprehensive performance metrics
        """
        # Create labeler with optimized parameters
        regime_config = MultiHorizonConfig(
            profit_take_multiplier=params.profit_take_multiplier,
            stop_loss_multiplier=params.stop_loss_multiplier,
            time_barrier_minutes=params.time_barrier_minutes,
            max_lookahead=params.max_lookahead,
            transaction_cost=params.transaction_cost,
            binary_classification=True,
            regime_aware=False
        )

        labeler = MultiHorizonProfitLabeler(regime_config)
        result = labeler.apply_labeling(data)
        labeled_data = result.labeled_data if result.success else pd.DataFrame()

        if len(labeled_data) == 0:
            return RegimePerformanceMetrics(
                regime_id=regime,
                regime_name=f'Regime_{regime}',
                total_samples=len(data),
                labeled_samples=0,
                win_rate=0.0,
                avg_profit=0.0,
                total_return=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                profit_factor=0.0,
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0
            )

        # Calculate metrics
        labels = labeled_data['label'].values
        profits = labeled_data.get('net_profit_pct', labeled_data.get('potential_profit_pct', np.zeros(len(labels)))).values

        # Basic metrics
        total_samples = len(data)
        labeled_samples = len(labeled_data)
        win_rate = np.mean(labels > 0) if len(labels) > 0 else 0.0
        avg_profit = np.mean(profits) if len(profits) > 0 else 0.0
        total_return = np.sum(profits)

        # Risk metrics
        sharpe_ratio = self._calculate_sharpe_ratio(profits)
        max_drawdown = self._calculate_max_drawdown(profits)
        profit_factor = self._calculate_profit_factor(profits)

        # Classification metrics
        if len(labels) > 0:
            # Create binary targets for classification metrics
            y_true = (labels > 0).astype(int)
            y_pred = (profits > 0).astype(int)

            try:
                accuracy = accuracy_score(y_true, y_pred)
                precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
            except:
                accuracy = 0.0
                precision = 0.0
                recall = 0.0
                f1 = 0.0
        else:
            accuracy = 0.0
            precision = 0.0
            recall = 0.0
            f1 = 0.0

        return RegimePerformanceMetrics(
            regime_id=regime,
            regime_name=f'Regime_{regime}',
            total_samples=total_samples,
            labeled_samples=labeled_samples,
            win_rate=win_rate,
            avg_profit=avg_profit,
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            profit_factor=profit_factor,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1
        )

    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        if len(returns) == 0:
            return 0.0

        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        return abs(np.min(drawdown))

    @traced(span_name='apply_optimized_labeling')
    @handles_errors(exceptions=(Exception,), default_return=pd.DataFrame())
    def apply_optimized_labeling(
        self,
        data: pd.DataFrame,
        regime_column: str = 'hmm_regime'
    ) -> pd.DataFrame:
        """Apply optimized triple barrier labeling using regime-specific parameters.

        Args:
            data: DataFrame with OHLCV data and regime information
            regime_column: Column containing regime labels

        Returns:
            DataFrame with optimized regime-aware labels
        """
        self.logger.info(f'🎯 Applying optimized regime-aware labeling')
        self.logger.info(f'   Data shape: {data.shape}')
        self.logger.info(f'   Regime column: {regime_column}')

        if not self.regime_parameters:
            self.logger.warning('⚠️ No optimized parameters available, using default labeling')
            config = MultiHorizonConfig()
            labeler = MultiHorizonProfitLabeler(config)
            return labeler.label(data)

        if regime_column not in data.columns:
            self.logger.error(f'❌ Regime column "{regime_column}" not found in data')
            return pd.DataFrame()

        # Initialize result arrays
        n = len(data)
        labels = np.zeros(n, dtype=np.int8)
        profit_pcts = np.zeros(n, dtype=np.float64)
        transaction_costs = np.zeros(n, dtype=np.float64)
        regime_params_used = np.zeros(n, dtype=object)

        # Process each regime with optimized parameters
        regimes = data[regime_column].unique()

        for regime in regimes:
            regime_mask = data[regime_column] == regime
            regime_data = data[regime_mask]

            if len(regime_data) < 2:
                continue

            # Get optimized parameters for this regime
            if regime in self.regime_parameters:
                params = self.regime_parameters[regime]
                self.logger.info(f'🎯 Using optimized parameters for regime {regime}')
                self.logger.info(f'   → Profit take: {params.profit_take_multiplier:.4f}')
                self.logger.info(f'   → Stop loss: {params.stop_loss_multiplier:.4f}')
            else:
                # Use default parameters if optimization failed
                params = RegimeBarrierParams()
                self.logger.warning(f'⚠️ Using default parameters for regime {regime}')

            # Apply labeling with regime-specific parameters
            regime_config = MultiHorizonConfig(
                profit_take_multiplier=params.profit_take_multiplier,
                stop_loss_multiplier=params.stop_loss_multiplier,
                time_barrier_minutes=params.time_barrier_minutes,
                max_lookahead=params.max_lookahead,
                transaction_cost=params.transaction_cost,
                binary_classification=True,
                regime_aware=False
            )

            labeler = MultiHorizonProfitLabeler(regime_config)
            result = labeler.apply_labeling(regime_data)
            labeled_regime_data = result.labeled_data if result.success else pd.DataFrame()

            if len(labeled_regime_data) > 0:
                # Store results
                labels[regime_mask] = labeled_regime_data['label'].values
                profit_pcts[regime_mask] = labeled_regime_data.get('net_profit_pct', labeled_regime_data.get('potential_profit_pct', np.zeros(len(labeled_regime_data)))).values
                transaction_costs[regime_mask] = labeled_regime_data.get('transaction_cost', np.zeros(len(labeled_regime_data))).values
                regime_params_used[regime_mask] = str(regime)

        # Create result dataframe
        result_data = data.copy()
        result_data['label'] = labels
        result_data['potential_profit_pct'] = profit_pcts
        result_data['transaction_cost'] = transaction_costs
        result_data['net_profit_pct'] = profit_pcts
        result_data['labeling_method'] = 'regime_optimized'
        result_data['regime_params_used'] = regime_params_used

        # Filter out HOLD samples for binary classification
        original_count = len(result_data)
        result_data = result_data[result_data['label'] != 0].copy()
        filtered_count = len(result_data)

        self.logger.info(f'✅ Optimized regime-aware labeling completed')
        self.logger.info(f'   → Total samples: {original_count}')
        self.logger.info(f'   → Labeled samples: {filtered_count}')
        self.logger.info(f'   → Regimes processed: {len(regimes)}')

        return result_data

    def save_optimization_results(self, file_path: Union[str, Path]):
        """Save optimization results to file.

        Args:
            file_path: Path to save the results
        """
        file_path = Path(file_path)

        results = {
            'regime_parameters': {
                str(regime): params.to_dict()
                for regime, params in self.regime_parameters.items()
            },
            'regime_performance': {
                str(regime): metrics.to_dict()
                for regime, metrics in self.regime_performance.items()
            },
            'optimization_params': self.optimization_params,
            'optimization_history': self.optimization_history,
            'timestamp': datetime.now().isoformat()
        }

        with open(file_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        self.logger.info(f'💾 Optimization results saved to {file_path}')

    def load_optimization_results(self, file_path: Union[str, Path]):
        """Load optimization results from file.

        Args:
            file_path: Path to load the results from
        """
        file_path = Path(file_path)

        if not file_path.exists():
            self.logger.error(f'❌ Optimization results file not found: {file_path}')
            return

        with open(file_path, 'r') as f:
            results = json.load(f)

        # Load regime parameters
        self.regime_parameters = {
            int(regime) if regime.isdigit() else regime: RegimeBarrierParams.from_dict(params)
            for regime, params in results.get('regime_parameters', {}).items()
        }

        # Load regime performance
        self.regime_performance = {}
        for regime, metrics_dict in results.get('regime_performance', {}).items():
            regime_id = int(regime) if regime.isdigit() else regime
            self.regime_performance[regime_id] = RegimePerformanceMetrics(**metrics_dict)

        self.logger.info(f'📂 Optimization results loaded from {file_path}')
        self.logger.info(f'   → Loaded parameters for {len(self.regime_parameters)} regimes')
        self.logger.info(f'   → Loaded performance for {len(self.regime_performance)} regimes')

    def generate_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report."""
        self.logger.info('📋 Generating optimization report...')

        # Calculate summary statistics
        total_regimes = len(self.regime_parameters)
        avg_win_rate = np.mean([metrics.win_rate for metrics in self.regime_performance.values()])
        avg_sharpe_ratio = np.mean([metrics.sharpe_ratio for metrics in self.regime_performance.values()])
        avg_profit_factor = np.mean([metrics.profit_factor for metrics in self.regime_performance.values()])

        # Find best and worst performing regimes
        best_regime = max(self.regime_performance.items(), key=lambda x: x[1].sharpe_ratio)[0] if self.regime_performance else None
        worst_regime = min(self.regime_performance.items(), key=lambda x: x[1].sharpe_ratio)[0] if self.regime_performance else None

        report = {
            'summary': {
                'total_regimes': total_regimes,
                'avg_win_rate': avg_win_rate,
                'avg_sharpe_ratio': avg_sharpe_ratio,
                'avg_profit_factor': avg_profit_factor,
                'best_regime': str(best_regime) if best_regime is not None else None,
                'worst_regime': str(worst_regime) if worst_regime is not None else None
            },
            'regime_details': {
                str(regime): {
                    'parameters': params.to_dict(),
                    'performance': metrics.to_dict()
                }
                for regime, params in self.regime_parameters.items()
                for metrics in [self.regime_performance.get(regime)]
                if metrics is not None
            },
            'optimization_params': self.optimization_params,
            'timestamp': datetime.now().isoformat()
        }

        self.logger.info('✅ Optimization report generated')
        return report

# Convenience functions
def optimize_regime_barriers(
    data: pd.DataFrame,
    regime_column: str = 'hmm_regime',
    config: Optional[Dict[str, Any]] = None,
    save_results: Optional[Union[str, Path]] = None
) -> RegimeAwareTripleBarrierOptimizer:
    """Optimize regime-specific triple barrier parameters.

    Args:
        data: DataFrame with OHLCV data and regime information
        regime_column: Column containing regime labels
        config: Optimization configuration
        save_results: Optional path to save results

    Returns:
        Optimized RegimeAwareTripleBarrierOptimizer instance
    """
    optimizer = RegimeAwareTripleBarrierOptimizer(config)
    optimizer.optimize_regime_parameters(data, regime_column)

    if save_results:
        optimizer.save_optimization_results(save_results)

    return optimizer

def apply_optimized_regime_labeling(
    data: pd.DataFrame,
    optimizer: RegimeAwareTripleBarrierOptimizer,
    regime_column: str = 'hmm_regime'
) -> pd.DataFrame:
    """Apply optimized regime-aware labeling.

    Args:
        data: DataFrame with OHLCV data and regime information
        optimizer: Optimized RegimeAwareTripleBarrierOptimizer instance
        regime_column: Column containing regime labels

    Returns:
        DataFrame with optimized regime-aware labels
    """
    return optimizer.apply_optimized_labeling(data, regime_column)

if __name__ == '__main__':
    # Test the regime-aware optimizer
    tprint('🧪 Testing Regime-Aware Triple Barrier Optimizer')

    # Create test data with regimes
    dates = pd.date_range('2024-01-01', periods=2000, freq='1min')
    data = pd.DataFrame({
        'open': np.random.uniform(100, 110, 2000),
        'high': np.random.uniform(105, 115, 2000),
        'low': np.random.uniform(95, 105, 2000),
        'close': np.random.uniform(100, 110, 2000),
        'volume': np.random.uniform(1000, 10000, 2000),
        'hmm_regime': np.random.choice([0, 1, 2], 2000, p=[0.4, 0.4, 0.2])  # Imbalanced regimes
    }, index=dates)

    # Test optimization
    tprint('\n🎯 Testing regime parameter optimization...')
    optimizer = optimize_regime_barriers(data, save_results='regime_optimization_results.json')

    # Test optimized labeling
    tprint('\n📊 Testing optimized regime labeling...')
    optimized_labeled = apply_optimized_regime_labeling(data, optimizer)
    tprint(f'Optimized labeling completed: {len(optimized_labeled)} samples labeled')

    # Generate report
    tprint('\n📋 Generating optimization report...')
    report = optimizer.generate_optimization_report()
    tprint(f'Optimization report generated with {report["summary"]["total_regimes"]} regimes')

    tprint('✅ Regime-Aware Triple Barrier Optimizer test completed successfully!')
