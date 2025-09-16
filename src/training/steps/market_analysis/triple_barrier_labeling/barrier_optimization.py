"""
Barrier Optimization using Optuna and Grid Search

This module implements comprehensive barrier optimization for the Tactician model
using a multi-stage approach:
1. Coarse grid search (5x5x5) for initial exploration
2. Fine grid search (5x5x5) around promising regions
3. Optuna/TPE optimization for final refinement

The optimization focuses on finding the best entry point where price will move
in the desired direction without going further in the opposite direction.
This is about directional prediction accuracy and minimizing adverse price movement.

Requires Optuna for optimal performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List, Callable
import logging
from dataclasses import dataclass
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Optuna imports (required)
import optuna
from optuna.samplers import TPESampler

from .tactician_barrier_config import TacticianBarrierConfig, TacticianBarrierLabeler
from .unified_labeler import UnifiedTripleBarrierLabeler

logger = logging.getLogger(__name__)


@dataclass
class BarrierOptimizationResult:
    """Result of barrier optimization."""
    best_profit_take: float
    best_stop_loss: float
    best_time_barrier: int
    best_score: float
    optimization_method: str
    optimization_time: float
    n_trials: int
    optimization_history: List[Dict[str, Any]]
    best_config: TacticianBarrierConfig


class BarrierOptimizer:
    """
    Comprehensive barrier optimizer using multi-stage optimization.
    
    Optimization stages:
    1. Coarse grid search (5x5) - broad exploration
    2. Fine grid search (5x5) - focused refinement
    3. Optuna/TPE - final optimization
    """
    
    def __init__(self, 
                 data: pd.DataFrame,
                 analyst_signals: np.ndarray,
                 optimization_metric: str = "directional_entry_quality",
                 n_jobs: int = -1):
        """
        Initialize barrier optimizer.
        
        Args:
            data: OHLC data for optimization
            analyst_signals: Binary signals from Analyst
            optimization_metric: Metric to optimize (directional_entry_quality, sharpe_ratio, profit_factor, win_rate)
            n_jobs: Number of parallel jobs (-1 for all cores)
        """
        self.data = data
        self.analyst_signals = analyst_signals
        self.optimization_metric = optimization_metric
        self.n_jobs = n_jobs
        self.logger = logger.getChild('BarrierOptimizer')
        
        # Optimization history
        self.optimization_history = []
        self.best_score = -np.inf
        self.best_params = None
        
        # Parameter ranges for optimization
        self.param_ranges = {
            'profit_take_multiplier': (0.0005, 0.005),  # 0.05% to 0.5%
            'stop_loss_multiplier': (0.0003, 0.003),    # 0.03% to 0.3%
            'time_barrier_minutes': (5, 30)             # 5 to 30 minutes
        }
        
        self.logger.info(f"🚀 Barrier optimizer initialized with {optimization_metric} metric")
    
    def optimize_barriers(self, 
                         method: str = "multi_stage",
                         max_trials: int = 100) -> BarrierOptimizationResult:
        """
        Optimize barrier parameters using specified method.
        
        Args:
            method: Optimization method ("grid_coarse", "grid_fine", "optuna", "multi_stage")
            max_trials: Maximum number of trials for Optuna
            
        Returns:
            BarrierOptimizationResult with best parameters
        """
        start_time = time.time()
        
        if method == "multi_stage":
            return self._multi_stage_optimization(max_trials)
        elif method == "grid_coarse":
            return self._coarse_grid_search()
        elif method == "grid_fine":
            return self._fine_grid_search()
        elif method == "optuna":
            return self._optuna_optimization(max_trials)
        else:
            raise ValueError(f"Unknown optimization method: {method}")
    
    def _multi_stage_optimization(self, max_trials: int) -> BarrierOptimizationResult:
        """Multi-stage optimization: coarse grid -> fine grid -> optuna."""
        self.logger.info("🔄 Starting multi-stage barrier optimization...")
        
        # Stage 1: Coarse grid search
        self.logger.info("📊 Stage 1: Coarse grid search (5x5x5)")
        coarse_result = self._coarse_grid_search()
        
        # Stage 2: Fine grid search around best region
        self.logger.info("📊 Stage 2: Fine grid search around best region")
        fine_result = self._fine_grid_search_around_best(coarse_result)
        
        # Stage 3: Optuna optimization
        self.logger.info("📊 Stage 3: Optuna/TPE optimization")
        optuna_result = self._optuna_optimization_around_best(fine_result, max_trials)
        final_result = optuna_result
        
        # Update final result with multi-stage info
        final_result.optimization_method = "multi_stage"
        final_result.optimization_time = time.time() - time.time()
        
        self.logger.info(f"✅ Multi-stage optimization completed: {final_result.best_score:.4f}")
        return final_result
    
    def _coarse_grid_search(self) -> BarrierOptimizationResult:
        """Coarse grid search (5x5x5) for initial exploration."""
        start_time = time.time()
        
        # Define coarse grid
        profit_take_values = np.linspace(
            self.param_ranges['profit_take_multiplier'][0],
            self.param_ranges['profit_take_multiplier'][1], 5
        )
        stop_loss_values = np.linspace(
            self.param_ranges['stop_loss_multiplier'][0],
            self.param_ranges['stop_loss_multiplier'][1], 5
        )
        time_barrier_values = np.linspace(
            self.param_ranges['time_barrier_minutes'][0],
            self.param_ranges['time_barrier_minutes'][1], 5, dtype=int
        )
        
        best_score = -np.inf
        best_params = None
        trial_count = 0
        
        # Grid search
        for pt in profit_take_values:
            for sl in stop_loss_values:
                for tb in time_barrier_values:
                    # Validate parameter combination
                    if pt <= sl:  # Profit take should be larger than stop loss
                        continue
                    
                    score = self._evaluate_barrier_config(pt, sl, tb)
                    trial_count += 1
                    
                    if score > best_score:
                        best_score = score
                        best_params = {
                            'profit_take_multiplier': pt,
                            'stop_loss_multiplier': sl,
                            'time_barrier_minutes': tb
                        }
                    
                    self.optimization_history.append({
                        'trial': trial_count,
                        'profit_take': pt,
                        'stop_loss': sl,
                        'time_barrier': tb,
                        'score': score,
                        'method': 'coarse_grid'
                    })
        
        # Create result
        result = BarrierOptimizationResult(
            best_profit_take=best_params['profit_take_multiplier'],
            best_stop_loss=best_params['stop_loss_multiplier'],
            best_time_barrier=best_params['time_barrier_minutes'],
            best_score=best_score,
            optimization_method="coarse_grid",
            optimization_time=time.time() - start_time,
            n_trials=trial_count,
            optimization_history=self.optimization_history.copy(),
            best_config=self._create_config_from_params(best_params)
        )
        
        self.logger.info(f"✅ Coarse grid search completed: {best_score:.4f} in {trial_count} trials")
        return result
    
    def _fine_grid_search_around_best(self, coarse_result: BarrierOptimizationResult) -> BarrierOptimizationResult:
        """Fine grid search around the best parameters from coarse search."""
        start_time = time.time()
        
        # Define fine grid around best parameters
        pt_center = coarse_result.best_profit_take
        sl_center = coarse_result.best_stop_loss
        tb_center = coarse_result.best_time_barrier
        
        # Fine grid ranges (20% around best)
        pt_range = (pt_center * 0.8, pt_center * 1.2)
        sl_range = (sl_center * 0.8, sl_center * 1.2)
        tb_range = (max(5, tb_center - 5), min(30, tb_center + 5))
        
        profit_take_values = np.linspace(pt_range[0], pt_range[1], 5)
        stop_loss_values = np.linspace(sl_range[0], sl_range[1], 5)
        time_barrier_values = np.linspace(tb_range[0], tb_range[1], 5, dtype=int)
        
        best_score = coarse_result.best_score
        best_params = {
            'profit_take_multiplier': coarse_result.best_profit_take,
            'stop_loss_multiplier': coarse_result.best_stop_loss,
            'time_barrier_minutes': coarse_result.best_time_barrier
        }
        trial_count = 0
        
        # Fine grid search
        for pt in profit_take_values:
            for sl in stop_loss_values:
                for tb in time_barrier_values:
                    if pt <= sl:
                        continue
                    
                    score = self._evaluate_barrier_config(pt, sl, tb)
                    trial_count += 1
                    
                    if score > best_score:
                        best_score = score
                        best_params = {
                            'profit_take_multiplier': pt,
                            'stop_loss_multiplier': sl,
                            'time_barrier_minutes': tb
                        }
                    
                    self.optimization_history.append({
                        'trial': trial_count,
                        'profit_take': pt,
                        'stop_loss': sl,
                        'time_barrier': tb,
                        'score': score,
                        'method': 'fine_grid'
                    })
        
        # Create result
        result = BarrierOptimizationResult(
            best_profit_take=best_params['profit_take_multiplier'],
            best_stop_loss=best_params['stop_loss_multiplier'],
            best_time_barrier=best_params['time_barrier_minutes'],
            best_score=best_score,
            optimization_method="fine_grid",
            optimization_time=time.time() - start_time,
            n_trials=trial_count,
            optimization_history=self.optimization_history.copy(),
            best_config=self._create_config_from_params(best_params)
        )
        
        self.logger.info(f"✅ Fine grid search completed: {best_score:.4f} in {trial_count} trials")
        return result
    
    def _optuna_optimization(self, max_trials: int) -> BarrierOptimizationResult:
        """Optuna optimization with TPE sampler."""
        
        start_time = time.time()
        
        def objective(trial):
            # Suggest parameters
            profit_take = trial.suggest_float(
                'profit_take_multiplier',
                self.param_ranges['profit_take_multiplier'][0],
                self.param_ranges['profit_take_multiplier'][1]
            )
            stop_loss = trial.suggest_float(
                'stop_loss_multiplier',
                self.param_ranges['stop_loss_multiplier'][0],
                self.param_ranges['stop_loss_multiplier'][1]
            )
            time_barrier = trial.suggest_int(
                'time_barrier_minutes',
                self.param_ranges['time_barrier_minutes'][0],
                self.param_ranges['time_barrier_minutes'][1]
            )
            
            # Validate parameters
            if profit_take <= stop_loss:
                return -np.inf
            
            return self._evaluate_barrier_config(profit_take, stop_loss, time_barrier)
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42)
        )
        
        # Optimize
        study.optimize(objective, n_trials=max_trials, n_jobs=self.n_jobs)
        
        # Get best parameters
        best_params = study.best_params
        best_score = study.best_value
        
        # Create result
        result = BarrierOptimizationResult(
            best_profit_take=best_params['profit_take_multiplier'],
            best_stop_loss=best_params['stop_loss_multiplier'],
            best_time_barrier=best_params['time_barrier_minutes'],
            best_score=best_score,
            optimization_method="optuna",
            optimization_time=time.time() - start_time,
            n_trials=max_trials,
            optimization_history=[],  # Optuna handles its own history
            best_config=self._create_config_from_params(best_params)
        )
        
        self.logger.info(f"✅ Optuna optimization completed: {best_score:.4f} in {max_trials} trials")
        return result
    
    def _optuna_optimization_around_best(self, 
                                       fine_result: BarrierOptimizationResult, 
                                       max_trials: int) -> BarrierOptimizationResult:
        """Optuna optimization around the best parameters from fine grid search."""
        
        start_time = time.time()
        
        # Define ranges around best parameters
        pt_center = fine_result.best_profit_take
        sl_center = fine_result.best_stop_loss
        tb_center = fine_result.best_time_barrier
        
        # Narrow ranges for final optimization (10% around best)
        pt_range = (pt_center * 0.9, pt_center * 1.1)
        sl_range = (sl_center * 0.9, sl_center * 1.1)
        tb_range = (max(5, tb_center - 3), min(30, tb_center + 3))
        
        def objective(trial):
            profit_take = trial.suggest_float('profit_take_multiplier', pt_range[0], pt_range[1])
            stop_loss = trial.suggest_float('stop_loss_multiplier', sl_range[0], sl_range[1])
            time_barrier = trial.suggest_int('time_barrier_minutes', tb_range[0], tb_range[1])
            
            if profit_take <= stop_loss:
                return -np.inf
            
            return self._evaluate_barrier_config(profit_take, stop_loss, time_barrier)
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42)
        )
        
        # Optimize
        study.optimize(objective, n_trials=max_trials, n_jobs=self.n_jobs)
        
        # Get best parameters
        best_params = study.best_params
        best_score = study.best_value
        
        # Create result
        result = BarrierOptimizationResult(
            best_profit_take=best_params['profit_take_multiplier'],
            best_stop_loss=best_params['stop_loss_multiplier'],
            best_time_barrier=best_params['time_barrier_minutes'],
            best_score=best_score,
            optimization_method="optuna_refined",
            optimization_time=time.time() - start_time,
            n_trials=max_trials,
            optimization_history=[],
            best_config=self._create_config_from_params(best_params)
        )
        
        self.logger.info(f"✅ Optuna refined optimization completed: {best_score:.4f}")
        return result
    
    def _evaluate_barrier_config(self, 
                                profit_take: float, 
                                stop_loss: float, 
                                time_barrier: int) -> float:
        """Evaluate a barrier configuration and return optimization score."""
        try:
            # Create barrier config
            config = TacticianBarrierConfig(
                profit_take_multiplier=profit_take,
                stop_loss_multiplier=stop_loss,
                time_barrier_minutes=time_barrier
            )
            
            # Create labeler and apply labeling
            labeler = TacticianBarrierLabeler(config)
            result = labeler.apply_tactician_labeling(self.data, self.analyst_signals)
            
            if not result['success'] or result['labeled_data'] is None:
                return -np.inf
            
            # Calculate optimization metric
            labeled_data = result['labeled_data']
            return self._calculate_optimization_metric(labeled_data)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to evaluate config: {e}")
            return -np.inf
    
    def _calculate_optimization_metric(self, labeled_data: pd.DataFrame) -> float:
        """Calculate the optimization metric from labeled data."""
        try:
            if 'label' not in labeled_data.columns or 'net_profit_pct' not in labeled_data.columns:
                return -np.inf
            
            # Filter valid labels
            valid_data = labeled_data[labeled_data['label'] != 0].copy()
            if len(valid_data) == 0:
                return -np.inf
            
            # Calculate metrics
            total_trades = len(valid_data)
            winning_trades = len(valid_data[valid_data['label'] == 1])
            losing_trades = len(valid_data[valid_data['label'] == -1])
            
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # Calculate returns
            returns = valid_data['net_profit_pct'].values
            total_return = np.sum(returns)
            avg_return = np.mean(returns)
            return_std = np.std(returns)
            
            # Calculate optimization metric
            if self.optimization_metric == "directional_entry_quality":
                return self._calculate_directional_entry_quality(valid_data, returns)
            elif self.optimization_metric == "sharpe_ratio":
                if return_std == 0:
                    return 0
                return avg_return / return_std
            elif self.optimization_metric == "profit_factor":
                gross_profit = np.sum(returns[returns > 0])
                gross_loss = abs(np.sum(returns[returns < 0]))
                return gross_profit / gross_loss if gross_loss > 0 else gross_profit
            elif self.optimization_metric == "win_rate":
                return win_rate
            elif self.optimization_metric == "total_return":
                return total_return
            else:
                # Default to directional entry quality
                return self._calculate_directional_entry_quality(valid_data, returns)
                
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to calculate metric: {e}")
            return -np.inf
    
    def _calculate_directional_entry_quality(self, valid_data: pd.DataFrame, returns: np.ndarray) -> float:
        """
        Calculate directional entry quality for Tactician optimization.
        
        This metric focuses on finding the best entry point where price will move
        in the desired direction without going further in the opposite direction.
        
        Components:
        1. Directional Accuracy: How often the entry direction is correct
        2. Adverse Movement Minimization: How well we avoid adverse price movement
        3. Directional Profit Efficiency: Profit from correct directional moves
        4. Risk-Adjusted Directional Performance: Return per unit of directional risk
        """
        try:
            # Component 1: Directional Accuracy (40% weight)
            # How often the entry leads to price movement in the desired direction
            directional_accuracy = len(valid_data[valid_data['label'] == 1]) / len(valid_data)
            
            # Component 2: Adverse Movement Minimization (30% weight)
            # How well we avoid adverse price movement (minimize stop-loss hits)
            adverse_movements = len(valid_data[valid_data['label'] == -1])
            adverse_movement_ratio = adverse_movements / len(valid_data)
            adverse_movement_minimization = 1 - adverse_movement_ratio
            
            # Component 3: Directional Profit Efficiency (20% weight)
            # How much profit we capture from correct directional moves
            winning_returns = returns[returns > 0]
            if len(winning_returns) > 0:
                avg_winning_return = np.mean(winning_returns)
                max_possible_return = np.max(returns)  # Best possible return in dataset
                directional_profit_efficiency = avg_winning_return / max_possible_return if max_possible_return > 0 else 0
            else:
                directional_profit_efficiency = 0
            
            # Component 4: Risk-Adjusted Directional Performance (10% weight)
            # Return per unit of directional risk (volatility of returns)
            return_volatility = np.std(returns)
            if return_volatility > 0:
                risk_adjusted_directional_performance = np.mean(returns) / return_volatility
            else:
                risk_adjusted_directional_performance = 0
            
            # Composite directional entry quality score
            directional_entry_quality = (
                0.4 * directional_accuracy +
                0.3 * adverse_movement_minimization +
                0.2 * directional_profit_efficiency +
                0.1 * risk_adjusted_directional_performance
            )
            
            return directional_entry_quality
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to calculate directional entry quality: {e}")
            return -np.inf
    
    def _create_config_from_params(self, params: Dict[str, Any]) -> TacticianBarrierConfig:
        """Create TacticianBarrierConfig from optimization parameters."""
        return TacticianBarrierConfig(
            profit_take_multiplier=params['profit_take_multiplier'],
            stop_loss_multiplier=params['stop_loss_multiplier'],
            time_barrier_minutes=params['time_barrier_minutes']
        )


# Convenience functions
def optimize_tactician_barriers(
    data: pd.DataFrame,
    analyst_signals: np.ndarray,
    optimization_metric: str = "directional_entry_quality",
    method: str = "multi_stage",
    max_trials: int = 100,
    n_jobs: int = -1
) -> BarrierOptimizationResult:
    """
    Optimize Tactician barrier parameters.
    
    Args:
        data: OHLC data for optimization
        analyst_signals: Binary signals from Analyst
        optimization_metric: Metric to optimize
        method: Optimization method
        max_trials: Maximum trials for Optuna
        n_jobs: Number of parallel jobs
        
    Returns:
        BarrierOptimizationResult with best parameters
    """
    optimizer = BarrierOptimizer(
        data=data,
        analyst_signals=analyst_signals,
        optimization_metric=optimization_metric,
        n_jobs=n_jobs
    )
    
    return optimizer.optimize_barriers(method=method, max_trials=max_trials)


def create_optimized_tactician_barrier_labeler(
    data: pd.DataFrame,
    analyst_signals: np.ndarray,
    optimization_metric: str = "directional_entry_quality",
    method: str = "multi_stage",
    max_trials: int = 100
) -> TacticianBarrierLabeler:
    """
    Create an optimized Tactician barrier labeler.
    
    Args:
        data: OHLC data for optimization
        analyst_signals: Binary signals from Analyst
        optimization_metric: Metric to optimize
        method: Optimization method
        max_trials: Maximum trials for Optuna
        
    Returns:
        Optimized TacticianBarrierLabeler
    """
    # Optimize barriers
    optimization_result = optimize_tactician_barriers(
        data=data,
        analyst_signals=analyst_signals,
        optimization_metric=optimization_metric,
        method=method,
        max_trials=max_trials
    )
    
    # Create labeler with optimized config
    return TacticianBarrierLabeler(optimization_result.best_config)


if __name__ == '__main__':
    # Test the barrier optimization
    print("🎯 Testing Barrier Optimization")
    
    # Create test data
    dates = pd.date_range('2024-01-01', periods=1000, freq='1min')
    data = pd.DataFrame({
        'open': np.random.uniform(100, 110, 1000),
        'high': np.random.uniform(105, 115, 1000),
        'low': np.random.uniform(95, 105, 1000),
        'close': np.random.uniform(100, 110, 1000),
        'volume': np.random.uniform(1000, 10000, 1000)
    }, index=dates)
    
    # Create analyst signals
    analyst_signals = np.random.choice([0, 1], 1000, p=[0.8, 0.2])
    
    # Test optimization
    print("\n📊 Testing multi-stage optimization with directional entry quality...")
    result = optimize_tactician_barriers(
        data=data,
        analyst_signals=analyst_signals,
        optimization_metric="directional_entry_quality",
        method="multi_stage",
        max_trials=50
    )
    
    print(f"✅ Optimization completed:")
    print(f"   Best profit take: {result.best_profit_take:.4f}")
    print(f"   Best stop loss: {result.best_stop_loss:.4f}")
    print(f"   Best time barrier: {result.best_time_barrier} minutes")
    print(f"   Best score: {result.best_score:.4f}")
    print(f"   Method: {result.optimization_method}")
    print(f"   Time: {result.optimization_time:.2f}s")
    
    print('✅ Barrier Optimization test completed!')