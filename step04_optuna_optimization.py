#!/usr/bin/env python3
"""
Step04 Optuna Optimization for Triple Barrier Parameters

This module provides comprehensive Optuna-based optimization for triple barrier method
parameters, addressing the fixed parameter issue identified in the review.

Features:
- Regime-specific parameter optimization
- Walk-forward validation integration
- Transaction cost consideration
- Risk-adjusted performance metrics
- Multi-objective optimization
"""

import pandas as pd
import numpy as np
import optuna
from typing import Dict, List, Optional, Tuple, Any, Callable
import logging
from datetime import datetime, timedelta
import json
from pathlib import Path

# Import our corrected triple barrier method
from step04_lookahead_bias_fix import CorrectedTripleBarrierMethod
import time

class TripleBarrierOptunaOptimizer:
    """
    Optuna-based optimizer for triple barrier method parameters.
    
    This optimizer addresses the fixed parameter issue by finding optimal
    profit/loss multipliers and time barriers for different market conditions.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Optimization configuration
        self.n_trials = config.get('n_trials', 100)
        self.timeout = config.get('timeout', 3600)  # 1 hour
        self.n_jobs = config.get('n_jobs', 1)
        self.study_name = config.get('study_name', 'triple_barrier_optimization')
        
        # Performance metrics weights
        self.metric_weights = {
            'sharpe_ratio': config.get('sharpe_weight', 0.4),
            'win_rate': config.get('win_rate_weight', 0.2),
            'profit_factor': config.get('profit_factor_weight', 0.2),
            'max_drawdown': config.get('max_drawdown_weight', 0.2)
        }
        
        # Parameter bounds
        self.param_bounds = {
            'profit_take_multiplier': (0.005, 0.05),  # 0.5% to 5%
            'stop_loss_multiplier': (0.005, 0.05),    # 0.5% to 5%
            'time_barrier_minutes': (5, 120),         # 5 minutes to 2 hours
            'max_lookahead': (10, 200)                # 10 to 200 periods
        }
        
        # Transaction costs
        self.transaction_cost_bps = config.get('transaction_cost_bps', 5)
        self.slippage_bps = config.get('slippage_bps', 2)
        
        self.logger.info("✅ Triple Barrier Optuna Optimizer initialized")
        self.logger.info(f"   Trials: {self.n_trials}")
        self.logger.info(f"   Timeout: {self.timeout}s")
        self.logger.info(f"   Transaction cost: {self.transaction_cost_bps} bps")
    
    def optimize_parameters(
        self,
        data: pd.DataFrame,
        regime_data: Optional[pd.DataFrame] = None,
        optimization_mode: str = 'global'
    ) -> Dict[str, Any]:
        """
        Optimize triple barrier parameters using Optuna.
        
        Args:
            data: Market data for optimization
            regime_data: Optional regime labels for regime-specific optimization
            optimization_mode: 'global', 'regime_specific', or 'walk_forward'
            
        Returns:
            Optimization results with best parameters
        """
        self.logger.info(f"🚀 Starting Optuna optimization: {optimization_mode}")
        self.logger.info(f"   Data shape: {data.shape}")
        self.logger.info(f"   Regime data available: {regime_data is not None}")
        
        if optimization_mode == 'global':
            return self._optimize_global_parameters(data)
        elif optimization_mode == 'regime_specific':
            if regime_data is None:
                raise ValueError("Regime data required for regime-specific optimization")
            return self._optimize_regime_specific_parameters(data, regime_data)
        elif optimization_mode == 'walk_forward':
            return self._optimize_walk_forward_parameters(data)
        else:
            raise ValueError(f"Unknown optimization mode: {optimization_mode}")
    
    def _optimize_global_parameters(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Optimize global parameters for all market conditions."""
        self.logger.info("🌍 Optimizing global parameters")
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(
                n_startup_trials=20,
                n_ei_candidates=24,
                multivariate=True
            ),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=10,
                n_warmup_steps=5,
                interval_steps=3
            )
        )
        
        # Define objective function
        def objective(trial):
            return self._evaluate_parameters(trial, data, None)
        
        # Optimize
        study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            n_jobs=self.n_jobs,
            callbacks=[optuna.callbacks.EarlyStoppingCallback(patience=20)]
        )
        
        return self._format_optimization_results(study, 'global')
    
    def _optimize_regime_specific_parameters(
        self, 
        data: pd.DataFrame, 
        regime_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Optimize parameters for each regime separately."""
        self.logger.info("🎯 Optimizing regime-specific parameters")
        
        # Merge data with regime information
        merged_data = pd.merge(data, regime_data[['timestamp', 'composite_cluster_id']], 
                              on='timestamp', how='inner')
        
        # Get unique regimes
        regimes = merged_data['composite_cluster_id'].unique()
        self.logger.info(f"   Found regimes: {regimes}")
        
        regime_results = {}
        
        for regime_id in regimes:
            self.logger.info(f"🔧 Optimizing regime {regime_id}")
            
            # Filter data for this regime
            regime_data_filtered = merged_data[merged_data['composite_cluster_id'] == regime_id]
            
            if len(regime_data_filtered) < 100:
                self.logger.warning(f"   Regime {regime_id} has insufficient data: {len(regime_data_filtered)} rows")
                continue
            
            # Create study for this regime
            study = optuna.create_study(
                direction='maximize',
                study_name=f"{self.study_name}_regime_{regime_id}",
                sampler=optuna.samplers.TPESampler(
                    n_startup_trials=15,
                    n_ei_candidates=20
                )
            )
            
            # Define objective function for this regime
            def objective(trial):
                return self._evaluate_parameters(trial, regime_data_filtered, regime_id)
            
            # Optimize
            study.optimize(
                objective,
                n_trials=max(50, self.n_trials // len(regimes)),
                timeout=self.timeout // len(regimes),
                callbacks=[optuna.callbacks.EarlyStoppingCallback(patience=15)]
            )
            
            regime_results[f'regime_{regime_id}'] = self._format_optimization_results(study, f'regime_{regime_id}')
        
        return {
            'optimization_mode': 'regime_specific',
            'regime_results': regime_results,
            'summary': self._create_regime_summary(regime_results)
        }
    
    def _optimize_walk_forward_parameters(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Optimize parameters using walk-forward validation."""
        self.logger.info("🔄 Optimizing with walk-forward validation")
        
        # Split data into training and validation
        split_ratio = 0.7
        split_idx = int(len(data) * split_ratio)
        train_data = data.iloc[:split_idx]
        val_data = data.iloc[split_idx:]
        
        self.logger.info(f"   Training data: {len(train_data)} rows")
        self.logger.info(f"   Validation data: {len(val_data)} rows")
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            study_name=f"{self.study_name}_walk_forward",
            sampler=optuna.samplers.TPESampler(
                n_startup_trials=20,
                n_ei_candidates=24
            )
        )
        
        # Define objective function with walk-forward validation
        def objective(trial):
            return self._evaluate_parameters_walk_forward(trial, train_data, val_data)
        
        # Optimize
        study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            callbacks=[optuna.callbacks.EarlyStoppingCallback(patience=20)]
        )
        
        return self._format_optimization_results(study, 'walk_forward')
    
    def _evaluate_parameters(
        self, 
        trial: optuna.Trial, 
        data: pd.DataFrame, 
        regime_id: Optional[int]
    ) -> float:
        """Evaluate parameter set and return performance score."""
        
        # Sample parameters
        params = {
            'profit_take_multiplier': trial.suggest_float(
                'profit_take_multiplier',
                *self.param_bounds['profit_take_multiplier'],
                log=True
            ),
            'stop_loss_multiplier': trial.suggest_float(
                'stop_loss_multiplier',
                *self.param_bounds['stop_loss_multiplier'],
                log=True
            ),
            'time_barrier_minutes': trial.suggest_int(
                'time_barrier_minutes',
                *self.param_bounds['time_barrier_minutes']
            ),
            'max_lookahead': trial.suggest_int(
                'max_lookahead',
                *self.param_bounds['max_lookahead']
            ),
            'transaction_cost_bps': self.transaction_cost_bps,
            'slippage_bps': self.slippage_bps
        }
        
        # Validate parameter constraints
        if not self._validate_parameters(params):
            return float('-inf')
        
        try:
            # Apply triple barrier method
            triple_barrier = CorrectedTripleBarrierMethod(params)
            labeled_data = triple_barrier.apply_corrected_triple_barrier(
                data, walk_forward=False
            )
            
            # Calculate performance metrics
            metrics = self._calculate_performance_metrics(labeled_data)
            
            # Calculate composite score
            score = self._calculate_composite_score(metrics)
            
            # Log trial results
            trial.set_user_attr('sharpe_ratio', metrics['sharpe_ratio'])
            trial.set_user_attr('win_rate', metrics['win_rate'])
            trial.set_user_attr('profit_factor', metrics['profit_factor'])
            trial.set_user_attr('max_drawdown', metrics['max_drawdown'])
            trial.set_user_attr('total_signals', metrics['total_signals'])
            
            return score
            
        except Exception as e:
            self.logger.warning(f"Trial failed: {e}")
            return float('-inf')
    
    def _evaluate_parameters_walk_forward(
        self, 
        trial: optuna.Trial, 
        train_data: pd.DataFrame, 
        val_data: pd.DataFrame
    ) -> float:
        """Evaluate parameters using walk-forward validation."""
        
        # Sample parameters
        params = {
            'profit_take_multiplier': trial.suggest_float(
                'profit_take_multiplier',
                *self.param_bounds['profit_take_multiplier'],
                log=True
            ),
            'stop_loss_multiplier': trial.suggest_float(
                'stop_loss_multiplier',
                *self.param_bounds['stop_loss_multiplier'],
                log=True
            ),
            'time_barrier_minutes': trial.suggest_int(
                'time_barrier_minutes',
                *self.param_bounds['time_barrier_minutes']
            ),
            'max_lookahead': trial.suggest_int(
                'max_lookahead',
                *self.param_bounds['max_lookahead']
            ),
            'transaction_cost_bps': self.transaction_cost_bps,
            'slippage_bps': self.slippage_bps
        }
        
        try:
            # Train on training data
            triple_barrier = CorrectedTripleBarrierMethod(params)
            train_labeled = triple_barrier.apply_corrected_triple_barrier(
                train_data, walk_forward=False
            )
            
            # Validate on validation data
            val_labeled = triple_barrier.apply_corrected_triple_barrier(
                val_data, walk_forward=False
            )
            
            # Calculate performance on validation set
            val_metrics = self._calculate_performance_metrics(val_labeled)
            
            # Calculate composite score
            score = self._calculate_composite_score(val_metrics)
            
            # Log trial results
            trial.set_user_attr('val_sharpe_ratio', val_metrics['sharpe_ratio'])
            trial.set_user_attr('val_win_rate', val_metrics['win_rate'])
            trial.set_user_attr('val_profit_factor', val_metrics['profit_factor'])
            trial.set_user_attr('val_max_drawdown', val_metrics['max_drawdown'])
            
            return score
            
        except Exception as e:
            self.logger.warning(f"Walk-forward trial failed: {e}")
            return float('-inf')
    
    def _validate_parameters(self, params: Dict[str, Any]) -> bool:
        """Validate parameter constraints."""
        
        # Profit take should be greater than stop loss for reasonable risk-reward
        if params['profit_take_multiplier'] <= params['stop_loss_multiplier']:
            return False
        
        # Risk-reward ratio should be reasonable (between 1:1 and 3:1)
        risk_reward_ratio = params['profit_take_multiplier'] / params['stop_loss_multiplier']
        if risk_reward_ratio < 1.0 or risk_reward_ratio > 3.0:
            return False
        
        # Time barrier should be reasonable
        if params['time_barrier_minutes'] < 5 or params['time_barrier_minutes'] > 120:
            return False
        
        return True
    
    def _calculate_performance_metrics(self, labeled_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        
        # Filter to only labeled data
        signals = labeled_data[labeled_data['label'] != 0].copy()
        
        if len(signals) == 0:
            return {
                'sharpe_ratio': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'max_drawdown': 1.0,
                'total_signals': 0,
                'avg_profit': 0.0,
                'total_return': 0.0
            }
        
        # Calculate returns
        returns = signals['net_profit_pct'].values
        
        # Sharpe ratio
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
        else:
            sharpe_ratio = 0.0
        
        # Win rate
        win_rate = (signals['label'] == 1).mean()
        
        # Profit factor
        profits = returns[returns > 0]
        losses = returns[returns < 0]
        
        if len(losses) > 0 and abs(np.sum(losses)) > 0:
            profit_factor = np.sum(profits) / abs(np.sum(losses))
        else:
            profit_factor = float('inf') if len(profits) > 0 else 0.0
        
        # Maximum drawdown
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = abs(np.min(drawdowns))
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'total_signals': len(signals),
            'avg_profit': np.mean(returns),
            'total_return': np.sum(returns)
        }
    
    def _calculate_composite_score(self, metrics: Dict[str, float]) -> float:
        """Calculate weighted composite score."""
        
        # Normalize metrics to 0-1 scale
        sharpe_norm = min(max(metrics['sharpe_ratio'] / 2.0, 0), 1)  # Cap at 2.0
        win_rate_norm = metrics['win_rate']
        profit_factor_norm = min(max(metrics['profit_factor'] / 2.0, 0), 1)  # Cap at 2.0
        drawdown_norm = 1 - min(max(metrics['max_drawdown'], 0), 1)  # Invert (lower is better)
        
        # Calculate weighted score
        score = (
            self.metric_weights['sharpe_ratio'] * sharpe_norm +
            self.metric_weights['win_rate'] * win_rate_norm +
            self.metric_weights['profit_factor'] * profit_factor_norm +
            self.metric_weights['max_drawdown'] * drawdown_norm
        )
        
        return score
    
    def _format_optimization_results(
        self, 
        study: optuna.Study, 
        mode: str
    ) -> Dict[str, Any]:
        """Format optimization results."""
        
        if len(study.trials) == 0:
            return {'error': 'No trials completed'}
        
        best_trial = study.best_trial
        
        return {
            'optimization_mode': mode,
            'best_score': best_trial.value,
            'best_params': best_trial.params,
            'best_trial_number': best_trial.number,
            'total_trials': len(study.trials),
            'optimization_time': datetime.now().isoformat(),
            'trial_history': [trial.value for trial in study.trials if trial.value is not None],
            'user_attrs': best_trial.user_attrs,
            'study_name': study.study_name
        }
    
    def _create_regime_summary(self, regime_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary of regime-specific optimization results."""
        
        summary = {
            'total_regimes': len(regime_results),
            'successful_optimizations': 0,
            'failed_optimizations': 0,
            'best_overall_score': 0.0,
            'best_regime': None,
            'regime_scores': {}
        }
        
        for regime_name, result in regime_results.items():
            if 'error' not in result:
                summary['successful_optimizations'] += 1
                score = result['best_score']
                summary['regime_scores'][regime_name] = score
                
                if score > summary['best_overall_score']:
                    summary['best_overall_score'] = score
                    summary['best_regime'] = regime_name
            else:
                summary['failed_optimizations'] += 1
        
        return summary
    
    def save_optimization_results(
        self, 
        results: Dict[str, Any], 
        filepath: str
    ) -> None:
        """Save optimization results to file."""
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Recursively convert all numpy types
        def recursive_convert(obj):
            if isinstance(obj, dict):
                return {k: recursive_convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [recursive_convert(item) for item in obj]
            else:
                return convert_numpy(obj)
        
        converted_results = recursive_convert(results)
        
        with open(filepath, 'w') as f:
            json.dump(converted_results, f, indent=2)
        
        self.logger.info(f"✅ Optimization results saved to {filepath}")


# Example usage and testing
def test_optuna_optimization():
    """Test the Optuna optimization for triple barrier parameters."""
    
    # Create sample data
    timestamps = pd.date_range('2024-01-01', periods=2000, freq='1min')
    data = pd.DataFrame({
        'timestamp': timestamps,
        'open': np.random.randn(2000).cumsum() + 100,
        'high': np.random.randn(2000).cumsum() + 102,
        'low': np.random.randn(2000).cumsum() + 98,
        'close': np.random.randn(2000).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 2000)
    })
    
    # Create sample regime data
    regime_data = pd.DataFrame({
        'timestamp': timestamps[::10],  # Every 10th timestamp
        'composite_cluster_id': np.random.randint(0, 3, len(timestamps[::10]))
    })
    
    # Test configuration
    config = {
        'n_trials': 50,  # Reduced for testing
        'timeout': 300,  # 5 minutes for testing
        'n_jobs': 1,
        'study_name': 'test_triple_barrier_optimization',
        'sharpe_weight': 0.4,
        'win_rate_weight': 0.2,
        'profit_factor_weight': 0.2,
        'max_drawdown_weight': 0.2,
        'transaction_cost_bps': 5,
        'slippage_bps': 2
    }
    
    # Initialize optimizer
    optimizer = TripleBarrierOptunaOptimizer(config)
    
    # Test global optimization
    print("=== Testing Global Optimization ===")
    global_results = optimizer.optimize_parameters(data, optimization_mode='global')
    print(f"Best score: {global_results['best_score']:.4f}")
    print(f"Best parameters: {global_results['best_params']}")
    
    # Test regime-specific optimization
    print("\n=== Testing Regime-Specific Optimization ===")
    regime_results = optimizer.optimize_parameters(
        data, regime_data, optimization_mode='regime_specific'
    )
    print(f"Regime summary: {regime_results['summary']}")
    
    # Test walk-forward optimization
    print("\n=== Testing Walk-Forward Optimization ===")
    walk_forward_results = optimizer.optimize_parameters(data, optimization_mode='walk_forward')
    print(f"Best score: {walk_forward_results['best_score']:.4f}")
    print(f"Best parameters: {walk_forward_results['best_params']}")
    
    # Save results
    all_results = {
        'global': global_results,
        'regime_specific': regime_results,
        'walk_forward': walk_forward_results
    }
    
    optimizer.save_optimization_results(all_results, 'triple_barrier_optimization_results.json')
    
    return all_results


if __name__ == "__main__":
    test_optuna_optimization()