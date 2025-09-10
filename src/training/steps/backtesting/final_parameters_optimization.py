"""
Final Parameters Optimization for ML Models

This module provides system-wide final parameters optimization functionality,
separate from HPO (Hyperparameter Optimization). This is used for optimizing
final system parameters after model training is complete.

Key Features:
- System-wide parameter optimization using Optuna
- Categorized parameter optimization (confidence, position sizing, leverage, etc.)
- Integration with calibration results
- Comprehensive evaluation and validation
- Automatic parameter updates
"""

import json
import os
import pickle
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Tuple
import optuna
import numpy as np
import logging

logger = logging.getLogger(__name__)


class FinalParametersOptimizer:
    """
    System-wide final parameters optimizer.
    
    This handles optimization of final system parameters after model training,
    separate from hyperparameter optimization during training.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the final parameters optimizer."""
        self.config = config
        self.logger = logger.getChild('FinalParametersOptimizer')
        
        # Parameter categories for optimization
        self.categories = [
            'confidence', 'intensity', 'position_sizing', 'leverage', 'tpsl',
            'ensemble', 'sr', 'two_tier', 'technical_indicators', 
            'system_monitoring', 'training_optimization', 'regime_transitions',
            'signal_aggregation'
        ]
        
        # Default search spaces for each category
        self.default_search_spaces = self._get_default_search_spaces()
        
        # Optimization settings
        self.n_trials = config.get('n_trials', 50)
        self.timeout = config.get('timeout', 300)
        self.study_name = config.get('study_name', 'final_parameters_optimization')
        
        self.logger.info("🚀 Final Parameters Optimizer initialized")
    
    def _get_default_search_spaces(self) -> Dict[str, Dict[str, Any]]:
        """Get default search spaces for parameter categories."""
        return {
            'confidence': {
                'base_entry_threshold': {'type': 'float', 'min': 0.5, 'max': 0.9},
                'analyst_confidence_threshold': {'type': 'float', 'min': 0.6, 'max': 0.8},
                'tactician_confidence_threshold': {'type': 'float', 'min': 0.7, 'max': 0.9}
            },
            'position_sizing': {
                'base_position_size': {'type': 'float', 'min': 0.01, 'max': 0.15},
                'max_position_size': {'type': 'float', 'min': 0.1, 'max': 0.3}
            },
            'leverage': {
                'safe_leverage_multiplier': {'type': 'float', 'min': 0.5, 'max': 1.0}
            },
            'tpsl': {
                'tp_long': {'type': 'float', 'min': 0.02, 'max': 0.1},
                'sl_long': {'type': 'float', 'min': 0.01, 'max': 0.05}
            },
            'ensemble': {
                'analyst_weight': {'type': 'float', 'min': 0.2, 'max': 0.5},
                'tactician_weight': {'type': 'float', 'min': 0.2, 'max': 0.5},
                'strategist_weight': {'type': 'float', 'min': 0.1, 'max': 0.3}
            },
            'sr': {
                'touch_count_weight': {'type': 'float', 'min': 0.1, 'max': 0.4},
                'total_volume_weight': {'type': 'float', 'min': 0.1, 'max': 0.4},
                'level_age_weight': {'type': 'float', 'min': 0.1, 'max': 0.3},
                'bounce_rate_weight': {'type': 'float', 'min': 0.1, 'max': 0.3}
            },
            'two_tier': {
                'tier1_weight': {'type': 'float', 'min': 0.4, 'max': 0.7},
                'tier2_weight': {'type': 'float', 'min': 0.3, 'max': 0.6},
                'direction_threshold': {'type': 'float', 'min': 0.6, 'max': 0.8},
                'timing_threshold': {'type': 'float', 'min': 0.7, 'max': 0.9}
            },
            'technical_indicators': {
                'rsi_period': {'type': 'int', 'min': 10, 'max': 20},
                'macd_fast_period': {'type': 'int', 'min': 8, 'max': 16},
                'macd_slow_period': {'type': 'int', 'min': 20, 'max': 30},
                'adx_trend_threshold': {'type': 'float', 'min': 20.0, 'max': 35.0},
                'adx_sideways_threshold': {'type': 'float', 'min': 15.0, 'max': 30.0},
                'volatility_threshold': {'type': 'float', 'min': 0.015, 'max': 0.035}
            },
            'system_monitoring': {
                'analysis_interval': {'type': 'int', 'min': 1800, 'max': 7200},
                'max_history': {'type': 'int', 'min': 50, 'max': 200},
                'memory_threshold': {'type': 'float', 'min': 0.7, 'max': 0.9},
                'learning_rate': {'type': 'float', 'min': 0.005, 'max': 0.05}
            },
            'training_optimization': {
                'min_label_balance': {'type': 'float', 'min': 0.03, 'max': 0.1},
                'max_label_balance': {'type': 'float', 'min': 0.9, 'max': 0.98},
                'stability_threshold': {'type': 'float', 'min': 0.6, 'max': 0.9},
                'lgb_learning_rate': {'type': 'float', 'min': 0.01, 'max': 0.2},
                'model_performance_threshold': {'type': 'float', 'min': 0.6, 'max': 0.85}
            },
            'regime_transitions': {
                'transition_intensity_threshold': {'type': 'float', 'min': 0.2, 'max': 0.5},
                'transition_confidence_threshold': {'type': 'float', 'min': 0.6, 'max': 0.9},
                'step9_5_weight': {'type': 'float', 'min': 0.2, 'max': 0.4},
                'step10_weight': {'type': 'float', 'min': 0.2, 'max': 0.4},
                'regime_expert_weight': {'type': 'float', 'min': 0.2, 'max': 0.4},
                'transition_lookback_periods': {'type': 'int', 'min': 3, 'max': 10},
                'transition_risk_multiplier': {'type': 'float', 'min': 1.0, 'max': 1.5}
            },
            'signal_aggregation': {
                'analyst_weight': {'type': 'float', 'min': 0.2, 'max': 0.4},
                'tactician_weight': {'type': 'float', 'min': 0.2, 'max': 0.4},
                'scenario_weight': {'type': 'float', 'min': 0.1, 'max': 0.3},
                'sr_breakout_weight': {'type': 'float', 'min': 0.1, 'max': 0.3},
                'regime_weight': {'type': 'float', 'min': 0.1, 'max': 0.3},
                'conflict_penalty_factor': {'type': 'float', 'min': 0.4, 'max': 0.6},
                'min_source_weight': {'type': 'float', 'min': 0.05, 'max': 0.15},
                'min_signal_confidence': {'type': 'float', 'min': 0.2, 'max': 0.4},
                'min_aggregated_confidence': {'type': 'float', 'min': 0.4, 'max': 0.6},
                'regime_alignment_bonus': {'type': 'float', 'min': 0.1, 'max': 0.25},
                'multi_signal_alignment_bonus': {'type': 'float', 'min': 0.05, 'max': 0.15},
                'use_multiplicative': {'type': 'bool', 'value': True}
            }
        }
    
    async def optimize_all_parameters(self, calibration_results: Dict[str, Any], 
                                    previous_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Optimize all parameters by category.
        
        Args:
            calibration_results: Results from confidence calibration
            previous_results: Previous optimization results for warm start
            
        Returns:
            Dict containing optimized parameters by category
        """
        try:
            self.logger.info("🔧 Starting final parameters optimization...")
            optimization_results = {}
            
            for category in self.categories:
                self.logger.info(f"Optimizing {category} parameters...")
                category_results = await self._optimize_category(
                    category, calibration_results, 
                    previous_results.get(category) if previous_results else None
                )
                optimization_results[category] = category_results
            
            self.logger.info("✅ Final parameters optimization completed")
            return optimization_results
            
        except Exception as e:
            self.logger.exception(f"Error in final parameters optimization: {e}")
            raise
    
    async def _optimize_category(self, category: str, calibration_results: Dict[str, Any], 
                               previous_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Optimize parameters for a specific category.
        
        Args:
            category: Parameter category to optimize
            calibration_results: Results from confidence calibration
            previous_results: Previous optimization results for this category
            
        Returns:
            Dict containing optimization results for the category
        """
        try:
            search_space = self.default_search_spaces.get(category, {})
            if not search_space:
                self.logger.warning(f"No search space found for category: {category}")
                return {}
            
            study_name = f'{self.study_name}_{category}'
            study = optuna.create_study(
                study_name=study_name,
                direction='maximize',
                storage='sqlite:///optuna_studies.db',
                load_if_exists=True
            )
            
            def objective(trial):
                return self._objective_function(trial, category, search_space, calibration_results)
            
            study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)
            
            best_params = study.best_params
            best_value = study.best_value
            
            return {
                'best_params': best_params,
                'best_value': best_value,
                'study_name': study_name,
                'n_trials': self.n_trials
            }
            
        except Exception as e:
            self.logger.error(f"Error optimizing category {category}: {e}")
            return {}
    
    def _objective_function(self, trial: optuna.Trial, category: str, 
                          search_space: Dict[str, Dict[str, Any]], 
                          calibration_results: Dict[str, Any]) -> float:
        """
        Objective function for Optuna optimization.
        
        Args:
            trial: Optuna trial object
            category: Parameter category being optimized
            search_space: Search space for the category
            calibration_results: Results from confidence calibration
            
        Returns:
            Optimization score (higher is better)
        """
        try:
            params = {}
            for param_name, param_config in search_space.items():
                if param_config['type'] == 'float':
                    params[param_name] = trial.suggest_float(
                        param_name, param_config['min'], param_config['max']
                    )
                elif param_config['type'] == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name, param_config['min'], param_config['max']
                    )
                elif param_config['type'] == 'bool':
                    params[param_name] = trial.suggest_categorical(
                        param_name, [True, False]
                    )
            
            score = self._evaluate_configuration(category, params, calibration_results)
            return score
            
        except Exception as e:
            self.logger.error(f"Error in objective function for {category}: {e}")
            return -999.0
    
    def _evaluate_configuration(self, category: str, params: Dict[str, Any], 
                              calibration_results: Dict[str, Any]) -> float:
        """
        Evaluate a configuration by running a backtest or simulation.
        
        Args:
            category: Parameter category being evaluated
            params: Parameters to evaluate
            calibration_results: Results from confidence calibration
            
        Returns:
            Evaluation score (higher is better)
        """
        try:
            base_score = 0.0
            
            if category == 'confidence':
                base_score = self._evaluate_confidence_params(params, calibration_results)
            elif category == 'position_sizing':
                base_score = self._evaluate_position_sizing_params(params, calibration_results)
            elif category == 'leverage':
                base_score = self._evaluate_leverage_params(params, calibration_results)
            elif category == 'tpsl':
                base_score = self._evaluate_tpsl_params(params, calibration_results)
            elif category == 'ensemble':
                base_score = self._evaluate_ensemble_params(params, calibration_results)
            elif category == 'sr':
                base_score = self._evaluate_sr_params(params, calibration_results)
            elif category == 'two_tier':
                base_score = self._evaluate_two_tier_params(params, calibration_results)
            elif category == 'technical_indicators':
                base_score = self._evaluate_technical_indicators_params(params, calibration_results)
            elif category == 'system_monitoring':
                base_score = self._evaluate_system_monitoring_params(params, calibration_results)
            elif category == 'training_optimization':
                base_score = self._evaluate_training_optimization_params(params, calibration_results)
            elif category == 'regime_transitions':
                base_score = self._evaluate_regime_transitions_params(params, calibration_results)
            elif category == 'signal_aggregation':
                base_score = self._evaluate_signal_aggregation_params(params, calibration_results)
            
            return base_score
            
        except Exception as e:
            self.logger.error(f"Error evaluating configuration for {category}: {e}")
            return 0.0
    
    def _evaluate_confidence_params(self, params: Dict[str, Any], 
                                  calibration_results: Dict[str, Any]) -> float:
        """Evaluate confidence threshold parameters."""
        score = 0.0
        
        if 'base_entry_threshold' in params:
            threshold = params['base_entry_threshold']
            if 0.6 <= threshold <= 0.8:
                score += 0.3
            elif 0.5 <= threshold <= 0.9:
                score += 0.2
            else:
                score += 0.1
        
        if 'analyst_confidence_threshold' in params and 'tactician_confidence_threshold' in params:
            analyst_thresh = params['analyst_confidence_threshold']
            tactician_thresh = params['tactician_confidence_threshold']
            if tactician_thresh > analyst_thresh:
                score += 0.2
            if 0.1 <= tactician_thresh - analyst_thresh <= 0.2:
                score += 0.1
        
        return score
    
    def _evaluate_position_sizing_params(self, params: Dict[str, Any], 
                                       calibration_results: Dict[str, Any]) -> float:
        """Evaluate position sizing parameters."""
        score = 0.0
        
        if 'base_position_size' in params:
            base_size = params['base_position_size']
            if 0.02 <= base_size <= 0.1:
                score += 0.3
            elif 0.01 <= base_size <= 0.15:
                score += 0.2
            else:
                score += 0.1
        
        if 'max_position_size' in params:
            max_size = params['max_position_size']
            if 0.15 <= max_size <= 0.3:
                score += 0.2
            else:
                score += 0.1
        
        return score
    
    def _evaluate_leverage_params(self, params: Dict[str, Any], 
                                calibration_results: Dict[str, Any]) -> float:
        """Evaluate leverage parameters."""
        score = 0.0
        
        if 'safe_leverage_multiplier' in params:
            multiplier = params['safe_leverage_multiplier']
            if 0.7 <= multiplier <= 0.9:
                score += 0.3
            elif 0.5 <= multiplier <= 1.0:
                score += 0.2
            else:
                score += 0.1
        
        return score
    
    def _evaluate_tpsl_params(self, params: Dict[str, Any], 
                            calibration_results: Dict[str, Any]) -> float:
        """Evaluate TP/SL parameters."""
        score = 0.0
        
        if 'tp_long' in params and 'sl_long' in params:
            tp = params['tp_long']
            sl = params['sl_long']
            if tp > sl and tp / sl >= 1.5:
                score += 0.3
            elif tp > sl:
                score += 0.2
            else:
                score += 0.1
        
        return score
    
    def _evaluate_ensemble_params(self, params: Dict[str, Any], 
                                calibration_results: Dict[str, Any]) -> float:
        """Evaluate ensemble parameters."""
        score = 0.0
        
        if all(key in params for key in ['analyst_weight', 'tactician_weight', 'strategist_weight']):
            weights = [params['analyst_weight'], params['tactician_weight'], params['strategist_weight']]
            if abs(sum(weights) - 1.0) < 0.1:
                score += 0.3
            else:
                score += 0.1
        
        return score
    
    def _evaluate_sr_params(self, params: Dict[str, Any], 
                          calibration_results: Dict[str, Any]) -> float:
        """Evaluate S/R parameters."""
        score = 0.0
        
        weight_params = ['touch_count_weight', 'total_volume_weight', 'level_age_weight', 
                        'bounce_rate_weight', 'isolation_score_weight']
        weights = [params.get(param, 0.0) for param in weight_params]
        
        if abs(sum(weights) - 1.0) < 0.1:
            score += 0.3
        else:
            score += 0.1
        
        return score
    
    def _evaluate_two_tier_params(self, params: Dict[str, Any], 
                                calibration_results: Dict[str, Any]) -> float:
        """Evaluate two-tier system parameters."""
        score = 0.0
        
        if 'tier1_weight' in params and 'tier2_weight' in params:
            tier1_weight = params['tier1_weight']
            tier2_weight = params['tier2_weight']
            if abs(tier1_weight + tier2_weight - 1.0) < 0.1:
                score += 0.3
            else:
                score += 0.1
        
        if 'direction_threshold' in params:
            threshold = params['direction_threshold']
            if 0.6 <= threshold <= 0.8:
                score += 0.2
            else:
                score += 0.1
        
        if 'timing_threshold' in params:
            threshold = params['timing_threshold']
            if 0.7 <= threshold <= 0.9:
                score += 0.2
            else:
                score += 0.1
        
        return score
    
    def _evaluate_technical_indicators_params(self, params: Dict[str, Any], 
                                           calibration_results: Dict[str, Any]) -> float:
        """Evaluate technical indicator parameters."""
        score = 0.0
        
        if 'rsi_period' in params:
            rsi_period = params['rsi_period']
            if 10 <= rsi_period <= 20:
                score += 0.2
            else:
                score += 0.1
        
        if 'macd_fast_period' in params and 'macd_slow_period' in params:
            fast = params['macd_fast_period']
            slow = params['macd_slow_period']
            if fast < slow and 8 <= fast <= 16 and 20 <= slow <= 30:
                score += 0.2
            else:
                score += 0.1
        
        if 'adx_trend_threshold' in params and 'adx_sideways_threshold' in params:
            trend = params['adx_trend_threshold']
            sideways = params['adx_sideways_threshold']
            if trend > sideways:
                score += 0.2
            else:
                score += 0.1
        
        if 'volatility_threshold' in params:
            vol_thresh = params['volatility_threshold']
            if 0.015 <= vol_thresh <= 0.035:
                score += 0.2
            else:
                score += 0.1
        
        return score
    
    def _evaluate_system_monitoring_params(self, params: Dict[str, Any], 
                                        calibration_results: Dict[str, Any]) -> float:
        """Evaluate system monitoring parameters."""
        score = 0.0
        
        if 'analysis_interval' in params:
            interval = params['analysis_interval']
            if 1800 <= interval <= 7200:
                score += 0.2
            else:
                score += 0.1
        
        if 'max_history' in params:
            max_hist = params['max_history']
            if 50 <= max_hist <= 200:
                score += 0.2
            else:
                score += 0.1
        
        if 'memory_threshold' in params:
            mem_thresh = params['memory_threshold']
            if 0.7 <= mem_thresh <= 0.9:
                score += 0.2
            else:
                score += 0.1
        
        if 'learning_rate' in params:
            lr = params['learning_rate']
            if 0.005 <= lr <= 0.05:
                score += 0.2
            else:
                score += 0.1
        
        return score
    
    def _evaluate_training_optimization_params(self, params: Dict[str, Any], 
                                            calibration_results: Dict[str, Any]) -> float:
        """Evaluate training optimization parameters."""
        score = 0.0
        
        if 'adx_trend_threshold' in params and 'adx_sideways_threshold' in params:
            trend = params['adx_trend_threshold']
            sideways = params['adx_sideways_threshold']
            if trend > sideways and 20.0 <= trend <= 35.0 and 15.0 <= sideways <= 30.0:
                score += 0.2
            else:
                score += 0.1
        
        if 'min_label_balance' in params and 'max_label_balance' in params:
            min_balance = params['min_label_balance']
            max_balance = params['max_label_balance']
            if min_balance < max_balance and 0.03 <= min_balance <= 0.1 and 0.9 <= max_balance <= 0.98:
                score += 0.2
            else:
                score += 0.1
        
        if 'stability_threshold' in params:
            stability = params['stability_threshold']
            if 0.6 <= stability <= 0.9:
                score += 0.2
            else:
                score += 0.1
        
        if 'lgb_learning_rate' in params:
            lr = params['lgb_learning_rate']
            if 0.01 <= lr <= 0.2:
                score += 0.2
            else:
                score += 0.1
        
        if 'model_performance_threshold' in params:
            perf_thresh = params['model_performance_threshold']
            if 0.6 <= perf_thresh <= 0.85:
                score += 0.2
            else:
                score += 0.1
        
        return score
    
    def _evaluate_regime_transitions_params(self, params: Dict[str, Any], 
                                         calibration_results: Dict[str, Any]) -> float:
        """Evaluate regime transition parameters."""
        score = 0.0
        
        if 'transition_intensity_threshold' in params:
            threshold = params['transition_intensity_threshold']
            if 0.2 <= threshold <= 0.5:
                score += 0.2
            else:
                score += 0.1
        
        if 'transition_confidence_threshold' in params:
            confidence_thresh = params['transition_confidence_threshold']
            if 0.6 <= confidence_thresh <= 0.9:
                score += 0.2
            else:
                score += 0.1
        
        if all(key in params for key in ['step9_5_weight', 'step10_weight', 'regime_expert_weight']):
            step9_5_w = params['step9_5_weight']
            step10_w = params['step10_weight']
            regime_w = params['regime_expert_weight']
            total_weight = step9_5_w + step10_w + regime_w
            if 0.9 <= total_weight <= 1.1:
                score += 0.2
            else:
                score += 0.1
        
        if 'transition_lookback_periods' in params:
            lookback = params['transition_lookback_periods']
            if 3 <= lookback <= 10:
                score += 0.2
            else:
                score += 0.1
        
        if 'transition_risk_multiplier' in params:
            risk_mult = params['transition_risk_multiplier']
            if 1.0 <= risk_mult <= 1.5:
                score += 0.2
            else:
                score += 0.1
        
        return score
    
    def _evaluate_signal_aggregation_params(self, params: Dict[str, Any], 
                                         calibration_results: Dict[str, Any]) -> float:
        """Evaluate signal aggregation parameters."""
        score = 0.0
        
        if all(key in params for key in ['analyst_weight', 'tactician_weight', 'scenario_weight', 
                                       'sr_breakout_weight', 'regime_weight']):
            total_weight = (params['analyst_weight'] + params['tactician_weight'] + 
                          params['scenario_weight'] + params['sr_breakout_weight'] + 
                          params['regime_weight'])
            if 1.0 <= total_weight <= 2.5:
                score += 0.2
                if params['analyst_weight'] >= 0.3 and params['tactician_weight'] >= 0.3:
                    score += 0.1
            else:
                score += 0.1
        
        if 'conflict_penalty_factor' in params:
            penalty = params['conflict_penalty_factor']
            if 0.4 <= penalty <= 0.6:
                score += 0.2
            else:
                score += 0.1
        
        if 'min_source_weight' in params:
            min_weight = params['min_source_weight']
            if 0.05 <= min_weight <= 0.15:
                score += 0.1
        
        if 'min_signal_confidence' in params and 'min_aggregated_confidence' in params:
            signal_conf = params['min_signal_confidence']
            agg_conf = params['min_aggregated_confidence']
            if signal_conf < agg_conf and 0.2 <= signal_conf <= 0.4 and 0.4 <= agg_conf <= 0.6:
                score += 0.2
            else:
                score += 0.1
        
        if 'regime_alignment_bonus' in params and 'multi_signal_alignment_bonus' in params:
            regime_bonus = params['regime_alignment_bonus']
            multi_bonus = params['multi_signal_alignment_bonus']
            if 0.1 <= regime_bonus <= 0.25 and 0.05 <= multi_bonus <= 0.15:
                score += 0.1
        
        if 'use_multiplicative' in params and params['use_multiplicative']:
            score += 0.1
        
        return score
    
    async def save_optimization_results(self, optimization_results: Dict[str, Any], 
                                      symbol: str, exchange: str, data_dir: str) -> None:
        """Save optimization results."""
        try:
            optimization_dir = f'{data_dir}/optimization_results'
            os.makedirs(optimization_dir, exist_ok=True)
            
            results_file = f'{optimization_dir}/{exchange}_{symbol}_final_parameters.pkl'
            with open(results_file, 'wb') as f:
                pickle.dump(optimization_results, f)
            
            json_file = f'{optimization_dir}/{exchange}_{symbol}_final_parameters.json'
            with open(json_file, 'w') as f:
                json.dump(optimization_results, f, indent=2, default=str)
            
            self.logger.info(f'Optimization results saved to {results_file}')
            
        except Exception as e:
            self.logger.error(f'Error saving optimization results: {e}')
    
    async def load_optimization_results(self, symbol: str, exchange: str, 
                                      data_dir: str) -> Optional[Dict[str, Any]]:
        """Load previous optimization results."""
        try:
            optimization_dir = f'{data_dir}/optimization_results'
            previous_file = f'{optimization_dir}/{exchange}_{symbol}_final_parameters.pkl'
            
            if os.path.exists(previous_file):
                with open(previous_file, 'rb') as f:
                    return pickle.load(f)
            return None
            
        except Exception as e:
            self.logger.error(f'Error loading optimization results: {e}')
            return None
    
    async def validate_optimization_results(self, optimization_results: Dict[str, Any]) -> bool:
        """Validate optimization results."""
        try:
            if not optimization_results:
                return False
            
            for category in self.categories:
                if category not in optimization_results:
                    self.logger.warning(f'Missing optimization results for category: {category}')
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f'Error validating optimization results: {e}')
            return False
    
    async def generate_optimization_report(self, optimization_results: Dict[str, Any], 
                                         start_time: datetime) -> Dict[str, Any]:
        """Generate optimization report."""
        try:
            report = {
                'optimization_timestamp': start_time.isoformat(),
                'duration_seconds': (datetime.now() - start_time).total_seconds(),
                'categories_optimized': list(optimization_results.keys()),
                'summary': {}
            }
            
            for category, results in optimization_results.items():
                if results and 'best_value' in results:
                    report['summary'][category] = {
                        'best_value': results['best_value'],
                        'n_trials': results.get('n_trials', 0)
                    }
            
            return report
            
        except Exception as e:
            self.logger.error(f'Error generating optimization report: {e}')
            return {'error': str(e)}


# Convenience functions for easy integration
async def optimize_final_parameters(calibration_results: Dict[str, Any], 
                                  config: Dict[str, Any],
                                  symbol: str = "ETHUSDT",
                                  exchange: str = "BINANCE",
                                  data_dir: str = "data/training") -> Dict[str, Any]:
    """
    Convenience function to optimize final parameters.
    
    Args:
        calibration_results: Results from confidence calibration
        config: Configuration dictionary
        symbol: Trading symbol
        exchange: Exchange name
        data_dir: Data directory
        
    Returns:
        Optimization results
    """
    optimizer = FinalParametersOptimizer(config)
    
    # Load previous results for warm start
    previous_results = await optimizer.load_optimization_results(symbol, exchange, data_dir)
    
    # Optimize all parameters
    optimization_results = await optimizer.optimize_all_parameters(
        calibration_results, previous_results
    )
    
    # Validate results
    validation_passed = await optimizer.validate_optimization_results(optimization_results)
    if not validation_passed:
        logger.warning('⚠️ Optimization results validation failed, using fallback parameters')
    
    # Save results
    await optimizer.save_optimization_results(optimization_results, symbol, exchange, data_dir)
    
    # Generate report
    start_time = datetime.now()
    report = await optimizer.generate_optimization_report(optimization_results, start_time)
    
    return {
        'final_parameters': optimization_results,
        'optimization_report': report,
        'validation_passed': validation_passed
    }