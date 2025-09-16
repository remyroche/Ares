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
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Tuple
import optuna
import numpy as np
import logging
from src.utils.enhanced_artifact_manager import get_artifact_manager
from src.utils.artifact_pickup_utils import get_artifact_pickup_utils
from src.utils.version_manager import get_version_manager
from src.utils.nonlinear_optimization_helpers import (
    NonLinearConfig, NonLinearParameterSampler, apply_nonlinear_scoring,
    create_enhanced_search_space, convert_parameters_to_original_space
)

logger = logging.getLogger(__name__)


class FinalParametersOptimizer:
    """
    System-wide final parameters optimizer.
    
    This handles optimization of final system parameters after model training,
    separate from hyperparameter optimization during training.
    """
    
    def __init__(self, config: Dict[str, Any], nonlinear_config: Optional[NonLinearConfig] = None):
        """Initialize the final parameters optimizer."""
        self.config = config
        self.logger = logger.getChild('FinalParametersOptimizer')
        
        # Non-linear optimization configuration
        self.nonlinear_config = nonlinear_config or NonLinearConfig()
        self.parameter_sampler = NonLinearParameterSampler(self.nonlinear_config)
        
        # Parameter categories for optimization
        self.categories = [
            'confidence', 'intensity', 'position_sizing', 'leverage', 'tpsl',
            'ensemble', 'sr', 'two_tier', 'technical_indicators',
            'system_monitoring', 'training_optimization', 'regime_transitions',
            'signal_aggregation', 'turnover_cost_penalty'
        ]
        
        # Default search spaces for each category
        self.default_search_spaces = self._get_default_search_spaces()
        
        # Enhanced search spaces with non-linear transformations
        self.enhanced_search_spaces = self._create_enhanced_search_spaces()
        
        # Optimization settings
        self.n_trials = config.get('n_trials', 50)
        self.timeout = config.get('timeout', 300)
        self.study_name = config.get('study_name', 'final_parameters_optimization')
        self.use_nonlinear_optimization = config.get('use_nonlinear_optimization', True)
        
        self.logger.info("🚀 Final Parameters Optimizer initialized")
        self.logger.info(f"📊 Optimization categories: {len(self.categories)}")
        self.logger.info(f"🔧 Number of trials: {self.n_trials}")
        self.logger.info(f"⏱️ Timeout: {self.timeout}s")
        self.logger.info(f"📝 Study name: {self.study_name}")
        self.logger.info(f"🎯 Categories to optimize: {', '.join(self.categories)}")
        self.logger.info(f"🚀 Non-linear optimization: {self.use_nonlinear_optimization}")
        if self.use_nonlinear_optimization:
            self.logger.info(f"   • Log sampling: {self.nonlinear_config.use_log_sampling}")
            self.logger.info(f"   • Fractional powers: {self.nonlinear_config.use_fractional_powers}")
            self.logger.info(f"   • Sigmoid transforms: {self.nonlinear_config.use_sigmoid_transforms}")
            self.logger.info(f"   • Adaptive transforms: {self.nonlinear_config.use_adaptive_transforms}")
    
        # Initialize artifact and version managers
        self.artifact_manager = get_artifact_manager()
        self.pickup_utils = get_artifact_pickup_utils()
        self.version_manager = get_version_manager()
    
    def _create_enhanced_search_spaces(self) -> Dict[str, Dict[str, Any]]:
        """Create enhanced search spaces with non-linear transformation metadata."""
        enhanced_spaces = {}
        
        for category, space in self.default_search_spaces.items():
            enhanced_spaces[category] = create_enhanced_search_space(space, self.nonlinear_config)
        
        return enhanced_spaces
    
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
            },
            'turnover_cost_penalty': {
                'turnover_penalty_weight': {'type': 'float', 'min': 0.1, 'max': 1.0},
                'commission_rate': {'type': 'float', 'min': 0.0005, 'max': 0.002},
                'slippage_rate': {'type': 'float', 'min': 0.0002, 'max': 0.001},
                'max_turnover_rate': {'type': 'float', 'min': 0.1, 'max': 0.5},
                'round_trip_multiplier': {'type': 'float', 'min': 1.5, 'max': 3.0}
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
            self.logger.info(f"📊 Calibration results available: {len(calibration_results)} keys")
            self.logger.info(f"🔄 Previous results available: {previous_results is not None}")
            
            optimization_results = {}
            start_time = time.time()
            
            for i, category in enumerate(self.categories, 1):
                self.logger.info(f"🔄 Optimizing {category} parameters ({i}/{len(self.categories)})...")
                category_start = time.time()
                
                category_results = await self._optimize_category(
                    category, calibration_results, 
                    previous_results.get(category) if previous_results else None
                )
                
                category_duration = time.time() - category_start
                optimization_results[category] = category_results
                
                if category_results and 'best_value' in category_results:
                    self.logger.info(f"✅ {category} optimization completed in {category_duration:.2f}s - Best value: {category_results['best_value']:.4f}")
                else:
                    self.logger.warning(f"⚠️ {category} optimization completed in {category_duration:.2f}s - No results obtained")
            
            total_duration = time.time() - start_time
            self.logger.info("✅ Final parameters optimization completed")
            self.logger.info(f"⏱️ Total optimization time: {total_duration:.2f}s")
            self.logger.info(f"📊 Categories optimized: {len(optimization_results)}")
            
            return optimization_results
            
        except Exception as e:
            self.logger.error(f"❌ Error in final parameters optimization: {e}")
            self.logger.exception("Full traceback:")
            raise
    
    async def _optimize_category(self, category: str, calibration_results: Dict[str, Any], 
                               previous_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Enhanced optimization for a specific category with coarse/fine grid + Optuna TPE.
        
        Args:
            category: Parameter category to optimize
            calibration_results: Results from confidence calibration
            previous_results: Previous optimization results for this category
            
        Returns:
            Dict containing optimization results for the category
        """
        try:
            self.logger.info(f"🔍 Analyzing search space for category: {category}")
            
            # Use enhanced search space if non-linear optimization is enabled
            if self.use_nonlinear_optimization and category in self.enhanced_search_spaces:
                search_space = self.enhanced_search_spaces[category]
                self.logger.info(f"🚀 Using enhanced non-linear search space for {category}")
            else:
                search_space = self.default_search_spaces.get(category, {})
                self.logger.info(f"📊 Using standard search space for {category}")
            
            if not search_space:
                self.logger.warning(f"⚠️ No search space found for category: {category}")
                return {}
            
            self.logger.info(f"📊 Search space parameters: {len(search_space)}")
            for param_name, param_config in search_space.items():
                if self.use_nonlinear_optimization and 'transform_type' in param_config:
                    transform_type = param_config['transform_type']
                    self.logger.debug(f"   • {param_name}: {param_config['type']} [{param_config.get('min', 'N/A')}-{param_config.get('max', 'N/A')}] (transform: {transform_type})")
                else:
                    self.logger.debug(f"   • {param_name}: {param_config['type']} [{param_config.get('min', 'N/A')}-{param_config.get('max', 'N/A')}]")
            
            # Stage 1: Coarse Grid Search
            self.logger.info(f"🎯 Stage 1: Coarse grid search for {category}")
            coarse_start = time.time()
            coarse_result = await self._coarse_grid_search(category, search_space, calibration_results)
            coarse_time = time.time() - coarse_start
            
            if not coarse_result or coarse_result.get('best_score', 0) <= 0:
                self.logger.warning(f"⚠️ Coarse grid search failed for {category}, using default parameters")
                return self._create_fallback_result(category)
            
            self.logger.info(f"✅ Coarse grid completed in {coarse_time:.2f}s - Best score: {coarse_result['best_score']:.4f}")
            
            # Stage 2: Fine Grid Search around best coarse parameters
            self.logger.info(f"🎯 Stage 2: Fine grid search for {category}")
            fine_start = time.time()
            fine_result = await self._fine_grid_search(category, search_space, coarse_result['best_params'], calibration_results)
            fine_time = time.time() - fine_start
            
            if not fine_result or fine_result.get('best_score', 0) <= coarse_result['best_score']:
                self.logger.info(f"ℹ️ Fine grid search did not improve results, using coarse grid results")
                best_params = coarse_result['best_params']
                best_score = coarse_result['best_score']
                grid_stage = 'coarse'
            else:
                self.logger.info(f"✅ Fine grid completed in {fine_time:.2f}s - Best score: {fine_result['best_score']:.4f}")
                best_params = fine_result['best_params']
                best_score = fine_result['best_score']
                grid_stage = 'fine'
            
            # Stage 3: Optuna TPE Optimization around best grid parameters
            self.logger.info(f"🎯 Stage 3: Optuna TPE optimization for {category}")
            optuna_start = time.time()
            optuna_result = await self._optuna_tpe_optimization(category, search_space, best_params, calibration_results)
            optuna_time = time.time() - optuna_start
            
            if optuna_result and optuna_result.get('best_score', 0) > best_score:
                self.logger.info(f"✅ Optuna TPE completed in {optuna_time:.2f}s - Best score: {optuna_result['best_score']:.4f}")
                final_params = optuna_result['best_params']
                final_score = optuna_result['best_score']
                final_stage = 'optuna'
            else:
                self.logger.info(f"ℹ️ Optuna TPE did not improve results, using grid search results")
                final_params = best_params
                final_score = best_score
                final_stage = grid_stage
            
            total_time = coarse_time + fine_time + optuna_time
            
            self.logger.info(f"🏆 Final parameters for {category}:")
            for param, value in final_params.items():
                self.logger.info(f"   • {param}: {value}")
            self.logger.info(f"📈 Final objective value: {final_score:.4f}")
            self.logger.info(f"⏱️ Total optimization time: {total_time:.2f}s")
            self.logger.info(f"🎯 Best stage: {final_stage}")
            
            result = {
                'best_params': final_params,
                'best_value': final_score,
                'optimization_method': 'coarse_fine_optuna',
                'coarse_result': coarse_result,
                'fine_result': fine_result,
                'optuna_result': optuna_result,
                'best_stage': final_stage,
                'coarse_time': coarse_time,
                'fine_time': fine_time,
                'optuna_time': optuna_time,
                'total_time': total_time,
                'convergence_analysis': {
                    'coarse_score': coarse_result.get('best_score', 0),
                    'fine_score': fine_result.get('best_score', 0) if fine_result else 0,
                    'optuna_score': optuna_result.get('best_score', 0) if optuna_result else 0,
                    'final_score': final_score
                }
            }
            
            if self.use_nonlinear_optimization:
                result['enhancement_methods_used'] = self._get_used_enhancement_methods(search_space)
                result['nonlinear_optimization'] = True
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error optimizing category {category}: {e}")
            self.logger.exception("Full traceback:")
            return {}
    
    def _objective_function(self, trial: optuna.Trial, category: str, 
                          search_space: Dict[str, Dict[str, Any]], 
                          calibration_results: Dict[str, Any]) -> float:
        """
        Enhanced objective function for Optuna optimization with non-linear sampling.
        
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
            
            # Use enhanced search space if non-linear optimization is enabled
            if self.use_nonlinear_optimization and category in self.enhanced_search_spaces:
                enhanced_space = self.enhanced_search_spaces[category]
                for param_name, param_config in enhanced_space.items():
                    if param_config['type'] == 'float':
                        # Use enhanced non-linear sampling
                        transform_type = param_config.get('transform_type', 'auto')
                        params[param_name] = self.parameter_sampler.suggest_enhanced_float(
                            trial, param_name, param_config['min'], param_config['max'], transform_type
                        )
                    elif param_config['type'] == 'int':
                        # Use enhanced non-linear sampling for integers
                        transform_type = param_config.get('transform_type', 'auto')
                        params[param_name] = self.parameter_sampler.suggest_enhanced_int(
                            trial, param_name, param_config['min'], param_config['max'], transform_type
                        )
                    elif param_config['type'] == 'bool':
                        params[param_name] = trial.suggest_categorical(param_name, [True, False])
            else:
                # Fallback to original linear sampling
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
                        params[param_name] = trial.suggest_categorical(param_name, [True, False])
            
            # Evaluate configuration
            score = self._evaluate_configuration(category, params, calibration_results)
            
            # Apply non-linear scoring enhancements
            if self.use_nonlinear_optimization:
                enhanced_score = apply_nonlinear_scoring(score, params, category)
                return enhanced_score
            
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
            elif category == 'turnover_cost_penalty':
                base_score = self._evaluate_turnover_cost_penalty_params(params, calibration_results)

            # Apply turnover cost penalty to all categories
            if base_score > 0.0:
                turnover_penalty = self._calculate_turnover_penalty(params, calibration_results)
                base_score -= turnover_penalty

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

    def _evaluate_turnover_cost_penalty_params(self, params: Dict[str, Any],
                                             calibration_results: Dict[str, Any]) -> float:
        """Evaluate turnover cost penalty parameters."""
        score = 0.0

        if 'turnover_penalty_weight' in params:
            weight = params['turnover_penalty_weight']
            if 0.2 <= weight <= 0.8:
                score += 0.3
            elif 0.1 <= weight <= 1.0:
                score += 0.2
            else:
                score += 0.1

        if 'commission_rate' in params:
            commission = params['commission_rate']
            if 0.0008 <= commission <= 0.0015:
                score += 0.2
            else:
                score += 0.1

        if 'slippage_rate' in params:
            slippage = params['slippage_rate']
            if 0.0003 <= slippage <= 0.0008:
                score += 0.2
            else:
                score += 0.1

        if 'round_trip_multiplier' in params:
            multiplier = params['round_trip_multiplier']
            if 1.8 <= multiplier <= 2.5:
                score += 0.2
            else:
                score += 0.1

        return score

    def _calculate_turnover_penalty(self, params: Dict[str, Any],
                                  calibration_results: Dict[str, Any]) -> float:
        """
        Calculate turnover penalty for a given configuration.

        The penalty is calculated as:
        turnover_penalty = turnover_rate * transaction_cost * round_trip_multiplier

        Where transaction_cost = commission_rate + slippage_rate

        Args:
            params: Current parameter configuration
            calibration_results: Results from calibration/backtesting

        Returns:
            Turnover penalty to subtract from base score
        """
        try:
            # Extract cost parameters from current params or use defaults
            commission_rate = params.get('commission_rate', 0.001)
            slippage_rate = params.get('slippage_rate', 0.0005)
            round_trip_multiplier = params.get('round_trip_multiplier', 2.0)
            turnover_penalty_weight = params.get('turnover_penalty_weight', 0.5)

            # Calculate transaction cost per trade
            transaction_cost = commission_rate + slippage_rate

            # Estimate turnover rate from calibration results or use default
            # In a real implementation, this would be calculated from actual backtesting results
            estimated_turnover_rate = self._estimate_turnover_rate(params, calibration_results)

            # Calculate round-trip cost
            round_trip_cost = transaction_cost * round_trip_multiplier

            # Calculate penalty
            turnover_penalty = estimated_turnover_rate * round_trip_cost * turnover_penalty_weight

            # Log the calculation for transparency
            if turnover_penalty > 0.001:  # Only log significant penalties
                self.logger.debug(f"⚠️ Turnover penalty: {turnover_penalty:.4f} "
                                f"(rate: {estimated_turnover_rate:.3f}, cost: {round_trip_cost:.6f})")

            return turnover_penalty

        except Exception as e:
            self.logger.warning(f"Error calculating turnover penalty: {e}")
            return 0.001  # Small default penalty

    def _estimate_turnover_rate(self, params: Dict[str, Any],
                               calibration_results: Dict[str, Any]) -> float:
        """
        Estimate turnover rate based on parameters and calibration results.

        Turnover rate represents how much of the portfolio changes per period.
        Higher trading frequency = higher turnover = higher costs.

        Args:
            params: Current parameter configuration
            calibration_results: Calibration/backtesting results

        Returns:
            Estimated turnover rate (0.0 to 1.0)
        """
        try:
            # Base turnover rate depends on trading frequency
            base_turnover = 0.15  # Default 15% portfolio turnover per period

            # Adjust based on confidence thresholds (lower thresholds = more trades)
            if 'base_entry_threshold' in params:
                threshold = params['base_entry_threshold']
                if threshold < 0.6:
                    base_turnover *= 1.3  # More aggressive = more trades
                elif threshold > 0.8:
                    base_turnover *= 0.7  # More conservative = fewer trades

            # Adjust based on position sizing (larger positions = potentially more turnover)
            if 'base_position_size' in params:
                position_size = params['base_position_size']
                if position_size > 0.1:
                    base_turnover *= 1.2
                elif position_size < 0.03:
                    base_turnover *= 0.8

            # Adjust based on TP/SL ratios (wider ranges = fewer trades)
            if all(key in params for key in ['tp_long', 'sl_long']):
                tp = params['tp_long']
                sl = params['sl_long']
                if tp > sl * 2:
                    base_turnover *= 0.8  # Wider profit targets = fewer trades
                elif tp < sl * 1.2:
                    base_turnover *= 1.2  # Narrow profit targets = more trades

            # Extract from calibration results if available
            if calibration_results and 'estimated_turnover' in calibration_results:
                calibrated_turnover = calibration_results['estimated_turnover']
                base_turnover = (base_turnover + calibrated_turnover) / 2  # Average with estimate

            # Ensure reasonable bounds
            max_turnover = params.get('max_turnover_rate', 0.5)
            base_turnover = min(base_turnover, max_turnover)

            return base_turnover

        except Exception as e:
            self.logger.warning(f"Error estimating turnover rate: {e}")
            return 0.15  # Default turnover rate

    async def save_optimization_results(self, optimization_results: Dict[str, Any],
                                      symbol: str, exchange: str, data_dir: str) -> None:
        """Save optimization results."""
        try:
            self.logger.info(f"💾 Saving optimization results for {exchange}_{symbol}")
            optimization_dir = f'{data_dir}/optimization_results'
            os.makedirs(optimization_dir, exist_ok=True)
            self.logger.info(f"📁 Optimization directory: {optimization_dir}")
            
            results_file = f'{optimization_dir}/{exchange}_{symbol}_final_parameters.pkl'
            self.logger.info(f"🔄 Saving pickle file: {results_file}")
            with open(results_file, 'wb') as f:
                pickle.dump(optimization_results, f)
            
            json_file = f'{optimization_dir}/{exchange}_{symbol}_final_parameters.json'
            self.logger.info(f"🔄 Saving JSON file: {json_file}")
            with open(json_file, 'w') as f:
                json.dump(optimization_results, f, indent=2, default=str)
            
            # Log file sizes
            pickle_size = os.path.getsize(results_file) / 1024  # KB
            json_size = os.path.getsize(json_file) / 1024  # KB
            
            self.logger.info(f'✅ Optimization results saved successfully')
            self.logger.info(f'📊 Pickle file size: {pickle_size:.1f} KB')
            self.logger.info(f'📊 JSON file size: {json_size:.1f} KB')
            self.logger.info(f'📁 Files saved to: {optimization_dir}')
            
        except Exception as e:
            self.logger.error(f'❌ Error saving optimization results: {e}')
            self.logger.exception("Full traceback:")
    
    async def load_optimization_results(self, symbol: str, exchange: str, 
                                      data_dir: str) -> Optional[Dict[str, Any]]:
        """Load previous optimization results."""
        try:
            self.logger.info(f"📂 Loading previous optimization results for {exchange}_{symbol}")
            optimization_dir = f'{data_dir}/optimization_results'
            previous_file = f'{optimization_dir}/{exchange}_{symbol}_final_parameters.pkl'
            
            self.logger.info(f"🔍 Checking for previous results: {previous_file}")
            
            if os.path.exists(previous_file):
                file_size = os.path.getsize(previous_file) / 1024  # KB
                self.logger.info(f"📁 Previous results found - File size: {file_size:.1f} KB")
                
                with open(previous_file, 'rb') as f:
                    results = pickle.load(f)
                
                if results:
                    self.logger.info(f"✅ Successfully loaded previous optimization results")
                    self.logger.info(f"📊 Categories in previous results: {len(results)}")
                    for category in results.keys():
                        self.logger.debug(f"   • {category}")
                else:
                    self.logger.warning(f"⚠️ Previous results file is empty")
                
                return results
            else:
                self.logger.info(f"ℹ️ No previous optimization results found")
                return None
            
        except Exception as e:
            self.logger.error(f'❌ Error loading optimization results: {e}')
            self.logger.exception("Full traceback:")
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
    
    def _analyze_convergence(self, study: optuna.Study) -> Dict[str, Any]:
        """Analyze convergence characteristics of the optimization."""
        try:
            if len(study.trials) < 5:
                return {'convergence_quality': 'insufficient_data'}
            
            values = [t.value for t in study.trials if t.value is not None]
            if not values:
                return {'convergence_quality': 'no_valid_trials'}
            
            # Calculate convergence metrics
            best_values = []
            current_best = float('-inf')
            for value in values:
                if value > current_best:
                    current_best = value
                best_values.append(current_best)
            
            # Improvement rate
            total_improvement = best_values[-1] - best_values[0]
            improvement_rate = total_improvement / len(values) if len(values) > 0 else 0
            
            # Convergence stability (variance in last 20% of trials)
            last_portion = int(len(best_values) * 0.2)
            if last_portion > 1:
                recent_values = best_values[-last_portion:]
                convergence_variance = np.var(recent_values)
            else:
                convergence_variance = 0
            
            # Convergence quality assessment
            if improvement_rate > 0.01 and convergence_variance < 0.001:
                convergence_quality = 'excellent'
            elif improvement_rate > 0.005 and convergence_variance < 0.01:
                convergence_quality = 'good'
            elif improvement_rate > 0.001:
                convergence_quality = 'fair'
            else:
                convergence_quality = 'poor'
            
            return {
                'convergence_quality': convergence_quality,
                'total_improvement': total_improvement,
                'improvement_rate': improvement_rate,
                'convergence_variance': convergence_variance,
                'final_best_value': best_values[-1],
                'n_trials': len(values)
            }
            
        except Exception as e:
            self.logger.warning(f"Convergence analysis failed: {e}")
            return {'convergence_quality': 'analysis_failed', 'error': str(e)}
    
    def _get_used_enhancement_methods(self, search_space: Dict[str, Dict[str, Any]]) -> List[str]:
        """Get list of enhancement methods used in the search space."""
        methods = set()
        for param_config in search_space.values():
            if 'transform_type' in param_config:
                transform_type = param_config['transform_type']
                if transform_type in ['log', 'power', 'sigmoid', 'adaptive']:
                    methods.add(transform_type)
        return list(methods)
    
    async def _coarse_grid_search(self, category: str, search_space: Dict[str, Dict[str, Any]], 
                                calibration_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform coarse grid search with fewer parameter combinations."""
        try:
            self.logger.info(f"🔍 Creating coarse grid for {category}")
            
            # Create coarse parameter grid
            coarse_grid = self._create_coarse_parameter_grid(search_space)
            self.logger.info(f"📊 Coarse grid size: {len(coarse_grid)} combinations")
            
            best_score = -np.inf
            best_params = {}
            parameter_scores = []
            
            # Evaluate each parameter combination
            for i, params in enumerate(coarse_grid):
                try:
                    score = self._evaluate_configuration(category, params, calibration_results)
                    parameter_scores.append((params, score))
                    
                    if score > best_score:
                        best_score = score
                        best_params = params.copy()
                    
                    if (i + 1) % 10 == 0:
                        self.logger.debug(f"   Evaluated {i + 1}/{len(coarse_grid)} combinations")
                        
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to evaluate parameters {params}: {e}")
                    continue
            
            if not parameter_scores:
                self.logger.error(f"❌ No valid parameter combinations found for {category}")
                return {}
            
            self.logger.info(f"✅ Coarse grid search completed - Best score: {best_score:.4f}")
            
            return {
                'best_params': best_params,
                'best_score': best_score,
                'n_combinations': len(coarse_grid),
                'valid_combinations': len(parameter_scores),
                'parameter_scores': parameter_scores[:10]  # Keep top 10 for analysis
            }
            
        except Exception as e:
            self.logger.error(f"❌ Coarse grid search failed for {category}: {e}")
            return {}
    
    async def _fine_grid_search(self, category: str, search_space: Dict[str, Dict[str, Any]], 
                              best_coarse_params: Dict[str, Any], calibration_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform fine grid search around best coarse parameters."""
        try:
            self.logger.info(f"🔍 Creating fine grid around best coarse parameters for {category}")
            
            # Create fine parameter grid around best coarse parameters
            fine_grid = self._create_fine_parameter_grid(search_space, best_coarse_params)
            self.logger.info(f"📊 Fine grid size: {len(fine_grid)} combinations")
            
            best_score = -np.inf
            best_params = {}
            parameter_scores = []
            
            # Evaluate each parameter combination
            for i, params in enumerate(fine_grid):
                try:
                    score = self._evaluate_configuration(category, params, calibration_results)
                    parameter_scores.append((params, score))
                    
                    if score > best_score:
                        best_score = score
                        best_params = params.copy()
                    
                    if (i + 1) % 10 == 0:
                        self.logger.debug(f"   Evaluated {i + 1}/{len(fine_grid)} combinations")
                        
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to evaluate parameters {params}: {e}")
                    continue
            
            if not parameter_scores:
                self.logger.error(f"❌ No valid parameter combinations found for {category}")
                return {}
            
            self.logger.info(f"✅ Fine grid search completed - Best score: {best_score:.4f}")
            
            return {
                'best_params': best_params,
                'best_score': best_score,
                'n_combinations': len(fine_grid),
                'valid_combinations': len(parameter_scores),
                'parameter_scores': parameter_scores[:10]  # Keep top 10 for analysis
            }
            
        except Exception as e:
            self.logger.error(f"❌ Fine grid search failed for {category}: {e}")
            return {}
    
    async def _optuna_tpe_optimization(self, category: str, search_space: Dict[str, Dict[str, Any]], 
                                     best_grid_params: Dict[str, Any], calibration_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform Optuna TPE optimization around best grid parameters."""
        try:
            self.logger.info(f"🎲 Starting Optuna TPE optimization for {category}")
            
            # Create narrowed search space around best grid parameters
            narrowed_space = self._create_narrowed_search_space(search_space, best_grid_params)
            
            study_name = f'{self.study_name}_{category}_tpe'
            if self.use_nonlinear_optimization:
                study_name += '_enhanced'
            
            # Use TPE sampler with enhanced settings
            sampler = optuna.samplers.TPESampler(
                n_startup_trials=5,  # Fewer startup trials since we have good starting point
                n_ei_candidates=24,
                gamma=lambda x: min(int(0.25 * x), 25),
                prior_weight=1.0,
                consider_magic_clip=True,
                consider_endpoints=True
            )
            
            study = optuna.create_study(
                study_name=study_name,
                direction='maximize',
                sampler=sampler,
                storage='sqlite:///optuna_studies_coarse_fine.db',
                load_if_exists=True
            )
            
            # Use fewer trials since we're fine-tuning around good parameters
            n_trials = min(self.n_trials // 3, 30)  # Use 1/3 of original trials or max 30
            timeout = min(self.timeout // 3, 120)   # Use 1/3 of original timeout or max 2 minutes
            
            self.logger.info(f"🎯 Starting TPE optimization with {n_trials} trials (timeout: {timeout}s)")
            
            def objective(trial):
                return self._objective_function(trial, category, narrowed_space, calibration_results)
            
            start_time = time.time()
            study.optimize(objective, n_trials=n_trials, timeout=timeout)
            optimization_time = time.time() - start_time
            
            best_params = study.best_params
            best_value = study.best_value
            
            # Convert parameters back to original space for reporting
            if self.use_nonlinear_optimization:
                converted_params = convert_parameters_to_original_space(best_params, narrowed_space)
            else:
                converted_params = best_params
            
            self.logger.info(f"✅ Optuna TPE optimization completed in {optimization_time:.2f}s")
            self.logger.info(f"📈 Best TPE score: {best_value:.4f}")
            
            # Enhanced convergence analysis
            convergence_analysis = self._analyze_convergence(study)
            
            return {
                'best_params': converted_params,
                'best_score': best_value,
                'study_name': study_name,
                'n_trials': len(study.trials),
                'optimization_time': optimization_time,
                'convergence_analysis': convergence_analysis,
                'narrowed_space': narrowed_space
            }
            
        except Exception as e:
            self.logger.error(f"❌ Optuna TPE optimization failed for {category}: {e}")
            return {}
    
    def _create_coarse_parameter_grid(self, search_space: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create coarse parameter grid with fewer combinations."""
        import itertools
        
        param_combinations = []
        
        for param_name, param_config in search_space.items():
            if param_config['type'] == 'float':
                # Use 3 points for coarse grid
                min_val, max_val = param_config['min'], param_config['max']
                if param_config.get('log', False) or (self.use_nonlinear_optimization and 
                    param_config.get('transform_type') == 'log'):
                    # Log-spaced values
                    values = np.logspace(np.log10(min_val), np.log10(max_val), 3)
                else:
                    # Linear-spaced values
                    values = np.linspace(min_val, max_val, 3)
                param_combinations.append([(param_name, v) for v in values])
                
            elif param_config['type'] == 'int':
                # Use 3 points for coarse grid
                min_val, max_val = param_config['min'], param_config['max']
                if max_val - min_val <= 2:
                    values = list(range(min_val, max_val + 1))
                else:
                    values = np.linspace(min_val, max_val, 3, dtype=int)
                param_combinations.append([(param_name, v) for v in values])
                
            elif param_config['type'] == 'bool':
                param_combinations.append([(param_name, v) for v in [True, False]])
        
        # Generate all combinations
        all_combinations = list(itertools.product(*param_combinations))
        
        # Convert to list of dictionaries
        grid = []
        for combination in all_combinations:
            param_dict = dict(combination)
            grid.append(param_dict)
        
        return grid
    
    def _create_fine_parameter_grid(self, search_space: Dict[str, Dict[str, Any]], 
                                  best_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create fine parameter grid around best parameters."""
        import itertools
        
        param_combinations = []
        
        for param_name, param_config in search_space.items():
            if param_name not in best_params:
                continue
                
            best_value = best_params[param_name]
            
            if param_config['type'] == 'float':
                min_val, max_val = param_config['min'], param_config['max']
                # Create fine grid around best value (±20% of range)
                range_size = max_val - min_val
                fine_range = range_size * 0.2
                fine_min = max(min_val, best_value - fine_range)
                fine_max = min(max_val, best_value + fine_range)
                
                # Use 5 points for fine grid
                if param_config.get('log', False) or (self.use_nonlinear_optimization and 
                    param_config.get('transform_type') == 'log'):
                    # Log-spaced values
                    values = np.logspace(np.log10(fine_min), np.log10(fine_max), 5)
                else:
                    # Linear-spaced values
                    values = np.linspace(fine_min, fine_max, 5)
                param_combinations.append([(param_name, v) for v in values])
                
            elif param_config['type'] == 'int':
                min_val, max_val = param_config['min'], param_config['max']
                # Create fine grid around best value (±2 values)
                fine_min = max(min_val, best_value - 2)
                fine_max = min(max_val, best_value + 2)
                values = list(range(fine_min, fine_max + 1))
                param_combinations.append([(param_name, v) for v in values])
                
            elif param_config['type'] == 'bool':
                param_combinations.append([(param_name, v) for v in [True, False]])
        
        # Generate all combinations
        all_combinations = list(itertools.product(*param_combinations))
        
        # Convert to list of dictionaries
        grid = []
        for combination in all_combinations:
            param_dict = dict(combination)
            grid.append(param_dict)
        
        return grid
    
    def _create_narrowed_search_space(self, search_space: Dict[str, Dict[str, Any]], 
                                    best_params: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Create narrowed search space around best parameters for Optuna."""
        narrowed_space = {}
        
        for param_name, param_config in search_space.items():
            if param_name not in best_params:
                narrowed_space[param_name] = param_config
                continue
            
            best_value = best_params[param_name]
            narrowed_config = param_config.copy()
            
            if param_config['type'] == 'float':
                min_val, max_val = param_config['min'], param_config['max']
                # Narrow range to ±10% of original range around best value
                range_size = max_val - min_val
                narrow_range = range_size * 0.1
                narrowed_config['min'] = max(min_val, best_value - narrow_range)
                narrowed_config['max'] = min(max_val, best_value + narrow_range)
                
            elif param_config['type'] == 'int':
                min_val, max_val = param_config['min'], param_config['max']
                # Narrow range to ±1 around best value
                narrowed_config['min'] = max(min_val, best_value - 1)
                narrowed_config['max'] = min(max_val, best_value + 1)
            
            narrowed_space[param_name] = narrowed_config
        
        return narrowed_space
    
    def _create_fallback_result(self, category: str) -> Dict[str, Any]:
        """Create fallback result with default parameters."""
        default_params = {}
        search_space = self.default_search_spaces.get(category, {})
        
        for param_name, param_config in search_space.items():
            if param_config['type'] == 'float':
                # Use middle value
                default_params[param_name] = (param_config['min'] + param_config['max']) / 2
            elif param_config['type'] == 'int':
                # Use middle value
                default_params[param_name] = (param_config['min'] + param_config['max']) // 2
            elif param_config['type'] == 'bool':
                default_params[param_name] = True
        
        return {
            'best_params': default_params,
            'best_value': 0.0,
            'optimization_method': 'fallback',
            'error': 'Grid search failed, using default parameters'
        }


# Convenience functions for easy integration
async def optimize_final_parameters(calibration_results: Dict[str, Any], 
                                  config: Dict[str, Any],
                                  symbol: str = "ETHUSDT",
                                  exchange: str = "BINANCE",
                                  data_dir: str = "data/training",
                                  nonlinear_config: Optional[NonLinearConfig] = None) -> Dict[str, Any]:
    """
    Enhanced convenience function to optimize final parameters with optional non-linear transformations.
    
    Args:
        calibration_results: Results from confidence calibration
        config: Configuration dictionary
        symbol: Trading symbol
        exchange: Exchange name
        data_dir: Data directory
        nonlinear_config: Non-linear optimization configuration (optional)
        
    Returns:
        Optimization results
    """
    optimizer = FinalParametersOptimizer(config, nonlinear_config)
    
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
    
    result = {
        'final_parameters': optimization_results,
        'optimization_report': report,
        'validation_passed': validation_passed
    }
    
    # Add non-linear optimization summary if used
    if optimizer.use_nonlinear_optimization:
        result['nonlinear_optimization'] = True
        result['enhancement_summary'] = {
            'use_log_sampling': optimizer.nonlinear_config.use_log_sampling,
            'use_fractional_powers': optimizer.nonlinear_config.use_fractional_powers,
            'use_sigmoid_transforms': optimizer.nonlinear_config.use_sigmoid_transforms,
            'use_adaptive_transforms': optimizer.nonlinear_config.use_adaptive_transforms
        }
    
    return result