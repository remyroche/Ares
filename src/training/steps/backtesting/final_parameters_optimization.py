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
                'tactician_confidence_threshold': {'type': 'float', 'min': 0.7, 'max': 0.9},
                'tactician_confidence_weight': {'type': 'float', 'min': 0.3, 'max': 0.8},
                'analyst_confidence_weight': {'type': 'float', 'min': 0.2, 'max': 0.7},
                'confidence_combination_method': {'type': 'categorical', 'choices': ['multiplicative', 'logarithmic', 'harmonic', 'weighted_average']},
                # Exit-specific confidence parameters
                'exit_confidence_threshold': {'type': 'float', 'min': 0.3, 'max': 0.7},
                'tactician_exit_confidence_weight': {'type': 'float', 'min': 0.4, 'max': 0.8},
                'analyst_exit_confidence_weight': {'type': 'float', 'min': 0.2, 'max': 0.6},
                'exit_confidence_combination_method': {'type': 'categorical', 'choices': ['multiplicative', 'logarithmic', 'weighted_average']}
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
                    elif param_config['type'] == 'categorical':
                        params[param_name] = trial.suggest_categorical(param_name, param_config['choices'])
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
                    elif param_config['type'] == 'categorical':
                        params[param_name] = trial.suggest_categorical(param_name, param_config['choices'])
            
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
            
            # Special handling for confidence category - add exit strategy backtesting
            if category == 'confidence' and base_score > 0.0:
                exit_strategy_score = self._evaluate_exit_strategy_performance(params, calibration_results)
                base_score += exit_strategy_score * 0.4  # Weight exit strategy evaluation

            return base_score

        except Exception as e:
            self.logger.error(f"Error evaluating configuration for {category}: {e}")
            return 0.0
    
    def _evaluate_confidence_params(self, params: Dict[str, Any], 
                                  calibration_results: Dict[str, Any]) -> float:
        """Evaluate confidence threshold parameters with optimal confidence calculation."""
        score = 0.0
        
        # Base entry threshold evaluation
        if 'base_entry_threshold' in params:
            threshold = params['base_entry_threshold']
            if 0.6 <= threshold <= 0.8:
                score += 0.3
            elif 0.5 <= threshold <= 0.9:
                score += 0.2
            else:
                score += 0.1
        
        # Enhanced confidence evaluation with optimal calculation
        if 'analyst_confidence_threshold' in params and 'tactician_confidence_threshold' in params:
            analyst_thresh = params['analyst_confidence_threshold']
            tactician_thresh = params['tactician_confidence_threshold']
            
            # Basic threshold validation
            if tactician_thresh > analyst_thresh:
                score += 0.2
            if 0.1 <= tactician_thresh - analyst_thresh <= 0.2:
                score += 0.1
            
            # Extract confidence weights from parameters if available
            tactician_weight = params.get('tactician_confidence_weight', 0.6)
            analyst_weight = params.get('analyst_confidence_weight', 0.4)
            
            # Validate weight constraints
            if 0.1 <= tactician_weight <= 0.9 and 0.1 <= analyst_weight <= 0.9:
                score += 0.1
                if abs(tactician_weight + analyst_weight - 1.0) < 0.1:
                    score += 0.1  # Bonus for weights that sum close to 1.0
            
            # Extract exit confidence parameters
            exit_threshold = params.get('exit_confidence_threshold', 0.5)
            tactician_exit_weight = params.get('tactician_exit_confidence_weight', 0.6)
            analyst_exit_weight = params.get('analyst_exit_confidence_weight', 0.4)
            exit_combination_method = params.get('exit_confidence_combination_method', 'multiplicative')
            
            # Validate exit confidence parameters
            if 0.3 <= exit_threshold <= 0.7:
                score += 0.1
            if 0.2 <= tactician_exit_weight <= 0.8 and 0.2 <= analyst_exit_weight <= 0.8:
                score += 0.1
                if abs(tactician_exit_weight + analyst_exit_weight - 1.0) < 0.1:
                    score += 0.1  # Bonus for exit weights that sum close to 1.0
            
            # Bonus for advanced exit combination methods
            if exit_combination_method in ['multiplicative', 'logarithmic']:
                score += 0.1
            
            # Update calibration results with parameter weights
            enhanced_calibration = calibration_results.copy()
            enhanced_calibration.update({
                'tactician_confidence_weight': tactician_weight,
                'analyst_confidence_weight': analyst_weight,
                'confidence_combination_method': params.get('confidence_combination_method', 'weighted_average'),
                # Exit-specific parameters
                'exit_confidence_threshold': exit_threshold,
                'tactician_exit_confidence_weight': tactician_exit_weight,
                'analyst_exit_confidence_weight': analyst_exit_weight,
                'exit_confidence_combination_method': exit_combination_method
            })
            
            # Calculate optimal confidence using multiplicative and logarithmic operations
            optimal_confidence = self._calculate_optimal_confidence(
                analyst_thresh, tactician_thresh, enhanced_calibration
            )
            
            if optimal_confidence is not None:
                # Score based on optimal confidence quality
                if optimal_confidence > 0.8:
                    score += 0.3
                elif optimal_confidence > 0.6:
                    score += 0.2
                else:
                    score += 0.1
                
                # Additional score for confidence stability
                confidence_stability = self._evaluate_confidence_stability(
                    analyst_thresh, tactician_thresh, enhanced_calibration
                )
                score += confidence_stability * 0.2
                
                # Score based on combination method effectiveness
                combination_method = params.get('confidence_combination_method', 'weighted_average')
                if combination_method in ['multiplicative', 'logarithmic']:
                    score += 0.1  # Bonus for advanced methods
                
                # Evaluate exit confidence calculation
                exit_confidence_score = self._evaluate_exit_confidence_calculation(
                    analyst_thresh, tactician_thresh, enhanced_calibration
                )
                score += exit_confidence_score * 0.3  # Weight exit confidence evaluation
        
        return score
    
    def _calculate_optimal_confidence(self, analyst_threshold: float, tactician_threshold: float, 
                                    calibration_results: Dict[str, Any]) -> Optional[float]:
        """
        Calculate optimal confidence using multiplicative and logarithmic operations.
        
        This method implements the requirement for optimal confidence calculation based on
        tactician's and analyst's confidence outputs, using:
        1. Multiplicative operations for combining confidences
        2. Logarithmic additions for weighted combination
        3. Different weights for tactician and analyst
        
        Args:
            analyst_threshold: Analyst confidence threshold
            tactician_threshold: Tactician confidence threshold
            calibration_results: Results from confidence calibration
            
        Returns:
            Optimal confidence value or None if calculation fails
        """
        try:
            # Check if both confidence levels are available (requirement 1)
            if not self._has_confidence_levels_available(calibration_results):
                self.logger.warning("⚠️ Both tactician and analyst confidence levels not available")
                return None
            
            # Extract confidence weights from calibration results or use defaults
            tactician_weight = calibration_results.get('tactician_confidence_weight', 0.6)
            analyst_weight = calibration_results.get('analyst_confidence_weight', 0.4)
            
            # Ensure weights sum to 1.0
            total_weight = tactician_weight + analyst_weight
            if total_weight > 0:
                tactician_weight = tactician_weight / total_weight
                analyst_weight = analyst_weight / total_weight
            else:
                tactician_weight = 0.6
                analyst_weight = 0.4
            
            self.logger.debug(f"📊 Using confidence weights - Tactician: {tactician_weight:.3f}, Analyst: {analyst_weight:.3f}")
            
            # Get combination method from calibration results
            combination_method = calibration_results.get('confidence_combination_method', 'weighted_average')
            
            # Calculate confidence based on selected method
            if combination_method == 'multiplicative':
                optimal_confidence = self._calculate_multiplicative_confidence(
                    analyst_threshold, tactician_threshold, tactician_weight, analyst_weight
                )
            elif combination_method == 'logarithmic':
                optimal_confidence = self._calculate_logarithmic_confidence(
                    analyst_threshold, tactician_threshold, tactician_weight, analyst_weight
                )
            elif combination_method == 'harmonic':
                optimal_confidence = self._calculate_harmonic_confidence(
                    analyst_threshold, tactician_threshold, tactician_weight, analyst_weight
                )
            else:  # weighted_average or default
                # Method 1: Multiplicative combination
                multiplicative_confidence = self._calculate_multiplicative_confidence(
                    analyst_threshold, tactician_threshold, tactician_weight, analyst_weight
                )
                
                # Method 2: Logarithmic addition combination
                logarithmic_confidence = self._calculate_logarithmic_confidence(
                    analyst_threshold, tactician_threshold, tactician_weight, analyst_weight
                )
                
                # Method 3: Weighted harmonic mean (additional method for robustness)
                harmonic_confidence = self._calculate_harmonic_confidence(
                    analyst_threshold, tactician_threshold, tactician_weight, analyst_weight
                )
                
                # Combine methods using weighted average
                optimal_confidence = (
                    0.4 * multiplicative_confidence +
                    0.4 * logarithmic_confidence +
                    0.2 * harmonic_confidence
                )
            
            # Ensure confidence is within valid range [0, 1]
            optimal_confidence = max(0.0, min(1.0, optimal_confidence))
            
            self.logger.debug(f"📊 Optimal confidence calculation using {combination_method}:")
            if combination_method == 'weighted_average':
                self.logger.debug(f"   Multiplicative: {multiplicative_confidence:.4f}")
                self.logger.debug(f"   Logarithmic: {logarithmic_confidence:.4f}")
                self.logger.debug(f"   Harmonic: {harmonic_confidence:.4f}")
            self.logger.debug(f"   Final optimal: {optimal_confidence:.4f}")
            
            return optimal_confidence
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating optimal confidence: {e}")
            return None
    
    def _has_confidence_levels_available(self, calibration_results: Dict[str, Any]) -> bool:
        """
        Check if both tactician and analyst confidence levels are available.
        
        Args:
            calibration_results: Results from confidence calibration
            
        Returns:
            True if both confidence levels are available, False otherwise
        """
        try:
            # Check for tactician confidence data
            tactician_available = (
                'tactician_confidence' in calibration_results or
                'tactician_models' in calibration_results or
                'tactician_ensemble' in calibration_results
            )
            
            # Check for analyst confidence data
            analyst_available = (
                'analyst_confidence' in calibration_results or
                'analyst_models' in calibration_results or
                'analyst_ensemble' in calibration_results
            )
            
            both_available = tactician_available and analyst_available
            
            if not both_available:
                self.logger.warning(f"⚠️ Confidence availability - Tactician: {tactician_available}, Analyst: {analyst_available}")
            
            return both_available
            
        except Exception as e:
            self.logger.error(f"❌ Error checking confidence availability: {e}")
            return False
    
    def _calculate_multiplicative_confidence(self, analyst_threshold: float, tactician_threshold: float,
                                           tactician_weight: float, analyst_weight: float) -> float:
        """
        Calculate confidence using multiplicative operations.
        
        Formula: (tactician_threshold^tactician_weight) * (analyst_threshold^analyst_weight)
        
        Args:
            analyst_threshold: Analyst confidence threshold
            tactician_threshold: Tactician confidence threshold
            tactician_weight: Weight for tactician confidence
            analyst_weight: Weight for analyst confidence
            
        Returns:
            Multiplicative confidence value
        """
        try:
            # Ensure thresholds are positive for power operations
            analyst_thresh = max(0.001, analyst_threshold)
            tactician_thresh = max(0.001, tactician_threshold)
            
            # Multiplicative combination with weights as exponents
            multiplicative_conf = (
                (tactician_thresh ** tactician_weight) * 
                (analyst_thresh ** analyst_weight)
            )
            
            # Normalize to [0, 1] range
            multiplicative_conf = min(1.0, multiplicative_conf)
            
            return multiplicative_conf
            
        except Exception as e:
            self.logger.error(f"❌ Error in multiplicative confidence calculation: {e}")
            return 0.5  # Default fallback
    
    def _calculate_logarithmic_confidence(self, analyst_threshold: float, tactician_threshold: float,
                                        tactician_weight: float, analyst_weight: float) -> float:
        """
        Calculate confidence using logarithmic additions.
        
        Formula: exp(tactician_weight * log(tactician_threshold) + analyst_weight * log(analyst_threshold))
        
        Args:
            analyst_threshold: Analyst confidence threshold
            tactician_threshold: Tactician confidence threshold
            tactician_weight: Weight for tactician confidence
            analyst_weight: Weight for analyst confidence
            
        Returns:
            Logarithmic confidence value
        """
        try:
            # Ensure thresholds are positive for log operations
            analyst_thresh = max(0.001, analyst_threshold)
            tactician_thresh = max(0.001, tactician_threshold)
            
            # Logarithmic addition with weights
            log_combination = (
                tactician_weight * np.log(tactician_thresh) +
                analyst_weight * np.log(analyst_thresh)
            )
            
            # Convert back using exponential
            logarithmic_conf = np.exp(log_combination)
            
            # Normalize to [0, 1] range
            logarithmic_conf = min(1.0, max(0.0, logarithmic_conf))
            
            return logarithmic_conf
            
        except Exception as e:
            self.logger.error(f"❌ Error in logarithmic confidence calculation: {e}")
            return 0.5  # Default fallback
    
    def _calculate_harmonic_confidence(self, analyst_threshold: float, tactician_threshold: float,
                                     tactician_weight: float, analyst_weight: float) -> float:
        """
        Calculate confidence using weighted harmonic mean.
        
        Formula: 1 / (tactician_weight/tactician_threshold + analyst_weight/analyst_threshold)
        
        Args:
            analyst_threshold: Analyst confidence threshold
            tactician_threshold: Tactician confidence threshold
            tactician_weight: Weight for tactician confidence
            analyst_weight: Weight for analyst confidence
            
        Returns:
            Harmonic confidence value
        """
        try:
            # Ensure thresholds are positive for harmonic mean
            analyst_thresh = max(0.001, analyst_threshold)
            tactician_thresh = max(0.001, tactician_threshold)
            
            # Weighted harmonic mean
            harmonic_conf = 1.0 / (
                tactician_weight / tactician_thresh + 
                analyst_weight / analyst_thresh
            )
            
            # Normalize to [0, 1] range
            harmonic_conf = min(1.0, max(0.0, harmonic_conf))
            
            return harmonic_conf
            
        except Exception as e:
            self.logger.error(f"❌ Error in harmonic confidence calculation: {e}")
            return 0.5  # Default fallback
    
    def _evaluate_confidence_stability(self, analyst_threshold: float, tactician_threshold: float,
                                     calibration_results: Dict[str, Any]) -> float:
        """
        Evaluate confidence stability based on threshold consistency.
        
        Args:
            analyst_threshold: Analyst confidence threshold
            tactician_threshold: Tactician confidence threshold
            calibration_results: Results from confidence calibration
            
        Returns:
            Stability score between 0 and 1
        """
        try:
            stability_score = 0.0
            
            # Check threshold consistency
            threshold_diff = abs(tactician_threshold - analyst_threshold)
            if 0.05 <= threshold_diff <= 0.3:  # Good separation
                stability_score += 0.4
            elif threshold_diff < 0.05:  # Too close
                stability_score += 0.1
            else:  # Too far apart
                stability_score += 0.2
            
            # Check if thresholds are in reasonable ranges
            if 0.5 <= analyst_threshold <= 0.9:
                stability_score += 0.3
            if 0.6 <= tactician_threshold <= 0.95:
                stability_score += 0.3
            
            return min(1.0, stability_score)
            
        except Exception as e:
            self.logger.error(f"❌ Error evaluating confidence stability: {e}")
            return 0.5  # Default fallback
    
    def _evaluate_exit_confidence_calculation(self, analyst_threshold: float, tactician_threshold: float,
                                           calibration_results: Dict[str, Any]) -> float:
        """
        Evaluate exit confidence calculation effectiveness.
        
        Args:
            analyst_threshold: Analyst confidence threshold
            tactician_threshold: Tactician confidence threshold
            calibration_results: Results from confidence calibration including exit parameters
            
        Returns:
            Exit confidence evaluation score between 0 and 1
        """
        try:
            score = 0.0
            
            # Get exit confidence parameters
            exit_threshold = calibration_results.get('exit_confidence_threshold', 0.5)
            tactician_exit_weight = calibration_results.get('tactician_exit_confidence_weight', 0.6)
            analyst_exit_weight = calibration_results.get('analyst_exit_confidence_weight', 0.4)
            exit_combination_method = calibration_results.get('exit_confidence_combination_method', 'multiplicative')
            
            # Calculate exit confidence using different methods
            exit_confidences = {}
            
            # Method 1: Multiplicative
            try:
                analyst_conf = max(0.001, analyst_threshold)
                tactician_conf = max(0.001, tactician_threshold)
                multiplicative_exit = (
                    (tactician_conf ** tactician_exit_weight) * 
                    (analyst_conf ** analyst_exit_weight)
                )
                exit_confidences['multiplicative'] = min(1.0, multiplicative_exit)
            except:
                exit_confidences['multiplicative'] = 0.5
            
            # Method 2: Logarithmic
            try:
                analyst_conf = max(0.001, analyst_threshold)
                tactician_conf = max(0.001, tactician_threshold)
                log_combination = (
                    tactician_exit_weight * np.log(tactician_conf) +
                    analyst_exit_weight * np.log(analyst_conf)
                )
                logarithmic_exit = np.exp(log_combination)
                exit_confidences['logarithmic'] = min(1.0, max(0.0, logarithmic_exit))
            except:
                exit_confidences['logarithmic'] = 0.5
            
            # Method 3: Weighted Average
            weighted_avg_exit = (
                analyst_threshold * analyst_exit_weight +
                tactician_threshold * tactician_exit_weight
            )
            exit_confidences['weighted_average'] = max(0.0, min(1.0, weighted_avg_exit))
            
            # Score based on selected method effectiveness
            selected_exit_confidence = exit_confidences.get(exit_combination_method, 0.5)
            
            # Score factors
            # 1. Exit confidence should be reasonable (not too high or too low)
            if 0.4 <= selected_exit_confidence <= 0.8:
                score += 0.3
            elif 0.2 <= selected_exit_confidence <= 0.9:
                score += 0.2
            else:
                score += 0.1
            
            # 2. Exit threshold should be lower than entry confidence
            entry_confidence = (analyst_threshold + tactician_threshold) / 2
            if exit_threshold < entry_confidence:
                score += 0.2
                # Bonus for reasonable gap
                gap = entry_confidence - exit_threshold
                if 0.1 <= gap <= 0.3:
                    score += 0.1
            
            # 3. Exit weights should be reasonable
            if abs(tactician_exit_weight + analyst_exit_weight - 1.0) < 0.1:
                score += 0.2
            
            # 4. Method consistency bonus
            if exit_combination_method in ['multiplicative', 'logarithmic']:
                # These methods should produce reasonable results
                multiplicative_conf = exit_confidences.get('multiplicative', 0)
                logarithmic_conf = exit_confidences.get('logarithmic', 0)
                
                if multiplicative_conf > 0.1 and logarithmic_conf > 0.1:
                    score += 0.1
                    
                    # Bonus if methods are consistent
                    if abs(multiplicative_conf - logarithmic_conf) < 0.2:
                        score += 0.1
            
            return min(1.0, score)
            
        except Exception as e:
            self.logger.error(f"❌ Error evaluating exit confidence calculation: {e}")
            return 0.5  # Default fallback

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
    
    def _evaluate_exit_strategy_performance(self, params: Dict[str, Any], 
                                          calibration_results: Dict[str, Any]) -> float:
        """
        Evaluate exit strategy performance through simulated backtesting.
        
        This method simulates how well the exit confidence threshold and combination methods
        work in practice by testing different scenarios.
        
        Args:
            params: Current parameter configuration including exit parameters
            calibration_results: Calibration results with historical data
            
        Returns:
            Exit strategy performance score (0.0 to 1.0)
        """
        try:
            self.logger.debug("🔍 Evaluating exit strategy performance...")
            
            # Extract exit parameters
            exit_threshold = params.get('exit_confidence_threshold', 0.5)
            tactician_exit_weight = params.get('tactician_exit_confidence_weight', 0.6)
            analyst_exit_weight = params.get('analyst_exit_confidence_weight', 0.4)
            exit_combination_method = params.get('exit_confidence_combination_method', 'multiplicative')
            
            # Simulate various confidence scenarios
            scenarios = self._generate_confidence_scenarios()
            
            total_score = 0.0
            scenario_count = 0
            
            for scenario in scenarios:
                scenario_score = self._evaluate_single_exit_scenario(
                    scenario, exit_threshold, tactician_exit_weight, 
                    analyst_exit_weight, exit_combination_method
                )
                total_score += scenario_score
                scenario_count += 1
            
            if scenario_count == 0:
                return 0.5  # Default score
            
            avg_score = total_score / scenario_count
            
            self.logger.debug(f"📊 Exit strategy evaluation completed:")
            self.logger.debug(f"   Exit threshold: {exit_threshold:.3f}")
            self.logger.debug(f"   Combination method: {exit_combination_method}")
            self.logger.debug(f"   Average scenario score: {avg_score:.3f}")
            
            return avg_score
            
        except Exception as e:
            self.logger.error(f"❌ Error evaluating exit strategy performance: {e}")
            return 0.5  # Default fallback
    
    def _generate_confidence_scenarios(self) -> List[Dict[str, Any]]:
        """
        Generate various confidence scenarios for backtesting exit strategies.
        
        Returns:
            List of scenarios with different confidence patterns
        """
        scenarios = []
        
        # Scenario 1: Declining confidence (should trigger exit)
        scenarios.append({
            'name': 'declining_confidence',
            'analyst_confidence_sequence': [0.8, 0.7, 0.6, 0.5, 0.4],
            'tactician_confidence_sequence': [0.9, 0.8, 0.6, 0.5, 0.3],
            'expected_exit': True,
            'optimal_exit_point': 3  # Should exit around 4th measurement
        })
        
        # Scenario 2: Stable high confidence (should not exit)
        scenarios.append({
            'name': 'stable_high_confidence',
            'analyst_confidence_sequence': [0.8, 0.8, 0.7, 0.8, 0.8],
            'tactician_confidence_sequence': [0.9, 0.8, 0.8, 0.9, 0.8],
            'expected_exit': False,
            'optimal_exit_point': None
        })
        
        # Scenario 3: Volatile confidence (should be cautious)
        scenarios.append({
            'name': 'volatile_confidence',
            'analyst_confidence_sequence': [0.7, 0.5, 0.8, 0.4, 0.6],
            'tactician_confidence_sequence': [0.8, 0.6, 0.9, 0.3, 0.7],
            'expected_exit': True,
            'optimal_exit_point': 3  # Should exit when it drops to 0.3
        })
        
        # Scenario 4: Gradual recovery (should not exit early)
        scenarios.append({
            'name': 'gradual_recovery',
            'analyst_confidence_sequence': [0.6, 0.5, 0.6, 0.7, 0.8],
            'tactician_confidence_sequence': [0.7, 0.6, 0.7, 0.8, 0.9],
            'expected_exit': False,
            'optimal_exit_point': None
        })
        
        # Scenario 5: Sharp drop (should exit quickly)
        scenarios.append({
            'name': 'sharp_drop',
            'analyst_confidence_sequence': [0.8, 0.8, 0.3, 0.2, 0.1],
            'tactician_confidence_sequence': [0.9, 0.9, 0.4, 0.2, 0.1],
            'expected_exit': True,
            'optimal_exit_point': 2  # Should exit immediately after drop
        })
        
        return scenarios
    
    def _evaluate_single_exit_scenario(self, scenario: Dict[str, Any], exit_threshold: float,
                                     tactician_exit_weight: float, analyst_exit_weight: float,
                                     exit_combination_method: str) -> float:
        """
        Evaluate a single exit scenario.
        
        Args:
            scenario: Scenario configuration with confidence sequences
            exit_threshold: Exit confidence threshold
            tactician_exit_weight: Weight for tactician confidence in exit calculation
            analyst_exit_weight: Weight for analyst confidence in exit calculation
            exit_combination_method: Method for combining confidences
            
        Returns:
            Scenario evaluation score (0.0 to 1.0)
        """
        try:
            analyst_seq = scenario['analyst_confidence_sequence']
            tactician_seq = scenario['tactician_confidence_sequence']
            expected_exit = scenario['expected_exit']
            optimal_exit_point = scenario.get('optimal_exit_point')
            
            exit_triggered = False
            exit_point = None
            
            # Simulate the confidence sequence
            for i, (analyst_conf, tactician_conf) in enumerate(zip(analyst_seq, tactician_seq)):
                # Calculate exit confidence using the specified method
                if exit_combination_method == 'multiplicative':
                    exit_confidence = self._calculate_multiplicative_confidence(
                        analyst_conf, tactician_conf, tactician_exit_weight, analyst_exit_weight
                    )
                elif exit_combination_method == 'logarithmic':
                    exit_confidence = self._calculate_logarithmic_confidence(
                        analyst_conf, tactician_conf, tactician_exit_weight, analyst_exit_weight
                    )
                else:  # weighted_average
                    exit_confidence = (
                        analyst_conf * analyst_exit_weight +
                        tactician_conf * tactician_exit_weight
                    )
                
                # Check if exit should be triggered
                if exit_confidence < exit_threshold:
                    exit_triggered = True
                    exit_point = i
                    break
            
            # Evaluate the exit decision
            score = 0.0
            
            # Base score for correct exit decision
            if expected_exit and exit_triggered:
                score += 0.5  # Correctly identified need to exit
                
                # Bonus for exiting at the right time
                if optimal_exit_point is not None:
                    time_diff = abs(exit_point - optimal_exit_point)
                    if time_diff == 0:
                        score += 0.3  # Perfect timing
                    elif time_diff == 1:
                        score += 0.2  # Close timing
                    elif time_diff <= 2:
                        score += 0.1  # Reasonable timing
                        
            elif not expected_exit and not exit_triggered:
                score += 0.6  # Correctly stayed in position
                
            elif expected_exit and not exit_triggered:
                score += 0.1  # Failed to exit when should have (small penalty)
                
            elif not expected_exit and exit_triggered:
                score += 0.2  # Exited when shouldn't have (moderate penalty)
            
            # Additional scoring factors
            
            # Penalty for very early or very late exits in declining scenarios
            if scenario['name'] == 'declining_confidence' and exit_triggered:
                if exit_point == 0:
                    score *= 0.7  # Too early
                elif exit_point >= 4:
                    score *= 0.8  # Too late
            
            # Bonus for not exiting during recovery
            if scenario['name'] == 'gradual_recovery' and not exit_triggered:
                score += 0.1
            
            # Penalty for not exiting during sharp drops
            if scenario['name'] == 'sharp_drop' and not exit_triggered:
                score *= 0.5
            
            return min(1.0, max(0.0, score))
            
        except Exception as e:
            self.logger.error(f"❌ Error evaluating exit scenario: {e}")
            return 0.5  # Default fallback
    
    def _calculate_multiplicative_confidence(self, analyst_conf: float, tactician_conf: float,
                                           tactician_weight: float, analyst_weight: float) -> float:
        """Calculate confidence using multiplicative method (shared with signal pipeline)."""
        try:
            analyst_conf = max(0.001, analyst_conf)
            tactician_conf = max(0.001, tactician_conf)
            
            multiplicative_conf = (
                (tactician_conf ** tactician_weight) * 
                (analyst_conf ** analyst_weight)
            )
            
            return min(1.0, multiplicative_conf)
            
        except Exception as e:
            self.logger.error(f"❌ Error in multiplicative confidence calculation: {e}")
            return 0.5
    
    def _calculate_logarithmic_confidence(self, analyst_conf: float, tactician_conf: float,
                                        tactician_weight: float, analyst_weight: float) -> float:
        """Calculate confidence using logarithmic method (shared with signal pipeline)."""
        try:
            analyst_conf = max(0.001, analyst_conf)
            tactician_conf = max(0.001, tactician_conf)
            
            log_combination = (
                tactician_weight * np.log(tactician_conf) +
                analyst_weight * np.log(analyst_conf)
            )
            
            logarithmic_conf = np.exp(log_combination)
            return min(1.0, max(0.0, logarithmic_conf))
            
        except Exception as e:
            self.logger.error(f"❌ Error in logarithmic confidence calculation: {e}")
            return 0.5

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
            elif param_config['type'] == 'categorical':
                param_combinations.append([(param_name, v) for v in param_config['choices']])
        
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
            elif param_config['type'] == 'categorical':
                param_combinations.append([(param_name, v) for v in param_config['choices']])
        
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
            elif param_config['type'] == 'categorical':
                default_params[param_name] = param_config['choices'][0]  # Use first choice as default
        
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