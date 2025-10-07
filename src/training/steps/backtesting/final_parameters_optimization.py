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
from pathlib import Path
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
        
        # Parameter categories for optimization (updated for new Analyst & Tactician models)
        self.categories = [
            'confidence', 'intensity', 'position_sizing', 'leverage', 'tpsl', 'exit_strategy',
            'ensemble', 'sr', 'two_tier', 'technical_indicators',
            'system_monitoring', 'training_optimization', 'regime_transitions',
            'signal_aggregation', 'turnover_cost_penalty', 'entry_timing_optimization', 
            'confidence_aware_ensemble', 'model_specific_parameters',
            # New directional categories
            'long_specific_parameters', 'short_specific_parameters', 
            'directional_thresholds', 'asymmetric_risk_management',
            # Merged Tactician & Analyst integration
            'tactician_analyst_integration', 'analyst_oof_weights', 'merged_feature_importance'
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

        # Load per-regime performance statistics for objective adjustments
        self.regime_performance_path: Optional[str] = None
        self.regime_performance_stats = self._load_regime_performance_stats()
        self.regime_performance_modifier = self._calculate_regime_performance_modifier()
        if self.regime_performance_stats:
            location = self.regime_performance_path or 'unknown location'
            self.logger.info(f"📊 Loaded per-regime performance stats from {location}")
            self.logger.info(f"   • Regime performance modifier: {self.regime_performance_modifier:.4f}")
    
        # Initialize artifact and version managers
        self.artifact_manager = get_artifact_manager()
        self.pickup_utils = get_artifact_pickup_utils()
        self.version_manager = get_version_manager()


class AsymmetricParametersOptimizer(FinalParametersOptimizer):
    """Enhanced optimizer with long/short parameter differentiation"""
    
    def __init__(self, config: Dict[str, Any], nonlinear_config: Optional[NonLinearConfig] = None):
        super().__init__(config, nonlinear_config)
        
        # Enhanced search spaces with directional parameters
        self.directional_search_spaces = self._get_directional_search_spaces()
        self.default_search_spaces.update(self.directional_search_spaces)
        
        # Re-create enhanced search spaces with new directional parameters
        self.enhanced_search_spaces = self._create_enhanced_search_spaces()
        
        self.logger.info("🎯 Asymmetric Parameters Optimizer initialized")
        self.logger.info(f"   Added directional parameter categories: {len(self.directional_search_spaces)}")
        
    def _get_directional_search_spaces(self):
        """Define search spaces for directional parameters"""
        return {
            'long_specific_parameters': {
                'long_entry_patience': {'type': 'float', 'low': 0.5, 'high': 2.0},
                'long_profit_target_multiplier': {'type': 'float', 'low': 0.8, 'high': 1.5},
                'long_stop_loss_multiplier': {'type': 'float', 'low': 0.8, 'high': 1.2},
                'long_position_size_multiplier': {'type': 'float', 'low': 0.8, 'high': 1.3},
                'long_confidence_threshold': {'type': 'float', 'low': 0.5, 'high': 0.8},
                'long_momentum_weight': {'type': 'float', 'low': 0.1, 'high': 0.6},
                'long_support_weight': {'type': 'float', 'low': 0.1, 'high': 0.5},
            },
            'short_specific_parameters': {
                'short_entry_urgency': {'type': 'float', 'low': 0.8, 'high': 1.5},
                'short_profit_target_multiplier': {'type': 'float', 'low': 0.6, 'high': 1.2},
                'short_stop_loss_multiplier': {'type': 'float', 'low': 1.0, 'high': 1.4},
                'short_position_size_multiplier': {'type': 'float', 'low': 0.7, 'high': 1.1},
                'short_confidence_threshold': {'type': 'float', 'low': 0.6, 'high': 0.85},
                'short_momentum_weight': {'type': 'float', 'low': 0.2, 'high': 0.7},
                'short_resistance_weight': {'type': 'float', 'low': 0.1, 'high': 0.5},
            },
            'directional_thresholds': {
                'long_vs_short_bias_threshold': {'type': 'float', 'low': 0.1, 'high': 0.4},
                'directional_confidence_weight': {'type': 'float', 'low': 0.1, 'high': 0.5},
                'asymmetric_volatility_adjustment': {'type': 'float', 'low': 0.8, 'high': 1.3},
                'directional_switch_penalty': {'type': 'float', 'low': 0.0, 'high': 0.1},
                'long_bias_boost': {'type': 'float', 'low': 0.9, 'high': 1.2},
                'short_bias_boost': {'type': 'float', 'low': 0.9, 'high': 1.2},
            },
            'asymmetric_risk_management': {
                'long_max_position_duration': {'type': 'int', 'low': 20, 'high': 40},
                'short_max_position_duration': {'type': 'int', 'low': 10, 'high': 25},
                'long_reassessment_frequency': {'type': 'int', 'low': 3, 'high': 8},
                'short_reassessment_frequency': {'type': 'int', 'low': 2, 'high': 5},
                'long_volatility_tolerance': {'type': 'float', 'low': 0.8, 'high': 1.1},
                'short_volatility_tolerance': {'type': 'float', 'low': 1.0, 'high': 1.3},
                'asymmetric_leverage_adjustment': {'type': 'float', 'low': 0.9, 'high': 1.1},
            },
            # Merged Tactician & Analyst integration parameters
            'tactician_analyst_integration': {
                'w_min': {'type': 'float', 'min': 0.1, 'max': 0.5},  # Minimum weight for sample weighting
                'analyst_feature_weight': {'type': 'float', 'min': 0.1, 'max': 1.0},  # Weight for Analyst OOF features
                'p_trade_weight': {'type': 'float', 'min': 0.2, 'max': 0.8},  # Weight for p_trade feature
                'u_trade_weight': {'type': 'float', 'min': 0.2, 'max': 0.8},  # Weight for u_trade feature
                'q_trade_weight': {'type': 'float', 'min': 0.2, 'max': 0.8},  # Weight for q_trade feature
                'analyst_expected_value_weight': {'type': 'float', 'min': 0.1, 'max': 0.6},  # Weight for expected value feature
                'analyst_weighted_prob_weight': {'type': 'float', 'min': 0.1, 'max': 0.6},  # Weight for weighted prob feature
                'integration_method': {'type': 'categorical', 'choices': ['additive', 'multiplicative', 'ensemble']},
                'feature_interaction_strength': {'type': 'float', 'min': 0.1, 'max': 1.0},  # Strength of feature interactions
            },
            'analyst_oof_weights': {
                'p_trade_threshold': {'type': 'float', 'min': 0.3, 'max': 0.8},  # Threshold for p_trade filtering
                'u_trade_threshold': {'type': 'float', 'min': -0.5, 'max': 0.5},  # Threshold for u_trade filtering
                'q_trade_threshold': {'type': 'float', 'min': 0.4, 'max': 0.9},  # Threshold for q_trade filtering
                'weight_scaling_factor': {'type': 'float', 'min': 0.5, 'max': 2.0},  # Scaling factor for weights
                'weight_smoothing': {'type': 'float', 'min': 0.0, 'max': 0.5},  # Smoothing factor for weights
                'adaptive_weighting': {'type': 'categorical', 'choices': ['static', 'dynamic', 'regime_based']},
            },
            'merged_feature_importance': {
                'analyst_feature_importance_boost': {'type': 'float', 'min': 1.0, 'max': 3.0},  # Boost for Analyst features
                'interaction_feature_importance': {'type': 'float', 'min': 0.5, 'max': 2.0},  # Importance of interaction features
                'feature_selection_threshold': {'type': 'float', 'min': 0.01, 'max': 0.1},  # Threshold for feature selection
                'analyst_feature_regularization': {'type': 'float', 'min': 0.0, 'max': 0.1},  # Regularization for Analyst features
                'feature_interaction_depth': {'type': 'int', 'min': 1, 'max': 3},  # Depth of feature interactions
            }
        }
    
    def optimize_per_regime_with_direction(self, regime_data: Dict[str, Any], regime_id: str):
        """
        Optimize parameters per regime with directional differentiation
        
        Args:
            regime_data: Data for specific regime including signals, directions, returns, etc.
            regime_id: Regime identifier
        """
        
        # Check if regime has enough samples for directional split
        total_samples = len(regime_data.get('signals', []))
        directions = regime_data.get('directions', np.array([]))
        
        if len(directions) == 0:
            self.logger.warning(f"⚠️ Regime {regime_id}: No direction data available, using standard optimization")
            return self.optimize_regime_parameters(regime_data, regime_id)
        
        long_samples = np.sum(directions > 0)
        short_samples = np.sum(directions < 0)
        
        min_samples_per_direction = self.config.get('min_samples_per_direction', 100)
        
        if long_samples >= min_samples_per_direction and short_samples >= min_samples_per_direction:
            # Sufficient samples: optimize separately
            self.logger.info(f"📊 Regime {regime_id}: Sufficient samples for directional optimization")
            self.logger.info(f"   Long samples: {long_samples}, Short samples: {short_samples}")
            
            return self._optimize_directional_parameters(regime_data, regime_id)
        
        else:
            # Insufficient samples: use averaged parameters with directional bias
            self.logger.info(f"📊 Regime {regime_id}: Using averaged parameters with directional bias")
            self.logger.info(f"   Long samples: {long_samples}, Short samples: {short_samples}")
            
            return self._optimize_averaged_parameters_with_bias(regime_data, regime_id)
    
    def _optimize_directional_parameters(self, regime_data: Dict[str, Any], regime_id: str):
        """Optimize separate parameters for longs and shorts"""
        
        results = {}
        
        # Separate data by direction
        directions = regime_data['directions']
        long_mask = directions > 0
        short_mask = directions < 0
        
        # Optimize long parameters
        long_data = self._filter_data_by_mask(regime_data, long_mask)
        
        long_study = optuna.create_study(
            direction='maximize',
            study_name=f'{self.study_name}_regime_{regime_id}_long',
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        long_objective = self._create_directional_objective(long_data, 'long', regime_id)
        
        try:
            long_study.optimize(long_objective, n_trials=self.n_trials // 2, timeout=self.timeout // 2)
            results['long_parameters'] = long_study.best_params
            results['long_score'] = long_study.best_value
            results['long_trials'] = len(long_study.trials)
        except Exception as e:
            self.logger.error(f"❌ Long parameter optimization failed for regime {regime_id}: {e}")
            results['long_parameters'] = {}
            results['long_score'] = 0.0
            results['long_trials'] = 0
        
        # Optimize short parameters
        short_data = self._filter_data_by_mask(regime_data, short_mask)
        
        short_study = optuna.create_study(
            direction='maximize',
            study_name=f'{self.study_name}_regime_{regime_id}_short',
            sampler=optuna.samplers.TPESampler(seed=43)
        )
        
        short_objective = self._create_directional_objective(short_data, 'short', regime_id)
        
        try:
            short_study.optimize(short_objective, n_trials=self.n_trials // 2, timeout=self.timeout // 2)
            results['short_parameters'] = short_study.best_params
            results['short_score'] = short_study.best_value
            results['short_trials'] = len(short_study.trials)
        except Exception as e:
            self.logger.error(f"❌ Short parameter optimization failed for regime {regime_id}: {e}")
            results['short_parameters'] = {}
            results['short_score'] = 0.0
            results['short_trials'] = 0
        
        # Create combined parameters
        results['combined_parameters'] = self._combine_directional_parameters(
            results.get('long_parameters', {}), 
            results.get('short_parameters', {})
        )
        
        # Calculate weighted score
        total_trials = results['long_trials'] + results['short_trials']
        if total_trials > 0:
            results['combined_score'] = (
                (results['long_score'] * results['long_trials'] + 
                 results['short_score'] * results['short_trials']) / total_trials
            )
        else:
            results['combined_score'] = 0.0
        
        self.logger.info(f"✅ Directional optimization completed for regime {regime_id}")
        self.logger.info(f"   Long score: {results['long_score']:.4f} ({results['long_trials']} trials)")
        self.logger.info(f"   Short score: {results['short_score']:.4f} ({results['short_trials']} trials)")
        self.logger.info(f"   Combined score: {results['combined_score']:.4f}")
        
        return results
    
    def _optimize_averaged_parameters_with_bias(self, regime_data: Dict[str, Any], regime_id: str):
        """Optimize averaged parameters with directional bias when samples are insufficient"""
        
        # Calculate directional bias
        directions = regime_data['directions']
        long_ratio = np.sum(directions > 0) / len(directions)
        short_ratio = np.sum(directions < 0) / len(directions)
        directional_bias = 'long' if long_ratio > short_ratio else 'short'
        bias_strength = abs(long_ratio - short_ratio)
        
        self.logger.info(f"   Directional bias: {directional_bias} (strength: {bias_strength:.2f})")
        self.logger.info(f"   Long ratio: {long_ratio:.1%}, Short ratio: {short_ratio:.1%}")
        
        # Create biased objective function
        study = optuna.create_study(
            direction='maximize',
            study_name=f'{self.study_name}_regime_{regime_id}_averaged',
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        biased_objective = self._create_biased_objective(
            regime_data, directional_bias, long_ratio, short_ratio, regime_id
        )
        
        try:
            study.optimize(biased_objective, n_trials=self.n_trials, timeout=self.timeout)
            base_parameters = study.best_params
            base_score = study.best_value
            trials_completed = len(study.trials)
        except Exception as e:
            self.logger.error(f"❌ Biased parameter optimization failed for regime {regime_id}: {e}")
            base_parameters = {}
            base_score = 0.0
            trials_completed = 0
        
        # Apply directional bias to parameters
        biased_parameters = self._apply_directional_bias(
            base_parameters, directional_bias, long_ratio, short_ratio
        )
        
        results = {
            'base_parameters': base_parameters,
            'biased_parameters': biased_parameters,
            'directional_bias': directional_bias,
            'bias_strength': bias_strength,
            'long_ratio': long_ratio,
            'short_ratio': short_ratio,
            'score': base_score,
            'trials_completed': trials_completed
        }
        
        self.logger.info(f"✅ Biased optimization completed for regime {regime_id}")
        self.logger.info(f"   Base score: {base_score:.4f} ({trials_completed} trials)")
        self.logger.info(f"   Bias applied: {directional_bias} (strength: {bias_strength:.2f})")
        
        return results
    
    def _filter_data_by_mask(self, data: Dict[str, Any], mask: np.ndarray) -> Dict[str, Any]:
        """Filter regime data by directional mask"""
        filtered_data = {}
        
        for key, value in data.items():
            if isinstance(value, np.ndarray) and len(value) == len(mask):
                filtered_data[key] = value[mask]
            elif isinstance(value, list) and len(value) == len(mask):
                filtered_data[key] = [value[i] for i in range(len(value)) if mask[i]]
            else:
                # Keep non-array data as-is
                filtered_data[key] = value
                
        return filtered_data
    
    def _create_directional_objective(self, data: Dict[str, Any], direction: str, regime_id: str):
        """Create objective function for specific direction"""
        
        def objective(trial):
            try:
                # Sample directional parameters
                params = {}
                
                # Sample direction-specific parameters
                direction_space = self.directional_search_spaces[f'{direction}_specific_parameters']
                for param_name, param_config in direction_space.items():
                    if param_config['type'] == 'float':
                        params[param_name] = trial.suggest_float(
                            param_name, param_config['low'], param_config['high']
                        )
                    elif param_config['type'] == 'int':
                        params[param_name] = trial.suggest_int(
                            param_name, param_config['low'], param_config['high']
                        )
                
                # Sample general directional parameters
                for category in ['directional_thresholds', 'asymmetric_risk_management']:
                    if category in self.directional_search_spaces:
                        category_space = self.directional_search_spaces[category]
                        for param_name, param_config in category_space.items():
                            if param_config['type'] == 'float':
                                params[param_name] = trial.suggest_float(
                                    param_name, param_config['low'], param_config['high']
                                )
                            elif param_config['type'] == 'int':
                                params[param_name] = trial.suggest_int(
                                    param_name, param_config['low'], param_config['high']
                                )
                
                # Sample some general parameters with directional adjustments
                general_params = self._sample_general_parameters_with_direction(trial, direction)
                params.update(general_params)
                
                # Evaluate performance with these parameters
                performance = self._evaluate_directional_performance(data, params, direction, regime_id)
                
                return performance
                
            except Exception as e:
                self.logger.error(f"❌ Objective evaluation failed: {e}")
                return 0.0  # Return poor score on failure
        
        return objective
    
    def _create_biased_objective(self, data: Dict[str, Any], bias: str, long_ratio: float, 
                                short_ratio: float, regime_id: str):
        """Create objective function with directional bias"""
        
        def objective(trial):
            try:
                # Sample base parameters
                params = self._sample_base_parameters(trial)
                
                # Apply directional bias during sampling
                biased_params = self._apply_directional_bias_to_sampling(
                    params, trial, bias, long_ratio, short_ratio
                )
                
                # Evaluate performance with biased parameters
                performance = self._evaluate_biased_performance(
                    data, biased_params, bias, long_ratio, short_ratio, regime_id
                )
                
                return performance
                
            except Exception as e:
                self.logger.error(f"❌ Biased objective evaluation failed: {e}")
                return 0.0  # Return poor score on failure
        
        return objective
    
    def _sample_general_parameters_with_direction(self, trial, direction: str) -> Dict[str, Any]:
        """Sample general parameters with directional adjustments"""
        params = {}
        
        # Adjust confidence thresholds based on direction
        if direction == 'long':
            params['confidence_threshold'] = trial.suggest_float('confidence_threshold', 0.5, 0.75)
            params['position_size_base'] = trial.suggest_float('position_size_base', 0.008, 0.015)
        else:  # short
            params['confidence_threshold'] = trial.suggest_float('confidence_threshold', 0.6, 0.85)
            params['position_size_base'] = trial.suggest_float('position_size_base', 0.006, 0.012)
        
        # Direction-agnostic parameters
        params['leverage_multiplier'] = trial.suggest_float('leverage_multiplier', 0.8, 1.2)
        params['risk_adjustment'] = trial.suggest_float('risk_adjustment', 0.9, 1.1)
        
        return params
    
    def _sample_base_parameters(self, trial) -> Dict[str, Any]:
        """Sample base parameters without directional bias"""
        return {
            'confidence_threshold': trial.suggest_float('confidence_threshold', 0.5, 0.8),
            'position_size_base': trial.suggest_float('position_size_base', 0.005, 0.015),
            'leverage_multiplier': trial.suggest_float('leverage_multiplier', 0.8, 1.2),
            'risk_adjustment': trial.suggest_float('risk_adjustment', 0.9, 1.1),
            'profit_target_multiplier': trial.suggest_float('profit_target_multiplier', 0.8, 1.4),
            'stop_loss_multiplier': trial.suggest_float('stop_loss_multiplier', 0.8, 1.3),
        }
    
    def _apply_directional_bias_to_sampling(self, params: Dict[str, Any], trial, bias: str, 
                                          long_ratio: float, short_ratio: float) -> Dict[str, Any]:
        """Apply directional bias during parameter sampling"""
        biased_params = params.copy()
        
        # Sample directional adjustment factors
        bias_adjustment = trial.suggest_float('bias_adjustment', 0.9, 1.1)
        
        if bias == 'long':
            # Long-friendly adjustments
            biased_params['confidence_threshold'] *= 0.95  # Slightly lower
            biased_params['position_size_base'] *= bias_adjustment * 1.05  # Slightly larger
            biased_params['profit_target_multiplier'] *= 1.1  # Higher profit targets
        else:  # short
            # Short-friendly adjustments
            biased_params['confidence_threshold'] *= 1.05  # Slightly higher
            biased_params['position_size_base'] *= bias_adjustment * 0.95  # Slightly smaller
            biased_params['stop_loss_multiplier'] *= 1.1  # Tighter stops
        
        return biased_params
    
    def _evaluate_directional_performance(self, data: Dict[str, Any], params: Dict[str, Any], 
                                        direction: str, regime_id: str) -> float:
        """Evaluate performance for specific direction"""
        try:
            # Extract relevant data
            signals = data.get('signals', np.array([]))
            returns = data.get('returns', np.array([]))
            directions = data.get('directions', np.array([]))
            
            if len(signals) == 0 or len(returns) == 0:
                return 0.0
            
            # Apply directional parameters to simulate performance
            confidence_threshold = params.get('confidence_threshold', 0.6)
            position_size = params.get('position_size_base', 0.01)
            
            # Filter signals by confidence
            confident_signals = signals >= confidence_threshold
            
            # Calculate directional returns
            directional_returns = returns[confident_signals] * position_size
            
            if len(directional_returns) == 0:
                return 0.0
            
            # Direction-specific performance metrics
            if direction == 'long':
                # For longs: reward sustained positive returns
                performance = np.mean(directional_returns) * np.sqrt(len(directional_returns))
                # Bonus for consistency
                if np.std(directional_returns) > 0:
                    sharpe_bonus = np.mean(directional_returns) / np.std(directional_returns) * 0.1
                    performance += sharpe_bonus
            else:  # short
                # For shorts: reward quick, sharp negative moves (positive returns for short positions)
                performance = np.mean(directional_returns) * np.sqrt(len(directional_returns))
                # Bonus for capturing volatility
                volatility_bonus = np.std(directional_returns) * 0.05
                performance += volatility_bonus
            
            # Apply risk adjustment
            risk_adjustment = params.get('risk_adjustment', 1.0)
            performance *= risk_adjustment
            
            return max(0.0, performance)  # Ensure non-negative
            
        except Exception as e:
            self.logger.error(f"❌ Directional performance evaluation failed: {e}")
            return 0.0
    
    def _evaluate_biased_performance(self, data: Dict[str, Any], params: Dict[str, Any], 
                                   bias: str, long_ratio: float, short_ratio: float, 
                                   regime_id: str) -> float:
        """Evaluate performance with directional bias"""
        try:
            # Extract data
            signals = data.get('signals', np.array([]))
            returns = data.get('returns', np.array([]))
            directions = data.get('directions', np.array([]))
            
            if len(signals) == 0 or len(returns) == 0:
                return 0.0
            
            # Apply parameters
            confidence_threshold = params.get('confidence_threshold', 0.6)
            position_size = params.get('position_size_base', 0.01)
            
            # Filter signals
            confident_signals = signals >= confidence_threshold
            filtered_returns = returns[confident_signals]
            filtered_directions = directions[confident_signals]
            
            if len(filtered_returns) == 0:
                return 0.0
            
            # Calculate weighted performance based on directional bias
            long_mask = filtered_directions > 0
            short_mask = filtered_directions < 0
            
            performance = 0.0
            
            if np.any(long_mask):
                long_returns = filtered_returns[long_mask] * position_size
                long_performance = np.mean(long_returns) * np.sqrt(len(long_returns))
                performance += long_performance * long_ratio
            
            if np.any(short_mask):
                short_returns = filtered_returns[short_mask] * position_size
                short_performance = np.mean(short_returns) * np.sqrt(len(short_returns))
                performance += short_performance * short_ratio
            
            # Apply bias boost
            bias_strength = abs(long_ratio - short_ratio)
            bias_boost = 1.0 + (bias_strength * 0.1)  # Up to 10% boost for strong bias
            performance *= bias_boost
            
            return max(0.0, performance)
            
        except Exception as e:
            self.logger.error(f"❌ Biased performance evaluation failed: {e}")
            return 0.0
    
    def _combine_directional_parameters(self, long_params: Dict[str, Any], 
                                      short_params: Dict[str, Any]) -> Dict[str, Any]:
        """Combine long and short parameters into unified set"""
        combined = {}
        
        # Combine parameters with directional prefixes
        for key, value in long_params.items():
            if not key.startswith('long_'):
                combined[f'long_{key}'] = value
            else:
                combined[key] = value
        
        for key, value in short_params.items():
            if not key.startswith('short_'):
                combined[f'short_{key}'] = value
            else:
                combined[key] = value
        
        # Create averaged parameters for general use
        general_params = {}
        for long_key, long_value in long_params.items():
            if long_key.startswith('long_'):
                base_key = long_key[5:]  # Remove 'long_' prefix
                short_key = f'short_{base_key}'
                if short_key in short_params:
                    general_params[base_key] = (long_value + short_params[short_key]) / 2
        
        combined.update(general_params)
        
        return combined
    
    def _apply_directional_bias(self, base_params: Dict[str, Any], bias: str, 
                              long_ratio: float, short_ratio: float) -> Dict[str, Any]:
        """Apply directional bias to base parameters"""
        
        biased_params = base_params.copy()
        bias_strength = abs(long_ratio - short_ratio)
        
        if bias == 'long':
            # Bias towards long-friendly parameters
            biased_params['confidence_threshold'] = biased_params.get('confidence_threshold', 0.6) * (1 - bias_strength * 0.1)
            biased_params['position_size_base'] = biased_params.get('position_size_base', 0.01) * (1 + bias_strength * 0.2)
            biased_params['profit_target_multiplier'] = biased_params.get('profit_target_multiplier', 1.0) * (1 + bias_strength * 0.3)
            biased_params['max_position_duration'] = int(biased_params.get('max_position_duration', 25) * (1 + bias_strength * 0.4))
            
        else:  # short bias
            # Bias towards short-friendly parameters  
            biased_params['confidence_threshold'] = biased_params.get('confidence_threshold', 0.6) * (1 + bias_strength * 0.1)
            biased_params['position_size_base'] = biased_params.get('position_size_base', 0.01) * (1 - bias_strength * 0.1)
            biased_params['stop_loss_multiplier'] = biased_params.get('stop_loss_multiplier', 1.0) * (1 + bias_strength * 0.2)
            biased_params['max_position_duration'] = int(biased_params.get('max_position_duration', 25) * (1 - bias_strength * 0.3))
        
        # Add directional metadata
        biased_params['directional_bias'] = bias
        biased_params['bias_strength'] = bias_strength
        biased_params['long_ratio'] = long_ratio
        biased_params['short_ratio'] = short_ratio
        
        return biased_params
    
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
                # 0.3% Micro Movement Entry Thresholds (immediate only)
                'micro_immediate_long_threshold': {'type': 'float', 'min': 0.65, 'max': 0.85},
                'micro_immediate_short_threshold': {'type': 'float', 'min': 0.68, 'max': 0.88},
                # Exit-specific confidence parameters for 0.3% micro movements
                'exit_confidence_threshold': {'type': 'float', 'min': 0.3, 'max': 0.7},
                'tactician_exit_confidence_weight': {'type': 'float', 'min': 0.4, 'max': 0.8},
                'analyst_exit_confidence_weight': {'type': 'float', 'min': 0.2, 'max': 0.6},
                'exit_confidence_combination_method': {'type': 'categorical', 'choices': ['multiplicative', 'logarithmic', 'weighted_average']},
                # 0.3% Micro Movement Exit Thresholds (immediate only)
                'exit_micro_immediate_long_threshold': {'type': 'float', 'min': 0.1, 'max': 0.5},
                'exit_micro_immediate_short_threshold': {'type': 'float', 'min': 0.1, 'max': 0.5},
                # Directional Reversal Detection (MAIN EXIT TRIGGER)
                'directional_confidence_min': {'type': 'float', 'min': 0.05, 'max': 0.5}
            },
            'intensity': {
                # Signal intensity and strength parameters
                'signal_intensity_threshold': {'type': 'float', 'min': 0.4, 'max': 0.8},
                'intensity_decay_factor': {'type': 'float', 'min': 0.85, 'max': 0.99},
                'intensity_amplification_factor': {'type': 'float', 'min': 1.05, 'max': 1.25},
                'min_intensity_duration': {'type': 'int', 'min': 3, 'max': 15},
                'max_intensity_duration': {'type': 'int', 'min': 30, 'max': 120},
                'intensity_combination_method': {'type': 'categorical', 'choices': ['weighted_average', 'maximum', 'harmonic_mean']}
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
            'exit_strategy': {
                # Confidence thresholds
                'confidence_very_low': {'type': 'float', 'min': 0.1, 'max': 0.3},
                'confidence_low': {'type': 'float', 'min': 0.3, 'max': 0.5},
                'confidence_medium': {'type': 'float', 'min': 0.5, 'max': 0.7},
                'confidence_high': {'type': 'float', 'min': 0.7, 'max': 0.9},
                
                # Profit-taking parameters
                'base_profit_target': {'type': 'float', 'min': 0.02, 'max': 0.08},
                'min_confidence_for_profit': {'type': 'float', 'min': 0.5, 'max': 0.8},
                'confidence_profit_multiplier': {'type': 'float', 'min': 0.2, 'max': 0.8},
                'profit_tier_1': {'type': 'float', 'min': 0.2, 'max': 0.4},
                'profit_tier_2': {'type': 'float', 'min': 0.4, 'max': 0.6},
                'profit_tier_3': {'type': 'float', 'min': 0.6, 'max': 0.8},
                
                # Stop-loss parameters
                'base_stop_loss': {'type': 'float', 'min': -0.08, 'max': -0.02},
                'atr_multiplier': {'type': 'float', 'min': 1.0, 'max': 3.0},
                'volatility_adjustment_factor': {'type': 'float', 'min': 0.5, 'max': 2.0},
                
                # Time-based parameters
                'max_hold_time': {'type': 'int', 'min': 3600, 'max': 14400},  # 1-4 hours
                'min_hold_time': {'type': 'int', 'min': 60, 'max': 1800},     # 1-30 minutes
                'confidence_time_scaling_factor': {'type': 'float', 'min': 0.5, 'max': 2.0},
                
                # Trailing stop parameters
                'trailing_atr_multiplier': {'type': 'float', 'min': 1.0, 'max': 3.0},
                'trailing_min_distance': {'type': 'float', 'min': 0.005, 'max': 0.03},
                'trailing_confidence_activation': {'type': 'float', 'min': 0.6, 'max': 0.9},
                
                # Regime-aware parameters
                'regime_transition_penalty': {'type': 'float', 'min': 0.05, 'max': 0.2},
                'regime_specific_scaling': {'type': 'float', 'min': 0.8, 'max': 1.2}
            },
            'ensemble': {
                'analyst_weight': {'type': 'float', 'min': 0.2, 'max': 0.5},
                'tactician_weight': {'type': 'float', 'min': 0.2, 'max': 0.5},
                'strategist_weight': {'type': 'float', 'min': 0.1, 'max': 0.3},
                # Ensemble method parameters for Analyst (Elastic Net meta) & Tactician (LightGBM meta)
                'ensemble_method': {'type': 'categorical', 'choices': ['stacking', 'weighted_average', 'voting', 'meta_learner']},
                'analyst_meta_model_type': {'type': 'categorical', 'choices': ['elastic_net']},
                'tactician_meta_model_type': {'type': 'categorical', 'choices': ['lightgbm']},
                'stacking_cv_folds': {'type': 'int', 'min': 3, 'max': 10},
                'meta_learner_weight': {'type': 'float', 'min': 0.1, 'max': 0.4}
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
            },
            'entry_timing_optimization': {
                # Entry timing parameters - Tactician naturally optimizes for 0-0.4% range
                'entry_timing_range': {'type': 'float', 'min': 0.002, 'max': 0.004},  # 0.2% to 0.4%
                'early_entry_penalty_weight': {'type': 'float', 'min': 0.1, 'max': 0.5},
                'late_entry_penalty_weight': {'type': 'float', 'min': 0.1, 'max': 0.5},
                'optimal_entry_reward_weight': {'type': 'float', 'min': 0.3, 'max': 0.7},
                'entry_timing_efficiency_weight': {'type': 'float', 'min': 0.2, 'max': 0.6},
                'directional_accuracy_threshold': {'type': 'float', 'min': 0.55, 'max': 0.75},
                'adverse_movement_threshold': {'type': 'float', 'min': 0.6, 'max': 0.8},
                'entry_timing_lookback_periods': {'type': 'int', 'min': 5, 'max': 20}
            },
            'confidence_aware_ensemble': {
                # Confidence-aware ensemble parameters for updated models
                'confidence_threshold_entry': {'type': 'float', 'min': 0.6, 'max': 0.85},
                'confidence_threshold_exit': {'type': 'float', 'min': 0.5, 'max': 0.75},
                'confidence_weight_analyst': {'type': 'float', 'min': 0.2, 'max': 0.5},
                'confidence_weight_tactician': {'type': 'float', 'min': 0.3, 'max': 0.6},
                'confidence_combination_method': {'type': 'categorical', 'choices': ['multiplicative', 'weighted_average', 'harmonic_mean', 'geometric_mean']},
                'ensemble_confidence_threshold': {'type': 'float', 'min': 0.65, 'max': 0.9},
                'base_model_confidence_weight': {'type': 'float', 'min': 0.4, 'max': 0.8},
                'meta_model_confidence_weight': {'type': 'float', 'min': 0.2, 'max': 0.6}
            },
            'model_specific_parameters': {
                # Analyst model weights (Base models)
                'analyst_tcn_weight': {'type': 'float', 'min': 0.2, 'max': 0.4},
                'analyst_catboost_weight': {'type': 'float', 'min': 0.2, 'max': 0.4},
                'analyst_lightgbm_weight': {'type': 'float', 'min': 0.2, 'max': 0.4},
                # Analyst meta-learner weight
                'analyst_elastic_net_weight': {'type': 'float', 'min': 0.1, 'max': 0.3},
                
                # Tactician model weights (Base models)
                'tactician_xgboost_weight': {'type': 'float', 'min': 0.2, 'max': 0.35},
                'tactician_randomforest_weight': {'type': 'float', 'min': 0.15, 'max': 0.3},
                'tactician_catboost_weight': {'type': 'float', 'min': 0.2, 'max': 0.35},
                'tactician_elastic_net_weight': {'type': 'float', 'min': 0.15, 'max': 0.3},
                # Tactician meta-learner weight
                'tactician_lightgbm_weight': {'type': 'float', 'min': 0.1, 'max': 0.3},
                
                # General model parameters
                'model_diversity_bonus': {'type': 'float', 'min': 0.05, 'max': 0.15},
                'model_complexity_penalty': {'type': 'float', 'min': 0.01, 'max': 0.1}
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
            elif category == 'exit_strategy':
                base_score = self._evaluate_exit_strategy_params(params, calibration_results)
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
            elif category == 'intensity':
                base_score = self._evaluate_intensity_params(params, calibration_results)
            elif category == 'entry_timing_optimization':
                base_score = self._evaluate_entry_timing_optimization_params(params, calibration_results)
            elif category == 'confidence_aware_ensemble':
                base_score = self._evaluate_confidence_aware_ensemble_params(params, calibration_results)
            elif category == 'model_specific_parameters':
                base_score = self._evaluate_model_specific_params(params, calibration_results)

            # Apply turnover cost penalty to all categories
            if base_score > 0.0:
                turnover_penalty = self._calculate_turnover_penalty(params, calibration_results)
                base_score -= turnover_penalty

            base_score = self._apply_regime_performance_adjustment(category, base_score)

            # Enhanced confidence evaluation includes exit confidence optimization
            # This is handled within _evaluate_confidence_params method

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
                
                # Evaluate exit confidence calculation using existing backtesting framework
                exit_confidence_score = self._evaluate_exit_confidence_calculation(
                    analyst_thresh, tactician_thresh, enhanced_calibration
                )
                score += exit_confidence_score * 0.2  # Weight exit confidence evaluation
                
                # Additional evaluation using the existing backtesting framework
                backtesting_score = self._evaluate_using_existing_backtesting_framework(
                    enhanced_calibration, params
                )
                score += backtesting_score * 0.1  # Weight backtesting evaluation
        
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
    
    def _evaluate_using_existing_backtesting_framework(self, calibration_results: Dict[str, Any], 
                                                     params: Dict[str, Any]) -> float:
        """
        Evaluate exit confidence parameters using the existing backtesting framework.
        
        This method integrates exit confidence optimization into the existing backtesting
        system rather than creating a separate backtesting strategy.
        
        Args:
            calibration_results: Results from confidence calibration
            params: Current parameter configuration
            
        Returns:
            Backtesting evaluation score (0.0 to 1.0)
        """
        try:
            # Extract exit parameters
            exit_threshold = calibration_results.get('exit_confidence_threshold', 0.5)
            exit_method = calibration_results.get('exit_confidence_combination_method', 'multiplicative')
            
            # Use existing calibration data for evaluation
            if 'analyst_confidence' in calibration_results and 'tactician_confidence' in calibration_results:
                analyst_confidences = calibration_results['analyst_confidence']
                tactician_confidences = calibration_results['tactician_confidence']
                
                # Evaluate exit timing using historical data
                exit_performance = self._evaluate_exit_timing_on_historical_data(
                    analyst_confidences, tactician_confidences, calibration_results
                )
                
                return exit_performance
            
            # Fallback evaluation based on parameter reasonableness
            score = 0.0
            
            # Exit threshold should be reasonable
            if 0.4 <= exit_threshold <= 0.6:
                score += 0.4
            elif 0.3 <= exit_threshold <= 0.7:
                score += 0.2
            
            # Method consistency
            if exit_method in ['multiplicative', 'logarithmic']:
                score += 0.3
            else:
                score += 0.2
            
            # Weight balance
            tactician_weight = calibration_results.get('tactician_exit_confidence_weight', 0.6)
            analyst_weight = calibration_results.get('analyst_exit_confidence_weight', 0.4)
            
            if abs(tactician_weight + analyst_weight - 1.0) < 0.1:
                score += 0.3
            
            return min(1.0, score)
            
        except Exception as e:
            self.logger.error(f"❌ Error in existing backtesting framework evaluation: {e}")
            return 0.5
    
    def _evaluate_exit_timing_on_historical_data(self, analyst_confidences: List[float], 
                                               tactician_confidences: List[float],
                                               calibration_results: Dict[str, Any]) -> float:
        """
        Evaluate exit timing using historical confidence data from calibration results.
        
        This uses the existing backtesting framework's historical data rather than
        creating synthetic scenarios.
        """
        try:
            if len(analyst_confidences) != len(tactician_confidences):
                return 0.5
            
            exit_threshold = calibration_results.get('exit_confidence_threshold', 0.5)
            tactician_weight = calibration_results.get('tactician_exit_confidence_weight', 0.6)
            analyst_weight = calibration_results.get('analyst_exit_confidence_weight', 0.4)
            exit_method = calibration_results.get('exit_confidence_combination_method', 'multiplicative')
            
            # Calculate exit points using historical data
            exit_signals = []
            for analyst_conf, tactician_conf in zip(analyst_confidences, tactician_confidences):
                if exit_method == 'multiplicative':
                    exit_conf = self._calculate_multiplicative_confidence(
                        analyst_conf, tactician_conf, tactician_weight, analyst_weight
                    )
                elif exit_method == 'logarithmic':
                    exit_conf = self._calculate_logarithmic_confidence(
                        analyst_conf, tactician_conf, tactician_weight, analyst_weight
                    )
                else:
                    exit_conf = analyst_conf * analyst_weight + tactician_conf * tactician_weight
                
                exit_signals.append(exit_conf < exit_threshold)
            
            # Evaluate exit signal quality using existing framework metrics
            if 'historical_returns' in calibration_results:
                returns = calibration_results['historical_returns']
                return self._score_exit_signals_against_returns(exit_signals, returns)
            
            # Fallback: evaluate signal consistency
            exit_rate = sum(exit_signals) / len(exit_signals) if exit_signals else 0
            
            # Reasonable exit rate (not too frequent, not too rare)
            if 0.1 <= exit_rate <= 0.3:
                return 0.8
            elif 0.05 <= exit_rate <= 0.4:
                return 0.6
            else:
                return 0.4
                
        except Exception as e:
            self.logger.error(f"❌ Error evaluating exit timing on historical data: {e}")
            return 0.5
    
    def _score_exit_signals_against_returns(self, exit_signals: List[bool], 
                                          returns: List[float]) -> float:
        """
        Score exit signals against historical returns using existing backtesting framework.
        """
        try:
            if len(exit_signals) != len(returns):
                return 0.5
            
            score = 0.0
            correct_exits = 0
            total_exits = sum(exit_signals)
            
            if total_exits == 0:
                return 0.3  # No exits might be too conservative
            
            # Check if exits preceded negative returns
            for i, (should_exit, return_val) in enumerate(zip(exit_signals[:-1], returns[1:])):
                if should_exit:
                    # Good exit if next return is negative
                    if return_val < 0:
                        correct_exits += 1
                    # Penalty for exiting before positive returns
                    elif return_val > 0.01:  # Significant positive return
                        correct_exits -= 0.5
            
            # Score based on exit accuracy
            if total_exits > 0:
                exit_accuracy = correct_exits / total_exits
                score = max(0.0, min(1.0, 0.5 + exit_accuracy * 0.5))
            
            return score
            
        except Exception as e:
            self.logger.error(f"❌ Error scoring exit signals against returns: {e}")
            return 0.5

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
    
    def _evaluate_exit_strategy_params(self, params: Dict[str, Any], 
                                     calibration_results: Dict[str, Any]) -> float:
        """Evaluate exit strategy parameters."""
        score = 0.0
        
        try:
            # 1. Confidence thresholds validation (0.3 weight)
            confidence_params = ['confidence_very_low', 'confidence_low', 'confidence_medium', 'confidence_high']
            if all(param in params for param in confidence_params):
                thresholds = [params[param] for param in confidence_params]
                # Check if thresholds are in ascending order
                if all(thresholds[i] <= thresholds[i+1] for i in range(len(thresholds)-1)):
                    score += 0.3
                    # Bonus for reasonable spacing
                    if all(thresholds[i+1] - thresholds[i] >= 0.1 for i in range(len(thresholds)-1)):
                        score += 0.1
                else:
                    score += 0.1  # Partial credit for having all parameters
            
            # 2. Profit-taking parameters validation (0.25 weight)
            profit_params = ['base_profit_target', 'min_confidence_for_profit', 'confidence_profit_multiplier']
            if all(param in params for param in profit_params):
                base_target = params['base_profit_target']
                min_conf = params['min_confidence_for_profit']
                conf_mult = params['confidence_profit_multiplier']
                
                # Validate profit target is positive and reasonable
                if 0.02 <= base_target <= 0.08:
                    score += 0.1
                # Validate confidence threshold is reasonable
                if 0.5 <= min_conf <= 0.8:
                    score += 0.1
                # Validate confidence multiplier is reasonable
                if 0.2 <= conf_mult <= 0.8:
                    score += 0.05
            
            # 3. Stop-loss parameters validation (0.2 weight)
            stop_params = ['base_stop_loss', 'atr_multiplier', 'volatility_adjustment_factor']
            if all(param in params for param in stop_params):
                stop_loss = params['base_stop_loss']
                atr_mult = params['atr_multiplier']
                vol_adj = params['volatility_adjustment_factor']
                
                # Validate stop loss is negative and reasonable
                if -0.08 <= stop_loss <= -0.02:
                    score += 0.1
                # Validate ATR multiplier is reasonable
                if 1.0 <= atr_mult <= 3.0:
                    score += 0.05
                # Validate volatility adjustment is reasonable
                if 0.5 <= vol_adj <= 2.0:
                    score += 0.05
            
            # 4. Time-based parameters validation (0.15 weight)
            time_params = ['max_hold_time', 'min_hold_time', 'confidence_time_scaling_factor']
            if all(param in params for param in time_params):
                max_time = params['max_hold_time']
                min_time = params['min_hold_time']
                time_scaling = params['confidence_time_scaling_factor']
                
                # Validate time constraints are reasonable
                if 3600 <= max_time <= 14400 and 60 <= min_time <= 1800 and min_time < max_time:
                    score += 0.1
                # Validate time scaling factor
                if 0.5 <= time_scaling <= 2.0:
                    score += 0.05
            
            # 5. Trailing stop parameters validation (0.1 weight)
            trailing_params = ['trailing_atr_multiplier', 'trailing_min_distance', 'trailing_confidence_activation']
            if all(param in params for param in trailing_params):
                trailing_atr = params['trailing_atr_multiplier']
                min_dist = params['trailing_min_distance']
                conf_act = params['trailing_confidence_activation']
                
                # Validate trailing stop parameters
                if (1.0 <= trailing_atr <= 3.0 and 
                    0.005 <= min_dist <= 0.03 and 
                    0.6 <= conf_act <= 0.9):
                    score += 0.1
            
            # 6. Regime-aware parameters validation (0.1 weight)
            regime_params = ['regime_transition_penalty', 'regime_specific_scaling']
            if all(param in params for param in regime_params):
                transition_penalty = params['regime_transition_penalty']
                regime_scaling = params['regime_specific_scaling']
                
                # Validate regime parameters
                if 0.05 <= transition_penalty <= 0.2 and 0.8 <= regime_scaling <= 1.2:
                    score += 0.1
            
            # 7. Profit tier validation (bonus)
            tier_params = ['profit_tier_1', 'profit_tier_2', 'profit_tier_3']
            if all(param in params for param in tier_params):
                tiers = [params[param] for param in tier_params]
                # Check if tiers are in ascending order
                if all(tiers[i] <= tiers[i+1] for i in range(len(tiers)-1)):
                    score += 0.05
            
            # 8. Risk-reward ratio validation (bonus)
            if 'base_profit_target' in params and 'base_stop_loss' in params:
                profit_target = params['base_profit_target']
                stop_loss = abs(params['base_stop_loss'])
                risk_reward_ratio = profit_target / stop_loss
                
                # Bonus for good risk-reward ratio
                if 1.5 <= risk_reward_ratio <= 3.0:
                    score += 0.1
                elif 1.0 <= risk_reward_ratio < 1.5:
                    score += 0.05
            
        except Exception as e:
            self.logger.error(f"❌ Error evaluating exit strategy parameters: {e}")
            score = 0.0
        
        return min(score, 1.0)  # Cap at 1.0
    
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

    def _evaluate_intensity_params(self, params: Dict[str, Any], 
                                 calibration_results: Dict[str, Any]) -> float:
        """Evaluate signal intensity parameters."""
        score = 0.0
        
        if 'signal_intensity_threshold' in params:
            threshold = params['signal_intensity_threshold']
            if 0.5 <= threshold <= 0.7:
                score += 0.3
            elif 0.4 <= threshold <= 0.8:
                score += 0.2
            else:
                score += 0.1
        
        if 'intensity_decay_factor' in params:
            decay = params['intensity_decay_factor']
            if 0.9 <= decay <= 0.95:
                score += 0.2
            elif 0.85 <= decay <= 0.99:
                score += 0.15
            else:
                score += 0.1
        
        return score
    
    def _evaluate_entry_timing_optimization_params(self, params: Dict[str, Any], 
                                                 calibration_results: Dict[str, Any]) -> float:
        """Evaluate entry timing optimization parameters for updated Tactician models."""
        score = 0.0
        
        if 'entry_timing_range' in params:
            range_val = params['entry_timing_range']
            # Optimal range is around 0.003-0.004 (0.3%-0.4%)
            if 0.003 <= range_val <= 0.004:
                score += 0.3
            elif 0.002 <= range_val <= 0.004:
                score += 0.2
            else:
                score += 0.1
        
        if 'optimal_entry_reward_weight' in params and 'early_entry_penalty_weight' in params:
            reward_weight = params['optimal_entry_reward_weight']
            penalty_weight = params['early_entry_penalty_weight']
            # Reward should be higher than penalty for optimal timing
            if reward_weight > penalty_weight and reward_weight >= 0.4:
                score += 0.25
            else:
                score += 0.15
        
        if 'directional_accuracy_threshold' in params:
            threshold = params['directional_accuracy_threshold']
            if 0.6 <= threshold <= 0.7:
                score += 0.2
            else:
                score += 0.1
        
        return score
    
    def _evaluate_confidence_aware_ensemble_params(self, params: Dict[str, Any], 
                                                 calibration_results: Dict[str, Any]) -> float:
        """Evaluate confidence-aware ensemble parameters for updated models."""
        score = 0.0
        
        if 'confidence_threshold_entry' in params and 'confidence_threshold_exit' in params:
            entry_thresh = params['confidence_threshold_entry']
            exit_thresh = params['confidence_threshold_exit']
            # Entry threshold should typically be higher than exit threshold
            if entry_thresh > exit_thresh and 0.65 <= entry_thresh <= 0.8:
                score += 0.3
            else:
                score += 0.15
        
        if 'confidence_weight_tactician' in params and 'confidence_weight_analyst' in params:
            tactician_weight = params['confidence_weight_tactician']
            analyst_weight = params['confidence_weight_analyst']
            # Tactician should have higher weight for timing decisions
            if tactician_weight > analyst_weight and tactician_weight >= 0.4:
                score += 0.25
            else:
                score += 0.15
        
        if 'ensemble_confidence_threshold' in params:
            threshold = params['ensemble_confidence_threshold']
            if 0.7 <= threshold <= 0.85:
                score += 0.2
            else:
                score += 0.1
        
        return score
    
    def _evaluate_model_specific_params(self, params: Dict[str, Any], 
                                      calibration_results: Dict[str, Any]) -> float:
        """Evaluate model-specific parameters for new Analyst & Tactician model types."""
        score = 0.0
        
        # Check if weights are balanced for different model types
        analyst_weights = []
        tactician_weights = []
        
        # Analyst model weights
        analyst_weight_keys = [
            'analyst_tcn_weight', 'analyst_catboost_weight', 'analyst_lightgbm_weight'
        ]
        
        # Tactician model weights  
        tactician_weight_keys = [
            'tactician_xgboost_weight', 'tactician_randomforest_weight', 
            'tactician_catboost_weight', 'tactician_elastic_net_weight'
        ]
        
        for key in analyst_weight_keys:
            if key in params:
                analyst_weights.append(params[key])
                
        for key in tactician_weight_keys:
            if key in params:
                tactician_weights.append(params[key])
        
        # Evaluate Analyst model balance
        if analyst_weights:
            max_weight = max(analyst_weights)
            min_weight = min(analyst_weights)
            weight_balance = min_weight / max_weight if max_weight > 0 else 0
            
            if weight_balance >= 0.6:  # Well balanced
                score += 0.15
            elif weight_balance >= 0.4:  # Moderately balanced
                score += 0.1
            else:
                score += 0.05
        
        # Evaluate Tactician model balance
        if tactician_weights:
            max_weight = max(tactician_weights)
            min_weight = min(tactician_weights)
            weight_balance = min_weight / max_weight if max_weight > 0 else 0
            
            if weight_balance >= 0.6:  # Well balanced
                score += 0.15
            elif weight_balance >= 0.4:  # Moderately balanced
                score += 0.1
            else:
                score += 0.05
        
        if 'model_diversity_bonus' in params:
            bonus = params['model_diversity_bonus']
            if 0.08 <= bonus <= 0.12:
                score += 0.15
            else:
                score += 0.1
        
        if 'model_complexity_penalty' in params:
            penalty = params['model_complexity_penalty']
            if 0.02 <= penalty <= 0.06:
                score += 0.15
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

    def _load_regime_performance_stats(self) -> Dict[str, Any]:
        """Load per-regime performance statistics if available.

        This first checks any explicit configuration overrides and then falls back
        to standard reporting directories such as ``reports/backtesting`` or
        ``generated/backtesting``.
        """
        path_candidates: List[Path] = []

        config_path = self.config.get('regime_performance_path') if isinstance(self.config, dict) else None
        if config_path:
            path_candidates.append(Path(config_path))

        reporting_dir = None
        if isinstance(self.config, dict):
            reporting_dir = self.config.get('reporting_output_dir')
            if reporting_dir is None and 'reporting' in self.config and isinstance(self.config['reporting'], dict):
                reporting_dir = self.config['reporting'].get('output_dir')
        if reporting_dir:
            path_candidates.append(Path(reporting_dir) / 'backtesting' / 'per_regime_performance.json')

        path_candidates.extend([
            Path('reports/backtesting/per_regime_performance.json'),
            Path('generated/backtesting/per_regime_performance.json'),
        ])

        for candidate in path_candidates:
            try:
                if candidate.exists():
                    with candidate.open('r') as fp:
                        stats = json.load(fp)
                    self.regime_performance_path = str(candidate)
                    return stats if isinstance(stats, dict) else {}
            except Exception as exc:
                self.logger.error(f"❌ Failed to load regime performance stats from {candidate}: {exc}")

        return {}

    def _calculate_regime_performance_modifier(self) -> float:
        """Compute an aggregate modifier from per-regime performance stats."""
        stats = getattr(self, 'regime_performance_stats', {})
        if not stats:
            return 0.0

        win_rates: List[float] = []
        profit_factors: List[float] = []
        rr_values: List[float] = []

        for metrics in stats.values():
            if not isinstance(metrics, dict):
                continue
            win_rates.append(float(metrics.get('win_rate', 0.0)))
            profit_factors.append(float(metrics.get('profit_factor', 0.0)))
            rr_values.append(float(metrics.get('average_rr', metrics.get('risk_reward_ratio', 0.0))))

        if not win_rates:
            return 0.0

        avg_win = float(np.mean(win_rates))
        min_win = float(np.min(win_rates))
        avg_profit_factor = float(np.mean(profit_factors)) if profit_factors else 0.0
        avg_rr = float(np.mean(rr_values)) if rr_values else 0.0
        stability_penalty = float(np.std(win_rates)) if len(win_rates) > 1 else 0.0

        normalized_win = avg_win - 0.5
        normalized_min_win = min_win - 0.5
        normalized_profit_factor = float(np.tanh(avg_profit_factor - 1.0))
        normalized_rr = float(np.tanh(avg_rr - 1.0))

        raw_modifier = (
            (normalized_win * 0.5)
            + (normalized_profit_factor * 0.2)
            + (normalized_rr * 0.2)
            + (normalized_min_win * 0.1)
            - (stability_penalty * 0.1)
        )

        return float(np.clip(raw_modifier, -0.25, 0.25))

    def _apply_regime_performance_adjustment(self, category: str, score: float) -> float:
        """Adjust objective score using per-regime performance insights."""
        modifier = getattr(self, 'regime_performance_modifier', 0.0)
        if modifier == 0.0:
            return score

        weight = 1.0
        if category in {'tpsl', 'exit_strategy', 'regime_transitions'}:
            weight = 1.2
        elif category in {'confidence', 'position_sizing'}:
            weight = 1.1
        elif category in {'ensemble', 'model_specific_parameters'}:
            weight = 0.9

        adjustment = float(np.clip(modifier * weight, -0.2, 0.2))
        return score + adjustment

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
            optimization_dir = f'generated/backtesting/optimization_results'
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