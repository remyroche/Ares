"""
Non-Linear Optimization Helpers

This module provides helper functions for non-linear parameter transformations
that can be integrated into existing optimization systems.

Key Features:
- Log-space parameter sampling
- Fractional power transformations
- Sigmoid transformations
- Adaptive transformation selection
- Parameter space conversion utilities
"""

import numpy as np
import logging
from typing import Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class NonLinearConfig:
    """Configuration for non-linear optimization methods."""
    use_log_sampling: bool = True
    use_fractional_powers: bool = True
    use_sigmoid_transforms: bool = True
    use_adaptive_transforms: bool = True
    log_threshold: float = 10.0  # Use log sampling for ranges > this value
    sigmoid_range: Tuple[float, float] = (-6.0, 6.0)
    power_exponents: list = None
    
    def __post_init__(self):
        if self.power_exponents is None:
            self.power_exponents = [0.3, 0.5, 0.7, 0.9]

class NonLinearParameterSampler:
    """Helper class for non-linear parameter sampling in optimization."""
    
    def __init__(self, config: Optional[NonLinearConfig] = None):
        self.config = config or NonLinearConfig()
        self.logger = logger.getChild('NonLinearParameterSampler')
    
    def suggest_enhanced_float(self, trial, param_name: str, min_val: float, max_val: float,
                             param_type: str = 'auto') -> float:
        """
        Suggest float parameter with enhanced non-linear sampling.
        
        Args:
            trial: Optuna trial object
            param_name: Name of the parameter
            min_val: Minimum value
            max_val: Maximum value
            param_type: Type of transformation ('auto', 'log', 'power', 'sigmoid', 'linear')
            
        Returns:
            Sampled parameter value
        """
        range_size = max_val - min_val
        
        # Auto-select transformation based on parameter characteristics
        if param_type == 'auto':
            param_type = self._select_optimal_transformation(min_val, max_val, range_size)
        
        if param_type == 'log' and self.config.use_log_sampling:
            return self._suggest_log_float(trial, param_name, min_val, max_val)
        elif param_type == 'power' and self.config.use_fractional_powers:
            return self._suggest_power_float(trial, param_name, min_val, max_val)
        elif param_type == 'sigmoid' and self.config.use_sigmoid_transforms:
            return self._suggest_sigmoid_float(trial, param_name, min_val, max_val)
        else:
            # Fallback to linear sampling
            return trial.suggest_float(param_name, min_val, max_val)
    
    def _select_optimal_transformation(self, min_val: float, max_val: float, range_size: float) -> str:
        """Select optimal transformation based on parameter characteristics."""
        if range_size > self.config.log_threshold and min_val > 0:
            return 'log'
        elif 0 < min_val < 1 and max_val > 1:
            return 'power'
        elif min_val >= 0 and max_val <= 1:
            return 'sigmoid'
        elif range_size < 1.0:
            return 'sigmoid'
        else:
            return 'linear'
    
    def _suggest_log_float(self, trial, param_name: str, min_val: float, max_val: float) -> float:
        """Suggest float parameter using logarithmic sampling."""
        log_min = np.log(max(min_val, 1e-10))  # Avoid log(0)
        log_max = np.log(max_val)
        log_param = trial.suggest_float(f"log_{param_name}", log_min, log_max)
        return np.exp(log_param)
    
    def _suggest_power_float(self, trial, param_name: str, min_val: float, max_val: float, 
                           power: float = 0.5) -> float:
        """Suggest float parameter using fractional power transformation."""
        raw_param = trial.suggest_float(f"raw_{param_name}", 0.0, 1.0)
        transformed = raw_param ** power
        return min_val + transformed * (max_val - min_val)
    
    def _suggest_sigmoid_float(self, trial, param_name: str, min_val: float, max_val: float) -> float:
        """Suggest float parameter using sigmoid transformation."""
        raw_param = trial.suggest_float(
            f"raw_{param_name}",
            self.config.sigmoid_range[0],
            self.config.sigmoid_range[1]
        )
        sigmoid_param = 1 / (1 + np.exp(-raw_param))
        return min_val + sigmoid_param * (max_val - min_val)
    
    def suggest_enhanced_int(self, trial, param_name: str, min_val: int, max_val: int,
                           param_type: str = 'auto') -> int:
        """
        Suggest integer parameter with enhanced non-linear sampling.
        
        Args:
            trial: Optuna trial object
            param_name: Name of the parameter
            min_val: Minimum value
            max_val: Maximum value
            param_type: Type of transformation
            
        Returns:
            Sampled parameter value
        """
        range_size = max_val - min_val
        
        if param_type == 'auto':
            param_type = self._select_optimal_transformation(min_val, max_val, range_size)
        
        if param_type == 'log' and self.config.use_log_sampling and min_val > 0:
            # Convert to float, apply log sampling, then round to int
            float_val = self._suggest_log_float(trial, param_name, float(min_val), float(max_val))
            return max(min_val, min(max_val, int(round(float_val))))
        elif param_type == 'power' and self.config.use_fractional_powers:
            # Convert to float, apply power sampling, then round to int
            float_val = self._suggest_power_float(trial, param_name, float(min_val), float(max_val))
            return max(min_val, min(max_val, int(round(float_val))))
        else:
            # Fallback to linear sampling
            return trial.suggest_int(param_name, min_val, max_val)

def apply_nonlinear_scoring(base_score: float, params: Dict[str, Any], 
                          category: str) -> float:
    """
    Apply non-linear scoring enhancements to base optimization score.
    
    Args:
        base_score: Base optimization score
        params: Parameter dictionary
        category: Parameter category
        
    Returns:
        Enhanced score with non-linear adjustments
    """
    try:
        enhanced_score = base_score
        
        # Apply category-specific non-linear enhancements
        if category == 'confidence':
            enhanced_score = _enhance_confidence_scoring(enhanced_score, params)
        elif category == 'position_sizing':
            enhanced_score = _enhance_position_sizing_scoring(enhanced_score, params)
        elif category == 'leverage':
            enhanced_score = _enhance_leverage_scoring(enhanced_score, params)
        elif category == 'ensemble':
            enhanced_score = _enhance_ensemble_scoring(enhanced_score, params)
        elif category == 'tpsl':
            enhanced_score = _enhance_tpsl_scoring(enhanced_score, params)
        
        return enhanced_score
        
    except Exception as e:
        logger.warning(f"Non-linear scoring failed: {e}")
        return base_score

def _enhance_confidence_scoring(score: float, params: Dict[str, Any]) -> float:
    """Enhance confidence parameter scoring with non-linear adjustments."""
    if 'base_entry_threshold' in params:
        threshold = params['base_entry_threshold']
        # Non-linear confidence scaling: higher confidence gets exponential bonus
        confidence_bonus = np.exp(threshold - 0.5) - 1
        score += confidence_bonus * 0.1
    
    if 'analyst_confidence_threshold' in params and 'tactician_confidence_threshold' in params:
        analyst_thresh = params['analyst_confidence_threshold']
        tactician_thresh = params['tactician_confidence_threshold']
        # Reward proper hierarchy with non-linear scaling
        if tactician_thresh > analyst_thresh:
            hierarchy_bonus = np.log(1 + (tactician_thresh - analyst_thresh) * 10)
            score += hierarchy_bonus * 0.05
    
    return score

def _enhance_position_sizing_scoring(score: float, params: Dict[str, Any]) -> float:
    """Enhance position sizing parameter scoring with non-linear adjustments."""
    if 'base_position_size' in params:
        position_size = params['base_position_size']
        # Risk-adjusted scoring: penalize very large positions exponentially
        risk_penalty = (position_size ** 2.5) * 0.2
        score -= risk_penalty
        
        # Reward moderate position sizes with log bonus
        if 0.02 <= position_size <= 0.1:
            moderate_bonus = np.log(1 + position_size * 10)
            score += moderate_bonus * 0.1
    
    if 'max_position_size' in params:
        max_size = params['max_position_size']
        # Penalize excessive max position sizes
        if max_size > 0.25:
            excess_penalty = ((max_size - 0.25) ** 2) * 2.0
            score -= excess_penalty
    
    return score

def _enhance_leverage_scoring(score: float, params: Dict[str, Any]) -> float:
    """Enhance leverage parameter scoring with non-linear adjustments."""
    if 'safe_leverage_multiplier' in params:
        leverage = params['safe_leverage_multiplier']
        # Leverage scoring: optimal around 0.7-0.8, penalize extremes
        optimal_leverage = 0.75
        leverage_penalty = ((leverage - optimal_leverage) ** 2) * 2.0
        score -= leverage_penalty
        
        # Reward conservative leverage with exponential bonus
        if leverage <= 0.8:
            conservative_bonus = np.exp(-leverage * 2) * 0.1
            score += conservative_bonus
    
    return score

def _enhance_ensemble_scoring(score: float, params: Dict[str, Any]) -> float:
    """Enhance ensemble parameter scoring with non-linear adjustments."""
    weight_keys = ['analyst_weight', 'tactician_weight', 'strategist_weight']
    weights = [params.get(key, 0) for key in weight_keys]
    
    if all(w > 0 for w in weights):
        # Diversity bonus: reward balanced weights using entropy
        weight_entropy = -sum(w * np.log(w) for w in weights if w > 0)
        diversity_bonus = weight_entropy * 0.1
        score += diversity_bonus
        
        # Penalty for extreme weights
        for weight in weights:
            if weight > 0.6 or weight < 0.1:
                extreme_penalty = ((weight - 0.35) ** 2) * 0.5
                score -= extreme_penalty
    
    return score

def _enhance_tpsl_scoring(score: float, params: Dict[str, Any]) -> float:
    """Enhance TP/SL parameter scoring with non-linear adjustments."""
    if 'tp_long' in params and 'sl_long' in params:
        tp = params['tp_long']
        sl = params['sl_long']
        
        if tp > sl:
            # Reward good risk-reward ratios with log scaling
            risk_reward_ratio = tp / sl
            if risk_reward_ratio >= 1.5:
                ratio_bonus = np.log(risk_reward_ratio) * 0.1
                score += ratio_bonus
            
            # Penalize extreme ratios
            if risk_reward_ratio > 5.0:
                extreme_penalty = ((risk_reward_ratio - 5.0) ** 2) * 0.05
                score -= extreme_penalty
    
    return score

def create_enhanced_search_space(original_space: Dict[str, Dict[str, Any]], 
                               config: Optional[NonLinearConfig] = None) -> Dict[str, Dict[str, Any]]:
    """
    Create enhanced search space with non-linear transformation metadata.
    
    Args:
        original_space: Original parameter search space
        config: Non-linear configuration
        
    Returns:
        Enhanced search space with transformation metadata
    """
    config = config or NonLinearConfig()
    enhanced_space = {}
    
    for param_name, param_config in original_space.items():
        if param_config['type'] == 'float':
            # Support both 'min'/'max' and 'low'/'high' formats for backward compatibility
            min_val = param_config.get('min', param_config.get('low', 0))
            max_val = param_config.get('max', param_config.get('high', 1))
            range_size = max_val - min_val
            
            # Determine optimal transformation
            if range_size > config.log_threshold and min_val > 0:
                transform_type = 'log'
            elif 0 < min_val < 1 and max_val > 1:
                transform_type = 'power'
            elif min_val >= 0 and max_val <= 1:
                transform_type = 'sigmoid'
            elif range_size < 1.0:
                transform_type = 'sigmoid'
            else:
                transform_type = 'linear'
            
            enhanced_space[param_name] = {
                **param_config,
                'transform_type': transform_type,
                'range_size': range_size
            }
        else:
            # Keep non-float parameters as-is
            enhanced_space[param_name] = param_config
    
    return enhanced_space

def convert_parameters_to_original_space(params: Dict[str, Any], 
                                       search_space: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Convert enhanced parameters back to original parameter space for reporting.
    
    Args:
        params: Parameters from enhanced optimization
        search_space: Enhanced search space metadata
        
    Returns:
        Parameters in original space
    """
    converted_params = {}
    
    for param_name, value in params.items():
        # Remove prefixes added by enhanced sampling
        original_name = param_name
        if param_name.startswith('log_'):
            original_name = param_name[4:]
        elif param_name.startswith('raw_'):
            original_name = param_name[4:]
        
        if original_name in search_space:
            # These are already converted to original space in the objective function
            converted_params[original_name] = value
        else:
            converted_params[param_name] = value
    
    return converted_params