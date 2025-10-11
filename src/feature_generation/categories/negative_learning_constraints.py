"""
Negative Learning Model Constraints Module

This module implements model constraints and validation for negative learning features
to keep models honest and prevent overfitting in challenging market conditions.

Key Features:
- Monotone constraints for tree-based models
- Sample weights for uncertain regions
- Model-specific constraint generation
- Validation framework integration
- Performance monitoring
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

from src.utils.logger import system_logger
from src.utils.math_validation import safe_divide, validate_finite

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None


class ModelType(Enum):
    """Supported model types for constraints"""
    LIGHTGBM = "lightgbm"
    XGBOOST = "xgboost"
    CATBOOST = "catboost"
    RANDOM_FOREST = "random_forest"
    ELASTIC_NET = "elastic_net"


@dataclass
class ConstraintConfig:
    """Configuration for model constraints"""
    enable_monotone_constraints: bool = True
    enable_sample_weights: bool = True
    enable_feature_caps: bool = True
    weight_uncertainty_factor: float = 0.3
    min_sample_weight: float = 0.1
    max_sample_weight: float = 2.0
    monotone_strength: float = 1.0


@dataclass
class ModelConstraints:
    """Model constraints for negative learning features"""
    monotone_constraints: List[int]
    sample_weights: Optional[pd.Series] = None
    feature_caps: Optional[Dict[str, Tuple[float, float]]] = None
    feature_importance_weights: Optional[Dict[str, float]] = None


class MonotoneConstraintGenerator:
    """
    Generates monotone constraints for tree-based models to encode domain knowledge
    about negative learning features.
    """
    
    def __init__(self, config: Optional[ConstraintConfig] = None):
        self.config = config or ConstraintConfig()
        self.logger = system_logger.getChild('MonotoneConstraintGenerator')
    
    def generate_monotone_constraints(
        self,
        feature_names: List[str],
        negative_learning_features: List[str],
        model_type: ModelType = ModelType.LIGHTGBM
    ) -> List[int]:
        """
        Generate monotone constraints for model features.
        
        Args:
            feature_names: Complete list of feature names in model order
            negative_learning_features: List of negative learning feature names
            model_type: Type of model (affects constraint format)
            
        Returns:
            List of monotone constraints (-1, 0, 1)
        """
        self.logger.debug(f"Generating monotone constraints for {model_type.value} model...")
        
        constraints = []
        
        for feature_name in feature_names:
            constraint = self._get_feature_constraint(
                feature_name, negative_learning_features, model_type
            )
            constraints.append(constraint)
        
        self.logger.debug(f"Generated {len(constraints)} monotone constraints")
        return constraints
    
    def _get_feature_constraint(
        self,
        feature_name: str,
        negative_learning_features: List[str],
        model_type: ModelType
    ) -> int:
        """Get monotone constraint for a specific feature"""
        
        # Base features - no constraint (let model learn freely)
        if feature_name not in negative_learning_features:
            return 0
        
        # Positive gated twins - should have positive monotonicity
        if feature_name.endswith('_pos'):
            return 1
        
        # Negative gated twins - should have negative monotonicity
        elif feature_name.endswith('_neg'):
            return -1
        
        # Exception interactions - no constraint (let model learn)
        elif feature_name.endswith('_x_fail'):
            return 0
        
        # Context indicators - no constraint (binary/categorical)
        elif feature_name.endswith('_p_'):
            return 0
        
        # Default - no constraint
        else:
            return 0
    
    def get_constraint_explanation(
        self,
        feature_names: List[str],
        negative_learning_features: List[str]
    ) -> Dict[str, str]:
        """Get human-readable explanation of constraints"""
        explanations = {}
        
        for feature_name in feature_names:
            if feature_name in negative_learning_features:
                if feature_name.endswith('_pos'):
                    explanations[feature_name] = "Positive monotonicity: higher values should increase prediction"
                elif feature_name.endswith('_neg'):
                    explanations[feature_name] = "Negative monotonicity: higher values should decrease prediction"
                elif feature_name.endswith('_x_fail'):
                    explanations[feature_name] = "No constraint: let model learn interaction effects"
                elif feature_name.endswith('_p_'):
                    explanations[feature_name] = "No constraint: context indicator"
                else:
                    explanations[feature_name] = "No constraint: unknown negative learning feature type"
            else:
                explanations[feature_name] = "No constraint: base feature"
        
        return explanations


class SampleWeightGenerator:
    """
    Generates sample weights to down-weight observations in uncertain failure zones
    and prevent overfitting to noisy regions.
    """
    
    def __init__(self, config: Optional[ConstraintConfig] = None):
        self.config = config or ConstraintConfig()
        self.logger = system_logger.getChild('SampleWeightGenerator')
    
    def generate_sample_weights(
        self,
        features_df: pd.DataFrame,
        failure_contexts: Dict[str, List[Any]],
        base_weights: Optional[pd.Series] = None,
        target: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Generate sample weights based on failure context uncertainty.
        
        Args:
            features_df: Feature matrix
            failure_contexts: Detected failure contexts per feature
            base_weights: Optional base sample weights
            target: Optional target variable for additional weighting
            
        Returns:
            Sample weights for training
        """
        self.logger.debug("Generating sample weights...")
        
        if base_weights is None:
            base_weights = pd.Series(1.0, index=features_df.index)
        
        # Calculate uncertainty weights
        uncertainty_weights = self._calculate_uncertainty_weights(
            features_df, failure_contexts
        )
        
        # Calculate target-based weights if available
        if target is not None:
            target_weights = self._calculate_target_weights(target)
        else:
            target_weights = pd.Series(1.0, index=features_df.index)
        
        # Combine weights
        combined_weights = base_weights * uncertainty_weights * target_weights
        
        # Apply constraints
        final_weights = self._apply_weight_constraints(combined_weights)
        
        self.logger.debug(f"Generated sample weights: mean={final_weights.mean():.3f}, std={final_weights.std():.3f}")
        return final_weights
    
    def _calculate_uncertainty_weights(
        self,
        features_df: pd.DataFrame,
        failure_contexts: Dict[str, List[Any]]
    ) -> pd.Series:
        """Calculate weights based on failure context uncertainty"""
        if not failure_contexts:
            return pd.Series(1.0, index=features_df.index)
        
        # Calculate maximum failure probability across all features
        p_fail_max = pd.Series(0.0, index=features_df.index)
        
        for feature_name, contexts in failure_contexts.items():
            if not contexts:
                continue
            
            # Generate context flags for this feature
            p_fail = self._calculate_feature_failure_probability(
                features_df, contexts, feature_name
            )
            p_fail_max = np.maximum(p_fail_max, p_fail)
        
        # Convert to uncertainty weights
        # Higher failure probability = lower weight (more uncertain)
        uncertainty_factor = self.config.weight_uncertainty_factor
        weights = 0.7 + 0.3 * (1 - p_fail_max)
        
        return weights
    
    def _calculate_feature_failure_probability(
        self,
        features_df: pd.DataFrame,
        contexts: List[Any],
        feature_name: str
    ) -> pd.Series:
        """Calculate failure probability for a specific feature"""
        # This is a simplified version - in practice, you'd use the actual
        # context detection logic from the main negative learning module
        try:
            # For now, use a simple heuristic based on volatility
            if 'volatility' in features_df.columns:
                vol = features_df['volatility']
                vol_threshold = vol.quantile(0.7)
                p_fail = (vol > vol_threshold).astype(float)
            else:
                p_fail = pd.Series(0.0, index=features_df.index)
            
            return p_fail
        except Exception as e:
            self.logger.warning(f"Error calculating failure probability for {feature_name}: {e}")
            return pd.Series(0.0, index=features_df.index)
    
    def _calculate_target_weights(self, target: pd.Series) -> pd.Series:
        """Calculate additional weights based on target distribution"""
        try:
            # Down-weight extreme outliers
            target_std = target.std()
            target_mean = target.mean()
            
            # Calculate z-scores
            z_scores = np.abs((target - target_mean) / target_std)
            
            # Weight inversely proportional to z-score (down-weight outliers)
            weights = 1.0 / (1.0 + z_scores)
            
            return weights
        except Exception as e:
            self.logger.warning(f"Error calculating target weights: {e}")
            return pd.Series(1.0, index=target.index)
    
    def _apply_weight_constraints(
        self, 
        weights: pd.Series
    ) -> pd.Series:
        """Apply min/max constraints to sample weights"""
        # Normalize to have mean of 1.0
        weights = weights / weights.mean()
        
        # Apply min/max constraints
        weights = np.clip(weights, self.config.min_sample_weight, self.config.max_sample_weight)
        
        return weights


class FeatureCapGenerator:
    """
    Generates feature caps to prevent extreme values in negative learning features.
    """
    
    def __init__(self, config: Optional[ConstraintConfig] = None):
        self.config = config or ConstraintConfig()
        self.logger = system_logger.getChild('FeatureCapGenerator')
    
    def generate_feature_caps(
        self,
        features_df: pd.DataFrame,
        negative_learning_features: List[str]
    ) -> Dict[str, Tuple[float, float]]:
        """
        Generate feature caps for negative learning features.
        
        Args:
            features_df: Feature matrix
            negative_learning_features: List of negative learning feature names
            
        Returns:
            Dictionary mapping feature names to (min, max) caps
        """
        if not self.config.enable_feature_caps:
            return {}
        
        self.logger.debug("Generating feature caps...")
        
        caps = {}
        
        for feature in negative_learning_features:
            if feature not in features_df.columns:
                continue
            
            feature_data = features_df[feature].dropna()
            if len(feature_data) == 0:
                continue
            
            # Calculate caps based on percentiles
            min_cap = feature_data.quantile(0.01)  # 1st percentile
            max_cap = feature_data.quantile(0.99)  # 99th percentile
            
            # Ensure reasonable bounds
            if np.isfinite(min_cap) and np.isfinite(max_cap):
                caps[feature] = (min_cap, max_cap)
        
        self.logger.debug(f"Generated caps for {len(caps)} features")
        return caps
    
    def apply_feature_caps(
        self,
        features_df: pd.DataFrame,
        caps: Dict[str, Tuple[float, float]]
    ) -> pd.DataFrame:
        """Apply feature caps to a dataframe"""
        if not caps:
            return features_df
        
        capped_df = features_df.copy()
        
        for feature, (min_cap, max_cap) in caps.items():
            if feature in capped_df.columns:
                capped_df[feature] = np.clip(capped_df[feature], min_cap, max_cap)
        
        return capped_df


class ModelConstraintManager:
    """
    Main manager for all model constraints related to negative learning.
    Provides a unified interface for constraint generation and application.
    """
    
    def __init__(self, config: Optional[ConstraintConfig] = None):
        self.config = config or ConstraintConfig()
        self.logger = system_logger.getChild('ModelConstraintManager')
        
        # Initialize components
        self.monotone_generator = MonotoneConstraintGenerator(config)
        self.weight_generator = SampleWeightGenerator(config)
        self.cap_generator = FeatureCapGenerator(config)
        
        # State
        self.current_constraints: Optional[ModelConstraints] = None
    
    def generate_constraints(
        self,
        features_df: pd.DataFrame,
        feature_names: List[str],
        negative_learning_features: List[str],
        failure_contexts: Dict[str, List[Any]],
        model_type: ModelType = ModelType.LIGHTGBM,
        target: Optional[pd.Series] = None,
        base_weights: Optional[pd.Series] = None
    ) -> ModelConstraints:
        """
        Generate all model constraints for negative learning features.
        
        Args:
            features_df: Feature matrix
            feature_names: Complete list of feature names in model order
            negative_learning_features: List of negative learning feature names
            failure_contexts: Detected failure contexts per feature
            model_type: Type of model
            target: Optional target variable
            base_weights: Optional base sample weights
            
        Returns:
            Complete model constraints
        """
        self.logger.info(f"🔧 Generating model constraints for {model_type.value}...")
        
        # Generate monotone constraints
        monotone_constraints = []
        if self.config.enable_monotone_constraints:
            monotone_constraints = self.monotone_generator.generate_monotone_constraints(
                feature_names, negative_learning_features, model_type
            )
        
        # Generate sample weights
        sample_weights = None
        if self.config.enable_sample_weights:
            sample_weights = self.weight_generator.generate_sample_weights(
                features_df, failure_contexts, base_weights, target
            )
        
        # Generate feature caps
        feature_caps = None
        if self.config.enable_feature_caps:
            feature_caps = self.cap_generator.generate_feature_caps(
                features_df, negative_learning_features
            )
        
        # Create constraints object
        constraints = ModelConstraints(
            monotone_constraints=monotone_constraints,
            sample_weights=sample_weights,
            feature_caps=feature_caps
        )
        
        self.current_constraints = constraints
        self.logger.info("✅ Model constraints generated successfully")
        
        return constraints
    
    def get_lightgbm_params(
        self,
        constraints: ModelConstraints,
        base_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Get LightGBM parameters with constraints applied"""
        params = base_params or {}
        
        if constraints.monotone_constraints:
            params['monotone_constraints'] = constraints.monotone_constraints
        
        return params
    
    def get_xgboost_params(
        self,
        constraints: ModelConstraints,
        base_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Get XGBoost parameters with constraints applied"""
        params = base_params or {}
        
        # XGBoost doesn't support monotone constraints directly
        # But we can use feature importance weights
        if constraints.feature_importance_weights:
            params['feature_weights'] = constraints.feature_importance_weights
        
        return params
    
    def get_catboost_params(
        self,
        constraints: ModelConstraints,
        base_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Get CatBoost parameters with constraints applied"""
        params = base_params or {}
        
        if constraints.monotone_constraints:
            params['monotone_constraints'] = constraints.monotone_constraints
        
        return params
    
    def validate_constraints(
        self,
        constraints: ModelConstraints,
        features_df: pd.DataFrame,
        target: pd.Series
    ) -> Dict[str, Any]:
        """
        Validate that constraints are reasonable and don't hurt performance.
        
        Args:
            constraints: Model constraints to validate
            features_df: Feature matrix
            target: Target variable
            
        Returns:
            Validation results
        """
        self.logger.info("🔍 Validating model constraints...")
        
        validation_results = {
            'monotone_constraints': self._validate_monotone_constraints(constraints, features_df, target),
            'sample_weights': self._validate_sample_weights(constraints, features_df, target),
            'feature_caps': self._validate_feature_caps(constraints, features_df)
        }
        
        self.logger.info("✅ Constraint validation complete")
        return validation_results
    
    def _validate_monotone_constraints(
        self,
        constraints: ModelConstraints,
        features_df: pd.DataFrame,
        target: pd.Series
    ) -> Dict[str, Any]:
        """Validate monotone constraints"""
        if not constraints.monotone_constraints:
            return {'status': 'skipped', 'reason': 'No monotone constraints'}
        
        # Check that constraints are reasonable
        positive_constraints = sum(1 for c in constraints.monotone_constraints if c > 0)
        negative_constraints = sum(1 for c in constraints.monotone_constraints if c < 0)
        
        return {
            'status': 'valid',
            'positive_constraints': positive_constraints,
            'negative_constraints': negative_constraints,
            'total_constraints': len(constraints.monotone_constraints)
        }
    
    def _validate_sample_weights(
        self,
        constraints: ModelConstraints,
        features_df: pd.DataFrame,
        target: pd.Series
    ) -> Dict[str, Any]:
        """Validate sample weights"""
        if constraints.sample_weights is None:
            return {'status': 'skipped', 'reason': 'No sample weights'}
        
        weights = constraints.sample_weights
        
        return {
            'status': 'valid',
            'mean_weight': float(weights.mean()),
            'std_weight': float(weights.std()),
            'min_weight': float(weights.min()),
            'max_weight': float(weights.max()),
            'weight_range': float(weights.max() - weights.min())
        }
    
    def _validate_feature_caps(
        self,
        constraints: ModelConstraints,
        features_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Validate feature caps"""
        if not constraints.feature_caps:
            return {'status': 'skipped', 'reason': 'No feature caps'}
        
        caps_info = {}
        for feature, (min_cap, max_cap) in constraints.feature_caps.items():
            if feature in features_df.columns:
                feature_data = features_df[feature].dropna()
                caps_info[feature] = {
                    'min_cap': min_cap,
                    'max_cap': max_cap,
                    'data_min': float(feature_data.min()),
                    'data_max': float(feature_data.max()),
                    'capped_min': feature_data[feature_data < min_cap].count(),
                    'capped_max': feature_data[feature_data > max_cap].count()
                }
        
        return {
            'status': 'valid',
            'capped_features': len(caps_info),
            'caps_details': caps_info
        }
    
    def get_constraint_summary(self) -> Dict[str, Any]:
        """Get summary of current constraints"""
        if not self.current_constraints:
            return {'status': 'no_constraints'}
        
        summary = {
            'monotone_constraints': {
                'enabled': bool(self.current_constraints.monotone_constraints),
                'count': len(self.current_constraints.monotone_constraints) if self.current_constraints.monotone_constraints else 0
            },
            'sample_weights': {
                'enabled': self.current_constraints.sample_weights is not None,
                'mean': float(self.current_constraints.sample_weights.mean()) if self.current_constraints.sample_weights is not None else None
            },
            'feature_caps': {
                'enabled': bool(self.current_constraints.feature_caps),
                'count': len(self.current_constraints.feature_caps) if self.current_constraints.feature_caps else 0
            }
        }
        
        return summary


# Convenience functions
def create_constraint_manager(config: Optional[ConstraintConfig] = None) -> ModelConstraintManager:
    """Create a new model constraint manager"""
    return ModelConstraintManager(config)


def get_default_constraint_config() -> ConstraintConfig:
    """Get default constraint configuration"""
    return ConstraintConfig(
        enable_monotone_constraints=True,
        enable_sample_weights=True,
        enable_feature_caps=True,
        weight_uncertainty_factor=0.3,
        min_sample_weight=0.1,
        max_sample_weight=2.0,
        monotone_strength=1.0
    )
    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and self.use_vectorbt and 
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and 
                VECTORBT_AVAILABLE)
    
    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str, 
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
        
        try:
            if operation == 'mean':
                return rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return rolling_sum(data, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
    
    def _pandas_rolling_operation(self, data: pd.Series, operation: str, 
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return data.rolling(window=window).mean()
        elif operation == 'std':
            return data.rolling(window=window).std()
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return data.rolling(window=window).min()
        elif operation == 'max':
            return data.rolling(window=window).max()
        elif operation == 'sum':
            return data.rolling(window=window).sum()
        else:
            raise ValueError(f"Unsupported operation: {operation}")
