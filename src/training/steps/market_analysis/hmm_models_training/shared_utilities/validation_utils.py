"""
Shared Validation Utilities

Provides centralized validation logic for HMM training components.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import logging

# Optional imports for external dependencies
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

logger = logging.getLogger(__name__)


class ValidationUtils:
    """Shared validation utilities for HMM training components."""
    
    @staticmethod
    def validate_data_shapes(X: Union[Any, Any], y: Any, regime_labels: Any) -> bool:
        """
        Validate data shapes are consistent.
        
        Args:
            X: Input features
            y: Target values
            regime_labels: Regime labels
            
        Returns:
            True if shapes are valid, False otherwise
        """
        if not NUMPY_AVAILABLE:
            logger.warning("NumPy not available, skipping shape validation")
            return True
            
        try:
            if len(X) != len(y):
                logger.error(f"Shape mismatch: X has {len(X)} samples, y has {len(y)} samples")
                return False
            
            if len(X) != len(regime_labels):
                logger.error(f"Shape mismatch: X has {len(X)} samples, regime_labels has {len(regime_labels)} samples")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Error validating data shapes: {e}")
            return False
    
    @staticmethod
    def validate_data_quality(X: Union[Any, Any], y: Any, regime_labels: Any) -> bool:
        """
        Validate data quality (no NaN, no infinite values).
        
        Args:
            X: Input features
            y: Target values
            regime_labels: Regime labels
            
        Returns:
            True if data quality is acceptable, False otherwise
        """
        if not NUMPY_AVAILABLE:
            logger.warning("NumPy not available, skipping data quality validation")
            return True
            
        try:
            # Check X for NaN and infinite values
            if NUMPY_AVAILABLE and isinstance(X, np.ndarray):
                if np.any(np.isnan(X)):
                    logger.error("X contains NaN values")
                    return False
                if np.any(np.isinf(X)):
                    logger.error("X contains infinite values")
                    return False
            elif PANDAS_AVAILABLE and isinstance(X, pd.DataFrame):
                if X.isnull().any().any():
                    logger.error("X contains NaN values")
                    return False
                if NUMPY_AVAILABLE and np.isinf(X.select_dtypes(include=[np.number])).any().any():
                    logger.error("X contains infinite values")
                    return False
            
            # Check y for NaN and infinite values
            if NUMPY_AVAILABLE:
                if np.any(np.isnan(y)):
                    logger.error("y contains NaN values")
                    return False
                if np.any(np.isinf(y)):
                    logger.error("y contains infinite values")
                    return False
                
                # Check regime_labels for NaN values
                if np.any(np.isnan(regime_labels)):
                    logger.error("regime_labels contains NaN values")
                    return False
            
            return True
        except Exception as e:
            logger.error(f"Error validating data quality: {e}")
            return False
    
    @staticmethod
    def validate_regime_distribution(regime_labels: Any, min_samples_per_regime: int = 10) -> bool:
        """
        Validate regime distribution is adequate.
        
        Args:
            regime_labels: Regime labels
            min_samples_per_regime: Minimum samples required per regime
            
        Returns:
            True if regime distribution is adequate, False otherwise
        """
        if not NUMPY_AVAILABLE:
            logger.warning("NumPy not available, skipping regime distribution validation")
            return True
            
        try:
            unique_regimes, regime_counts = np.unique(regime_labels, return_counts=True)
            
            if len(unique_regimes) < 2:
                logger.error("Need at least 2 regimes")
                return False
            
            min_count = np.min(regime_counts)
            if min_count < min_samples_per_regime:
                logger.error(f"Some regimes have insufficient samples (minimum: {min_count}, required: {min_samples_per_regime})")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Error validating regime distribution: {e}")
            return False
    
    @staticmethod
    def validate_config(config: Any) -> bool:
        """
        Validate configuration parameters with range checks.
        
        Args:
            config: Configuration object to validate
            
        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            # Check required attributes
            required_attrs = ['model_types', 'n_features', 'sequence_length', 'n_regimes', 'timeframe']
            for attr in required_attrs:
                if not hasattr(config, attr):
                    logger.error(f"Configuration missing required attribute: {attr}")
                    return False
            
            # Validate model types
            if not config.model_types or len(config.model_types) == 0:
                logger.error("No model types specified")
                return False
            
            # Enhanced: Validate numeric parameters with ranges
            if config.n_features <= 0:
                logger.error("n_features must be positive")
                return False
            elif config.n_features > 10000:
                logger.warning(f"n_features is very large ({config.n_features}), may cause performance issues")
            
            if config.sequence_length <= 0:
                logger.error("sequence_length must be positive")
                return False
            elif config.sequence_length > 1000:
                logger.warning(f"sequence_length is very large ({config.sequence_length}), may cause memory issues")
            
            if config.n_regimes < 2:
                logger.error("n_regimes must be at least 2")
                return False
            elif config.n_regimes > 20:
                logger.warning(f"n_regimes is very large ({config.n_regimes}), may cause overfitting")
            
            # Enhanced: Validate HPO parameters if present
            if hasattr(config, 'hpo_trials'):
                if config.hpo_trials < 0:
                    logger.error("hpo_trials must be non-negative")
                    return False
                elif config.hpo_trials > 1000:
                    logger.warning(f"hpo_trials is very large ({config.hpo_trials}), may take very long")
            
            # Enhanced: Validate learning rate if present
            if hasattr(config, 'learning_rate'):
                if config.learning_rate <= 0 or config.learning_rate > 1:
                    logger.error("learning_rate must be between 0 and 1")
                    return False
            
            # Enhanced: Validate batch size if present
            if hasattr(config, 'batch_size'):
                if config.batch_size <= 0:
                    logger.error("batch_size must be positive")
                    return False
                elif config.batch_size > 10000:
                    logger.warning(f"batch_size is very large ({config.batch_size}), may cause memory issues")
            
            # Validate timeframe
            valid_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
            if config.timeframe not in valid_timeframes:
                logger.error(f"Invalid timeframe: {config.timeframe}. Valid timeframes: {valid_timeframes}")
                return False
            
            # Enhanced: Cross-parameter validation
            if hasattr(config, 'n_features') and hasattr(config, 'sequence_length'):
                total_params = config.n_features * config.sequence_length
                if total_params > 1000000:
                    logger.warning(f"Total parameter space is very large ({total_params}), may cause memory issues")
            
            return True
        except Exception as e:
            logger.error(f"Error validating configuration: {e}")
            return False
    
    @staticmethod
    def validate_model_type(model_type: str, available_types: List[str]) -> bool:
        """
        Validate model type is supported.
        
        Args:
            model_type: Model type to validate
            available_types: List of available model types
            
        Returns:
            True if model type is valid, False otherwise
        """
        if model_type not in available_types:
            logger.error(f"Invalid model type: {model_type}. Available types: {available_types}")
            return False
        return True
    
    @staticmethod
    def comprehensive_validation(
        X: Union[Any, Any], 
        y: Any, 
        regime_labels: Any,
        config: Any,
        min_samples_per_regime: int = 10
    ) -> Tuple[bool, List[str]]:
        """
        Perform comprehensive validation of all inputs.
        
        Args:
            X: Input features
            y: Target values
            regime_labels: Regime labels
            config: Configuration object
            min_samples_per_regime: Minimum samples per regime
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Validate shapes
        if not ValidationUtils.validate_data_shapes(X, y, regime_labels):
            errors.append("Data shape validation failed")
        
        # Validate data quality
        if not ValidationUtils.validate_data_quality(X, y, regime_labels):
            errors.append("Data quality validation failed")
        
        # Validate regime distribution
        if not ValidationUtils.validate_regime_distribution(regime_labels, min_samples_per_regime):
            errors.append("Regime distribution validation failed")
        
        # Validate configuration
        if not ValidationUtils.validate_config(config):
            errors.append("Configuration validation failed")
        
        is_valid = len(errors) == 0
        return is_valid, errors