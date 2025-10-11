"""
Generalized Validation Utilities for Pre-Training Steps.

This module provides a unified interface for data validation across all pre-training
steps, using the enhanced fast_failing_validation module with comprehensive logging.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass
from enum import Enum

# Import the enhanced validation utilities
from ..feature_lookback_optimization.utils.fast_failing_validation import (
    FastFailingValidator, ValidationResult, ValidationSeverity,
    validate_dataframe_basic, validate_feature_data, validate_target_data,
    validate_preprocessing_inputs, validate_model_inputs,
    validate_optimization_inputs_fast_fail, validate_feature_calculation_inputs
)

# Import tprint utilities
try:
    from src.utils.tprint import (
        tprint, tprint_debug, tprint_info, tprint_warning, 
        tprint_error, tprint_success, tprint_performance
    )
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print("TPRINT:", *args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_performance(*args, **kwargs): print("PERF:", *args, **kwargs)


class ValidationContext(Enum):
    """Pre-defined validation contexts for different pre-training and model training steps."""
    # Pre-training contexts
    FEATURE_GENERATION = "feature_generation"
    FEATURE_SELECTION = "feature_selection"
    DATA_PREPROCESSING = "data_preprocessing"
    CROSS_VALIDATION = "cross_validation"
    HYPERPARAMETER_TUNING = "hyperparameter_tuning"
    FEATURE_OPTIMIZATION = "feature_optimization"
    LABEL_GENERATION = "label_generation"
    INTERACTION_FEATURES = "interaction_features"
    FINAL_SELECTION = "final_selection"
    
    # Model training contexts
    MODEL_TRAINING = "model_training"
    ENSEMBLE_TRAINING = "ensemble_training"
    TACTICIAN_TRAINING = "tactician_training"
    ANALYST_TRAINING = "analyst_training"
    NAS_TAS_TRAINING = "nas_tas_training"
    REGIME_AWARE_TRAINING = "regime_aware_training"
    MODEL_VALIDATION = "model_validation"
    MODEL_DEPLOYMENT = "model_deployment"
    MODEL_MONITORING = "model_monitoring"
    NEGATIVE_LEARNING = "negative_learning"


@dataclass
class ValidationConfig:
    """Configuration for validation operations."""
    min_samples: int = 100
    min_variance: float = 1e-8
    enable_logging: bool = True
    strict_mode: bool = True
    context: ValidationContext = ValidationContext.FEATURE_GENERATION


class PreTrainingValidator:
    """
    Unified validator for all pre-training steps with context-aware validation.
    """
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        """
        Initialize the pre-training validator.
        
        Args:
            config: Validation configuration
        """
        self.config = config or ValidationConfig()
        self.validator = FastFailingValidator(
            min_samples=self.config.min_samples,
            min_variance=self.config.min_variance,
            enable_logging=self.config.enable_logging,
            validation_context=self.config.context.value
        )
        self.validation_history = []
    
    def validate_dataframe(
        self, 
        data: pd.DataFrame,
        min_rows: int = None,
        min_cols: int = None,
        context: Optional[ValidationContext] = None
    ) -> ValidationResult:
        """
        Validate a DataFrame with context-aware logging.
        
        Args:
            data: Input DataFrame
            min_rows: Minimum number of rows (uses config default if None)
            min_cols: Minimum number of columns (uses config default if None)
            context: Validation context (uses config default if None)
            
        Returns:
            ValidationResult with validation status
        """
        context_str = context.value if context else self.config.context.value
        
        result = validate_dataframe_basic(
            data, 
            min_rows=min_rows or 1,
            min_cols=min_cols or 1,
            validation_context=context_str
        )
        
        self._record_validation("dataframe", result)
        return result
    
    def validate_features(
        self,
        data: pd.DataFrame,
        feature_columns: List[str],
        context: Optional[ValidationContext] = None
    ) -> ValidationResult:
        """
        Validate feature columns with context-aware logging.
        
        Args:
            data: Input DataFrame
            feature_columns: List of feature column names
            context: Validation context
            
        Returns:
            ValidationResult with validation status
        """
        context_str = context.value if context else self.config.context.value
        
        result = validate_feature_data(
            data,
            feature_columns,
            validation_context=context_str
        )
        
        self._record_validation("features", result)
        return result
    
    def validate_targets(
        self,
        data: pd.DataFrame,
        target_columns: List[str],
        context: Optional[ValidationContext] = None
    ) -> ValidationResult:
        """
        Validate target columns with context-aware logging.
        
        Args:
            data: Input DataFrame
            target_columns: List of target column names
            context: Validation context
            
        Returns:
            ValidationResult with validation status
        """
        context_str = context.value if context else self.config.context.value
        
        result = validate_target_data(
            data,
            target_columns,
            validation_context=context_str
        )
        
        self._record_validation("targets", result)
        return result
    
    def validate_preprocessing(
        self,
        data: pd.DataFrame,
        required_columns: List[str],
        context: Optional[ValidationContext] = None
    ) -> ValidationResult:
        """
        Validate preprocessing inputs with context-aware logging.
        
        Args:
            data: Input DataFrame
            required_columns: List of required column names
            context: Validation context
            
        Returns:
            ValidationResult with validation status
        """
        context_str = context.value if context else self.config.context.value
        
        result = validate_preprocessing_inputs(
            data,
            required_columns,
            validation_context=context_str
        )
        
        self._record_validation("preprocessing", result)
        return result
    
    def validate_model_inputs(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        context: Optional[ValidationContext] = None
    ) -> ValidationResult:
        """
        Validate model inputs with context-aware logging.
        
        Args:
            X: Feature matrix
            y: Target vector
            context: Validation context
            
        Returns:
            ValidationResult with validation status
        """
        context_str = context.value if context else self.config.context.value
        
        result = validate_model_inputs(
            X, y,
            validation_context=context_str
        )
        
        self._record_validation("model_inputs", result)
        return result
    
    def validate_optimization_data(
        self,
        data: pd.DataFrame,
        feature_columns: List[str],
        target_columns: List[str],
        lookback_range: Tuple[int, int],
        context: Optional[ValidationContext] = None
    ) -> ValidationResult:
        """
        Validate optimization data with context-aware logging.
        
        Args:
            data: Input DataFrame
            feature_columns: List of feature column names
            target_columns: List of target column names
            lookback_range: (min_lookback, max_lookback) tuple
            context: Validation context
            
        Returns:
            ValidationResult with validation status
        """
        context_str = context.value if context else self.config.context.value
        
        result = validate_optimization_inputs_fast_fail(
            data,
            feature_columns,
            target_columns,
            lookback_range,
            min_samples=self.config.min_samples,
            validation_context=context_str
        )
        
        self._record_validation("optimization", result)
        return result
    
    def validate_feature_calculation(
        self,
        data: pd.DataFrame,
        feature_name: str,
        lookback: int,
        context: Optional[ValidationContext] = None
    ) -> None:
        """
        Validate feature calculation inputs with context-aware logging.
        
        Args:
            data: Input DataFrame
            feature_name: Name of feature to calculate
            lookback: Lookback period
            context: Validation context
            
        Raises:
            DataValidationError: If inputs are invalid
        """
        context_str = context.value if context else self.config.context.value
        
        validate_feature_calculation_inputs(
            data, feature_name, lookback,
            validation_context=context_str
        )
    
    def _record_validation(self, validation_type: str, result: ValidationResult):
        """Record validation result in history."""
        self.validation_history.append({
            "type": validation_type,
            "timestamp": pd.Timestamp.now(),
            "result": result,
            "context": self.config.context.value
        })
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get a summary of all validations performed."""
        if not self.validation_history:
            return {"total_validations": 0, "success_rate": 0.0}
        
        total = len(self.validation_history)
        successful = sum(1 for record in record["result"].is_valid for record in self.validation_history)
        
        return {
            "total_validations": total,
            "successful_validations": successful,
            "success_rate": successful / total if total > 0 else 0.0,
            "context": self.config.context.value,
            "recent_validations": self.validation_history[-5:]  # Last 5 validations
        }
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get detailed validation statistics."""
        return self.validator.get_validation_stats()


# ============================================================================
# CONVENIENCE FUNCTIONS FOR COMMON VALIDATION PATTERNS
# ============================================================================

def validate_feature_generation_inputs(
    data: pd.DataFrame,
    feature_columns: List[str],
    required_columns: List[str] = None
) -> ValidationResult:
    """
    Validate inputs for feature generation steps.
    
    Args:
        data: Input DataFrame
        feature_columns: List of feature column names to generate
        required_columns: List of required input columns
        
    Returns:
        ValidationResult with validation status
    """
    validator = PreTrainingValidator(ValidationConfig(context=ValidationContext.FEATURE_GENERATION))
    
    # Basic DataFrame validation
    basic_result = validator.validate_dataframe(data)
    if not basic_result.is_valid:
        return basic_result
    
    # Validate required columns if provided
    if required_columns:
        preprocessing_result = validator.validate_preprocessing(data, required_columns)
        if not preprocessing_result.is_valid:
            return preprocessing_result
    
    # Validate existing features
    if feature_columns:
        feature_result = validator.validate_features(data, feature_columns)
        if not feature_result.is_valid:
            return feature_result
    
    return ValidationResult(
        is_valid=True,
        severity=ValidationSeverity.LOW,
        error_message="Feature generation inputs validation passed",
        details={
            "data_shape": data.shape,
            "feature_columns": feature_columns,
            "required_columns": required_columns or []
        },
        should_fail_fast=False
    )


def validate_feature_selection_inputs(
    data: pd.DataFrame,
    feature_columns: List[str],
    target_columns: List[str]
) -> ValidationResult:
    """
    Validate inputs for feature selection steps.
    
    Args:
        data: Input DataFrame
        feature_columns: List of feature column names
        target_columns: List of target column names
        
    Returns:
        ValidationResult with validation status
    """
    validator = PreTrainingValidator(ValidationConfig(context=ValidationContext.FEATURE_SELECTION))
    
    # Validate features
    feature_result = validator.validate_features(data, feature_columns)
    if not feature_result.is_valid:
        return feature_result
    
    # Validate targets
    target_result = validator.validate_targets(data, target_columns)
    if not target_result.is_valid:
        return target_result
    
    return ValidationResult(
        is_valid=True,
        severity=ValidationSeverity.LOW,
        error_message="Feature selection inputs validation passed",
        details={
            "data_shape": data.shape,
            "feature_columns": feature_columns,
            "target_columns": target_columns
        },
        should_fail_fast=False
    )


def validate_cross_validation_inputs(
    data: pd.DataFrame,
    feature_columns: List[str],
    target_columns: List[str],
    cv_folds: int = 5
) -> ValidationResult:
    """
    Validate inputs for cross-validation steps.
    
    Args:
        data: Input DataFrame
        feature_columns: List of feature column names
        target_columns: List of target column names
        cv_folds: Number of cross-validation folds
        
    Returns:
        ValidationResult with validation status
    """
    validator = PreTrainingValidator(ValidationConfig(context=ValidationContext.CROSS_VALIDATION))
    
    # Validate features and targets
    feature_result = validator.validate_features(data, feature_columns)
    if not feature_result.is_valid:
        return feature_result
    
    target_result = validator.validate_targets(data, target_columns)
    if not target_result.is_valid:
        return target_result
    
    # Check if we have enough data for CV
    min_samples_per_fold = len(data) // cv_folds
    if min_samples_per_fold < 10:  # Minimum 10 samples per fold
        return ValidationResult(
            is_valid=False,
            severity=ValidationSeverity.CRITICAL,
            error_message=f"Insufficient data for {cv_folds}-fold CV: {len(data)} samples, need at least {cv_folds * 10}",
            details={"cv_folds": cv_folds, "min_samples_per_fold": min_samples_per_fold},
            should_fail_fast=True
        )
    
    return ValidationResult(
        is_valid=True,
        severity=ValidationSeverity.LOW,
        error_message="Cross-validation inputs validation passed",
        details={
            "data_shape": data.shape,
            "feature_columns": feature_columns,
            "target_columns": target_columns,
            "cv_folds": cv_folds,
            "min_samples_per_fold": min_samples_per_fold
        },
        should_fail_fast=False
    )


def validate_label_generation_inputs(
    data: pd.DataFrame,
    price_columns: List[str],
    required_columns: List[str] = None
) -> ValidationResult:
    """
    Validate inputs for label generation steps.
    
    Args:
        data: Input DataFrame
        price_columns: List of price column names
        required_columns: List of additional required columns
        
    Returns:
        ValidationResult with validation status
    """
    validator = PreTrainingValidator(ValidationConfig(context=ValidationContext.LABEL_GENERATION))
    
    # Basic DataFrame validation
    basic_result = validator.validate_dataframe(data)
    if not basic_result.is_valid:
        return basic_result
    
    # Validate price columns
    price_result = validator.validate_features(data, price_columns)
    if not price_result.is_valid:
        return price_result
    
    # Validate additional required columns
    if required_columns:
        required_result = validator.validate_preprocessing(data, required_columns)
        if not required_result.is_valid:
            return required_result
    
    return ValidationResult(
        is_valid=True,
        severity=ValidationSeverity.LOW,
        error_message="Label generation inputs validation passed",
        details={
            "data_shape": data.shape,
            "price_columns": price_columns,
            "required_columns": required_columns or []
        },
        should_fail_fast=False
    )


# ============================================================================
# MODEL TRAINING VALIDATION FUNCTIONS
# ============================================================================

def validate_training_data(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    context: ValidationContext = ValidationContext.MODEL_TRAINING
) -> ValidationResult:
    """
    Validate training data for model training.
    
    Args:
        X: Feature matrix
        y: Target vector
        context: Validation context
        
    Returns:
        ValidationResult with validation status
    """
    validator = PreTrainingValidator(ValidationConfig(context=context))
    
    # Use the existing model inputs validation
    result = validator.validate_model_inputs(X, y)
    
    # Additional model training specific checks
    if result.is_valid:
        # Check for sufficient training samples
        n_samples = len(X) if hasattr(X, '__len__') else X.shape[0]
        if n_samples < 50:  # Minimum samples for training
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.CRITICAL,
                error_message=f"Insufficient training samples: {n_samples} < 50",
                details={"n_samples": n_samples, "min_required": 50},
                should_fail_fast=True
            )
        
        # Check for class balance in classification tasks
        if hasattr(y, 'value_counts'):  # pandas Series
            class_counts = y.value_counts()
            if len(class_counts) > 1:  # Multi-class
                min_class_count = class_counts.min()
                if min_class_count < 5:  # Minimum samples per class
                    return ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.HIGH,
                        error_message=f"Imbalanced classes: minimum class has {min_class_count} samples",
                        details={"class_counts": class_counts.to_dict()},
                        should_fail_fast=False
                    )
    
    return result


def validate_ensemble_training_inputs(
    training_data: pd.DataFrame,
    feature_columns: List[str],
    target_columns: List[str],
    base_models: List[Any],
    context: ValidationContext = ValidationContext.ENSEMBLE_TRAINING
) -> ValidationResult:
    """
    Validate inputs for ensemble training.
    
    Args:
        training_data: Training DataFrame
        feature_columns: List of feature column names
        target_columns: List of target column names
        base_models: List of base models for ensemble
        context: Validation context
        
    Returns:
        ValidationResult with validation status
    """
    validator = PreTrainingValidator(ValidationConfig(context=context))
    
    # Validate basic training data
    basic_result = validator.validate_dataframe(training_data)
    if not basic_result.is_valid:
        return basic_result
    
    # Validate features
    feature_result = validator.validate_features(training_data, feature_columns)
    if not feature_result.is_valid:
        return feature_result
    
    # Validate targets
    target_result = validator.validate_targets(training_data, target_columns)
    if not target_result.is_valid:
        return target_result
    
    # Validate base models
    if not base_models or len(base_models) == 0:
        return ValidationResult(
            is_valid=False,
            severity=ValidationSeverity.CRITICAL,
            error_message="No base models provided for ensemble training",
            details={"base_models_count": 0},
            should_fail_fast=True
        )
    
    # Check for sufficient data for ensemble training
    min_samples = len(base_models) * 20  # At least 20 samples per model
    if len(training_data) < min_samples:
        return ValidationResult(
            is_valid=False,
            severity=ValidationSeverity.CRITICAL,
            error_message=f"Insufficient data for ensemble training: {len(training_data)} < {min_samples}",
            details={"n_samples": len(training_data), "min_required": min_samples, "n_models": len(base_models)},
            should_fail_fast=True
        )
    
    return ValidationResult(
        is_valid=True,
        severity=ValidationSeverity.LOW,
        error_message="Ensemble training inputs validation passed",
        details={
            "data_shape": training_data.shape,
            "feature_columns": feature_columns,
            "target_columns": target_columns,
            "base_models_count": len(base_models)
        },
        should_fail_fast=False
    )


def validate_model_config(
    model_config: Dict[str, Any],
    context: ValidationContext = ValidationContext.MODEL_TRAINING
) -> ValidationResult:
    """
    Validate model configuration parameters.
    
    Args:
        model_config: Model configuration dictionary
        context: Validation context
        
    Returns:
        ValidationResult with validation status
    """
    if not isinstance(model_config, dict):
        return ValidationResult(
            is_valid=False,
            severity=ValidationSeverity.CRITICAL,
            error_message="Model config must be a dictionary",
            details={"config_type": type(model_config).__name__},
            should_fail_fast=True
        )
    
    required_keys = ['model_type', 'hyperparameters']
    missing_keys = [key for key in required_keys if key not in model_config]
    
    if missing_keys:
        return ValidationResult(
            is_valid=False,
            severity=ValidationSeverity.CRITICAL,
            error_message=f"Missing required config keys: {missing_keys}",
            details={"missing_keys": missing_keys, "required_keys": required_keys},
            should_fail_fast=True
        )
    
    # Validate hyperparameters
    hyperparams = model_config.get('hyperparameters', {})
    if not isinstance(hyperparams, dict):
        return ValidationResult(
            is_valid=False,
            severity=ValidationSeverity.CRITICAL,
            error_message="Hyperparameters must be a dictionary",
            details={"hyperparams_type": type(hyperparams).__name__},
            should_fail_fast=True
        )
    
    return ValidationResult(
        is_valid=True,
        severity=ValidationSeverity.LOW,
        error_message="Model configuration validation passed",
        details={
            "model_type": model_config.get('model_type'),
            "hyperparams_count": len(hyperparams),
            "config_keys": list(model_config.keys())
        },
        should_fail_fast=False
    )


def validate_regime_data(
    data: pd.DataFrame,
    regime_column: str,
    context: ValidationContext = ValidationContext.REGIME_AWARE_TRAINING
) -> ValidationResult:
    """
    Validate regime-aware training data.
    
    Args:
        data: Training DataFrame
        regime_column: Name of the regime column
        context: Validation context
        
    Returns:
        ValidationResult with validation status
    """
    validator = PreTrainingValidator(ValidationConfig(context=context))
    
    # Basic DataFrame validation
    basic_result = validator.validate_dataframe(data)
    if not basic_result.is_valid:
        return basic_result
    
    # Check regime column exists
    if regime_column not in data.columns:
        return ValidationResult(
            is_valid=False,
            severity=ValidationSeverity.CRITICAL,
            error_message=f"Regime column '{regime_column}' not found in data",
            details={"available_columns": list(data.columns), "regime_column": regime_column},
            should_fail_fast=True
        )
    
    # Check regime column has valid values
    regime_values = data[regime_column].dropna()
    if len(regime_values) == 0:
        return ValidationResult(
            is_valid=False,
            severity=ValidationSeverity.CRITICAL,
            error_message=f"Regime column '{regime_column}' contains only NaN values",
            details={"regime_column": regime_column},
            should_fail_fast=True
        )
    
    # Check for sufficient samples per regime
    regime_counts = regime_values.value_counts()
    min_samples_per_regime = 10
    insufficient_regimes = regime_counts[regime_counts < min_samples_per_regime]
    
    if len(insufficient_regimes) > 0:
        return ValidationResult(
            is_valid=False,
            severity=ValidationSeverity.HIGH,
            error_message=f"Some regimes have insufficient samples: {insufficient_regimes.to_dict()}",
            details={
                "regime_counts": regime_counts.to_dict(),
                "min_samples_per_regime": min_samples_per_regime,
                "insufficient_regimes": insufficient_regimes.to_dict()
            },
            should_fail_fast=False
        )
    
    return ValidationResult(
        is_valid=True,
        severity=ValidationSeverity.LOW,
        error_message="Regime data validation passed",
        details={
            "data_shape": data.shape,
            "regime_column": regime_column,
            "regime_counts": regime_counts.to_dict(),
            "n_regimes": len(regime_counts)
        },
        should_fail_fast=False
    )


def validate_nas_tas_inputs(
    data: pd.DataFrame,
    feature_columns: List[str],
    target_columns: List[str],
    architecture_config: Dict[str, Any],
    context: ValidationContext = ValidationContext.NAS_TAS_TRAINING
) -> ValidationResult:
    """
    Validate inputs for NAS-TAS (Neural Architecture Search - Tree-based Architecture Search) training.
    
    Args:
        data: Training DataFrame
        feature_columns: List of feature column names
        target_columns: List of target column names
        architecture_config: Architecture search configuration
        context: Validation context
        
    Returns:
        ValidationResult with validation status
    """
    validator = PreTrainingValidator(ValidationConfig(context=context))
    
    # Validate basic training data
    basic_result = validator.validate_dataframe(data)
    if not basic_result.is_valid:
        return basic_result
    
    # Validate features and targets
    feature_result = validator.validate_features(data, feature_columns)
    if not feature_result.is_valid:
        return feature_result
    
    target_result = validator.validate_targets(data, target_columns)
    if not target_result.is_valid:
        return target_result
    
    # Validate architecture config
    if not isinstance(architecture_config, dict):
        return ValidationResult(
            is_valid=False,
            severity=ValidationSeverity.CRITICAL,
            error_message="Architecture config must be a dictionary",
            details={"config_type": type(architecture_config).__name__},
            should_fail_fast=True
        )
    
    # Check for required architecture parameters
    required_params = ['search_space', 'max_trials', 'objective']
    missing_params = [param for param in required_params if param not in architecture_config]
    
    if missing_params:
        return ValidationResult(
            is_valid=False,
            severity=ValidationSeverity.CRITICAL,
            error_message=f"Missing required architecture parameters: {missing_params}",
            details={"missing_params": missing_params, "required_params": required_params},
            should_fail_fast=True
        )
    
    # Check for sufficient data for architecture search
    min_samples = architecture_config.get('max_trials', 100) * 5  # At least 5 samples per trial
    if len(data) < min_samples:
        return ValidationResult(
            is_valid=False,
            severity=ValidationSeverity.CRITICAL,
            error_message=f"Insufficient data for NAS-TAS: {len(data)} < {min_samples}",
            details={"n_samples": len(data), "min_required": min_samples, "max_trials": architecture_config.get('max_trials')},
            should_fail_fast=True
        )
    
    return ValidationResult(
        is_valid=True,
        severity=ValidationSeverity.LOW,
        error_message="NAS-TAS inputs validation passed",
        details={
            "data_shape": data.shape,
            "feature_columns": feature_columns,
            "target_columns": target_columns,
            "architecture_config": architecture_config
        },
        should_fail_fast=False
    )


def validate_negative_learning_inputs(
    data: pd.DataFrame,
    feature_columns: List[str],
    target_columns: List[str],
    negative_samples: pd.DataFrame,
    context: ValidationContext = ValidationContext.NEGATIVE_LEARNING
) -> ValidationResult:
    """
    Validate inputs for negative learning training.
    
    Args:
        data: Positive training DataFrame
        feature_columns: List of feature column names
        target_columns: List of target column names
        negative_samples: Negative samples DataFrame
        context: Validation context
        
    Returns:
        ValidationResult with validation status
    """
    validator = PreTrainingValidator(ValidationConfig(context=context))
    
    # Validate positive data
    positive_result = validator.validate_dataframe(data)
    if not positive_result.is_valid:
        return positive_result
    
    # Validate negative samples
    negative_result = validator.validate_dataframe(negative_samples)
    if not negative_result.is_valid:
        return negative_result
    
    # Validate features in both datasets
    feature_result = validator.validate_features(data, feature_columns)
    if not feature_result.is_valid:
        return feature_result
    
    # Check feature consistency between positive and negative data
    missing_features = set(feature_columns) - set(negative_samples.columns)
    if missing_features:
        return ValidationResult(
            is_valid=False,
            severity=ValidationSeverity.CRITICAL,
            error_message=f"Missing features in negative samples: {missing_features}",
            details={"missing_features": list(missing_features), "feature_columns": feature_columns},
            should_fail_fast=True
        )
    
    # Check for sufficient negative samples
    min_negative_ratio = 0.1  # At least 10% negative samples
    min_negative_samples = int(len(data) * min_negative_ratio)
    if len(negative_samples) < min_negative_samples:
        return ValidationResult(
            is_valid=False,
            severity=ValidationSeverity.HIGH,
            error_message=f"Insufficient negative samples: {len(negative_samples)} < {min_negative_samples}",
            details={
                "negative_samples": len(negative_samples),
                "min_required": min_negative_samples,
                "positive_samples": len(data),
                "min_ratio": min_negative_ratio
            },
            should_fail_fast=False
        )
    
    return ValidationResult(
        is_valid=True,
        severity=ValidationSeverity.LOW,
        error_message="Negative learning inputs validation passed",
        details={
            "positive_data_shape": data.shape,
            "negative_data_shape": negative_samples.shape,
            "feature_columns": feature_columns,
            "target_columns": target_columns,
            "negative_ratio": len(negative_samples) / len(data)
        },
        should_fail_fast=False
    )


# ============================================================================
# VALIDATION DECORATORS
# ============================================================================

def validate_inputs(validation_type: str, context: ValidationContext = ValidationContext.FEATURE_GENERATION):
    """
    Decorator to automatically validate function inputs.
    
    Args:
        validation_type: Type of validation to perform
        context: Validation context
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Extract data from function arguments
            data = None
            for arg in args:
                if isinstance(arg, pd.DataFrame):
                    data = arg
                    break
            
            if data is not None:
                validator = PreTrainingValidator(ValidationConfig(context=context))
                
                if validation_type == "dataframe":
                    result = validator.validate_dataframe(data)
                elif validation_type == "features":
                    feature_columns = kwargs.get('feature_columns', [])
                    result = validator.validate_features(data, feature_columns)
                elif validation_type == "targets":
                    target_columns = kwargs.get('target_columns', [])
                    result = validator.validate_targets(data, target_columns)
                elif validation_type == "training_data":
                    X = kwargs.get('X', data)
                    y = kwargs.get('y', None)
                    if y is not None:
                        result = validate_training_data(X, y, context)
                    else:
                        result = validator.validate_dataframe(data)
                else:
                    # Skip validation for unknown types
                    result = ValidationResult(True, ValidationSeverity.LOW, "Skipped", {})
                
                if not result.is_valid and result.should_fail_fast:
                    raise ValueError(f"Input validation failed: {result.error_message}")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator