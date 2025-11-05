"""
Validation utilities for Statsmodels Clustering

This module provides validation functions for statsmodels regime switching models.
"""

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

# Import utilities
try:
    from src.utils.tprint import (
        tprint_info, tprint_success, tprint_warning, tprint_error
    )
except ImportError:
    def tprint_info(msg): print(f'ℹ️  {msg}')
    def tprint_success(msg): print(f'✅ {msg}')
    def tprint_warning(msg): print(f'⚠️  {msg}')
    def tprint_error(msg): print(f'❌ {msg}')


class ValidationConfig:
    """Configuration for model validation."""
    
    def __init__(self, 
                 min_samples: int = 100,
                 max_regimes: int = 10,
                 cv_folds: int = 5,
                 test_size: float = 0.2):
        self.min_samples = min_samples
        self.max_regimes = max_regimes
        self.cv_folds = cv_folds
        self.test_size = test_size


class ValidationResult:
    """Result container for validation operations."""
    
    def __init__(self,
                 success: bool = True,
                 is_valid: bool = True,
                 warnings: Optional[List[str]] = None,
                 errors: Optional[List[str]] = None,
                 metrics: Optional[Dict[str, Any]] = None):
        self.success = success
        self.is_valid = is_valid
        self.warnings = warnings or []
        self.errors = errors or []
        self.metrics = metrics or {}


class ModelValidator:
    """Validate models and data for regime switching analysis."""
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()
        tprint_info("🔧 Initialized Model Validator")
    
    def validate_input_data(self, data: pd.DataFrame) -> ValidationResult:
        """Validate input data for regime switching models."""
        tprint_info("🔍 Validating input data for regime switching models")
        
        try:
            warnings = []
            errors = []
            
            # Check if data is empty
            if len(data) == 0:
                tprint_error("❌ Data is empty")
                errors.append("Data is empty")
                return ValidationResult(success=False, is_valid=False, errors=errors)
            
            # Check minimum samples
            if len(data) < self.config.min_samples:
                warning_msg = f"Data has only {len(data)} samples, minimum recommended is {self.config.min_samples}"
                tprint_warning(f"⚠️ {warning_msg}")
                warnings.append(warning_msg)
            
            # Check for missing values
            missing_values = data.isnull().sum().sum()
            if missing_values > 0:
                warning_msg = f"Data contains {missing_values} missing values"
                tprint_warning(f"⚠️ {warning_msg}")
                warnings.append(warning_msg)
            
            # Check for infinite values
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            infinite_values = np.isinf(data[numeric_cols]).sum().sum()
            if infinite_values > 0:
                warning_msg = f"Data contains {infinite_values} infinite values"
                tprint_warning(f"⚠️ {warning_msg}")
                warnings.append(warning_msg)
            
            is_valid = len(errors) == 0
            
            tprint_info(f"📊 Data shape: {data.shape}")
            tprint_info(f"📊 Sample count: {len(data)}, Feature count: {len(data.columns)}")
            
            result = ValidationResult(
                success=True,
                is_valid=is_valid,
                warnings=warnings,
                errors=errors,
                metrics={
                    "sample_count": len(data),
                    "feature_count": len(data.columns),
                    "missing_values": missing_values,
                    "infinite_values": infinite_values
                }
            )
            
            if is_valid:
                tprint_success("✅ Input data validation passed")
            else:
                tprint_warning("⚠️ Input data validation failed")
            
            return result
        except Exception as e:
            tprint_error(f"❌ Input data validation failed: {e}")
            return ValidationResult(
                success=False,
                is_valid=False,
                errors=[f"Validation failed: {str(e)}"]
            )
    
    def validate_model_fit(self, model: Any, data: pd.DataFrame) -> ValidationResult:
        """Validate fitted model."""
        tprint_info("🔍 Validating fitted model")
        
        try:
            warnings = []
            errors = []
            
            # Check if model has required attributes
            tprint_info("📊 Checking for required model attributes")
            required_attrs = ['params', 'llf', 'aic', 'bic']
            for attr in required_attrs:
                if not hasattr(model, attr):
                    error_msg = f"Model missing required attribute: {attr}"
                    tprint_error(f"❌ {error_msg}")
                    errors.append(error_msg)
            
            # Check parameter values
            if hasattr(model, 'params'):
                tprint_info("📊 Checking model parameter values")
                if np.any(np.isnan(model.params)):
                    warning_msg = "Model parameters contain NaN values"
                    tprint_warning(f"⚠️ {warning_msg}")
                    warnings.append(warning_msg)
                
                if np.any(np.isinf(model.params)):
                    warning_msg = "Model parameters contain infinite values"
                    tprint_warning(f"⚠️ {warning_msg}")
                    warnings.append(warning_msg)
            
            is_valid = len(errors) == 0
            
            # Extract metrics
            log_likelihood = getattr(model, 'llf', None)
            aic = getattr(model, 'aic', None)
            bic = getattr(model, 'bic', None)
            param_count = len(getattr(model, 'params', []))
            
            if log_likelihood is not None:
                tprint_info(f"📈 Log likelihood: {log_likelihood:.4f}")
            if aic is not None:
                tprint_info(f"📈 AIC: {aic:.4f}")
            if bic is not None:
                tprint_info(f"📈 BIC: {bic:.4f}")
            tprint_info(f"📊 Parameter count: {param_count}")
            
            result = ValidationResult(
                success=True,
                is_valid=is_valid,
                warnings=warnings,
                errors=errors,
                metrics={
                    "log_likelihood": log_likelihood,
                    "aic": aic,
                    "bic": bic,
                    "param_count": param_count
                }
            )
            
            if is_valid:
                tprint_success("✅ Model fit validation passed")
            else:
                tprint_warning("⚠️ Model fit validation failed")
            
            return result
        except Exception as e:
            tprint_error(f"❌ Model fit validation failed: {e}")
            return ValidationResult(
                success=False,
                is_valid=False,
                errors=[f"Model validation failed: {str(e)}"]
            )
    
    def cross_validate_regime_model(self, model_class: Any, data: pd.DataFrame, k_regimes: int) -> ValidationResult:
        """Perform cross-validation for regime models."""
        tprint_info(f"🔍 Performing cross-validation for regime model (k={k_regimes})")
        
        try:
            from sklearn.model_selection import TimeSeriesSplit
            
            warnings = []
            errors = []
            
            # Create time series split
            tprint_info(f"📊 Creating {self.config.cv_folds}-fold time series split")
            tscv = TimeSeriesSplit(n_splits=self.config.cv_folds)
            
            cv_scores = []
            for i, (train_idx, test_idx) in enumerate(tscv.split(data)):
                tprint_info(f"🔄 Processing CV fold {i+1}/{self.config.cv_folds}")
                try:
                    train_data = data.iloc[train_idx]
                    test_data = data.iloc[test_idx]
                    
                    tprint_info(f"📊 Train data shape: {train_data.shape}, Test data shape: {test_data.shape}")
                    
                    # Fit model on training data
                    model = model_class(k_regimes=k_regimes, trend='n')
                    model.fit(train_data)
                    
                    # Evaluate on test data
                    if hasattr(model, 'llf'):
                        cv_scores.append(model.llf)
                        tprint_info(f"📈 Fold {i+1} log likelihood: {model.llf:.4f}")
                    else:
                        warning_msg = "Model doesn't have log likelihood for CV scoring"
                        tprint_warning(f"⚠️ {warning_msg}")
                        warnings.append(warning_msg)
                        
                except Exception as e:
                    warning_msg = f"CV fold {i+1} failed: {str(e)}"
                    tprint_warning(f"⚠️ {warning_msg}")
                    warnings.append(warning_msg)
            
            if len(cv_scores) == 0:
                tprint_error("❌ All CV folds failed")
                errors.append("All CV folds failed")
                return ValidationResult(success=False, is_valid=False, errors=errors)
            
            is_valid = len(errors) == 0
            
            # Calculate CV metrics
            mean_score = np.mean(cv_scores)
            std_score = np.std(cv_scores)
            
            tprint_info(f"📊 CV mean score: {mean_score:.4f}")
            tprint_info(f"📊 CV std score: {std_score:.4f}")
            tprint_info(f"📊 Successful folds: {len(cv_scores)}/{self.config.cv_folds}")
            
            result = ValidationResult(
                success=True,
                is_valid=is_valid,
                warnings=warnings,
                errors=errors,
                metrics={
                    "cv_scores": cv_scores,
                    "mean_score": mean_score,
                    "std_score": std_score,
                    "successful_folds": len(cv_scores),
                    "total_folds": self.config.cv_folds
                }
            )
            
            if is_valid:
                tprint_success("✅ Cross-validation completed successfully")
            else:
                tprint_warning("⚠️ Cross-validation completed with warnings")
            
            return result
        except Exception as e:
            tprint_error(f"❌ Cross-validation failed: {e}")
            return ValidationResult(
                success=False,
                is_valid=False,
                errors=[f"Cross-validation failed: {str(e)}"]
            )


def validate_input_data(data: pd.DataFrame, config: Optional[ValidationConfig] = None) -> ValidationResult:
    """Convenience function to validate input data."""
    tprint_info("🏭 Convenience function: validating input data")
    validator = ModelValidator(config)
    result = validator.validate_input_data(data)
    if result.is_valid:
        tprint_success("✅ Input data validation complete")
    else:
        tprint_warning("⚠️ Input data validation complete with issues")
    return result


def validate_model_fit(model: Any, data: pd.DataFrame, config: Optional[ValidationConfig] = None) -> ValidationResult:
    """Convenience function to validate model fit."""
    tprint_info("🏭 Convenience function: validating model fit")
    validator = ModelValidator(config)
    result = validator.validate_model_fit(model, data)
    if result.is_valid:
        tprint_success("✅ Model fit validation complete")
    else:
        tprint_warning("⚠️ Model fit validation complete with issues")
    return result


def cross_validate_regime_model(model_class: Any, data: pd.DataFrame, k_regimes: int, config: Optional[ValidationConfig] = None) -> ValidationResult:
    """Convenience function to perform cross-validation."""
    tprint_info("🏭 Convenience function: performing cross-validation")
    validator = ModelValidator(config)
    result = validator.cross_validate_regime_model(model_class, data, k_regimes)
    if result.is_valid:
        tprint_success("✅ Cross-validation complete")
    else:
        tprint_warning("⚠️ Cross-validation complete with issues")
    return result