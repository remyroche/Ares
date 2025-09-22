"""
Universal Validation Integration for ML Training Pipelines

Automatically wires universal validation into all ML training/optimization pipelines
by default, ensuring comprehensive validation for all models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass
import logging
from datetime import datetime
import json
from pathlib import Path

# Import universal validation components
from ..validation import (
    validate_ml_model,
    get_ml_validator,
    get_overfitting_detector,
    get_temporal_validator,
    UniversalMLValidationConfig,
    OverfittingConfig,
    TemporalValidationConfig
)
from ..config.universal_timeframe_config import (
    get_timeframe_manager,
    validate_timeframe_consistency
)

logger = logging.getLogger(__name__)

@dataclass
class ValidationIntegrationConfig:
    """Configuration for validation integration into training pipelines."""
    
    # Validation settings
    enable_validation: bool = True
    enable_overfitting_detection: bool = True
    enable_temporal_validation: bool = True
    enable_timeframe_validation: bool = True
    
    # Validation thresholds
    validation_failure_threshold: float = 0.5  # Minimum validation score to pass
    critical_issue_threshold: int = 1  # Number of critical issues to fail
    
    # Reporting settings
    save_validation_reports: bool = True
    validation_report_directory: str = "reports/validation"
    enable_validation_logging: bool = True
    
    # Integration behavior
    fail_on_validation_error: bool = False  # Whether to fail training on validation errors
    warn_on_validation_issues: bool = True  # Whether to warn on validation issues
    auto_fix_validation_issues: bool = False  # Whether to attempt automatic fixes
    
    # Model-specific settings
    model_validation_overrides: Dict[str, Dict[str, Any]] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.model_validation_overrides is None:
            self.model_validation_overrides = {}

class UniversalValidationIntegrator:
    """Integrates universal validation into ML training pipelines."""
    
    def __init__(self, config: Optional[ValidationIntegrationConfig] = None):
        """
        Initialize validation integrator.
        
        Args:
            config: Validation integration configuration
        """
        self.config = config or ValidationIntegrationConfig()
        
        # Initialize validation components
        self.overfitting_detector = get_overfitting_detector()
        self.temporal_validator = get_temporal_validator()
        self.timeframe_manager = get_timeframe_manager()
        
        # Create validation report directory
        if self.config.save_validation_reports:
            Path(self.config.validation_report_directory).mkdir(parents=True, exist_ok=True)
        
        # Track validation history
        self.validation_history = []
        
        logger.info("✅ Universal Validation Integrator initialized")
    
    def validate_training_data(self, 
                              X_train: np.ndarray,
                              X_val: np.ndarray,
                              y_train: np.ndarray,
                              y_val: np.ndarray,
                              timestamps: Optional[np.ndarray] = None,
                              feature_names: Optional[List[str]] = None,
                              model_type: str = "unknown") -> Dict[str, Any]:
        """
        Validate training data before model training.
        
        Args:
            X_train: Training features
            X_val: Validation features
            y_train: Training labels
            y_val: Validation labels
            timestamps: Optional timestamps for temporal validation
            feature_names: Optional feature names
            model_type: Type of model
            
        Returns:
            Dict: Validation results
        """
        if not self.config.enable_validation:
            return {'valid': True, 'message': 'Validation disabled'}
        
        validation_results = {
            'valid': True,
            'warnings': [],
            'critical_issues': [],
            'recommendations': []
        }
        
        try:
            # 1. Timeframe validation
            if self.config.enable_timeframe_validation:
                timeframe_valid = self._validate_timeframe(model_type)
                if not timeframe_valid:
                    validation_results['warnings'].append("Timeframe validation failed")
                    if self.config.fail_on_validation_error:
                        validation_results['valid'] = False
                        validation_results['critical_issues'].append("Timeframe validation failed")
            
            # 2. Temporal validation
            if self.config.enable_temporal_validation and timestamps is not None:
                temporal_report = self.temporal_validator.validate_temporal_split(
                    X_train, X_val, y_train, y_val, timestamps, 
                    f"TrainingData_{model_type}", model_type
                )
                
                if not temporal_report.temporal_order_valid:
                    validation_results['warnings'].append("Temporal order violation detected")
                    if self.config.fail_on_validation_error:
                        validation_results['valid'] = False
                        validation_results['critical_issues'].append("Temporal order violation")
                
                if temporal_report.leakage_detected:
                    validation_results['warnings'].append("Data leakage detected")
                    validation_results['critical_issues'].append("Data leakage detected")
                    validation_results['valid'] = False
            
            # 3. Data quality validation
            data_quality_issues = self._validate_data_quality(X_train, X_val, y_train, y_val)
            validation_results['warnings'].extend(data_quality_issues)
            
            if len(data_quality_issues) > 0:
                validation_results['recommendations'].append("Review data quality issues")
            
            # 4. Generate recommendations
            if not validation_results['valid']:
                validation_results['recommendations'].extend([
                    "Fix critical validation issues before training",
                    "Review data preprocessing pipeline",
                    "Check for data leakage and temporal order"
                ])
            
            # Log validation results
            if self.config.enable_validation_logging:
                self._log_validation_results("TrainingData", validation_results)
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Training data validation failed: {e}")
            return {
                'valid': False,
                'warnings': [f"Validation error: {str(e)}"],
                'critical_issues': [f"Validation failed: {str(e)}"],
                'recommendations': ["Fix validation error and retry"]
            }
    
    def validate_trained_model(self, 
                              model: Any,
                              X_train: np.ndarray,
                              X_val: np.ndarray,
                              y_train: np.ndarray,
                              y_val: np.ndarray,
                              timestamps: Optional[np.ndarray] = None,
                              feature_names: Optional[List[str]] = None,
                              model_name: str = "unknown",
                              model_type: str = "unknown",
                              fold_number: Optional[int] = None) -> Dict[str, Any]:
        """
        Validate trained model with comprehensive analysis.
        
        Args:
            model: Trained ML model
            X_train: Training features
            X_val: Validation features
            y_train: Training labels
            y_val: Validation labels
            timestamps: Optional timestamps for temporal validation
            feature_names: Optional feature names
            model_name: Name of the model
            model_type: Type of model
            fold_number: Fold number for cross-validation
            
        Returns:
            Dict: Comprehensive validation results
        """
        if not self.config.enable_validation:
            return {'valid': True, 'message': 'Validation disabled'}
        
        try:
            # Get model-specific validation overrides
            model_config = self.config.model_validation_overrides.get(model_type, {})
            
            # Create validation configuration
            validation_config = UniversalMLValidationConfig(
                enable_overfitting_detection=self.config.enable_overfitting_detection,
                enable_temporal_validation=self.config.enable_temporal_validation,
                enable_timeframe_validation=self.config.enable_timeframe_validation,
                save_comprehensive_reports=self.config.save_validation_reports,
                report_directory=self.config.validation_report_directory,
                enable_visualization=True,
                detailed_logging=self.config.enable_validation_logging
            )
            
            # Override with model-specific settings
            for key, value in model_config.items():
                if hasattr(validation_config, key):
                    setattr(validation_config, key, value)
            
            # Get validator with configuration
            validator = get_ml_validator(validation_config)
            
            # Perform comprehensive validation
            validation_report = validator.validate_model(
                model=model,
                X_train=X_train,
                X_val=X_val,
                y_train=y_train,
                y_val=y_val,
                timestamps=timestamps,
                feature_names=feature_names,
                model_name=model_name,
                model_type=model_type,
                fold_number=fold_number
            )
            
            # Evaluate validation results
            validation_results = {
                'valid': validation_report.overall_validation_passed,
                'validation_score': validation_report.validation_score,
                'warnings': validation_report.warnings,
                'critical_issues': validation_report.critical_issues,
                'recommendations': validation_report.recommendations,
                'overfitting_analysis': validation_report.overfitting_analysis,
                'temporal_validation': validation_report.temporal_validation,
                'timeframe_validation': validation_report.timeframe_validation
            }
            
            # Check validation thresholds
            if validation_report.validation_score < self.config.validation_failure_threshold:
                validation_results['valid'] = False
                validation_results['critical_issues'].append(
                    f"Validation score {validation_report.validation_score:.3f} below threshold {self.config.validation_failure_threshold}"
                )
            
            if len(validation_report.critical_issues) >= self.config.critical_issue_threshold:
                validation_results['valid'] = False
            
            # Track validation history
            self.validation_history.append({
                'timestamp': datetime.now().isoformat(),
                'model_name': model_name,
                'model_type': model_type,
                'fold_number': fold_number,
                'validation_score': validation_report.validation_score,
                'valid': validation_results['valid'],
                'critical_issues': len(validation_report.critical_issues)
            })
            
            # Log validation results
            if self.config.enable_validation_logging:
                self._log_validation_results(model_name, validation_results)
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Model validation failed for {model_name}: {e}")
            return {
                'valid': False,
                'validation_score': 0.0,
                'warnings': [f"Validation error: {str(e)}"],
                'critical_issues': [f"Model validation failed: {str(e)}"],
                'recommendations': ["Fix validation error and retry"]
            }
    
    def validate_hpo_trial(self, 
                          model: Any,
                          X_train: np.ndarray,
                          X_val: np.ndarray,
                          y_train: np.ndarray,
                          y_val: np.ndarray,
                          trial_params: Dict[str, Any],
                          model_name: str = "unknown",
                          model_type: str = "unknown",
                          trial_number: int = 0) -> Dict[str, Any]:
        """
        Validate HPO trial with comprehensive analysis.
        
        Args:
            model: Trained model from HPO trial
            X_train: Training features
            X_val: Validation features
            y_train: Training labels
            y_val: Validation labels
            trial_params: HPO trial parameters
            model_name: Name of the model
            model_type: Type of model
            trial_number: HPO trial number
            
        Returns:
            Dict: HPO trial validation results
        """
        if not self.config.enable_validation:
            return {'valid': True, 'message': 'Validation disabled'}
        
        try:
            # Validate the model
            validation_results = self.validate_trained_model(
                model=model,
                X_train=X_train,
                X_val=X_val,
                y_train=y_train,
                y_val=y_val,
                model_name=f"{model_name}_trial_{trial_number}",
                model_type=model_type,
                fold_number=trial_number
            )
            
            # Add HPO-specific analysis
            hpo_validation = {
                'trial_number': trial_number,
                'trial_params': trial_params,
                'validation_score': validation_results['validation_score'],
                'valid': validation_results['valid'],
                'warnings': validation_results['warnings'],
                'critical_issues': validation_results['critical_issues'],
                'recommendations': validation_results['recommendations']
            }
            
            # Check if trial should be pruned based on validation
            if not validation_results['valid']:
                hpo_validation['should_prune'] = True
                hpo_validation['prune_reason'] = "Validation failed"
            elif validation_results['validation_score'] < self.config.validation_failure_threshold:
                hpo_validation['should_prune'] = True
                hpo_validation['prune_reason'] = f"Low validation score: {validation_results['validation_score']:.3f}"
            else:
                hpo_validation['should_prune'] = False
            
            return hpo_validation
            
        except Exception as e:
            logger.error(f"HPO trial validation failed for {model_name} trial {trial_number}: {e}")
            return {
                'trial_number': trial_number,
                'trial_params': trial_params,
                'valid': False,
                'validation_score': 0.0,
                'should_prune': True,
                'prune_reason': f"Validation error: {str(e)}",
                'warnings': [f"HPO trial validation error: {str(e)}"],
                'critical_issues': [f"HPO trial validation failed: {str(e)}"],
                'recommendations': ["Fix validation error and retry"]
            }
    
    def _validate_timeframe(self, model_type: str) -> bool:
        """Validate timeframe for model type."""
        try:
            primary_timeframe = self.timeframe_manager.config.primary_timeframe
            return self.timeframe_manager.validate_timeframe_consistency(
                primary_timeframe, model_type, f"TrainingPipeline_{model_type}"
            )
        except Exception as e:
            logger.error(f"Timeframe validation failed: {e}")
            return False
    
    def _validate_data_quality(self, 
                              X_train: np.ndarray,
                              X_val: np.ndarray,
                              y_train: np.ndarray,
                              y_val: np.ndarray) -> List[str]:
        """Validate data quality."""
        warnings = []
        
        try:
            # Check for empty data
            if len(X_train) == 0 or len(X_val) == 0:
                warnings.append("Empty training or validation data")
            
            # Check for NaN values
            if np.isnan(X_train).any():
                warnings.append("NaN values found in training data")
            if np.isnan(X_val).any():
                warnings.append("NaN values found in validation data")
            
            # Check for infinite values
            if np.isinf(X_train).any():
                warnings.append("Infinite values found in training data")
            if np.isinf(X_val).any():
                warnings.append("Infinite values found in validation data")
            
            # Check for constant features
            constant_features = np.var(X_train, axis=0) == 0
            if constant_features.any():
                warnings.append(f"Constant features found: {np.sum(constant_features)}")
            
            # Check for high correlation between train and val
            if X_train.shape[1] > 1 and X_val.shape[1] > 1:
                train_mean = np.mean(X_train, axis=0)
                val_mean = np.mean(X_val, axis=0)
                correlation = np.corrcoef(train_mean, val_mean)[0, 1]
                if correlation > 0.95:
                    warnings.append(f"High correlation between train and val means: {correlation:.3f}")
            
        except Exception as e:
            logger.error(f"Data quality validation failed: {e}")
            warnings.append(f"Data quality validation error: {str(e)}")
        
        return warnings
    
    def _log_validation_results(self, model_name: str, validation_results: Dict[str, Any]):
        """Log validation results."""
        logger.info(f"Validation results for {model_name}:")
        logger.info(f"  Valid: {validation_results['valid']}")
        logger.info(f"  Validation Score: {validation_results.get('validation_score', 'N/A')}")
        logger.info(f"  Warnings: {len(validation_results.get('warnings', []))}")
        logger.info(f"  Critical Issues: {len(validation_results.get('critical_issues', []))}")
        logger.info(f"  Recommendations: {len(validation_results.get('recommendations', []))}")
        
        if validation_results.get('critical_issues'):
            for issue in validation_results['critical_issues']:
                logger.error(f"  Critical: {issue}")
        
        if validation_results.get('warnings'):
            for warning in validation_results['warnings'][:3]:  # Show first 3 warnings
                logger.warning(f"  Warning: {warning}")
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of all validations."""
        if not self.validation_history:
            return {'message': 'No validations performed'}
        
        total_validations = len(self.validation_history)
        valid_validations = sum(1 for v in self.validation_history if v['valid'])
        success_rate = valid_validations / total_validations
        
        # Model type distribution
        model_type_counts = {}
        for validation in self.validation_history:
            model_type = validation['model_type']
            model_type_counts[model_type] = model_type_counts.get(model_type, 0) + 1
        
        # Average validation scores
        avg_validation_score = np.mean([v['validation_score'] for v in self.validation_history])
        
        return {
            'total_validations': total_validations,
            'valid_validations': valid_validations,
            'success_rate': success_rate,
            'model_type_distribution': model_type_counts,
            'average_validation_score': avg_validation_score,
            'validation_history': self.validation_history
        }

# Global integrator instance
DEFAULT_VALIDATION_INTEGRATOR = UniversalValidationIntegrator()

def get_validation_integrator(config: Optional[ValidationIntegrationConfig] = None) -> UniversalValidationIntegrator:
    """Get validation integrator instance."""
    if config is None:
        return DEFAULT_VALIDATION_INTEGRATOR
    return UniversalValidationIntegrator(config)

def validate_training_data(X_train: np.ndarray,
                          X_val: np.ndarray,
                          y_train: np.ndarray,
                          y_val: np.ndarray,
                          timestamps: Optional[np.ndarray] = None,
                          feature_names: Optional[List[str]] = None,
                          model_type: str = "unknown",
                          config: Optional[ValidationIntegrationConfig] = None) -> Dict[str, Any]:
    """Convenience function to validate training data."""
    integrator = get_validation_integrator(config)
    return integrator.validate_training_data(
        X_train, X_val, y_train, y_val, timestamps, feature_names, model_type
    )

def validate_trained_model(model: Any,
                          X_train: np.ndarray,
                          X_val: np.ndarray,
                          y_train: np.ndarray,
                          y_val: np.ndarray,
                          timestamps: Optional[np.ndarray] = None,
                          feature_names: Optional[List[str]] = None,
                          model_name: str = "unknown",
                          model_type: str = "unknown",
                          fold_number: Optional[int] = None,
                          config: Optional[ValidationIntegrationConfig] = None) -> Dict[str, Any]:
    """Convenience function to validate trained model."""
    integrator = get_validation_integrator(config)
    return integrator.validate_trained_model(
        model, X_train, X_val, y_train, y_val, timestamps, feature_names,
        model_name, model_type, fold_number
    )

def validate_hpo_trial(model: Any,
                      X_train: np.ndarray,
                      X_val: np.ndarray,
                      y_train: np.ndarray,
                      y_val: np.ndarray,
                      trial_params: Dict[str, Any],
                      model_name: str = "unknown",
                      model_type: str = "unknown",
                      trial_number: int = 0,
                      config: Optional[ValidationIntegrationConfig] = None) -> Dict[str, Any]:
    """Convenience function to validate HPO trial."""
    integrator = get_validation_integrator(config)
    return integrator.validate_hpo_trial(
        model, X_train, X_val, y_train, y_val, trial_params,
        model_name, model_type, trial_number
    )