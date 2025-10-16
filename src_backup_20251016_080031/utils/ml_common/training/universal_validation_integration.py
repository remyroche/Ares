from __future__ import annotations

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

# Import and integrate with existing modules
from ...lookahead_bias_detector import LookaheadBiasDetector
from ..validation.enhanced_overfitting_detection import (
    UniversalOverfittingDetector,
    OverfittingConfig,
    OverfittingReport,
    get_overfitting_detector
)
from ..validation.universal_ml_validation import (
    get_ml_validator,
    UniversalMLValidationConfig
)
from ..optimization.overfitting_prevention import (
    OverfittingPrevention,
    OverfittingPreventionConfig
)

# Import new utility modules that complement existing ones
try:
    from ..validation.data_leakage_prevention import (
        get_data_leakage_prevention,
        DataLeakageConfig,
        detect_temporal_leakage,
        detect_train_test_leakage
    )
    DATA_LEAKAGE_AVAILABLE = True
except ImportError:
    DATA_LEAKAGE_AVAILABLE = False
    get_data_leakage_prevention = None
    DataLeakageConfig = None
    detect_temporal_leakage = None
    detect_train_test_leakage = None

try:
    from ..validation.overfitting_monitoring import (
        get_overfitting_monitor,
        OverfittingMonitoringConfig,
        start_monitoring_session,
        monitor_training_step
    )
    OVERFITTING_MONITORING_AVAILABLE = True
except ImportError:
    OVERFITTING_MONITORING_AVAILABLE = False
    get_overfitting_monitor = None
    OverfittingMonitoringConfig = None
    start_monitoring_session = None
    monitor_training_step = None

try:
    from ..validation.enhanced_validation import (
        get_enhanced_validator,
        EnhancedValidationConfig,
        validate_model_comprehensively
    )
    ENHANCED_VALIDATION_AVAILABLE = True
except ImportError:
    ENHANCED_VALIDATION_AVAILABLE = False
    get_enhanced_validator = None
    EnhancedValidationConfig = None
    validate_model_comprehensively = None

try:
    from ..validation.hpo_overfitting_prevention import (
        get_hpo_with_overfitting_prevention,
        HPOOverfittingPreventionConfig,
        optimize_hyperparameters_safely
    )
    HPO_PREVENTION_AVAILABLE = True
except ImportError:
    HPO_PREVENTION_AVAILABLE = False
    get_hpo_with_overfitting_prevention = None
    HPOOverfittingPreventionConfig = None
    optimize_hyperparameters_safely = None

try:
    from ..validation.model_complexity_analysis import (
        get_model_complexity_analyzer,
        ModelComplexityConfig,
        analyze_model_complexity
    )
    COMPLEXITY_ANALYSIS_AVAILABLE = True
except ImportError:
    COMPLEXITY_ANALYSIS_AVAILABLE = False
    get_model_complexity_analyzer = None
    ModelComplexityConfig = None
    analyze_model_complexity = None

# Import reporting integration
from ..reporting.validation_reporting_integration import (
    get_validation_reporting_integrator,
    process_validation_with_reporting
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

    # New utility settings
    enable_data_leakage_prevention: bool = True
    enable_overfitting_monitoring: bool = True
    enable_enhanced_validation: bool = True
    enable_hpo_overfitting_prevention: bool = True
    enable_model_complexity_analysis: bool = True

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

    # Utility-specific configurations
    data_leakage_config: Optional[DataLeakageConfig] = None
    overfitting_monitoring_config: Optional[OverfittingMonitoringConfig] = None
    enhanced_validation_config: Optional[EnhancedValidationConfig] = None
    hpo_overfitting_prevention_config: Optional[HPOOverfittingPreventionConfig] = None
    model_complexity_config: Optional[ModelComplexityConfig] = None

    # Intelligent utility selection
    auto_select_utilities: bool = True
    utility_selection_criteria: Dict[str, Any] = None

    def __post_init__(self):
        """Initialize default values."""
        if self.model_validation_overrides is None:
            self.model_validation_overrides = {}
        if self.utility_selection_criteria is None:
            self.utility_selection_criteria = {
                'min_samples_for_enhanced_validation': 100,
                'min_features_for_complexity_analysis': 10,
                'enable_hpo_prevention_for_trials': 50,
                'enable_leakage_prevention_for_time_series': True,
                'enable_monitoring_for_long_training': True
            }

        # Initialize default configurations if not provided
        # Only instantiate configs if corresponding classes are available
        try:
            if self.data_leakage_config is None and 'DataLeakageConfig' in globals() and DataLeakageConfig is not None:
                self.data_leakage_config = DataLeakageConfig()
        except Exception:
            pass
        try:
            if self.overfitting_monitoring_config is None and 'OverfittingMonitoringConfig' in globals() and OverfittingMonitoringConfig is not None:
                self.overfitting_monitoring_config = OverfittingMonitoringConfig()
        except Exception:
            pass
        try:
            if self.enhanced_validation_config is None and 'EnhancedValidationConfig' in globals() and EnhancedValidationConfig is not None:
                self.enhanced_validation_config = EnhancedValidationConfig()
        except Exception:
            pass
        try:
            if self.hpo_overfitting_prevention_config is None and 'HPOOverfittingPreventionConfig' in globals() and HPOOverfittingPreventionConfig is not None:
                self.hpo_overfitting_prevention_config = HPOOverfittingPreventionConfig()
        except Exception:
            pass
        try:
            if self.model_complexity_config is None and 'ModelComplexityConfig' in globals() and ModelComplexityConfig is not None:
                self.model_complexity_config = ModelComplexityConfig()
        except Exception:
            pass

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

        # Initialize new utility components (guarded)
        self.data_leakage_prevention = (
            get_data_leakage_prevention(self.config.data_leakage_config)
            if 'get_data_leakage_prevention' in globals() and callable(get_data_leakage_prevention)
            else None
        )
        self.overfitting_monitor = (
            get_overfitting_monitor(self.config.overfitting_monitoring_config)
            if 'get_overfitting_monitor' in globals() and callable(get_overfitting_monitor)
            else None
        )
        self.enhanced_validator = (
            get_enhanced_validator(self.config.enhanced_validation_config)
            if 'get_enhanced_validator' in globals() and callable(get_enhanced_validator)
            else None
        )
        self.hpo_prevention = (
            get_hpo_with_overfitting_prevention(self.config.hpo_overfitting_prevention_config)
            if 'get_hpo_with_overfitting_prevention' in globals() and callable(get_hpo_with_overfitting_prevention)
            else None
        )
        self.complexity_analyzer = (
            get_model_complexity_analyzer(self.config.model_complexity_config)
            if 'get_model_complexity_analyzer' in globals() and callable(get_model_complexity_analyzer)
            else None
        )

        # Initialize reporting integration
        self.reporting_integrator = get_validation_reporting_integrator()

        # Create validation report directory
        if self.config.save_validation_reports:
            Path(self.config.validation_report_directory).mkdir(parents=True, exist_ok=True)

        # Track validation history
        self.validation_history = []
        self.monitoring_sessions = {}

        logger.info("✅ Universal Validation Integrator initialized with new utilities")
    
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
            
            # Process with reporting system
            if self.config.save_validation_reports:
                self.reporting_integrator.process_validation_report(
                    validation_report=validation_report,
                    model_name=model_name,
                    model_type=model_type,
                    fold_number=fold_number,
                    validation_duration=None,  # Could be calculated if needed
                    model_metadata={'config': self.config.__dict__}
                )
            
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
    
    def intelligently_select_utilities(self,
                                     X: np.ndarray,
                                     y: np.ndarray,
                                     model_type: str,
                                     task_type: str = "training",
                                     n_samples: Optional[int] = None,
                                     n_features: Optional[int] = None) -> Dict[str, Any]:
        """
        Intelligently select which validation utilities to use based on data characteristics.

        Args:
            X: Feature matrix
            y: Target vector
            model_type: Type of model
            task_type: Type of task (training, hpo, evaluation)
            n_samples: Number of samples (if None, inferred from X)
            n_features: Number of features (if None, inferred from X)

        Returns:
            Dict with utility selection decisions and rationale
        """
        if not self.config.auto_select_utilities:
            return {
                'utilities_selected': {
                    'data_leakage_prevention': self.config.enable_data_leakage_prevention,
                    'overfitting_monitoring': self.config.enable_overfitting_monitoring,
                    'enhanced_validation': self.config.enable_enhanced_validation,
                    'hpo_overfitting_prevention': self.config.enable_hpo_overfitting_prevention,
                    'model_complexity_analysis': self.config.enable_model_complexity_analysis
                },
                'selection_rationale': 'Manual configuration used'
            }

        # Infer data characteristics
        if n_samples is None and X is not None:
            n_samples = len(X)
        if n_features is None and X is not None:
            n_features = X.shape[1]

        # Default selections
        selections = {
            'data_leakage_prevention': False,
            'overfitting_monitoring': False,
            'enhanced_validation': False,
            'hpo_overfitting_prevention': False,
            'model_complexity_analysis': False
        }

        rationale_parts = []

        # Data leakage prevention selection
        if (self.config.enable_data_leakage_prevention and
            n_samples is not None and n_samples > 50):
            # Enable for time series or large datasets
            if (self.config.utility_selection_criteria['enable_leakage_prevention_for_time_series'] or
                n_samples > 500):
                selections['data_leakage_prevention'] = True
                rationale_parts.append("Data leakage prevention enabled (time series/large dataset)")

        # Overfitting monitoring selection
        if (self.config.enable_overfitting_monitoring and
            task_type == "training" and n_samples is not None):
            # Enable for longer training sessions
            if (self.config.utility_selection_criteria['enable_monitoring_for_long_training'] and
                n_samples > 200):
                selections['overfitting_monitoring'] = True
                rationale_parts.append("Overfitting monitoring enabled (long training session)")

        # Enhanced validation selection
        if (self.config.enable_enhanced_validation and
            n_samples is not None and
            n_samples >= self.config.utility_selection_criteria['min_samples_for_enhanced_validation']):
            selections['enhanced_validation'] = True
            rationale_parts.append("Enhanced validation enabled (sufficient data)")

        # Model complexity analysis selection
        if (self.config.enable_model_complexity_analysis and
            n_features is not None and
            n_features >= self.config.utility_selection_criteria['min_features_for_complexity_analysis']):
            selections['model_complexity_analysis'] = True
            rationale_parts.append("Model complexity analysis enabled (sufficient features)")

        # HPO overfitting prevention selection
        if (self.config.enable_hpo_overfitting_prevention and
            task_type == "hpo" and n_samples is not None):
            # Enable for HPO with many trials
            if n_samples > 100:  # Large enough for meaningful HPO
                selections['hpo_overfitting_prevention'] = True
                rationale_parts.append("HPO overfitting prevention enabled (HPO task)")

        # Model-specific adjustments
        selections = self._adjust_for_model_type(selections, model_type, rationale_parts)

        # Data-specific adjustments
        selections = self._adjust_for_data_characteristics(selections, X, y, rationale_parts)

        return {
            'utilities_selected': selections,
            'selection_rationale': '; '.join(rationale_parts),
            'data_characteristics': {
                'n_samples': n_samples,
                'n_features': n_features,
                'model_type': model_type,
                'task_type': task_type
            }
        }

    def _adjust_for_model_type(self, selections: Dict[str, bool], model_type: str, rationale_parts: List[str]) -> Dict[str, bool]:
        """Adjust utility selections based on model type."""
        model_type_lower = model_type.lower()

        # Tree-based models benefit from complexity analysis
        if model_type_lower in ['xgboost', 'lightgbm', 'catboost', 'random_forest', 'extra_trees']:
            if not selections['model_complexity_analysis']:
                selections['model_complexity_analysis'] = True
                rationale_parts.append("Complexity analysis enabled (tree-based model)")

        # Neural networks benefit from overfitting monitoring
        elif model_type_lower in ['neural_network', 'deep_learning', 'cnn', 'rnn']:
            if not selections['overfitting_monitoring']:
                selections['overfitting_monitoring'] = True
                rationale_parts.append("Overfitting monitoring enabled (neural network)")

        return selections

    def _adjust_for_data_characteristics(self, selections: Dict[str, bool], X: np.ndarray, y: np.ndarray, rationale_parts: List[str]) -> Dict[str, bool]:
        """Adjust utility selections based on data characteristics."""
        try:
            # Check for temporal patterns that suggest time series
            if X is not None and len(X) > 10:
                # Simple heuristic: check if data looks like it has temporal structure
                # This could be enhanced with more sophisticated time series detection
                temporal_likelihood = self._assess_temporal_likelihood(X, y)

                if temporal_likelihood > 0.7:
                    if not selections['data_leakage_prevention']:
                        selections['data_leakage_prevention'] = True
                        rationale_parts.append("Data leakage prevention enabled (temporal patterns detected)")

        except Exception as e:
            logger.warning(f"Data characteristics adjustment failed: {e}")

        return selections

    def _assess_temporal_likelihood(self, X: np.ndarray, y: np.ndarray) -> float:
        """Assess likelihood that data has temporal structure."""
        try:
            # Simple heuristics for temporal data detection
            likelihood = 0.0

            # Check for autocorrelation in target
            if len(y) > 20:
                from scipy.stats import pearsonr
                # Check autocorrelation at lag 1
                if len(y) > 1:
                    corr, _ = pearsonr(y[:-1], y[1:])
                    if abs(corr) > 0.5:
                        likelihood += 0.4

            # Check for feature autocorrelation
            if X.shape[1] > 0:
                feature_corr_sum = 0
                n_features_checked = min(5, X.shape[1])

                for i in range(n_features_checked):
                    if len(X) > 1:
                        corr, _ = pearsonr(X[:-1, i], X[1:, i])
                        feature_corr_sum += abs(corr)

                avg_feature_corr = feature_corr_sum / n_features_checked
                if avg_feature_corr > 0.3:
                    likelihood += 0.3

            # Check for time-like features (e.g., increasing indices)
            if X.shape[1] > 0:
                # Check if first feature looks like time index
                first_feature = X[:, 0]
                if len(first_feature) > 1:
                    # Check if it's monotonically increasing
                    if np.all(np.diff(first_feature) >= 0):
                        likelihood += 0.3

            return min(1.0, likelihood)

        except Exception as e:
            logger.error(f"Temporal likelihood assessment failed: {e}")
            return 0.5

    def start_monitoring_session(self, model_name: str, model_type: str = "unknown") -> str:
        """Start a monitoring session for overfitting monitoring."""
        session_id = start_monitoring_session(model_name, model_type)
        self.monitoring_sessions[session_id] = {
            'model_name': model_name,
            'model_type': model_type,
            'start_time': datetime.now()
        }
        return session_id

    def monitor_training_step(self,
                             session_id: str,
                             epoch: int,
                             train_accuracy: float,
                             val_accuracy: float,
                             train_loss: float,
                             val_loss: float,
                             additional_metrics: Optional[Dict[str, float]] = None):
        """Monitor a training step using overfitting monitoring."""
        return monitor_training_step(
            session_id, epoch, train_accuracy, val_accuracy, train_loss, val_loss, additional_metrics
        )

    def end_monitoring_session(self, session_id: str):
        """End a monitoring session."""
        if session_id in self.monitoring_sessions:
            del self.monitoring_sessions[session_id]
        self.overfitting_monitor.end_monitoring_session(session_id)

    def perform_data_leakage_check(self,
                                  data: pd.DataFrame,
                                  timestamp_column: str,
                                  target_column: Optional[str] = None,
                                  dataset_name: str = "dataset") -> Dict[str, Any]:
        """Perform data leakage check using the prevention utility."""
        leakage_report = detect_temporal_leakage(data, timestamp_column, target_column, dataset_name)
        return {
            'leakage_detected': leakage_report.temporal_leakage_detected or leakage_report.lookahead_bias_detected,
            'leakage_rate': leakage_report.overall_leakage_rate,
            'severity': leakage_report.severity_level,
            'recommendations': leakage_report.recommendations,
            'report': leakage_report
        }

    def perform_enhanced_validation(self,
                                   model: Any,
                                   X: np.ndarray,
                                   y: np.ndarray,
                                   model_name: str = "unknown",
                                   model_type: str = "unknown",
                                   is_classification: bool = True) -> Dict[str, Any]:
        """Perform enhanced validation using the enhanced validator."""
        validation_report = validate_model_comprehensively(
            model, X, y, model_name, model_type, is_classification=is_classification
        )
        return {
            'validation_score': validation_report.validation_quality_score,
            'performance_stability': validation_report.performance_stability,
            'validation_reliability': validation_report.validation_reliability,
            'recommendations': validation_report.recommendations,
            'report': validation_report
        }

    def perform_complexity_analysis(self,
                                   model: Any,
                                   X: np.ndarray,
                                   y: np.ndarray,
                                   model_name: str = "unknown",
                                   model_type: str = "unknown") -> Dict[str, Any]:
        """Perform model complexity analysis."""
        complexity_report = analyze_model_complexity(
            model, X, y, model_name, model_type
        )
        return {
            'complexity_score': complexity_report.overall_complexity_score,
            'complexity_level': complexity_report.complexity_level,
            'overfitting_risk': complexity_report.overfitting_risk_score,
            'simplification_potential': complexity_report.simplification_potential,
            'recommendations': complexity_report.primary_recommendations,
            'report': complexity_report
        }

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

        # Utility usage statistics
        utility_usage = {
            'data_leakage_prevention': sum(1 for v in self.validation_history if 'leakage_check' in v),
            'overfitting_monitoring': len(self.monitoring_sessions),
            'enhanced_validation': sum(1 for v in self.validation_history if 'enhanced_validation' in v),
            'complexity_analysis': sum(1 for v in self.validation_history if 'complexity_analysis' in v)
        }

        return {
            'total_validations': total_validations,
            'valid_validations': valid_validations,
            'success_rate': success_rate,
            'model_type_distribution': model_type_counts,
            'average_validation_score': avg_validation_score,
            'validation_history': self.validation_history,
            'utility_usage': utility_usage,
            'monitoring_sessions_active': len(self.monitoring_sessions)
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

def intelligently_select_utilities(X: np.ndarray,
                                  y: np.ndarray,
                                  model_type: str,
                                  task_type: str = "training",
                                  n_samples: Optional[int] = None,
                                  n_features: Optional[int] = None,
                                  config: Optional[ValidationIntegrationConfig] = None) -> Dict[str, Any]:
    """Convenience function to intelligently select validation utilities."""
    integrator = get_validation_integrator(config)
    return integrator.intelligently_select_utilities(
        X, y, model_type, task_type, n_samples, n_features
    )

def start_monitoring_session(model_name: str, model_type: str = "unknown", config: Optional[ValidationIntegrationConfig] = None) -> str:
    """Convenience function to start monitoring session."""
    integrator = get_validation_integrator(config)
    return integrator.start_monitoring_session(model_name, model_type)

def monitor_training_step(session_id: str,
                         epoch: int,
                         train_accuracy: float,
                         val_accuracy: float,
                         train_loss: float,
                         val_loss: float,
                         additional_metrics: Optional[Dict[str, float]] = None,
                         config: Optional[ValidationIntegrationConfig] = None):
    """Convenience function to monitor training step."""
    integrator = get_validation_integrator(config)
    return integrator.monitor_training_step(
        session_id, epoch, train_accuracy, val_accuracy, train_loss, val_loss, additional_metrics
    )

def perform_data_leakage_check(data: pd.DataFrame,
                              timestamp_column: str,
                              target_column: Optional[str] = None,
                              dataset_name: str = "dataset",
                              config: Optional[ValidationIntegrationConfig] = None) -> Dict[str, Any]:
    """Convenience function to perform data leakage check."""
    integrator = get_validation_integrator(config)
    return integrator.perform_data_leakage_check(data, timestamp_column, target_column, dataset_name)

def perform_enhanced_validation(model: Any,
                               X: np.ndarray,
                               y: np.ndarray,
                               model_name: str = "unknown",
                               model_type: str = "unknown",
                               is_classification: bool = True,
                               config: Optional[ValidationIntegrationConfig] = None) -> Dict[str, Any]:
    """Convenience function to perform enhanced validation."""
    integrator = get_validation_integrator(config)
    return integrator.perform_enhanced_validation(
        model, X, y, model_name, model_type, is_classification
    )

def perform_complexity_analysis(model: Any,
                               X: np.ndarray,
                               y: np.ndarray,
                               model_name: str = "unknown",
                               model_type: str = "unknown",
                               config: Optional[ValidationIntegrationConfig] = None) -> Dict[str, Any]:
    """Convenience function to perform model complexity analysis."""
    integrator = get_validation_integrator(config)
    return integrator.perform_complexity_analysis(model, X, y, model_name, model_type)