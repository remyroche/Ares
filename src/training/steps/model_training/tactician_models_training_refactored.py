"""
Tactician Models Training Step - Enhanced & Streamlined

This step handles per-regime training of individual Tactician models using common dependencies.
The Tactician operates on 1m timeframe and decides WHEN to trade based on Analyst's green light signals.

ENHANCED FEATURES:
- Comprehensive error handling with detailed failure reporting
- Enhanced progress tracking and sub-step reporting
- Input validation and data quality checks
- Optimized vectorization with intelligent fallback
- Structured logging with performance metrics
- Health monitoring throughout training process
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import time
import traceback
from dataclasses import dataclass
from enum import Enum

from src.utils.logger import system_logger
from src.utils.ml_common.config import PerRegimeTrainingConfig, TacticianTrainingConfig
from src.utils.ml_common.training import PerRegimeTrainingStep

# Import vectorized training manager for enhanced capabilities
try:
    from src.utils.ml_common.training.vectorized_training_manager import VectorizedTrainingManager
    VECTORIZED_TRAINING_AVAILABLE = True
except ImportError:
    VECTORIZED_TRAINING_AVAILABLE = False

logger = system_logger.getChild('TacticianModelsTrainingEnhanced')


class TrainingPhase(Enum):
    """Training phases for progress tracking."""
    INITIALIZATION = "initialization"
    DATA_VALIDATION = "data_validation"
    FEATURE_PREPARATION = "feature_preparation"
    REGIME_ANALYSIS = "regime_analysis"
    MODEL_TRAINING = "model_training"
    EVALUATION = "evaluation"
    MODEL_SAVING = "model_saving"
    FINALIZATION = "finalization"


@dataclass
class TrainingMetrics:
    """Training metrics for comprehensive reporting."""
    phase: TrainingPhase
    start_time: float
    end_time: Optional[float] = None
    samples_processed: int = 0
    features_count: int = 0
    regimes_count: int = 0
    models_trained: int = 0
    errors_encountered: int = 0
    warnings_issued: int = 0
    memory_usage_mb: float = 0.0
    success: bool = False
    error_message: Optional[str] = None
    
    @property
    def duration(self) -> float:
        """Get phase duration in seconds."""
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time


class TacticianModelsTrainingStepRefactored(PerRegimeTrainingStep):
    """
    Enhanced Tactician Models Training Step with comprehensive error handling and reporting.
    
    The Tactician operates on 1m timeframe and is trained on:
    1. Only periods where the Analyst gives a green light
    2. Using the Analyst's model outputs as input features
    
    ENHANCED FEATURES:
    - Comprehensive input validation and data quality checks
    - Detailed progress tracking with phase-based metrics
    - Enhanced error handling with specific failure reporting
    - Optimized vectorization with intelligent fallback
    - Structured logging with performance monitoring
    """
    
    def __init__(self, config: Optional[TacticianTrainingConfig] = None, enable_vectorization: bool = True):
        """
        Initialize enhanced Tactician models training step.

        Args:
            config: Per-regime training configuration
            enable_vectorization: Whether to enable vectorized training
        """
        # Initialize training metrics tracking
        self.training_metrics: Dict[TrainingPhase, TrainingMetrics] = {}
        self.overall_start_time = time.time()
        self.phase_start_time = time.time()
        
        # Set default configuration for tactician models with enhanced settings
        if config is None:
            config = TacticianTrainingConfig(
                model_name="tactician_models",
                timeframe="1m",
                model_types=["NeuralObliviousDecisionEnsembles", "CatBoostRegressor", "LGBMRegressor", "ElasticNetCV"],
                hpo_n_trials=100,
                hpo_timeout_seconds=3600,
                min_samples_per_regime=1000,
                enable_data_augmentation=True,
                augmentation_method="smote",
                model_save_path="./models/tactician_models",
                evaluation_metrics=["mse", "mae", "r2", "mape", "smape"],
                use_single_model=True,
                single_model_name="tactician_unified_model",
                enable_ensemble_training=True,
                ensemble_method="stacking",
                meta_model="ElasticNetCV",
                ensemble_name="tactician_ensemble",
                enable_entry_timing_optimization=True,
                entry_timing_range=0.005,
                expected_movement=0.01
            )

        try:
            super().__init__(config)
            self.logger = logger.getChild('TacticianModelsTrainingEnhanced')
            
            # Vectorization support with enhanced validation
            self.enable_vectorization = enable_vectorization and VECTORIZED_TRAINING_AVAILABLE
            self.vectorization_fallback_used = False
            
            # Initialize training metrics for initialization phase
            self._start_phase(TrainingPhase.INITIALIZATION)
            
            # Validate configuration
            self._validate_configuration(config)
            
            # Log initialization success
            if self.enable_vectorization:
                self.logger.info("🚀 Enhanced Tactician Models Training Step initialized with vectorization")
            else:
                self.logger.info("✅ Enhanced Tactician Models Training Step initialized (standard mode)")
            
            self._complete_phase(TrainingPhase.INITIALIZATION, success=True)
            
        except Exception as e:
            self._handle_initialization_error(e)
            raise
    
    def _start_phase(self, phase: TrainingPhase, context: Optional[Dict[str, Any]] = None) -> None:
        """Start tracking a training phase with structured logging."""
        self.training_metrics[phase] = TrainingMetrics(
            phase=phase,
            start_time=time.time()
        )
        
        # Log phase start with structured format
        self._log_phase_start(phase, context)
    
    def _complete_phase(self, phase: TrainingPhase, success: bool = True, 
                       error_message: Optional[str] = None, **kwargs) -> None:
        """Complete a training phase with metrics and structured logging."""
        if phase in self.training_metrics:
            metrics = self.training_metrics[phase]
            metrics.end_time = time.time()
            metrics.success = success
            metrics.error_message = error_message
            
            # Update metrics with provided values
            for key, value in kwargs.items():
                if hasattr(metrics, key):
                    setattr(metrics, key, value)
            
            duration = metrics.duration
            
            # Prepare metrics for structured logging
            phase_metrics = {
                'duration': duration,
                'samples_processed': metrics.samples_processed,
                'features_count': metrics.features_count,
                'models_trained': metrics.models_trained,
                'warnings_issued': metrics.warnings_issued,
                'errors_encountered': metrics.errors_encountered,
                'memory_usage_mb': metrics.memory_usage_mb
            }
            
            # Log phase completion with structured format
            self._log_phase_complete(phase, success, duration, phase_metrics)
            
            if not success and error_message:
                self._log_structured_event(
                    event_type="phase_error",
                    phase=phase.value,
                    message=f"Phase failed: {error_message}",
                    level="error"
                )
    
    def _validate_configuration(self, config: PerRegimeTrainingConfig) -> None:
        """Validate training configuration."""
        try:
            # Validate model types
            if not config.model_types:
                raise ValueError("No model types specified in configuration")
            
            # Validate timeframe
            if not config.timeframe:
                raise ValueError("No timeframe specified in configuration")
            
            # Validate minimum samples
            if config.min_samples_per_regime < 100:
                self.logger.warning(f"⚠️ Very low minimum samples per regime: {config.min_samples_per_regime}")
            
            # Validate HPO settings
            if config.enable_hpo and config.hpo_n_trials < 10:
                self.logger.warning(f"⚠️ Very low HPO trials: {config.hpo_n_trials}")
            
            self.logger.info("✅ Configuration validation passed")
            
        except Exception as e:
            self.logger.error(f"❌ Configuration validation failed: {e}")
            raise
    
    def _handle_initialization_error(self, error: Exception) -> None:
        """Handle initialization errors with detailed reporting."""
        error_msg = f"Initialization failed: {str(error)}"
        self.logger.error(f"❌ {error_msg}")
        self.logger.error(f"❌ Traceback: {traceback.format_exc()}")
        
        if TrainingPhase.INITIALIZATION in self.training_metrics:
            self._complete_phase(TrainingPhase.INITIALIZATION, success=False, error_message=error_msg)
    
    def _validate_input_data(self, X: np.ndarray, y: np.ndarray, 
                           regime_labels: np.ndarray) -> Dict[str, Any]:
        """Comprehensive input data validation with detailed reporting."""
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'data_quality_metrics': {},
            'regime_analysis': {}
        }
        
        try:
            # Check data shapes
            if X.shape[0] != y.shape[0]:
                error_msg = f"Feature and target sample counts don't match: {X.shape[0]} vs {y.shape[0]}"
                validation_results['errors'].append(error_msg)
                validation_results['is_valid'] = False
            
            if X.shape[0] != regime_labels.shape[0]:
                error_msg = f"Feature and regime label sample counts don't match: {X.shape[0]} vs {regime_labels.shape[0]}"
                validation_results['errors'].append(error_msg)
                validation_results['is_valid'] = False
            
            # Check for empty data
            if X.shape[0] == 0:
                error_msg = "No samples provided in input data"
                validation_results['errors'].append(error_msg)
                validation_results['is_valid'] = False
            
            if X.shape[1] == 0:
                error_msg = "No features provided in input data"
                validation_results['errors'].append(error_msg)
                validation_results['is_valid'] = False
            
            # Data quality analysis using utility functions
            data_quality = {}
            
            # Validate features quality
            feature_quality = self._validate_data_quality(X, "features", max_nan_percentage=10.0, max_inf_percentage=1.0)
            data_quality['features'] = feature_quality
            validation_results['warnings'].extend(feature_quality['warnings'])
            validation_results['errors'].extend(feature_quality['errors'])
            if not feature_quality['is_valid']:
                validation_results['is_valid'] = False
            
            # Validate targets quality (stricter thresholds)
            target_quality = self._validate_data_quality(y, "targets", max_nan_percentage=5.0, max_inf_percentage=1.0)
            data_quality['targets'] = target_quality
            validation_results['warnings'].extend(target_quality['warnings'])
            validation_results['errors'].extend(target_quality['errors'])
            if not target_quality['is_valid']:
                validation_results['is_valid'] = False
            
            # Regime distribution analysis
            unique_regimes = np.unique(regime_labels)
            regime_counts = np.bincount(regime_labels)
            min_regime_size = np.min(regime_counts)
            max_regime_size = np.max(regime_counts)
            regime_balance = min_regime_size / max_regime_size if max_regime_size > 0 else 0
            
            regime_analysis = {
                'unique_regimes_count': len(unique_regimes),
                'min_regime_size': min_regime_size,
                'max_regime_size': max_regime_size,
                'regime_balance': regime_balance,
                'regime_distribution': dict(zip(unique_regimes, regime_counts))
            }
            
            validation_results['regime_analysis'] = regime_analysis
            validation_results['data_quality_metrics'] = data_quality
            
            # Check regime sufficiency
            insufficient_regimes = regime_counts < self.config.min_samples_per_regime
            insufficient_count = np.sum(insufficient_regimes)
            
            if insufficient_count > 0:
                warning_msg = f"{insufficient_count} regimes have fewer than {self.config.min_samples_per_regime} samples"
                validation_results['warnings'].append(warning_msg)
                
                # Check if too many regimes are insufficient
                if insufficient_count > len(unique_regimes) * 0.5:
                    error_msg = f"Critical: {insufficient_count}/{len(unique_regimes)} regimes have insufficient data"
                    validation_results['errors'].append(error_msg)
                    validation_results['is_valid'] = False
            
            # Log comprehensive validation results
            self.logger.info(f"📊 Data validation: {X.shape[0]} samples, {X.shape[1]} features, {len(unique_regimes)} regimes")
            self.logger.info(f"📊 Regime balance: {regime_balance:.3f} (min={min_regime_size}, max={max_regime_size})")
            
            if validation_results['warnings']:
                for warning in validation_results['warnings']:
                    self.logger.warning(f"⚠️ {warning}")
            
            if validation_results['errors']:
                for error in validation_results['errors']:
                    self.logger.error(f"❌ {error}")
                raise ValueError(f"Data validation failed: {'; '.join(validation_results['errors'])}")
            
            return validation_results
            
        except Exception as e:
            validation_results['is_valid'] = False
            validation_results['errors'].append(str(e))
            self.logger.error(f"❌ Data validation failed: {e}")
            raise
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
    
    def _validate_data_quality(self, data: np.ndarray, data_name: str, 
                              max_nan_percentage: float = 10.0, 
                              max_inf_percentage: float = 1.0) -> Dict[str, Any]:
        """Validate data quality with configurable thresholds."""
        quality_metrics = {
            'nan_count': 0,
            'inf_count': 0,
            'nan_percentage': 0.0,
            'inf_percentage': 0.0,
            'is_valid': True,
            'warnings': [],
            'errors': []
        }
        
        try:
            # Check for NaN values
            nan_count = np.sum(np.isnan(data))
            if nan_count > 0:
                nan_percentage = (nan_count / data.size) * 100
                quality_metrics['nan_count'] = nan_count
                quality_metrics['nan_percentage'] = nan_percentage
                
                if nan_percentage > max_nan_percentage:
                    error_msg = f"{data_name} contains {nan_percentage:.2f}% NaN values (threshold: {max_nan_percentage}%)"
                    quality_metrics['errors'].append(error_msg)
                    quality_metrics['is_valid'] = False
                else:
                    warning_msg = f"{data_name} contains {nan_count} NaN values ({nan_percentage:.2f}%)"
                    quality_metrics['warnings'].append(warning_msg)
            
            # Check for infinite values
            inf_count = np.sum(np.isinf(data))
            if inf_count > 0:
                inf_percentage = (inf_count / data.size) * 100
                quality_metrics['inf_count'] = inf_count
                quality_metrics['inf_percentage'] = inf_percentage
                
                if inf_percentage > max_inf_percentage:
                    error_msg = f"{data_name} contains {inf_percentage:.2f}% infinite values (threshold: {max_inf_percentage}%)"
                    quality_metrics['errors'].append(error_msg)
                    quality_metrics['is_valid'] = False
                else:
                    warning_msg = f"{data_name} contains {inf_count} infinite values ({inf_percentage:.2f}%)"
                    quality_metrics['warnings'].append(warning_msg)
            
            return quality_metrics
            
        except Exception as e:
            quality_metrics['is_valid'] = False
            quality_metrics['errors'].append(f"Failed to validate {data_name}: {e}")
            return quality_metrics
    
    def _validate_array_shapes(self, arrays: Dict[str, np.ndarray], expected_samples: int) -> Dict[str, Any]:
        """Validate that all arrays have consistent sample counts."""
        validation_results = {
            'is_valid': True,
            'errors': [],
            'shape_info': {}
        }
        
        try:
            for name, array in arrays.items():
                if array.shape[0] != expected_samples:
                    error_msg = f"{name} shape mismatch: expected {expected_samples}, got {array.shape[0]}"
                    validation_results['errors'].append(error_msg)
                    validation_results['is_valid'] = False
                else:
                    validation_results['shape_info'][name] = array.shape
            
            if not validation_results['is_valid']:
                raise ValueError(f"Shape validation failed: {'; '.join(validation_results['errors'])}")
            
            return validation_results
            
        except Exception as e:
            validation_results['is_valid'] = False
            validation_results['errors'].append(str(e))
            return validation_results
    
    def _comprehensive_validation_check(self, 
                                      X: np.ndarray, 
                                      y: np.ndarray, 
                                      regime_labels: np.ndarray,
                                      phase_name: str,
                                      additional_arrays: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, Any]:
        """Comprehensive validation check for any phase of training."""
        validation_summary = {
            'phase': phase_name,
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'metrics': {},
            'recommendations': []
        }
        
        try:
            # Basic shape validation
            arrays_to_validate = {
                'features': X,
                'targets': y,
                'regime_labels': regime_labels
            }
            
            if additional_arrays:
                arrays_to_validate.update(additional_arrays)
            
            shape_validation = self._validate_array_shapes(arrays_to_validate, X.shape[0])
            if not shape_validation['is_valid']:
                validation_summary['errors'].extend(shape_validation['errors'])
                validation_summary['is_valid'] = False
            
            # Data quality validation
            feature_quality = self._validate_data_quality(X, f"{phase_name}_features", max_nan_percentage=10.0, max_inf_percentage=1.0)
            target_quality = self._validate_data_quality(y, f"{phase_name}_targets", max_nan_percentage=5.0, max_inf_percentage=1.0)
            
            validation_summary['warnings'].extend(feature_quality['warnings'])
            validation_summary['warnings'].extend(target_quality['warnings'])
            validation_summary['errors'].extend(feature_quality['errors'])
            validation_summary['errors'].extend(target_quality['errors'])
            
            if not feature_quality['is_valid'] or not target_quality['is_valid']:
                validation_summary['is_valid'] = False
            
            # Regime analysis
            unique_regimes = np.unique(regime_labels)
            regime_counts = np.bincount(regime_labels)
            min_regime_size = np.min(regime_counts)
            max_regime_size = np.max(regime_counts)
            regime_balance = min_regime_size / max_regime_size if max_regime_size > 0 else 0
            
            validation_summary['metrics'] = {
                'samples': X.shape[0],
                'features': X.shape[1],
                'regimes': len(unique_regimes),
                'min_regime_size': min_regime_size,
                'max_regime_size': max_regime_size,
                'regime_balance': regime_balance,
                'feature_nan_percentage': feature_quality['nan_percentage'],
                'target_nan_percentage': target_quality['nan_percentage']
            }
            
            # Generate recommendations
            if regime_balance < 0.1:
                validation_summary['recommendations'].append("Very low regime balance - consider data augmentation")
            
            if feature_quality['nan_percentage'] > 5:
                validation_summary['recommendations'].append("High NaN percentage in features - consider imputation")
            
            if target_quality['nan_percentage'] > 2:
                validation_summary['recommendations'].append("High NaN percentage in targets - review data pipeline")
            
            if min_regime_size < self.config.min_samples_per_regime:
                validation_summary['recommendations'].append("Some regimes have insufficient samples - consider reducing min_samples_per_regime")
            
            # Log validation results
            if validation_summary['is_valid']:
                self.logger.info(f"✅ {phase_name} validation passed")
            else:
                self.logger.error(f"❌ {phase_name} validation failed")
                for error in validation_summary['errors']:
                    self.logger.error(f"❌ {error}")
            
            if validation_summary['warnings']:
                for warning in validation_summary['warnings']:
                    self.logger.warning(f"⚠️ {warning}")
            
            if validation_summary['recommendations']:
                for recommendation in validation_summary['recommendations']:
                    self.logger.info(f"💡 {recommendation}")
            
            return validation_summary
            
        except Exception as e:
            validation_summary['is_valid'] = False
            validation_summary['errors'].append(f"Validation check failed: {e}")
            self.logger.error(f"❌ {phase_name} validation check failed: {e}")
            return validation_summary
    
    def _log_structured_event(self, event_type: str, phase: str, message: str, 
                             metrics: Optional[Dict[str, Any]] = None, 
                             level: str = "info") -> None:
        """Log structured events with consistent formatting."""
        try:
            log_data = {
                'event_type': event_type,
                'phase': phase,
                'message': message,
                'timestamp': time.time(),
                'memory_mb': self._get_memory_usage()
            }
            
            if metrics:
                log_data['metrics'] = metrics
            
            # Format structured log message
            structured_msg = f"[{event_type.upper()}] {phase}: {message}"
            if metrics:
                structured_msg += f" | Metrics: {metrics}"
            
            # Log with appropriate level
            if level == "error":
                self.logger.error(structured_msg)
            elif level == "warning":
                self.logger.warning(structured_msg)
            elif level == "debug":
                self.logger.debug(structured_msg)
            else:
                self.logger.info(structured_msg)
                
        except Exception as e:
            self.logger.error(f"Failed to log structured event: {e}")
    
    def _log_phase_start(self, phase: TrainingPhase, context: Optional[Dict[str, Any]] = None) -> None:
        """Log phase start with context."""
        self._log_structured_event(
            event_type="phase_start",
            phase=phase.value,
            message=f"Starting {phase.value} phase",
            metrics=context,
            level="info"
        )
    
    def _log_phase_complete(self, phase: TrainingPhase, success: bool, 
                           duration: float, metrics: Optional[Dict[str, Any]] = None) -> None:
        """Log phase completion with results."""
        event_type = "phase_success" if success else "phase_failure"
        message = f"Completed {phase.value} phase in {duration:.2f}s"
        
        if not success:
            message += " (FAILED)"
        
        self._log_structured_event(
            event_type=event_type,
            phase=phase.value,
            message=message,
            metrics=metrics,
            level="error" if not success else "info"
        )
    
    def _log_data_quality_issue(self, issue_type: str, details: Dict[str, Any]) -> None:
        """Log data quality issues with structured format."""
        self._log_structured_event(
            event_type="data_quality_issue",
            phase="validation",
            message=f"Data quality issue: {issue_type}",
            metrics=details,
            level="warning"
        )
    
    def _log_performance_metric(self, metric_name: str, value: float, unit: str = "") -> None:
        """Log performance metrics with structured format."""
        self._log_structured_event(
            event_type="performance_metric",
            phase="training",
            message=f"Performance metric: {metric_name}",
            metrics={metric_name: f"{value:.2f}{unit}"},
            level="info"
        )
    
    def execute(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regime_labels: np.ndarray,
        feature_names: Optional[List[str]] = None,
        hmm_states: Optional[np.ndarray] = None,
        analyst_signals: Optional[np.ndarray] = None,
        analyst_model_outputs: Optional[np.ndarray] = None,
        hmm_regime_features: Optional[np.ndarray] = None,
        all_analyst_models_outputs: Optional[Dict[str, np.ndarray]] = None,
        hmm_model_outputs: Optional[np.ndarray] = None,
        analyst_ensemble_outputs: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Execute enhanced Tactician models training step with comprehensive error handling.
        
        Args:
            X: Input features (1m timeframe with cross-timeframe features)
            y: Target values (tactician outputs - timing decisions)
            regime_labels: Regime labels for each sample
            feature_names: Names of input features
            hmm_states: HMM cluster/regime states
            analyst_signals: Binary signals from Analyst (green light indicators)
            analyst_model_outputs: Analyst model predictions used as features
            hmm_regime_features: HMM regime features (probabilities, characteristics)
            all_analyst_models_outputs: All individual analyst ML model outputs
            hmm_model_outputs: HMM model outputs (predictions, probabilities, etc.)
            analyst_ensemble_outputs: Analyst ensemble model outputs
            
        Returns:
            Dictionary containing training results and metadata with comprehensive reporting
        """
        try:
            self.logger.info("🚀 Starting Enhanced Tactician models training step")
            self.overall_start_time = time.time()
            
            # Phase 1: Data Validation
            validation_context = {
                'samples': X.shape[0],
                'features': X.shape[1],
                'regimes': len(np.unique(regime_labels))
            }
            self._start_phase(TrainingPhase.DATA_VALIDATION, validation_context)
            
            try:
                validation_results = self._validate_input_data(X, y, regime_labels)
                
                # Log data quality issues if any
                if validation_results.get('warnings'):
                    for warning in validation_results['warnings']:
                        self._log_data_quality_issue("warning", {'message': warning})
                
                if validation_results.get('errors'):
                    for error in validation_results['errors']:
                        self._log_data_quality_issue("error", {'message': error})
                
                self._complete_phase(TrainingPhase.DATA_VALIDATION, success=True, 
                                   samples_processed=X.shape[0], features_count=X.shape[1],
                                   warnings_issued=len(validation_results.get('warnings', [])),
                                   errors_encountered=len(validation_results.get('errors', [])))
            except Exception as e:
                self._complete_phase(TrainingPhase.DATA_VALIDATION, success=False, error_message=str(e))
                raise
            
            # Phase 2: Feature Preparation
            feature_context = {
                'original_samples': X.shape[0],
                'original_features': X.shape[1],
                'has_analyst_signals': analyst_signals is not None,
                'has_hmm_features': hmm_regime_features is not None,
                'has_analyst_models': all_analyst_models_outputs is not None,
                'has_analyst_ensemble': analyst_ensemble_outputs is not None
            }
            self._start_phase(TrainingPhase.FEATURE_PREPARATION, feature_context)
            
            try:
                X, y, regime_labels, feature_names, preparation_metrics = self._prepare_features(
                    X, y, regime_labels, feature_names, hmm_states, 
                    analyst_signals, analyst_model_outputs, hmm_regime_features, 
                    all_analyst_models_outputs, hmm_model_outputs, analyst_ensemble_outputs
                )
                
                # Log feature preparation metrics
                if preparation_metrics.get('green_light_filtering'):
                    gl_filtering = preparation_metrics['green_light_filtering']
                    self._log_performance_metric(
                        "green_light_rate", 
                        gl_filtering.get('green_light_rate', 0) * 100, 
                        "%"
                    )
                
                self._complete_phase(TrainingPhase.FEATURE_PREPARATION, success=True,
                                   samples_processed=X.shape[0], features_count=X.shape[1],
                                   warnings_issued=len(preparation_metrics.get('warnings', [])),
                                   errors_encountered=len(preparation_metrics.get('errors', [])))
            except Exception as e:
                self._complete_phase(TrainingPhase.FEATURE_PREPARATION, success=False, error_message=str(e))
                raise
            
            # Phase 3: Model Training
            self._start_phase(TrainingPhase.MODEL_TRAINING)
            try:
                results = self._execute_training(X, y, regime_labels, feature_names, hmm_states)
                self._complete_phase(TrainingPhase.MODEL_TRAINING, success=True,
                                   models_trained=len(results.get('models', {})),
                                   memory_usage_mb=self._get_memory_usage())
            except Exception as e:
                self._complete_phase(TrainingPhase.MODEL_TRAINING, success=False, error_message=str(e))
                raise
            
            # Phase 4: Finalization
            self._start_phase(TrainingPhase.FINALIZATION)
            try:
                results = self._finalize_results(results, analyst_signals)
                total_time = time.time() - self.overall_start_time
                self._complete_phase(TrainingPhase.FINALIZATION, success=True)
                
                # Generate comprehensive training report
                self._generate_training_report(total_time)
                
                return results
                
            except Exception as e:
                self._complete_phase(TrainingPhase.FINALIZATION, success=False, error_message=str(e))
                raise
                
        except Exception as e:
            self.logger.error(f"❌ Enhanced Tactician training failed: {e}")
            self.logger.error(f"❌ Traceback: {traceback.format_exc()}")
            return self._create_error_result(str(e))
    
    def _prepare_features(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regime_labels: np.ndarray,
        feature_names: Optional[List[str]],
        hmm_states: Optional[np.ndarray],
        analyst_signals: Optional[np.ndarray],
        analyst_model_outputs: Optional[np.ndarray],
        hmm_regime_features: Optional[np.ndarray],
        all_analyst_models_outputs: Optional[Dict[str, np.ndarray]],
        hmm_model_outputs: Optional[np.ndarray],
        analyst_ensemble_outputs: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[List[str]], Dict[str, Any]]:
        """Prepare and combine all features with comprehensive error handling and validation."""
        preparation_metrics = {
            'original_samples': X.shape[0],
            'original_features': X.shape[1],
            'green_light_filtering': {},
            'feature_combinations': {},
            'final_samples': 0,
            'final_features': 0,
            'warnings': [],
            'errors': []
        }
        
        try:
            # Filter data to only include periods where Analyst gives green light
            if analyst_signals is not None:
                green_light_mask = analyst_signals == 1
                green_light_count = np.sum(green_light_mask)
                green_light_rate = green_light_count / len(analyst_signals)
                
                preparation_metrics['green_light_filtering'] = {
                    'total_signals': len(analyst_signals),
                    'green_light_count': green_light_count,
                    'green_light_rate': green_light_rate
                }
                
                self.logger.info(f"📊 Filtering to {green_light_count} samples with Analyst green light signals ({green_light_rate:.2%})")
                
                # Validate green light filtering results
                if green_light_count == 0:
                    error_msg = "No samples with Analyst green light signals found"
                    preparation_metrics['errors'].append(error_msg)
                    raise ValueError(error_msg)
                
                if green_light_rate < 0.01:  # Less than 1%
                    warning_msg = f"Very low green light rate: {green_light_rate:.2%}"
                    preparation_metrics['warnings'].append(warning_msg)
                    self.logger.warning(f"⚠️ {warning_msg}")
                
                # Apply filtering with validation
                X_filtered = X[green_light_mask]
                y_filtered = y[green_light_mask]
                regime_labels_filtered = regime_labels[green_light_mask]
                
                # Validate filtered data shapes
                if X_filtered.shape[0] != green_light_count:
                    error_msg = f"Filtered data shape mismatch: expected {green_light_count}, got {X_filtered.shape[0]}"
                    preparation_metrics['errors'].append(error_msg)
                    raise ValueError(error_msg)
                
                X, y, regime_labels = X_filtered, y_filtered, regime_labels_filtered
                
                if hmm_states is not None:
                    hmm_states = hmm_states[green_light_mask]
                    if hmm_states.shape[0] != green_light_count:
                        error_msg = f"HMM states filtering mismatch: expected {green_light_count}, got {hmm_states.shape[0]}"
                        preparation_metrics['errors'].append(error_msg)
                        raise ValueError(error_msg)
            
            # Combine all features: base features + HMM regime features + HMM model outputs + all analyst model outputs + analyst ensemble outputs
            additional_features = []
            additional_feature_names = []
            feature_combination_details = {}
            
            # Add HMM regime features if provided
            if hmm_regime_features is not None:
                try:
                    if analyst_signals is not None:
                        hmm_regime_features = hmm_regime_features[green_light_mask]
                    
                    # Validate HMM features shape
                    if hmm_regime_features.shape[0] != X.shape[0]:
                        error_msg = f"HMM regime features shape mismatch: expected {X.shape[0]}, got {hmm_regime_features.shape[0]}"
                        preparation_metrics['errors'].append(error_msg)
                        raise ValueError(error_msg)
                    
                    # Check for NaN/Inf in HMM features
                    hmm_nan_count = np.sum(np.isnan(hmm_regime_features))
                    hmm_inf_count = np.sum(np.isinf(hmm_regime_features))
                    
                    if hmm_nan_count > 0:
                        warning_msg = f"HMM features contain {hmm_nan_count} NaN values"
                        preparation_metrics['warnings'].append(warning_msg)
                        self.logger.warning(f"⚠️ {warning_msg}")
                    
                    if hmm_inf_count > 0:
                        warning_msg = f"HMM features contain {hmm_inf_count} infinite values"
                        preparation_metrics['warnings'].append(warning_msg)
                        self.logger.warning(f"⚠️ {warning_msg}")
                    
                    additional_features.append(hmm_regime_features)
                    additional_feature_names.extend([f"hmm_regime_{i}" for i in range(hmm_regime_features.shape[1])])
                    
                    feature_combination_details['hmm_regime_features'] = {
                        'count': hmm_regime_features.shape[1],
                        'nan_count': hmm_nan_count,
                        'inf_count': hmm_inf_count
                    }
                    
                    self.logger.info(f"📊 Added {hmm_regime_features.shape[1]} HMM regime features")
                    
                except Exception as e:
                    error_msg = f"Failed to add HMM regime features: {e}"
                    preparation_metrics['errors'].append(error_msg)
                    self.logger.error(f"❌ {error_msg}")
                    raise
            
            # Add HMM model outputs if provided
            if hmm_model_outputs is not None:
                try:
                    if analyst_signals is not None:
                        hmm_model_outputs = hmm_model_outputs[green_light_mask]
                    
                    # Validate HMM model outputs shape
                    if hmm_model_outputs.shape[0] != X.shape[0]:
                        error_msg = f"HMM model outputs shape mismatch: expected {X.shape[0]}, got {hmm_model_outputs.shape[0]}"
                        preparation_metrics['errors'].append(error_msg)
                        raise ValueError(error_msg)
                    
                    # Check for NaN/Inf in HMM model outputs
                    hmm_outputs_nan_count = np.sum(np.isnan(hmm_model_outputs))
                    hmm_outputs_inf_count = np.sum(np.isinf(hmm_model_outputs))
                    
                    if hmm_outputs_nan_count > 0:
                        warning_msg = f"HMM model outputs contain {hmm_outputs_nan_count} NaN values"
                        preparation_metrics['warnings'].append(warning_msg)
                        self.logger.warning(f"⚠️ {warning_msg}")
                    
                    if hmm_outputs_inf_count > 0:
                        warning_msg = f"HMM model outputs contain {hmm_outputs_inf_count} infinite values"
                        preparation_metrics['warnings'].append(warning_msg)
                        self.logger.warning(f"⚠️ {warning_msg}")
                    
                    additional_features.append(hmm_model_outputs)
                    additional_feature_names.extend([f"hmm_model_{i}" for i in range(hmm_model_outputs.shape[1])])
                    
                    feature_combination_details['hmm_model_outputs'] = {
                        'count': hmm_model_outputs.shape[1],
                        'nan_count': hmm_outputs_nan_count,
                        'inf_count': hmm_outputs_inf_count
                    }
                    
                    self.logger.info(f"📊 Added {hmm_model_outputs.shape[1]} HMM model outputs")
                    
                except Exception as e:
                    error_msg = f"Failed to add HMM model outputs: {e}"
                    preparation_metrics['errors'].append(error_msg)
                    self.logger.error(f"❌ {error_msg}")
                    raise
            
            # Add all individual analyst model outputs if provided
            if all_analyst_models_outputs is not None:
                analyst_features_added = 0
                for model_name, model_outputs in all_analyst_models_outputs.items():
                    try:
                        if analyst_signals is not None:
                            model_outputs = model_outputs[green_light_mask]
                        
                        # Validate model outputs shape
                        if model_outputs.shape[0] != X.shape[0]:
                            error_msg = f"Analyst model {model_name} output shape mismatch: expected {X.shape[0]}, got {model_outputs.shape[0]}"
                            preparation_metrics['errors'].append(error_msg)
                            raise ValueError(error_msg)
                        
                        # Check for NaN/Inf in model outputs
                        model_nan_count = np.sum(np.isnan(model_outputs))
                        model_inf_count = np.sum(np.isinf(model_outputs))
                        
                        if model_nan_count > 0:
                            warning_msg = f"Analyst model {model_name} outputs contain {model_nan_count} NaN values"
                            preparation_metrics['warnings'].append(warning_msg)
                            self.logger.warning(f"⚠️ {warning_msg}")
                        
                        if model_inf_count > 0:
                            warning_msg = f"Analyst model {model_name} outputs contain {model_inf_count} infinite values"
                            preparation_metrics['warnings'].append(warning_msg)
                            self.logger.warning(f"⚠️ {warning_msg}")
                        
                        additional_features.append(model_outputs)
                        additional_feature_names.extend([f"analyst_{model_name}_{i}" for i in range(model_outputs.shape[1])])
                        analyst_features_added += model_outputs.shape[1]
                        
                    except Exception as e:
                        error_msg = f"Failed to add analyst model {model_name} outputs: {e}"
                        preparation_metrics['errors'].append(error_msg)
                        self.logger.error(f"❌ {error_msg}")
                        raise
                
                feature_combination_details['analyst_models'] = {
                    'model_count': len(all_analyst_models_outputs),
                    'total_features': analyst_features_added
                }
                
                self.logger.info(f"📊 Added outputs from {len(all_analyst_models_outputs)} analyst models ({analyst_features_added} features)")
            
            # Add analyst ensemble outputs if provided
            if analyst_ensemble_outputs is not None:
                try:
                    if analyst_signals is not None:
                        analyst_ensemble_outputs = analyst_ensemble_outputs[green_light_mask]
                    
                    # Validate analyst ensemble outputs shape
                    if analyst_ensemble_outputs.shape[0] != X.shape[0]:
                        error_msg = f"Analyst ensemble outputs shape mismatch: expected {X.shape[0]}, got {analyst_ensemble_outputs.shape[0]}"
                        preparation_metrics['errors'].append(error_msg)
                        raise ValueError(error_msg)
                    
                    # Check for NaN/Inf in analyst ensemble outputs
                    ensemble_nan_count = np.sum(np.isnan(analyst_ensemble_outputs))
                    ensemble_inf_count = np.sum(np.isinf(analyst_ensemble_outputs))
                    
                    if ensemble_nan_count > 0:
                        warning_msg = f"Analyst ensemble outputs contain {ensemble_nan_count} NaN values"
                        preparation_metrics['warnings'].append(warning_msg)
                        self.logger.warning(f"⚠️ {warning_msg}")
                    
                    if ensemble_inf_count > 0:
                        warning_msg = f"Analyst ensemble outputs contain {ensemble_inf_count} infinite values"
                        preparation_metrics['warnings'].append(warning_msg)
                        self.logger.warning(f"⚠️ {warning_msg}")
                    
                    additional_features.append(analyst_ensemble_outputs)
                    additional_feature_names.extend([f"analyst_ensemble_{i}" for i in range(analyst_ensemble_outputs.shape[1])])
                    
                    feature_combination_details['analyst_ensemble_outputs'] = {
                        'count': analyst_ensemble_outputs.shape[1],
                        'nan_count': ensemble_nan_count,
                        'inf_count': ensemble_inf_count
                    }
                    
                    self.logger.info(f"📊 Added {analyst_ensemble_outputs.shape[1]} analyst ensemble outputs")
                    
                except Exception as e:
                    error_msg = f"Failed to add analyst ensemble outputs: {e}"
                    preparation_metrics['errors'].append(error_msg)
                    self.logger.error(f"❌ {error_msg}")
                    raise
            
            # Add legacy analyst model outputs for backward compatibility
            if analyst_model_outputs is not None:
                try:
                    if analyst_signals is not None:
                        analyst_model_outputs = analyst_model_outputs[green_light_mask]
                    
                    # Validate legacy outputs shape
                    if analyst_model_outputs.shape[0] != X.shape[0]:
                        error_msg = f"Legacy analyst outputs shape mismatch: expected {X.shape[0]}, got {analyst_model_outputs.shape[0]}"
                        preparation_metrics['errors'].append(error_msg)
                        raise ValueError(error_msg)
                    
                    # Check for NaN/Inf in legacy outputs
                    legacy_nan_count = np.sum(np.isnan(analyst_model_outputs))
                    legacy_inf_count = np.sum(np.isinf(analyst_model_outputs))
                    
                    if legacy_nan_count > 0:
                        warning_msg = f"Legacy analyst outputs contain {legacy_nan_count} NaN values"
                        preparation_metrics['warnings'].append(warning_msg)
                        self.logger.warning(f"⚠️ {warning_msg}")
                    
                    if legacy_inf_count > 0:
                        warning_msg = f"Legacy analyst outputs contain {legacy_inf_count} infinite values"
                        preparation_metrics['warnings'].append(warning_msg)
                        self.logger.warning(f"⚠️ {warning_msg}")
                    
                    additional_features.append(analyst_model_outputs)
                    additional_feature_names.extend([f"analyst_legacy_{i}" for i in range(analyst_model_outputs.shape[1])])
                    
                    feature_combination_details['analyst_legacy'] = {
                        'count': analyst_model_outputs.shape[1],
                        'nan_count': legacy_nan_count,
                        'inf_count': legacy_inf_count
                    }
                    
                    self.logger.info(f"📊 Added {analyst_model_outputs.shape[1]} legacy analyst outputs")
                    
                except Exception as e:
                    error_msg = f"Failed to add legacy analyst outputs: {e}"
                    preparation_metrics['errors'].append(error_msg)
                    self.logger.error(f"❌ {error_msg}")
                    raise
            
            # Concatenate all additional features with validation
            if additional_features:
                try:
                    # Validate all features have same number of samples
                    for i, feature_array in enumerate(additional_features):
                        if feature_array.shape[0] != X.shape[0]:
                            error_msg = f"Feature array {i} shape mismatch: expected {X.shape[0]}, got {feature_array.shape[0]}"
                            preparation_metrics['errors'].append(error_msg)
                            raise ValueError(error_msg)
                    
                    X_combined = np.column_stack([X] + additional_features)
                    
                    # Validate combined features
                    if X_combined.shape[0] != X.shape[0]:
                        error_msg = f"Combined features sample count mismatch: expected {X.shape[0]}, got {X_combined.shape[0]}"
                        preparation_metrics['errors'].append(error_msg)
                        raise ValueError(error_msg)
                    
                    X = X_combined
                    
                    # Update feature names
                    if feature_names is not None:
                        feature_names = feature_names + additional_feature_names
                    else:
                        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
                    
                    preparation_metrics['feature_combinations'] = feature_combination_details
                    self.logger.info(f"📊 Total features: {X.shape[1]} (base + HMM regime + HMM model + all analyst models + analyst ensemble)")
                    
                except Exception as e:
                    error_msg = f"Failed to combine features: {e}"
                    preparation_metrics['errors'].append(error_msg)
                    self.logger.error(f"❌ {error_msg}")
                    raise
            
            # Final validation
            preparation_metrics['final_samples'] = X.shape[0]
            preparation_metrics['final_features'] = X.shape[1]
            
            # Check for final data quality issues
            final_nan_count = np.sum(np.isnan(X))
            final_inf_count = np.sum(np.isinf(X))
            
            if final_nan_count > 0:
                warning_msg = f"Final feature matrix contains {final_nan_count} NaN values"
                preparation_metrics['warnings'].append(warning_msg)
                self.logger.warning(f"⚠️ {warning_msg}")
            
            if final_inf_count > 0:
                warning_msg = f"Final feature matrix contains {final_inf_count} infinite values"
                preparation_metrics['warnings'].append(warning_msg)
                self.logger.warning(f"⚠️ {warning_msg}")
            
            return X, y, regime_labels, feature_names, preparation_metrics
            
        except Exception as e:
            preparation_metrics['errors'].append(str(e))
            self.logger.error(f"❌ Feature preparation failed: {e}")
            raise
    
    def _execute_training(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regime_labels: np.ndarray,
        feature_names: Optional[List[str]],
        hmm_states: Optional[np.ndarray]
    ) -> Dict[str, Any]:
        """Execute training with enhanced vectorization and comprehensive error handling."""
        training_metrics = {
            'vectorization_attempted': False,
            'vectorization_successful': False,
            'fallback_used': False,
            'training_method': 'unknown',
            'errors': [],
            'warnings': [],
            'performance_metrics': {}
        }
        
        try:
            # Pre-training validation
            self._validate_training_inputs(X, y, regime_labels, feature_names, hmm_states)
            
            # VECTORIZED: Use ultra-fast vectorized training by default
            self.logger.info("🚀 Using VECTORIZED tactician models training")
            training_metrics['vectorization_attempted'] = True
            
            if self.enable_vectorization:
                try:
                    vectorization_start_time = time.time()
                    
                    results = super().execute_vectorized(
                        X=X,
                        y=y,
                        regime_labels=regime_labels,
                        feature_names=feature_names,
                        hmm_states=hmm_states,
                        is_classification=False,  # Tactician models are typically regression
                        symbol=None,
                        exchange=None,
                        timeframe=self.config.timeframe
                    )
                    
                    vectorization_duration = time.time() - vectorization_start_time
                    training_metrics['performance_metrics']['vectorization_duration'] = vectorization_duration
                    
                    if results.get('vectorized', False):
                        training_metrics['vectorization_successful'] = True
                        training_metrics['training_method'] = 'vectorized'
                        self.logger.info(f"✅ VECTORIZED tactician training completed successfully in {vectorization_duration:.2f}s")
                        return results
                    else:
                        warning_msg = "VECTORIZED tactician training failed, falling back to standard method"
                        training_metrics['warnings'].append(warning_msg)
                        training_metrics['fallback_used'] = True
                        self.logger.warning(f"⚠️ {warning_msg}")
                        
                except Exception as e:
                    error_msg = f"VECTORIZED tactician training failed: {e}"
                    training_metrics['warnings'].append(error_msg)
                    training_metrics['fallback_used'] = True
                    self.logger.warning(f"⚠️ {error_msg}, falling back to standard method")
            else:
                training_metrics['fallback_used'] = True
                self.logger.info("🔄 Vectorization disabled, using standard training")
            
            # Fallback to standard training
            self.logger.info("🔄 Using standard tactician models training")
            standard_start_time = time.time()
            
            results = super().execute(
                X=X,
                y=y,
                regime_labels=regime_labels,
                feature_names=feature_names,
                hmm_states=hmm_states,
                is_classification=False,
                symbol=None,
                exchange=None,
                timeframe=self.config.timeframe
            )
            
            standard_duration = time.time() - standard_start_time
            training_metrics['performance_metrics']['standard_training_duration'] = standard_duration
            training_metrics['training_method'] = 'standard'
            
            self.logger.info(f"✅ Standard tactician training completed in {standard_duration:.2f}s")
            
            # Add training metrics to results
            results['training_execution_metrics'] = training_metrics
            
            # Add entry timing optimization if enabled
            if hasattr(self.config, 'enable_entry_timing_optimization') and self.config.enable_entry_timing_optimization:
                self.logger.info("🔄 Applying entry timing optimization for 0-0.5% range...")
                entry_timing_results = self._apply_entry_timing_optimization(X, y, feature_names, results)
                results.update(entry_timing_results)
            
            # Always add ensemble training for Tactician (core requirement)
            self.logger.info("🔄 Training ensemble model (always enabled for Tactician)...")
            ensemble_results = self._train_ensemble_model(X, y, feature_names, results)
            results.update(ensemble_results)
            
            return results
            
        except Exception as e:
            training_metrics['errors'].append(str(e))
            self.logger.error(f"❌ Training execution failed: {e}")
            self.logger.error(f"❌ Training metrics: {training_metrics}")
            raise
    
    def _validate_training_inputs(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regime_labels: np.ndarray,
        feature_names: Optional[List[str]],
        hmm_states: Optional[np.ndarray]
    ) -> None:
        """Validate training inputs before execution."""
        try:
            # Check data shapes
            if X.shape[0] != y.shape[0]:
                raise ValueError(f"Feature and target sample counts don't match: {X.shape[0]} vs {y.shape[0]}")
            
            if X.shape[0] != regime_labels.shape[0]:
                raise ValueError(f"Feature and regime label sample counts don't match: {X.shape[0]} vs {regime_labels.shape[0]}")
            
            if hmm_states is not None and hmm_states.shape[0] != X.shape[0]:
                raise ValueError(f"HMM states sample count mismatch: expected {X.shape[0]}, got {hmm_states.shape[0]}")
            
            # Check feature names consistency
            if feature_names is not None and len(feature_names) != X.shape[1]:
                raise ValueError(f"Feature names count mismatch: expected {X.shape[1]}, got {len(feature_names)}")
            
            # Check for critical data quality issues
            if np.any(np.isnan(X)):
                nan_percentage = (np.sum(np.isnan(X)) / X.size) * 100
                if nan_percentage > 5:  # More than 5% NaN
                    raise ValueError(f"Critical: {nan_percentage:.2f}% of features are NaN")
            
            if np.any(np.isnan(y)):
                nan_percentage = (np.sum(np.isnan(y)) / y.size) * 100
                if nan_percentage > 2:  # More than 2% NaN in targets
                    raise ValueError(f"Critical: {nan_percentage:.2f}% of targets are NaN")
            
            # Check regime distribution
            unique_regimes = np.unique(regime_labels)
            regime_counts = np.bincount(regime_labels)
            min_regime_size = np.min(regime_counts)
            
            if min_regime_size < self.config.min_samples_per_regime:
                insufficient_regimes = np.sum(regime_counts < self.config.min_samples_per_regime)
                if insufficient_regimes > len(unique_regimes) * 0.5:
                    raise ValueError(f"Critical: {insufficient_regimes}/{len(unique_regimes)} regimes have insufficient data")
            
            self.logger.info("✅ Training input validation passed")
            
        except Exception as e:
            self.logger.error(f"❌ Training input validation failed: {e}")
            raise
    
    def _apply_entry_timing_optimization(self,
                                      X: np.ndarray,
                                      y: np.ndarray,
                                      feature_names: Optional[List[str]],
                                      base_results: Dict[str, Any]) -> Dict[str, Any]:
        """Apply entry timing optimization for 0-0.5% range."""
        try:
            from .tactician_directional_optimization import EntryTimingTacticianOptimizer
            
            # Initialize entry timing optimizer
            entry_timing_optimizer = EntryTimingTacticianOptimizer(self.config)
            
            # Get entry timing range from config
            entry_timing_range = getattr(self.config, 'entry_timing_range', 0.005)  # 0-0.5% range
            
            # Filter targets for entry timing range (0-0.5%)
            entry_timing_mask = np.abs(y) <= entry_timing_range
            X_entry_timing = X[entry_timing_mask]
            y_entry_timing = y[entry_timing_mask]
            
            self.logger.info(f"📊 Entry timing filtering: {len(y_entry_timing)}/{len(y)} samples (≤{entry_timing_range:.1%} range)")
            
            if len(y_entry_timing) < 100:  # Need minimum samples for optimization
                self.logger.warning("⚠️ Insufficient entry timing samples for optimization")
                return {}
            
            # Apply entry timing optimization
            entry_timing_result = entry_timing_optimizer.optimize_tactician_entry_timing(
                X=X_entry_timing, y=y_entry_timing, regime_labels=np.zeros(len(y_entry_timing)),
                feature_names=feature_names, hmm_states=None,
                max_trials=getattr(self.config, 'hpo_n_trials', 100) // 2  # Half trials for entry timing
            )
            
            # Create entry timing optimization results
            entry_timing_results = {
                'entry_timing_optimization': {
                    'enabled': True,
                    'entry_timing_range': entry_timing_range,
                    'entry_timing_samples': len(y_entry_timing),
                    'total_samples': len(y),
                    'objectives': getattr(self.config, 'entry_timing_objectives', {}),
                    'optimization_time': entry_timing_result.optimization_time,
                    'n_trials': entry_timing_result.n_trials
                },
                'entry_timing_model': entry_timing_result.model,
                'entry_timing_metrics': {
                    'early_entry_penalty': entry_timing_result.directional_accuracy,
                    'late_entry_penalty': entry_timing_result.adverse_movement_minimization,
                    'optimal_entry_reward': entry_timing_result.directional_profit_efficiency,
                    'entry_timing_efficiency': entry_timing_result.risk_adjusted_performance,
                    'composite_score': entry_timing_result.composite_score
                }
            }
            
            self.logger.info(f"✅ Entry timing optimization completed for 0-0.5% range")
            self.logger.info(f"   Early entry penalty: {entry_timing_result.directional_accuracy:.4f}")
            self.logger.info(f"   Late entry penalty: {entry_timing_result.adverse_movement_minimization:.4f}")
            self.logger.info(f"   Optimal entry reward: {entry_timing_result.directional_profit_efficiency:.4f}")
            self.logger.info(f"   Entry timing efficiency: {entry_timing_result.risk_adjusted_performance:.4f}")
            self.logger.info(f"   Composite score: {entry_timing_result.composite_score:.4f}")
            
            return entry_timing_results
            
        except Exception as e:
            self.logger.error(f"❌ Entry timing optimization failed: {e}")
            return {}
    
    def _train_ensemble_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]],
        base_models_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Train ensemble model using base models as inputs."""
        try:
            self.logger.info("🔄 Training ensemble model from base models...")
            
            # Get base models from results
            base_models = base_models_results.get('models', {})
            if not base_models:
                self.logger.warning("⚠️ No base models found for ensemble training")
                return {}
            
            # Generate base model predictions for ensemble training
            base_predictions = []
            base_model_names = []
            
            for model_name, model in base_models.items():
                try:
                    # Generate predictions using the base model
                    if hasattr(model, 'predict'):
                        predictions = model.predict(X)
                        base_predictions.append(predictions.reshape(-1, 1))
                        base_model_names.append(model_name)
                        self.logger.info(f"📊 Generated predictions from {model_name}")
                    else:
                        self.logger.warning(f"⚠️ Model {model_name} does not have predict method")
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to generate predictions from {model_name}: {e}")
            
            if not base_predictions:
                self.logger.warning("⚠️ No valid base model predictions for ensemble training")
                return {}
            
            # Combine base model predictions
            X_ensemble = np.column_stack(base_predictions)
            ensemble_feature_names = [f"base_model_{name}" for name in base_model_names]
            
            self.logger.info(f"📊 Ensemble training data: {X_ensemble.shape[0]} samples, {X_ensemble.shape[1]} base model predictions")
            
            # Train ensemble model
            ensemble_method = getattr(self.config, 'ensemble_method', 'stacking')
            meta_model_type = getattr(self.config, 'meta_model', 'ElasticNetCV')
            ensemble_name = getattr(self.config, 'ensemble_name', 'tactician_ensemble')
            
            if ensemble_method == 'stacking':
                # Use stacking ensemble
                ensemble_model = self._train_stacking_ensemble(
                    X_ensemble, y, meta_model_type, ensemble_name
                )
            else:
                # Use simple meta-model
                ensemble_model = self.training_utils.train_single_model(
                    model_type=meta_model_type,
                    X=X_ensemble,
                    y=y,
                    model_name=ensemble_name
                )
            
            # Evaluate ensemble model
            ensemble_evaluation = self.training_utils.evaluate_model(
                model=ensemble_model,
                X=X_ensemble,
                y=y,
                metrics=self.config.evaluation_metrics
            )
            
            ensemble_results = {
                'ensemble_model': ensemble_model,
                'ensemble_evaluation': ensemble_evaluation,
                'ensemble_method': ensemble_method,
                'meta_model_type': meta_model_type,
                'base_models_used': base_model_names,
                'ensemble_feature_names': ensemble_feature_names
            }
            
            self.logger.info(f"✅ Ensemble training completed: {ensemble_method} with {meta_model_type}")
            self.logger.info(f"📊 Ensemble performance: {ensemble_evaluation}")
            
            return ensemble_results
            
        except Exception as e:
            self.logger.error(f"❌ Ensemble training failed: {e}")
            return {}
    
    def _train_stacking_ensemble(
        self,
        X_ensemble: np.ndarray,
        y: np.ndarray,
        meta_model_type: str,
        ensemble_name: str
    ):
        """Train a stacking ensemble model."""
        try:
            from sklearn.ensemble import StackingRegressor
            from sklearn.model_selection import cross_val_predict
            
            # Create base estimators from the ensemble features
            base_estimators = []
            for i in range(X_ensemble.shape[1]):
                # Use simple models as base estimators for stacking
                base_estimator = self.training_utils.create_model('Ridge')
                base_estimators.append((f'base_{i}', base_estimator))
            
            # Create meta-model (ElasticNetCV for better performance)
            if meta_model_type == 'ElasticNetCV':
                from sklearn.linear_model import ElasticNetCV
                meta_model = ElasticNetCV(
                    cv=5,
                    random_state=42,
                    l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0],
                    alphas=np.logspace(-4, 1, 50)
                )
            else:
                meta_model = self.training_utils.create_model(meta_model_type)
            
            # Create stacking regressor
            stacking_regressor = StackingRegressor(
                estimators=base_estimators,
                final_estimator=meta_model,
                cv=5,  # 5-fold cross-validation
                stack_method='predict'
            )
            
            # Train the stacking ensemble
            stacking_regressor.fit(X_ensemble, y)
            
            self.logger.info(f"✅ Stacking ensemble trained with {len(base_estimators)} base estimators")
            
            return stacking_regressor
            
        except Exception as e:
            self.logger.error(f"❌ Stacking ensemble training failed: {e}")
            # Fallback to simple meta-model with ElasticNetCV
            if meta_model_type == 'ElasticNetCV':
                from sklearn.linear_model import ElasticNetCV
                from sklearn.model_selection import cross_val_score
                
                # Create and train ElasticNetCV directly
                elastic_net = ElasticNetCV(
                    cv=5,
                    random_state=42,
                    l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0],
                    alphas=np.logspace(-4, 1, 50)
                )
                elastic_net.fit(X_ensemble, y)
                return elastic_net
            else:
                return self.training_utils.train_single_model(
                    model_type=meta_model_type,
                    X=X_ensemble,
                    y=y,
                    model_name=ensemble_name
                )
    
    def _finalize_results(self, results: Dict[str, Any], analyst_signals: Optional[np.ndarray]) -> Dict[str, Any]:
        """Finalize results with tactician-specific metadata and comprehensive reporting."""
        try:
            # Add tactician-specific post-processing if needed
            if 'error' not in results:
                results = self._add_tactician_specific_metadata(results, analyst_signals)
            
            # Add training metrics to results
            results['training_metrics'] = {
                phase.value: {
                    'duration': metrics.duration,
                    'success': metrics.success,
                    'samples_processed': metrics.samples_processed,
                    'features_count': metrics.features_count,
                    'models_trained': metrics.models_trained,
                    'memory_usage_mb': metrics.memory_usage_mb,
                    'error_message': metrics.error_message
                }
                for phase, metrics in self.training_metrics.items()
            }
            
            # Add vectorization information
            results['vectorization_info'] = {
                'vectorization_enabled': self.enable_vectorization,
                'vectorization_fallback_used': self.vectorization_fallback_used,
                'vectorized_training_available': VECTORIZED_TRAINING_AVAILABLE
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Results finalization failed: {e}")
            return results
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create error result with comprehensive error information."""
        return {
            'error': True,
            'error_message': error_message,
            'training_metrics': {
                phase.value: {
                    'duration': metrics.duration,
                    'success': metrics.success,
                    'error_message': metrics.error_message
                }
                for phase, metrics in self.training_metrics.items()
            },
            'vectorization_info': {
                'vectorization_enabled': self.enable_vectorization,
                'vectorization_fallback_used': self.vectorization_fallback_used,
                'vectorized_training_available': VECTORIZED_TRAINING_AVAILABLE
            }
        }
    
    def _generate_training_report(self, total_time: float) -> None:
        """Generate comprehensive training report with actionable insights."""
        try:
            self.logger.info("📊 " + "="*80)
            self.logger.info("📊 ENHANCED TACTICIAN TRAINING REPORT")
            self.logger.info("📊 " + "="*80)
            
            # Overall statistics
            self.logger.info(f"📊 Total training time: {total_time:.2f}s")
            self.logger.info(f"📊 Vectorization enabled: {self.enable_vectorization}")
            self.logger.info(f"📊 Vectorization fallback used: {self.vectorization_fallback_used}")
            
            # Calculate efficiency metrics
            total_warnings = sum(len(metrics.warnings_issued) for metrics in self.training_metrics.values())
            total_errors = sum(len(metrics.errors_encountered) for metrics in self.training_metrics.values())
            total_samples = sum(metrics.samples_processed for metrics in self.training_metrics.values())
            total_features = sum(metrics.features_count for metrics in self.training_metrics.values())
            
            self.logger.info(f"📊 Total samples processed: {total_samples:,}")
            self.logger.info(f"📊 Total features: {total_features}")
            self.logger.info(f"📊 Total warnings: {total_warnings}")
            self.logger.info(f"📊 Total errors: {total_errors}")
            
            # Performance efficiency
            if total_samples > 0:
                samples_per_second = total_samples / total_time
                self.logger.info(f"📊 Processing rate: {samples_per_second:,.0f} samples/second")
            
            # Phase-by-phase breakdown with detailed metrics
            self.logger.info("📊 " + "-"*60)
            self.logger.info("📊 PHASE BREAKDOWN:")
            self.logger.info("📊 " + "-"*60)
            
            for phase, metrics in self.training_metrics.items():
                status = "✅" if metrics.success else "❌"
                efficiency = f"({metrics.samples_processed/1000:.1f}k samples)" if metrics.samples_processed > 0 else ""
                
                self.logger.info(f"📊   {status} {phase.value.upper()}: {metrics.duration:.2f}s {efficiency}")
                
                # Detailed phase metrics
                if metrics.samples_processed > 0:
                    self.logger.info(f"📊     └─ Samples: {metrics.samples_processed:,}")
                if metrics.features_count > 0:
                    self.logger.info(f"📊     └─ Features: {metrics.features_count}")
                if metrics.models_trained > 0:
                    self.logger.info(f"📊     └─ Models trained: {metrics.models_trained}")
                if metrics.warnings_issued > 0:
                    self.logger.info(f"📊     └─ Warnings: {metrics.warnings_issued}")
                if metrics.errors_encountered > 0:
                    self.logger.info(f"📊     └─ Errors: {metrics.errors_encountered}")
                if metrics.memory_usage_mb > 0:
                    self.logger.info(f"📊     └─ Memory: {metrics.memory_usage_mb:.1f} MB")
                
                if not metrics.success and metrics.error_message:
                    self.logger.info(f"📊     └─ ❌ Error: {metrics.error_message}")
            
            # Data quality summary
            self.logger.info("📊 " + "-"*60)
            self.logger.info("📊 DATA QUALITY SUMMARY:")
            self.logger.info("📊 " + "-"*60)
            
            # Analyze data validation results if available
            if TrainingPhase.DATA_VALIDATION in self.training_metrics:
                data_phase = self.training_metrics[TrainingPhase.DATA_VALIDATION]
                if data_phase.success:
                    self.logger.info("📊   ✅ Data validation passed")
                else:
                    self.logger.info("📊   ❌ Data validation failed")
            
            # Analyze feature preparation results if available
            if TrainingPhase.FEATURE_PREPARATION in self.training_metrics:
                feature_phase = self.training_metrics[TrainingPhase.FEATURE_PREPARATION]
                if feature_phase.success:
                    self.logger.info("📊   ✅ Feature preparation completed")
                else:
                    self.logger.info("📊   ❌ Feature preparation failed")
            
            # Training method analysis
            self.logger.info("📊 " + "-"*60)
            self.logger.info("📊 TRAINING METHOD ANALYSIS:")
            self.logger.info("📊 " + "-"*60)
            
            if self.enable_vectorization:
                if self.vectorization_fallback_used:
                    self.logger.info("📊   ⚠️ Vectorization attempted but fallback used")
                    self.logger.info("📊   💡 Consider investigating vectorization issues")
                else:
                    self.logger.info("📊   ✅ Vectorization used successfully")
                    self.logger.info("📊   🚀 Optimal performance achieved")
            else:
                self.logger.info("📊   ℹ️ Standard training used (vectorization disabled)")
            
            # Memory usage analysis
            current_memory = self._get_memory_usage()
            self.logger.info("📊 " + "-"*60)
            self.logger.info("📊 MEMORY USAGE:")
            self.logger.info("📊 " + "-"*60)
            self.logger.info(f"📊 Current memory usage: {current_memory:.1f} MB")
            
            if current_memory > 1000:  # More than 1GB
                self.logger.info("📊   ⚠️ High memory usage detected")
                self.logger.info("📊   💡 Consider reducing batch size or using data streaming")
            elif current_memory < 100:  # Less than 100MB
                self.logger.info("📊   ✅ Low memory usage - efficient processing")
            
            # Recommendations
            self.logger.info("📊 " + "-"*60)
            self.logger.info("📊 RECOMMENDATIONS:")
            self.logger.info("📊 " + "-"*60)
            
            if total_warnings > 5:
                self.logger.info("📊   ⚠️ High warning count - review data quality")
            if total_errors > 0:
                self.logger.info("📊   ❌ Errors detected - review error logs")
            if self.vectorization_fallback_used:
                self.logger.info("📊   🔧 Vectorization fallback used - investigate vectorization issues")
            if total_time > 3600:  # More than 1 hour
                self.logger.info("📊   ⏱️ Long training time - consider optimizing hyperparameters")
            
            # Success indicators
            if total_errors == 0 and total_warnings < 3:
                self.logger.info("📊   ✅ Training completed successfully with minimal issues")
            
            self.logger.info("📊 " + "="*80)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate training report: {e}")
            self.logger.error(f"❌ Traceback: {traceback.format_exc()}")
    
    def _add_tactician_specific_metadata(self, results: Dict[str, Any], analyst_signals: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Add tactician-specific metadata to results with enhanced reporting.
        
        Args:
            results: Training results
            analyst_signals: Analyst green light signals for analysis
            
        Returns:
            Enhanced results with tactician-specific metadata
        """
        try:
            # Add tactician-specific analysis
            if 'regime_analysis' in results:
                regime_analysis = results['regime_analysis']
                
                # Calculate tactician-specific metrics
                tactician_metrics = {
                    'total_regimes': len(regime_analysis.get('unique_regimes', [])),
                    'sufficient_regimes': len(regime_analysis.get('sufficient_regimes', [])),
                    'insufficient_regimes': len(regime_analysis.get('insufficient_regimes', [])),
                    'regime_balance': regime_analysis.get('regime_balance_train', 0.0),
                    'timeframe': self.config.timeframe,
                    'model_types': self.config.model_types
                }
                
                # Add analyst signal analysis if available
                if analyst_signals is not None:
                    green_light_rate = np.mean(analyst_signals)
                    tactician_metrics.update({
                        'analyst_green_light_rate': green_light_rate,
                        'total_samples_with_green_light': int(np.sum(analyst_signals)),
                        'total_samples_analyzed': len(analyst_signals)
                    })
                
                results['tactician_metrics'] = tactician_metrics
            
            # Add model performance summary
            if 'evaluation_results' in results:
                evaluation_results = results['evaluation_results']
                
                # Calculate best performing model per regime
                best_models = {}
                for regime, regime_metrics in evaluation_results.items():
                    if isinstance(regime_metrics, dict) and 'error' not in regime_metrics:
                        best_model = None
                        best_r2 = -np.inf
                        
                        for model_name, metrics in regime_metrics.items():
                            if isinstance(metrics, dict) and 'r2' in metrics:
                                if metrics['r2'] > best_r2:
                                    best_r2 = metrics['r2']
                                    best_model = model_name
                        
                        if best_model:
                            best_models[regime] = {
                                'model': best_model,
                                'r2_score': best_r2
                            }
                
                results['best_models_per_regime'] = best_models
            
            # Add timing-specific analysis
            timing_analysis = {
                'base_timeframe': self.config.timeframe,
                'cross_timeframe_features': True,
                'analyst_dependency': True,
                'timing_decision_role': True
            }
            results['timing_analysis'] = timing_analysis
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Failed to add tactician-specific metadata: {e}")
            return results


# Enhanced convenience functions with better error handling
def create_tactician_models_training_step_refactored(
    config: Optional[TacticianTrainingConfig] = None,
    enable_vectorization: bool = True
) -> TacticianModelsTrainingStepRefactored:
    """
    Create enhanced Tactician models training step with comprehensive error handling.
    
    Args:
        config: Per-regime training configuration
        enable_vectorization: Whether to enable vectorized training
        
    Returns:
        Enhanced Tactician models training step
        
    Raises:
        Exception: If initialization fails
    """
    try:
        return TacticianModelsTrainingStepRefactored(config, enable_vectorization)
    except Exception as e:
        logger.error(f"❌ Failed to create tactician training step: {e}")
        raise


def execute_tactician_models_training_refactored(
    X: np.ndarray,
    y: np.ndarray,
    regime_labels: np.ndarray,
    config: Optional[TacticianTrainingConfig] = None,
    feature_names: Optional[List[str]] = None,
    hmm_states: Optional[np.ndarray] = None,
    analyst_signals: Optional[np.ndarray] = None,
    analyst_model_outputs: Optional[np.ndarray] = None,
    hmm_regime_features: Optional[np.ndarray] = None,
    all_analyst_models_outputs: Optional[Dict[str, np.ndarray]] = None,
    hmm_model_outputs: Optional[np.ndarray] = None,
    analyst_ensemble_outputs: Optional[np.ndarray] = None,
    enable_vectorization: bool = True
) -> Dict[str, Any]:
    """
    Execute enhanced Tactician models training step with comprehensive error handling.
    
    Args:
        X: Input features
        y: Target values
        regime_labels: Regime labels for each sample
        config: Per-regime training configuration
        feature_names: Names of input features
        hmm_states: HMM cluster/regime states
        analyst_signals: Binary signals from Analyst
        analyst_model_outputs: Analyst model predictions
        hmm_regime_features: HMM regime features
        all_analyst_models_outputs: All individual analyst ML model outputs
        hmm_model_outputs: HMM model outputs (predictions, probabilities, etc.)
        analyst_ensemble_outputs: Analyst ensemble model outputs
        enable_vectorization: Whether to enable vectorized training
        
    Returns:
        Dictionary containing training results and metadata
        
    Raises:
        Exception: If training fails
    """
    try:
        step = create_tactician_models_training_step_refactored(config, enable_vectorization)
        return step.execute(
            X, y, regime_labels, feature_names, hmm_states, 
            analyst_signals, analyst_model_outputs, hmm_regime_features, 
            all_analyst_models_outputs, hmm_model_outputs, analyst_ensemble_outputs
        )
    except Exception as e:
        logger.error(f"❌ Failed to execute tactician training: {e}")
        raise


# Enhanced example usage and comparison
if __name__ == "__main__":
    # Example of how to use the enhanced version
    print("Enhanced Tactician Models Training Step")
    print("=" * 50)
    
    # Create configuration with enhanced settings
    config = TacticianTrainingConfig(
        model_name="tactician_models",
        timeframe="1m",
        model_types=["NeuralObliviousDecisionEnsembles", "CatBoostRegressor", "LGBMRegressor", "Ridge"],
        hpo_n_trials=50,  # Reduced for demo
        enable_hpo=True,
        save_models=True,
        model_save_path="./models/tactician_models_enhanced",
        use_single_model=True,
        single_model_name="tactician_unified_model",
        enable_ensemble_training=True,
        ensemble_method="stacking",
        meta_model="ElasticNetCV",
        ensemble_name="tactician_ensemble"
    )
    
    # Create enhanced training step
    try:
        training_step = create_tactician_models_training_step_refactored(config)
        
        print(f"✅ Created enhanced tactician training step with {len(config.model_types)} model types")
        print(f"📊 HPO enabled: {config.enable_hpo}")
        print(f"💾 Save models: {config.save_models}")
        print(f"📁 Save path: {config.model_save_path}")
        print(f"⏰ Base timeframe: {config.timeframe}")
        
        # The actual training would be called with:
        # results = training_step.execute(X, y, regime_labels, feature_names, hmm_states, analyst_signals, analyst_model_outputs)
        
        print("\n🎯 Enhanced Tactician Module Features:")
        print("- Comprehensive error handling with detailed failure reporting")
        print("- Enhanced progress tracking with phase-based metrics")
        print("- Input validation and data quality checks")
        print("- Optimized vectorization with intelligent fallback")
        print("- Structured logging with performance monitoring")
        print("- Health monitoring throughout training process")
        
        print("\n🔄 Integration with Analyst:")
        print("- Receives green light signals from Analyst")
        print("- Uses Analyst predictions as additional features")
        print("- Focuses on timing rather than trade decision")
        print("- Operates on higher frequency (1m vs 5m)")
        
        print("\n📊 Enhanced Reporting Features:")
        print("- Phase-by-phase progress tracking")
        print("- Comprehensive training metrics")
        print("- Memory usage monitoring")
        print("- Vectorization status reporting")
        print("- Detailed error reporting with stack traces")
        
    except Exception as e:
        print(f"❌ Failed to create enhanced tactician training step: {e}")
        print("This demonstrates the enhanced error handling capabilities")