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
from src.utils.ml_common.config import PerRegimeTrainingConfig
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
    
    def __init__(self, config: Optional[PerRegimeTrainingConfig] = None, enable_vectorization: bool = True):
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
            config = PerRegimeTrainingConfig(
                model_name="tactician_models",
                timeframe="1m",
                model_types=["NeuralObliviousDecisionEnsembles", "CatBoostRegressor", "LGBMRegressor", "Ridge"],
                hpo_n_trials=100,
                hpo_timeout_seconds=3600,
                min_samples_per_regime=1000,
                enable_data_augmentation=True,
                augmentation_method="smote",
                model_save_path="./models/tactician_models",
                evaluation_metrics=["mse", "mae", "r2", "mape", "smape"]
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
    
    def _start_phase(self, phase: TrainingPhase) -> None:
        """Start tracking a training phase."""
        self.training_metrics[phase] = TrainingMetrics(
            phase=phase,
            start_time=time.time()
        )
        self.logger.info(f"🔄 Starting phase: {phase.value}")
    
    def _complete_phase(self, phase: TrainingPhase, success: bool = True, 
                       error_message: Optional[str] = None, **kwargs) -> None:
        """Complete a training phase with metrics."""
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
            status = "✅" if success else "❌"
            self.logger.info(f"{status} Completed phase: {phase.value} in {duration:.2f}s")
            
            if not success and error_message:
                self.logger.error(f"❌ Phase {phase.value} failed: {error_message}")
    
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
            
            # Data quality analysis
            data_quality = {}
            
            # Check for NaN values with detailed analysis
            x_nan_count = np.sum(np.isnan(X))
            y_nan_count = np.sum(np.isnan(y))
            
            if x_nan_count > 0:
                nan_percentage = (x_nan_count / X.size) * 100
                warning_msg = f"Found {x_nan_count} NaN values in features ({nan_percentage:.2f}%)"
                validation_results['warnings'].append(warning_msg)
                data_quality['feature_nan_count'] = x_nan_count
                data_quality['feature_nan_percentage'] = nan_percentage
                
                # Check if NaN percentage is critical
                if nan_percentage > 10:
                    error_msg = f"Critical: {nan_percentage:.2f}% of features are NaN"
                    validation_results['errors'].append(error_msg)
                    validation_results['is_valid'] = False
            
            if y_nan_count > 0:
                nan_percentage = (y_nan_count / y.size) * 100
                warning_msg = f"Found {y_nan_count} NaN values in targets ({nan_percentage:.2f}%)"
                validation_results['warnings'].append(warning_msg)
                data_quality['target_nan_count'] = y_nan_count
                data_quality['target_nan_percentage'] = nan_percentage
                
                # Check if NaN percentage is critical
                if nan_percentage > 5:
                    error_msg = f"Critical: {nan_percentage:.2f}% of targets are NaN"
                    validation_results['errors'].append(error_msg)
                    validation_results['is_valid'] = False
            
            # Check for infinite values with detailed analysis
            x_inf_count = np.sum(np.isinf(X))
            y_inf_count = np.sum(np.isinf(y))
            
            if x_inf_count > 0:
                inf_percentage = (x_inf_count / X.size) * 100
                warning_msg = f"Found {x_inf_count} infinite values in features ({inf_percentage:.2f}%)"
                validation_results['warnings'].append(warning_msg)
                data_quality['feature_inf_count'] = x_inf_count
                data_quality['feature_inf_percentage'] = inf_percentage
                
                if inf_percentage > 1:
                    error_msg = f"Critical: {inf_percentage:.2f}% of features are infinite"
                    validation_results['errors'].append(error_msg)
                    validation_results['is_valid'] = False
            
            if y_inf_count > 0:
                inf_percentage = (y_inf_count / y.size) * 100
                warning_msg = f"Found {y_inf_count} infinite values in targets ({inf_percentage:.2f}%)"
                validation_results['warnings'].append(warning_msg)
                data_quality['target_inf_count'] = y_inf_count
                data_quality['target_inf_percentage'] = inf_percentage
                
                if inf_percentage > 1:
                    error_msg = f"Critical: {inf_percentage:.2f}% of targets are infinite"
                    validation_results['errors'].append(error_msg)
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
        all_analyst_models_outputs: Optional[Dict[str, np.ndarray]] = None
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
            
        Returns:
            Dictionary containing training results and metadata with comprehensive reporting
        """
        try:
            self.logger.info("🚀 Starting Enhanced Tactician models training step")
            self.overall_start_time = time.time()
            
            # Phase 1: Data Validation
            self._start_phase(TrainingPhase.DATA_VALIDATION)
            try:
                validation_results = self._validate_input_data(X, y, regime_labels)
                self._complete_phase(TrainingPhase.DATA_VALIDATION, success=True, 
                                   samples_processed=X.shape[0], features_count=X.shape[1],
                                   warnings_issued=len(validation_results.get('warnings', [])),
                                   errors_encountered=len(validation_results.get('errors', [])))
            except Exception as e:
                self._complete_phase(TrainingPhase.DATA_VALIDATION, success=False, error_message=str(e))
                raise
            
            # Phase 2: Feature Preparation
            self._start_phase(TrainingPhase.FEATURE_PREPARATION)
            try:
                X, y, regime_labels, feature_names, preparation_metrics = self._prepare_features(
                    X, y, regime_labels, feature_names, hmm_states, 
                    analyst_signals, analyst_model_outputs, hmm_regime_features, 
                    all_analyst_models_outputs
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
        all_analyst_models_outputs: Optional[Dict[str, np.ndarray]]
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
            
            # Combine all features: base features + HMM regime features + all analyst model outputs
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
                    self.logger.info(f"📊 Total features: {X.shape[1]} (base + HMM + all analyst models)")
                    
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
        """Generate comprehensive training report."""
        try:
            self.logger.info("📊 " + "="*60)
            self.logger.info("📊 ENHANCED TACTICIAN TRAINING REPORT")
            self.logger.info("📊 " + "="*60)
            
            # Overall statistics
            self.logger.info(f"📊 Total training time: {total_time:.2f}s")
            self.logger.info(f"📊 Vectorization enabled: {self.enable_vectorization}")
            self.logger.info(f"📊 Vectorization fallback used: {self.vectorization_fallback_used}")
            
            # Phase-by-phase breakdown
            self.logger.info("📊 Phase breakdown:")
            for phase, metrics in self.training_metrics.items():
                status = "✅" if metrics.success else "❌"
                self.logger.info(f"📊   {status} {phase.value}: {metrics.duration:.2f}s")
                if not metrics.success and metrics.error_message:
                    self.logger.info(f"📊     Error: {metrics.error_message}")
            
            # Memory usage
            current_memory = self._get_memory_usage()
            self.logger.info(f"📊 Current memory usage: {current_memory:.1f} MB")
            
            self.logger.info("📊 " + "="*60)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate training report: {e}")
    
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
    config: Optional[PerRegimeTrainingConfig] = None,
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
    config: Optional[PerRegimeTrainingConfig] = None,
    feature_names: Optional[List[str]] = None,
    hmm_states: Optional[np.ndarray] = None,
    analyst_signals: Optional[np.ndarray] = None,
    analyst_model_outputs: Optional[np.ndarray] = None,
    hmm_regime_features: Optional[np.ndarray] = None,
    all_analyst_models_outputs: Optional[Dict[str, np.ndarray]] = None,
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
            all_analyst_models_outputs
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
    config = PerRegimeTrainingConfig(
        model_name="tactician_models",
        timeframe="1m",
        model_types=["NeuralObliviousDecisionEnsembles", "CatBoostRegressor", "LGBMRegressor", "Ridge"],
        hpo_n_trials=50,  # Reduced for demo
        enable_hpo=True,
        save_models=True,
        model_save_path="./models/tactician_models_enhanced"
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