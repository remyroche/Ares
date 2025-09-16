"""
Tactician Single Model Training Step - Enhanced for Entry Point Optimization

This step handles training of a single Tactician model (not per-regime) that operates on 1m timeframe
and is trained to find the best entry points after the Analyst gives its green light.

Key Features:
1. Only trains on periods where the Analyst gives a green light to start a trade
2. Includes the Analyst's outputs as input features for the Tactician's ML models
3. Uses optimal barrier settings for finding the best entry points
4. Single model for all regimes (not per-regime like Analyst)
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
from src.utils.ml_common.config import TacticianTrainingConfig
from src.utils.ml_common.training.base_training_step import BaseTrainingStep

logger = system_logger.getChild('TacticianSingleModelTraining')


class TrainingPhase(Enum):
    """Training phases for progress tracking."""
    INITIALIZATION = "initialization"
    DATA_VALIDATION = "data_validation"
    GREEN_LIGHT_FILTERING = "green_light_filtering"
    FEATURE_PREPARATION = "feature_preparation"
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


class TacticianSingleModelTrainingStep(BaseTrainingStep):
    """
    Enhanced Tactician Single Model Training Step for entry point optimization.
    
    The Tactician operates on 1m timeframe and is trained on:
    1. Only periods where the Analyst gives a green light
    2. Using the Analyst's model outputs as input features
    3. Single model for all regimes (not per-regime)
    4. Optimized barriers for finding best entry points
    """
    
    def __init__(self, config: Optional[TacticianTrainingConfig] = None):
        """
        Initialize enhanced Tactician single model training step.

        Args:
            config: Tactician training configuration
        """
        # Initialize training metrics tracking
        self.training_metrics: Dict[TrainingPhase, TrainingMetrics] = {}
        self.overall_start_time = time.time()
        
        # Set default configuration for tactician models
        if config is None:
            config = TacticianTrainingConfig(
                model_name="tactician_single_model",
                timeframe="1m",
                model_types=["NeuralObliviousDecisionEnsembles", "CatBoostRegressor", "LGBMRegressor", "ElasticNetCV"],
                hpo_n_trials=100,
                hpo_timeout_seconds=3600,
                min_samples_per_regime=1000,  # Not used in single model, but kept for compatibility
                enable_data_augmentation=True,
                augmentation_method="smote",
                model_save_path="./models/tactician_single_model",
                evaluation_metrics=["mse", "mae", "r2", "mape", "smape"],
                use_single_model=True,
                single_model_name="tactician_unified_model"
            )

        try:
            super().__init__(config)
            self.logger = logger.getChild('TacticianSingleModelTraining')
            
            # Initialize training metrics for initialization phase
            self._start_phase(TrainingPhase.INITIALIZATION)
            
            # Validate configuration
            self._validate_configuration(config)
            
            # Log initialization success
            self.logger.info("🚀 Enhanced Tactician Single Model Training Step initialized")
            
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
        
        # Log phase start
        self.logger.info(f"🔄 Starting {phase.value} phase")
    
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
            
            # Log phase completion
            status = "✅" if success else "❌"
            self.logger.info(f"{status} Completed {phase.value} phase in {duration:.2f}s")
            
            if not success and error_message:
                self.logger.error(f"❌ Phase failed: {error_message}")
    
    def _validate_configuration(self, config: TacticianTrainingConfig) -> None:
        """Validate training configuration."""
        try:
            # Validate model types
            if not config.model_types:
                raise ValueError("No model types specified in configuration")
            
            # Validate timeframe
            if not config.timeframe:
                raise ValueError("No timeframe specified in configuration")
            
            # Validate single model configuration
            if not config.use_single_model:
                self.logger.warning("⚠️ Single model training is disabled - this may not be optimal for Tactician")
            
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
        Execute enhanced Tactician single model training step.
        
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
            Dictionary containing training results and metadata
        """
        try:
            self.logger.info("🚀 Starting Enhanced Tactician single model training step")
            self.overall_start_time = time.time()
            
            # Phase 1: Data Validation
            self._start_phase(TrainingPhase.DATA_VALIDATION)
            try:
                validation_results = self._validate_input_data(X, y, regime_labels)
                self._complete_phase(TrainingPhase.DATA_VALIDATION, success=True, 
                                   samples_processed=X.shape[0], features_count=X.shape[1])
            except Exception as e:
                self._complete_phase(TrainingPhase.DATA_VALIDATION, success=False, error_message=str(e))
                raise
            
            # Phase 2: Green Light Filtering
            self._start_phase(TrainingPhase.GREEN_LIGHT_FILTERING)
            try:
                X_filtered, y_filtered, regime_labels_filtered, filtering_metrics = self._filter_green_light_samples(
                    X, y, regime_labels, analyst_signals
                )
                self._complete_phase(TrainingPhase.GREEN_LIGHT_FILTERING, success=True,
                                   samples_processed=X_filtered.shape[0])
            except Exception as e:
                self._complete_phase(TrainingPhase.GREEN_LIGHT_FILTERING, success=False, error_message=str(e))
                raise
            
            # Phase 3: Feature Preparation
            self._start_phase(TrainingPhase.FEATURE_PREPARATION)
            try:
                X_combined, feature_names_combined, preparation_metrics = self._prepare_combined_features(
                    X_filtered, feature_names, hmm_regime_features, 
                    analyst_model_outputs, all_analyst_models_outputs
                )
                self._complete_phase(TrainingPhase.FEATURE_PREPARATION, success=True,
                                   features_count=X_combined.shape[1])
            except Exception as e:
                self._complete_phase(TrainingPhase.FEATURE_PREPARATION, success=False, error_message=str(e))
                raise
            
            # Phase 4: Single Model Training
            self._start_phase(TrainingPhase.MODEL_TRAINING)
            try:
                results = self._train_single_model(
                    X_combined, y_filtered, feature_names_combined
                )
                self._complete_phase(TrainingPhase.MODEL_TRAINING, success=True,
                                   models_trained=len(results.get('models', {})))
            except Exception as e:
                self._complete_phase(TrainingPhase.MODEL_TRAINING, success=False, error_message=str(e))
                raise
            
            # Phase 5: Finalization
            self._start_phase(TrainingPhase.FINALIZATION)
            try:
                results = self._finalize_results(results, filtering_metrics, preparation_metrics)
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
    
    def _validate_input_data(self, X: np.ndarray, y: np.ndarray, 
                           regime_labels: np.ndarray) -> Dict[str, Any]:
        """Comprehensive input data validation."""
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'errors': []
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
            
            # Log validation results
            self.logger.info(f"📊 Data validation: {X.shape[0]} samples, {X.shape[1]} features")
            
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
    
    def _filter_green_light_samples(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regime_labels: np.ndarray,
        analyst_signals: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """Filter data to only include periods where Analyst gives green light."""
        filtering_metrics = {
            'original_samples': X.shape[0],
            'green_light_samples': 0,
            'green_light_rate': 0.0,
            'filtered_samples': 0
        }
        
        try:
            if analyst_signals is None:
                self.logger.warning("⚠️ No analyst signals provided - training on all samples")
                return X, y, regime_labels, filtering_metrics
            
            # Create green light mask
            green_light_mask = analyst_signals == 1
            green_light_count = np.sum(green_light_mask)
            green_light_rate = green_light_count / len(analyst_signals)
            
            filtering_metrics.update({
                'green_light_samples': green_light_count,
                'green_light_rate': green_light_rate,
                'filtered_samples': green_light_count
            })
            
            self.logger.info(f"📊 Filtering to {green_light_count} samples with Analyst green light signals ({green_light_rate:.2%})")
            
            # Validate green light filtering results
            if green_light_count == 0:
                raise ValueError("No samples with Analyst green light signals found")
            
            if green_light_rate < 0.01:  # Less than 1%
                self.logger.warning(f"⚠️ Very low green light rate: {green_light_rate:.2%}")
            
            # Apply filtering
            X_filtered = X[green_light_mask]
            y_filtered = y[green_light_mask]
            regime_labels_filtered = regime_labels[green_light_mask]
            
            # Validate filtered data shapes
            if X_filtered.shape[0] != green_light_count:
                raise ValueError(f"Filtered data shape mismatch: expected {green_light_count}, got {X_filtered.shape[0]}")
            
            self.logger.info(f"✅ Green light filtering completed: {X_filtered.shape[0]} samples")
            
            return X_filtered, y_filtered, regime_labels_filtered, filtering_metrics
            
        except Exception as e:
            self.logger.error(f"❌ Green light filtering failed: {e}")
            raise
    
    def _prepare_combined_features(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]],
        hmm_regime_features: Optional[np.ndarray],
        analyst_model_outputs: Optional[np.ndarray],
        all_analyst_models_outputs: Optional[Dict[str, np.ndarray]]
    ) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
        """Prepare and combine all features including Analyst outputs."""
        preparation_metrics = {
            'base_features': X.shape[1],
            'hmm_features': 0,
            'analyst_features': 0,
            'total_features': 0,
            'feature_combinations': {}
        }
        
        try:
            # Start with base features
            additional_features = []
            additional_feature_names = []
            
            # Add HMM regime features if provided
            if hmm_regime_features is not None:
                # Apply same filtering as base features (should already be filtered)
                if hmm_regime_features.shape[0] != X.shape[0]:
                    raise ValueError(f"HMM regime features shape mismatch: expected {X.shape[0]}, got {hmm_regime_features.shape[0]}")
                
                additional_features.append(hmm_regime_features)
                additional_feature_names.extend([f"hmm_regime_{i}" for i in range(hmm_regime_features.shape[1])])
                preparation_metrics['hmm_features'] = hmm_regime_features.shape[1]
                
                self.logger.info(f"📊 Added {hmm_regime_features.shape[1]} HMM regime features")
            
            # Add all individual analyst model outputs if provided
            if all_analyst_models_outputs is not None:
                analyst_features_added = 0
                for model_name, model_outputs in all_analyst_models_outputs.items():
                    # Apply same filtering as base features (should already be filtered)
                    if model_outputs.shape[0] != X.shape[0]:
                        raise ValueError(f"Analyst model {model_name} output shape mismatch: expected {X.shape[0]}, got {model_outputs.shape[0]}")
                    
                    additional_features.append(model_outputs)
                    additional_feature_names.extend([f"analyst_{model_name}_{i}" for i in range(model_outputs.shape[1])])
                    analyst_features_added += model_outputs.shape[1]
                
                preparation_metrics['analyst_features'] = analyst_features_added
                self.logger.info(f"📊 Added outputs from {len(all_analyst_models_outputs)} analyst models ({analyst_features_added} features)")
            
            # Add legacy analyst model outputs for backward compatibility
            if analyst_model_outputs is not None:
                if analyst_model_outputs.shape[0] != X.shape[0]:
                    raise ValueError(f"Legacy analyst outputs shape mismatch: expected {X.shape[0]}, got {analyst_model_outputs.shape[0]}")
                
                additional_features.append(analyst_model_outputs)
                additional_feature_names.extend([f"analyst_legacy_{i}" for i in range(analyst_model_outputs.shape[1])])
                preparation_metrics['analyst_features'] += analyst_model_outputs.shape[1]
                
                self.logger.info(f"📊 Added {analyst_model_outputs.shape[1]} legacy analyst outputs")
            
            # Combine all features
            if additional_features:
                X_combined = np.column_stack([X] + additional_features)
                
                # Update feature names
                if feature_names is not None:
                    feature_names_combined = feature_names + additional_feature_names
                else:
                    feature_names_combined = [f"feature_{i}" for i in range(X_combined.shape[1])]
                
                preparation_metrics['total_features'] = X_combined.shape[1]
                self.logger.info(f"📊 Total features: {X_combined.shape[1]} (base + HMM + analyst models)")
            else:
                X_combined = X
                feature_names_combined = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
                preparation_metrics['total_features'] = X.shape[1]
                self.logger.info(f"📊 Using base features only: {X.shape[1]} features")
            
            return X_combined, feature_names_combined, preparation_metrics
            
        except Exception as e:
            self.logger.error(f"❌ Feature preparation failed: {e}")
            raise
    
    def _train_single_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Train a single model for all regimes (not per-regime)."""
        try:
            self.logger.info("🔄 Training single Tactician model for all regimes...")
            
            # Use the first model type as the primary model (can be extended to ensemble)
            primary_model_type = self.config.model_types[0]
            
            # Train the model
            if self.config.enable_hpo:
                search_space = self.config.hpo_search_spaces.get(primary_model_type, {})
                trained_model = self.training_utils.optimize_model_with_hpo(
                    model_type=primary_model_type,
                    X=X,
                    y=y,
                    search_space=search_space,
                    model_name=self.config.single_model_name
                )
            else:
                trained_model = self.training_utils.train_single_model(
                    model_type=primary_model_type,
                    X=X,
                    y=y,
                    model_name=self.config.single_model_name
                )
            
            # Evaluate the model
            evaluation_results = self.training_utils.evaluate_model(
                model=trained_model,
                X=X,
                y=y,
                metrics=self.config.evaluation_metrics
            )
            
            results = {
                'models': {
                    self.config.single_model_name: trained_model
                },
                'evaluation_results': {
                    'single_model': evaluation_results
                },
                'training_method': 'single_model',
                'model_type': primary_model_type,
                'samples_trained': X.shape[0],
                'features_used': X.shape[1]
            }
            
            self.logger.info(f"✅ Single model training completed: {primary_model_type}")
            self.logger.info(f"📊 Model performance: {evaluation_results}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Single model training failed: {e}")
            raise
    
    def _finalize_results(
        self, 
        results: Dict[str, Any], 
        filtering_metrics: Dict[str, Any],
        preparation_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Finalize results with tactician-specific metadata."""
        try:
            # Add tactician-specific metadata
            results['tactician_metadata'] = {
                'training_type': 'single_model',
                'timeframe': self.config.timeframe,
                'green_light_filtering': filtering_metrics,
                'feature_preparation': preparation_metrics,
                'model_config': {
                    'model_types': self.config.model_types,
                    'use_single_model': self.config.use_single_model,
                    'single_model_name': self.config.single_model_name
                }
            }
            
            # Add training metrics
            results['training_metrics'] = {
                phase.value: {
                    'duration': metrics.duration,
                    'success': metrics.success,
                    'samples_processed': metrics.samples_processed,
                    'features_count': metrics.features_count,
                    'models_trained': metrics.models_trained,
                    'error_message': metrics.error_message
                }
                for phase, metrics in self.training_metrics.items()
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
            }
        }
    
    def _generate_training_report(self, total_time: float) -> None:
        """Generate comprehensive training report."""
        try:
            self.logger.info("📊 " + "="*80)
            self.logger.info("📊 ENHANCED TACTICIAN SINGLE MODEL TRAINING REPORT")
            self.logger.info("📊 " + "="*80)
            
            # Overall statistics
            self.logger.info(f"📊 Total training time: {total_time:.2f}s")
            self.logger.info(f"📊 Training type: Single model (not per-regime)")
            self.logger.info(f"📊 Timeframe: {self.config.timeframe}")
            
            # Phase-by-phase breakdown
            self.logger.info("📊 " + "-"*60)
            self.logger.info("📊 PHASE BREAKDOWN:")
            self.logger.info("📊 " + "-"*60)
            
            for phase, metrics in self.training_metrics.items():
                status = "✅" if metrics.success else "❌"
                self.logger.info(f"📊   {status} {phase.value.upper()}: {metrics.duration:.2f}s")
                
                if metrics.samples_processed > 0:
                    self.logger.info(f"📊     └─ Samples: {metrics.samples_processed:,}")
                if metrics.features_count > 0:
                    self.logger.info(f"📊     └─ Features: {metrics.features_count}")
                if metrics.models_trained > 0:
                    self.logger.info(f"📊     └─ Models trained: {metrics.models_trained}")
                
                if not metrics.success and metrics.error_message:
                    self.logger.info(f"📊     └─ ❌ Error: {metrics.error_message}")
            
            self.logger.info("📊 " + "="*80)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate training report: {e}")


# Convenience functions
def create_tactician_single_model_training_step(
    config: Optional[TacticianTrainingConfig] = None
) -> TacticianSingleModelTrainingStep:
    """Create enhanced Tactician single model training step."""
    try:
        return TacticianSingleModelTrainingStep(config)
    except Exception as e:
        logger.error(f"❌ Failed to create tactician training step: {e}")
        raise


def execute_tactician_single_model_training(
    X: np.ndarray,
    y: np.ndarray,
    regime_labels: np.ndarray,
    config: Optional[TacticianTrainingConfig] = None,
    feature_names: Optional[List[str]] = None,
    hmm_states: Optional[np.ndarray] = None,
    analyst_signals: Optional[np.ndarray] = None,
    analyst_model_outputs: Optional[np.ndarray] = None,
    hmm_regime_features: Optional[np.ndarray] = None,
    all_analyst_models_outputs: Optional[Dict[str, np.ndarray]] = None
) -> Dict[str, Any]:
    """Execute enhanced Tactician single model training step."""
    try:
        step = create_tactician_single_model_training_step(config)
        return step.execute(
            X, y, regime_labels, feature_names, hmm_states, 
            analyst_signals, analyst_model_outputs, hmm_regime_features, 
            all_analyst_models_outputs
        )
    except Exception as e:
        logger.error(f"❌ Failed to execute tactician training: {e}")
        raise


if __name__ == "__main__":
    # Example usage
    print("Enhanced Tactician Single Model Training Step")
    print("=" * 50)
    
    # Create configuration
    config = TacticianTrainingConfig(
        model_name="tactician_single_model",
        timeframe="1m",
        model_types=["NeuralObliviousDecisionEnsembles", "CatBoostRegressor"],
        hpo_n_trials=50,
        enable_hpo=True,
        save_models=True,
        model_save_path="./models/tactician_single_model",
        use_single_model=True,
        single_model_name="tactician_unified_model"
    )
    
    # Create training step
    try:
        training_step = create_tactician_single_model_training_step(config)
        print(f"✅ Created enhanced tactician single model training step")
        print(f"📊 Model types: {config.model_types}")
        print(f"📊 Single model: {config.use_single_model}")
        print(f"📊 Timeframe: {config.timeframe}")
        
        print("\n🎯 Enhanced Tactician Features:")
        print("- Only trains on Analyst green light periods")
        print("- Includes Analyst outputs as features")
        print("- Single model for all regimes")
        print("- Optimized for entry point finding")
        
    except Exception as e:
        print(f"❌ Failed to create training step: {e}")