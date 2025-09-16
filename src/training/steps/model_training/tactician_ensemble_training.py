"""
Tactician Ensemble Training Step

This step handles all-regime ensemble training of Tactician models using common dependencies.
The Tactician Ensemble operates on 1m timeframe and combines individual tactician models
with all previous model inputs (HMM, Analyst) to create the final meta-learner for timing decisions.

Enhanced with vectorized training capabilities for improved performance.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import time
import traceback
from dataclasses import dataclass

from src.utils.logger import system_logger
from src.utils.ml_common.config.base_training_config import EnsembleTrainingConfig
from src.utils.ml_common.training.ensemble_training_step import EnsembleTrainingStep

# Import vectorized training manager
try:
    from src.utils.ml_common.training.vectorized_training_manager import VectorizedTrainingManager
    VECTORIZED_TRAINING_AVAILABLE = True
except ImportError:
    VECTORIZED_TRAINING_AVAILABLE = False

logger = system_logger.getChild('TacticianEnsembleTraining')


@dataclass
class TrainingProgress:
    """Track training progress and metrics."""
    step_name: str
    start_time: float
    end_time: Optional[float] = None
    success: bool = False
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}
    
    @property
    def duration(self) -> float:
        """Get training duration in seconds."""
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time
    
    def complete(self, success: bool = True, error_message: Optional[str] = None, metrics: Optional[Dict[str, Any]] = None):
        """Mark step as complete."""
        self.end_time = time.time()
        self.success = success
        self.error_message = error_message
        if metrics:
            self.metrics.update(metrics)


class TacticianEnsembleTrainingStep(EnsembleTrainingStep):
    """
    Tactician Ensemble Training Step with all-regime ensemble training, HPO, saving, and metrics.
    
    The Tactician Ensemble operates on 1m timeframe and combines individual tactician models
    with all previous model inputs (HMM, Analyst) to create the final meta-learner for timing decisions.
    """
    
    def __init__(self, config: Optional[EnsembleTrainingConfig] = None, enable_vectorization: bool = True):
        """
        Initialize Tactician ensemble training step with vectorization support.

        Args:
            config: Per-regime training configuration
            enable_vectorization: Whether to enable vectorized training
        """
        try:
            # Set default configuration for tactician ensemble models
            if config is None:
                config = EnsembleTrainingConfig(
                    model_name="tactician_ensemble_models",
                    timeframe="1m",
                    model_types=["XGBOOST_CUSTOM"],
                    hpo_n_trials=100,
                    hpo_timeout_seconds=3600,
                    min_samples_per_regime=1000,
                    enable_data_augmentation=True,
                    augmentation_method="smote",
                    model_save_path="./models/tactician_ensemble_models",
                    evaluation_metrics=["mse", "mae", "r2", "mape", "smape"]
                )

            # Validate configuration
            self._validate_config(config)
            
            super().__init__(config, enable_vectorization=enable_vectorization and VECTORIZED_TRAINING_AVAILABLE)
            self.logger = logger.getChild('TacticianEnsembleTrainingStep')
            
            # Initialize progress tracking
            self.progress_tracker: List[TrainingProgress] = []
            self.current_step: Optional[TrainingProgress] = None

            if self.enable_vectorization:
                self.logger.info("🚀 Tactician Ensemble Training Step initialized with vectorization")
            else:
                self.logger.info("✅ Tactician Ensemble Training Step initialized (standard mode)")
                
        except Exception as e:
            error_msg = f"Failed to initialize TacticianEnsembleTrainingStep: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(error_msg) from e
    
    def _validate_config(self, config: EnsembleTrainingConfig) -> None:
        """Validate configuration parameters."""
        validation_errors = []
        
        if not config.model_name or not isinstance(config.model_name, str):
            validation_errors.append("model_name must be a non-empty string")
        
        if not config.timeframe or not isinstance(config.timeframe, str):
            validation_errors.append("timeframe must be a non-empty string")
            
        if not config.model_types or not isinstance(config.model_types, list) or len(config.model_types) == 0:
            validation_errors.append("model_types must be a non-empty list")
            
        if config.hpo_n_trials <= 0:
            validation_errors.append("hpo_n_trials must be positive")
            
        if config.min_samples_per_regime <= 0:
            validation_errors.append("min_samples_per_regime must be positive")
            
        if validation_errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(validation_errors)}")
    
    def _start_step(self, step_name: str) -> TrainingProgress:
        """Start tracking a training step."""
        progress = TrainingProgress(step_name=step_name, start_time=time.time())
        self.progress_tracker.append(progress)
        self.current_step = progress
        self.logger.info(f"🔄 Starting step: {step_name}")
        return progress
    
    def _complete_step(self, success: bool = True, error_message: Optional[str] = None, metrics: Optional[Dict[str, Any]] = None):
        """Complete the current training step."""
        if self.current_step:
            self.current_step.complete(success, error_message, metrics)
            if success:
                self.logger.info(f"✅ Completed step: {self.current_step.step_name} in {self.current_step.duration:.2f}s")
            else:
                self.logger.error(f"❌ Failed step: {self.current_step.step_name} - {error_message}")
            self.current_step = None
    
    def execute(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regime_labels: np.ndarray,
        feature_names: Optional[List[str]] = None,
        hmm_states: Optional[np.ndarray] = None,
        base_tactician_models: Optional[Dict[str, Any]] = None,
        tactician_training_metrics: Optional[Dict[str, Any]] = None,
        analyst_models: Optional[Dict[str, Any]] = None,
        analyst_ensembles: Optional[Dict[str, Any]] = None,
        analyst_ensemble_metrics: Optional[Dict[str, Any]] = None,
        hmm_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute Tactician ensemble training step with comprehensive error handling and progress tracking.
        
        Args:
            X: Input features (1m timeframe with cross-timeframe features)
            y: Target values (tactician outputs - timing decisions)
            regime_labels: Regime labels for each sample
            feature_names: Names of input features
            hmm_states: HMM cluster/regime states
            base_tactician_models: Individual tactician models to ensemble
            tactician_training_metrics: Performance metrics of base tactician models
            analyst_models: Individual analyst models
            analyst_ensembles: Analyst ensemble models
            analyst_ensemble_metrics: Performance metrics of analyst ensembles
            hmm_data: HMM regime data and features
            
        Returns:
            Dictionary containing training results and metadata
        """
        overall_start_time = time.time()
        self.logger.info("🚀 Starting Tactician ensemble training step (meta-learner)")
        
        try:
            # Step 1: Input validation
            self._start_step("Input Validation")
            self._validate_inputs(X, y, regime_labels, feature_names)
            self._complete_step(True, metrics={'samples': len(X), 'features': X.shape[1]})
            
            # Step 2: Base model validation and preparation
            self._start_step("Base Model Preparation")
            base_tactician_models = self._prepare_base_models(base_tactician_models)
            self._complete_step(True, metrics={'base_models_count': len(base_tactician_models)})
            
            # Step 3: Feature enhancement
            self._start_step("Feature Enhancement")
            X_enhanced = self._combine_all_model_inputs(
                X, analyst_models, analyst_ensembles, hmm_data, feature_names
            )
            enhancement_metrics = {
                'original_features': X.shape[1],
                'enhanced_features': X_enhanced.shape[1],
                'feature_increase': X_enhanced.shape[1] - X.shape[1]
            }
            self._complete_step(True, metrics=enhancement_metrics)
            
            # Step 4: Ensemble training
            self._start_step("Ensemble Training")
            results = super().execute(
                X=X_enhanced,
                y=y,
                regime_labels=regime_labels,
                feature_names=feature_names,
                hmm_states=hmm_states,
                is_classification=False,  # Tactician ensemble models are typically regression
                symbol=None,  # Can be passed as kwargs
                exchange=None,
                timeframe=self.config.timeframe
            )
            
            if 'error' in results:
                self._complete_step(False, f"Parent training failed: {results['error']}")
                return self._create_error_result("Ensemble training failed", results['error'])
            
            training_metrics = {
                'regimes_trained': len(results.get('models', {})),
                'training_time': results.get('training_time', 0)
            }
            self._complete_step(True, metrics=training_metrics)
            
            # Step 5: Meta-learner metadata enhancement
            self._start_step("Meta-learner Enhancement")
            results = self._add_meta_learner_metadata(
                results, base_tactician_models, tactician_training_metrics,
                analyst_models, analyst_ensembles, analyst_ensemble_metrics, hmm_data
            )
            self._complete_step(True)
            
            # Step 6: Final reporting
            self._start_step("Final Reporting")
            results = self._add_comprehensive_reporting(results, overall_start_time)
            self._complete_step(True)
            
            return results
            
        except Exception as e:
            error_msg = f"Tactician ensemble training failed: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            
            if self.current_step:
                self._complete_step(False, error_msg)
            
            return self._create_error_result("Training execution failed", error_msg)
    
    def _validate_inputs(self, X: np.ndarray, y: np.ndarray, regime_labels: np.ndarray, feature_names: Optional[List[str]]) -> None:
        """Validate input data with comprehensive checks."""
        validation_errors = []
        
        # Check data types and shapes
        if not isinstance(X, np.ndarray):
            validation_errors.append("X must be a numpy array")
        elif X.ndim != 2:
            validation_errors.append("X must be a 2D array")
        elif X.shape[0] == 0:
            validation_errors.append("X cannot be empty")
            
        if not isinstance(y, np.ndarray):
            validation_errors.append("y must be a numpy array")
        elif y.ndim != 1:
            validation_errors.append("y must be a 1D array")
        elif y.shape[0] == 0:
            validation_errors.append("y cannot be empty")
            
        if not isinstance(regime_labels, np.ndarray):
            validation_errors.append("regime_labels must be a numpy array")
        elif regime_labels.ndim != 1:
            validation_errors.append("regime_labels must be a 1D array")
            
        # Check shape consistency
        if isinstance(X, np.ndarray) and isinstance(y, np.ndarray) and X.shape[0] != y.shape[0]:
            validation_errors.append(f"X and y must have same number of samples: {X.shape[0]} vs {y.shape[0]}")
            
        if isinstance(y, np.ndarray) and isinstance(regime_labels, np.ndarray) and y.shape[0] != regime_labels.shape[0]:
            validation_errors.append(f"y and regime_labels must have same number of samples: {y.shape[0]} vs {regime_labels.shape[0]}")
            
        # Check for NaN or infinite values
        if isinstance(X, np.ndarray):
            if np.any(np.isnan(X)):
                validation_errors.append("X contains NaN values")
            if np.any(np.isinf(X)):
                validation_errors.append("X contains infinite values")
                
        if isinstance(y, np.ndarray):
            if np.any(np.isnan(y)):
                validation_errors.append("y contains NaN values")
            if np.any(np.isinf(y)):
                validation_errors.append("y contains infinite values")
        
        # Check feature names consistency
        if feature_names is not None and isinstance(X, np.ndarray):
            if len(feature_names) != X.shape[1]:
                validation_errors.append(f"feature_names length ({len(feature_names)}) must match X features ({X.shape[1]})")
        
        if validation_errors:
            raise ValueError(f"Input validation failed: {'; '.join(validation_errors)}")
    
    def _prepare_base_models(self, base_tactician_models: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare and validate base tactician models."""
        if base_tactician_models is None or not base_tactician_models:
            self.logger.info("📊 No base tactician models provided, creating from configuration...")
            base_tactician_models = self._create_base_models_from_config()
        
        # Validate base models
        valid_models = {}
        for name, model in base_tactician_models.items():
            if model is not None:
                valid_models[name] = model
            else:
                self.logger.warning(f"⚠️ Base model '{name}' is None, skipping")
        
        if not valid_models:
            self.logger.error("❌ No valid base models found. All provided models are None.")
            raise ValueError("At least one valid base tactician model is required for ensemble training.")
        
        self.logger.info(f"✅ Using {len(valid_models)} base tactician models: {list(valid_models.keys())}")
        return valid_models
    
    def _create_base_models_from_config(self) -> Dict[str, Any]:
        """Create base tactician models from configuration."""
        try:
            from src.utils.ml_common.models.model_factory import create_tactician_models
            
            self.logger.info("🏭 Creating tactician models from configuration...")
            models = create_tactician_models()
            
            if not models:
                raise ValueError("Failed to create any tactician models from configuration")
            
            self.logger.info(f"✅ Created {len(models)} tactician models: {list(models.keys())}")
            return models
            
        except ImportError as e:
            self.logger.error(f"❌ Failed to import model factory: {e}")
            raise RuntimeError("Cannot create tactician models: model factory not available") from e
        except Exception as e:
            self.logger.error(f"❌ Failed to create tactician models from configuration: {e}")
            raise RuntimeError(f"Tactician model creation failed: {e}") from e
    
    def _create_error_result(self, error_type: str, error_message: str) -> Dict[str, Any]:
        """Create standardized error result."""
        return {
            'error': error_type,
            'error_message': error_message,
            'success': False,
            'training_time': 0,
            'progress_tracker': [progress.__dict__ for progress in self.progress_tracker]
        }
    
    
    def _combine_all_model_inputs(
        self,
        X: np.ndarray,
        analyst_models: Optional[Dict[str, Any]],
        analyst_ensembles: Optional[Dict[str, Any]],
        hmm_data: Optional[Dict[str, Any]],
        feature_names: Optional[List[str]]
    ) -> np.ndarray:
        """
        Combine all model inputs for meta-learner training with comprehensive error handling.
        
        Args:
            X: Base features
            analyst_models: Individual analyst models
            analyst_ensembles: Analyst ensemble models
            hmm_data: HMM regime data
            feature_names: Feature names for tracking
            
        Returns:
            Enhanced feature matrix with all model inputs
        """
        try:
            enhanced_features = [X]
            feature_count = X.shape[1]
            integration_stats = {
                'hmm_features_added': 0,
                'analyst_models_integrated': 0,
                'analyst_ensembles_integrated': 0,
                'integration_errors': []
            }
            
            # Add HMM regime features if available
            if hmm_data and 'regime_features' in hmm_data:
                try:
                    hmm_features = hmm_data['regime_features']
                    if isinstance(hmm_features, np.ndarray) and hmm_features.shape[0] == X.shape[0]:
                        enhanced_features.append(hmm_features)
                        feature_count += hmm_features.shape[1]
                        integration_stats['hmm_features_added'] = hmm_features.shape[1]
                        self.logger.info(f"📊 Added {hmm_features.shape[1]} HMM regime features")
                    else:
                        self.logger.warning("⚠️ HMM features shape mismatch or invalid format")
                        integration_stats['integration_errors'].append("HMM features shape mismatch")
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to integrate HMM features: {e}")
                    integration_stats['integration_errors'].append(f"HMM integration failed: {e}")
            
            # Add analyst model predictions if available
            if analyst_models:
                for model_name, model in analyst_models.items():
                    try:
                        predictions = self._generate_model_predictions(model, X, model_name)
                        if predictions is not None:
                            enhanced_features.append(predictions)
                            feature_count += predictions.shape[1]
                            integration_stats['analyst_models_integrated'] += 1
                            self.logger.info(f"📊 Added predictions from analyst model: {model_name}")
                        else:
                            integration_stats['integration_errors'].append(f"Failed to generate predictions for {model_name}")
                    except Exception as e:
                        self.logger.warning(f"⚠️ Could not add predictions from {model_name}: {e}")
                        integration_stats['integration_errors'].append(f"Analyst model {model_name} failed: {e}")
            
            # Add analyst ensemble predictions if available
            if analyst_ensembles:
                for ensemble_name, ensemble in analyst_ensembles.items():
                    try:
                        predictions = self._generate_model_predictions(ensemble, X, ensemble_name)
                        if predictions is not None:
                            enhanced_features.append(predictions)
                            feature_count += predictions.shape[1]
                            integration_stats['analyst_ensembles_integrated'] += 1
                            self.logger.info(f"📊 Added predictions from analyst ensemble: {ensemble_name}")
                        else:
                            integration_stats['integration_errors'].append(f"Failed to generate predictions for {ensemble_name}")
                    except Exception as e:
                        self.logger.warning(f"⚠️ Could not add predictions from {ensemble_name}: {e}")
                        integration_stats['integration_errors'].append(f"Analyst ensemble {ensemble_name} failed: {e}")
            
            # Combine all features
            if len(enhanced_features) > 1:
                X_enhanced = np.column_stack(enhanced_features)
                self.logger.info(f"📊 Meta-learner features: {X.shape[1]} base + {feature_count - X.shape[1]} model inputs = {feature_count} total")
            else:
                X_enhanced = X
                self.logger.info(f"📊 Using base features only: {X.shape[1]} features")
            
            # Log integration summary
            self.logger.info(f"📊 Integration summary: {integration_stats}")
            
            return X_enhanced
            
        except Exception as e:
            self.logger.error(f"Failed to combine model inputs: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            # Return original features if combination fails
            self.logger.warning("⚠️ Returning original features due to combination failure")
            return X
    
    def _generate_model_predictions(self, model: Any, X: np.ndarray, model_name: str) -> Optional[np.ndarray]:
        """Generate predictions from a model with proper error handling."""
        try:
            # Check if model has predict method
            if not hasattr(model, 'predict'):
                self.logger.warning(f"⚠️ Model {model_name} does not have predict method")
                return None
            
            # Generate predictions
            predictions = model.predict(X)
            
            # Ensure predictions are 2D
            if predictions.ndim == 1:
                predictions = predictions.reshape(-1, 1)
            
            # Validate predictions
            if predictions.shape[0] != X.shape[0]:
                self.logger.warning(f"⚠️ Model {model_name} predictions shape mismatch: {predictions.shape[0]} vs {X.shape[0]}")
                return None
            
            # Check for NaN or infinite values
            if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
                self.logger.warning(f"⚠️ Model {model_name} produced invalid predictions (NaN/Inf)")
                return None
            
            return predictions
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to generate predictions from {model_name}: {e}")
            return None
    
    def _add_meta_learner_metadata(
        self,
        results: Dict[str, Any],
        base_models: Dict[str, Any],
        tactician_metrics: Optional[Dict[str, Any]],
        analyst_models: Optional[Dict[str, Any]],
        analyst_ensembles: Optional[Dict[str, Any]],
        analyst_metrics: Optional[Dict[str, Any]],
        hmm_data: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Add meta-learner specific metadata to results.
        
        Args:
            results: Training results
            base_models: Base tactician models used in ensemble
            tactician_metrics: Performance metrics of base tactician models
            analyst_models: Individual analyst models
            analyst_ensembles: Analyst ensemble models
            analyst_metrics: Performance metrics of analyst ensembles
            hmm_data: HMM regime data
            
        Returns:
            Enhanced results with meta-learner specific metadata
        """
        # Add meta-learner specific analysis
        if 'regime_analysis' in results:
            regime_analysis = results['regime_analysis']
            
            # Calculate meta-learner specific metrics
            meta_learner_metrics = {
                'total_regimes': len(regime_analysis.get('unique_regimes', [])),
                'sufficient_regimes': len(regime_analysis.get('sufficient_regimes', [])),
                'insufficient_regimes': len(regime_analysis.get('insufficient_regimes', [])),
                'regime_balance': regime_analysis.get('regime_balance_train', 0.0),
                'timeframe': self.config.timeframe,
                'ensemble_model_types': self.config.model_types,
                'base_tactician_models_count': len(base_models) if base_models else 0,
                'analyst_models_integrated': len(analyst_models) if analyst_models else 0,
                'analyst_ensembles_integrated': len(analyst_ensembles) if analyst_ensembles else 0,
                'hmm_data_integrated': bool(hmm_data)
            }
            
            # Add performance metrics from all integrated models
            integrated_metrics = {}
            if tactician_metrics:
                integrated_metrics['tactician_models'] = tactician_metrics
            if analyst_metrics:
                integrated_metrics['analyst_ensembles'] = analyst_metrics
            if hmm_data and 'metrics' in hmm_data:
                integrated_metrics['hmm_models'] = hmm_data['metrics']
            
            if integrated_metrics:
                meta_learner_metrics['integrated_model_performance'] = integrated_metrics
                self.logger.info("📊 Integrated performance metrics from all model types")
            
            results['meta_learner_metrics'] = meta_learner_metrics
        
        # Add meta-learner performance summary
        if 'evaluation_results' in results:
            evaluation_results = results['evaluation_results']
            
            # Calculate best performing meta-learner per regime
            best_meta_learners = {}
            for regime, regime_metrics in evaluation_results.items():
                if isinstance(regime_metrics, dict) and 'error' not in regime_metrics:
                    best_meta_learner = None
                    best_r2 = -np.inf
                    
                    for meta_learner_name, metrics in regime_metrics.items():
                        if isinstance(metrics, dict) and 'r2' in metrics:
                            if metrics['r2'] > best_r2:
                                best_r2 = metrics['r2']
                                best_meta_learner = meta_learner_name
                    
                    if best_meta_learner:
                        best_meta_learners[regime] = {
                            'meta_learner': best_meta_learner,
                            'r2_score': best_r2
                        }
            
            results['best_meta_learners_per_regime'] = best_meta_learners
        
        # Add meta-learner specific analysis
        meta_learner_analysis = {
            'base_timeframe': self.config.timeframe,
            'cross_timeframe_features': True,
            'ensemble_method': 'all_regime_meta_learner',
            'tactician_models_integrated': len(base_models) if base_models else 0,
            'analyst_models_integrated': len(analyst_models) if analyst_models else 0,
            'analyst_ensembles_integrated': len(analyst_ensembles) if analyst_ensembles else 0,
            'hmm_data_integrated': bool(hmm_data),
            'meta_learner_role': 'final_timing_decision',
            'comprehensive_intelligence': True
        }
        results['meta_learner_analysis'] = meta_learner_analysis
        
        # Add proper artifact formatting for ensemble training
        tactician_ensembles = []
        ensemble_metrics = {}
        tactician_ensemble_performance = {}
        
        # Extract ensemble models from results
        if 'models' in results:
            for regime_id, regime_models in results['models'].items():
                if isinstance(regime_models, dict):
                    for model_name, model_data in regime_models.items():
                        if 'error' not in model_data and model_data.get('model') is not None:
                            tactician_ensembles.append({
                                'regime_id': regime_id,
                                'model_name': model_name,
                                'model_type': model_name,
                                'model_object': model_data.get('model'),
                                'hyperparameters': model_data.get('hyperparameters', {})
                            })
                            
                            # Add ensemble metrics
                            ensemble_metrics[f"{regime_id}_{model_name}"] = {
                                'regime_id': regime_id,
                                'model_name': model_name,
                                'training_time': model_data.get('training_time', 0.0),
                                'evaluation_metrics': model_data.get('evaluation_metrics', {}),
                                'feature_importance': model_data.get('feature_importance', {}),
                                'model_performance': model_data.get('model_performance', {})
                            }
                            
                            # Add performance data
                            tactician_ensemble_performance[f"{regime_id}_{model_name}"] = {
                                'regime_id': regime_id,
                                'model_name': model_name,
                                'performance_available': bool(model_data.get('evaluation_metrics')),
                                'feature_importance_available': bool(model_data.get('feature_importance')),
                                'training_successful': 'error' not in model_data,
                                'model_available': model_data.get('model') is not None
                            }
        
        # Add artifacts to results
        results['artifacts'] = {
            'tactician_ensembles': tactician_ensembles,
            'ensemble_metrics': ensemble_metrics,
            'tactician_ensemble_performance': tactician_ensemble_performance
        }
        
        return results
    
    def _add_comprehensive_reporting(self, results: Dict[str, Any], overall_start_time: float) -> Dict[str, Any]:
        """Add comprehensive reporting and progress tracking to results."""
        try:
            total_time = time.time() - overall_start_time
            
            # Create comprehensive report
            comprehensive_report = {
                'training_summary': {
                    'total_training_time': total_time,
                    'steps_completed': len([p for p in self.progress_tracker if p.success]),
                    'steps_failed': len([p for p in self.progress_tracker if not p.success]),
                    'vectorization_enabled': self.enable_vectorization,
                    'configuration': {
                        'model_name': self.config.model_name,
                        'timeframe': self.config.timeframe,
                        'model_types': self.config.model_types,
                        'hpo_enabled': self.config.enable_hpo,
                        'hpo_trials': self.config.hpo_n_trials
                    }
                },
                'step_breakdown': [
                    {
                        'step_name': progress.step_name,
                        'duration': progress.duration,
                        'success': progress.success,
                        'error_message': progress.error_message,
                        'metrics': progress.metrics
                    }
                    for progress in self.progress_tracker
                ],
                'performance_metrics': {
                    'total_regimes': len(results.get('models', {})),
                    'successful_regimes': len([r for r in results.get('models', {}).values() if 'error' not in r]),
                    'failed_regimes': len([r for r in results.get('models', {}).values() if 'error' in r]),
                    'average_training_time_per_regime': total_time / max(len(results.get('models', {})), 1)
                }
            }
            
            # Add evaluation summary if available
            if 'evaluation_results' in results:
                evaluation_summary = self._summarize_evaluation_results(results['evaluation_results'])
                comprehensive_report['evaluation_summary'] = evaluation_summary
            
            # Add to results
            results['comprehensive_report'] = comprehensive_report
            results['progress_tracker'] = [progress.__dict__ for progress in self.progress_tracker]
            
            # Log summary
            self._log_comprehensive_summary(comprehensive_report)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to add comprehensive reporting: {e}")
            # Return results without reporting if it fails
            return results
    
    def _summarize_evaluation_results(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize evaluation results across all regimes."""
        try:
            summary = {
                'total_regimes_evaluated': len(evaluation_results),
                'regime_metrics': {},
                'overall_performance': {}
            }
            
            all_metrics = []
            for regime, metrics in evaluation_results.items():
                if isinstance(metrics, dict) and 'error' not in metrics:
                    summary['regime_metrics'][regime] = metrics
                    all_metrics.append(metrics)
            
            # Calculate overall performance if we have metrics
            if all_metrics:
                metric_names = set()
                for metrics in all_metrics:
                    metric_names.update(metrics.keys())
                
                for metric_name in metric_names:
                    values = [m.get(metric_name) for m in all_metrics if metric_name in m and m[metric_name] is not None]
                    if values:
                        summary['overall_performance'][metric_name] = {
                            'mean': np.mean(values),
                            'std': np.std(values),
                            'min': np.min(values),
                            'max': np.max(values),
                            'count': len(values)
                        }
            
            return summary
            
        except Exception as e:
            self.logger.warning(f"Failed to summarize evaluation results: {e}")
            return {'error': str(e)}
    
    def _log_comprehensive_summary(self, report: Dict[str, Any]) -> None:
        """Log comprehensive training summary."""
        try:
            summary = report['training_summary']
            performance = report['performance_metrics']
            
            self.logger.info("=" * 80)
            self.logger.info("🎯 TACTICIAN ENSEMBLE TRAINING SUMMARY")
            self.logger.info("=" * 80)
            self.logger.info(f"⏱️  Total Training Time: {summary['total_training_time']:.2f}s")
            self.logger.info(f"✅ Steps Completed: {summary['steps_completed']}")
            self.logger.info(f"❌ Steps Failed: {summary['steps_failed']}")
            self.logger.info(f"🚀 Vectorization: {'Enabled' if summary['vectorization_enabled'] else 'Disabled'}")
            self.logger.info(f"📊 Total Regimes: {performance['total_regimes']}")
            self.logger.info(f"✅ Successful Regimes: {performance['successful_regimes']}")
            self.logger.info(f"❌ Failed Regimes: {performance['failed_regimes']}")
            
            # Log step breakdown
            self.logger.info("\n📋 Step Breakdown:")
            for step in report['step_breakdown']:
                status = "✅" if step['success'] else "❌"
                self.logger.info(f"  {status} {step['step_name']}: {step['duration']:.2f}s")
                if not step['success'] and step['error_message']:
                    self.logger.info(f"    Error: {step['error_message']}")
            
            # Log evaluation summary if available
            if 'evaluation_summary' in report:
                eval_summary = report['evaluation_summary']
                if 'overall_performance' in eval_summary and eval_summary['overall_performance']:
                    self.logger.info("\n📈 Overall Performance:")
                    for metric, stats in eval_summary['overall_performance'].items():
                        self.logger.info(f"  {metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")
            
            self.logger.info("=" * 80)
            
        except Exception as e:
            self.logger.error(f"Failed to log comprehensive summary: {e}")


# Convenience functions for backward compatibility
def create_tactician_ensemble_training_step(
    config: Optional[EnsembleTrainingConfig] = None
) -> TacticianEnsembleTrainingStep:
    """Create Tactician ensemble training step."""
    return TacticianEnsembleTrainingStep(config)


def execute_tactician_ensemble_training(
    X: np.ndarray,
    y: np.ndarray,
    regime_labels: np.ndarray,
    config: Optional[EnsembleTrainingConfig] = None,
    feature_names: Optional[List[str]] = None,
    hmm_states: Optional[np.ndarray] = None,
    base_tactician_models: Optional[Dict[str, Any]] = None,
    tactician_training_metrics: Optional[Dict[str, Any]] = None,
    analyst_models: Optional[Dict[str, Any]] = None,
    analyst_ensembles: Optional[Dict[str, Any]] = None,
    analyst_ensemble_metrics: Optional[Dict[str, Any]] = None,
    hmm_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Execute Tactician ensemble training step."""
    step = create_tactician_ensemble_training_step(config)
    return step.execute(
        X, y, regime_labels, feature_names, hmm_states,
        base_tactician_models, tactician_training_metrics,
        analyst_models, analyst_ensembles, analyst_ensemble_metrics, hmm_data
    )


# Example usage and comparison
if __name__ == "__main__":
    # Example of how to use the meta-learner ensemble training version
    print("Tactician Ensemble Training Step (Meta-Learner)")
    print("=" * 60)
    
    # Create configuration
    config = EnsembleTrainingConfig(
        model_name="tactician_ensemble_models",
        timeframe="1m",
        model_types=["XGBOOST_CUSTOM"],
        hpo_n_trials=50,  # Reduced for demo
        enable_hpo=True,
        save_models=True,
        model_save_path="./models/tactician_ensemble_models_refactored"
    )
    
    # Create training step
    training_step = create_tactician_ensemble_training_step(config)
    
    print(f"✅ Created tactician ensemble training step with {len(config.model_types)} ensemble types")
    print(f"📊 HPO enabled: {config.enable_hpo}")
    print(f"💾 Save models: {config.save_models}")
    print(f"📁 Save path: {config.model_save_path}")
    print(f"⏰ Base timeframe: {config.timeframe}")
    
    # The actual training would be called with:
    # results = training_step.execute(X, y, regime_labels, feature_names, hmm_states, ...)
    
    print("\n🎯 Tactician Ensemble Module Features:")
    print("- Operates on 1m timeframe with cross-timeframe features")
    print("- Meta-learner combining ALL previous model inputs")
    print("- All-regime ensemble training for comprehensive intelligence")
    print("- Final timing decision optimization")
    print("- Models: VotingRegressor, StackingRegressor, BaggingRegressor, AdaBoostRegressor")
    print("- Comprehensive context from ALL model types")
    
    print("\n🔄 Integration with ALL Previous Models:")
    print("- Receives individual tactician model predictions")
    print("- Integrates analyst model predictions")
    print("- Integrates analyst ensemble predictions")
    print("- Integrates HMM regime data and features")
    print("- Creates final meta-learner for optimal timing decisions")
    print("- Provides comprehensive market intelligence")