"""
Base Training Step

Base class for all training steps with common functionality.
Uses existing utilities for maximum efficiency and consistency.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
import time
from abc import ABC, abstractmethod

# Import tprint utilities
from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error,
    tprint_success, tprint_progress, tprint_performance, tprint_timer,
    tprint_data_format, LogLevel
)

# Use existing utilities
from src.utils.logger import system_logger
from src.utils.common_operations import safe_file_exists, safe_json_dump
from src.utils.parquet_utils import ParquetUtils
from src.utils.data.unified_data_utils import UnifiedDataUtils

from src.utils.ml_common.config.base_training_config import BaseTrainingConfig
from src.utils.ml_common.data_processing.regime_processing import RegimeProcessor
from src.utils.ml_common.data_processing.feature_preparation import FeaturePreparator
from src.utils.ml_common.training.training_utils import TrainingUtils
from src.utils.model_manager import ModelManager
from src.utils.ml_common.evaluation.evaluation_utils import EvaluationUtils

# Import universal validation integration
from .universal_validation_integration import (
    get_validation_integrator,
    validate_training_data,
    validate_trained_model,
    ValidationIntegrationConfig
)

# Import enhanced training utilities (lazy loading)
def _get_enhanced_training_utils():
    """Lazy import enhanced training utilities."""
    try:
        from .enhanced_training_utils import (
            EnhancedTrainingUtils,
            EarlyStoppingConfig,
            PurgedCVConfig,
            OverfittingMonitorConfig,
            RegularizationConfig
        )
        return {
            'EnhancedTrainingUtils': EnhancedTrainingUtils,
            'EarlyStoppingConfig': EarlyStoppingConfig,
            'PurgedCVConfig': PurgedCVConfig,
            'OverfittingMonitorConfig': OverfittingMonitorConfig,
            'RegularizationConfig': RegularizationConfig
        }
    except ImportError:
        return {
            'EnhancedTrainingUtils': None,
            'EarlyStoppingConfig': None,
            'PurgedCVConfig': None,
            'OverfittingMonitorConfig': None,
            'RegularizationConfig': None
        }

def _get_training_integration():
    """Lazy import training integration utilities."""
    try:
        from .training_integration import (
            TrainingStepEnhancer,
            TrainingIntegrationConfig
        )
        return {
            'TrainingStepEnhancer': TrainingStepEnhancer,
            'TrainingIntegrationConfig': TrainingIntegrationConfig
        }
    except ImportError:
        return {
            'TrainingStepEnhancer': None,
            'TrainingIntegrationConfig': None
        }

logger = system_logger.getChild('BaseTrainingStep')

class BaseTrainingStep(ABC):
    """
    Abstract base class for all training steps with common functionality.

    This class provides a comprehensive interface for training steps with
    production-ready features including error handling, validation, logging,
    and hardware optimization.
    """

    def __init__(self, config: BaseTrainingConfig):
        """
        Initialize base training step.

        Args:
            config: Training configuration object
        """
        tprint("🏗️ [BASE_TRAINING_STEP] Initializing Base Training Step", color="blue")
        self.config = config
        self.logger = logger.getChild(self.__class__.__name__)

        # Initialize common components
        tprint("🔧 [BASE_TRAINING_STEP] Initializing common components", color="cyan")
        self._initialize_common_components()

        # Initialize validation integration
        tprint("✅ [BASE_TRAINING_STEP] Initializing validation integration", color="cyan")
        self._initialize_validation_integration()

        # Initialize enhanced training utilities (lazy loading)
        tprint("🚀 [BASE_TRAINING_STEP] Initializing enhanced training utilities", color="cyan")
        self._initialize_enhanced_training()

        # Training results
        self.training_results = {}

        tprint_success("✅ [BASE_TRAINING_STEP] Base Training Step initialized successfully")
        self.logger.info("✅ Base Training Step initialized")

    def execute_training(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the main training logic.
        
        Args:
            data: Training data dictionary
            
        Returns:
            Training results dictionary
        """
        # Default implementation - subclasses should override
        self.logger.warning("Using default execute_training implementation - subclasses should override")
        
        try:
            # Basic validation
            if not self.validate_training_data(data):
                raise ValueError("Training data validation failed")
            
            # Prepare data
            prepared_data = self.prepare_training_data(data)
            
            # Return basic results structure
            return {
                'success': True,
                'training_time': 0.0,
                'models': {},
                'evaluation_results': {},
                'metadata': {
                    'training_step': self.__class__.__name__,
                    'data_shape': str(data.get('X', 'unknown').shape) if 'X' in data else 'unknown'
                }
            }
        except Exception as e:
            self.logger.error(f"Default training execution failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'training_time': 0.0,
                'models': {},
                'evaluation_results': {}
            }

    def validate_training_data(self, data: Dict[str, Any]) -> bool:
        """
        Validate training data before training.
        
        Args:
            data: Training data dictionary
            
        Returns:
            True if data is valid, False otherwise
        """
        # Default implementation - subclasses should override
        try:
            if not isinstance(data, dict):
                self.logger.error("Training data must be a dictionary")
                return False
            
            # Check for required keys
            required_keys = ['X', 'y']
            for key in required_keys:
                if key not in data:
                    self.logger.error(f"Required key '{key}' not found in training data")
                    return False
            
            # Basic shape validation
            X = data['X']
            y = data['y']
            
            if hasattr(X, 'shape') and hasattr(y, 'shape'):
                if X.shape[0] != y.shape[0]:
                    self.logger.error(f"Sample count mismatch: X={X.shape[0]}, y={y.shape[0]}")
                    return False
            
            return True
        except Exception as e:
            self.logger.error(f"Data validation failed: {e}")
            return False

    def prepare_training_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare raw data for training.
        
        Args:
            raw_data: Raw training data
            
        Returns:
            Prepared training data
        """
        # Default implementation - subclasses should override
        self.logger.info("Using default data preparation - subclasses should override")
        
        try:
            # Basic preparation - just return a copy
            prepared_data = raw_data.copy()
            
            # Add basic metadata
            prepared_data['prepared_at'] = time.time()
            prepared_data['preparation_method'] = 'default'
            
            return prepared_data
        except Exception as e:
            self.logger.error(f"Data preparation failed: {e}")
            return raw_data

    def evaluate_model(self, model: Any, test_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate trained model on test data.
        
        Args:
            model: Trained model
            test_data: Test data dictionary
            
        Returns:
            Evaluation metrics dictionary
        """
        # Default implementation - subclasses should override
        self.logger.warning("Using default model evaluation - subclasses should override")
        
        try:
            if model is None:
                return {'error': 'Model is None'}
            
            if 'X' not in test_data or 'y' not in test_data:
                return {'error': 'Missing X or y in test data'}
            
            X_test = test_data['X']
            y_test = test_data['y']
            
            # Basic evaluation if model has predict method
            if hasattr(model, 'predict'):
                try:
                    y_pred = model.predict(X_test)
                    
                    # Calculate basic metrics
                    if hasattr(y_test, 'shape') and len(y_test.shape) > 1 and y_test.shape[1] > 1:
                        # Multi-output case
                        mse = np.mean((y_test - y_pred) ** 2)
                        mae = np.mean(np.abs(y_test - y_pred))
                    else:
                        # Single output case
                        y_test_flat = y_test.flatten() if hasattr(y_test, 'flatten') else y_test
                        y_pred_flat = y_pred.flatten() if hasattr(y_pred, 'flatten') else y_pred
                        mse = np.mean((y_test_flat - y_pred_flat) ** 2)
                        mae = np.mean(np.abs(y_test_flat - y_pred_flat))
                    
                    return {
                        'mse': float(mse),
                        'mae': float(mae),
                        'evaluation_method': 'default'
                    }
                except Exception as e:
                    return {'error': f'Prediction failed: {e}'}
            else:
                return {'error': 'Model does not support predict method'}
                
        except Exception as e:
            self.logger.error(f"Model evaluation failed: {e}")
            return {'error': str(e)}

    def save_training_results(self, results: Dict[str, Any], filepath: str) -> bool:
        """
        Save training results to file.
        
        Args:
            results: Training results dictionary
            filepath: Path to save results
            
        Returns:
            True if saved successfully, False otherwise
        """
        # Default implementation - subclasses should override
        try:
            # Ensure directory exists
            import os
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Use existing safe_json_dump utility
            success = safe_json_dump(results, filepath)
            
            if success:
                self.logger.info(f"Training results saved to {filepath}")
            else:
                self.logger.error(f"Failed to save training results to {filepath}")
            
            return success
        except Exception as e:
            self.logger.error(f"Failed to save training results: {e}")
            return False

    def load_training_results(self, filepath: str) -> Optional[Dict[str, Any]]:
        """
        Load training results from file.
        
        Args:
            filepath: Path to load results from
            
        Returns:
            Training results dictionary or None if failed
        """
        # Default implementation - subclasses should override
        try:
            if not safe_file_exists(filepath):
                self.logger.error(f"Training results file not found: {filepath}")
                return None
            
            # Use existing safe_json_load utility
            results = safe_json_load(filepath)
            
            if results is not None:
                self.logger.info(f"Training results loaded from {filepath}")
            else:
                self.logger.error(f"Failed to load training results from {filepath}")
            
            return results
        except Exception as e:
            self.logger.error(f"Failed to load training results: {e}")
            return None

    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive training summary.
        
        Returns:
            Training summary dictionary
        """
        # Default implementation - subclasses should override
        try:
            summary = {
                'training_step': self.__class__.__name__,
                'config': self.config.__dict__ if hasattr(self.config, '__dict__') else str(self.config),
                'training_results': self.training_results,
                'timestamp': time.time(),
                'summary_method': 'default'
            }
            
            return summary
        except Exception as e:
            self.logger.error(f"Failed to get training summary: {e}")
            return {'error': str(e)}

    def _initialize_enhanced_training(self):
        """Initialize enhanced training utilities with lazy loading."""
        # Get enhanced training utilities
        enhanced_utils = _get_enhanced_training_utils()
        training_integration = _get_training_integration()

        self.enhanced_training_available = enhanced_utils['EnhancedTrainingUtils'] is not None
        self.enhanced_training_utils = enhanced_utils['EnhancedTrainingUtils']
        self.training_enhancer = training_integration['TrainingStepEnhancer']
        self.enhanced_training_config = training_integration['TrainingIntegrationConfig']

        # Initialize training step enhancer if available
        if self.training_enhancer:
            self.logger.info("✅ Enhanced training utilities initialized")
        else:
            self.logger.info("⚠️ Enhanced training utilities not available (optional)")

    def _initialize_common_components(self):
        """Initialize common components using existing utilities."""
        tprint("🔧 [BASE_TRAINING_STEP] Setting up training utilities", color="blue")
        # Initialize training utilities with hardware optimization
        self.training_utils = TrainingUtils(self.config)

        tprint("📊 [BASE_TRAINING_STEP] Setting up data processors", color="blue")
        # Initialize data processors
        self.regime_processor = RegimeProcessor()
        self.feature_preparator = FeaturePreparator()

        tprint("💾 [BASE_TRAINING_STEP] Setting up model manager", color="blue")
        # Initialize model manager with existing serialization utilities
        self.model_manager = ModelManager(
            save_path=self.config.model_save_path,
            save_format=self.config.save_format
        )

        tprint("📈 [BASE_TRAINING_STEP] Setting up evaluation utilities", color="blue")
        # Initialize evaluation utilities
        self.evaluation_utils = EvaluationUtils()

        tprint("🗄️ [BASE_TRAINING_STEP] Setting up data utilities", color="blue")
        # Initialize existing data utilities
        self.data_utils = UnifiedDataUtils()
        self.parquet_utils = ParquetUtils()

        tprint_success("✅ [BASE_TRAINING_STEP] Common components initialized with existing utilities")
        self.logger.info("✅ Common components initialized with existing utilities")

    def _initialize_validation_integration(self):
        """Initialize universal validation integration."""
        # Create validation configuration
        validation_config = ValidationIntegrationConfig(
            enable_validation=getattr(self.config, 'enable_validation', True),
            enable_overfitting_detection=getattr(self.config, 'enable_overfitting_detection', True),
            enable_temporal_validation=getattr(self.config, 'enable_temporal_validation', True),
            enable_timeframe_validation=getattr(self.config, 'enable_timeframe_validation', True),
            save_validation_reports=getattr(self.config, 'save_validation_reports', True),
            validation_report_directory=getattr(self.config, 'validation_report_directory', "reports/validation"),
            enable_validation_logging=getattr(self.config, 'enable_validation_logging', True),
            fail_on_validation_error=getattr(self.config, 'fail_on_validation_error', False),
            warn_on_validation_issues=getattr(self.config, 'warn_on_validation_issues', True)
        )

        # Initialize validation integrator
        self.validation_integrator = get_validation_integrator(validation_config)

        self.logger.info("✅ Universal validation integration initialized")

    def validate_training_data(self,
                              X: np.ndarray,
                              y: np.ndarray,
                              regime_labels: np.ndarray,
                              feature_names: Optional[List[str]] = None,
                              timestamps: Optional[np.ndarray] = None,
                              model_type: str = "unknown") -> Dict[str, Any]:
        """
        Validate training data before model training.

        Args:
            X: Input features
            y: Target values
            regime_labels: Regime labels for each sample
            feature_names: Names of input features
            timestamps: Optional timestamps for temporal validation
            model_type: Type of model

        Returns:
            Dict: Validation results
        """
        tprint(f"🔍 [BASE_TRAINING_STEP] validate_training_data() called for {model_type}", color="blue")
        tprint(f"📊 [BASE_TRAINING_STEP] Input: X={X.shape}, y={y.shape}, regimes={len(np.unique(regime_labels))}", color="cyan")
        
        # Validate data format for troubleshooting
        tprint_data_format(X, f"training_data_X_{model_type}", level=LogLevel.DEBUG)
        tprint_data_format(y, f"training_data_y_{model_type}", level=LogLevel.DEBUG)
        tprint_data_format(regime_labels, f"training_data_regimes_{model_type}", level=LogLevel.DEBUG)
        # Split data for validation
        from sklearn.model_selection import train_test_split
        # Use stratified split only if every class has at least 2 samples; otherwise fallback
        stratify_labels = None
        try:
            unique, counts = np.unique(y, return_counts=True)
            if len(unique) > 1 and np.all(counts >= 2):
                stratify_labels = y
        except Exception:
            stratify_labels = None

        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=self.config.validation_split if hasattr(self.config, 'validation_split') else 0.2,
                random_state=42, stratify=stratify_labels
            )
        except ValueError:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=self.config.validation_split if hasattr(self.config, 'validation_split') else 0.2,
                random_state=42, stratify=None
            )

        # Validate training data
        validation_results = self.validation_integrator.validate_training_data(
            X_train=X_train,
            X_val=X_val,
            y_train=y_train,
            y_val=y_val,
            timestamps=timestamps,
            feature_names=feature_names,
            model_type=model_type
        )

        # Log validation results
        if validation_results['valid']:
            tprint_success("✅ [BASE_TRAINING_STEP] validate_training_data() completed successfully")
            self.logger.info("✅ Training data validation passed")
        else:
            tprint_error("❌ [BASE_TRAINING_STEP] validate_training_data() failed validation")
            self.logger.warning("⚠️ Training data validation failed")
            for issue in validation_results.get('critical_issues', []):
                tprint_error(f"❌ [BASE_TRAINING_STEP] Critical issue: {issue}")
                self.logger.error(f"Critical issue: {issue}")
            for warning in validation_results.get('warnings', []):
                tprint_warning(f"⚠️ [BASE_TRAINING_STEP] Warning: {warning}")
                self.logger.warning(f"Warning: {warning}")

        tprint(f"📊 [BASE_TRAINING_STEP] validate_training_data() outcome: valid={validation_results['valid']}", color="green" if validation_results['valid'] else "red")
        return validation_results

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
        tprint(f"🔍 [BASE_TRAINING_STEP] validate_trained_model() called for {model_name} ({model_type})", color="blue")
        tprint(f"📊 [BASE_TRAINING_STEP] Model validation input: train={X_train.shape}, val={X_val.shape}", color="cyan")
        
        # Validate data format for troubleshooting
        tprint_data_format(model, f"trained_model_{model_name}", level=LogLevel.DEBUG)
        tprint_data_format(X_train, f"validation_X_train_{model_name}", level=LogLevel.DEBUG)
        tprint_data_format(X_val, f"validation_X_val_{model_name}", level=LogLevel.DEBUG)
        tprint_data_format(y_train, f"validation_y_train_{model_name}", level=LogLevel.DEBUG)
        tprint_data_format(y_val, f"validation_y_val_{model_name}", level=LogLevel.DEBUG)
        # Validate trained model
        validation_results = self.validation_integrator.validate_trained_model(
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

        # Log validation results
        if validation_results['valid']:
            tprint_success(f"✅ [BASE_TRAINING_STEP] validate_trained_model() passed for {model_name}")
            tprint(f"📊 [BASE_TRAINING_STEP] Validation score: {validation_results.get('validation_score', 'N/A')}", color="green")
            self.logger.info(f"✅ Model validation passed for {model_name}")
            self.logger.info(f"  Validation score: {validation_results.get('validation_score', 'N/A')}")
        else:
            tprint_error(f"❌ [BASE_TRAINING_STEP] validate_trained_model() failed for {model_name}")
            self.logger.warning(f"⚠️ Model validation failed for {model_name}")
            for issue in validation_results.get('critical_issues', []):
                tprint_error(f"❌ [BASE_TRAINING_STEP] Critical issue: {issue}")
                self.logger.error(f"Critical issue: {issue}")
            for warning in validation_results.get('warnings', []):
                tprint_warning(f"⚠️ [BASE_TRAINING_STEP] Warning: {warning}")
                self.logger.warning(f"Warning: {warning}")

        # Store validation results in training results
        if 'validation_results' not in self.training_results:
            self.training_results['validation_results'] = {}

        self.training_results['validation_results'][model_name] = validation_results

        # Process validation with reporting system
        if self.config.save_validation_reports:
            from ..reporting.validation_reporting_integration import process_validation_with_reporting
            process_validation_with_reporting(
                validation_report=validation_results,
                model_name=model_name,
                model_type=model_type,
                fold_number=fold_number,
                model_metadata={'training_step': self.__class__.__name__}
            )

        tprint(f"📊 [BASE_TRAINING_STEP] validate_trained_model() outcome: valid={validation_results['valid']}", color="green" if validation_results['valid'] else "red")
        return validation_results

    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of all validations performed."""
        tprint("📊 [BASE_TRAINING_STEP] get_validation_summary() called", color="blue")
        summary = self.validation_integrator.get_validation_summary()
        tprint(f"📊 [BASE_TRAINING_STEP] get_validation_summary() outcome: {len(summary)} validation entries", color="green")
        return summary

    def execute(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regime_labels: np.ndarray,
        feature_names: Optional[List[str]] = None,
        hmm_states: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute training step with common workflow.

        This method provides a default implementation that handles the common
        training workflow. Subclasses can override this method for specialized
        training logic while still benefiting from the common infrastructure.

        Args:
            X: Input features
            y: Target values
            regime_labels: Regime labels for each sample
            feature_names: Names of input features
            hmm_states: HMM cluster/regime states
            **kwargs: Additional arguments

        Returns:
            Dictionary containing training results and metadata
        """
        start_time = time.time()
        tprint("🚀 [BASE_TRAINING_STEP] Starting training execution", color="cyan", bold=True)
        tprint(f"📊 [BASE_TRAINING_STEP] Input data shape: {X.shape}, targets: {y.shape}", color="blue")
        self.logger.info("🚀 Starting training execution")

        try:
            # Step 1: Validate training data
            self.logger.info("📊 Validating training data...")
            validation_results = self.validate_training_data(
                X=X,
                y=y,
                regime_labels=regime_labels,
                feature_names=feature_names,
                model_type=self.__class__.__name__
            )

            if not validation_results.get('valid', False) and self.config.fail_on_validation_error:
                raise ValueError(f"Training data validation failed: {validation_results.get('critical_issues', [])}")

            # Step 2: Analyze regimes
            self.logger.info("🔍 Analyzing regime distribution...")
            regime_analysis = self.analyze_regimes(regime_labels)

            # Step 3: Prepare regime-specific data
            self.logger.info("📋 Preparing regime-specific data...")
            regime_data = self.prepare_regime_data(
                X=X,
                y=y,
                regime_labels=regime_labels,
                regime_analysis=regime_analysis,
                hmm_states=hmm_states
            )

            # Step 4: Prepare combined features
            self.logger.info("🔧 Preparing combined features...")
            X_combined, combined_feature_names = self.prepare_combined_features(
                X=X,
                regime_labels=regime_labels,
                hmm_states=hmm_states,
                feature_names=feature_names
            )

            # Step 5: Train models (subclasses should override this part)
            self.logger.info("🤖 Training models...")
            model_types = kwargs.get('model_types', ['RandomForest', 'XGBoost', 'LightGBM'])

            # Use default training if no specific model training is implemented
            models = {}
            training_metadata = {}
            evaluation_results = {}

            # Try to train models using the common training utilities
            try:
                training_results = self.train_models(
                    model_types=model_types,
                    X=X_combined,
                    y=y,
                    enable_hpo=kwargs.get('enable_hpo', True)
                )

                models = training_results.get('models', {})
                training_metadata = training_results.get('metadata', {})
                evaluation_results = training_results.get('evaluation_results', {})

            except Exception as training_error:
                self.logger.warning(f"⚠️ Model training failed, using fallback: {training_error}")
                # Fallback: create dummy models for structure
                models = {'fallback_model': None}
                training_metadata = {'fallback': True, 'error': str(training_error)}
                evaluation_results = {'fallback_model': {'error': str(training_error)}}

            # Step 6: Validate trained models
            self.logger.info("✅ Validating trained models...")
            for model_name, model in models.items():
                if model is not None:
                    validation_results = self.validate_trained_model(
                        model=model,
                        X_train=X_combined,
                        X_val=X_combined[:min(1000, len(X_combined)//5)],  # Use subset for validation
                        y_train=y,
                        y_val=y[:min(1000, len(y)//5)],
                        model_name=model_name,
                        model_type=self.__class__.__name__
                    )

                    # Update evaluation results with validation
                    if model_name in evaluation_results:
                        evaluation_results[model_name].update(validation_results)

            # Step 7: Save models and metadata
            self.logger.info("💾 Saving models and metadata...")
            saved_paths = []
            if models:
                saved_paths = self.save_models(
                    models=models,
                    model_type=self.__class__.__name__,
                    symbol=kwargs.get('symbol'),
                    exchange=kwargs.get('exchange'),
                    timeframe=kwargs.get('timeframe')
                )

            # Create comprehensive metadata
            final_metadata = self.get_model_metadata(
                model=list(models.values())[0] if models else None,
                model_name=self.__class__.__name__,
                training_time=time.time() - start_time,
                samples=len(X),
                features=X_combined.shape[1] if len(X_combined.shape) > 1 else X_combined.shape[0]
            )

            final_metadata.update({
                'regime_analysis': regime_analysis,
                'training_data_shape': X.shape,
                'target_distribution': dict(zip(*np.unique(y, return_counts=True))),
                'regime_data': regime_data,
                'feature_names': combined_feature_names,
                'saved_model_paths': saved_paths,
                'execution_time': time.time() - start_time,
                'training_step_type': self.__class__.__name__
            })

            # Step 8: Create final results
            final_results = self._create_final_results(
                models=models,
                metadata=final_metadata,
                evaluation_results=evaluation_results,
                training_time=time.time() - start_time
            )

            # Step 9: Log training summary
            self._log_training_summary(
                final_results,
                model_type=self.__class__.__name__,
                n_models=len(models)
            )

            self.logger.info(f"✅ Training execution completed successfully in {time.time() - start_time:.2f}s")
            return final_results

        except Exception as e:
            self.logger.error(f"❌ Training execution failed: {e}")
            return self._handle_training_error(e, "training execution")

    def analyze_regimes(self, regime_labels: np.ndarray) -> Dict[str, Any]:
        """
        Analyze regime distribution and characteristics.

        Args:
            regime_labels: Array of regime labels for each sample

        Returns:
            Dictionary containing regime analysis results
        """
        tprint(f"🔍 [BASE_TRAINING_STEP] analyze_regimes() called for {len(regime_labels)} samples", color="blue")
        tprint(f"📊 [BASE_TRAINING_STEP] Unique regimes: {len(np.unique(regime_labels))}", color="cyan")
        result = self.regime_processor.analyze_regimes(
            regime_labels=regime_labels,
            min_samples=self.config.min_samples_per_regime,
            enable_regime_merging=self.config.enable_regime_merging,
            regime_merge_threshold=self.config.regime_merge_threshold
        )
        tprint_success("✅ [BASE_TRAINING_STEP] analyze_regimes() completed successfully")
        tprint(f"📊 [BASE_TRAINING_STEP] analyze_regimes() outcome: {len(result)} analysis results", color="green")
        return result

    def prepare_regime_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regime_labels: np.ndarray,
        regime_analysis: Dict[str, Any],
        hmm_states: Optional[np.ndarray] = None
    ) -> Dict[int, Dict[str, np.ndarray]]:
        """
        Prepare data for each regime with HMM state integration.

        Args:
            X: Input features
            y: Target values
            regime_labels: Regime labels for each sample
            regime_analysis: Results from regime analysis
            hmm_states: Optional HMM cluster/regime states

        Returns:
            Dictionary containing prepared data for each regime
        """
        return self.regime_processor.prepare_regime_data(
            X=X,
            y=y,
            regime_labels=regime_labels,
            regime_analysis=regime_analysis,
            hmm_states=hmm_states,
            min_samples=self.config.min_samples_per_regime,
            enable_data_augmentation=self.config.enable_data_augmentation,
            augmentation_method=self.config.augmentation_method,
            augmentation_ratio=self.config.augmentation_ratio
        )

    def prepare_combined_features(
        self,
        X: np.ndarray,
        regime_labels: np.ndarray,
        hmm_states: Optional[np.ndarray] = None,
        analyst_outputs: Optional[np.ndarray] = None,
        analyst_output_names: Optional[List[str]] = None,
        feature_names: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare combined features with HMM states, analyst outputs, and regime features.

        Args:
            X: Input features
            regime_labels: Array of regime labels
            hmm_states: Optional HMM cluster/regime states
            analyst_outputs: Optional analyst model outputs
            analyst_output_names: Names of analyst output features
            feature_names: Names of input features

        Returns:
            Tuple of combined features and feature names
        """
        return self.feature_preparator.prepare_combined_features(
            X=X,
            regime_labels=regime_labels,
            hmm_states=hmm_states,
            analyst_outputs=analyst_outputs,
            analyst_output_names=analyst_output_names,
            feature_names=feature_names
        )

    def train_models(
        self,
        model_types: List[str],
        X: np.ndarray,
        y: np.ndarray,
        enable_hpo: bool = True,
        search_spaces: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Train multiple models.

        Args:
            model_types: List of model types to train
            X: Input features
            y: Target values
            enable_hpo: Whether to use HPO
            search_spaces: HPO search spaces for each model type

        Returns:
            Dictionary containing training results
        """
        tprint(f"🤖 [BASE_TRAINING_STEP] train_models() called for {len(model_types)} model types", color="blue")
        tprint(f"📊 [BASE_TRAINING_STEP] Input: X={X.shape}, y={y.shape}, HPO={enable_hpo}", color="cyan")
        tprint(f"🤖 [BASE_TRAINING_STEP] Model types: {model_types}", color="yellow")
        result = self.training_utils.train_models(
            model_types=model_types,
            X=X,
            y=y,
            enable_hpo=enable_hpo,
            search_spaces=search_spaces
        )
        tprint_success("✅ [BASE_TRAINING_STEP] train_models() completed successfully")
        tprint(f"📊 [BASE_TRAINING_STEP] train_models() outcome: {len(result.get('models', {}))} models trained", color="green")
        return result

    def evaluate_models(
        self,
        models: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        is_classification: bool = True
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate multiple models.

        Args:
            models: Dictionary of trained models
            X: Input features
            y: True target values
            is_classification: Whether this is a classification task

        Returns:
            Dictionary containing evaluation results for each model
        """
        return self.training_utils.evaluate_models(
            models=models,
            X=X,
            y=y,
            is_classification=is_classification
        )

    def save_models(
        self,
        models: Dict[str, Any],
        model_type: str,
        symbol: Optional[str] = None,
        exchange: Optional[str] = None,
        timeframe: Optional[str] = None,
        regime: Optional[int] = None
    ) -> List[str]:
        """
        Save trained models.

        Args:
            models: Dictionary of models to save
            model_type: Type of models
            symbol: Optional symbol identifier
            exchange: Optional exchange identifier
            timeframe: Optional timeframe identifier
            regime: Optional regime identifier

        Returns:
            List of saved model file paths
        """
        return self.model_manager.save_models(
            models=models,
            model_type=model_type,
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            regime=regime
        )

    def save_metadata(
        self,
        metadata: Dict[str, Any],
        model_type: str,
        symbol: Optional[str] = None,
        exchange: Optional[str] = None,
        timeframe: Optional[str] = None,
        regime: Optional[int] = None
    ) -> str:
        """
        Save model metadata.

        Args:
            metadata: Model metadata to save
            model_type: Type of models
            symbol: Optional symbol identifier
            exchange: Optional exchange identifier
            timeframe: Optional timeframe identifier
            regime: Optional regime identifier

        Returns:
            Path to saved metadata file
        """
        return self.model_manager.save_metadata(
            metadata=metadata,
            model_type=model_type,
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            regime=regime
        )

    def get_model_metadata(
        self,
        model: Any,
        model_name: str,
        training_time: float = 0.0,
        optimization_time: float = 0.0,
        samples: int = 0,
        features: int = 0
    ) -> Dict[str, Any]:
        """
        Extract common model metadata.

        Args:
            model: Trained model
            model_name: Name of the model
            training_time: Training time in seconds
            optimization_time: Optimization time in seconds
            samples: Number of training samples
            features: Number of features

        Returns:
            Dictionary containing model metadata
        """
        return self.model_manager.get_model_metadata(
            model=model,
            model_name=model_name,
            training_time=training_time,
            optimization_time=optimization_time,
            samples=samples,
            features=features
        )

    def _create_final_results(
        self,
        models: Dict[str, Any],
        metadata: Dict[str, Any],
        evaluation_results: Dict[str, Any],
        training_time: float,
        additional_results: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create final results dictionary.

        Args:
            models: Trained models
            metadata: Training metadata
            evaluation_results: Evaluation results
            training_time: Total training time
            additional_results: Additional results to include

        Returns:
            Dictionary containing final results
        """
        # Add direction information to metadata if available
        enhanced_metadata = dict(metadata)
        if hasattr(self.config, 'enable_long_positions') and hasattr(self.config, 'enable_short_positions'):
            enhanced_metadata['direction_settings'] = {
                'enable_long_positions': self.config.enable_long_positions,
                'enable_short_positions': self.config.enable_short_positions,
            }

        results = {
            'models': models,
            'metadata': enhanced_metadata,
            'evaluation_results': evaluation_results,
            'training_time': training_time,
            'config': self.config
        }

        if additional_results:
            results.update(additional_results)

        return results

    def _log_training_summary(
        self,
        results: Dict[str, Any],
        model_type: str,
        n_models: int = 0
    ):
        """
        Log training summary.

        Args:
            results: Training results
            model_type: Type of models trained
            n_models: Number of models trained
        """
        training_time = results.get('training_time', 0)
        self.logger.info(f"✅ {model_type} training completed in {training_time:.2f}s")

        if n_models > 0:
            self.logger.info(f"📊 Models trained: {n_models}")

        # Log evaluation results if available
        evaluation_results = results.get('evaluation_results', {})
        if evaluation_results:
            self.logger.info("📊 Evaluation results:")
            for model_name, metrics in evaluation_results.items():
                if isinstance(metrics, dict) and 'error' not in metrics:
                    # Log key metrics
                    key_metrics = ['accuracy', 'f1_score', 'r2', 'mse']
                    metric_values = []
                    for metric in key_metrics:
                        if metric in metrics:
                            metric_values.append(f"{metric}={metrics[metric]:.4f}")

                    if metric_values:
                        self.logger.info(f"📊 - {model_name}: {', '.join(metric_values)}")

    def _handle_training_error(self, error: Exception, context: str = ""):
        """
        Handle training errors with proper logging.

        Args:
            error: Exception that occurred
            context: Additional context about where the error occurred
        """
        error_msg = f"❌ Training error{f' in {context}' if context else ''}: {error}"
        self.logger.error(error_msg)

        # Return empty results structure
        return {
            'models': {},
            'metadata': {},
            'evaluation_results': {},
            'training_time': 0,
            'config': self.config,
            'error': str(error)
        }


class PerRegimeTrainingStep(BaseTrainingStep):
    """
    Per-Regime Training Step
    
    Base class for training models per regime with enhanced validation.
    """
    
    def __init__(self, config: BaseTrainingConfig, logger=None):
        """
        Initialize per-regime training step.
        
        Args:
            config: Training configuration
            logger: Logger instance
        """
        super().__init__(config, logger)
        self.regime_processor = RegimeProcessor(config)
        
    def train_per_regime(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train models per regime.
        
        Args:
            data: Training data with regime information
            
        Returns:
            Training results per regime
        """
        try:
            tprint_info("🎯 Starting per-regime training...")
            
            # Process regime data
            regime_data = self.regime_processor.process_regime_data(data)
            
            # Train models for each regime
            regime_results = {}
            for regime_id, regime_info in regime_data.items():
                tprint_info(f"🎯 Training models for regime {regime_id}...")
                
                # Train models for this regime
                regime_result = self._train_regime_models(regime_id, regime_info)
                regime_results[regime_id] = regime_result
                
            tprint_success("✅ Per-regime training completed")
            return {
                'success': True,
                'regime_results': regime_results,
                'total_regimes': len(regime_results)
            }
            
        except Exception as e:
            tprint_error(f"❌ Per-regime training failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _train_regime_models(self, regime_id: str, regime_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train models for a specific regime.
        
        Args:
            regime_id: Regime identifier
            regime_data: Data for this regime
            
        Returns:
            Training results for this regime
        """
        try:
            import time
            training_start = time.time()
            
            # Extract features and targets from regime data
            if isinstance(regime_data, dict):
                X_regime = regime_data.get('features')
                y_regime = regime_data.get('targets')
            else:
                # Assume regime_data is a tuple of (X, y)
                X_regime, y_regime = regime_data
            
            if X_regime is None or y_regime is None:
                raise ValueError("Regime data must contain features and targets")
            
            # Train regime-specific models
            models_trained = 0
            
            # Train primary model for this regime
            if hasattr(self, 'model_manager') and self.model_manager:
                regime_model = self.model_manager.create_model(
                    model_type='regime_specific',
                    regime_id=regime_id,
                    config=self.config
                )
                
                if regime_model:
                    regime_model.fit(X_regime, y_regime)
                    models_trained += 1
            
            # Train ensemble models if configured
            if hasattr(self.config, 'enable_ensemble') and self.config.enable_ensemble:
                ensemble_models = self._create_regime_ensemble(regime_id, X_regime, y_regime)
                models_trained += len(ensemble_models)
            
            training_time = time.time() - training_start
            
            return {
                'success': True,
                'regime_id': regime_id,
                'models_trained': models_trained,
                'training_time': training_time,
                'samples_processed': len(X_regime)
            }
        except Exception as e:
            return {
                'success': False,
                'regime_id': regime_id,
                'error': str(e)
            }
    
    def _create_regime_ensemble(self, regime_id: str, X_regime, y_regime):
        """Create ensemble models for a specific regime."""
        ensemble_models = []
        
        try:
            # Create different model types for ensemble
            model_types = ['linear', 'tree', 'neural_network']
            
            for model_type in model_types:
                if hasattr(self, 'model_manager') and self.model_manager:
                    ensemble_model = self.model_manager.create_model(
                        model_type=model_type,
                        regime_id=f"{regime_id}_{model_type}",
                        config=self.config
                    )
                    
                    if ensemble_model:
                        ensemble_model.fit(X_regime, y_regime)
                        ensemble_models.append(ensemble_model)
            
            return ensemble_models
            
        except Exception as e:
            self.logger.warning(f"Failed to create ensemble for regime {regime_id}: {e}")
            return []
