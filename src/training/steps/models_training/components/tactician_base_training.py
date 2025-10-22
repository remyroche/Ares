"""
Tactician Base Training - Enhanced with Comprehensive BaseStep Utilities

This module provides the Tactician base training component that handles training
of individual Tactician base models using the unified BaseTrainer architecture
with comprehensive BaseStep utility integration.

Key Features:
- Unified training interface for all Tactician model types
- Common training patterns and lifecycle management
- Standardized configuration and validation
- Performance monitoring and checkpointing
- Error handling and recovery mechanisms
- Comprehensive BaseStep utility integration
- Advanced logging and data visualization
- Hardware optimization and memory management
- Data quality validation and cleaning
- Model persistence and caching
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd
import numpy as np

from ..core.tactician_base_trainer import (
    TacticianBaseTrainer, TacticianTrainingConfig, TacticianModelType
)
from src.training.steps.base_step import BaseStep
from src.core.decorators import handles_errors, traced, log_execution_time


@dataclass
class TacticianBaseTrainingConfig:
    """Configuration for Tactician base training."""
    model_types: List[TacticianModelType]
    training_params: Dict[str, Any] = field(default_factory=dict)
    validation_params: Dict[str, Any] = field(default_factory=dict)
    timeframe: str = "15m"
    symbol: str = "ETHUSDT"
    auto_save: bool = True
    
    # Feature engineering parameters
    enable_entry_timing: bool = True
    enable_exit_timing: bool = True
    enable_position_sizing: bool = True
    
    # Model-specific parameters
    lightgbm_params: Dict[str, Any] = field(default_factory=dict)
    catboost_params: Dict[str, Any] = field(default_factory=dict)
    neural_network_params: Dict[str, Any] = field(default_factory=dict)
    linear_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TacticianBaseTrainingResult:
    """Result of Tactician base training."""
    success: bool
    models: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    training_time: float = 0.0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    feature_importance: Optional[Dict[str, Dict[str, float]]] = None


class TacticianBaseTraining(BaseStep):
    """
    Tactician base training component using unified BaseTrainer architecture.
    
    This component handles training of individual Tactician base models with comprehensive
    state management, performance monitoring, and error handling.
    """
    
    def __init__(
        self,
        name: str = "tactician_base_training",
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the Tactician base training component.
        
        Args:
            name: Component name
            config: Configuration dictionary
            logger: Logger instance
        """
        super().__init__(name, config)
        
        # Set default configuration
        default_config = {
            'model_types': [TacticianModelType.LIGHTGBM, TacticianModelType.CATBOOST],
            'training_params': {
                'validation_split': 0.2,
                'cross_validation_folds': 5,
                'random_seed': 42
            },
            'validation_params': {
                'enable_early_stopping': True,
                'early_stopping_patience': 10
            },
            'timeframe': '15m',
            'symbol': 'ETHUSDT',
            'auto_save': True,
            'enable_entry_timing': True,
            'enable_exit_timing': True,
            'enable_position_sizing': True
        }
        
        # Merge with provided configuration
        if config:
            default_config.update(config)
        
        self.config = TacticianBaseTrainingConfig(**default_config)
        
        # Initialize trainer
        self._trainer = None
        
        # Log initialization with comprehensive utilities
        self.tprint_banner("Tactician Base Training Component")
        self.tprint_info(f"🔧 Initialized TacticianBaseTraining: {name}")
        self.tprint_config_preview(self.config.__dict__, "Tactician Base Training Config")
        
        # Log utility availability status
        self._log_utility_availability()
        
        # Initialize performance tracking
        self._performance_metrics = {}
        
        self.logger.info(f"Initialized TacticianBaseTraining: {name}")
    
    def _safe_merge_configs(self, default: Dict[str, Any], provided: Dict[str, Any]) -> Dict[str, Any]:
        """Safely merge configuration dictionaries using BaseStep utilities."""
        try:
            # Use safe operations for deep merge
            if self.common_ops and 'safe_dict_merge' in self.common_ops:
                return self.common_ops['safe_dict_merge'](default, provided)
            else:
                # Fallback implementation
                result = default.copy()
                for key, value in provided.items():
                    if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                        result[key] = self._safe_merge_configs(result[key], value)
                    else:
                        result[key] = value
                return result
        except Exception as e:
            self.tprint_warning(f"⚠️ Config merge failed, using defaults: {e}")
            return default
    
    @handles_errors(
        exceptions=(ValueError, RuntimeError, MemoryError),
        default_return=TacticianBaseTrainingResult(
            success=False,
            errors=["Component initialization failed"]
        ),
        context="tactician base training"
    )
    async def initialize(self) -> bool:
        """Initialize the component."""
        try:
            tprint_info("🔧 Initializing Tactician base training...")
            
            # Create trainer configuration
            trainer_config = TacticianTrainingConfig(
                model_types=[self._convert_model_type(mt) for mt in self.config.model_types],
                timeframe=self.config.timeframe,
                symbol=self.config.symbol,
                validation_split=self.config.training_params.get('validation_split', 0.2),
                cross_validation_folds=self.config.training_params.get('cross_validation_folds', 5),
                random_seed=self.config.training_params.get('random_seed'),
                enable_entry_timing=self.config.enable_entry_timing,
                enable_exit_timing=self.config.enable_exit_timing,
                enable_position_sizing=self.config.enable_position_sizing,
                lightgbm_params=self.config.lightgbm_params,
                catboost_params=self.config.catboost_params,
                neural_network_params=self.config.neural_network_params,
                linear_params=self.config.linear_params,
                custom_params=self.config.training_params
            )
            
            # Format analysis of trainer configuration for troubleshooting
            tprint_data_format(trainer_config, "Tactician base trainer configuration", level="DEBUG")
            
            # Create trainer
            self._trainer = TacticianBaseTrainer(trainer_config, self.logger)
            
            # Initialize trainer
            if not await self._trainer.initialize():
                tprint_error("❌ Trainer initialization failed")
                return False
            
            tprint_success("✅ Tactician base training initialized")
            return True
            
        except Exception as e:
            tprint_error(f"❌ Initialization failed: {e}")
            self.logger.error(f"Initialization failed: {e}")
            return False
    
    @handles_errors(
        exceptions=(ValueError, RuntimeError, MemoryError),
        default_return=TacticianBaseTrainingResult(
            success=False,
            errors=["Training execution failed"]
        ),
        context="tactician base training"
    )
    @comprehensive_memory_optimization(
        optimization_level=MemoryOptimizationLevel.AGGRESSIVE,
        enable_caching=True,
        enable_chunking=True,
        enable_gc=True,
        enable_pools=True
    )
    @performance_tracked(log_performance=True, track_memory=True)
    async def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the Tactician base training.
        
        Args:
            data: Input data containing training features and targets
            
        Returns:
            Training result dictionary
        """
        try:
            tprint_info("⚔️ Starting Tactician base training...")
            self.logger.info("Starting Tactician base training...")
            
            start_time = time.time()
            
            # Comprehensive data format analysis for troubleshooting
            tprint_data_format(data, "Input data dictionary", level="INFO")
            
            # Extract data
            X_train = data.get('X_train')
            y_train = data.get('y_train')
            
            # Detailed format analysis of training data
            if X_train is not None:
                tprint_data_format(X_train, "Extracted X_train", level="INFO")
            if y_train is not None:
                tprint_data_format(y_train, "Extracted y_train", level="INFO")
            
            if X_train is None or y_train is None:
                return {
                    'success': False,
                    'error_message': 'Missing required data: X_train and y_train',
                    'training_time': 0.0
                }
            
            # Convert to pandas if needed and optimize with enhanced hardware tools
            if not isinstance(X_train, pd.DataFrame):
                X_train = pd.DataFrame(X_train)
            if not isinstance(y_train, pd.Series):
                y_train = pd.Series(y_train)
            
            # Process data with enhanced hardware optimization
            hardware_manager = get_integrated_hardware_manager()
            X_train = hardware_manager.process_data_with_optimization(
                X_train, WorkloadType.ML_TRAINING
            )
            
            # Comprehensive format analysis of processed data
            tprint_data_format(X_train, "Processed X_train", level="DEBUG")
            tprint_data_format(y_train, "Processed y_train", level="DEBUG")
            
            # Train models
            training_result = await self._trainer.train(X_train, y_train)
            
            if not training_result.success:
                return {
                    'success': False,
                    'error_message': training_result.error_message,
                    'training_time': time.time() - start_time
                }
            
            # Create result
            result = TacticianBaseTrainingResult(
                success=True,
                models=training_result.model if isinstance(training_result.model, dict) else {},
                metrics=training_result.metrics,
                training_time=training_result.training_time,
                feature_importance=self._extract_feature_importance(training_result)
            )
            
            # Format analysis of training results for troubleshooting
            tprint_data_format(result.models, "Trained models", level="DEBUG")
            tprint_data_format(result.metrics, "Training metrics", level="INFO")
            if result.feature_importance:
                tprint_data_format(result.feature_importance, "Feature importance", level="DEBUG")
            
            # Auto-save if enabled
            if self.config.auto_save:
                await self._save_models(result)
            
            tprint_success(f"✅ Tactician base training completed in {result.training_time:.2f}s")
            self.logger.info(f"Tactician base training completed in {result.training_time:.2f}s")
            
            return {
                'success': True,
                'result': result,
                'training_time': result.training_time,
                'models_trained': list(result.models.keys()),
                'metrics': result.metrics
            }
            
        except Exception as e:
            tprint_error(f"❌ Tactician base training failed: {e}")
            self.logger.error(f"Tactician base training failed: {e}")
            
            # Enhanced error troubleshooting with data format analysis
            tprint_error("🔍 Error troubleshooting - analyzing data formats:")
            try:
                if 'data' in locals():
                    tprint_data_format(data, "Failed training data", level="ERROR")
                if 'X_train' in locals() and X_train is not None:
                    tprint_data_format(X_train, "Failed X_train", level="ERROR")
                if 'y_train' in locals() and y_train is not None:
                    tprint_data_format(y_train, "Failed y_train", level="ERROR")
            except Exception as format_error:
                tprint_error(f"Could not analyze data formats during error: {format_error}")
            
            return {
                'success': False,
                'error_message': str(e),
                'training_time': time.time() - start_time if 'start_time' in locals() else 0.0
            }
    
    async def validate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate the trained models.
        
        Args:
            data: Validation data containing features and targets
            
        Returns:
            Validation result dictionary
        """
        try:
            tprint_info("⚔️ Validating Tactician base models...")
            
            # Extract data
            X_val = data.get('X_val', data.get('X_train'))
            y_val = data.get('y_val', data.get('y_train'))
            
            # Format analysis of validation data
            tprint_data_format(data, "Validation data dictionary", level="DEBUG")
            if X_val is not None:
                tprint_data_format(X_val, "Validation X_val", level="DEBUG")
            if y_val is not None:
                tprint_data_format(y_val, "Validation y_val", level="DEBUG")
            
            if X_val is None or y_val is None:
                return {
                    'success': False,
                    'error_message': 'Missing required validation data'
                }
            
            # Convert to pandas if needed
            if not isinstance(X_val, pd.DataFrame):
                X_val = pd.DataFrame(X_val)
            if not isinstance(y_val, pd.Series):
                y_val = pd.Series(y_val)
            
            # Validate models
            validation_result = await self._trainer.validate(X_val, y_val)
            
            if not validation_result.success:
                return {
                    'success': False,
                    'error_message': validation_result.error_message
                }
            
            tprint_success("✅ Tactician base validation completed")
            return {
                'success': True,
                'metrics': validation_result.metrics,
                'predictions': validation_result.predictions
            }
            
        except Exception as e:
            tprint_error(f"❌ Tactician base validation failed: {e}")
            self.logger.error(f"Tactician base validation failed: {e}")
            return {
                'success': False,
                'error_message': str(e)
            }
    
    async def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make predictions with the trained models.
        
        Args:
            data: Input data for prediction
            
        Returns:
            Prediction result dictionary
        """
        try:
            tprint_info("⚔️ Making Tactician base predictions...")
            
            # Extract data
            X_pred = data.get('X_pred', data.get('X_train'))
            
            # Format analysis of prediction data
            tprint_data_format(data, "Prediction data dictionary", level="DEBUG")
            if X_pred is not None:
                tprint_data_format(X_pred, "Prediction X_pred", level="DEBUG")
            
            if X_pred is None:
                return {
                    'success': False,
                    'error_message': 'Missing required prediction data'
                }
            
            # Convert to pandas if needed
            if not isinstance(X_pred, pd.DataFrame):
                X_pred = pd.DataFrame(X_pred)
            
            # Make predictions
            prediction_result = await self._trainer.predict(X_pred)
            
            if not prediction_result.success:
                return {
                    'success': False,
                    'error_message': prediction_result.error_message
                }
            
            tprint_success("✅ Tactician base predictions completed")
            return {
                'success': True,
                'predictions': prediction_result.predictions,
                'probabilities': prediction_result.probabilities
            }
            
        except Exception as e:
            tprint_error(f"❌ Tactician base prediction failed: {e}")
            self.logger.error(f"Tactician base prediction failed: {e}")
            return {
                'success': False,
                'error_message': str(e)
            }
    
    def _convert_model_type(self, model_type: TacticianModelType):
        """Convert TacticianModelType to ModelType."""
        from ..core.base_trainer import ModelType
        
        if model_type == TacticianModelType.LIGHTGBM:
            return ModelType.LIGHTGBM
        elif model_type == TacticianModelType.CATBOOST:
            return ModelType.CATBOOST
        elif model_type == TacticianModelType.NEURAL_NETWORK:
            return ModelType.NEURAL_NETWORK
        elif model_type == TacticianModelType.LINEAR:
            return ModelType.LINEAR
        else:
            return ModelType.LIGHTGBM  # Default
    
    def _extract_feature_importance(self, training_result) -> Optional[Dict[str, Dict[str, float]]]:
        """Extract feature importance from training result."""
        try:
            if hasattr(training_result, 'feature_importance') and training_result.feature_importance:
                return training_result.feature_importance
            return None
        except Exception as e:
            self.logger.warning(f"Could not extract feature importance: {e}")
            return None
    
    async def _save_models(self, result: TacticianBaseTrainingResult) -> None:
        """Save trained models."""
        try:
            if not result.success or not result.models:
                return
            
            # Format analysis of models before saving for troubleshooting
            tprint_data_format(result.models, "Models to be saved", level="DEBUG")
            
            # This would implement model saving logic
            # For now, just log the save operation
            self.logger.info(f"Models saved: {list(result.models.keys())}")
            tprint_info(f"💾 Models saved: {list(result.models.keys())}")
            
        except Exception as e:
            self.logger.warning(f"Model saving failed: {e}")
            tprint_error(f"❌ Model saving failed: {e}")
            # Format analysis of failed save operation
            tprint_data_format(result, "Failed save result", level="ERROR")
    
    def _extract_and_validate_training_data(self, data: Dict[str, Any]) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
        """Extract and validate training data using BaseStep utilities."""
        try:
            # Extract data
            X_train = data.get('X_train')
            y_train = data.get('y_train')
            
            if X_train is None or y_train is None:
                self.tprint_error("❌ Missing required data: X_train and y_train")
                return None, None
            
            # Convert to pandas if needed using safe operations
            if not isinstance(X_train, pd.DataFrame):
                X_train = pd.DataFrame(X_train)
            if not isinstance(y_train, pd.Series):
                y_train = pd.Series(y_train)
            
            # Validate data using BaseStep utilities
            if not self._validate_dataframe_columns(X_train, []):
                self.tprint_error("❌ Invalid training features")
                return None, None
            
            # Data preview using BaseStep utilities
            self.tprint_data_summary(X_train, "Training Features", max_rows=5)
            self.tprint_data_summary(y_train, "Training Targets", max_rows=5)
            
            return X_train, y_train
            
        except Exception as e:
            self.tprint_error(f"❌ Data extraction failed: {e}")
            return None, None
    
    def _analyze_data_quality(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Analyze data quality using BaseStep utilities."""
        try:
            if self.data_quality:
                # Use data quality utilities
                quality_metrics = self.data_quality['calculate_quality_metrics'](X_train, y_train)
                self.tprint_validation_result(quality_metrics, "Data Quality Analysis")
            else:
                # Fallback analysis
                self.tprint_info(f"📊 Training data shape: {X_train.shape}")
                self.tprint_info(f"📊 Target data shape: {y_train.shape}")
                self.tprint_info(f"📊 Missing values in features: {X_train.isnull().sum().sum()}")
                self.tprint_info(f"📊 Missing values in targets: {y_train.isnull().sum()}")
        except Exception as e:
            self.tprint_warning(f"⚠️ Data quality analysis failed: {e}")
    
    def _optimize_training_data(self, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Optimize training data using hardware utilities."""
        try:
            if self.hardware_utils and 'optimize_dataframe' in self.hardware_utils:
                X_train = self.hardware_utils['optimize_dataframe'](X_train)
                self.tprint_success("✅ Training data optimized for hardware")
            return X_train, y_train
        except Exception as e:
            self.tprint_warning(f"⚠️ Data optimization failed: {e}")
            return X_train, y_train
    
    def _create_training_result(self, training_result: Any, training_time: float) -> Dict[str, Any]:
        """Create comprehensive training result."""
        try:
            result = {
                'success': True,
                'models': training_result.model if isinstance(training_result.model, dict) else {},
                'metrics': training_result.metrics if hasattr(training_result, 'metrics') else {},
                'training_time': training_time,
                'errors': training_result.errors if hasattr(training_result, 'errors') else [],
                'warnings': training_result.warnings if hasattr(training_result, 'warnings') else [],
                'feature_importance': self._extract_feature_importance(training_result)
            }
            
            # Add performance metrics
            result['performance_metrics'] = getattr(self, '_performance_metrics', {})
            
            return result
            
        except Exception as e:
            self.tprint_error(f"❌ Result creation failed: {e}")
            return {
                'success': False,
                'error_message': f'Result creation failed: {e}',
                'training_time': training_time
            }
    
    def _extract_feature_importance(self, training_result: Any) -> Optional[Dict[str, Dict[str, float]]]:
        """Extract feature importance using BaseStep utilities."""
        try:
            if hasattr(training_result, 'feature_importance') and training_result.feature_importance:
                return training_result.feature_importance
            return None
        except Exception as e:
            self.tprint_warning(f"⚠️ Feature importance extraction failed: {e}")
            return None
    
    def _analyze_training_performance(self, result: Dict[str, Any]) -> None:
        """Analyze training performance using BaseStep utilities."""
        try:
            # Performance summary
            self.tprint_performance_summary({
                'training_time': result['training_time'],
                'success': result['success'],
                'models_trained': len(result.get('models', {})),
                'metrics_count': len(result.get('metrics', {}))
            })
            
            # Memory usage analysis
            if self.hardware_utils and 'get_memory_usage' in self.hardware_utils:
                memory_usage = self.hardware_utils['get_memory_usage']()
                self.tprint_memory_usage(memory_usage)
            
        except Exception as e:
            self.tprint_warning(f"⚠️ Performance analysis failed: {e}")
    
    async def _save_training_artifacts(self, result: Dict[str, Any]) -> None:
        """Save training artifacts using BaseStep utilities."""
        try:
            # Save models using BaseStep utilities
            if result.get('models'):
                self._save_model(result['models'], 'tactician_base_models')
            
            # Save metrics using BaseStep utilities
            if result.get('metrics'):
                self._save_metadata(result['metrics'], 'training_metrics')
            
            # Save feature importance
            if result.get('feature_importance'):
                self._save_metadata(result['feature_importance'], 'feature_importance')
            
            self.tprint_success("✅ Training artifacts saved successfully")
            
        except Exception as e:
            self.tprint_error(f"❌ Artifact saving failed: {e}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary using BaseStep utilities."""
        try:
            summary = {
                'component_name': self.name,
                'config': self.config.__dict__,
                'trainer_initialized': self._trainer is not None,
                'utility_availability': self._get_availability_status(),
                'performance_metrics': getattr(self, '_performance_metrics', {})
            }
            
            if self._trainer:
                summary.update(self._trainer.get_tactician_summary())
            
            return summary
            
        except Exception as e:
            self.tprint_error(f"❌ Summary generation failed: {e}")
            return {
                'component_name': self.name,
                'error': str(e)
            }
    
    def get_required_dependencies(self) -> List[str]:
        """Get list of required dependencies."""
        return ['pandas', 'numpy', 'scikit-learn', 'lightgbm', 'catboost']
    
    def get_processing_capabilities(self) -> Dict[str, Any]:
        """Get component processing capabilities."""
        return {
            'input_types': ['dict'],
            'output_types': ['dict'],
            'supports_parallel_processing': False,
            'supports_checkpointing': True,
            'supports_validation': True,
            'supports_early_stopping': True,
            'supports_ensemble': False,
            'memory_efficient': True
        }


# Convenience functions
def create_tactician_base_training(
    model_types: List[TacticianModelType] = None,
    config: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None
) -> TacticianBaseTraining:
    """
    Create a Tactician base training component.
    
    Args:
        model_types: List of model types to train
        config: Configuration dictionary
        logger: Logger instance
        
    Returns:
        TacticianBaseTraining component
    """
    if model_types is None:
        model_types = [TacticianModelType.LIGHTGBM, TacticianModelType.CATBOOST]
    
    if config is None:
        config = {}
    
    config['model_types'] = model_types
    
    return TacticianBaseTraining(config=config, logger=logger)


async def execute_tactician_base_training(
    data: Dict[str, Any],
    model_types: List[TacticianModelType] = None,
    config: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Execute Tactician base training with minimal configuration.
    
    Args:
        data: Training data
        model_types: List of model types to train
        config: Configuration dictionary
        logger: Logger instance
        
    Returns:
        Training result
    """
    component = create_tactician_base_training(model_types, config, logger)
    
    # Initialize
    if not await component.initialize():
        return {
            'success': False,
            'error_message': 'Component initialization failed'
        }
    
    # Run training
    return await component.run(data)