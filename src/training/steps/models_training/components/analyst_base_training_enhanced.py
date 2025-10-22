"""
Analyst Base Training - Enhanced with Comprehensive BaseStep Utilities

This module provides the Analyst base training component that handles training
of individual Analyst base models using the unified BaseTrainer architecture
with comprehensive BaseStep utility integration.

Key Features:
- Unified training interface for all Analyst model types
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

from ..core.analyst_base_trainer import (
    AnalystBaseTrainer, AnalystTrainingConfig, AnalystModelType
)
from src.training.steps.base_step import BaseStep
from src.core.decorators import handles_errors, traced, log_execution_time


@dataclass
class AnalystBaseTrainingConfig:
    """Configuration for Analyst base training."""
    model_types: List[AnalystModelType]
    training_params: Dict[str, Any] = field(default_factory=dict)
    validation_params: Dict[str, Any] = field(default_factory=dict)
    timeframe: str = "15m"
    symbol: str = "ETHUSDT"
    auto_save: bool = True
    
    # Feature engineering parameters
    enable_patchtst_features: bool = True
    enable_regime_features: bool = True
    enable_multi_timeframe: bool = True
    
    # Model-specific parameters
    lightgbm_params: Dict[str, Any] = field(default_factory=dict)
    catboost_params: Dict[str, Any] = field(default_factory=dict)
    stacker_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalystBaseTrainingResult:
    """Result of Analyst base training."""
    success: bool
    models: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    training_time: float = 0.0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    feature_importance: Optional[Dict[str, Dict[str, float]]] = None


class AnalystBaseTraining(BaseStep):
    """
    Analyst base training component using unified BaseTrainer architecture
    with comprehensive BaseStep utility integration.
    
    This component demonstrates the full power of BaseStep utilities including:
    - Comprehensive logging and data visualization
    - Hardware optimization and memory management
    - Data quality validation and cleaning
    - Model persistence and caching
    - Safe operations with fallbacks
    - Performance monitoring and analytics
    """
    
    def __init__(
        self,
        name: str = "analyst_base_training",
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the Analyst base training component with comprehensive utilities.
        
        Args:
            name: Component name
            config: Configuration dictionary
            logger: Logger instance
        """
        super().__init__(name, config)
        
        # Set default configuration using BaseStep utilities
        default_config = {
            'model_types': [AnalystModelType.LIGHTGBM, AnalystModelType.CATBOOST],
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
            'enable_patchtst_features': True,
            'enable_regime_features': True,
            'enable_multi_timeframe': True
        }
        
        # Merge with provided configuration using safe operations
        if config:
            default_config = self._safe_merge_configs(default_config, config)
        
        self.config = AnalystBaseTrainingConfig(**default_config)
        
        # Initialize trainer
        self._trainer = None
        
        # Log initialization with comprehensive utilities
        self.tprint_banner("Analyst Base Training Component")
        self.tprint_info(f"🔧 Initialized AnalystBaseTraining: {name}")
        self.tprint_config_preview(self.config.__dict__, "Analyst Base Training Config")
        
        # Log utility availability status
        self._log_utility_availability()
        
        # Initialize performance tracking
        self._performance_metrics = {}
        
        self.logger.info(f"Initialized AnalystBaseTraining: {name}")
    
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
        default_return=AnalystBaseTrainingResult(
            success=False,
            errors=["Component initialization failed"]
        ),
        context="analyst base training"
    )
    async def initialize(self) -> bool:
        """Initialize the component with comprehensive utility integration."""
        try:
            self.tprint_step_start("Component Initialization")
            
            # Log hardware optimization status
            if self.hardware_utils:
                self.tprint_success("✅ Hardware optimization available")
                # Get hardware stats
                hw_stats = self._get_hardware_stats()
                self.tprint_hardware_stats(hw_stats)
            else:
                self.tprint_warning("⚠️ Hardware optimization not available")
            
            # Create trainer configuration using safe operations
            trainer_config = AnalystTrainingConfig(
                model_types=[self._convert_model_type(mt) for mt in self.config.model_types],
                timeframe=self.config.timeframe,
                symbol=self.config.symbol,
                training_params=self.config.training_params,
                validation_params=self.config.validation_params,
                lightgbm_params=self.config.lightgbm_params,
                catboost_params=self.config.catboost_params,
                stacker_params=self.config.stacker_params
            )
            
            # Validate configuration using BaseStep utilities
            if not self._validate_training_config(trainer_config):
                return False
            
            # Create trainer
            self._trainer = AnalystBaseTrainer(trainer_config, self.logger)
            
            # Initialize trainer
            if not await self._trainer.initialize():
                self.tprint_error("❌ Trainer initialization failed")
                return False
            
            # Log successful initialization
            self.tprint_step_end("Component Initialization", success=True)
            self.tprint_success("✅ Analyst base training component initialized successfully")
            
            return True
            
        except Exception as e:
            self.tprint_error(f"❌ Initialization failed: {e}")
            self.tprint_step_end("Component Initialization", success=False)
            return False
    
    def _validate_training_config(self, config: AnalystTrainingConfig) -> bool:
        """Validate training configuration using BaseStep utilities."""
        try:
            # Validate model types
            if not config.model_types:
                self.tprint_error("❌ No model types specified")
                return False
            
            # Validate timeframe using safe operations
            if not self._validate_finite(len(config.timeframe), default=0) > 0:
                self.tprint_error("❌ Invalid timeframe")
                return False
            
            # Validate symbol
            if not config.symbol or not isinstance(config.symbol, str):
                self.tprint_error("❌ Invalid symbol")
                return False
            
            # Validate training parameters
            validation_split = config.training_params.get('validation_split', 0.2)
            if not self._validate_range(validation_split, 0.0, 1.0):
                self.tprint_error(f"❌ Invalid validation split: {validation_split}")
                return False
            
            self.tprint_success("✅ Configuration validation passed")
            return True
            
        except Exception as e:
            self.tprint_error(f"❌ Configuration validation failed: {e}")
            return False
    
    def _validate_range(self, value: float, min_val: float, max_val: float) -> bool:
        """Validate value is within range using BaseStep utilities."""
        if self.math_validation and 'validate_range' in self.math_validation:
            return self.math_validation['validate_range'](value, min_val, max_val)
        else:
            return min_val <= value <= max_val
    
    def _get_hardware_stats(self) -> Dict[str, Any]:
        """Get hardware statistics using BaseStep utilities."""
        if self.hardware_utils and 'get_hardware_stats' in self.hardware_utils:
            return self.hardware_utils['get_hardware_stats']()
        else:
            return {'status': 'unavailable'}
    
    @handles_errors(
        exceptions=(ValueError, RuntimeError, MemoryError, KeyError),
        default_return={'success': False, 'error_message': 'Training failed'},
        context="analyst base training"
    )
    async def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the Analyst base training with comprehensive utility integration.
        
        Args:
            data: Input data containing training features and targets
            
        Returns:
            Training result dictionary
        """
        try:
            self.tprint_step_start("Analyst Base Training")
            self.tprint_info("📊 Starting Analyst base training...")
            
            start_time = time.time()
            
            # Extract and validate data using BaseStep utilities
            X_train, y_train = self._extract_and_validate_training_data(data)
            if X_train is None or y_train is None:
                return {
                    'success': False,
                    'error_message': 'Data extraction/validation failed',
                    'training_time': 0.0
                }
            
            # Data quality analysis using BaseStep utilities
            self._analyze_data_quality(X_train, y_train)
            
            # Hardware optimization if available
            if self.hardware_utils:
                X_train, y_train = self._optimize_training_data(X_train, y_train)
            
            # Train models
            training_result = await self._trainer.train(X_train, y_train)
            
            if not training_result.success:
                self.tprint_error(f"❌ Training failed: {training_result.error_message}")
                return {
                    'success': False,
                    'error_message': training_result.error_message,
                    'training_time': time.time() - start_time
                }
            
            # Create result with comprehensive metrics
            result = self._create_training_result(training_result, time.time() - start_time)
            
            # Performance analysis using BaseStep utilities
            self._analyze_training_performance(result)
            
            # Auto-save if enabled using BaseStep utilities
            if self.config.auto_save:
                await self._save_training_artifacts(result)
            
            self.tprint_step_end("Analyst Base Training", success=True)
            self.tprint_success(f"✅ Analyst base training completed in {result.training_time:.2f}s")
            
            return result
            
        except Exception as e:
            self.tprint_error(f"❌ Training failed: {e}")
            self.tprint_step_end("Analyst Base Training", success=False)
            return {
                'success': False,
                'error_message': str(e),
                'training_time': time.time() - start_time if 'start_time' in locals() else 0.0
            }
    
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
            result['performance_metrics'] = self._performance_metrics
            
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
                self._save_model(result['models'], 'analyst_base_models')
            
            # Save metrics using BaseStep utilities
            if result.get('metrics'):
                self._save_metadata(result['metrics'], 'training_metrics')
            
            # Save feature importance
            if result.get('feature_importance'):
                self._save_metadata(result['feature_importance'], 'feature_importance')
            
            self.tprint_success("✅ Training artifacts saved successfully")
            
        except Exception as e:
            self.tprint_error(f"❌ Artifact saving failed: {e}")
    
    def _convert_model_type(self, model_type: AnalystModelType) -> AnalystModelType:
        """Convert model type with validation."""
        try:
            if isinstance(model_type, str):
                return AnalystModelType(model_type.upper())
            return model_type
        except (ValueError, AttributeError):
            self.tprint_warning(f"⚠️ Invalid model type: {model_type}, using LIGHTGBM")
            return AnalystModelType.LIGHTGBM
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary using BaseStep utilities."""
        try:
            summary = {
                'component_name': self.name,
                'config': self.config.__dict__,
                'trainer_initialized': self._trainer is not None,
                'utility_availability': self._get_availability_status(),
                'performance_metrics': self._performance_metrics
            }
            
            if self._trainer:
                summary.update(self._trainer.get_analyst_summary())
            
            return summary
            
        except Exception as e:
            self.tprint_error(f"❌ Summary generation failed: {e}")
            return {
                'component_name': self.name,
                'error': str(e)
            }


# Factory functions for backward compatibility
def create_analyst_base_training(
    model_types: List[AnalystModelType] = None,
    config: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None
) -> AnalystBaseTraining:
    """Create Analyst base training component with comprehensive utilities."""
    if model_types is None:
        model_types = [AnalystModelType.LIGHTGBM, AnalystModelType.CATBOOST]
    
    component_config = config or {}
    component_config['model_types'] = model_types
    
    return AnalystBaseTraining(
        name="analyst_base_training",
        config=component_config,
        logger=logger
    )


async def execute_analyst_base_training(
    data: Dict[str, Any],
    model_types: List[AnalystModelType] = None,
    config: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Execute Analyst base training with comprehensive utility integration.
    
    Args:
        data: Training data
        model_types: List of model types to train
        config: Configuration dictionary
        logger: Logger instance
        
    Returns:
        Training result
    """
    component = create_analyst_base_training(model_types, config, logger)
    
    # Initialize
    if not await component.initialize():
        return {
            'success': False,
            'error_message': 'Component initialization failed'
        }
    
    # Run training
    return await component.run(data)