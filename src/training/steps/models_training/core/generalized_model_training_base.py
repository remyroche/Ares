"""
Generalized Model Training Base - Comprehensive Tools Integration

This module provides a generalized base class for all model training components
that inherits from BaseStep and provides comprehensive access to all utility tools,
hardware optimization, and advanced logging capabilities.

Key Features:
- Full BaseStep comprehensive tools integration
- Model-specific training utilities and patterns
- Hardware optimization for ML workloads
- Advanced logging and performance monitoring
- Unified configuration and validation
- Error handling and recovery mechanisms
- Artifact management and persistence
- Memory optimization and cleanup
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union, Type
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import pandas as pd
import numpy as np

from src.training.steps.base_step import BaseStep
from src.utils.logger import system_logger
from src.utils.tprint import (
    tprint, tprint_info, tprint_warning, tprint_error, tprint_success, 
    tprint_debug, tprint_performance, tprint_data_format, LogLevel,
    tprint_banner, tprint_separator, tprint_header, tprint_footer,
    tprint_step_start, tprint_step_end, tprint_operation_start, tprint_operation_end,
    tprint_data_summary, tprint_config_preview, tprint_validation_result,
    tprint_performance_summary, tprint_memory_usage, tprint_hardware_stats,
    tprint_dict, tprint_list, tprint_dataframe_info, tprint_model_info,
    tprint_artifact_info, tprint_execution_summary
)
from src.core.decorators import handles_errors, traced, log_execution_time
from src.utils.hardware.integrated_hardware_manager import (
    get_integrated_hardware_manager, IntegratedHardwareConfig, WorkloadType
)
from src.utils.hardware.unified_hardware_manager import (
    get_unified_hardware_manager, OptimizationLevel
)
from src.utils.hardware.optimization_decorators import (
    smart_cache, auto_optimize, memory_efficient, performance_tracked
)
from src.utils.hardware.memory_optimized_decorators import (
    memory_optimized, gc_optimized, comprehensive_memory_optimization,
    MemoryOptimizationLevel
)


class ModelTrainingRole(Enum):
    """Model training roles in the system."""
    ANALYST = "analyst"
    TACTICIAN = "tactician"
    ENSEMBLE = "ensemble"
    REGIME = "regime"
    CUSTOM = "custom"


class ModelType(Enum):
    """Types of ML models."""
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"
    NEURAL_NETWORK = "neural_network"
    ENSEMBLE = "ensemble"
    LINEAR = "linear"
    TREE_BASED = "tree_based"
    STACKER = "stacker"


@dataclass
class ModelTrainingConfig:
    """Comprehensive model training configuration."""
    # Core configuration
    role: ModelTrainingRole
    model_types: List[ModelType]
    timeframe: str = "15m"
    symbol: str = "ETHUSDT"
    exchange: str = "binance"
    
    # Training parameters
    validation_split: float = 0.2
    cross_validation_folds: int = 5
    random_seed: Optional[int] = None
    
    # Model-specific parameters
    enable_hyperparameter_optimization: bool = True
    enable_ensemble: bool = True
    enable_early_stopping: bool = True
    early_stopping_patience: int = 10
    
    # Performance parameters
    max_training_time: Optional[float] = None  # seconds
    memory_limit_mb: Optional[int] = None
    
    # Feature configuration
    feature_selection_method: str = "multi_objective"
    max_features: int = 100
    correlation_threshold: float = 0.85
    
    # Hardware optimization
    enable_hardware_optimization: bool = True
    enable_memory_optimization: bool = True
    enable_gpu_acceleration: bool = False
    
    # Logging and monitoring
    enable_detailed_logging: bool = True
    enable_performance_monitoring: bool = True
    enable_artifact_management: bool = True
    
    # Custom parameters
    custom_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelTrainingResult:
    """Result of model training operation."""
    success: bool
    models: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    training_time: float = 0.0
    validation_metrics: Dict[str, float] = field(default_factory=dict)
    feature_importance: Optional[Dict[str, Dict[str, float]]] = None
    artifacts: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class GeneralizedModelTrainingBase(BaseStep, ABC):
    """
    Generalized base class for all model training components.
    
    This class provides comprehensive access to all BaseStep utilities while
    adding model-specific training patterns and optimizations.
    
    Key Features:
    - Full BaseStep comprehensive tools integration
    - Model-specific training utilities
    - Hardware optimization for ML workloads
    - Advanced logging and performance monitoring
    - Unified configuration and validation
    - Error handling and recovery mechanisms
    - Artifact management and persistence
    """
    
    def __init__(
        self,
        step_name: str,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the generalized model training base.
        
        Args:
            step_name: Name of the training step
            config: Configuration dictionary
            logger: Logger instance (optional)
        """
        # Initialize BaseStep with comprehensive tools
        super().__init__(step_name, config)
        
        # Set up model training specific logger
        self.training_logger = logger or system_logger.getChild(f"{self.__class__.__name__}")
        
        # Model training configuration
        self.training_config = self._parse_training_config(config)
        
        # Training state management
        self._training_state = {
            'initialized': False,
            'training_started': False,
            'training_completed': False,
            'models_created': False,
            'best_models_saved': False,
            'current_epoch': 0,
            'best_epoch': 0,
            'best_metrics': {},
            'training_history': [],
            'validation_history': [],
            'model_checkpoints': []
        }
        
        # Performance tracking
        self._performance_metrics = {
            'training_time': 0.0,
            'validation_time': 0.0,
            'prediction_time': 0.0,
            'memory_usage_mb': 0.0,
            'cpu_usage_percent': 0.0,
            'gpu_usage_percent': 0.0
        }
        
        # Model state
        self._model_state = {
            'models': {},
            'best_models': {},
            'feature_importance': {},
            'training_artifacts': []
        }
        
        # Initialize comprehensive tools
        self._initialize_comprehensive_tools()
        
        tprint_banner(f"Generalized Model Training Base: {step_name}")
        tprint_info(f"🔧 Initialized {self.__class__.__name__} for {self.training_config.role.value}")
        self.training_logger.info(f"Initialized {self.__class__.__name__} for {self.training_config.role.value}")
    
    def _parse_training_config(self, config: Optional[Dict[str, Any]]) -> ModelTrainingConfig:
        """Parse and validate training configuration."""
        if not config:
            config = {}
        
        # Extract configuration with defaults
        training_config = ModelTrainingConfig(
            role=ModelTrainingRole(config.get('role', 'analyst')),
            model_types=[ModelType(mt) for mt in config.get('model_types', ['lightgbm'])],
            timeframe=config.get('timeframe', '15m'),
            symbol=config.get('symbol', 'ETHUSDT'),
            exchange=config.get('exchange', 'binance'),
            validation_split=config.get('validation_split', 0.2),
            cross_validation_folds=config.get('cross_validation_folds', 5),
            random_seed=config.get('random_seed'),
            enable_hyperparameter_optimization=config.get('enable_hyperparameter_optimization', True),
            enable_ensemble=config.get('enable_ensemble', True),
            enable_early_stopping=config.get('enable_early_stopping', True),
            early_stopping_patience=config.get('early_stopping_patience', 10),
            max_training_time=config.get('max_training_time'),
            memory_limit_mb=config.get('memory_limit_mb'),
            feature_selection_method=config.get('feature_selection_method', 'multi_objective'),
            max_features=config.get('max_features', 100),
            correlation_threshold=config.get('correlation_threshold', 0.85),
            enable_hardware_optimization=config.get('enable_hardware_optimization', True),
            enable_memory_optimization=config.get('enable_memory_optimization', True),
            enable_gpu_acceleration=config.get('enable_gpu_acceleration', False),
            enable_detailed_logging=config.get('enable_detailed_logging', True),
            enable_performance_monitoring=config.get('enable_performance_monitoring', True),
            enable_artifact_management=config.get('enable_artifact_management', True),
            custom_params=config.get('custom_params', {})
        )
        
        return training_config
    
    def _initialize_comprehensive_tools(self) -> None:
        """Initialize comprehensive tools from BaseStep."""
        try:
            tprint_info("🔧 Initializing comprehensive tools...")
            
            # Log utility availability
            self._log_utility_availability()
            
            # Initialize hardware optimizers if enabled
            if self.training_config.enable_hardware_optimization:
                self._initialize_hardware_optimizers()
            
            # Set up performance monitoring
            if self.training_config.enable_performance_monitoring:
                self._setup_performance_monitoring()
            
            # Initialize artifact management
            if self.training_config.enable_artifact_management:
                self._setup_artifact_management()
            
            self._training_state['initialized'] = True
            tprint_success("✅ Comprehensive tools initialized successfully")
            
        except Exception as e:
            tprint_error(f"❌ Failed to initialize comprehensive tools: {e}")
            self.training_logger.error(f"Failed to initialize comprehensive tools: {e}")
            raise
    
    def _initialize_hardware_optimizers(self) -> None:
        """Initialize hardware optimization tools."""
        try:
            tprint_debug("🔧 Initializing hardware optimizers...")
            
            # Use BaseStep hardware utilities
            if self.hardware_utils:
                # Initialize integrated hardware manager
                integrated_config = IntegratedHardwareConfig(
                    enable_automatic_optimization=True,
                    enable_caching=True,
                    enable_memory_monitoring=True,
                    enable_performance_tracking=True,
                    memory_limit_gb=self.training_config.memory_limit_mb / 1024.0 if self.training_config.memory_limit_mb else 8.0,
                    cache_memory_limit_mb=1024.0
                )
                
                # Store hardware managers for use
                self._integrated_hardware_manager = self.hardware_utils['get_integrated_hardware_manager'](integrated_config)
                self._unified_hardware_manager = get_unified_hardware_manager()
                
                tprint_success("✅ Hardware optimizers initialized")
            else:
                tprint_warning("⚠️ Hardware utilities not available, using fallbacks")
                
        except Exception as e:
            tprint_warning(f"⚠️ Hardware optimizer initialization failed: {e}")
            self.training_logger.warning(f"Hardware optimizer initialization failed: {e}")
    
    def _setup_performance_monitoring(self) -> None:
        """Set up performance monitoring."""
        try:
            tprint_debug("📊 Setting up performance monitoring...")
            
            # Use BaseStep performance monitoring utilities
            self._performance_start_time = time.time()
            self._memory_start = self._get_memory_usage()
            
            tprint_success("✅ Performance monitoring setup complete")
            
        except Exception as e:
            tprint_warning(f"⚠️ Performance monitoring setup failed: {e}")
            self.training_logger.warning(f"Performance monitoring setup failed: {e}")
    
    def _setup_artifact_management(self) -> None:
        """Set up artifact management."""
        try:
            tprint_debug("📦 Setting up artifact management...")
            
            # Use BaseStep artifact management utilities
            # Artifacts will be automatically managed through BaseStep methods
            self._artifact_prefix = f"{self.training_config.role.value}_{self.training_config.symbol}_{self.training_config.timeframe}"
            
            tprint_success("✅ Artifact management setup complete")
            
        except Exception as e:
            tprint_warning(f"⚠️ Artifact management setup failed: {e}")
            self.training_logger.warning(f"Artifact management setup failed: {e}")
    
    # ============================================================================
    # COMPREHENSIVE TOOLS ACCESS METHODS
    # ============================================================================
    
    def get_comprehensive_tools_status(self) -> Dict[str, Any]:
        """Get status of all comprehensive tools."""
        return {
            'utility_availability': self._get_availability_status(),
            'hardware_optimization': self.training_config.enable_hardware_optimization,
            'memory_optimization': self.training_config.enable_memory_optimization,
            'performance_monitoring': self.training_config.enable_performance_monitoring,
            'artifact_management': self.training_config.enable_artifact_management,
            'training_state': self._training_state.copy(),
            'performance_metrics': self._performance_metrics.copy()
        }
    
    def print_comprehensive_tools_help(self) -> None:
        """Print help for all available comprehensive tools."""
        self._print_utility_help()
    
    # ============================================================================
    # MODEL TRAINING SPECIFIC METHODS
    # ============================================================================
    
    @abstractmethod
    async def train_models(self, data: pd.DataFrame, targets: Optional[pd.Series] = None) -> ModelTrainingResult:
        """
        Train models with given data.
        
        Args:
            data: Training data
            targets: Target variables (optional, can be inferred from data)
            
        Returns:
            Model training result
        """
        pass
    
    @abstractmethod
    async def validate_models(self, data: pd.DataFrame, targets: Optional[pd.Series] = None) -> Dict[str, float]:
        """
        Validate trained models.
        
        Args:
            data: Validation data
            targets: Target variables (optional)
            
        Returns:
            Validation metrics
        """
        pass
    
    @abstractmethod
    async def predict(self, data: pd.DataFrame, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Make predictions with trained models.
        
        Args:
            data: Input data for prediction
            model_name: Specific model to use (optional)
            
        Returns:
            Prediction results
        """
        pass
    
    @abstractmethod
    def _create_model(self, model_type: ModelType) -> Any:
        """
        Create a model instance.
        
        Args:
            model_type: Type of model to create
            
        Returns:
            Model instance
        """
        pass
    
    # ============================================================================
    # ENHANCED DATA PROCESSING WITH COMPREHENSIVE TOOLS
    # ============================================================================
    
    @handles_errors(
        exceptions=(ValueError, AttributeError, RuntimeError),
        default_return=None,
        context="data preprocessing with comprehensive tools"
    )
    @comprehensive_memory_optimization(
        optimization_level=MemoryOptimizationLevel.AGGRESSIVE,
        enable_caching=True,
        enable_chunking=True,
        enable_gc=True,
        enable_pools=True
    )
    def preprocess_data_with_comprehensive_tools(
        self, 
        data: pd.DataFrame, 
        targets: Optional[pd.Series] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Preprocess data using comprehensive tools from BaseStep.
        
        Args:
            data: Input data
            targets: Target variables
            
        Returns:
            Preprocessed data and targets
        """
        try:
            tprint_operation_start("Data Preprocessing with Comprehensive Tools")
            
            # Use BaseStep data preview and validation
            tprint_data_preview(data, "Raw input data", max_rows=5, level="INFO")
            tprint_data_format(data, "Raw input data", level=LogLevel.DEBUG)
            
            # Calculate data quality metrics using BaseStep utilities
            if self.common_utils and 'calculate_data_quality_metrics' in self.common_utils:
                quality_metrics = self.common_utils['calculate_data_quality_metrics'](data)
                tprint_dict(quality_metrics, "Data Quality Metrics")
            else:
                # Fallback quality metrics
                quality_metrics = {
                    'shape': data.shape,
                    'missing_values': data.isnull().sum().sum(),
                    'infinite_values': np.isinf(data).sum().sum(),
                    'memory_usage_mb': data.memory_usage(deep=True).sum() / (1024 * 1024)
                }
                tprint_dict(quality_metrics, "Data Quality Metrics (Fallback)")
            
            # Handle missing values using BaseStep safe operations
            if data.isnull().any().any():
                tprint_warning("⚠️ Found missing values, filling with median")
                data = self._safe_dataframe_operation(data, "fillna", method="median")
            
            # Handle infinite values using BaseStep safe operations
            if np.isinf(data).any().any():
                tprint_warning("⚠️ Found infinite values, replacing with finite values")
                data = data.replace([np.inf, -np.inf], np.nan)
                data = self._safe_dataframe_operation(data, "fillna", method="median")
            
            # Optimize memory usage using BaseStep hardware utilities
            if self.training_config.enable_hardware_optimization and self.hardware_utils:
                data = self.hardware_utils['optimize_dataframe'](data)
                tprint_debug("🧠 Hardware optimization applied")
            
            # Feature selection using BaseStep utilities
            if self.training_config.max_features < len(data.columns):
                selected_features = self._select_features_with_comprehensive_tools(data, targets)
                data = data[selected_features]
                tprint_info(f"📊 Selected {len(selected_features)} features using comprehensive tools")
            
            # Extract targets if not provided
            if targets is None:
                targets = self._extract_targets_with_comprehensive_tools(data)
            
            # Validate targets using BaseStep utilities
            if targets is not None:
                targets = self._validate_finite(targets)
            
            # Final data preview
            tprint_data_preview(data, "Final preprocessed data", max_rows=5, level="INFO")
            if targets is not None:
                tprint_data_preview(targets, "Final preprocessed targets", max_rows=10, level="INFO")
            
            tprint_operation_end("Data Preprocessing with Comprehensive Tools", success=True)
            tprint_success(f"✅ Data preprocessed: {data.shape[0]} samples, {data.shape[1]} features")
            
            return data, targets
            
        except Exception as e:
            tprint_operation_end("Data Preprocessing with Comprehensive Tools", success=False)
            tprint_error(f"❌ Data preprocessing failed: {e}")
            self.training_logger.error(f"Data preprocessing failed: {e}")
            raise
    
    def _select_features_with_comprehensive_tools(
        self, 
        data: pd.DataFrame, 
        targets: Optional[pd.Series] = None
    ) -> List[str]:
        """Select features using comprehensive tools."""
        try:
            if self.training_config.feature_selection_method == "correlation":
                return self._select_features_by_correlation_with_tools(data)
            elif self.training_config.feature_selection_method == "variance":
                return self._select_features_by_variance_with_tools(data)
            elif self.training_config.feature_selection_method == "mutual_info":
                return self._select_features_by_mutual_info_with_tools(data, targets)
            else:
                # Default: use BaseStep utilities for correlation-based selection
                if targets is not None:
                    correlations = data.corrwith(targets).abs().sort_values(ascending=False)
                    return correlations.head(self.training_config.max_features).index.tolist()
                else:
                    return data.columns[:self.training_config.max_features].tolist()
                    
        except Exception as e:
            self.training_logger.warning(f"Feature selection failed, using all features: {e}")
            return data.columns.tolist()
    
    def _select_features_by_correlation_with_tools(self, data: pd.DataFrame) -> List[str]:
        """Select features by correlation using comprehensive tools."""
        # Use BaseStep safe operations for correlation calculation
        corr_matrix = data.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find features to drop using safe operations
        to_drop = [column for column in upper_tri.columns 
                  if any(upper_tri[column] > self.training_config.correlation_threshold)]
        
        # Select remaining features
        selected = [col for col in data.columns if col not in to_drop]
        return selected[:self.training_config.max_features]
    
    def _select_features_by_variance_with_tools(self, data: pd.DataFrame) -> List[str]:
        """Select features by variance using comprehensive tools."""
        # Use BaseStep safe operations for variance calculation
        variances = data.var().sort_values(ascending=False)
        return variances.head(self.training_config.max_features).index.tolist()
    
    def _select_features_by_mutual_info_with_tools(
        self, 
        data: pd.DataFrame, 
        targets: pd.Series
    ) -> List[str]:
        """Select features by mutual information using comprehensive tools."""
        try:
            if targets is None:
                return data.columns[:self.training_config.max_features].tolist()
            
            # Use BaseStep ML utilities if available
            if self.ml_common and 'mutual_info_regression' in self.ml_common:
                mi_scores = self.ml_common['mutual_info_regression'](data, targets)
            else:
                # Fallback to sklearn
                from sklearn.feature_selection import mutual_info_regression
                mi_scores = mutual_info_regression(data, targets)
            
            feature_scores = pd.Series(mi_scores, index=data.columns).sort_values(ascending=False)
            return feature_scores.head(self.training_config.max_features).index.tolist()
            
        except ImportError:
            self.training_logger.warning("sklearn not available, using correlation-based selection")
            return self._select_features_by_correlation_with_tools(data)
    
    def _extract_targets_with_comprehensive_tools(self, data: pd.DataFrame) -> pd.Series:
        """Extract targets using comprehensive tools."""
        target_columns = ['target', 'y', 'label', 'price_target', 'return_target']
        
        for col in target_columns:
            if col in data.columns:
                targets = data[col]
                data = data.drop(columns=[col])
                tprint_data_preview(targets, f"Extracted targets from {col}", max_rows=10, level="INFO")
                return targets
        
        raise ValueError("No target column found in data")
    
    # ============================================================================
    # ENHANCED MODEL MANAGEMENT WITH COMPREHENSIVE TOOLS
    # ============================================================================
    
    def save_models_with_comprehensive_tools(
        self, 
        models: Dict[str, Any], 
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """
        Save models using comprehensive tools from BaseStep.
        
        Args:
            models: Dictionary of models to save
            metadata: Additional metadata
            
        Returns:
            Dictionary of saved model paths
        """
        try:
            tprint_operation_start("Model Saving with Comprehensive Tools")
            
            saved_paths = {}
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            
            for model_name, model in models.items():
                # Create comprehensive metadata
                model_metadata = {
                    'model_name': model_name,
                    'model_type': type(model).__name__,
                    'timestamp': timestamp,
                    'training_config': self.training_config.__dict__,
                    'performance_metrics': self._performance_metrics.copy(),
                    'training_state': self._training_state.copy()
                }
                
                if metadata:
                    model_metadata.update(metadata)
                
                # Use BaseStep model saving utilities
                artifact_name = f"{self._artifact_prefix}_{model_name}_{timestamp}"
                model_path = self._save_model(model, artifact_name, model_metadata)
                saved_paths[model_name] = model_path
                
                # Log model information using BaseStep utilities
                tprint_model_info(model, f"Saved {model_name}")
            
            tprint_operation_end("Model Saving with Comprehensive Tools", success=True)
            tprint_success(f"✅ Saved {len(models)} models using comprehensive tools")
            
            return saved_paths
            
        except Exception as e:
            tprint_operation_end("Model Saving with Comprehensive Tools", success=False)
            tprint_error(f"❌ Model saving failed: {e}")
            self.training_logger.error(f"Model saving failed: {e}")
            raise
    
    def load_models_with_comprehensive_tools(self, model_paths: Dict[str, str]) -> Dict[str, Any]:
        """
        Load models using comprehensive tools from BaseStep.
        
        Args:
            model_paths: Dictionary of model paths to load
            
        Returns:
            Dictionary of loaded models
        """
        try:
            tprint_operation_start("Model Loading with Comprehensive Tools")
            
            loaded_models = {}
            
            for model_name, model_path in model_paths.items():
                # Use BaseStep model loading utilities
                model = self._load_model(model_path)
                loaded_models[model_name] = model
                
                # Log model information using BaseStep utilities
                tprint_model_info(model, f"Loaded {model_name}")
            
            tprint_operation_end("Model Loading with Comprehensive Tools", success=True)
            tprint_success(f"✅ Loaded {len(model_paths)} models using comprehensive tools")
            
            return loaded_models
            
        except Exception as e:
            tprint_operation_end("Model Loading with Comprehensive Tools", success=False)
            tprint_error(f"❌ Model loading failed: {e}")
            self.training_logger.error(f"Model loading failed: {e}")
            raise
    
    # ============================================================================
    # ENHANCED PERFORMANCE MONITORING WITH COMPREHENSIVE TOOLS
    # ============================================================================
    
    def get_comprehensive_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary using BaseStep utilities."""
        try:
            # Use BaseStep performance utilities
            base_performance = self._get_performance_metrics()
            memory_analytics = self._get_memory_analytics()
            comprehensive_stats = self._get_comprehensive_stats()
            
            # Combine with training-specific metrics
            performance_summary = {
                'base_performance': base_performance,
                'memory_analytics': memory_analytics,
                'comprehensive_stats': comprehensive_stats,
                'training_metrics': self._performance_metrics.copy(),
                'training_state': self._training_state.copy(),
                'model_state': self._model_state.copy(),
                'utility_availability': self._get_availability_status()
            }
            
            return performance_summary
            
        except Exception as e:
            tprint_error(f"❌ Failed to get comprehensive performance summary: {e}")
            self.training_logger.error(f"Failed to get comprehensive performance summary: {e}")
            return {}
    
    def log_comprehensive_training_summary(self) -> None:
        """Log comprehensive training summary using BaseStep utilities."""
        try:
            tprint_banner("Comprehensive Training Summary")
            
            # Log training configuration
            tprint_config_preview(self.training_config.__dict__, "Training Configuration")
            
            # Log performance metrics
            performance_summary = self.get_comprehensive_performance_summary()
            tprint_performance_summary(performance_summary['training_metrics'])
            
            # Log memory usage
            tprint_memory_usage(performance_summary['memory_analytics'])
            
            # Log hardware stats if available
            if self.hardware_utils:
                tprint_hardware_stats(performance_summary['comprehensive_stats'])
            
            # Log training state
            tprint_dict(self._training_state, "Training State")
            
            # Log model state
            tprint_dict(self._model_state, "Model State")
            
            # Log utility availability
            tprint_dict(performance_summary['utility_availability'], "Utility Availability")
            
            tprint_footer("End of Training Summary")
            
        except Exception as e:
            tprint_error(f"❌ Failed to log comprehensive training summary: {e}")
            self.training_logger.error(f"Failed to log comprehensive training summary: {e}")
    
    # ============================================================================
    # UTILITY METHODS
    # ============================================================================
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0.0
    
    def get_training_capabilities(self) -> Dict[str, Any]:
        """Get training capabilities."""
        return {
            'supports_parallel_processing': False,
            'supports_checkpointing': True,
            'supports_early_stopping': True,
            'supports_ensemble': self.training_config.enable_ensemble,
            'supports_hardware_optimization': self.training_config.enable_hardware_optimization,
            'supports_memory_optimization': self.training_config.enable_memory_optimization,
            'supports_performance_monitoring': self.training_config.enable_performance_monitoring,
            'supports_artifact_management': self.training_config.enable_artifact_management,
            'comprehensive_tools_available': sum(self._get_availability_status().values()),
            'total_utilities': len(self._get_availability_status())
        }
    
    def estimate_training_time(self, data_size: int) -> float:
        """Estimate training time for given data size."""
        base_time = 5.0  # seconds
        size_factor = data_size / 1000
        model_factor = len(self.training_config.model_types)
        complexity_factor = 1.0 if self.training_config.enable_hardware_optimization else 1.5
        
        return base_time * size_factor * model_factor * complexity_factor
    
    def get_memory_requirements(self, data_size: int) -> Dict[str, float]:
        """Get memory requirements for training."""
        base_memory = 200  # MB
        data_memory = data_size * 0.001  # Rough estimate
        model_memory = len(self.training_config.model_types) * 100  # MB per model
        optimization_overhead = 1.5 if self.training_config.enable_memory_optimization else 2.0
        
        estimated_memory = (base_memory + data_memory + model_memory) * optimization_overhead
        peak_memory = estimated_memory * 1.5  # 50% buffer
        
        return {
            'estimated_memory_mb': estimated_memory,
            'peak_memory_mb': peak_memory,
            'data_memory_mb': data_memory,
            'model_memory_mb': model_memory,
            'base_memory_mb': base_memory
        }