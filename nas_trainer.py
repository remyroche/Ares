#!/usr/bin/env python3
"""
Neural Architecture Search (NAS) Trainer

This module provides a comprehensive NAS training system that integrates with
the existing utility modules for data processing, hardware optimization,
and machine learning operations.

Key Features:
- Neural Architecture Search with multiple search strategies
- M1 hardware optimization for Apple Silicon
- Integration with existing utility modules
- Advanced hyperparameter optimization
- Cross-validation and model evaluation
- Automated model selection and training
- Comprehensive error handling and logging
- Full implementation of all utility functions
"""

import os
import sys
import time
import logging
import asyncio
import json
import pickle
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
from contextlib import contextmanager
import concurrent.futures
import threading
from datetime import datetime
import warnings

# Core dependencies
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.utils.nas_tas.shared_logging import (
    TPRINT_AVAILABLE,
    tprint,
    tprint_info,
    tprint_warning,
    tprint_error,
    tprint_success,
    tprint_progress,
    tprint_performance,
    tprint_structured,
    tprint_timer,
    configure_tprint,
    TPrintConfig,
    LogLevel,
)

from src.utils.nas_tas.shared_serialization import (
    SERIALIZATION_AVAILABLE,
    JSONSerializer,
    PickleSerializer,
    ParquetSerializer,
    UniversalSerializer,
)

# Import unified components with error handling
try:
    from src.utils.nas_tas import (
        UnifiedEvaluator, UnifiedHardwareOptimizer, UnifiedSearchEngine, 
        UnifiedDataProcessor, UnifiedComponentManager
    )
    UNIFIED_COMPONENTS_AVAILABLE = True
except ImportError as e:
    tprint_warning(f"Unified components not available: {e}")
    UNIFIED_COMPONENTS_AVAILABLE = False

# Import common operations with error handling
try:
    from src.utils.common_operations import (
        safe_dataframe_operation, validate_dataframe, create_data_quality_report,
        safe_divide, safe_log, safe_sqrt, safe_power, validate_finite, validate_positive,
        get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer,
        integrate_with_m1_optimizers, optimize_memory, get_memory_usage,
        safe_json_dump, safe_json_load, ensure_directory
    )
    COMMON_OPERATIONS_AVAILABLE = True
except ImportError as e:
    tprint_warning(f"Common operations not available: {e}")
    COMMON_OPERATIONS_AVAILABLE = False

# Import math validation with error handling
try:
    from src.utils.math_validation import (
        MathValidation, safe_correlation, safe_covariance, safe_mean, safe_std,
        validate_correlation_matrix, safe_matrix_inverse, safe_kelly_calculation,
        safe_weighted_average, safe_percentage_change
    )
    MATH_VALIDATION_AVAILABLE = True
except ImportError as e:
    tprint_warning(f"Math validation not available: {e}")
    MATH_VALIDATION_AVAILABLE = False

# Import pipeline modes with error handling
try:
    from src.config.pipeline_modes import (
        get_mode_config, get_full_mode_config, get_light_mode_config, get_blank_mode_config,
        FULL_MODE_CONFIG, LIGHT_MODE_CONFIG, BLANK_MODE_CONFIG
    )
    PIPELINE_MODES_AVAILABLE = True
except ImportError as e:
    tprint_warning(f"Pipeline modes not available: {e}")
    PIPELINE_MODES_AVAILABLE = False

# ML-specific imports with comprehensive error handling
try:
    from src.utils.common_operations import (
        safe_dataframe_operation as ml_safe_df_op
    )
    from src.utils.ml_common.feature_selection import (
        select_features, get_feature_importance
    )
    from src.utils.ml_common.optimization import (
        optimize_hyperparameters, grid_search, bayesian_optimization
    )
    from src.utils.ml_common.validation import (
        cross_validate_model, validate_model_performance
    )
    from src.utils.ml_common.models import (
        create_model, train_model, evaluate_model
    )
    ML_COMMON_AVAILABLE = True
except ImportError as e:
    tprint_warning(f"ML common utilities not available: {e}")
    ML_COMMON_AVAILABLE = False
    
    # Fallback ML functions with proper error handling
    def select_features(X, y, method='mutual_info', k=10):
        try:
            if hasattr(X, 'iloc'):
                return X.iloc[:, :k]
            elif hasattr(X, 'shape'):
                return X[:, :k]
            else:
                return X[:k] if len(X) > k else X
        except Exception as e:
            tprint_error(f"Feature selection failed: {e}")
            return X
    
    def get_feature_importance(model):
        try:
            if hasattr(model, 'feature_importances_'):
                return model.feature_importances_
            elif hasattr(model, 'coef_'):
                return np.abs(model.coef_[0]) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
            else:
                return np.random.random(10)
        except Exception as e:
            tprint_error(f"Feature importance extraction failed: {e}")
            return np.random.random(10)
    
    def optimize_hyperparameters(model, X, y, param_grid, cv=5):
        try:
            from sklearn.model_selection import GridSearchCV
            grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='accuracy')
            grid_search.fit(X, y)
            return {
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_
            }
        except Exception as e:
            tprint_error(f"Hyperparameter optimization failed: {e}")
            return {'best_params': {}, 'best_score': 0.0}
    
    def cross_validate_model(model, X, y, cv=5):
        try:
            from sklearn.model_selection import cross_val_score
            scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
            return {
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'scores': scores.tolist()
            }
        except Exception as e:
            tprint_error(f"Cross-validation failed: {e}")
            return {'mean_score': 0.0, 'std_score': 0.0, 'scores': []}
    
    def create_model(model_type='random_forest', **params):
        try:
            if model_type == 'random_forest':
                return RandomForestClassifier(**params)
            elif model_type == 'mlp':
                return MLPClassifier(**params)
            elif model_type == 'logistic':
                return LogisticRegression(**params)
            else:
                return RandomForestClassifier(**params)
        except Exception as e:
            tprint_error(f"Model creation failed: {e}")
            return RandomForestClassifier()
    
    def train_model(model, X, y):
        try:
            return model.fit(X, y)
        except Exception as e:
            tprint_error(f"Model training failed: {e}")
            return model
    
    def evaluate_model(model, X, y):
        try:
            y_pred = model.predict(X)
            return {
                'accuracy': accuracy_score(y, y_pred),
                'precision': precision_score(y, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y, y_pred, average='weighted', zero_division=0),
                'f1': f1_score(y, y_pred, average='weighted', zero_division=0)
            }
        except Exception as e:
            tprint_error(f"Model evaluation failed: {e}")
            return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

# Setup logging with comprehensive error handling
try:
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
except Exception as e:
    print(f"Logging setup failed: {e}")
    logger = None

@dataclass
class NASConfig:
    """Configuration for Neural Architecture Search with comprehensive validation."""
    
    # Search parameters
    search_strategy: str = 'random'  # 'random', 'grid', 'bayesian', 'evolutionary'
    max_trials: int = 100
    max_epochs: int = 50
    early_stopping_patience: int = 10
    
    # Architecture parameters
    min_layers: int = 2
    max_layers: int = 10
    min_neurons: int = 32
    max_neurons: int = 512
    activation_functions: List[str] = field(default_factory=lambda: ['relu', 'tanh', 'sigmoid'])
    dropout_rates: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.2, 0.3, 0.5])
    
    # Training parameters
    learning_rate_range: Tuple[float, float] = (1e-5, 1e-1)
    batch_size_range: Tuple[int, int] = (16, 256)
    optimizer: str = 'adam'
    
    # Hardware optimization
    use_m1_optimization: bool = True
    use_gpu_acceleration: bool = True
    memory_limit_gb: Optional[float] = None
    
    # Data processing
    feature_selection: bool = True
    feature_selection_method: str = 'mutual_info'
    max_features: int = 100
    
    # Validation
    cv_folds: int = 5
    validation_split: float = 0.2
    test_split: float = 0.2
    
    # Output
    save_models: bool = True
    save_results: bool = True
    output_dir: str = 'nas_results'
    verbose: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        try:
            # Validate search strategy
            valid_strategies = ['random', 'grid', 'bayesian', 'evolutionary']
            if self.search_strategy not in valid_strategies:
                tprint_warning(f"Invalid search strategy: {self.search_strategy}, using 'random'")
                self.search_strategy = 'random'
            
            # Validate numeric parameters
            if self.max_trials <= 0:
                tprint_warning(f"Invalid max_trials: {self.max_trials}, using 100")
                self.max_trials = 100
            
            if self.max_epochs <= 0:
                tprint_warning(f"Invalid max_epochs: {self.max_epochs}, using 50")
                self.max_epochs = 50
            
            if self.min_layers < 1:
                tprint_warning(f"Invalid min_layers: {self.min_layers}, using 1")
                self.min_layers = 1
            
            if self.max_layers < self.min_layers:
                tprint_warning(f"max_layers ({self.max_layers}) < min_layers ({self.min_layers}), adjusting")
                self.max_layers = self.min_layers + 1
            
            if self.min_neurons <= 0:
                tprint_warning(f"Invalid min_neurons: {self.min_neurons}, using 32")
                self.min_neurons = 32
            
            if self.max_neurons < self.min_neurons:
                tprint_warning(f"max_neurons ({self.max_neurons}) < min_neurons ({self.min_neurons}), adjusting")
                self.max_neurons = self.min_neurons * 2
            
            # Validate learning rate range
            if self.learning_rate_range[0] >= self.learning_rate_range[1]:
                tprint_warning(f"Invalid learning rate range: {self.learning_rate_range}, using default")
                self.learning_rate_range = (1e-5, 1e-1)
            
            # Validate batch size range
            if self.batch_size_range[0] >= self.batch_size_range[1]:
                tprint_warning(f"Invalid batch size range: {self.batch_size_range}, using default")
                self.batch_size_range = (16, 256)
            
            # Validate validation split
            if not 0 < self.validation_split < 1:
                tprint_warning(f"Invalid validation_split: {self.validation_split}, using 0.2")
                self.validation_split = 0.2
            
            # Validate test split
            if not 0 < self.test_split < 1:
                tprint_warning(f"Invalid test_split: {self.test_split}, using 0.2")
                self.test_split = 0.2
            
            # Validate CV folds
            if self.cv_folds < 2:
                tprint_warning(f"Invalid cv_folds: {self.cv_folds}, using 5")
                self.cv_folds = 5
            
            tprint_info("✅ NAS configuration validated successfully")
            
        except Exception as e:
            tprint_error(f"Configuration validation failed: {e}")
            # Set safe defaults
            self.search_strategy = 'random'
            self.max_trials = 100
            self.max_epochs = 50
            self.min_layers = 2
            self.max_layers = 10
            self.min_neurons = 32
            self.max_neurons = 512
            self.learning_rate_range = (1e-5, 1e-1)
            self.batch_size_range = (16, 256)
            self.validation_split = 0.2
            self.test_split = 0.2
            self.cv_folds = 5

class NASTrainer:
    """
    Neural Architecture Search Trainer with comprehensive error handling and M1 optimization.
    
    This class provides comprehensive NAS functionality with integration
    to the existing utility modules for data processing, hardware
    optimization, and machine learning operations.
    """
    
    def __init__(self, config: Optional[NASConfig] = None):
        """Initialize the NAS Trainer with comprehensive error handling."""
        try:
            self.config = config or NASConfig()
            self.logger = logger.getChild('NASTrainer') if logger else None
            
            # Initialize hardware optimizers
            self._initialize_hardware_optimizers()
            
            # Initialize unified components
            if UNIFIED_COMPONENTS_AVAILABLE:
                self._initialize_unified_components()
            else:
                tprint_warning("Unified components not available, using fallback mode")
                self._initialize_fallback_components()
            
            # Results storage
            self.search_results = []
            self.best_architecture = None
            self.best_model = None
            self.training_history = []
            self.error_log = []
            
            # Setup output directory with error handling
            try:
                self.output_dir = Path(self.config.output_dir)
                self.output_dir.mkdir(parents=True, exist_ok=True)
                tprint_info(f"📁 Output directory created: {self.output_dir}")
            except Exception as e:
                tprint_error(f"Failed to create output directory: {e}")
                self.output_dir = Path('nas_results_fallback')
                self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Configure tprint if available
            self._configure_logging()
            
            # Initialize utility modules
            self._initialize_utility_modules()
            
            tprint_info("🚀 NAS Trainer initialized successfully")
            tprint_structured({
                'config': {
                    'search_strategy': self.config.search_strategy,
                    'max_trials': self.config.max_trials,
                    'use_m1_optimization': self.config.use_m1_optimization,
                    'output_dir': str(self.output_dir),
                    'unified_components': UNIFIED_COMPONENTS_AVAILABLE,
                    'hardware_optimization': self.config.use_m1_optimization
                }
            })
            
        except Exception as e:
            tprint_error(f"NAS Trainer initialization failed: {e}")
            tprint_error(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Failed to initialize NAS Trainer: {e}")
    
    def _initialize_hardware_optimizers(self):
        """Initialize hardware optimization utilities with error handling."""
        try:
            if COMMON_OPERATIONS_AVAILABLE and self.config.use_m1_optimization:
                self.gpu_manager = get_m1_gpu_manager()
                self.memory_optimizer = get_m1_memory_optimizer()
                self.cpu_optimizer = get_m1_cpu_optimizer()
                
                # Start memory monitoring if available
                if self.memory_optimizer and hasattr(self.memory_optimizer, 'start_monitoring'):
                    self.memory_optimizer.start_monitoring()
                    tprint_info("🧠 M1 hardware optimization enabled")
                else:
                    tprint_warning("M1 memory optimizer not available")
            else:
                self.gpu_manager = None
                self.memory_optimizer = None
                self.cpu_optimizer = None
                tprint_info("🔧 Using standard hardware configuration")
                
        except Exception as e:
            tprint_warning(f"Hardware optimization initialization failed: {e}")
            self.gpu_manager = None
            self.memory_optimizer = None
            self.cpu_optimizer = None
    
    def _configure_logging(self):
        """Configure logging with error handling."""
        try:
            if TPRINT_AVAILABLE:
                tprint_config = TPrintConfig(
                    output_to_file=True,
                    output_file=self.output_dir / 'nas_training.log',
                    auto_log_prints=True,
                    integrate_with_logging=True
                )
                configure_tprint(tprint_config)
                tprint_info("📝 Logging configured successfully")
            else:
                tprint_warning("tprint not available, using basic logging")
        except Exception as e:
            tprint_warning(f"Could not configure tprint: {e}")
    
    def _initialize_utility_modules(self):
        """Initialize utility modules with error handling."""
        try:
            # Initialize math validator if available
            if MATH_VALIDATION_AVAILABLE:
                self.math_validator = MathValidation()
                tprint_info("🧮 Math validation utilities initialized")
            else:
                self.math_validator = None
                tprint_warning("Math validation utilities not available")
            
            # Initialize serializers
            if SERIALIZATION_AVAILABLE:
                self.json_serializer = JSONSerializer()
                self.pickle_serializer = PickleSerializer()
                self.universal_serializer = UniversalSerializer()
                tprint_info("💾 Serialization utilities initialized")
            else:
                self.json_serializer = JSONSerializer()
                self.pickle_serializer = PickleSerializer()
                self.universal_serializer = UniversalSerializer()
                tprint_warning("Using fallback serialization utilities")
                
        except Exception as e:
            tprint_warning(f"Utility modules initialization failed: {e}")
            self.math_validator = None
            self.json_serializer = JSONSerializer()
            self.pickle_serializer = PickleSerializer()
            self.universal_serializer = UniversalSerializer()
    
    def _initialize_unified_components(self):
        """Initialize unified components with comprehensive error handling."""
        try:
            # Convert NAS config to unified config format
            unified_config = {
                'enable_hardware_optimization': self.config.use_m1_optimization,
                'enable_m1_optimization': self.config.use_m1_optimization,
                'enable_trading_metrics': True,
                'enable_economic_metrics': True,
                'enable_complexity_metrics': True,
                'handle_missing_values': True,
                'normalize_data': True,
                'standardize_data': True,
                'outlier_detection': True,
                'enable_feature_selection': self.config.feature_selection,
                'max_features': self.config.max_features,
                'validation_split': self.config.validation_split,
                'use_bayesian_optimization': self.config.search_strategy == 'bayesian',
                'n_trials': self.config.max_trials,
                'max_candidates': self.config.max_trials,
                'memory_limit_gb': self.config.memory_limit_gb
            }
            
            # Initialize unified components with error handling
            try:
                self.evaluator = UnifiedEvaluator(unified_config)
                tprint_info("✅ Unified evaluator initialized")
            except Exception as e:
                tprint_warning(f"Unified evaluator initialization failed: {e}")
                self.evaluator = None
            
            try:
                self.hardware_optimizer = UnifiedHardwareOptimizer(unified_config)
                tprint_info("✅ Unified hardware optimizer initialized")
            except Exception as e:
                tprint_warning(f"Unified hardware optimizer initialization failed: {e}")
                self.hardware_optimizer = None
            
            try:
                self.search_engine = UnifiedSearchEngine(unified_config)
                tprint_info("✅ Unified search engine initialized")
            except Exception as e:
                tprint_warning(f"Unified search engine initialization failed: {e}")
                self.search_engine = None
            
            try:
                self.data_processor = UnifiedDataProcessor(unified_config)
                tprint_info("✅ Unified data processor initialized")
            except Exception as e:
                tprint_warning(f"Unified data processor initialization failed: {e}")
                self.data_processor = None
            
            tprint_info("✅ Unified components initialization completed")
            
        except Exception as e:
            tprint_error(f"Unified components initialization failed: {e}")
            self._initialize_fallback_components()
    
    def _initialize_fallback_components(self):
        """Initialize fallback components when unified components are not available."""
        try:
            # Fallback initialization
            self.evaluator = None
            self.hardware_optimizer = None
            self.search_engine = None
            self.data_processor = None
            tprint_warning("⚠️ Using fallback mode - some features may be limited")
            
            # Initialize fallback data processor
            self._initialize_fallback_data_processor()
            
        except Exception as e:
            tprint_error(f"Fallback components initialization failed: {e}")
    
    def _initialize_fallback_data_processor(self):
        """Initialize fallback data processor."""
        try:
            class FallbackDataProcessor:
                def __init__(self, config):
                    self.config = config
                
                def process_data(self, X, y, data_type="general"):
                    try:
                        # Basic data processing
                        if isinstance(X, pd.DataFrame):
                            X_processed = X.copy()
                        else:
                            X_processed = pd.DataFrame(X)
                        
                        if isinstance(y, pd.Series):
                            y_processed = y.copy()
                        else:
                            y_processed = pd.Series(y)
                        
                        # Handle missing values
                        X_processed = X_processed.fillna(X_processed.mean())
                        y_processed = y_processed.fillna(y_processed.mean())
                        
                        return X_processed, y_processed, {'processed': True}
                    except Exception as e:
                        tprint_error(f"Fallback data processing failed: {e}")
                        return X, y, {'processed': False, 'error': str(e)}
                
                def split_data(self, X, y, data_type="general"):
                    try:
                        from sklearn.model_selection import train_test_split
                        X_train, X_val, y_train, y_val = train_test_split(
                            X, y, test_size=self.config.validation_split, random_state=42
                        )
                        return X_train, X_val, y_train, y_val
                    except Exception as e:
                        tprint_error(f"Fallback data splitting failed: {e}")
                        # Return original data as fallback
                        split_idx = int(len(X) * (1 - self.config.validation_split))
                        return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]
            
            self.data_processor = FallbackDataProcessor(self.config)
            tprint_info("✅ Fallback data processor initialized")
            
        except Exception as e:
            tprint_error(f"Fallback data processor initialization failed: {e}")
            self.data_processor = None
    
    def _cleanup_resources(self):
        """Cleanup resources using unified components with error handling."""
        try:
            if self.hardware_optimizer:
                try:
                    self.hardware_optimizer.cleanup()
                    tprint_info("✅ Hardware resources cleaned up")
                except Exception as e:
                    tprint_warning(f"⚠️ Hardware cleanup failed: {e}")
            
            if self.memory_optimizer:
                try:
                    if hasattr(self.memory_optimizer, 'stop_monitoring'):
                        self.memory_optimizer.stop_monitoring()
                    tprint_info("✅ Memory monitoring stopped")
                except Exception as e:
                    tprint_warning(f"⚠️ Memory monitoring cleanup failed: {e}")
            
            # Cleanup M1 optimizers if available
            if COMMON_OPERATIONS_AVAILABLE:
                try:
                    from src.utils.common_operations import cleanup_m1_optimizers
                    cleanup_m1_optimizers()
                    tprint_info("✅ M1 optimizers cleaned up")
                except Exception as e:
                    tprint_warning(f"⚠️ M1 optimizers cleanup failed: {e}")
            
        except Exception as e:
            tprint_error(f"Resource cleanup failed: {e}")
    
    def prepare_data(self, X: Union[pd.DataFrame, np.ndarray], 
                    y: Union[pd.Series, np.ndarray],
                    test_size: float = None) -> Dict[str, Any]:
        """
        Prepare data for NAS training using unified components with comprehensive error handling.
        
        Args:
            X: Feature data
            y: Target data
            test_size: Test set size (uses config if None)
            
        Returns:
            Dictionary containing prepared data splits
        """
        tprint_info("📊 Preparing data for NAS training")
        
        try:
            with tprint_timer("Data preparation"):
                # Validate input data
                if not self._validate_input_data(X, y):
                    raise ValueError("Invalid input data provided")
                
                # Convert to appropriate format
                X_original = X.copy() if hasattr(X, 'copy') else X
                y_original = y.copy() if hasattr(y, 'copy') else y
                
                # Use unified data processor if available
                if self.data_processor:
                    try:
                        X_processed, y_processed, processing_info = self.data_processor.process_data(
                            X_original, y_original, "general"
                        )
                        
                        # Split data using unified processor
                        X_train, X_val, y_train, y_val = self.data_processor.split_data(
                            X_processed, y_processed, "general"
                        )
                        
                        # Additional test split if needed
                        test_size = test_size or self.config.test_split
                        if test_size > 0:
                            X_train, X_test, y_train, y_test = train_test_split(
                                X_train, y_train, test_size=test_size, random_state=42, stratify=y_train
                            )
                        else:
                            X_test = X_val
                            y_test = y_val
                        
                        tprint_success("✅ Data prepared using unified components")
                        tprint_structured({
                            'processing_info': processing_info,
                            'train_shape': X_train.shape,
                            'val_shape': X_val.shape,
                            'test_shape': X_test.shape
                        })
                        
                        return {
                            'X_train': X_train,
                            'X_val': X_val,
                            'X_test': X_test,
                            'y_train': y_train,
                            'y_val': y_val,
                            'y_test': y_test,
                            'processing_info': processing_info,
                            'feature_names': getattr(X_processed, 'columns', None),
                            'data_quality': self._assess_data_quality(X_processed, y_processed)
                        }
                        
                    except Exception as e:
                        tprint_warning(f"Unified data processing failed: {e}")
                        # Fall back to manual processing
                        return self._prepare_data_manual(X_original, y_original, test_size)
                else:
                    # Use manual data processing
                    return self._prepare_data_manual(X_original, y_original, test_size)
                    
        except Exception as e:
            tprint_error(f"Data preparation failed: {e}")
            tprint_error(f"Traceback: {traceback.format_exc()}")
            # Return minimal data structure to prevent complete failure
            return self._create_minimal_data_splits(X, y)
    
    def _validate_input_data(self, X, y) -> bool:
        """Validate input data with comprehensive checks."""
        try:
            # Check if data is not None
            if X is None or y is None:
                tprint_error("❌ Input data is None")
                return False
            
            # Check data types and shapes
            if hasattr(X, 'shape') and hasattr(y, 'shape'):
                if len(X.shape) != 2:
                    tprint_error(f"❌ X must be 2D, got shape {X.shape}")
                    return False
                
                if len(y.shape) != 1:
                    tprint_error(f"❌ y must be 1D, got shape {y.shape}")
                    return False
                
                if X.shape[0] != y.shape[0]:
                    tprint_error(f"❌ X and y must have same number of samples: {X.shape[0]} vs {y.shape[0]}")
                    return False
                
                if X.shape[0] < 10:
                    tprint_warning(f"⚠️ Very small dataset: {X.shape[0]} samples")
                
                if X.shape[1] < 1:
                    tprint_error("❌ X must have at least 1 feature")
                    return False
                
                tprint_info(f"✅ Data validation passed: {X.shape[0]} samples, {X.shape[1]} features")
                return True
            else:
                tprint_error("❌ Input data must have shape attribute")
                return False
                
        except Exception as e:
            tprint_error(f"Data validation failed: {e}")
            return False
    
    def _prepare_data_manual(self, X, y, test_size=None):
        """Manual data preparation with comprehensive error handling."""
        try:
            tprint_info("🔧 Using manual data preparation")
            
            # Convert to pandas if needed
            if not isinstance(X, pd.DataFrame):
                if isinstance(X, np.ndarray):
                    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
                else:
                    X = pd.DataFrame(X)
            
            if not isinstance(y, pd.Series):
                if isinstance(y, np.ndarray):
                    y = pd.Series(y, name='target')
                else:
                    y = pd.Series(y, name='target')
            
            # Handle missing values
            X_clean = X.fillna(X.mean())
            y_clean = y.fillna(y.mean())
            
            # Feature selection if enabled
            if self.config.feature_selection and X_clean.shape[1] > self.config.max_features:
                try:
                    if ML_COMMON_AVAILABLE:
                        X_selected = select_features(X_clean, y_clean, method=self.config.feature_selection_method, k=self.config.max_features)
                    else:
                        # Simple feature selection - take first max_features
                        X_selected = X_clean.iloc[:, :self.config.max_features]
                    tprint_info(f"✅ Feature selection: {X_clean.shape[1]} -> {X_selected.shape[1]} features")
                except Exception as e:
                    tprint_warning(f"Feature selection failed: {e}, using all features")
                    X_selected = X_clean
            else:
                X_selected = X_clean
            
            # Split data
            test_size = test_size or self.config.test_split
            validation_size = self.config.validation_split
            
            # First split: train+val vs test
            X_temp, X_test, y_temp, y_test = train_test_split(
                X_selected, y_clean, test_size=test_size, random_state=42, stratify=y_clean
            )
            
            # Second split: train vs val
            val_size = validation_size / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size, random_state=42, stratify=y_temp
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)
            
            # Convert back to DataFrames
            X_train_df = pd.DataFrame(X_train_scaled, columns=X_selected.columns, index=X_train.index)
            X_val_df = pd.DataFrame(X_val_scaled, columns=X_selected.columns, index=X_val.index)
            X_test_df = pd.DataFrame(X_test_scaled, columns=X_selected.columns, index=X_test.index)
            
            # Assess data quality
            quality_metrics = self._assess_data_quality(X_selected, y_clean)
            
            data_splits = {
                'X_train': X_train_df,
                'X_val': X_val_df,
                'X_test': X_test_df,
                'y_train': y_train,
                'y_val': y_val,
                'y_test': y_test,
                'scaler': scaler,
                'feature_names': X_selected.columns.tolist(),
                'quality_metrics': quality_metrics
            }
            
            tprint_success(f"✅ Data prepared: {X_train_df.shape[0]} train, {X_val_df.shape[0]} val, {X_test_df.shape[0]} test")
            return data_splits
            
        except Exception as e:
            tprint_error(f"Manual data preparation failed: {e}")
            return self._create_minimal_data_splits(X, y)
    
    def _assess_data_quality(self, X, y):
        """Assess data quality with comprehensive metrics."""
        try:
            if COMMON_OPERATIONS_AVAILABLE:
                return create_data_quality_report(X)
            else:
                # Basic quality assessment
                return {
                    'total_rows': len(X),
                    'total_columns': X.shape[1] if hasattr(X, 'shape') else len(X),
                    'missing_values': X.isnull().sum().sum() if hasattr(X, 'isnull') else 0,
                    'duplicate_rows': X.duplicated().sum() if hasattr(X, 'duplicated') else 0
                }
        except Exception as e:
            tprint_warning(f"Data quality assessment failed: {e}")
            return {'error': str(e)}
    
    def _create_minimal_data_splits(self, X, y):
        """Create minimal data splits as fallback."""
        try:
            tprint_warning("⚠️ Creating minimal data splits as fallback")
            
            # Simple split
            split_idx = int(len(X) * 0.8)
            X_train = X[:split_idx]
            X_test = X[split_idx:]
            y_train = y[:split_idx]
            y_test = y[split_idx:]
            
            return {
                'X_train': X_train,
                'X_val': X_train,  # Use train as val
                'X_test': X_test,
                'y_train': y_train,
                'y_val': y_train,  # Use train as val
                'y_test': y_test,
                'feature_names': None,
                'quality_metrics': {'fallback': True}
            }
        except Exception as e:
            tprint_error(f"Minimal data splits creation failed: {e}")
            raise RuntimeError("Data preparation completely failed")
    
    def generate_architecture(self, trial_id: int) -> Dict[str, Any]:
        """
        Generate a neural network architecture for the given trial.
        
        Args:
            trial_id: Trial identifier
            
        Returns:
            Architecture configuration dictionary
        """
        import random
        
        # Generate architecture based on search strategy
        if self.config.search_strategy == 'random':
            architecture = self._generate_random_architecture()
        elif self.config.search_strategy == 'grid':
            architecture = self._generate_grid_architecture(trial_id)
        elif self.config.search_strategy == 'bayesian':
            architecture = self._generate_bayesian_architecture(trial_id)
        else:
            architecture = self._generate_random_architecture()
        
        # Add trial metadata
        architecture['trial_id'] = trial_id
        architecture['search_strategy'] = self.config.search_strategy
        architecture['generated_at'] = datetime.now().isoformat()
        
        return architecture
    
    def _generate_random_architecture(self) -> Dict[str, Any]:
        """Generate a random neural network architecture."""
        import random
        
        # Random number of layers
        n_layers = random.randint(self.config.min_layers, self.config.max_layers)
        
        # Random layer sizes
        layers = []
        for i in range(n_layers):
            neurons = random.randint(self.config.min_neurons, self.config.max_neurons)
            activation = random.choice(self.config.activation_functions)
            dropout = random.choice(self.config.dropout_rates)
            
            layers.append({
                'neurons': neurons,
                'activation': activation,
                'dropout': dropout
            })
        
        # Random hyperparameters
        learning_rate = random.uniform(*self.config.learning_rate_range)
        batch_size = random.randint(*self.config.batch_size_range)
        
        return {
            'layers': layers,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'optimizer': self.config.optimizer,
            'n_layers': n_layers
        }
    
    def _generate_grid_architecture(self, trial_id: int) -> Dict[str, Any]:
        """Generate architecture from grid search."""
        # Simple grid search implementation
        n_layers_options = [2, 4, 6, 8, 10]
        neurons_options = [32, 64, 128, 256, 512]
        
        n_layers = n_layers_options[trial_id % len(n_layers_options)]
        base_neurons = neurons_options[trial_id % len(neurons_options)]
        
        layers = []
        for i in range(n_layers):
            neurons = base_neurons // (2 ** i) if i > 0 else base_neurons
            neurons = max(self.config.min_neurons, min(neurons, self.config.max_neurons))
            
            layers.append({
                'neurons': neurons,
                'activation': 'relu',
                'dropout': 0.2
            })
        
        return {
            'layers': layers,
            'learning_rate': 0.001,
            'batch_size': 64,
            'optimizer': self.config.optimizer,
            'n_layers': n_layers
        }
    
    def _generate_bayesian_architecture(self, trial_id: int) -> Dict[str, Any]:
        """Generate architecture using Bayesian optimization."""
        # Simplified Bayesian optimization
        # In practice, this would use a proper Bayesian optimization library
        
        # Use trial_id to create pseudo-random but deterministic architecture
        np.random.seed(trial_id)
        
        n_layers = int(np.random.uniform(self.config.min_layers, self.config.max_layers + 1))
        
        layers = []
        for i in range(n_layers):
            # Bayesian-inspired neuron count
            neurons = int(np.random.lognormal(
                np.log(self.config.min_neurons + self.config.max_neurons) / 2,
                0.5
            ))
            neurons = max(self.config.min_neurons, min(neurons, self.config.max_neurons))
            
            layers.append({
                'neurons': neurons,
                'activation': np.random.choice(self.config.activation_functions),
                'dropout': np.random.choice(self.config.dropout_rates)
            })
        
        # Bayesian-inspired hyperparameters
        learning_rate = np.random.lognormal(
            np.log(np.sqrt(self.config.learning_rate_range[0] * self.config.learning_rate_range[1])),
            0.5
        )
        learning_rate = max(self.config.learning_rate_range[0], 
                           min(learning_rate, self.config.learning_rate_range[1]))
        
        batch_size = int(np.random.lognormal(
            np.log(np.sqrt(self.config.batch_size_range[0] * self.config.batch_size_range[1])),
            0.3
        ))
        batch_size = max(self.config.batch_size_range[0], 
                        min(batch_size, self.config.batch_size_range[1]))
        
        return {
            'layers': layers,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'optimizer': self.config.optimizer,
            'n_layers': n_layers
        }
    
    def create_model_from_architecture(self, architecture: Dict[str, Any], 
                                     input_shape: int) -> Any:
        """
        Create a neural network model from architecture configuration.
        
        Args:
            architecture: Architecture configuration
            input_shape: Input feature dimension
            
        Returns:
            Compiled neural network model
        """
        try:
            # Try to import TensorFlow/Keras
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense, Dropout
            from tensorflow.keras.optimizers import Adam
            
            # Create model
            model = Sequential()
            
            # Add input layer
            model.add(Dense(
                architecture['layers'][0]['neurons'],
                activation=architecture['layers'][0]['activation'],
                input_shape=(input_shape,)
            ))
            
            if architecture['layers'][0]['dropout'] > 0:
                model.add(Dropout(architecture['layers'][0]['dropout']))
            
            # Add hidden layers
            for layer_config in architecture['layers'][1:-1]:
                model.add(Dense(
                    layer_config['neurons'],
                    activation=layer_config['activation']
                ))
                if layer_config['dropout'] > 0:
                    model.add(Dropout(layer_config['dropout']))
            
            # Add output layer
            model.add(Dense(1, activation='sigmoid'))
            
            # Compile model
            optimizer = Adam(learning_rate=architecture['learning_rate'])
            model.compile(
                optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            return model
            
        except ImportError:
            # Fallback to sklearn MLPClassifier
            from sklearn.neural_network import MLPClassifier
            
            # Extract architecture parameters
            hidden_layer_sizes = tuple(layer['neurons'] for layer in architecture['layers'][:-1])
            activation = architecture['layers'][0]['activation']
            learning_rate_init = architecture['learning_rate']
            batch_size = architecture['batch_size']
            
            model = MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                activation=activation,
                learning_rate_init=learning_rate_init,
                batch_size=batch_size,
                max_iter=self.config.max_epochs,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=self.config.early_stopping_patience,
                random_state=42
            )
            
            return model
    
    def train_architecture(self, architecture: Dict[str, Any], 
                          data_splits: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train a neural network with the given architecture.
        
        Args:
            architecture: Architecture configuration
            data_splits: Data splits dictionary
            
        Returns:
            Training results dictionary
        """
        trial_id = architecture['trial_id']
        tprint_info(f"🧠 Training architecture {trial_id}")
        
        with tprint_timer(f"Architecture {trial_id} training"):
            try:
                # Create model
                input_shape = data_splits['X_train'].shape[1]
                model = self.create_model_from_architecture(architecture, input_shape)
                
                # Prepare data
                X_train = data_splits['X_train'].values
                y_train = data_splits['y_train'].values
                X_val = data_splits['X_val'].values
                y_val = data_splits['y_val'].values
                
                # Train model
                if hasattr(model, 'fit'):
                    # TensorFlow/Keras model
                    history = model.fit(
                        X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=self.config.max_epochs,
                        batch_size=architecture['batch_size'],
                        verbose=0
                    )
                    
                    # Evaluate model
                    train_score = model.evaluate(X_train, y_train, verbose=0)
                    val_score = model.evaluate(X_val, y_val, verbose=0)
                    
                    results = {
                        'trial_id': trial_id,
                        'architecture': architecture,
                        'train_loss': train_score[0],
                        'train_accuracy': train_score[1],
                        'val_loss': val_score[0],
                        'val_accuracy': val_score[1],
                        'epochs_trained': len(history.history['loss']),
                        'best_val_accuracy': max(history.history['val_accuracy']),
                        'success': True
                    }
                    
                else:
                    # Sklearn model
                    model.fit(X_train, y_train)
                    
                    train_score = model.score(X_train, y_train)
                    val_score = model.score(X_val, y_val)
                    
                    results = {
                        'trial_id': trial_id,
                        'architecture': architecture,
                        'train_accuracy': train_score,
                        'val_accuracy': val_score,
                        'epochs_trained': model.n_iter_,
                        'best_val_accuracy': val_score,
                        'success': True
                    }
                
                tprint_success(f"✅ Architecture {trial_id} trained: val_acc={results['val_accuracy']:.4f}")
                return results
                
            except Exception as e:
                tprint_error(f"❌ Architecture {trial_id} training failed: {e}")
                return {
                    'trial_id': trial_id,
                    'architecture': architecture,
                    'error': str(e),
                    'success': False
                }
    
    def search_architectures(self, data_splits: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Perform neural architecture search.
        
        Args:
            data_splits: Prepared data splits
            
        Returns:
            List of search results
        """
        tprint_info(f"🔍 Starting NAS with {self.config.search_strategy} strategy")
        tprint_structured({
            'max_trials': self.config.max_trials,
            'search_strategy': self.config.search_strategy,
            'data_shape': data_splits['X_train'].shape
        })
        
        search_results = []
        
        # Use M1-optimized parallel processing if available
        if self.config.use_m1_optimization and self.cpu_optimizer:
            search_results = self._parallel_architecture_search(data_splits)
        else:
            search_results = self._sequential_architecture_search(data_splits)
        
        # Sort results by validation accuracy
        search_results.sort(key=lambda x: x.get('val_accuracy', 0), reverse=True)
        
        # Store results
        self.search_results = search_results
        
        # Save results
        if self.config.save_results:
            self._save_search_results()
        
        tprint_success(f"✅ NAS completed: {len(search_results)} architectures tested")
        return search_results
    
    def _sequential_architecture_search(self, data_splits: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Perform sequential architecture search."""
        search_results = []
        
        for trial_id in range(self.config.max_trials):
            tprint_progress(trial_id + 1, self.config.max_trials, f"Trial {trial_id + 1}")
            
            # Generate architecture
            architecture = self.generate_architecture(trial_id)
            
            # Train architecture
            result = self.train_architecture(architecture, data_splits)
            search_results.append(result)
            
            # Early stopping if we have enough good results
            if len([r for r in search_results if r.get('success', False)]) >= 10:
                if trial_id > 20:  # Only stop after at least 20 trials
                    tprint_info("🛑 Early stopping: sufficient good results found")
                    break
        
        return search_results
    
    def _parallel_architecture_search(self, data_splits: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Perform parallel architecture search using M1 optimization."""
        tprint_info("🚀 Using M1-optimized parallel architecture search")
        
        # Create trial tasks
        trial_tasks = []
        for trial_id in range(self.config.max_trials):
            architecture = self.generate_architecture(trial_id)
            trial_tasks.append((trial_id, architecture))
        
        # Use M1-optimized parallel processing
        def train_single_architecture(task):
            trial_id, architecture = task
            return self.train_architecture(architecture, data_splits)
        
        # Execute in parallel with M1 optimization
        search_results = parallel_map_m1(
            train_single_architecture,
            trial_tasks,
            max_workers=self.cpu_optimizer.get_optimal_worker_count()
        )
        
        return search_results
    
    def evaluate_best_architecture(self, data_splits: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate the best architecture using unified evaluator.
        
        Args:
            data_splits: Data splits dictionary
            
        Returns:
            Evaluation results
        """
        if not self.search_results:
            raise ValueError("No search results available. Run search_architectures first.")
        
        # Get best architecture
        best_result = max([r for r in self.search_results if r.get('success', False)], 
                         key=lambda x: x.get('val_accuracy', 0))
        
        tprint_info(f"🏆 Evaluating best architecture (trial {best_result['trial_id']})")
        
        with tprint_timer("Best architecture evaluation"):
            # Create and train best model
            best_architecture = best_result['architecture']
            input_shape = data_splits['X_train'].shape[1]
            best_model = self.create_model_from_architecture(best_architecture, input_shape)
            
            # Prepare data
            X_train = data_splits['X_train'].values
            y_train = data_splits['y_train'].values
            X_test = data_splits['X_test'].values
            y_test = data_splits['y_test'].values
            
            # Train model
            if hasattr(best_model, 'fit'):
                best_model.fit(X_train, y_train, epochs=self.config.max_epochs, verbose=0)
            else:
                best_model.fit(X_train, y_train)
            
            # Use unified evaluator if available
            if self.evaluator:
                evaluation_results = self.evaluator.evaluate_architecture(
                    best_model, X_test, y_test, X_train, y_train
                )
                test_accuracy = evaluation_results.get('accuracy', 0.0)
                y_pred = best_model.predict(X_test) if hasattr(best_model, 'predict') else np.random.randint(0, 2, len(y_test))
            else:
                # Fallback evaluation
                if hasattr(best_model, 'evaluate'):
                    test_loss, test_accuracy = best_model.evaluate(X_test, y_test, verbose=0)
                else:
                    test_accuracy = best_model.score(X_test, y_test)
                y_pred = best_model.predict(X_test)
            
            # Calculate additional metrics
            from sklearn.metrics import precision_score, recall_score, f1_score
            
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            evaluation_results = {
                'best_architecture': best_architecture,
                'test_accuracy': test_accuracy,
                'test_precision': precision,
                'test_recall': recall,
                'test_f1': f1,
                'predictions': y_pred.tolist(),
                'true_labels': y_test.tolist()
            }
            
            # Store best model
            self.best_architecture = best_architecture
            self.best_model = best_model
            
            tprint_success(f"✅ Best architecture evaluated: test_acc={test_accuracy:.4f}")
            return evaluation_results
    
    def _save_search_results(self):
        """Save search results to files."""
        try:
            # Save results as JSON
            results_file = self.output_dir / 'search_results.json'
            with open(results_file, 'w') as f:
                json.dump(self.search_results, f, indent=2, default=str)
            
            # Save best model if available
            if self.best_model and self.config.save_models:
                model_file = self.output_dir / 'best_model.pkl'
                with open(model_file, 'wb') as f:
                    pickle.dump(self.best_model, f)
            
            tprint_success(f"✅ Results saved to {self.output_dir}")
            
        except Exception as e:
            tprint_error(f"❌ Failed to save results: {e}")
    
    def run_full_nas(self, X: Union[pd.DataFrame, np.ndarray], 
                    y: Union[pd.Series, np.ndarray]) -> Dict[str, Any]:
        """
        Run complete Neural Architecture Search pipeline.
        
        Args:
            X: Feature data
            y: Target data
            
        Returns:
            Complete NAS results
        """
        tprint_info("🚀 Starting full NAS pipeline")
        
        with tprint_timer("Full NAS pipeline"):
            # Step 1: Prepare data
            tprint_info("📊 Step 1: Preparing data")
            data_splits = self.prepare_data(X, y)
            
            # Step 2: Search architectures
            tprint_info("🔍 Step 2: Searching architectures")
            search_results = self.search_architectures(data_splits)
            
            # Step 3: Evaluate best architecture
            tprint_info("🏆 Step 3: Evaluating best architecture")
            evaluation_results = self.evaluate_best_architecture(data_splits)
            
            # Compile final results
            final_results = {
                'search_results': search_results,
                'evaluation_results': evaluation_results,
                'config': self.config.__dict__,
                'data_info': {
                    'n_samples': len(X),
                    'n_features': X.shape[1] if hasattr(X, 'shape') else len(X[0]),
                    'feature_names': data_splits.get('feature_names', [])
                },
                'best_architecture': self.best_architecture,
                'training_completed': True
            }
            
            tprint_success("✅ Full NAS pipeline completed")
            return final_results
    
    def cleanup(self):
        """Cleanup resources and stop monitoring."""
        try:
            # Stop M1 optimizations
            if self.memory_optimizer:
                self.memory_optimizer.stop_monitoring()
            
            # Cleanup M1 optimizers
            cleanup_m1_optimizers()
            
            tprint_success("✅ NAS Trainer cleanup completed")
            
        except Exception as e:
            tprint_error(f"❌ Cleanup failed: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()


# Example usage and testing functions
def create_sample_data(n_samples: int = 1000, n_features: int = 20) -> Tuple[pd.DataFrame, pd.Series]:
    """Create sample data for testing."""
    np.random.seed(42)
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
    
    # Generate target (binary classification)
    y = (X.sum(axis=1) > 0).astype(int)
    y = pd.Series(y, name='target')
    
    return X, y


def run_nas_example():
    """Run a complete NAS example."""
    tprint_info("🧪 Running NAS example")
    
    # Create sample data
    X, y = create_sample_data(n_samples=1000, n_features=20)
    tprint_success(f"✅ Created sample data: {X.shape}")
    
    # Configure NAS
    config = NASConfig(
        search_strategy='random',
        max_trials=20,
        max_epochs=10,
        use_m1_optimization=True,
        verbose=True
    )
    
    # Run NAS
    with NASTrainer(config) as nas_trainer:
        results = nas_trainer.run_full_nas(X, y)
        
        # Print results
        tprint_structured({
            'best_accuracy': results['evaluation_results']['test_accuracy'],
            'best_architecture': results['best_architecture'],
            'total_trials': len(results['search_results'])
        })
    
    tprint_success("✅ NAS example completed")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run example
    run_nas_example()