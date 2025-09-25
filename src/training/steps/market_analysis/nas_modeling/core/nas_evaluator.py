"""
NAS Evaluator - Updated to use Unified Evaluator

This module provides NAS-specific evaluation capabilities using
the unified evaluation framework.
"""

import logging
import time
import asyncio
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import pandas as pd

# Import unified evaluator
try:
    from src.utils.nas_tas import UnifiedEvaluator, EvaluationConfig, EvaluationResult
    UNIFIED_EVALUATOR_AVAILABLE = True
except ImportError:
    UNIFIED_EVALUATOR_AVAILABLE = False

# Import utility modules with comprehensive error handling
try:
    from src.utils.common_operations import (
        safe_dataframe_operation, validate_dataframe_columns, 
        safe_convert_dtypes, calculate_data_quality_metrics,
        safe_merge_dataframes, create_summary_statistics,
        optimize_dataframe_dtypes, get_dataframe_info,
        integrate_with_m1_optimizers, memory_checkpoint, gpu_context,
        safe_json_dump, safe_json_load, safe_file_exists, ensure_directory,
        validate_finite, validate_positive, validate_range, safe_divide,
        get_current_datetime, format_datetime, safe_deepcopy,
        safe_to_parquet, safe_read_parquet, list_parquet_files,
        get_memory_usage, optimize_memory, cleanup_m1_optimizers
    )
    COMMON_OPERATIONS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"⚠️ Common operations not available: {e}")
    COMMON_OPERATIONS_AVAILABLE = False

try:
    from src.utils.common_utilities import (
        safe_dataframe_operation as safe_df_op,
        validate_dataframe_columns as validate_df_cols,
        safe_convert_dtypes as safe_convert,
        calculate_data_quality_metrics as calc_quality,
        create_summary_statistics as create_summary,
        CommonUtilities
    )
    COMMON_UTILITIES_AVAILABLE = True
except ImportError as e:
    logging.warning(f"⚠️ Common utilities not available: {e}")
    COMMON_UTILITIES_AVAILABLE = False

try:
    from src.utils.math_validation import (
        safe_divide, safe_log, safe_sqrt, safe_power,
        validate_finite, validate_positive, validate_range,
        safe_correlation, safe_covariance, safe_mean, safe_std,
        safe_percentile, safe_kelly_calculation, safe_weighted_average,
        safe_percentage_change, validate_correlation_matrix,
        safe_matrix_inverse, math_safe, MathValidation, MathValidationError
    )
    MATH_VALIDATION_AVAILABLE = True
except ImportError as e:
    logging.warning(f"⚠️ Math validation not available: {e}")
    MATH_VALIDATION_AVAILABLE = False

try:
    from src.utils.serialization_utils import (
        JSONSerializer, PickleSerializer, ParquetSerializer, UniversalSerializer
    )
    SERIALIZATION_AVAILABLE = True
except ImportError as e:
    logging.warning(f"⚠️ Serialization utilities not available: {e}")
    SERIALIZATION_AVAILABLE = False

try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_warning, tprint_error, tprint_success,
        tprint_progress, tprint_performance, tprint_structured,
        tprint_timer, tprint_with_level, LogLevel, tprint_debug
    )
    TPRINT_AVAILABLE = True
except ImportError as e:
    logging.warning(f"⚠️ TPrint utilities not available: {e}")
    TPRINT_AVAILABLE = False

try:
    from src.utils.hardware.m1_gpu_utils import (
        get_m1_gpu_manager, is_m1_available, is_mps_available,
        M1GPUManager
    )
    M1_GPU_AVAILABLE = True
except ImportError as e:
    logging.warning(f"⚠️ M1 GPU utilities not available: {e}")
    M1_GPU_AVAILABLE = False

try:
    from src.utils.hardware.m1_memory_optimizer import (
        get_m1_memory_optimizer, M1MemoryOptimizer
    )
    M1_MEMORY_AVAILABLE = True
except ImportError as e:
    logging.warning(f"⚠️ M1 memory optimizer not available: {e}")
    M1_MEMORY_AVAILABLE = False

try:
    from src.utils.hardware.m1_cpu_optimizer import (
        get_m1_cpu_optimizer, M1CPUOptimizer
    )
    M1_CPU_AVAILABLE = True
except ImportError as e:
    logging.warning(f"⚠️ M1 CPU optimizer not available: {e}")
    M1_CPU_AVAILABLE = False

try:
    from src.utils.data.klines_parquet import (
        load_klines_data, save_klines_data, validate_klines_data
    )
    KLINES_PARQUET_AVAILABLE = True
except ImportError as e:
    logging.warning(f"⚠️ Klines parquet utilities not available: {e}")
    KLINES_PARQUET_AVAILABLE = False

try:
    from src.utils.matrix_operations.unified_operations import (
        MatrixOperations, VectorizedOperations, BatchOperations
    )
    MATRIX_OPERATIONS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"⚠️ Matrix operations not available: {e}")
    MATRIX_OPERATIONS_AVAILABLE = False

try:
    from src.utils.ml_common.common_operations import (
        MLCommonOperations, CrossValidationManager, HyperparameterOptimizer
    )
    ML_COMMON_AVAILABLE = True
except ImportError as e:
    logging.warning(f"⚠️ ML common utilities not available: {e}")
    ML_COMMON_AVAILABLE = False

# Import hybrid NAS-TAS shared utilities
try:
    from src.training.steps.market_analysis.hybrid_nas_tas_regime.shared_utils.enhanced_utility_integration import (
        EnhancedUtilityIntegration, UtilityIntegrationConfig
    )
    from src.training.steps.market_analysis.hybrid_nas_tas_regime.shared_utils.unified_evaluation_framework import (
        UnifiedEvaluationFramework, EvaluationType, EvaluationMetric, EvaluationResult
    )
    from src.training.steps.market_analysis.hybrid_nas_tas_regime.shared_utils.unified_architecture_config import (
        ArchitectureType, SearchStrategy, OptimizationObjective, MarketRegime
    )
    from src.training.steps.market_analysis.hybrid_nas_tas_regime.shared_utils.unified_hardware_manager import (
        UnifiedHardwareManager, HardwareType, WorkloadType, HardwareMetrics
    )
    from src.training.steps.market_analysis.hybrid_nas_tas_regime.shared_utils.enhanced_ml_integration import (
        EnhancedMLIntegration, MLIntegrationConfig
    )
    HYBRID_NAS_TAS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"⚠️ Hybrid NAS-TAS shared utilities not available: {e}")
    HYBRID_NAS_TAS_AVAILABLE = False

# Setup logging
logger = logging.getLogger(__name__)

@dataclass
class ArchitectureMetrics:
    """Metrics for architecture evaluation."""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    auc_roc: float = 0.0
    training_time: float = 0.0
    inference_time: float = 0.0
    memory_usage: float = 0.0
    model_size: float = 0.0
    convergence_epochs: int = 0
    loss_value: float = 0.0
    validation_loss: float = 0.0
    generalization_gap: float = 0.0
    stability_score: float = 0.0
    efficiency_score: float = 0.0
    complexity_score: float = 0.0

@dataclass
class EvaluationConfig:
    """Configuration for NAS evaluation."""
    # Evaluation parameters
    max_evaluations: int = 100
    timeout_seconds: int = 3600
    early_stopping_patience: int = 10
    validation_split: float = 0.2
    test_split: float = 0.1
    
    # Cross-validation
    cv_folds: int = 5
    cv_strategy: str = "stratified"
    cv_shuffle: bool = True
    
    # Hyperparameter optimization
    hpo_enabled: bool = True
    hpo_trials: int = 50
    hpo_strategy: str = "bayesian"  # bayesian, grid, random
    
    # Hardware optimization
    use_m1_optimizations: bool = True
    memory_limit_gb: Optional[float] = None
    gpu_acceleration: bool = True
    
    # Data processing
    data_validation: bool = True
    feature_engineering: bool = True
    data_quality_threshold: float = 0.8
    
    # Logging and monitoring
    verbose: bool = True
    log_level: str = "INFO"
    save_intermediate_results: bool = True
    results_directory: str = "nas_evaluation_results"
    
    # Performance optimization
    parallel_evaluation: bool = True
    max_workers: int = 4
    batch_size: int = 32
    cache_results: bool = True

class NASEvaluator:
    """NAS-specific evaluator using unified components."""
    
    def __init__(self, config: Optional[EvaluationConfig] = None):
        """Initialize NAS evaluator with unified components."""
        if UNIFIED_EVALUATOR_AVAILABLE:
            self.unified_evaluator = UnifiedEvaluator(config)
        else:
            raise ImportError("Unified evaluator not available")
    
    def __getattr__(self, name):
        """Delegate to unified evaluator."""
        return getattr(self.unified_evaluator, name)


class LegacyNASEvaluator:
    """
    Comprehensive Neural Architecture Search Evaluator.
    
    This class provides advanced evaluation capabilities for neural architectures
    with integration of hardware optimizations, data processing, and ML utilities.
    """
    
    def __init__(self, config: Optional[EvaluationConfig] = None):
        """
        Initialize NASEvaluator.
        
        Args:
            config: Evaluation configuration
        """
        self.config = config or EvaluationConfig()
        self.logger = logger.getChild('NASEvaluator')
        
        # Initialize utility modules
        self._initialize_utilities()
        
        # Initialize hardware optimizations
        self._initialize_hardware_optimizations()
        
        # Initialize evaluation state
        self.evaluation_history: List[Dict[str, Any]] = []
        self.best_architecture: Optional[Dict[str, Any]] = None
        self.best_metrics: Optional[ArchitectureMetrics] = None
        
        # Initialize results storage
        self._setup_results_directory()
        
        # Log initialization
        if TPRINT_AVAILABLE:
            tprint_success("🚀 NASEvaluator initialized successfully")
        else:
            self.logger.info("🚀 NASEvaluator initialized successfully")
    
    def _initialize_utilities(self):
        """Initialize utility modules."""
        try:
            # Initialize common utilities
            if COMMON_UTILITIES_AVAILABLE:
                self.common_utils = CommonUtilities()
            else:
                self.common_utils = None
            
            # Initialize math validation
            if MATH_VALIDATION_AVAILABLE:
                self.math_validator = MathValidation()
            else:
                self.math_validator = None
            
            # Initialize serialization
            if SERIALIZATION_AVAILABLE:
                self.serializer = UniversalSerializer()
            else:
                self.serializer = None
            
            # Initialize matrix operations
            if MATRIX_OPERATIONS_AVAILABLE:
                self.matrix_ops = MatrixOperations()
                self.vectorized_ops = VectorizedOperations()
                self.batch_ops = BatchOperations()
            else:
                self.matrix_ops = None
                self.vectorized_ops = None
                self.batch_ops = None
            
            # Initialize ML common operations
            if ML_COMMON_AVAILABLE:
                self.ml_ops = MLCommonOperations()
                self.cv_manager = CrossValidationManager()
                self.hpo_optimizer = HyperparameterOptimizer()
            else:
                self.ml_ops = None
                self.cv_manager = None
                self.hpo_optimizer = None
            
            # Initialize hybrid NAS-TAS shared utilities
            if HYBRID_NAS_TAS_AVAILABLE:
                # Initialize enhanced utility integration
                utility_config = UtilityIntegrationConfig()
                self.enhanced_utils = EnhancedUtilityIntegration(utility_config)
                
                # Initialize unified evaluation framework
                self.unified_evaluator = UnifiedEvaluationFramework()
                
                # Initialize enhanced ML integration
                ml_config = MLIntegrationConfig()
                self.enhanced_ml = EnhancedMLIntegration(ml_config)
                
                # Initialize unified hardware manager
                self.unified_hardware = UnifiedHardwareManager()
            else:
                self.enhanced_utils = None
                self.unified_evaluator = None
                self.enhanced_ml = None
                self.unified_hardware = None
                
        except Exception as e:
            self.logger.error(f"❌ Error initializing utilities: {e}")
            raise
    
    def _initialize_hardware_optimizations(self):
        """Initialize hardware optimization modules."""
        try:
            # Initialize M1 GPU manager
            if M1_GPU_AVAILABLE:
                self.gpu_manager = get_m1_gpu_manager()
                self.m1_available = is_m1_available()
                self.mps_available = is_mps_available()
            else:
                self.gpu_manager = None
                self.m1_available = False
                self.mps_available = False
            
            # Initialize M1 memory optimizer
            if M1_MEMORY_AVAILABLE:
                self.memory_optimizer = get_m1_memory_optimizer()
                if self.memory_optimizer and self.config.memory_limit_gb:
                    self.memory_optimizer.set_memory_limit(self.config.memory_limit_gb)
            else:
                self.memory_optimizer = None
            
            # Initialize M1 CPU optimizer
            if M1_CPU_AVAILABLE:
                self.cpu_optimizer = get_m1_cpu_optimizer()
            else:
                self.cpu_optimizer = None
            
            # Log hardware status
            if TPRINT_AVAILABLE:
                tprint_info(f"🔧 Hardware Status: M1={self.m1_available}, MPS={self.mps_available}")
            else:
                self.logger.info(f"🔧 Hardware Status: M1={self.m1_available}, MPS={self.mps_available}")
                
        except Exception as e:
            self.logger.error(f"❌ Error initializing hardware optimizations: {e}")
            raise
    
    def _setup_results_directory(self):
        """Setup results directory for saving evaluation results."""
        try:
            if COMMON_OPERATIONS_AVAILABLE:
                self.results_dir = Path(self.config.results_directory)
                ensure_directory(self.results_dir)
            else:
                self.results_dir = Path(self.config.results_directory)
                self.results_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            self.logger.error(f"❌ Error setting up results directory: {e}")
            self.results_dir = Path("nas_evaluation_results")
            self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def evaluate_architecture(self, 
                             architecture: Dict[str, Any], 
                             data: Union[pd.DataFrame, np.ndarray, Dict[str, Any]],
                             target: Optional[Union[pd.Series, np.ndarray]] = None,
                             **kwargs) -> ArchitectureMetrics:
        """
        Evaluate a neural architecture.
        
        Args:
            architecture: Architecture definition
            data: Training data
            target: Target values (for supervised learning)
            **kwargs: Additional evaluation parameters
            
        Returns:
            ArchitectureMetrics: Evaluation metrics
        """
        start_time = time.time()
        
        try:
            if TPRINT_AVAILABLE:
                tprint_info(f"🔍 Evaluating architecture: {architecture.get('name', 'Unknown')}")
            else:
                self.logger.info(f"🔍 Evaluating architecture: {architecture.get('name', 'Unknown')}")
            
            # Validate inputs
            self._validate_evaluation_inputs(architecture, data, target)
            
            # Preprocess data
            processed_data = self._preprocess_data(data, target)
            
            # Setup hardware context
            with self._get_hardware_context():
                # Perform evaluation
                metrics = self._perform_architecture_evaluation(
                    architecture, processed_data, **kwargs
                )
            
            # Calculate evaluation time
            evaluation_time = time.time() - start_time
            metrics.training_time = evaluation_time
            
            # Store results
            self._store_evaluation_result(architecture, metrics)
            
            # Log results
            if TPRINT_AVAILABLE:
                tprint_success(f"✅ Architecture evaluation completed in {evaluation_time:.2f}s")
                tprint_structured({
                    "accuracy": metrics.accuracy,
                    "f1_score": metrics.f1_score,
                    "training_time": metrics.training_time
                })
            else:
                self.logger.info(f"✅ Architecture evaluation completed in {evaluation_time:.2f}s")
            
            return metrics
            
        except Exception as e:
            if TPRINT_AVAILABLE:
                tprint_error(f"❌ Architecture evaluation failed: {e}")
            else:
                self.logger.error(f"❌ Architecture evaluation failed: {e}")
            raise
    
    def _validate_evaluation_inputs(self, 
                                   architecture: Dict[str, Any], 
                                   data: Union[pd.DataFrame, np.ndarray, Dict[str, Any]],
                                   target: Optional[Union[pd.Series, np.ndarray]]):
        """Validate evaluation inputs."""
        try:
            # Validate architecture
            if not isinstance(architecture, dict):
                raise ValueError("Architecture must be a dictionary")
            
            required_keys = ['layers', 'activation', 'optimizer']
            missing_keys = [key for key in required_keys if key not in architecture]
            if missing_keys:
                raise ValueError(f"Architecture missing required keys: {missing_keys}")
            
            # Validate data
            if isinstance(data, pd.DataFrame):
                if data.empty:
                    raise ValueError("DataFrame is empty")
            elif isinstance(data, np.ndarray):
                if data.size == 0:
                    raise ValueError("Data array is empty")
            elif isinstance(data, dict):
                if not data:
                    raise ValueError("Data dictionary is empty")
            else:
                raise ValueError("Data must be DataFrame, numpy array, or dictionary")
            
            # Validate target if provided
            if target is not None:
                if isinstance(target, pd.Series):
                    if target.empty:
                        raise ValueError("Target series is empty")
                elif isinstance(target, np.ndarray):
                    if target.size == 0:
                        raise ValueError("Target array is empty")
                else:
                    raise ValueError("Target must be Series or numpy array")
            
        except Exception as e:
            self.logger.error(f"❌ Input validation failed: {e}")
            raise
    
    def _preprocess_data(self, 
                        data: Union[pd.DataFrame, np.ndarray, Dict[str, Any]],
                        target: Optional[Union[pd.Series, np.ndarray]]) -> Dict[str, Any]:
        """Preprocess data for evaluation."""
        try:
            processed_data = {}
            
            # Handle DataFrame data
            if isinstance(data, pd.DataFrame):
                if COMMON_OPERATIONS_AVAILABLE:
                    # Use common operations for data processing
                    processed_data['X'] = data.values
                    processed_data['feature_names'] = list(data.columns)
                    processed_data['data_info'] = get_dataframe_info(data)
                else:
                    processed_data['X'] = data.values
                    processed_data['feature_names'] = list(data.columns)
            
            # Handle numpy array data
            elif isinstance(data, np.ndarray):
                processed_data['X'] = data
                processed_data['feature_names'] = [f"feature_{i}" for i in range(data.shape[1])]
            
            # Handle dictionary data
            elif isinstance(data, dict):
                processed_data.update(data)
            
            # Process target
            if target is not None:
                if isinstance(target, pd.Series):
                    processed_data['y'] = target.values
                else:
                    processed_data['y'] = target
            
            # Validate data quality
            if self.config.data_validation:
                self._validate_data_quality(processed_data)
            
            return processed_data
            
        except Exception as e:
            self.logger.error(f"❌ Data preprocessing failed: {e}")
            raise
    
    def _validate_data_quality(self, data: Dict[str, Any]):
        """Validate data quality."""
        try:
            X = data.get('X')
            if X is None:
                raise ValueError("No input data found")
            
            # Check for missing values
            if isinstance(X, np.ndarray):
                missing_count = np.isnan(X).sum()
                if missing_count > 0:
                    self.logger.warning(f"⚠️ Found {missing_count} missing values in data")
            
            # Check data shape
            if len(X.shape) < 2:
                raise ValueError("Data must be at least 2-dimensional")
            
            # Check for sufficient samples
            if X.shape[0] < 10:
                self.logger.warning("⚠️ Very small dataset, evaluation may be unreliable")
            
            # Calculate data quality score
            if COMMON_OPERATIONS_AVAILABLE:
                # Create temporary DataFrame for quality assessment
                temp_df = pd.DataFrame(X)
                quality_metrics = calculate_data_quality_metrics(temp_df)
                quality_score = 1.0 - (quality_metrics['missing_percentage'] / 100.0)
                
                if quality_score < self.config.data_quality_threshold:
                    self.logger.warning(f"⚠️ Low data quality score: {quality_score:.2f}")
            
        except Exception as e:
            self.logger.error(f"❌ Data quality validation failed: {e}")
            raise
    
    def _get_hardware_context(self):
        """Get hardware optimization context manager."""
        if self.config.use_m1_optimizations and self.m1_available:
            if COMMON_OPERATIONS_AVAILABLE:
                return memory_checkpoint("nas_evaluation")
            else:
                return self._dummy_context()
        else:
            return self._dummy_context()
    
    def _dummy_context(self):
        """Dummy context manager for when hardware optimizations are not available."""
        from contextlib import contextmanager
        
        @contextmanager
        def dummy_context():
            yield
        
        return dummy_context()
    
    def _perform_architecture_evaluation(self, 
                                        architecture: Dict[str, Any],
                                        data: Dict[str, Any],
                                        **kwargs) -> ArchitectureMetrics:
        """Perform the actual architecture evaluation."""
        try:
            # Initialize metrics
            metrics = ArchitectureMetrics()
            
            # Extract data
            X = data['X']
            y = data.get('y')
            
            # Use enhanced evaluation if available
            if HYBRID_NAS_TAS_AVAILABLE and self.unified_evaluator:
                metrics = self._perform_enhanced_evaluation(architecture, data, **kwargs)
            else:
                # Fallback to standard evaluation
                if y is not None and self.cv_manager is not None:
                    cv_scores = self._perform_cross_validation(architecture, X, y)
                    metrics.accuracy = np.mean(cv_scores['accuracy'])
                    metrics.precision = np.mean(cv_scores['precision'])
                    metrics.recall = np.mean(cv_scores['recall'])
                    metrics.f1_score = np.mean(cv_scores['f1_score'])
                    metrics.auc_roc = np.mean(cv_scores['auc_roc'])
                else:
                    # Unsupervised evaluation
                    metrics = self._perform_unsupervised_evaluation(architecture, X)
            
            # Calculate additional metrics
            metrics = self._calculate_additional_metrics(metrics, architecture, X, y)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"❌ Architecture evaluation failed: {e}")
            raise
    
    def _perform_enhanced_evaluation(self, 
                                   architecture: Dict[str, Any],
                                   data: Dict[str, Any],
                                   **kwargs) -> ArchitectureMetrics:
        """Perform enhanced evaluation using hybrid NAS-TAS utilities."""
        try:
            if not self.unified_evaluator:
                raise ValueError("Unified evaluator not available")
            
            # Create evaluation result using unified framework
            evaluation_result = self.unified_evaluator.evaluate_architecture(
                architecture=architecture,
                data=data,
                evaluation_type=EvaluationType.COMPREHENSIVE,
                architecture_type=ArchitectureType.NAS
            )
            
            # Convert to ArchitectureMetrics
            metrics = ArchitectureMetrics()
            
            # Basic metrics
            if EvaluationMetric.ACCURACY in evaluation_result.basic_metrics:
                metrics.accuracy = evaluation_result.basic_metrics[EvaluationMetric.ACCURACY]
            if EvaluationMetric.PRECISION in evaluation_result.basic_metrics:
                metrics.precision = evaluation_result.basic_metrics[EvaluationMetric.PRECISION]
            if EvaluationMetric.RECALL in evaluation_result.basic_metrics:
                metrics.recall = evaluation_result.basic_metrics[EvaluationMetric.RECALL]
            if EvaluationMetric.F1_SCORE in evaluation_result.basic_metrics:
                metrics.f1_score = evaluation_result.basic_metrics[EvaluationMetric.F1_SCORE]
            if EvaluationMetric.ROC_AUC in evaluation_result.basic_metrics:
                metrics.auc_roc = evaluation_result.basic_metrics[EvaluationMetric.ROC_AUC]
            
            # Trading metrics (if available)
            if evaluation_result.trading_metrics:
                # Extract trading performance metrics
                if EvaluationMetric.SHARPE_RATIO in evaluation_result.trading_metrics:
                    # Use Sharpe ratio as efficiency score
                    metrics.efficiency_score = evaluation_result.trading_metrics[EvaluationMetric.SHARPE_RATIO]
            
            # Economic metrics (if available)
            if evaluation_result.economic_metrics:
                # Extract economic significance
                if EvaluationMetric.ECONOMIC_SIGNIFICANCE in evaluation_result.economic_metrics:
                    metrics.stability_score = evaluation_result.economic_metrics[EvaluationMetric.ECONOMIC_SIGNIFICANCE]
            
            # Risk metrics (if available)
            if evaluation_result.risk_metrics:
                # Extract risk metrics
                if EvaluationMetric.VOLATILITY in evaluation_result.risk_metrics:
                    # Use inverse volatility as stability
                    volatility = evaluation_result.risk_metrics[EvaluationMetric.VOLATILITY]
                    metrics.stability_score = 1.0 / (1.0 + volatility) if volatility > 0 else 0.5
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"❌ Enhanced evaluation failed: {e}")
            # Fallback to standard evaluation
            return self._perform_unsupervised_evaluation(architecture, data.get('X'))
    
    def _perform_cross_validation(self, 
                                 architecture: Dict[str, Any],
                                 X: np.ndarray, 
                                 y: np.ndarray) -> Dict[str, List[float]]:
        """Perform cross-validation evaluation."""
        try:
            if not self.cv_manager:
                # Fallback to simple train-test split
                return self._simple_train_test_split(architecture, X, y)
            
            # Use cross-validation manager
            cv_results = self.cv_manager.cross_validate(
                architecture=architecture,
                X=X,
                y=y,
                cv_folds=self.config.cv_folds,
                cv_strategy=self.config.cv_strategy
            )
            
            return cv_results
            
        except Exception as e:
            self.logger.error(f"❌ Cross-validation failed: {e}")
            # Fallback to simple evaluation
            return self._simple_train_test_split(architecture, X, y)
    
    def _simple_train_test_split(self, 
                                architecture: Dict[str, Any],
                                X: np.ndarray, 
                                y: np.ndarray) -> Dict[str, List[float]]:
        """Simple train-test split evaluation."""
        try:
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.config.test_split, random_state=42
            )
            
            # Train model (simplified)
            model = self._create_model(architecture)
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            try:
                auc_roc = roc_auc_score(y_test, y_pred_proba)
            except:
                auc_roc = 0.5  # Default for binary classification
            
            return {
                'accuracy': [accuracy],
                'precision': [precision],
                'recall': [recall],
                'f1_score': [f1],
                'auc_roc': [auc_roc]
            }
            
        except Exception as e:
            self.logger.error(f"❌ Simple train-test split failed: {e}")
            # Return default metrics
            return {
                'accuracy': [0.5],
                'precision': [0.5],
                'recall': [0.5],
                'f1_score': [0.5],
                'auc_roc': [0.5]
            }
    
    def _create_model(self, architecture: Dict[str, Any]):
        """Create model from architecture definition."""
        try:
            # This is a simplified model creation
            # In practice, you would implement proper model creation based on architecture
            from sklearn.ensemble import RandomForestClassifier
            
            # Extract parameters from architecture
            n_estimators = architecture.get('n_estimators', 100)
            max_depth = architecture.get('max_depth', None)
            
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42
            )
            
            return model
            
        except Exception as e:
            self.logger.error(f"❌ Model creation failed: {e}")
            raise
    
    def _perform_unsupervised_evaluation(self, 
                                       architecture: Dict[str, Any],
                                       X: np.ndarray) -> ArchitectureMetrics:
        """Perform unsupervised evaluation."""
        try:
            metrics = ArchitectureMetrics()
            
            # Use clustering or dimensionality reduction for unsupervised evaluation
            from sklearn.cluster import KMeans
            from sklearn.decomposition import PCA
            from sklearn.metrics import silhouette_score
            
            # Clustering evaluation
            n_clusters = architecture.get('n_clusters', 3)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(X)
            
            # Calculate silhouette score
            if len(np.unique(cluster_labels)) > 1:
                metrics.accuracy = silhouette_score(X, cluster_labels)
            else:
                metrics.accuracy = 0.0
            
            # Use accuracy as other metrics for unsupervised
            metrics.precision = metrics.accuracy
            metrics.recall = metrics.accuracy
            metrics.f1_score = metrics.accuracy
            metrics.auc_roc = metrics.accuracy
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"❌ Unsupervised evaluation failed: {e}")
            # Return default metrics
            return ArchitectureMetrics()
    
    def _calculate_additional_metrics(self, 
                                     metrics: ArchitectureMetrics,
                                     architecture: Dict[str, Any],
                                     X: np.ndarray,
                                     y: Optional[np.ndarray]) -> ArchitectureMetrics:
        """Calculate additional metrics."""
        try:
            # Calculate model complexity
            metrics.complexity_score = self._calculate_complexity_score(architecture)
            
            # Calculate efficiency score
            metrics.efficiency_score = self._calculate_efficiency_score(metrics)
            
            # Calculate stability score
            metrics.stability_score = self._calculate_stability_score(architecture, X)
            
            # Calculate generalization gap
            if y is not None:
                metrics.generalization_gap = abs(metrics.accuracy - metrics.validation_loss)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"❌ Additional metrics calculation failed: {e}")
            return metrics
    
    def _calculate_complexity_score(self, architecture: Dict[str, Any]) -> float:
        """Calculate model complexity score."""
        try:
            # Simple complexity calculation based on architecture
            layers = architecture.get('layers', [])
            complexity = len(layers)
            
            # Add parameters complexity
            for layer in layers:
                if isinstance(layer, dict):
                    complexity += layer.get('units', 0) * 0.1
            
            return min(complexity / 100.0, 1.0)  # Normalize to [0, 1]
            
        except Exception as e:
            self.logger.error(f"❌ Complexity score calculation failed: {e}")
            return 0.5
    
    def _calculate_efficiency_score(self, metrics: ArchitectureMetrics) -> float:
        """Calculate efficiency score."""
        try:
            # Combine accuracy and speed metrics
            accuracy_score = metrics.accuracy
            speed_score = 1.0 / (1.0 + metrics.training_time)  # Higher is better
            
            efficiency = (accuracy_score + speed_score) / 2.0
            return min(efficiency, 1.0)
            
        except Exception as e:
            self.logger.error(f"❌ Efficiency score calculation failed: {e}")
            return 0.5
    
    def _calculate_stability_score(self, architecture: Dict[str, Any], X: np.ndarray) -> float:
        """Calculate stability score."""
        try:
            # Simple stability calculation based on data variance
            if X.size > 0:
                variance = np.var(X)
                stability = 1.0 / (1.0 + variance)  # Higher variance = lower stability
                return min(stability, 1.0)
            else:
                return 0.5
                
        except Exception as e:
            self.logger.error(f"❌ Stability score calculation failed: {e}")
            return 0.5
    
    def _store_evaluation_result(self, architecture: Dict[str, Any], metrics: ArchitectureMetrics):
        """Store evaluation result."""
        try:
            result = {
                'timestamp': datetime.now().isoformat(),
                'architecture': architecture,
                'metrics': {
                    'accuracy': metrics.accuracy,
                    'precision': metrics.precision,
                    'recall': metrics.recall,
                    'f1_score': metrics.f1_score,
                    'auc_roc': metrics.auc_roc,
                    'training_time': metrics.training_time,
                    'inference_time': metrics.inference_time,
                    'memory_usage': metrics.memory_usage,
                    'model_size': metrics.model_size,
                    'convergence_epochs': metrics.convergence_epochs,
                    'loss_value': metrics.loss_value,
                    'validation_loss': metrics.validation_loss,
                    'generalization_gap': metrics.generalization_gap,
                    'stability_score': metrics.stability_score,
                    'efficiency_score': metrics.efficiency_score,
                    'complexity_score': metrics.complexity_score
                }
            }
            
            self.evaluation_history.append(result)
            
            # Update best architecture if this is better
            if self.best_architecture is None or metrics.accuracy > self.best_metrics.accuracy:
                self.best_architecture = architecture.copy()
                self.best_metrics = metrics
            
            # Save intermediate results if configured
            if self.config.save_intermediate_results:
                self._save_intermediate_results()
                
        except Exception as e:
            self.logger.error(f"❌ Error storing evaluation result: {e}")
    
    def _save_intermediate_results(self):
        """Save intermediate results."""
        try:
            if self.serializer:
                results_file = self.results_dir / f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                self.serializer.save(self.evaluation_history, str(results_file))
            elif COMMON_OPERATIONS_AVAILABLE:
                results_file = self.results_dir / f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                safe_json_dump(self.evaluation_history, results_file)
            else:
                # Fallback to pickle
                results_file = self.results_dir / f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
                with open(results_file, 'wb') as f:
                    pickle.dump(self.evaluation_history, f)
                    
        except Exception as e:
            self.logger.error(f"❌ Error saving intermediate results: {e}")
    
    def get_best_architecture(self) -> Optional[Dict[str, Any]]:
        """Get the best architecture found so far."""
        return self.best_architecture
    
    def get_best_metrics(self) -> Optional[ArchitectureMetrics]:
        """Get the best metrics found so far."""
        return self.best_metrics
    
    def get_evaluation_history(self) -> List[Dict[str, Any]]:
        """Get evaluation history."""
        return self.evaluation_history.copy()
    
    def save_results(self, filepath: Optional[str] = None) -> str:
        """Save evaluation results."""
        try:
            if filepath is None:
                filepath = self.results_dir / f"nas_evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            else:
                filepath = Path(filepath)
            
            results = {
                'config': {
                    'max_evaluations': self.config.max_evaluations,
                    'timeout_seconds': self.config.timeout_seconds,
                    'cv_folds': self.config.cv_folds,
                    'hpo_enabled': self.config.hpo_enabled,
                    'use_m1_optimizations': self.config.use_m1_optimizations
                },
                'best_architecture': self.best_architecture,
                'best_metrics': self.best_metrics.__dict__ if self.best_metrics else None,
                'evaluation_history': self.evaluation_history,
                'summary': {
                    'total_evaluations': len(self.evaluation_history),
                    'best_accuracy': self.best_metrics.accuracy if self.best_metrics else 0.0,
                    'average_accuracy': np.mean([h['metrics']['accuracy'] for h in self.evaluation_history]) if self.evaluation_history else 0.0
                }
            }
            
            if self.serializer:
                self.serializer.save(results, str(filepath))
            elif COMMON_OPERATIONS_AVAILABLE:
                safe_json_dump(results, filepath)
            else:
                with open(filepath, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
            
            if TPRINT_AVAILABLE:
                tprint_success(f"💾 Results saved to {filepath}")
            else:
                self.logger.info(f"💾 Results saved to {filepath}")
            
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"❌ Error saving results: {e}")
            raise
    
    def load_results(self, filepath: str) -> Dict[str, Any]:
        """Load evaluation results."""
        try:
            if self.serializer:
                results = self.serializer.load(filepath)
            elif COMMON_OPERATIONS_AVAILABLE:
                results = safe_json_load(filepath)
            else:
                with open(filepath, 'r') as f:
                    results = json.load(f)
            
            # Restore state
            if 'evaluation_history' in results:
                self.evaluation_history = results['evaluation_history']
            if 'best_architecture' in results:
                self.best_architecture = results['best_architecture']
            if 'best_metrics' in results and results['best_metrics']:
                self.best_metrics = ArchitectureMetrics(**results['best_metrics'])
            
            if TPRINT_AVAILABLE:
                tprint_success(f"📁 Results loaded from {filepath}")
            else:
                self.logger.info(f"📁 Results loaded from {filepath}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Error loading results: {e}")
            raise
    
    def cleanup(self):
        """Cleanup resources."""
        try:
            # Cleanup hardware optimizations
            if COMMON_OPERATIONS_AVAILABLE:
                cleanup_m1_optimizers()
            
            # Close file handles
            if hasattr(self, 'results_dir'):
                pass  # Results directory cleanup handled automatically
            
            if TPRINT_AVAILABLE:
                tprint_info("🧹 NASEvaluator cleanup completed")
            else:
                self.logger.info("🧹 NASEvaluator cleanup completed")
                
        except Exception as e:
            self.logger.error(f"❌ Error during cleanup: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
    
    def __del__(self):
        """Destructor."""
        try:
            self.cleanup()
        except:
            pass  # Ignore errors during destruction
    
    def evaluate_with_enhanced_utilities(self, 
                                       architecture: Dict[str, Any], 
                                       data: Union[pd.DataFrame, np.ndarray, Dict[str, Any]],
                                       target: Optional[Union[pd.Series, np.ndarray]] = None,
                                       evaluation_type: str = "comprehensive",
                                       **kwargs) -> Dict[str, Any]:
        """
        Evaluate architecture using enhanced hybrid NAS-TAS utilities.
        
        Args:
            architecture: Architecture definition
            data: Training data
            target: Target values (for supervised learning)
            evaluation_type: Type of evaluation (basic, trading, economic, comprehensive)
            **kwargs: Additional evaluation parameters
            
        Returns:
            Dict containing enhanced evaluation results
        """
        try:
            if not HYBRID_NAS_TAS_AVAILABLE:
                raise ValueError("Hybrid NAS-TAS utilities not available")
            
            if TPRINT_AVAILABLE:
                tprint_info(f"🔍 Enhanced evaluation: {evaluation_type}")
            else:
                self.logger.info(f"🔍 Enhanced evaluation: {evaluation_type}")
            
            # Use enhanced utility integration
            if self.enhanced_utils:
                enhanced_data = self.enhanced_utils.preprocess_data(data, target)
            else:
                enhanced_data = self._preprocess_data(data, target)
            
            # Use unified hardware manager for optimization
            if self.unified_hardware:
                with self.unified_hardware.get_optimization_context(WorkloadType.EVALUATION):
                    result = self._perform_enhanced_evaluation(architecture, enhanced_data, **kwargs)
            else:
                result = self._perform_enhanced_evaluation(architecture, enhanced_data, **kwargs)
            
            # Add enhanced metrics using ML integration
            if self.enhanced_ml:
                enhanced_metrics = self.enhanced_ml.analyze_performance(
                    architecture, enhanced_data, result
                )
                result.update(enhanced_metrics)
            
            if TPRINT_AVAILABLE:
                tprint_success("✅ Enhanced evaluation completed")
            else:
                self.logger.info("✅ Enhanced evaluation completed")
            
            return result
            
        except Exception as e:
            if TPRINT_AVAILABLE:
                tprint_error(f"❌ Enhanced evaluation failed: {e}")
            else:
                self.logger.error(f"❌ Enhanced evaluation failed: {e}")
            raise
    
    def get_enhanced_hardware_metrics(self) -> Optional[HardwareMetrics]:
        """Get enhanced hardware metrics using unified hardware manager."""
        try:
            if not HYBRID_NAS_TAS_AVAILABLE or not self.unified_hardware:
                return None
            
            return self.unified_hardware.get_current_metrics()
            
        except Exception as e:
            self.logger.error(f"❌ Error getting hardware metrics: {e}")
            return None
    
    def optimize_with_enhanced_utilities(self, 
                                      architecture: Dict[str, Any],
                                      data: Union[pd.DataFrame, np.ndarray, Dict[str, Any]],
                                      target: Optional[Union[pd.Series, np.ndarray]] = None,
                                      optimization_objective: str = "accuracy",
                                      **kwargs) -> Dict[str, Any]:
        """
        Optimize architecture using enhanced hybrid NAS-TAS utilities.
        
        Args:
            architecture: Architecture definition
            data: Training data
            target: Target values
            optimization_objective: Objective for optimization
            **kwargs: Additional optimization parameters
            
        Returns:
            Dict containing optimization results
        """
        try:
            if not HYBRID_NAS_TAS_AVAILABLE:
                raise ValueError("Hybrid NAS-TAS utilities not available")
            
            if TPRINT_AVAILABLE:
                tprint_info(f"🔧 Enhanced optimization: {optimization_objective}")
            else:
                self.logger.info(f"🔧 Enhanced optimization: {optimization_objective}")
            
            # Use enhanced ML integration for optimization
            if self.enhanced_ml:
                optimization_result = self.enhanced_ml.optimize_architecture(
                    architecture=architecture,
                    data=data,
                    target=target,
                    objective=optimization_objective,
                    **kwargs
                )
            else:
                # Fallback to standard optimization
                optimization_result = self._standard_optimization(architecture, data, target, **kwargs)
            
            if TPRINT_AVAILABLE:
                tprint_success("✅ Enhanced optimization completed")
            else:
                self.logger.info("✅ Enhanced optimization completed")
            
            return optimization_result
            
        except Exception as e:
            if TPRINT_AVAILABLE:
                tprint_error(f"❌ Enhanced optimization failed: {e}")
            else:
                self.logger.error(f"❌ Enhanced optimization failed: {e}")
            raise
    
    def _standard_optimization(self, 
                             architecture: Dict[str, Any],
                             data: Union[pd.DataFrame, np.ndarray, Dict[str, Any]],
                             target: Optional[Union[pd.Series, np.ndarray]] = None,
                             **kwargs) -> Dict[str, Any]:
        """Standard optimization fallback."""
        try:
            # Simple optimization using existing methods
            metrics = self.evaluate_architecture(architecture, data, target, **kwargs)
            
            return {
                'optimized_architecture': architecture,
                'metrics': metrics.__dict__,
                'optimization_success': True,
                'method': 'standard_fallback'
            }
            
        except Exception as e:
            self.logger.error(f"❌ Standard optimization failed: {e}")
            return {
                'optimized_architecture': architecture,
                'metrics': {},
                'optimization_success': False,
                'error': str(e),
                'method': 'standard_fallback'
            }
    
    def get_utility_integration_status(self) -> Dict[str, bool]:
        """Get status of utility integrations."""
        return {
            'common_operations': COMMON_OPERATIONS_AVAILABLE,
            'common_utilities': COMMON_UTILITIES_AVAILABLE,
            'math_validation': MATH_VALIDATION_AVAILABLE,
            'serialization': SERIALIZATION_AVAILABLE,
            'tprint': TPRINT_AVAILABLE,
            'm1_gpu': M1_GPU_AVAILABLE,
            'm1_memory': M1_MEMORY_AVAILABLE,
            'm1_cpu': M1_CPU_AVAILABLE,
            'klines_parquet': KLINES_PARQUET_AVAILABLE,
            'matrix_operations': MATRIX_OPERATIONS_AVAILABLE,
            'ml_common': ML_COMMON_AVAILABLE,
            'hybrid_nas_tas': HYBRID_NAS_TAS_AVAILABLE
        }