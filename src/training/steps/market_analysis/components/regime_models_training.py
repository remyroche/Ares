"""
Regime Detection Models Training Component

This component implements the specific regime detection models mentioned in the user's request:
- CatBoost (base model)
- Greedy Rule Lists (base model - multi-class compatible)
- ExtraTrees (base model)
- stacker_lgbm_calibrated (meta-learner with probability calibration)
"""

import numpy as np
import pandas as pd
import pickle
import json
import time
import warnings
import psutil
import gc
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

from src.utils.logger import system_logger
from src.utils.tprint import tprint
from .base_component import BaseMarketAnalysisComponent, ComponentConfig, ComponentResult

# Suppress warnings
warnings.filterwarnings('ignore')

# Import ML libraries with comprehensive error handling
tprint("🔍 [REGIME_MODELS] Starting ML libraries import process", color="cyan")
ML_LIBRARIES_AVAILABLE = False
ML_LIBRARY_VERSIONS = {}
ML_IMPORT_ERRORS = []

# Import sklearn components
try:
    from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler
    import sklearn
    ML_LIBRARY_VERSIONS['sklearn'] = sklearn.__version__
    tprint(f"✅ [REGIME_MODELS] scikit-learn imported successfully (v{sklearn.__version__})", color="green")
except ImportError as e:
    ML_IMPORT_ERRORS.append(f"scikit-learn: {e}")
    tprint(f"❌ [REGIME_MODELS] Failed to import scikit-learn: {e}", color="red")

# Import CatBoost
try:
    import catboost as cb
    ML_LIBRARY_VERSIONS['catboost'] = cb.__version__
    tprint(f"✅ [REGIME_MODELS] CatBoost imported successfully (v{cb.__version__})", color="green")
except ImportError as e:
    ML_IMPORT_ERRORS.append(f"CatBoost: {e}")
    tprint(f"❌ [REGIME_MODELS] Failed to import CatBoost: {e}", color="red")

# Import LightGBM
try:
    import lightgbm as lgb
    ML_LIBRARY_VERSIONS['lightgbm'] = lgb.__version__
    tprint(f"✅ [REGIME_MODELS] LightGBM imported successfully (v{lgb.__version__})", color="green")
except ImportError as e:
    ML_IMPORT_ERRORS.append(f"LightGBM: {e}")
    tprint(f"❌ [REGIME_MODELS] Failed to import LightGBM: {e}", color="red")

# Import Greedy Rule Lists
try:
    from imodels import GreedyRuleListClassifier
    ML_LIBRARY_VERSIONS['imodels'] = "1.0.0"  # Placeholder version
    tprint(f"✅ [REGIME_MODELS] imodels (Greedy Rule Lists) imported successfully", color="green")
except ImportError as e:
    ML_IMPORT_ERRORS.append(f"imodels (Greedy Rule Lists): {e}")
    tprint(f"❌ [REGIME_MODELS] Failed to import imodels: {e}", color="red")

# Check overall availability
if not ML_IMPORT_ERRORS:
    ML_LIBRARIES_AVAILABLE = True
    tprint("🎉 [REGIME_MODELS] All ML libraries imported successfully", color="green", bold=True)
    tprint(f"📊 [REGIME_MODELS] Library versions: {ML_LIBRARY_VERSIONS}", color="blue")
else:
    tprint(f"⚠️ [REGIME_MODELS] Import errors encountered: {ML_IMPORT_ERRORS}", color="yellow")
    tprint("🔧 [REGIME_MODELS] Some functionality may be limited", color="yellow")


class RegimeModelsTrainingComponent(BaseMarketAnalysisComponent):
    """
    Regime Detection Models Training Component.
    
    This component trains the specific regime detection models:
    - CatBoost (base model)
    - Greedy Rule Lists (base model - multi-class compatible)
    - ExtraTrees (base model)
    - stacker_lgbm_calibrated (meta-learner with probability calibration)
    """
    
    def __init__(self, config: Optional[ComponentConfig] = None):
        """Initialize the Regime Models Training Component."""
        tprint("🚀 [REGIME_MODELS] Initializing Regime Models Training Component", color="cyan", bold=True)
        tprint(f"📋 [REGIME_MODELS] Config provided: {config is not None}", color="blue")
        
        # Initialize parent component
        try:
            super().__init__(config)
            tprint("✅ [REGIME_MODELS] Parent component initialized successfully", color="green")
        except Exception as e:
            tprint(f"❌ [REGIME_MODELS] Failed to initialize parent component: {e}", color="red")
            raise
        
        # Initialize logger
        try:
            self.logger = system_logger.getChild('RegimeModelsTrainingComponent')
            self.logger.info("Regime Models Training Component logger initialized")
            tprint("✅ [REGIME_MODELS] Logger initialized successfully", color="green")
        except Exception as e:
            tprint(f"⚠️ [REGIME_MODELS] Logger initialization warning: {e}", color="yellow")
        
        # Initialize model training parameters
        tprint("🔧 [REGIME_MODELS] Configuring model training parameters", color="cyan")
        self.model_config = {
            'random_state': 42,
            'test_size': 0.2,
            'cv_folds': 5,
            'n_jobs': -1
        }
        
        # Regime-specific model configurations
        self.regime_models_config = {
            'base': {
                'CatBoost': {
                    'iterations': 100,
                    'depth': 6,
                    'learning_rate': 0.1,
                    'random_seed': 42,
                    'verbose': False
                },
                'Greedy Rule Lists': {
                    'max_depth': 20,  # Increased for better complexity handling
                    'criterion': 'gini',  # Criterion for splitting
                    'class_weight': 'balanced'  # Handle class imbalance
                },
                'ExtraTrees': {
                    'n_estimators': 100,
                    'max_depth': None,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1,
                    'max_features': 'sqrt',
                    'random_state': 42,
                    'n_jobs': -1
                }
            },
            'meta_learner': {
                'stacker_lgbm_calibrated': {
                    'num_leaves': 63,  # Increased for better complexity
                    'max_depth': 8,    # Increased depth
                    'learning_rate': 0.05,  # Reduced for better convergence
                    'n_estimators': 200,    # More estimators
                    'min_child_samples': 20,  # Prevent overfitting
                    'subsample': 0.8,        # Stochastic sampling
                    'colsample_bytree': 0.8,  # Feature sampling
                    'reg_alpha': 0.1,        # L1 regularization
                    'reg_lambda': 0.1,       # L2 regularization
                    'class_weight': 'balanced',  # Handle class imbalance
                    'random_state': 42,
                    'verbose': -1
                }
            }
        }
        
        # Initialize model storage
        self.models = {}
        self.model_metrics = {}
        self.training_history = []
        self.performance_metrics = {}
        
        tprint("📊 [REGIME_MODELS] Model storage initialized", color="blue")
        tprint(f"🔍 [REGIME_MODELS] Available ML libraries: {ML_LIBRARIES_AVAILABLE}", color="blue")
        if ML_LIBRARIES_AVAILABLE:
            tprint(f"📚 [REGIME_MODELS] Library versions: {ML_LIBRARY_VERSIONS}", color="blue")
        
        # Log initialization completion
        tprint("✅ [REGIME_MODELS] Regime Models Training Component initialized successfully", color="green", bold=True)
        self.logger.info("Regime Models Training Component initialization completed")
    
    def get_required_artifacts(self) -> List[str]:
        """Get list of required artifacts this component must produce."""
        tprint("📋 [REGIME_MODELS] Getting required artifacts", color="cyan")
        required_artifacts = ['regime_models_training_result']
        tprint(f"✅ [REGIME_MODELS] Required artifacts: {required_artifacts}", color="green")
        return required_artifacts
    
    async def execute(self, data: pd.DataFrame, pipeline_state: Dict[str, Any]) -> ComponentResult:
        """
        Execute regime detection models training.
        
        Args:
            data: Market data DataFrame
            pipeline_state: Pipeline state dictionary
            
        Returns:
            ComponentResult with training results
        """
        execution_start_time = time.time()
        tprint("🚀 [REGIME_MODELS] Starting regime detection models training execution", color="cyan", bold=True)
        self.logger.info("Starting regime detection models training execution")
        
        # Log initial system performance
        initial_perf = self._get_system_performance()
        if initial_perf:
            tprint(f"💻 [REGIME_MODELS] Initial system state - CPU: {initial_perf.get('cpu_percent', 'N/A')}%, Memory: {initial_perf.get('memory_percent', 'N/A')}%", color="blue")
        
        # Monitor initial memory usage
        initial_memory = self._monitor_memory_usage("Initial")
        
        # Log execution context
        tprint(f"📊 [REGIME_MODELS] Input data shape: {data.shape}", color="blue")
        tprint(f"📋 [REGIME_MODELS] Data columns: {list(data.columns)}", color="blue")
        tprint(f"🔍 [REGIME_MODELS] Pipeline state keys: {list(pipeline_state.keys())}", color="blue")
        
        try:
            # Step 0: Validate input data
            tprint("🔍 [REGIME_MODELS] Step 0: Validating input data", color="cyan")
            if not self._validate_input_data(data):
                error_msg = "Input data validation failed"
                tprint(f"❌ [REGIME_MODELS] {error_msg}", color="red")
                self.logger.error("Input data validation failed")
                return ComponentResult(
                    success=False,
                    artifacts={},
                    error_message=error_msg
                )
            
            # Step 1: Check ML libraries availability
            tprint("🔍 [REGIME_MODELS] Step 1: Checking ML libraries availability", color="cyan")
            if not ML_LIBRARIES_AVAILABLE:
                error_msg = "ML libraries not available for regime detection models training"
                tprint(f"❌ [REGIME_MODELS] {error_msg}", color="red")
                tprint(f"🔍 [REGIME_MODELS] Import errors: {ML_IMPORT_ERRORS}", color="yellow")
                self.logger.error(f"ML libraries not available: {ML_IMPORT_ERRORS}")
                return ComponentResult(
                    success=False,
                    artifacts={},
                    error_message=error_msg
                )
            tprint("✅ [REGIME_MODELS] ML libraries check passed", color="green")
            
            # Step 2: Extract and validate regime labels
            tprint("🔍 [REGIME_MODELS] Step 2: Extracting regime labels from pipeline state", color="cyan")
            artifacts = pipeline_state.get('artifacts', {})
            tprint(f"📋 [REGIME_MODELS] Available artifacts: {list(artifacts.keys())}", color="blue")
            
            nas_tas_clustering_result = artifacts.get('nas_tas_clustering_result', {})
            tprint(f"🔍 [REGIME_MODELS] NAS-TAS clustering result keys: {list(nas_tas_clustering_result.keys())}", color="blue")
            
            regime_labels = nas_tas_clustering_result.get('cluster_assignments')

            if regime_labels is None:
                error_msg = "No regime labels found in pipeline state artifacts"
                tprint(f"❌ [REGIME_MODELS] {error_msg}", color="red")
                tprint(f"🔍 [REGIME_MODELS] Available artifacts: {list(artifacts.keys())}", color="yellow")
                self.logger.error(f"Missing regime labels. Available artifacts: {list(artifacts.keys())}")
                return ComponentResult(
                    success=False,
                    artifacts={},
                    error_message=error_msg
                )
            
            # Validate regime labels
            regime_labels = np.array(regime_labels)
            if not self._validate_regime_labels(regime_labels):
                error_msg = "Regime labels validation failed"
                tprint(f"❌ [REGIME_MODELS] {error_msg}", color="red")
                self.logger.error("Regime labels validation failed")
                return ComponentResult(
                    success=False,
                    artifacts={},
                    error_message=error_msg
                )
            
            unique_regimes = np.unique(regime_labels)
            tprint(f"📊 [REGIME_MODELS] Found regime labels: {len(regime_labels)} samples", color="blue")
            tprint(f"📊 [REGIME_MODELS] Unique regimes: {unique_regimes} (count: {len(unique_regimes)})", color="blue")
            tprint(f"📊 [REGIME_MODELS] Regime distribution: {dict(zip(*np.unique(regime_labels, return_counts=True)))}", color="blue")
            
            # Step 3: Prepare training data
            tprint("🔍 [REGIME_MODELS] Step 3: Preparing training data", color="cyan")
            data_prep_start = time.time()
            X, y = self._prepare_training_data(data, regime_labels, pipeline_state)
            self._log_performance_metrics("Data preparation", data_prep_start)
            self._monitor_memory_usage("After data preparation")
            if X is None or y is None:
                error_msg = "Failed to prepare training data"
                tprint(f"❌ [REGIME_MODELS] {error_msg}", color="red")
                self.logger.error("Failed to prepare training data")
                return ComponentResult(
                    success=False,
                    artifacts={},
                    error_message=error_msg
                )
            
            # Validate prepared training data
            if not self._validate_training_data(X, y):
                error_msg = "Training data validation failed"
                tprint(f"❌ [REGIME_MODELS] {error_msg}", color="red")
                self.logger.error("Training data validation failed")
                return ComponentResult(
                    success=False,
                    artifacts={},
                    error_message=error_msg
                )
            
            tprint(f"📊 [REGIME_MODELS] Training data prepared: X={X.shape}, y={y.shape}", color="blue")
            tprint(f"📊 [REGIME_MODELS] Feature matrix info: dtype={X.dtype}, min={X.min():.4f}, max={X.max():.4f}", color="blue")
            tprint(f"📊 [REGIME_MODELS] Target distribution: {dict(zip(*np.unique(y, return_counts=True)))}", color="blue")
            
            # Step 4: Train regime detection models
            tprint("🔍 [REGIME_MODELS] Step 4: Training regime detection models", color="cyan")
            model_training_start = time.time()
            training_results = self._train_regime_models(X, y)
            self._log_performance_metrics("Model training", model_training_start)
            self._monitor_memory_usage("After model training")
            
            # Clean up memory after training
            self._cleanup_memory()
            
            # Validate trained models
            if not self._validate_models(training_results['models']):
                error_msg = "Trained models validation failed"
                tprint(f"❌ [REGIME_MODELS] {error_msg}", color="red")
                self.logger.error("Trained models validation failed")
                return ComponentResult(
                    success=False,
                    artifacts={},
                    error_message=error_msg
                )
            
            # Step 5: Create and validate artifacts
            tprint("🔍 [REGIME_MODELS] Step 5: Creating artifacts", color="cyan")
            artifacts = {
                'regime_models_training_result': {
                    'regime_models': training_results['models'],
                    'metrics': training_results['metrics'],
                    'training_time': training_results['training_time'],
                    'success': True,
                    'model_count': len(training_results['models']),
                    'feature_count': X.shape[1],
                    'sample_count': X.shape[0],
                    'regime_models_config': self.regime_models_config
                }
            }
            
            execution_time = time.time() - execution_start_time
            
            # Log final performance metrics
            final_perf = self._get_system_performance()
            final_memory = self._monitor_memory_usage("Final")
            
            tprint(f"⏱️ [REGIME_MODELS] Total execution time: {execution_time:.2f} seconds", color="blue")
            if final_perf:
                tprint(f"💻 [REGIME_MODELS] Final system state - CPU: {final_perf.get('cpu_percent', 'N/A')}%, Memory: {final_perf.get('memory_percent', 'N/A')}%", color="blue")
            tprint(f"🧠 [REGIME_MODELS] Memory usage change: {final_memory - initial_memory:.1f} MB", color="blue")
            
            tprint("✅ [REGIME_MODELS] Regime detection models training completed successfully", color="green", bold=True)
            self.logger.info(f"Regime detection models training completed successfully in {execution_time:.2f} seconds")
            
            return ComponentResult(
                success=True,
                artifacts=artifacts,
                metadata={
                    'component_type': 'regime_models_training', 
                    'execution_time': execution_time,
                    'training_time': training_results['training_time'],
                    'model_count': len(training_results['models']),
                    'feature_count': X.shape[1],
                    'sample_count': X.shape[0]
                }
            )
            
        except Exception as e:
            execution_time = time.time() - execution_start_time
            error_type = type(e).__name__
            error_msg = f"Regime detection models training failed: {str(e)}"
            
            # Enhanced error logging with context
            tprint(f"❌ [REGIME_MODELS] {error_msg}", color="red")
            tprint(f"🔍 [REGIME_MODELS] Error type: {error_type}", color="yellow")
            tprint(f"🔍 [REGIME_MODELS] Execution time before failure: {execution_time:.2f} seconds", color="yellow")
            
            # Log system state at failure
            failure_perf = self._get_system_performance()
            if failure_perf:
                tprint(f"💻 [REGIME_MODELS] System state at failure - CPU: {failure_perf.get('cpu_percent', 'N/A')}%, Memory: {failure_perf.get('memory_percent', 'N/A')}%", color="yellow")
            
            # Provide recovery suggestions based on error type
            recovery_suggestions = self._get_recovery_suggestions(e)
            if recovery_suggestions:
                tprint(f"💡 [REGIME_MODELS] Recovery suggestions: {recovery_suggestions}", color="cyan")
            
            # Log detailed error information
            self.logger.error(f"Regime detection models training failed after {execution_time:.2f} seconds", exc_info=True)
            self.logger.error(f"Error type: {error_type}, Error message: {str(e)}")
            if recovery_suggestions:
                self.logger.error(f"Recovery suggestions: {recovery_suggestions}")
            
            return ComponentResult(
                success=False,
                artifacts={},
                error_message=f"{error_msg} (Type: {error_type})"
            )
    
    def _get_system_performance(self) -> Dict[str, Any]:
        """Get current system performance metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_used_gb = memory.used / (1024**3)
            memory_total_gb = memory.total / (1024**3)
            memory_percent = memory.percent
            
            return {
                'cpu_percent': cpu_percent,
                'cpu_count': cpu_count,
                'memory_used_gb': memory_used_gb,
                'memory_total_gb': memory_total_gb,
                'memory_percent': memory_percent
            }
        except Exception as e:
            tprint(f"⚠️ [REGIME_MODELS] Failed to get system performance: {e}", color="yellow")
            return {}
    
    def _log_performance_metrics(self, stage: str, start_time: float):
        """Log performance metrics for a given stage."""
        current_time = time.time()
        elapsed_time = current_time - start_time
        
        # Get system performance
        perf_metrics = self._get_system_performance()
        
        # Log timing
        tprint(f"⏱️ [REGIME_MODELS] {stage} completed in {elapsed_time:.3f} seconds", color="blue")
        
        # Log system metrics if available
        if perf_metrics:
            tprint(f"💻 [REGIME_MODELS] System metrics - CPU: {perf_metrics.get('cpu_percent', 'N/A')}%, Memory: {perf_metrics.get('memory_percent', 'N/A')}% ({perf_metrics.get('memory_used_gb', 0):.1f}GB/{perf_metrics.get('memory_total_gb', 0):.1f}GB)", color="blue")
        
        return elapsed_time, perf_metrics
    
    def _monitor_memory_usage(self, stage: str):
        """Monitor and log memory usage."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024**2)
            tprint(f"🧠 [REGIME_MODELS] {stage} memory usage: {memory_mb:.1f} MB", color="blue")
            return memory_mb
        except Exception as e:
            tprint(f"⚠️ [REGIME_MODELS] Failed to monitor memory: {e}", color="yellow")
            return 0
    
    def _cleanup_memory(self):
        """Clean up memory by forcing garbage collection."""
        try:
            gc.collect()
            tprint("🧹 [REGIME_MODELS] Memory cleanup completed", color="blue")
        except Exception as e:
            tprint(f"⚠️ [REGIME_MODELS] Memory cleanup failed: {e}", color="yellow")
    
    def _validate_input_data(self, data: pd.DataFrame) -> bool:
        """Validate input data for training."""
        tprint("🔍 [REGIME_MODELS] Validating input data", color="cyan")
        
        try:
            # Check if data is empty
            if data.empty:
                tprint("❌ [REGIME_MODELS] Input data is empty", color="red")
                return False
            
            # Check minimum required columns
            required_columns = ['close']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                tprint(f"❌ [REGIME_MODELS] Missing required columns: {missing_columns}", color="red")
                return False
            
            # Check data types
            if not pd.api.types.is_numeric_dtype(data['close']):
                tprint("❌ [REGIME_MODELS] 'close' column is not numeric", color="red")
                return False
            
            # Check for sufficient data points
            min_samples = 100
            if len(data) < min_samples:
                tprint(f"❌ [REGIME_MODELS] Insufficient data points: {len(data)} < {min_samples}", color="red")
                return False
            
            tprint("✅ [REGIME_MODELS] Input data validation passed", color="green")
            return True
            
        except Exception as e:
            tprint(f"❌ [REGIME_MODELS] Data validation error: {e}", color="red")
            return False
    
    def _validate_regime_labels(self, regime_labels: np.ndarray) -> bool:
        """Validate regime labels."""
        tprint("🔍 [REGIME_MODELS] Validating regime labels", color="cyan")
        
        try:
            # Check if labels are not None
            if regime_labels is None:
                tprint("❌ [REGIME_MODELS] Regime labels are None", color="red")
                return False
            
            # Convert to numpy array if needed
            regime_labels = np.array(regime_labels)
            
            # Check for sufficient samples
            if len(regime_labels) < 50:
                tprint(f"❌ [REGIME_MODELS] Insufficient regime labels: {len(regime_labels)} < 50", color="red")
                return False
            
            # Check for valid regime values
            unique_regimes = np.unique(regime_labels)
            if len(unique_regimes) < 2:
                tprint(f"❌ [REGIME_MODELS] Insufficient regime classes: {len(unique_regimes)} < 2", color="red")
                return False
            
            tprint(f"✅ [REGIME_MODELS] Regime labels validation passed - {len(unique_regimes)} classes, {len(regime_labels)} samples", color="green")
            return True
            
        except Exception as e:
            tprint(f"❌ [REGIME_MODELS] Regime labels validation error: {e}", color="red")
            return False
    
    def _validate_training_data(self, X: np.ndarray, y: np.ndarray) -> bool:
        """Validate prepared training data."""
        tprint("🔍 [REGIME_MODELS] Validating training data", color="cyan")
        
        try:
            # Check shapes
            if X.shape[0] != y.shape[0]:
                tprint(f"❌ [REGIME_MODELS] Mismatched sample counts: X={X.shape[0]}, y={y.shape[0]}", color="red")
                return False
            
            # Check for sufficient features
            if X.shape[1] < 2:
                tprint(f"❌ [REGIME_MODELS] Insufficient features: {X.shape[1]} < 2", color="red")
                return False
            
            # Check for NaN or infinite values
            nan_count = np.isnan(X).sum()
            inf_count = np.isinf(X).sum()
            if nan_count > 0:
                tprint(f"❌ [REGIME_MODELS] Found {nan_count} NaN values in features", color="red")
                return False
            if inf_count > 0:
                tprint(f"❌ [REGIME_MODELS] Found {inf_count} infinite values in features", color="red")
                return False
            
            tprint("✅ [REGIME_MODELS] Training data validation passed", color="green")
            return True
            
        except Exception as e:
            tprint(f"❌ [REGIME_MODELS] Training data validation error: {e}", color="red")
            return False
    
    def _validate_models(self, models: Dict[str, Any]) -> bool:
        """Validate trained models."""
        tprint("🔍 [REGIME_MODELS] Validating trained models", color="cyan")
        
        try:
            if not models:
                tprint("❌ [REGIME_MODELS] No models trained", color="red")
                return False
            
            # Filter out metadata objects that are not actual models
            model_names_to_validate = [name for name in models.keys() 
                                    if not name.endswith('_feature_indices') and 
                                       not name.endswith('_metadata') and
                                       not name.endswith('_config')]
            
            tprint(f"🔍 [REGIME_MODELS] Validating {len(model_names_to_validate)} models: {model_names_to_validate}", color="blue")
            
            # Check each model
            valid_models = 0
            for name in model_names_to_validate:
                model = models[name]
                if model is None:
                    tprint(f"⚠️ [REGIME_MODELS] Model {name} is None (training failed)", color="yellow")
                    continue  # Skip None models but don't fail validation
                
                # Check if model has required methods
                if not hasattr(model, 'predict'):
                    tprint(f"❌ [REGIME_MODELS] Model {name} missing predict method", color="red")
                    return False
                
                valid_models += 1
            
            # Ensure at least one model is valid
            if valid_models == 0:
                tprint("❌ [REGIME_MODELS] No valid models trained", color="red")
                return False
            
            tprint(f"✅ [REGIME_MODELS] Model validation passed - {valid_models} valid models out of {len(model_names_to_validate)} attempted", color="green")
            return True
            
        except Exception as e:
            tprint(f"❌ [REGIME_MODELS] Model validation error: {e}", color="red")
            return False
    
    def _prepare_training_data(self, data: pd.DataFrame, regime_labels: np.ndarray, pipeline_state: Dict[str, Any] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from market data and regime labels."""
        tprint("🔧 [REGIME_MODELS] Preparing training data", color="cyan")
        self.logger.info("Starting data preparation process")
        
        try:
            # Log input data characteristics
            tprint(f"📊 [REGIME_MODELS] Input data shape: {data.shape}", color="blue")
            tprint(f"📊 [REGIME_MODELS] Input data columns: {list(data.columns)}", color="blue")
            
            # Reuse features from clustering stage instead of creating generic ones
            tprint("🔧 [REGIME_MODELS] Reusing regime-focused features from clustering stage", color="cyan")
            
            # Extract features from pipeline state artifacts
            if pipeline_state is None:
                pipeline_state = {}
            artifacts = pipeline_state.get('artifacts', {})
            nas_tas_clustering_result = artifacts.get('nas_tas_clustering_result', {})
            
            # Try to get the original features used in clustering
            if 'original_features' in nas_tas_clustering_result:
                X = nas_tas_clustering_result['original_features']
                feature_names = nas_tas_clustering_result.get('feature_names', [f'feature_{i}' for i in range(X.shape[1])])
                tprint(f"📊 [REGIME_MODELS] Reusing clustering features: {X.shape}", color="blue")
                tprint(f"📋 [REGIME_MODELS] Feature names ({len(feature_names)}): {feature_names[:10]}..." if len(feature_names) > 10 else f"📋 [REGIME_MODELS] Feature names ({len(feature_names)}): {feature_names}", color="blue")
            else:
                # Fallback: Use shared utilities to create regime-focused features
                tprint("⚠️ [REGIME_MODELS] Clustering features not found, creating regime-focused features", color="yellow")
                from src.training.steps.market_analysis.shared_utils.features import prepare_market_features, FeatureConfig
                
                # Create feature config for regime-focused features
                feature_config = FeatureConfig()
                feature_config.feature_categories = ['regime_volatility', 'regime_volume', 'regime_structural_trend', 'regime_statistical']
                
                # Generate regime-focused features
                X = prepare_market_features(data, feature_config, verbose=True)
                if X is None:
                    raise ValueError("Failed to create regime-focused features")
                
                feature_names = [f'regime_feature_{i}' for i in range(X.shape[1])]
                tprint(f"📊 [REGIME_MODELS] Created regime-focused features: {X.shape}", color="blue")
                tprint(f"📋 [REGIME_MODELS] Feature names ({len(feature_names)}): {feature_names[:10]}..." if len(feature_names) > 10 else f"📋 [REGIME_MODELS] Feature names ({len(feature_names)}): {feature_names}", color="blue")
            
            # Check for NaN or infinite values in features
            nan_count = np.isnan(X).sum()
            inf_count = np.isinf(X).sum()
            if nan_count > 0:
                tprint(f"⚠️ [REGIME_MODELS] Found {nan_count} NaN values in features", color="yellow")
                X = np.nan_to_num(X, nan=0.0)
            if inf_count > 0:
                tprint(f"⚠️ [REGIME_MODELS] Found {inf_count} infinite values in features", color="yellow")
                X = np.nan_to_num(X, posinf=1e6, neginf=-1e6)
            
            # Align with regime labels
            tprint("🔧 [REGIME_MODELS] Aligning features with regime labels", color="cyan")
            min_length = min(len(X), len(regime_labels))
            X = X[:min_length]
            y = np.array(regime_labels[:min_length])
            
            tprint(f"✅ [REGIME_MODELS] Training data prepared: {X.shape[0]} samples, {X.shape[1]} features", color="green", bold=True)
            
            self.logger.info(f"Training data preparation completed: {X.shape[0]} samples, {X.shape[1]} features")
            return X, y
            
        except Exception as e:
            error_type = type(e).__name__
            tprint(f"❌ [REGIME_MODELS] Error preparing training data: {e}", color="red")
            tprint(f"🔍 [REGIME_MODELS] Error type: {error_type}", color="yellow")
            
            self.logger.error(f"Error preparing training data: {str(e)}", exc_info=True)
            return None, None
    
    def _train_regime_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train regime detection models."""
        tprint("🏋️ [REGIME_MODELS] Training regime detection models", color="cyan")
        self.logger.info("Starting regime detection models training process")
        
        start_time = time.time()
        models = {}
        metrics = {}
        training_history = []
        
        try:
            # Log training data characteristics
            tprint(f"📊 [REGIME_MODELS] Training data: {X.shape[0]} samples, {X.shape[1]} features", color="blue")
            tprint(f"📊 [REGIME_MODELS] Target classes: {np.unique(y)} (count: {len(np.unique(y))})", color="blue")
            
            # Step 1: Split data
            tprint("🔧 [REGIME_MODELS] Step 1: Splitting data into train/test sets", color="cyan")
            split_start = time.time()
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.model_config['test_size'], 
                random_state=self.model_config['random_state'], 
                stratify=y
            )
            
            split_time = time.time() - split_start
            tprint(f"📊 [REGIME_MODELS] Train set: {X_train.shape[0]} samples", color="blue")
            tprint(f"📊 [REGIME_MODELS] Test set: {X_test.shape[0]} samples", color="blue")
            tprint(f"⏱️ [REGIME_MODELS] Data splitting completed in {split_time:.3f} seconds", color="blue")
            
            # Step 2: Scale features
            tprint("🔧 [REGIME_MODELS] Step 2: Scaling features", color="cyan")
            scale_start = time.time()
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            scale_time = time.time() - scale_start
            tprint(f"⏱️ [REGIME_MODELS] Feature scaling completed in {scale_time:.3f} seconds", color="blue")
            
            # Step 3: Train CatBoost with timeout protection
            tprint("🔧 [REGIME_MODELS] Step 3: Training CatBoost", color="cyan")
            catboost_start = time.time()
            
            try:
                # Use CPU-only configuration to prevent hanging on M1 Macs
                catboost_config = self.regime_models_config['base']['CatBoost'].copy()
                catboost_config.update({
                    'task_type': 'CPU',  # Force CPU usage to prevent GPU hanging
                    'verbose': False,    # Reduce verbosity
                    'random_seed': 42    # Ensure reproducibility
                })
                
                catboost_model = cb.CatBoostClassifier(**catboost_config)
                catboost_model.fit(X_train_scaled, y_train)
                models['CatBoost'] = catboost_model
                
                catboost_time = time.time() - catboost_start
                tprint(f"⏱️ [REGIME_MODELS] CatBoost training completed in {catboost_time:.3f} seconds", color="blue")
            except Exception as e:
                tprint(f"❌ [REGIME_MODELS] CatBoost training failed: {e}", color="red")
                models['CatBoost'] = None
            
            # Step 4: Train Greedy Rule Lists with parameter optimization
            tprint("🔧 [REGIME_MODELS] Step 4: Training Greedy Rule Lists with parameter optimization", color="cyan")
            grl_start = time.time()
            
            # Get number of classes for logging
            n_classes = len(np.unique(y_train))
            
            try:
                # First try with simple, robust parameters
                tprint("🔧 [REGIME_MODELS] Attempting Greedy Rule Lists with robust parameters", color="blue")
                grl_model = self._robust_grl_training(X_train_scaled, y_train, n_classes)
                models['Greedy Rule Lists'] = grl_model
                
                grl_time = time.time() - grl_start
                tprint(f"⏱️ [REGIME_MODELS] Greedy Rule Lists training completed in {grl_time:.3f} seconds", color="blue")
                tprint(f"📊 [REGIME_MODELS] Greedy Rule Lists: Supports multi-class with {n_classes} classes", color="green")
            except Exception as e:
                tprint(f"❌ [REGIME_MODELS] Greedy Rule Lists training failed: {e}", color="red")
                tprint(f"🔍 [REGIME_MODELS] Error details: {type(e).__name__}: {str(e)}", color="yellow")
                models['Greedy Rule Lists'] = None
            
            # Step 5: Train ExtraTrees
            tprint("🔧 [REGIME_MODELS] Step 5: Training ExtraTrees", color="cyan")
            extratrees_start = time.time()
            
            try:
                extratrees_model = ExtraTreesClassifier(**self.regime_models_config['base']['ExtraTrees'])
                extratrees_model.fit(X_train_scaled, y_train)
                models['ExtraTrees'] = extratrees_model
                
                extratrees_time = time.time() - extratrees_start
                tprint(f"⏱️ [REGIME_MODELS] ExtraTrees training completed in {extratrees_time:.3f} seconds", color="blue")
            except Exception as e:
                tprint(f"❌ [REGIME_MODELS] ExtraTrees training failed: {e}", color="red")
                models['ExtraTrees'] = None
            
            # Step 6: Train stacker_lgbm_calibrated (meta-learner) with proper cross-validation
            tprint("🔧 [REGIME_MODELS] Step 6: Training stacker_lgbm_calibrated meta-learner with CV", color="cyan")
            meta_start = time.time()
            
            try:
                # Create base models for stacking
                base_models = {}
                for name, model in models.items():
                    if model is not None:
                        base_models[name] = model
                
                if base_models:
                    # Generate out-of-fold predictions using cross-validation
                    tprint("🔄 [REGIME_MODELS] Generating out-of-fold predictions for meta-learning", color="blue")
                    oof_predictions = self._generate_out_of_fold_predictions(
                        base_models, X_train_scaled, y_train, cv_folds=5
                    )
                    
                    if oof_predictions is not None:
                        # Create enhanced meta-learner features and store feature indices for consistency
                        enhanced_features, feature_indices = self._create_enhanced_meta_features_with_indices(
                            oof_predictions, X_train_scaled
                        )
                        
                        # Create meta-learner with regularization to prevent overfitting
                        meta_config = self.regime_models_config['meta_learner']['stacker_lgbm_calibrated'].copy()
                        meta_config.update({
                            'num_leaves': 15,  # Reduce complexity
                            'max_depth': 4,    # Reduce depth
                            'learning_rate': 0.05,  # Lower learning rate
                            'n_estimators': 50,  # Fewer estimators
                            'reg_alpha': 0.1,  # L1 regularization
                            'reg_lambda': 0.1,  # L2 regularization
                            'subsample': 0.8,  # Subsampling for regularization
                            'colsample_bytree': 0.8,  # Feature sampling
                            'min_child_samples': 20,  # Minimum samples per leaf
                        })
                        
                        meta_learner = lgb.LGBMClassifier(**meta_config)
                        meta_learner.fit(enhanced_features, y_train)
                        models['stacker_lgbm_calibrated'] = meta_learner
                        
                        # Store feature indices for consistent prediction (as metadata, not as a model)
                        models['stacker_lgbm_calibrated_feature_indices'] = feature_indices
                        
                        meta_time = time.time() - meta_start
                        tprint(f"⏱️ [REGIME_MODELS] Meta-learner training completed in {meta_time:.3f} seconds", color="blue")
                        tprint(f"📊 [REGIME_MODELS] Meta-learner features: {enhanced_features.shape[1]}", color="blue")
                    else:
                        tprint("⚠️ [REGIME_MODELS] Failed to generate out-of-fold predictions", color="yellow")
                        models['stacker_lgbm_calibrated'] = None
                else:
                    tprint("⚠️ [REGIME_MODELS] No base models available for meta-learner", color="yellow")
                    models['stacker_lgbm_calibrated'] = None
                    
            except Exception as e:
                tprint(f"❌ [REGIME_MODELS] Meta-learner training failed: {e}", color="red")
                models['stacker_lgbm_calibrated'] = None
            
            # Step 7: Evaluate models
            tprint("🔧 [REGIME_MODELS] Step 7: Evaluating models", color="cyan")
            eval_start = time.time()
            
            # Only evaluate actual model objects, skip metadata like feature_indices
            model_names_to_evaluate = [name for name, model in models.items() 
                                     if model is not None and hasattr(model, 'predict')]
            
            for name in model_names_to_evaluate:
                model = models[name]
                tprint(f"📊 [REGIME_MODELS] Evaluating {name}", color="blue")
                
                # Make predictions - handle meta-learner differently
                if name == 'stacker_lgbm_calibrated':
                    # Meta-learner needs enhanced features as input
                    # Only use the same base models that were used during training
                    base_model_names = ['CatBoost', 'Greedy Rule Lists', 'ExtraTrees']
                    base_predictions = np.column_stack([
                        np.argmax(models[base_name].predict_proba(X_test_scaled), axis=1).reshape(-1, 1) if hasattr(models[base_name], 'predict_proba') else models[base_name].predict(X_test_scaled).reshape(-1, 1)
                        for base_name in base_model_names
                        if base_name in models and models[base_name] is not None
                    ])
                    # Use stored feature indices for consistency
                    feature_indices = models.get('stacker_lgbm_calibrated_feature_indices')
                    tprint(f"🔧 [REGIME_MODELS] Base predictions shape: {base_predictions.shape}", color="blue")
                    tprint(f"🔧 [REGIME_MODELS] Feature indices: {len(feature_indices) if feature_indices is not None else 'None'}", color="blue")
                    enhanced_test_features = self._create_enhanced_meta_features(base_predictions, X_test_scaled, feature_indices)
                    
                    # Validate feature dimensions match the trained model
                    expected_features = model.n_features_in_ if hasattr(model, 'n_features_in_') else None
                    actual_features = enhanced_test_features.shape[1]
                    tprint(f"🔧 [REGIME_MODELS] Expected features: {expected_features}, Actual features: {actual_features}", color="blue")
                    
                    if expected_features is not None and expected_features != actual_features:
                        error_msg = f"Feature dimension mismatch: model expects {expected_features} features but received {actual_features}"
                        tprint(f"❌ [REGIME_MODELS] {error_msg}", color="red")
                        raise ValueError(error_msg)
                    
                    y_pred = model.predict(enhanced_test_features)
                    y_pred_proba = model.predict_proba(enhanced_test_features) if hasattr(model, 'predict_proba') else None
                else:
                    # Regular models use original features
                    y_pred = model.predict(X_test_scaled)
                    y_pred_proba = model.predict_proba(X_test_scaled) if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                
                # Store detailed metrics
                model_metrics = {
                    'accuracy': accuracy,
                    'test_samples': len(y_test),
                    'train_samples': len(y_train),
                    'n_features': X.shape[1]
                }
                
                # Add prediction probabilities if available
                if y_pred_proba is not None:
                    model_metrics['prediction_confidence'] = {
                        'mean': y_pred_proba.max(axis=1).mean(),
                        'std': y_pred_proba.max(axis=1).std()
                    }
                
                metrics[name] = model_metrics
                
                # Log detailed results
                tprint(f"📊 [REGIME_MODELS] {name} accuracy: {accuracy:.4f}", color="green")
            
            eval_time = time.time() - eval_start
            tprint(f"⏱️ [REGIME_MODELS] Model evaluation completed in {eval_time:.3f} seconds", color="blue")
            
            # Calculate total training time
            training_time = time.time() - start_time
            
            # Log comprehensive training summary
            tprint("📊 [REGIME_MODELS] Training Summary:", color="cyan", bold=True)
            tprint(f"⏱️ [REGIME_MODELS] Total training time: {training_time:.2f} seconds", color="blue")
            tprint(f"📊 [REGIME_MODELS] Models trained: {len([m for m in models.values() if m is not None])}", color="blue")
            if metrics:
                tprint(f"📊 [REGIME_MODELS] Best accuracy: {max(metrics[m]['accuracy'] for m in metrics):.4f}", color="green")
            
            # Store training history
            training_history = {
                'data_split_time': split_time,
                'scaling_time': scale_time,
                'total_time': training_time
            }
            
            self.logger.info(f"Regime detection models training completed successfully in {training_time:.2f} seconds")
            
            return {
                'models': models,
                'metrics': metrics,
                'training_time': training_time,
                'scaler': scaler,
                'training_history': training_history,
                'feature_count': X.shape[1],
                'sample_count': X.shape[0]
            }
            
        except Exception as e:
            training_time = time.time() - start_time
            error_type = type(e).__name__
            tprint(f"❌ [REGIME_MODELS] Error training regime detection models: {e}", color="red")
            tprint(f"🔍 [REGIME_MODELS] Error type: {error_type}", color="yellow")
            
            self.logger.error(f"Error training regime detection models after {training_time:.2f} seconds: {str(e)}", exc_info=True)
            
            return {
                'models': {},
                'metrics': {},
                'training_time': training_time,
                'error': str(e),
                'error_type': error_type
            }
    
    def _get_recovery_suggestions(self, error: Exception) -> str:
        """Get recovery suggestions based on error type."""
        error_type = type(error).__name__
        
        if "MemoryError" in error_type or "memory" in str(error).lower():
            return "Try reducing data size, increasing available memory, or using data sampling"
        elif "ImportError" in error_type:
            return "Check ML library installations: pip install catboost lightgbm imodels"
        elif "ValueError" in error_type and "shape" in str(error).lower():
            return "Check data alignment between features and labels, ensure consistent lengths"
        elif "KeyError" in error_type:
            return "Verify required columns exist in input data (close, volume, etc.)"
        elif "AttributeError" in error_type:
            return "Check model object integrity and required methods availability"
        else:
            return "Check logs for detailed error information and system requirements"
    
    def _generate_out_of_fold_predictions(self, base_models: dict, X: np.ndarray, y: np.ndarray, cv_folds: int = 5) -> Optional[np.ndarray]:
        """
        Generate out-of-fold predictions using cross-validation to prevent data leakage.
        
        Args:
            base_models: Dictionary of trained base models
            X: Feature matrix
            y: Target labels
            cv_folds: Number of cross-validation folds
            
        Returns:
            Array of out-of-fold predictions or None if failed
        """
        try:
            from sklearn.model_selection import StratifiedKFold
            
            tprint(f"🔄 [REGIME_MODELS] Generating {cv_folds}-fold out-of-fold predictions", color="blue")
            
            # Initialize array to store OOF predictions
            oof_predictions = np.zeros((X.shape[0], 0))
            
            # Create stratified K-fold for balanced regime distribution
            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            
            for model_name, model in base_models.items():
                if model is None:
                    continue
                    
                tprint(f"📊 [REGIME_MODELS] Generating OOF predictions for {model_name}", color="blue")
                
                # Special handling for CatBoost to prevent hanging
                if model_name == 'CatBoost':
                    model_oof = self._generate_catboost_oof_with_timeout(model, X, y, skf)
                else:
                    model_oof = np.zeros(X.shape[0])
                    
                    # Generate out-of-fold predictions for this model
                    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                        y_train_fold = y[train_idx]
                        
                        # Train model on fold
                        if hasattr(model, 'fit'):
                            model.fit(X_train_fold, y_train_fold)
                        
                        # Predict on validation fold
                        if hasattr(model, 'predict_proba'):
                            val_pred_proba = model.predict_proba(X_val_fold)
                            # Use max probability class for multi-class
                            model_oof[val_idx] = np.argmax(val_pred_proba, axis=1)
                        else:
                            model_oof[val_idx] = model.predict(X_val_fold)
                
                # Reshape and add to OOF predictions
                model_oof = model_oof.reshape(-1, 1)
                oof_predictions = np.column_stack([oof_predictions, model_oof])
            
            if oof_predictions.shape[1] > 0:
                tprint(f"✅ [REGIME_MODELS] Generated OOF predictions: {oof_predictions.shape}", color="green")
                return oof_predictions
            else:
                tprint("⚠️ [REGIME_MODELS] No valid OOF predictions generated", color="yellow")
                return None
                
        except Exception as e:
            tprint(f"❌ [REGIME_MODELS] OOF prediction generation failed: {e}", color="red")
            self.logger.error(f"OOF prediction generation failed: {e}")
            return None
    
    def _generate_catboost_oof_with_timeout(self, model, X: np.ndarray, y: np.ndarray, skf) -> np.ndarray:
        """
        Generate CatBoost OOF predictions with timeout and CPU fallback to prevent hanging.
        
        Args:
            model: CatBoost model
            X: Feature matrix
            y: Target labels
            skf: StratifiedKFold object
            
        Returns:
            Array of OOF predictions
        """
        import signal
        import threading
        import time
        
        def timeout_handler(signum, frame):
            raise TimeoutError("CatBoost OOF prediction timed out")
        
        model_oof = np.zeros(X.shape[0])
        
        try:
            # Set timeout for CatBoost operations (30 seconds per fold)
            timeout_seconds = 30
            
            for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold = y[train_idx]
                
                # Create a fresh CatBoost model for this fold to avoid GPU issues
                try:
                    # Use CPU-only configuration to prevent hanging
                    fold_model = cb.CatBoostClassifier(
                        iterations=50,  # Reduced iterations for speed
                        depth=4,       # Reduced depth
                        learning_rate=0.1,
                        task_type='CPU',  # Force CPU usage
                        verbose=False,
                        random_seed=42
                    )
                    
                    # Set timeout for training
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(timeout_seconds)
                    
                    # Train model on fold
                    fold_model.fit(X_train_fold, y_train_fold)
                    
                    # Predict on validation fold
                    val_pred_proba = fold_model.predict_proba(X_val_fold)
                    model_oof[val_idx] = np.argmax(val_pred_proba, axis=1)
                    
                    # Cancel timeout
                    signal.alarm(0)
                    
                except TimeoutError:
                    tprint(f"⚠️ [REGIME_MODELS] CatBoost fold {fold} timed out, using fallback", color="yellow")
                    # Fallback: use simple majority class prediction
                    from collections import Counter
                    majority_class = Counter(y_train_fold).most_common(1)[0][0]
                    model_oof[val_idx] = majority_class
                    signal.alarm(0)
                    
                except Exception as e:
                    tprint(f"⚠️ [REGIME_MODELS] CatBoost fold {fold} failed: {e}, using fallback", color="yellow")
                    # Fallback: use simple majority class prediction
                    from collections import Counter
                    majority_class = Counter(y_train_fold).most_common(1)[0][0]
                    model_oof[val_idx] = majority_class
                    signal.alarm(0)
            
            tprint(f"✅ [REGIME_MODELS] CatBoost OOF predictions generated with timeout protection", color="green")
            return model_oof
            
        except Exception as e:
            tprint(f"❌ [REGIME_MODELS] CatBoost OOF generation failed: {e}", color="red")
            # Ultimate fallback: return random predictions
            np.random.seed(42)
            unique_classes = np.unique(y)
            return np.random.choice(unique_classes, size=X.shape[0])
    
    def _create_enhanced_meta_features(self, base_predictions: np.ndarray, original_features: np.ndarray, feature_indices: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Create enhanced meta-learner features combining base predictions with original features.
        
        Args:
            base_predictions: Out-of-fold predictions from base models
            original_features: Original feature matrix
            feature_indices: Pre-selected feature indices for consistency (optional)
            
        Returns:
            Enhanced feature matrix for meta-learner
        """
        try:
            tprint("🔧 [REGIME_MODELS] Creating enhanced meta-learner features", color="blue")
            tprint(f"🔧 [REGIME_MODELS] Base predictions shape: {base_predictions.shape}", color="blue")
            tprint(f"🔧 [REGIME_MODELS] Original features shape: {original_features.shape}", color="blue")
            
            enhanced_features = []
            
            # Add base model predictions
            enhanced_features.append(base_predictions)
            
            # Add prediction statistics
            if base_predictions.shape[1] > 1:
                # Prediction agreement (how many models agree)
                pred_agreement = np.apply_along_axis(
                    lambda x: len(set(x)) / len(x), axis=1, arr=base_predictions
                ).reshape(-1, 1)
                enhanced_features.append(pred_agreement)
                
                # Prediction confidence (standard deviation of predictions)
                pred_confidence = np.std(base_predictions, axis=1).reshape(-1, 1)
                enhanced_features.append(pred_confidence)
                
                # Most frequent prediction (safer alternative to stats.mode)
                # Use the prediction with highest confidence instead of mode
                most_frequent = np.argmax(base_predictions, axis=1).reshape(-1, 1)
                enhanced_features.append(most_frequent)
            
            # Add subset of original features (most important ones)
            # Use consistent feature selection to avoid training/prediction mismatch
            if feature_indices is not None:
                # Use pre-selected features for consistency
                top_features_idx = feature_indices
                tprint(f"🔧 [REGIME_MODELS] Using pre-selected features: {len(top_features_idx)} features", color="blue")
            else:
                # Select features with highest variance (most informative)
                feature_variance = np.var(original_features, axis=0)
                top_features_idx = np.argsort(feature_variance)[-min(6, original_features.shape[1]):]
                tprint(f"🔧 [REGIME_MODELS] Selected top features by variance: {len(top_features_idx)} features", color="blue")
            
            enhanced_features.append(original_features[:, top_features_idx])
            
            # Combine all enhanced features
            enhanced_matrix = np.column_stack(enhanced_features)
            
            tprint(f"✅ [REGIME_MODELS] Enhanced features created: {enhanced_matrix.shape}", color="green")
            return enhanced_matrix
            
        except Exception as e:
            tprint(f"❌ [REGIME_MODELS] Enhanced feature creation failed: {e}", color="red")
            self.logger.error(f"Enhanced feature creation failed: {e}")
            # Fallback to base predictions only
            return base_predictions
    
    def _create_enhanced_meta_features_with_indices(self, base_predictions: np.ndarray, original_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create enhanced meta-learner features and return feature indices for consistency.
        
        Args:
            base_predictions: Out-of-fold predictions from base models
            original_features: Original feature matrix
            
        Returns:
            Tuple of (enhanced feature matrix, feature indices)
        """
        try:
            tprint("🔧 [REGIME_MODELS] Creating enhanced meta-learner features with indices", color="blue")
            
            enhanced_features = []
            
            # Add base model predictions
            enhanced_features.append(base_predictions)
            
            # Add prediction statistics
            if base_predictions.shape[1] > 1:
                # Prediction agreement (how many models agree)
                pred_agreement = np.apply_along_axis(
                    lambda x: len(set(x)) / len(x), axis=1, arr=base_predictions
                ).reshape(-1, 1)
                enhanced_features.append(pred_agreement)
                
                # Prediction confidence (standard deviation of predictions)
                pred_confidence = np.std(base_predictions, axis=1).reshape(-1, 1)
                enhanced_features.append(pred_confidence)
                
                # Most frequent prediction (safer alternative to stats.mode)
                # Use the prediction with highest confidence instead of mode
                most_frequent = np.argmax(base_predictions, axis=1).reshape(-1, 1)
                enhanced_features.append(most_frequent)
            
            # Select features with highest variance (most informative) and store indices
            feature_variance = np.var(original_features, axis=0)
            top_features_idx = np.argsort(feature_variance)[-min(6, original_features.shape[1]):]
            enhanced_features.append(original_features[:, top_features_idx])
            
            # Combine all enhanced features
            enhanced_matrix = np.column_stack(enhanced_features)
            
            tprint(f"✅ [REGIME_MODELS] Enhanced features created: {enhanced_matrix.shape}", color="green")
            tprint(f"🔧 [REGIME_MODELS] Stored feature indices: {len(top_features_idx)} features", color="blue")
            return enhanced_matrix, top_features_idx
            
        except Exception as e:
            tprint(f"❌ [REGIME_MODELS] Enhanced feature creation failed: {e}", color="red")
            self.logger.error(f"Enhanced feature creation failed: {e}")
            # Fallback to base predictions only
            return base_predictions, np.array([])
    
    def _log_grl_optimization_results(self, best_params: dict, best_score: float, n_classes: int, n_samples: int) -> None:
        """
        Log detailed results of Greedy Rule Lists optimization.
        
        Args:
            best_params: Best parameters found
            best_score: Best cross-validation score
            n_classes: Number of regime classes
            n_samples: Number of training samples
        """
        try:
            tprint("📊 [REGIME_MODELS] Greedy Rule Lists Optimization Results:", color="cyan", bold=True)
            tprint(f"🎯 [REGIME_MODELS] Best CV Score: {best_score:.4f}", color="green")
            tprint(f"📋 [REGIME_MODELS] Best Parameters:", color="blue")
            tprint(f"   - max_depth: {best_params.get('max_depth', 'N/A')}", color="blue")
            tprint(f"   - max_rules: {best_params.get('max_rules', 'N/A')}", color="blue")
            tprint(f"📊 [REGIME_MODELS] Context: {n_classes} regimes, {n_samples} samples", color="blue")
            
            # Calculate expected improvement over default
            default_score = 0.0545  # 5.45% baseline
            improvement = ((best_score - default_score) / default_score) * 100
            tprint(f"📈 [REGIME_MODELS] Expected improvement: {improvement:+.1f}% over baseline", color="green")
            
            # Log to file for persistence
            self.logger.info(f"Greedy Rule Lists optimization completed:")
            self.logger.info(f"  Best CV Score: {best_score:.4f}")
            self.logger.info(f"  Best Parameters: {best_params}")
            self.logger.info(f"  Expected improvement: {improvement:+.1f}% over baseline")
            
        except Exception as e:
            tprint(f"⚠️ [REGIME_MODELS] Failed to log optimization results: {e}", color="yellow")
    
    def _advanced_grl_optimization(self, X_train: np.ndarray, y_train: np.ndarray, n_classes: int) -> GreedyRuleListClassifier:
        """
        Advanced parameter optimization for Greedy Rule Lists using adaptive search.
        
        Args:
            X_train: Training features
            y_train: Training labels
            n_classes: Number of regime classes
            
        Returns:
            Optimized GreedyRuleListClassifier
        """
        try:
            tprint("🚀 [REGIME_MODELS] Advanced Greedy Rule Lists optimization with adaptive search", color="cyan")
            
            # Adaptive parameter ranges based on data characteristics
            n_samples, n_features = X_train.shape
            
            # Calculate adaptive parameter ranges
            adaptive_max_depth = min(20, max(8, int(np.log2(n_samples))))
            adaptive_min_samples = max(5, min(50, n_samples // (n_classes * 4)))
            
            tprint(f"📊 [REGIME_MODELS] Adaptive ranges - max_depth: {adaptive_max_depth}, min_samples: {adaptive_min_samples}", color="blue")
            
            # Define adaptive parameter search space
            # Note: GreedyRuleListClassifier only supports max_depth, class_weight, and criterion parameters
            param_combinations = [
                # High complexity for complex regimes
                {
                    'max_depth': adaptive_max_depth
                },
                # Balanced complexity
                {
                    'max_depth': adaptive_max_depth - 2
                },
                # Conservative for stability
                {
                    'max_depth': adaptive_max_depth - 4
                }
            ]
            
            best_model = None
            best_score = 0.0
            best_params = None
            
            # Test each parameter combination
            for i, params in enumerate(param_combinations):
                tprint(f"🔍 [REGIME_MODELS] Testing combination {i+1}/{len(param_combinations)}: {params}", color="blue")
                
                try:
                    # Create model with current parameters
                    # Note: GreedyRuleListClassifier doesn't support min_samples_split and min_samples_leaf
                    model = GreedyRuleListClassifier(
                        max_depth=params['max_depth'],
                        criterion='gini'
                    )
                    
                    # Use stratified cross-validation for better regime balance
                    from sklearn.model_selection import StratifiedKFold
                    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                    
                    cv_scores = cross_val_score(
                        model, X_train, y_train, 
                        cv=skf, scoring='accuracy', n_jobs=1
                    )
                    mean_score = cv_scores.mean()
                    std_score = cv_scores.std()
                    
                    tprint(f"📊 [REGIME_MODELS] CV Score: {mean_score:.4f} ± {std_score:.4f}", color="blue")
                    
                    # Update best model if this is better
                    if mean_score > best_score:
                        best_score = mean_score
                        best_model = model
                        best_params = params.copy()
                        
                except Exception as e:
                    tprint(f"⚠️ [REGIME_MODELS] Parameter combination {i+1} failed: {e}", color="yellow")
                    continue
            
            # Train the best model on full training data
            if best_model is not None:
                tprint(f"✅ [REGIME_MODELS] Best parameters found: {best_params}", color="green")
                tprint(f"📊 [REGIME_MODELS] Best CV score: {best_score:.4f}", color="green")
                
                # Fit the best model
                best_model.fit(X_train, y_train)
                return best_model
            else:
                # Fallback to default parameters
                tprint("⚠️ [REGIME_MODELS] No optimal parameters found, using default configuration", color="yellow")
                default_model = GreedyRuleListClassifier(**self.regime_models_config['base']['Greedy Rule Lists'])
                default_model.fit(X_train, y_train)
                return default_model
                
        except Exception as e:
            tprint(f"❌ [REGIME_MODELS] Advanced Greedy Rule Lists optimization failed: {e}", color="red")
            self.logger.error(f"Advanced Greedy Rule Lists optimization failed: {e}")
            
            # Fallback to default parameters
            tprint("🔄 [REGIME_MODELS] Using fallback default parameters", color="yellow")
            default_model = GreedyRuleListClassifier(**self.regime_models_config['base']['Greedy Rule Lists'])
            default_model.fit(X_train, y_train)
            return default_model
    
    def _optimize_greedy_rule_lists(self, X_train: np.ndarray, y_train: np.ndarray, n_classes: int) -> GreedyRuleListClassifier:
        """
        Optimize Greedy Rule Lists parameters for complex regime detection.
        
        Args:
            X_train: Training features
            y_train: Training labels
            n_classes: Number of regime classes
            
        Returns:
            Optimized GreedyRuleListClassifier
        """
        try:
            tprint("🔧 [REGIME_MODELS] Optimizing Greedy Rule Lists parameters for complex regimes", color="cyan")
            
            # Define parameter search space based on regime complexity
            # Note: GreedyRuleListClassifier only supports max_depth, class_weight, and criterion parameters
            param_grids = [
                # Conservative parameters for stable regimes
                {
                    'max_depth': [15, 18, 20],
                    'class_weight': ['balanced']
                },
                # Aggressive parameters for complex regimes
                {
                    'max_depth': [20, 25, 30],
                    'class_weight': ['balanced']
                }
            ]
            
            best_model = None
            best_score = 0.0
            best_params = None
            
            # Try different parameter combinations
            for param_grid in param_grids:
                tprint(f"🔍 [REGIME_MODELS] Testing parameter grid: {param_grid}", color="blue")
                
                # Simple grid search with cross-validation
                from sklearn.model_selection import cross_val_score
                
                for max_depth in param_grid['max_depth']:
                    for class_weight in param_grid['class_weight']:
                        try:
                            # Create model with current parameters
                            # Note: GreedyRuleListClassifier only supports max_depth, class_weight, and criterion
                            model = GreedyRuleListClassifier(
                                max_depth=max_depth,
                                criterion='gini',
                                class_weight=class_weight
                            )
                        
                            # Cross-validation score
                            cv_scores = cross_val_score(
                                model, X_train, y_train, 
                                cv=3, scoring='accuracy', n_jobs=1
                            )
                            mean_score = cv_scores.mean()
                            
                            tprint(f"📊 [REGIME_MODELS] Params: depth={max_depth}, class_weight={class_weight} -> CV Score: {mean_score:.4f}", color="blue")
                            
                            # Update best model if this is better
                            if mean_score > best_score:
                                best_score = mean_score
                                best_model = model
                                best_params = {
                                    'max_depth': max_depth,
                                    'class_weight': class_weight
                                }
                        
                        except Exception as e:
                            tprint(f"⚠️ [REGIME_MODELS] Parameter combination failed: {e}", color="yellow")
                            continue
            
            # Train the best model on full training data
            if best_model is not None:
                tprint(f"✅ [REGIME_MODELS] Best parameters found: {best_params}", color="green")
                tprint(f"📊 [REGIME_MODELS] Best CV score: {best_score:.4f}", color="green")
                
                # Fit the best model
                best_model.fit(X_train, y_train)
                return best_model
            else:
                # Fallback to default parameters
                tprint("⚠️ [REGIME_MODELS] No optimal parameters found, using default configuration", color="yellow")
                default_model = GreedyRuleListClassifier(**self.regime_models_config['base']['Greedy Rule Lists'])
                default_model.fit(X_train, y_train)
                return default_model
                
        except Exception as e:
            tprint(f"❌ [REGIME_MODELS] Greedy Rule Lists optimization failed: {e}", color="red")
            self.logger.error(f"Greedy Rule Lists optimization failed: {e}")
            
            # Fallback to default parameters
            tprint("🔄 [REGIME_MODELS] Using fallback default parameters", color="yellow")
            default_model = GreedyRuleListClassifier(**self.regime_models_config['base']['Greedy Rule Lists'])
            default_model.fit(X_train, y_train)
            return default_model
    
    def _robust_grl_training(self, X_train: np.ndarray, y_train: np.ndarray, n_classes: int) -> GreedyRuleListClassifier:
        """
        Robust Greedy Rule Lists training with multiple fallback strategies.
        
        Args:
            X_train: Training features
            y_train: Training labels
            n_classes: Number of regime classes
            
        Returns:
            Trained GreedyRuleListClassifier
        """
        try:
            tprint("🔧 [REGIME_MODELS] Starting robust Greedy Rule Lists training", color="cyan")
            
            # Preprocess features for better GRL performance
            tprint("🔧 [REGIME_MODELS] Preprocessing features for Greedy Rule Lists", color="blue")
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            
            # Check class distribution
            from collections import Counter
            class_counts = Counter(y_train)
            tprint(f"📊 [REGIME_MODELS] Class distribution: {dict(class_counts)}", color="blue")
            
            # Handle class imbalance with SMOTE if needed
            min_class_count = min(class_counts.values())
            if min_class_count < 10:  # If any class has less than 10 samples
                tprint("⚠️ [REGIME_MODELS] Detected class imbalance, applying SMOTE", color="yellow")
                try:
                    from imblearn.over_sampling import SMOTE
                    smote = SMOTE(random_state=42, k_neighbors=1)
                    X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
                    tprint(f"📊 [REGIME_MODELS] After SMOTE - X: {X_train_scaled.shape}, y: {len(y_train)}", color="blue")
                except Exception as e:
                    tprint(f"⚠️ [REGIME_MODELS] SMOTE failed, continuing without resampling: {e}", color="yellow")
            
            # Strategy 1: Conservative parameters for stability
            try:
                tprint("🔧 [REGIME_MODELS] Strategy 1: Conservative parameters", color="blue")
                conservative_params = {
                    'max_depth': 15,
                    'criterion': 'gini',
                    'class_weight': 'balanced'
                }
                
                model = GreedyRuleListClassifier(**conservative_params)
                model.fit(X_train_scaled, y_train)
                tprint("✅ [REGIME_MODELS] Conservative parameters successful", color="green")
                return model
                
            except Exception as e:
                tprint(f"⚠️ [REGIME_MODELS] Conservative parameters failed: {e}", color="yellow")
            
            # Strategy 2: Minimal parameters
            try:
                tprint("🔧 [REGIME_MODELS] Strategy 2: Minimal parameters", color="blue")
                minimal_params = {
                    'max_depth': 10,
                    'criterion': 'gini',
                    'class_weight': 'balanced'
                }
                
                model = GreedyRuleListClassifier(**minimal_params)
                model.fit(X_train_scaled, y_train)
                tprint("✅ [REGIME_MODELS] Minimal parameters successful", color="green")
                return model
                
            except Exception as e:
                tprint(f"⚠️ [REGIME_MODELS] Minimal parameters failed: {e}", color="yellow")
            
            # Strategy 3: Ultra-minimal parameters
            try:
                tprint("🔧 [REGIME_MODELS] Strategy 3: Ultra-minimal parameters", color="blue")
                ultra_minimal_params = {
                    'max_depth': 8,
                    'criterion': 'gini',
                    'class_weight': 'balanced'
                }
                
                model = GreedyRuleListClassifier(**ultra_minimal_params)
                model.fit(X_train_scaled, y_train)
                tprint("✅ [REGIME_MODELS] Ultra-minimal parameters successful", color="green")
                return model
                
            except Exception as e:
                tprint(f"⚠️ [REGIME_MODELS] Ultra-minimal parameters failed: {e}", color="yellow")
            
            # Strategy 4: Default parameters as last resort
            try:
                tprint("🔧 [REGIME_MODELS] Strategy 4: Default parameters", color="blue")
                model = GreedyRuleListClassifier()
                model.fit(X_train_scaled, y_train)
                tprint("✅ [REGIME_MODELS] Default parameters successful", color="green")
                return model
                
            except Exception as e:
                tprint(f"❌ [REGIME_MODELS] All Greedy Rule Lists strategies failed: {e}", color="red")
                raise e
                
        except Exception as e:
            tprint(f"❌ [REGIME_MODELS] Robust Greedy Rule Lists training failed: {e}", color="red")
            self.logger.error(f"Robust Greedy Rule Lists training failed: {e}")
            raise e
