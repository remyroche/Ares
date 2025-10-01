"""
NAS-TAS Models Training Component

This component implements base model training for NAS-TAS (Neural Architecture Search - Tree-based Architecture Search) based regime detection models.
It trains individual base models using NAS-TAS regime labels for regime classification.
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

# Suppress LightGBM warnings about no further splits
warnings.filterwarnings('ignore', message='.*No further splits with positive gain.*')

# Import ML libraries with comprehensive error handling and version logging
tprint("🔍 [NAS_TAS_MODELS] Starting ML libraries import process", color="cyan")
ML_LIBRARIES_AVAILABLE = False
ML_LIBRARY_VERSIONS = {}
ML_IMPORT_ERRORS = []

# Import sklearn components with detailed logging
try:
    from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler
    import sklearn
    ML_LIBRARY_VERSIONS['sklearn'] = sklearn.__version__
    tprint(f"✅ [NAS_TAS_MODELS] scikit-learn imported successfully (v{sklearn.__version__})", color="green")
except ImportError as e:
    ML_IMPORT_ERRORS.append(f"scikit-learn: {e}")
    tprint(f"❌ [NAS_TAS_MODELS] Failed to import scikit-learn: {e}", color="red")

# Import CatBoost with version logging
try:
    import catboost as cb
    ML_LIBRARY_VERSIONS['catboost'] = cb.__version__
    tprint(f"✅ [NAS_TAS_MODELS] CatBoost imported successfully (v{cb.__version__})", color="green")
except ImportError as e:
    ML_IMPORT_ERRORS.append(f"CatBoost: {e}")
    tprint(f"❌ [NAS_TAS_MODELS] Failed to import CatBoost: {e}", color="red")

# Import Bayesian Rule Lists with version logging
try:
    from imodels import BayesianRuleListClassifier
    # Get actual version if available
    try:
        import imodels
        ML_LIBRARY_VERSIONS['imodels'] = getattr(imodels, '__version__', "2.0.3")
    except:
        ML_LIBRARY_VERSIONS['imodels'] = "2.0.3"  # Known version from pip list
    tprint(f"✅ [NAS_TAS_MODELS] Bayesian Rule Lists imported successfully", color="green")
except ImportError as e:
    ML_IMPORT_ERRORS.append(f"Bayesian Rule Lists: {e}")
    tprint(f"❌ [NAS_TAS_MODELS] Failed to import Bayesian Rule Lists: {e}", color="red")

# Check overall availability
if not ML_IMPORT_ERRORS:
    ML_LIBRARIES_AVAILABLE = True
    tprint("🎉 [NAS_TAS_MODELS] All ML libraries imported successfully", color="green", bold=True)
    tprint(f"📊 [NAS_TAS_MODELS] Library versions: {ML_LIBRARY_VERSIONS}", color="blue")
else:
    tprint(f"⚠️ [NAS_TAS_MODELS] Import errors encountered: {ML_IMPORT_ERRORS}", color="yellow")
    tprint("🔧 [NAS_TAS_MODELS] Some functionality may be limited", color="yellow")


class NASTASModelsTrainingComponent(BaseMarketAnalysisComponent):
    """
    NAS-TAS Models Training Component.
    
    This component trains base models using NAS-TAS regime labels for regime classification.
    It creates individual models for regime detection and classification.
    """
    
    def __init__(self, config: Optional[ComponentConfig] = None):
        """Initialize the NAS-TAS Models Training Component."""
        tprint("🚀 [NAS_TAS_MODELS] Initializing NAS-TAS Models Training Component", color="cyan", bold=True)
        tprint(f"📋 [NAS_TAS_MODELS] Config provided: {config is not None}", color="blue")
        
        # Initialize parent component with detailed logging
        try:
            super().__init__(config)
            tprint("✅ [NAS_TAS_MODELS] Parent component initialized successfully", color="green")
        except Exception as e:
            tprint(f"❌ [NAS_TAS_MODELS] Failed to initialize parent component: {e}", color="red")
            raise
        
        # Initialize logger with detailed configuration
        try:
            self.logger = system_logger.getChild('NASTASModelsTrainingComponent')
            self.logger.info("NAS-TAS Models Training Component logger initialized")
            tprint("✅ [NAS_TAS_MODELS] Logger initialized successfully", color="green")
        except Exception as e:
            tprint(f"⚠️ [NAS_TAS_MODELS] Logger initialization warning: {e}", color="yellow")
        
        # Initialize model training parameters with validation
        tprint("🔧 [NAS_TAS_MODELS] Configuring model training parameters", color="cyan")
        self.model_config = {
            'random_state': 42,
            'test_size': 0.2,
            'cv_folds': 5,
            'n_jobs': -1
        }
        
        # Validate configuration parameters
        self._validate_model_config()
        tprint(f"📊 [NAS_TAS_MODELS] Model configuration: {self.model_config}", color="blue")
        
        # Initialize model storage with detailed logging
        self.models = {}
        self.model_metrics = {}
        self.training_history = []
        self.performance_metrics = {}
        
        tprint("📊 [NAS_TAS_MODELS] Model storage initialized", color="blue")
        tprint(f"🔍 [NAS_TAS_MODELS] Available ML libraries: {ML_LIBRARIES_AVAILABLE}", color="blue")
        if ML_LIBRARIES_AVAILABLE:
            tprint(f"📚 [NAS_TAS_MODELS] Library versions: {ML_LIBRARY_VERSIONS}", color="blue")
        
        # Log initialization completion
        tprint("✅ [NAS_TAS_MODELS] NAS-TAS Models Training Component initialized successfully", color="green", bold=True)
        self.logger.info("NAS-TAS Models Training Component initialization completed")
    
    def _validate_model_config(self):
        """Validate model configuration parameters."""
        tprint("🔍 [NAS_TAS_MODELS] Validating model configuration", color="cyan")
        
        # Validate random_state
        if not isinstance(self.model_config['random_state'], int) or self.model_config['random_state'] < 0:
            tprint(f"⚠️ [NAS_TAS_MODELS] Invalid random_state: {self.model_config['random_state']}, using default 42", color="yellow")
            self.model_config['random_state'] = 42
        
        # Validate test_size
        if not (0 < self.model_config['test_size'] < 1):
            tprint(f"⚠️ [NAS_TAS_MODELS] Invalid test_size: {self.model_config['test_size']}, using default 0.2", color="yellow")
            self.model_config['test_size'] = 0.2
        
        # Validate cv_folds
        if not isinstance(self.model_config['cv_folds'], int) or self.model_config['cv_folds'] < 2:
            tprint(f"⚠️ [NAS_TAS_MODELS] Invalid cv_folds: {self.model_config['cv_folds']}, using default 5", color="yellow")
            self.model_config['cv_folds'] = 5
        
        # Validate n_jobs
        if self.model_config['n_jobs'] != -1 and (not isinstance(self.model_config['n_jobs'], int) or self.model_config['n_jobs'] < 1):
            tprint(f"⚠️ [NAS_TAS_MODELS] Invalid n_jobs: {self.model_config['n_jobs']}, using default -1", color="yellow")
            self.model_config['n_jobs'] = -1
        
        tprint("✅ [NAS_TAS_MODELS] Model configuration validation completed", color="green")
    
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
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_used_gb = disk.used / (1024**3)
            disk_total_gb = disk.total / (1024**3)
            disk_percent = (disk.used / disk.total) * 100
            
            return {
                'cpu_percent': cpu_percent,
                'cpu_count': cpu_count,
                'memory_used_gb': memory_used_gb,
                'memory_total_gb': memory_total_gb,
                'memory_percent': memory_percent,
                'disk_used_gb': disk_used_gb,
                'disk_total_gb': disk_total_gb,
                'disk_percent': disk_percent
            }
        except Exception as e:
            tprint(f"⚠️ [NAS_TAS_MODELS] Failed to get system performance: {e}", color="yellow")
            return {}
    
    def _log_performance_metrics(self, stage: str, start_time: float):
        """Log performance metrics for a given stage."""
        current_time = time.time()
        elapsed_time = current_time - start_time
        
        # Get system performance
        perf_metrics = self._get_system_performance()
        
        # Log timing
        tprint(f"⏱️ [NAS_TAS_MODELS] {stage} completed in {elapsed_time:.3f} seconds", color="blue")
        
        # Log system metrics if available
        if perf_metrics:
            tprint(f"💻 [NAS_TAS_MODELS] System metrics - CPU: {perf_metrics.get('cpu_percent', 'N/A')}%, Memory: {perf_metrics.get('memory_percent', 'N/A')}% ({perf_metrics.get('memory_used_gb', 0):.1f}GB/{perf_metrics.get('memory_total_gb', 0):.1f}GB)", color="blue")
        
        return elapsed_time, perf_metrics
    
    def _monitor_memory_usage(self, stage: str):
        """Monitor and log memory usage."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024**2)
            tprint(f"🧠 [NAS_TAS_MODELS] {stage} memory usage: {memory_mb:.1f} MB", color="blue")
            return memory_mb
        except Exception as e:
            tprint(f"⚠️ [NAS_TAS_MODELS] Failed to monitor memory: {e}", color="yellow")
            return 0
    
    def _cleanup_memory(self):
        """Clean up memory by forcing garbage collection."""
        try:
            gc.collect()
            tprint("🧹 [NAS_TAS_MODELS] Memory cleanup completed", color="blue")
        except Exception as e:
            tprint(f"⚠️ [NAS_TAS_MODELS] Memory cleanup failed: {e}", color="yellow")
    
    def _validate_input_data(self, data: pd.DataFrame) -> bool:
        """Validate input data for training."""
        tprint("🔍 [NAS_TAS_MODELS] Validating input data", color="cyan")
        
        try:
            # Check if data is empty
            if data.empty:
                tprint("❌ [NAS_TAS_MODELS] Input data is empty", color="red")
                return False
            
            # Check minimum required columns
            required_columns = ['close']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                tprint(f"❌ [NAS_TAS_MODELS] Missing required columns: {missing_columns}", color="red")
                return False
            
            # Check data types
            if not pd.api.types.is_numeric_dtype(data['close']):
                tprint("❌ [NAS_TAS_MODELS] 'close' column is not numeric", color="red")
                return False
            
            # Check for sufficient data points
            min_samples = 100
            if len(data) < min_samples:
                tprint(f"❌ [NAS_TAS_MODELS] Insufficient data points: {len(data)} < {min_samples}", color="red")
                return False
            
            # Check for excessive missing values
            missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
            if missing_ratio > 0.5:
                tprint(f"⚠️ [NAS_TAS_MODELS] High missing value ratio: {missing_ratio:.2%}", color="yellow")
            
            tprint("✅ [NAS_TAS_MODELS] Input data validation passed", color="green")
            return True
            
        except Exception as e:
            tprint(f"❌ [NAS_TAS_MODELS] Data validation error: {e}", color="red")
            return False
    
    def _validate_regime_labels(self, regime_labels: np.ndarray) -> bool:
        """Validate regime labels."""
        tprint("🔍 [NAS_TAS_MODELS] Validating regime labels", color="cyan")
        
        try:
            # Check if labels are not None
            if regime_labels is None:
                tprint("❌ [NAS_TAS_MODELS] Regime labels are None", color="red")
                return False
            
            # Convert to numpy array if needed
            regime_labels = np.array(regime_labels)
            
            # Check for sufficient samples
            if len(regime_labels) < 50:
                tprint(f"❌ [NAS_TAS_MODELS] Insufficient regime labels: {len(regime_labels)} < 50", color="red")
                return False
            
            # Check for valid regime values
            unique_regimes = np.unique(regime_labels)
            if len(unique_regimes) < 2:
                tprint(f"❌ [NAS_TAS_MODELS] Insufficient regime classes: {len(unique_regimes)} < 2", color="red")
                return False
            
            # Check for class balance
            class_counts = np.bincount(regime_labels)
            min_class_count = class_counts.min()
            max_class_count = class_counts.max()
            imbalance_ratio = max_class_count / min_class_count if min_class_count > 0 else float('inf')
            
            if imbalance_ratio > 10:
                tprint(f"⚠️ [NAS_TAS_MODELS] High class imbalance ratio: {imbalance_ratio:.1f}", color="yellow")
            
            tprint(f"✅ [NAS_TAS_MODELS] Regime labels validation passed - {len(unique_regimes)} classes, {len(regime_labels)} samples", color="green")
            return True
            
        except Exception as e:
            tprint(f"❌ [NAS_TAS_MODELS] Regime labels validation error: {e}", color="red")
            return False
    
    def _validate_training_data(self, X: np.ndarray, y: np.ndarray) -> bool:
        """Validate prepared training data."""
        tprint("🔍 [NAS_TAS_MODELS] Validating training data", color="cyan")
        
        try:
            # Check shapes
            if X.shape[0] != y.shape[0]:
                tprint(f"❌ [NAS_TAS_MODELS] Mismatched sample counts: X={X.shape[0]}, y={y.shape[0]}", color="red")
                return False
            
            # Check for sufficient features
            if X.shape[1] < 2:
                tprint(f"❌ [NAS_TAS_MODELS] Insufficient features: {X.shape[1]} < 2", color="red")
                return False
            
            # Check for NaN or infinite values
            nan_count = np.isnan(X).sum()
            inf_count = np.isinf(X).sum()
            if nan_count > 0:
                tprint(f"❌ [NAS_TAS_MODELS] Found {nan_count} NaN values in features", color="red")
                return False
            if inf_count > 0:
                tprint(f"❌ [NAS_TAS_MODELS] Found {inf_count} infinite values in features", color="red")
                return False
            
            # Check feature variance
            feature_vars = np.var(X, axis=0)
            zero_var_features = np.sum(feature_vars == 0)
            if zero_var_features > 0:
                tprint(f"⚠️ [NAS_TAS_MODELS] Found {zero_var_features} features with zero variance", color="yellow")
            
            tprint("✅ [NAS_TAS_MODELS] Training data validation passed", color="green")
            return True
            
        except Exception as e:
            tprint(f"❌ [NAS_TAS_MODELS] Training data validation error: {e}", color="red")
            return False
    
    def _validate_models(self, models: Dict[str, Any]) -> bool:
        """Validate trained models."""
        tprint("🔍 [NAS_TAS_MODELS] Validating trained models", color="cyan")
        
        try:
            if not models:
                tprint("❌ [NAS_TAS_MODELS] No models trained", color="red")
                return False
            
            # Check each model
            for name, model in models.items():
                if model is None:
                    tprint(f"❌ [NAS_TAS_MODELS] Model {name} is None", color="red")
                    return False
                
                # Check if model has required methods
                if not hasattr(model, 'predict'):
                    tprint(f"❌ [NAS_TAS_MODELS] Model {name} missing predict method", color="red")
                    return False
            
            tprint(f"✅ [NAS_TAS_MODELS] Model validation passed - {len(models)} models validated", color="green")
            return True
            
        except Exception as e:
            tprint(f"❌ [NAS_TAS_MODELS] Model validation error: {e}", color="red")
            return False
    
    def get_required_artifacts(self) -> List[str]:
        """Get list of required artifacts this component must produce."""
        tprint("📋 [NAS_TAS_MODELS] Getting required artifacts", color="cyan")
        required_artifacts = ['nas_tas_models_training_result']
        tprint(f"✅ [NAS_TAS_MODELS] Required artifacts: {required_artifacts}", color="green")
        return required_artifacts
    
    async def execute(self, data: pd.DataFrame, pipeline_state: Dict[str, Any]) -> ComponentResult:
        """
        Execute NAS-TAS models training.
        
        Args:
            data: Market data DataFrame
            pipeline_state: Pipeline state dictionary
            
        Returns:
            ComponentResult with training results
        """
        execution_start_time = time.time()
        tprint("🚀 [NAS_TAS_MODELS] Starting NAS-TAS models training execution", color="cyan", bold=True)
        self.logger.info("Starting NAS-TAS models training execution")
        
        # Log initial system performance
        initial_perf = self._get_system_performance()
        if initial_perf:
            tprint(f"💻 [NAS_TAS_MODELS] Initial system state - CPU: {initial_perf.get('cpu_percent', 'N/A')}%, Memory: {initial_perf.get('memory_percent', 'N/A')}%", color="blue")
        
        # Monitor initial memory usage
        initial_memory = self._monitor_memory_usage("Initial")
        
        # Log execution context
        tprint(f"📊 [NAS_TAS_MODELS] Input data shape: {data.shape}", color="blue")
        tprint(f"📋 [NAS_TAS_MODELS] Data columns: {list(data.columns)}", color="blue")
        tprint(f"🔍 [NAS_TAS_MODELS] Pipeline state keys: {list(pipeline_state.keys())}", color="blue")
        
        try:
            # Step 0: Validate input data
            tprint("🔍 [NAS_TAS_MODELS] Step 0: Validating input data", color="cyan")
            if not self._validate_input_data(data):
                error_msg = "Input data validation failed"
                tprint(f"❌ [NAS_TAS_MODELS] {error_msg}", color="red")
                self.logger.error("Input data validation failed")
                return ComponentResult(
                    success=False,
                    artifacts={},
                    error_message=error_msg
                )
            
            # Step 1: Check ML libraries availability
            tprint("🔍 [NAS_TAS_MODELS] Step 1: Checking ML libraries availability", color="cyan")
            if not ML_LIBRARIES_AVAILABLE:
                error_msg = "ML libraries not available for NAS-TAS models training"
                tprint(f"❌ [NAS_TAS_MODELS] {error_msg}", color="red")
                tprint(f"🔍 [NAS_TAS_MODELS] Import errors: {ML_IMPORT_ERRORS}", color="yellow")
                self.logger.error(f"ML libraries not available: {ML_IMPORT_ERRORS}")
                return ComponentResult(
                    success=False,
                    artifacts={},
                    error_message=error_msg
                )
            tprint("✅ [NAS_TAS_MODELS] ML libraries check passed", color="green")
            
            # Step 2: Extract and validate regime labels
            tprint("🔍 [NAS_TAS_MODELS] Step 2: Extracting regime labels from pipeline state", color="cyan")
            artifacts = pipeline_state.get('artifacts', {})
            tprint(f"📋 [NAS_TAS_MODELS] Available artifacts: {list(artifacts.keys())}", color="blue")
            
            # Try to get regime labels from regime discovery result first
            nas_tas_regime_discovery_result = artifacts.get('nas_tas_regime_discovery_result', {})
            tprint(f"🔍 [NAS_TAS_MODELS] NAS-TAS regime discovery result keys: {list(nas_tas_regime_discovery_result.keys())}", color="blue")
            
            # Extract regime labels from both TAS and NAS systems
            tas_assignments = nas_tas_regime_discovery_result.get('tas_assignments', [])
            nas_assignments = nas_tas_regime_discovery_result.get('nas_assignments', [])
            
            tprint(f"📊 [NAS_TAS_MODELS] TAS assignments: {len(tas_assignments)} samples", color="blue")
            tprint(f"📊 [NAS_TAS_MODELS] NAS assignments: {len(nas_assignments)} samples", color="blue")
            
            # Use TAS assignments as primary regime labels (fallback to NAS if TAS is empty)
            if len(tas_assignments) > 0:
                regime_labels = tas_assignments
                tprint("✅ [NAS_TAS_MODELS] Using TAS assignments as regime labels", color="green")
            elif len(nas_assignments) > 0:
                regime_labels = nas_assignments
                tprint("✅ [NAS_TAS_MODELS] Using NAS assignments as regime labels (TAS fallback)", color="green")
            else:
                # Fallback to clustering result if available
                nas_tas_clustering_result = artifacts.get('nas_tas_clustering_result', {})
                tprint(f"🔍 [NAS_TAS_MODELS] NAS-TAS clustering result keys: {list(nas_tas_clustering_result.keys())}", color="blue")
                regime_labels = nas_tas_clustering_result.get('cluster_assignments')

                if regime_labels is None:
                    error_msg = "No regime labels found in pipeline state artifacts"
                    tprint(f"❌ [NAS_TAS_MODELS] {error_msg}", color="red")
                    tprint(f"🔍 [NAS_TAS_MODELS] Available artifacts: {list(artifacts.keys())}", color="yellow")
                    self.logger.error(f"Missing regime labels. Available artifacts: {list(artifacts.keys())}")
                    return ComponentResult(
                        success=False,
                        artifacts={},
                        error_message=error_msg
                    )
                else:
                    tprint("✅ [NAS_TAS_MODELS] Using clustering result regime labels as fallback", color="green")
            
            # Validate regime labels
            regime_labels = np.array(regime_labels)
            if not self._validate_regime_labels(regime_labels):
                error_msg = "Regime labels validation failed"
                tprint(f"❌ [NAS_TAS_MODELS] {error_msg}", color="red")
                self.logger.error("Regime labels validation failed")
                return ComponentResult(
                    success=False,
                    artifacts={},
                    error_message=error_msg
                )
            
            unique_regimes = np.unique(regime_labels)
            tprint(f"📊 [NAS_TAS_MODELS] Found regime labels: {len(regime_labels)} samples", color="blue")
            tprint(f"📊 [NAS_TAS_MODELS] Unique regimes: {unique_regimes} (count: {len(unique_regimes)})", color="blue")
            tprint(f"📊 [NAS_TAS_MODELS] Regime distribution: {dict(zip(*np.unique(regime_labels, return_counts=True)))}", color="blue")
            
            # Log additional context about the regime labels source
            if len(tas_assignments) > 0 and len(nas_assignments) > 0:
                tprint(f"📊 [NAS_TAS_MODELS] Both TAS ({len(tas_assignments)}) and NAS ({len(nas_assignments)}) assignments available", color="blue")
                tprint(f"📊 [NAS_TAS_MODELS] TAS regimes: {len(set(tas_assignments))}, NAS regimes: {len(set(nas_assignments))}", color="blue")
            elif len(tas_assignments) > 0:
                tprint(f"📊 [NAS_TAS_MODELS] Using TAS assignments only ({len(tas_assignments)} samples)", color="blue")
            elif len(nas_assignments) > 0:
                tprint(f"📊 [NAS_TAS_MODELS] Using NAS assignments only ({len(nas_assignments)} samples)", color="blue")
            
            # Step 3: Prepare training data
            tprint("🔍 [NAS_TAS_MODELS] Step 3: Preparing training data", color="cyan")
            data_prep_start = time.time()
            X, y = self._prepare_training_data(data, regime_labels)
            self._log_performance_metrics("Data preparation", data_prep_start)
            self._monitor_memory_usage("After data preparation")
            if X is None or y is None:
                error_msg = "Failed to prepare training data"
                tprint(f"❌ [NAS_TAS_MODELS] {error_msg}", color="red")
                self.logger.error("Failed to prepare training data")
                return ComponentResult(
                    success=False,
                    artifacts={},
                    error_message=error_msg
                )
            
            # Validate prepared training data
            if not self._validate_training_data(X, y):
                error_msg = "Training data validation failed"
                tprint(f"❌ [NAS_TAS_MODELS] {error_msg}", color="red")
                self.logger.error("Training data validation failed")
                return ComponentResult(
                    success=False,
                    artifacts={},
                    error_message=error_msg
                )
            
            tprint(f"📊 [NAS_TAS_MODELS] Training data prepared: X={X.shape}, y={y.shape}", color="blue")
            tprint(f"📊 [NAS_TAS_MODELS] Feature matrix info: dtype={X.dtype}, min={X.min():.4f}, max={X.max():.4f}", color="blue")
            tprint(f"📊 [NAS_TAS_MODELS] Target distribution: {dict(zip(*np.unique(y, return_counts=True)))}", color="blue")
            
            # Step 4: Train models
            tprint("🔍 [NAS_TAS_MODELS] Step 4: Training models", color="cyan")
            model_training_start = time.time()
            training_results = self._train_models(X, y)
            self._log_performance_metrics("Model training", model_training_start)
            self._monitor_memory_usage("After model training")
            
            # Clean up memory after training
            self._cleanup_memory()
            
            # Validate trained models
            if not self._validate_models(training_results['models']):
                error_msg = "Trained models validation failed"
                tprint(f"❌ [NAS_TAS_MODELS] {error_msg}", color="red")
                self.logger.error("Trained models validation failed")
                return ComponentResult(
                    success=False,
                    artifacts={},
                    error_message=error_msg
                )
            
            # Step 5: Create and validate artifacts
            tprint("🔍 [NAS_TAS_MODELS] Step 5: Creating artifacts", color="cyan")
            artifacts = {
                'nas_tas_models_training_result': {
                    'models': training_results['models'],
                    'metrics': training_results['metrics'],
                    'training_time': training_results['training_time'],
                    'success': True,
                    'model_count': len(training_results['models']),
                    'feature_count': X.shape[1],
                    'sample_count': X.shape[0]
                }
            }
            
            execution_time = time.time() - execution_start_time
            
            # Log final performance metrics
            final_perf = self._get_system_performance()
            final_memory = self._monitor_memory_usage("Final")
            
            tprint(f"⏱️ [NAS_TAS_MODELS] Total execution time: {execution_time:.2f} seconds", color="blue")
            if final_perf:
                tprint(f"💻 [NAS_TAS_MODELS] Final system state - CPU: {final_perf.get('cpu_percent', 'N/A')}%, Memory: {final_perf.get('memory_percent', 'N/A')}%", color="blue")
            tprint(f"🧠 [NAS_TAS_MODELS] Memory usage change: {final_memory - initial_memory:.1f} MB", color="blue")
            
            tprint("✅ [NAS_TAS_MODELS] NAS-TAS models training completed successfully", color="green", bold=True)
            self.logger.info(f"NAS-TAS models training completed successfully in {execution_time:.2f} seconds")
            
            return ComponentResult(
                success=True,
                artifacts=artifacts,
                metadata={
                    'component_type': 'nas_tas_models_training', 
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
            error_msg = f"NAS-TAS models training failed: {str(e)}"
            
            # Enhanced error logging with context
            tprint(f"❌ [NAS_TAS_MODELS] {error_msg}", color="red")
            tprint(f"🔍 [NAS_TAS_MODELS] Error type: {error_type}", color="yellow")
            tprint(f"🔍 [NAS_TAS_MODELS] Execution time before failure: {execution_time:.2f} seconds", color="yellow")
            
            # Log system state at failure
            failure_perf = self._get_system_performance()
            if failure_perf:
                tprint(f"💻 [NAS_TAS_MODELS] System state at failure - CPU: {failure_perf.get('cpu_percent', 'N/A')}%, Memory: {failure_perf.get('memory_percent', 'N/A')}%", color="yellow")
            
            # Provide recovery suggestions based on error type
            recovery_suggestions = self._get_recovery_suggestions(e)
            if recovery_suggestions:
                tprint(f"💡 [NAS_TAS_MODELS] Recovery suggestions: {recovery_suggestions}", color="cyan")
            
            # Log detailed error information
            self.logger.error(f"NAS-TAS models training failed after {execution_time:.2f} seconds", exc_info=True)
            self.logger.error(f"Error type: {error_type}, Error message: {str(e)}")
            if recovery_suggestions:
                self.logger.error(f"Recovery suggestions: {recovery_suggestions}")
            
            return ComponentResult(
                success=False,
                artifacts={},
                error_message=f"{error_msg} (Type: {error_type})"
            )
    
    def _get_recovery_suggestions(self, error: Exception) -> str:
        """Get recovery suggestions based on error type."""
        error_type = type(error).__name__
        
        if "MemoryError" in error_type or "memory" in str(error).lower():
            return "Try reducing data size, increasing available memory, or using data sampling"
        elif "ImportError" in error_type:
            return "Check ML library installations: pip install scikit-learn xgboost lightgbm"
        elif "ValueError" in error_type and "shape" in str(error).lower():
            return "Check data alignment between features and labels, ensure consistent lengths"
        elif "KeyError" in error_type:
            return "Verify required columns exist in input data (close, volume, etc.)"
        elif "AttributeError" in error_type:
            return "Check model object integrity and required methods availability"
        elif "TimeoutError" in error_type:
            return "Increase timeout limits or reduce model complexity"
        else:
            return "Check logs for detailed error information and system requirements"
    
    def _prepare_training_data(self, data: pd.DataFrame, regime_labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from market data and regime labels."""
        tprint("🔧 [NAS_TAS_MODELS] Preparing training data", color="cyan", bold=True)
        self.logger.info("Starting data preparation process")
        
        try:
            # Log input data characteristics
            tprint(f"📊 [NAS_TAS_MODELS] Input data shape: {data.shape}", color="blue")
            tprint(f"📊 [NAS_TAS_MODELS] Input data columns: {list(data.columns)}", color="blue")
            tprint(f"📊 [NAS_TAS_MODELS] Input data dtypes: {data.dtypes.to_dict()}", color="blue")
            tprint(f"📊 [NAS_TAS_MODELS] Input data memory usage: {data.memory_usage(deep=True).sum() / 1024**2:.2f} MB", color="blue")
            tprint(f"📊 [NAS_TAS_MODELS] Regime labels shape: {regime_labels.shape}", color="blue")
            tprint(f"📊 [NAS_TAS_MODELS] Regime labels unique values: {np.unique(regime_labels)}", color="blue")
            
            # Check for missing values
            missing_values = data.isnull().sum()
            if missing_values.any():
                tprint(f"⚠️ [NAS_TAS_MODELS] Missing values detected: {missing_values[missing_values > 0].to_dict()}", color="yellow")
            else:
                tprint("✅ [NAS_TAS_MODELS] No missing values in input data", color="green")
            
            # Create basic features from OHLCV data
            features = []
            feature_names = []
            
            tprint("🔧 [NAS_TAS_MODELS] Creating price-based features", color="cyan", bold=True)
            if 'close' in data.columns:
                # Price-based features
                tprint("📈 [NAS_TAS_MODELS] Computing price returns", color="blue")
                returns = data['close'].pct_change().fillna(0)
                features.append(returns.values)
                feature_names.append('price_returns')
                tprint(f"📊 [NAS_TAS_MODELS] Returns stats: mean={returns.mean():.6f}, std={returns.std():.6f}, min={returns.min():.6f}, max={returns.max():.6f}", color="blue")
                tprint(f"📊 [NAS_TAS_MODELS] Returns distribution: {returns.describe().to_dict()}", color="blue")
                
                # Moving averages
                tprint("📈 [NAS_TAS_MODELS] Computing moving averages", color="blue")
                sma_20 = data['close'].rolling(20).mean().fillna(data['close'].iloc[0])
                sma_50 = data['close'].rolling(50).mean().fillna(data['close'].iloc[0])
                features.append(sma_20.values)
                features.append(sma_50.values)
                feature_names.extend(['sma_20', 'sma_50'])
                tprint(f"📊 [NAS_TAS_MODELS] SMA20 stats: mean={sma_20.mean():.2f}, std={sma_20.std():.2f}", color="blue")
                tprint(f"📊 [NAS_TAS_MODELS] SMA50 stats: mean={sma_50.mean():.2f}, std={sma_50.std():.2f}", color="blue")
                
                # Volatility
                tprint("📈 [NAS_TAS_MODELS] Computing volatility", color="blue")
                volatility = returns.rolling(20).std().fillna(0)
                features.append(volatility.values)
                feature_names.append('volatility_20')
                tprint(f"📊 [NAS_TAS_MODELS] Volatility stats: mean={volatility.mean():.6f}, std={volatility.std():.6f}", color="blue")
                
                # Additional technical indicators
                tprint("📈 [NAS_TAS_MODELS] Computing additional technical indicators", color="blue")
                
                # RSI-like indicator
                price_change = data['close'].diff()
                gain = price_change.where(price_change > 0, 0)
                loss = -price_change.where(price_change < 0, 0)
                avg_gain = gain.rolling(14).mean().fillna(0)
                avg_loss = loss.rolling(14).mean().fillna(0)
                rs = avg_gain / (avg_loss + 1e-8)  # Avoid division by zero
                rsi = 100 - (100 / (1 + rs))
                features.append(rsi.values)
                feature_names.append('rsi_14')
                tprint(f"📊 [NAS_TAS_MODELS] RSI stats: mean={rsi.mean():.2f}, std={rsi.std():.2f}", color="blue")
                
            else:
                tprint("⚠️ [NAS_TAS_MODELS] No 'close' column found, skipping price-based features", color="yellow")
            
            tprint("🔧 [NAS_TAS_MODELS] Creating volume-based features", color="cyan")
            if 'volume' in data.columns:
                # Volume features
                tprint("📊 [NAS_TAS_MODELS] Computing volume features", color="blue")
                volume_ratio = data['volume'] / data['volume'].rolling(20).mean().fillna(data['volume'].mean())
                features.append(volume_ratio.fillna(1).values)
                feature_names.append('volume_ratio')
                tprint(f"📊 [NAS_TAS_MODELS] Volume ratio stats: mean={volume_ratio.mean():.2f}, std={volume_ratio.std():.2f}", color="blue")
                
                # Volume volatility
                volume_returns = data['volume'].pct_change().fillna(0)
                volume_volatility = volume_returns.rolling(20).std().fillna(0)
                features.append(volume_volatility.values)
                feature_names.append('volume_volatility')
                tprint(f"📊 [NAS_TAS_MODELS] Volume volatility stats: mean={volume_volatility.mean():.6f}, std={volume_volatility.std():.6f}", color="blue")
            else:
                tprint("⚠️ [NAS_TAS_MODELS] No 'volume' column found, skipping volume-based features", color="yellow")
            
            # Log feature creation summary
            tprint(f"📊 [NAS_TAS_MODELS] Created {len(features)} feature sets: {feature_names}", color="blue")
            
            # Combine features
            tprint("🔧 [NAS_TAS_MODELS] Combining features into feature matrix", color="cyan")
            X = np.column_stack(features)
            tprint(f"📊 [NAS_TAS_MODELS] Feature matrix shape: {X.shape}", color="blue")
            tprint(f"📊 [NAS_TAS_MODELS] Feature matrix dtype: {X.dtype}", color="blue")
            tprint(f"📊 [NAS_TAS_MODELS] Feature matrix memory usage: {X.nbytes / 1024**2:.2f} MB", color="blue")
            
            # Check for NaN or infinite values in features
            nan_count = np.isnan(X).sum()
            inf_count = np.isinf(X).sum()
            if nan_count > 0:
                tprint(f"⚠️ [NAS_TAS_MODELS] Found {nan_count} NaN values in features", color="yellow")
                X = np.nan_to_num(X, nan=0.0)
                tprint("🔧 [NAS_TAS_MODELS] Replaced NaN values with 0", color="blue")
            if inf_count > 0:
                tprint(f"⚠️ [NAS_TAS_MODELS] Found {inf_count} infinite values in features", color="yellow")
                X = np.nan_to_num(X, posinf=1e6, neginf=-1e6)
                tprint("🔧 [NAS_TAS_MODELS] Replaced infinite values with finite bounds", color="blue")
            
            # Align with regime labels
            tprint("🔧 [NAS_TAS_MODELS] Aligning features with regime labels", color="cyan")
            min_length = min(len(X), len(regime_labels))
            tprint(f"📊 [NAS_TAS_MODELS] Aligning to minimum length: {min_length}", color="blue")
            
            X = X[:min_length]
            y = np.array(regime_labels[:min_length])
            
            # Final validation
            tprint(f"✅ [NAS_TAS_MODELS] Training data prepared: {X.shape[0]} samples, {X.shape[1]} features", color="green")
            tprint(f"📊 [NAS_TAS_MODELS] Feature matrix final stats: min={X.min():.4f}, max={X.max():.4f}, mean={X.mean():.4f}, std={X.std():.4f}", color="blue")
            tprint(f"📊 [NAS_TAS_MODELS] Target distribution: {dict(zip(*np.unique(y, return_counts=True)))}", color="blue")
            
            self.logger.info(f"Training data preparation completed: {X.shape[0]} samples, {X.shape[1]} features")
            return X, y
            
        except Exception as e:
            error_type = type(e).__name__
            tprint(f"❌ [NAS_TAS_MODELS] Error preparing training data: {e}", color="red")
            tprint(f"🔍 [NAS_TAS_MODELS] Error type: {error_type}", color="yellow")
            
            # Provide specific error context
            if "KeyError" in error_type:
                tprint(f"💡 [NAS_TAS_MODELS] Missing required columns. Available: {list(data.columns)}", color="cyan")
            elif "ValueError" in error_type and "shape" in str(e).lower():
                tprint(f"💡 [NAS_TAS_MODELS] Data shape mismatch. Data shape: {data.shape}, Labels length: {len(regime_labels)}", color="cyan")
            elif "MemoryError" in error_type:
                tprint(f"💡 [NAS_TAS_MODELS] Insufficient memory for data processing", color="cyan")
            
            self.logger.error(f"Error preparing training data: {str(e)}", exc_info=True)
            self.logger.error(f"Error type: {error_type}")
            return None, None
    
    def _train_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train individual models for regime classification."""
        tprint("🏋️ [NAS_TAS_MODELS] Training models", color="cyan")
        self.logger.info("Starting model training process")
        
        start_time = time.time()
        models = {}
        metrics = {}
        training_history = []
        
        try:
            # Log training data characteristics
            tprint(f"📊 [NAS_TAS_MODELS] Training data: {X.shape[0]} samples, {X.shape[1]} features", color="blue")
            tprint(f"📊 [NAS_TAS_MODELS] Target classes: {np.unique(y)} (count: {len(np.unique(y))})", color="blue")
            tprint(f"📊 [NAS_TAS_MODELS] Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}", color="blue")
            
            # Step 1: Split data with detailed logging
            tprint("🔧 [NAS_TAS_MODELS] Step 1: Splitting data into train/test sets", color="cyan")
            split_start = time.time()
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.model_config['test_size'], 
                random_state=self.model_config['random_state'], 
                stratify=y
            )
            
            split_time = time.time() - split_start
            tprint(f"📊 [NAS_TAS_MODELS] Train set: {X_train.shape[0]} samples", color="blue")
            tprint(f"📊 [NAS_TAS_MODELS] Test set: {X_test.shape[0]} samples", color="blue")
            tprint(f"📊 [NAS_TAS_MODELS] Train class distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}", color="blue")
            tprint(f"📊 [NAS_TAS_MODELS] Test class distribution: {dict(zip(*np.unique(y_test, return_counts=True)))}", color="blue")
            tprint(f"⏱️ [NAS_TAS_MODELS] Data splitting completed in {split_time:.3f} seconds", color="blue")
            
            # Step 2: Scale features with detailed logging
            tprint("🔧 [NAS_TAS_MODELS] Step 2: Scaling features", color="cyan")
            scale_start = time.time()
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            scale_time = time.time() - scale_start
            tprint(f"📊 [NAS_TAS_MODELS] Scaled features - Train: mean={X_train_scaled.mean():.4f}, std={X_train_scaled.std():.4f}", color="blue")
            tprint(f"📊 [NAS_TAS_MODELS] Scaled features - Test: mean={X_test_scaled.mean():.4f}, std={X_test_scaled.std():.4f}", color="blue")
            tprint(f"⏱️ [NAS_TAS_MODELS] Feature scaling completed in {scale_time:.3f} seconds", color="blue")
            
            # Step 3: Train CatBoost with detailed logging
            tprint("🔧 [NAS_TAS_MODELS] Step 3: Training CatBoost", color="cyan", bold=True)
            cb_start = time.time()
            
            cb_params = {
                'iterations': 100,
                'depth': 6,
                'learning_rate': 0.1,
                'l2_leaf_reg': 3.0,
                'random_seed': self.model_config['random_state'],
                'thread_count': self.model_config['n_jobs'],
                'verbose': False
            }
            tprint(f"🐱 [NAS_TAS_MODELS] CatBoost parameters: {cb_params}", color="blue")
            tprint(f"🐱 [NAS_TAS_MODELS] CatBoost training data: {X_train_scaled.shape[0]} samples, {X_train_scaled.shape[1]} features", color="blue")
            tprint(f"🐱 [NAS_TAS_MODELS] CatBoost target classes: {np.unique(y_train)}", color="blue")
            
            cb_model = cb.CatBoostClassifier(**cb_params)
            tprint("🏋️ [NAS_TAS_MODELS] Starting CatBoost training...", color="yellow")
            cb_model.fit(X_train_scaled, y_train)
            models['catboost'] = cb_model
            
            cb_time = time.time() - cb_start
            tprint(f"⏱️ [NAS_TAS_MODELS] CatBoost training completed in {cb_time:.3f} seconds", color="blue")
            tprint(f"📊 [NAS_TAS_MODELS] CatBoost feature importance: {cb_model.feature_importances_[:5]}", color="blue")
            tprint(f"📊 [NAS_TAS_MODELS] CatBoost training iterations: {cb_model.get_best_iteration()}", color="blue")
            tprint("✅ [NAS_TAS_MODELS] CatBoost model trained successfully", color="green")
            
            # Step 4: Train Bayesian Rule Lists with detailed logging
            tprint("🔧 [NAS_TAS_MODELS] Step 4: Training Bayesian Rule Lists", color="cyan", bold=True)
            brl_start = time.time()
            
            brl_params = {
                'listlengthprior': 3,
                'listwidthprior': 1,
                'maxcardinality': 2,
                'minsupport': 0.1,
                'n_chains': 3,
                'max_iter': 10000,
                'verbose': False,
                'random_state': 42
            }
            tprint(f"📋 [NAS_TAS_MODELS] Bayesian Rule Lists parameters: {brl_params}", color="blue")
            tprint(f"📋 [NAS_TAS_MODELS] Bayesian Rule Lists training data: {X_train_scaled.shape[0]} samples, {X_train_scaled.shape[1]} features", color="blue")
            tprint(f"📋 [NAS_TAS_MODELS] Bayesian Rule Lists target classes: {np.unique(y_train)}", color="blue")
            
            brl_model = BayesianRuleListClassifier(**brl_params)
            tprint("🏋️ [NAS_TAS_MODELS] Starting Bayesian Rule Lists training...", color="yellow")
            brl_model.fit(X_train_scaled, y_train)
            models['bayesian_rule_lists'] = brl_model
            
            brl_time = time.time() - brl_start
            tprint(f"⏱️ [NAS_TAS_MODELS] Bayesian Rule Lists training completed in {brl_time:.3f} seconds", color="blue")
            tprint(f"📊 [NAS_TAS_MODELS] Bayesian Rule Lists rules: {len(brl_model.rules_) if hasattr(brl_model, 'rules_') else 'N/A'}", color="blue")
            tprint(f"📊 [NAS_TAS_MODELS] Bayesian Rule Lists rule complexity: {brl_params['max_rule_length']} conditions per rule", color="blue")
            tprint("✅ [NAS_TAS_MODELS] Bayesian Rule Lists model trained successfully", color="green")
            
            # Step 5: Train ExtraTrees with detailed logging
            tprint("🔧 [NAS_TAS_MODELS] Step 5: Training ExtraTrees", color="cyan", bold=True)
            et_start = time.time()
            
            et_params = {
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'max_features': 'sqrt',
                'random_state': self.model_config['random_state'],
                'n_jobs': self.model_config['n_jobs']
            }
            tprint(f"🌳 [NAS_TAS_MODELS] ExtraTrees parameters: {et_params}", color="blue")
            tprint(f"🌳 [NAS_TAS_MODELS] ExtraTrees training data: {X_train_scaled.shape[0]} samples, {X_train_scaled.shape[1]} features", color="blue")
            tprint(f"🌳 [NAS_TAS_MODELS] ExtraTrees target classes: {np.unique(y_train)}", color="blue")
            
            et_model = ExtraTreesClassifier(**et_params)
            tprint("🏋️ [NAS_TAS_MODELS] Starting ExtraTrees training...", color="yellow")
            et_model.fit(X_train_scaled, y_train)
            models['extratrees'] = et_model
            
            et_time = time.time() - et_start
            tprint(f"⏱️ [NAS_TAS_MODELS] ExtraTrees training completed in {et_time:.3f} seconds", color="blue")
            tprint(f"📊 [NAS_TAS_MODELS] ExtraTrees feature importance: {et_model.feature_importances_[:5]}", color="blue")
            tprint(f"📊 [NAS_TAS_MODELS] ExtraTrees trees trained: {et_params['n_estimators']}", color="blue")
            tprint("✅ [NAS_TAS_MODELS] ExtraTrees model trained successfully", color="green")
            
            # Step 7: Evaluate models with comprehensive metrics
            tprint("🔧 [NAS_TAS_MODELS] Step 7: Evaluating models", color="cyan")
            eval_start = time.time()
            
            for name, model in models.items():
                tprint(f"📊 [NAS_TAS_MODELS] Evaluating {name}", color="blue")
                
                # Make predictions
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
                tprint(f"📊 [NAS_TAS_MODELS] {name} accuracy: {accuracy:.4f}", color="green")
                tprint(f"📊 [NAS_TAS_MODELS] {name} test samples: {len(y_test)}", color="blue")
                if y_pred_proba is not None:
                    tprint(f"📊 [NAS_TAS_MODELS] {name} prediction confidence: {y_pred_proba.max(axis=1).mean():.4f} ± {y_pred_proba.max(axis=1).std():.4f}", color="blue")
            
            eval_time = time.time() - eval_start
            tprint(f"⏱️ [NAS_TAS_MODELS] Model evaluation completed in {eval_time:.3f} seconds", color="blue")
            
            # Calculate total training time
            training_time = time.time() - start_time
            
            # Log comprehensive training summary
            tprint("📊 [NAS_TAS_MODELS] Training Summary:", color="cyan", bold=True)
            tprint(f"⏱️ [NAS_TAS_MODELS] Total training time: {training_time:.2f} seconds", color="blue")
            tprint(f"📊 [NAS_TAS_MODELS] Models trained: {len(models)}", color="blue")
            tprint(f"📊 [NAS_TAS_MODELS] Best accuracy: {max(metrics[m]['accuracy'] for m in metrics):.4f}", color="green")
            
            # Store training history
            training_history = {
                'data_split_time': split_time,
                'scaling_time': scale_time,
                'catboost_time': cb_time,
                'bayesian_rule_lists_time': brl_time,
                'extratrees_time': et_time,
                'evaluation_time': eval_time,
                'total_time': training_time
            }
            
            self.logger.info(f"Model training completed successfully in {training_time:.2f} seconds")
            self.logger.info(f"Trained {len(models)} models with best accuracy: {max(metrics[m]['accuracy'] for m in metrics):.4f}")
            
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
            tprint(f"❌ [NAS_TAS_MODELS] Error training models: {e}", color="red")
            tprint(f"🔍 [NAS_TAS_MODELS] Error type: {error_type}", color="yellow")
            tprint(f"🔍 [NAS_TAS_MODELS] Training time before failure: {training_time:.2f} seconds", color="yellow")
            
            # Provide specific error context for model training
            if "ValueError" in error_type and "n_features" in str(e).lower():
                tprint(f"💡 [NAS_TAS_MODELS] Feature count mismatch. Expected: {X.shape[1]} features", color="cyan")
            elif "MemoryError" in error_type:
                tprint(f"💡 [NAS_TAS_MODELS] Insufficient memory for model training. Try reducing n_jobs or model complexity", color="cyan")
            elif "ImportError" in error_type:
                tprint(f"💡 [NAS_TAS_MODELS] Missing ML library. Check installations", color="cyan")
            elif "AttributeError" in error_type:
                tprint(f"💡 [NAS_TAS_MODELS] Model object issue. Check model initialization", color="cyan")
            
            self.logger.error(f"Error training models after {training_time:.2f} seconds: {str(e)}", exc_info=True)
            self.logger.error(f"Error type: {error_type}")
            
            return {
                'models': {},
                'metrics': {},
                'training_time': training_time,
                'error': str(e),
                'error_type': error_type
            }
