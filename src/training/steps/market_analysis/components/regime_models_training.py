"""
Regime Detection Models Training Component

This component implements the specific regime detection models mentioned in the user's request:
- CatBoost (base model)
- Bayesian Rule Lists (base model) 
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

# Import Bayesian Rule Lists
try:
    from imodels import BayesianRuleListClassifier
    ML_LIBRARY_VERSIONS['imodels'] = "1.0.0"  # Placeholder version
    tprint(f"✅ [REGIME_MODELS] Bayesian Rule Lists imported successfully", color="green")
except ImportError as e:
    ML_IMPORT_ERRORS.append(f"Bayesian Rule Lists: {e}")
    tprint(f"❌ [REGIME_MODELS] Failed to import Bayesian Rule Lists: {e}", color="red")

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
    - Bayesian Rule Lists (base model)
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
                'Bayesian Rule Lists': {
                    'max_rules': 12,
                    'max_rule_length': 3,
                    'n_chains': 3,
                    'n_iter': 10000
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
                    'num_leaves': 31,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'n_estimators': 100,
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
            
            regime_labels = nas_tas_clustering_result.get('regime_assignments')
            
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
            X, y = self._prepare_training_data(data, regime_labels)
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
            
            # Check each model
            for name, model in models.items():
                if model is None:
                    tprint(f"❌ [REGIME_MODELS] Model {name} is None", color="red")
                    return False
                
                # Check if model has required methods
                if not hasattr(model, 'predict'):
                    tprint(f"❌ [REGIME_MODELS] Model {name} missing predict method", color="red")
                    return False
            
            tprint(f"✅ [REGIME_MODELS] Model validation passed - {len(models)} models validated", color="green")
            return True
            
        except Exception as e:
            tprint(f"❌ [REGIME_MODELS] Model validation error: {e}", color="red")
            return False
    
    def _prepare_training_data(self, data: pd.DataFrame, regime_labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from market data and regime labels."""
        tprint("🔧 [REGIME_MODELS] Preparing training data", color="cyan")
        self.logger.info("Starting data preparation process")
        
        try:
            # Log input data characteristics
            tprint(f"📊 [REGIME_MODELS] Input data shape: {data.shape}", color="blue")
            tprint(f"📊 [REGIME_MODELS] Input data columns: {list(data.columns)}", color="blue")
            
            # Create basic features from OHLCV data
            features = []
            feature_names = []
            
            tprint("🔧 [REGIME_MODELS] Creating price-based features", color="cyan")
            if 'close' in data.columns:
                # Price-based features
                tprint("📈 [REGIME_MODELS] Computing price returns", color="blue")
                returns = data['close'].pct_change().fillna(0)
                features.append(returns.values)
                feature_names.append('price_returns')
                
                # Moving averages
                tprint("📈 [REGIME_MODELS] Computing moving averages", color="blue")
                sma_20 = data['close'].rolling(20).mean().fillna(data['close'].iloc[0])
                sma_50 = data['close'].rolling(50).mean().fillna(data['close'].iloc[0])
                features.append(sma_20.values)
                features.append(sma_50.values)
                feature_names.extend(['sma_20', 'sma_50'])
                
                # Volatility
                tprint("📈 [REGIME_MODELS] Computing volatility", color="blue")
                volatility = returns.rolling(20).std().fillna(0)
                features.append(volatility.values)
                feature_names.append('volatility_20')
                
                # RSI-like indicator
                tprint("📈 [REGIME_MODELS] Computing RSI", color="blue")
                price_change = data['close'].diff()
                gain = price_change.where(price_change > 0, 0)
                loss = -price_change.where(price_change < 0, 0)
                avg_gain = gain.rolling(14).mean().fillna(0)
                avg_loss = loss.rolling(14).mean().fillna(0)
                rs = avg_gain / (avg_loss + 1e-8)
                rsi = 100 - (100 / (1 + rs))
                features.append(rsi.values)
                feature_names.append('rsi_14')
            
            # Combine features
            tprint("🔧 [REGIME_MODELS] Combining features into feature matrix", color="cyan")
            X = np.column_stack(features)
            tprint(f"📊 [REGIME_MODELS] Feature matrix shape: {X.shape}", color="blue")
            
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
            
            tprint(f"✅ [REGIME_MODELS] Training data prepared: {X.shape[0]} samples, {X.shape[1]} features", color="green")
            
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
            
            # Step 3: Train CatBoost
            tprint("🔧 [REGIME_MODELS] Step 3: Training CatBoost", color="cyan")
            catboost_start = time.time()
            
            try:
                catboost_model = cb.CatBoostClassifier(**self.regime_models_config['base']['CatBoost'])
                catboost_model.fit(X_train_scaled, y_train)
                models['CatBoost'] = catboost_model
                
                catboost_time = time.time() - catboost_start
                tprint(f"⏱️ [REGIME_MODELS] CatBoost training completed in {catboost_time:.3f} seconds", color="blue")
            except Exception as e:
                tprint(f"❌ [REGIME_MODELS] CatBoost training failed: {e}", color="red")
                models['CatBoost'] = None
            
            # Step 4: Train Bayesian Rule Lists
            tprint("🔧 [REGIME_MODELS] Step 4: Training Bayesian Rule Lists", color="cyan")
            brl_start = time.time()
            
            try:
                brl_model = BayesianRuleListClassifier(**self.regime_models_config['base']['Bayesian Rule Lists'])
                brl_model.fit(X_train_scaled, y_train)
                models['Bayesian Rule Lists'] = brl_model
                
                brl_time = time.time() - brl_start
                tprint(f"⏱️ [REGIME_MODELS] Bayesian Rule Lists training completed in {brl_time:.3f} seconds", color="blue")
            except Exception as e:
                tprint(f"❌ [REGIME_MODELS] Bayesian Rule Lists training failed: {e}", color="red")
                models['Bayesian Rule Lists'] = None
            
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
            
            # Step 6: Train stacker_lgbm_calibrated (meta-learner)
            tprint("🔧 [REGIME_MODELS] Step 6: Training stacker_lgbm_calibrated meta-learner", color="cyan")
            meta_start = time.time()
            
            try:
                # Create base models for stacking
                base_models = {}
                for name, model in models.items():
                    if model is not None:
                        base_models[name] = model
                
                if base_models:
                    # Create LightGBM meta-learner with calibration
                    meta_learner = lgb.LGBMClassifier(**self.regime_models_config['meta_learner']['stacker_lgbm_calibrated'])
                    
                    # Train meta-learner on base model predictions
                    base_predictions = np.column_stack([
                        model.predict_proba(X_train_scaled) if hasattr(model, 'predict_proba') else model.predict(X_train_scaled).reshape(-1, 1)
                        for model in base_models.values()
                    ])
                    
                    meta_learner.fit(base_predictions, y_train)
                    models['stacker_lgbm_calibrated'] = meta_learner
                    
                    meta_time = time.time() - meta_start
                    tprint(f"⏱️ [REGIME_MODELS] Meta-learner training completed in {meta_time:.3f} seconds", color="blue")
                else:
                    tprint("⚠️ [REGIME_MODELS] No base models available for meta-learner", color="yellow")
                    models['stacker_lgbm_calibrated'] = None
                    
            except Exception as e:
                tprint(f"❌ [REGIME_MODELS] Meta-learner training failed: {e}", color="red")
                models['stacker_lgbm_calibrated'] = None
            
            # Step 7: Evaluate models
            tprint("🔧 [REGIME_MODELS] Step 7: Evaluating models", color="cyan")
            eval_start = time.time()
            
            for name, model in models.items():
                if model is not None:
                    tprint(f"📊 [REGIME_MODELS] Evaluating {name}", color="blue")
                    
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