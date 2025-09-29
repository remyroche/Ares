"""
NAS-TAS Ensemble Training Component

This component implements ensemble training for NAS-TAS (Neural Architecture Search - Tree-based Architecture Search) based regime detection models.
It creates meta-models that combine multiple base models trained on NAS-TAS regime labels.
"""

import numpy as np
import pandas as pd
import pickle
import json
import os
import warnings
import sys
import traceback
import psutil
import time
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from pathlib import Path

from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, 
    tprint_success, tprint_progress, tprint_performance, tprint_timer,
    tprint_structured, tprint_with_level, LogLevel
)
from src.utils.logger import system_logger
from .base_component import BaseMarketAnalysisComponent, ComponentConfig, ComponentResult

# Suppress LightGBM warnings about no further splits
warnings.filterwarnings('ignore', message='.*No further splits with positive gain.*')

tprint("🚀 [NAS_TAS_ENSEMBLE] Starting NAS-TAS Ensemble Training Component initialization", color="cyan", bold=True)
tprint_debug("📦 [NAS_TAS_ENSEMBLE] Importing core dependencies...")

# Import ensemble training classes
tprint_debug("🔧 [NAS_TAS_ENSEMBLE] Attempting to import ensemble training classes...")
try:
    from src.analyst.predictive_ensembles.ensemble_orchestrator import RegimePredictiveEnsembles
    from src.analyst.predictive_ensembles.regime_ensembles.volatile_regime_ensemble import VolatileRegimeEnsemble
    ENSEMBLE_AVAILABLE = True
    tprint_success("✅ [NAS_TAS_ENSEMBLE] Ensemble training classes imported successfully")
    tprint_debug(f"📊 [NAS_TAS_ENSEMBLE] Available ensemble classes: RegimePredictiveEnsembles, VolatileRegimeEnsemble")
except ImportError as e:
    ENSEMBLE_AVAILABLE = False
    tprint_error(f"❌ [NAS_TAS_ENSEMBLE] Failed to import ensemble training classes: {e}")
    tprint_debug(f"🔍 [NAS_TAS_ENSEMBLE] Import error details: {traceback.format_exc()}")

# Import ML libraries
tprint_debug("🔧 [NAS_TAS_ENSEMBLE] Attempting to import ML libraries...")
ML_LIBRARIES_IMPORTED = []
ML_LIBRARIES_FAILED = []

try:
    from sklearn.ensemble import VotingClassifier, StackingClassifier
    ML_LIBRARIES_IMPORTED.append("sklearn.ensemble")
    tprint_debug("✅ [NAS_TAS_ENSEMBLE] sklearn.ensemble imported successfully")
except ImportError as e:
    ML_LIBRARIES_FAILED.append(f"sklearn.ensemble: {e}")

try:
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    ML_LIBRARIES_IMPORTED.append("sklearn.model_selection")
    tprint_debug("✅ [NAS_TAS_ENSEMBLE] sklearn.model_selection imported successfully")
except ImportError as e:
    ML_LIBRARIES_FAILED.append(f"sklearn.model_selection: {e}")

try:
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    ML_LIBRARIES_IMPORTED.append("sklearn.metrics")
    tprint_debug("✅ [NAS_TAS_ENSEMBLE] sklearn.metrics imported successfully")
except ImportError as e:
    ML_LIBRARIES_FAILED.append(f"sklearn.metrics: {e}")

try:
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    ML_LIBRARIES_IMPORTED.append("sklearn.preprocessing")
    tprint_debug("✅ [NAS_TAS_ENSEMBLE] sklearn.preprocessing imported successfully")
except ImportError as e:
    ML_LIBRARIES_FAILED.append(f"sklearn.preprocessing: {e}")

try:
    from lightgbm import LGBMClassifier
    ML_LIBRARIES_IMPORTED.append("lightgbm")
    tprint_debug("✅ [NAS_TAS_ENSEMBLE] lightgbm imported successfully")
except ImportError as e:
    ML_LIBRARIES_FAILED.append(f"lightgbm: {e}")

try:
    from xgboost import XGBClassifier
    ML_LIBRARIES_IMPORTED.append("xgboost")
    tprint_debug("✅ [NAS_TAS_ENSEMBLE] xgboost imported successfully")
except ImportError as e:
    ML_LIBRARIES_FAILED.append(f"xgboost: {e}")

try:
    from sklearn.linear_model import LogisticRegression
    ML_LIBRARIES_IMPORTED.append("sklearn.linear_model")
    tprint_debug("✅ [NAS_TAS_ENSEMBLE] sklearn.linear_model imported successfully")
except ImportError as e:
    ML_LIBRARIES_FAILED.append(f"sklearn.linear_model: {e}")

try:
    from sklearn.svm import SVC
    ML_LIBRARIES_IMPORTED.append("sklearn.svm")
    tprint_debug("✅ [NAS_TAS_ENSEMBLE] sklearn.svm imported successfully")
except ImportError as e:
    ML_LIBRARIES_FAILED.append(f"sklearn.svm: {e}")

try:
    from sklearn.ensemble import RandomForestClassifier
    ML_LIBRARIES_IMPORTED.append("sklearn.ensemble.RandomForestClassifier")
    tprint_debug("✅ [NAS_TAS_ENSEMBLE] sklearn.ensemble.RandomForestClassifier imported successfully")
except ImportError as e:
    ML_LIBRARIES_FAILED.append(f"sklearn.ensemble.RandomForestClassifier: {e}")

# Check overall ML libraries availability
if len(ML_LIBRARIES_IMPORTED) >= 5:  # Minimum required libraries
    ML_LIBRARIES_AVAILABLE = True
    tprint_success(f"✅ [NAS_TAS_ENSEMBLE] ML libraries imported successfully ({len(ML_LIBRARIES_IMPORTED)}/{len(ML_LIBRARIES_IMPORTED) + len(ML_LIBRARIES_FAILED)})")
    tprint_debug(f"📊 [NAS_TAS_ENSEMBLE] Successfully imported: {', '.join(ML_LIBRARIES_IMPORTED)}")
    if ML_LIBRARIES_FAILED:
        tprint_warning(f"⚠️ [NAS_TAS_ENSEMBLE] Some libraries failed to import: {', '.join(ML_LIBRARIES_FAILED)}")
else:
    ML_LIBRARIES_AVAILABLE = False
    tprint_error(f"❌ [NAS_TAS_ENSEMBLE] Insufficient ML libraries available ({len(ML_LIBRARIES_IMPORTED)}/{len(ML_LIBRARIES_IMPORTED) + len(ML_LIBRARIES_FAILED)})")
    tprint_error(f"🔍 [NAS_TAS_ENSEMBLE] Failed imports: {', '.join(ML_LIBRARIES_FAILED)}")

# Log system information
tprint_debug(f"🖥️ [NAS_TAS_ENSEMBLE] System information:")
tprint_debug(f"   - Python version: {sys.version}")
tprint_debug(f"   - Available memory: {psutil.virtual_memory().available / (1024**3):.2f} GB")
tprint_debug(f"   - CPU count: {psutil.cpu_count()}")
tprint_debug(f"   - Current working directory: {os.getcwd()}")

tprint_success("✅ [NAS_TAS_ENSEMBLE] All imports completed successfully", color="green", bold=True)


class NASTASEnsembleTrainingComponent(BaseMarketAnalysisComponent):
    """
    NAS-TAS Ensemble Training Component.
    
    This component trains ensemble models using NAS-TAS regime labels for meta-learning.
    It combines multiple base models into voting and stacking ensembles.
    """
    
    def __init__(self, config: Optional[ComponentConfig] = None):
        """Initialize the NAS-TAS Ensemble Training Component."""
        init_start_time = time.perf_counter()
        tprint("🚀 [NAS_TAS_ENSEMBLE] Initializing NAS-TAS Ensemble Training Component", color="cyan", bold=True)
        
        # Log initialization parameters
        tprint_debug(f"🔧 [NAS_TAS_ENSEMBLE] Initialization parameters:")
        tprint_debug(f"   - Config provided: {config is not None}")
        tprint_debug(f"   - Config type: {type(config)}")
        if config:
            tprint_debug(f"   - Config details: {config.__dict__ if hasattr(config, '__dict__') else 'No attributes'}")
        
        # Initialize base component
        tprint_debug("📦 [NAS_TAS_ENSEMBLE] Calling parent class initialization...")
        try:
            super().__init__(config)
            tprint_success("✅ [NAS_TAS_ENSEMBLE] Parent class initialization completed")
        except Exception as e:
            tprint_error(f"❌ [NAS_TAS_ENSEMBLE] Parent class initialization failed: {e}")
            tprint_debug(f"🔍 [NAS_TAS_ENSEMBLE] Parent init error details: {traceback.format_exc()}")
            raise
        
        # Initialize logger with detailed configuration
        tprint_debug("📝 [NAS_TAS_ENSEMBLE] Setting up logger...")
        try:
            self.logger = system_logger.getChild('NASTASEnsembleTrainingComponent')
            self.logger.setLevel('DEBUG')
            tprint_success("✅ [NAS_TAS_ENSEMBLE] Logger initialized successfully")
            tprint_debug(f"📊 [NAS_TAS_ENSEMBLE] Logger name: {self.logger.name}")
            tprint_debug(f"📊 [NAS_TAS_ENSEMBLE] Logger level: {self.logger.level}")
        except Exception as e:
            tprint_error(f"❌ [NAS_TAS_ENSEMBLE] Logger initialization failed: {e}")
            tprint_debug(f"🔍 [NAS_TAS_ENSEMBLE] Logger error details: {traceback.format_exc()}")
        
        # Initialize ensemble training parameters with detailed logging
        tprint_debug("⚙️ [NAS_TAS_ENSEMBLE] Setting up ensemble configuration...")
        self.ensemble_config = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Log configuration details
        tprint_structured({
            "ensemble_config": self.ensemble_config,
            "config_source": "default_values",
            "timestamp": datetime.now().isoformat()
        }, LogLevel.DEBUG)
        
        tprint_success("✅ [NAS_TAS_ENSEMBLE] Ensemble configuration set")
        tprint_debug(f"📊 [NAS_TAS_ENSEMBLE] Configuration details:")
        for key, value in self.ensemble_config.items():
            tprint_debug(f"   - {key}: {value} (type: {type(value).__name__})")
        
        # Initialize ensemble models with detailed setup
        tprint_debug("📊 [NAS_TAS_ENSEMBLE] Initializing ensemble models...")
        self.voting_ensemble = None
        self.stacking_ensemble = None
        self.base_models = {}
        self.ensemble_metrics = {}
        
        # Log model initialization
        tprint_debug(f"📊 [NAS_TAS_ENSEMBLE] Model states initialized:")
        tprint_debug(f"   - voting_ensemble: {type(self.voting_ensemble).__name__}")
        tprint_debug(f"   - stacking_ensemble: {type(self.stacking_ensemble).__name__}")
        tprint_debug(f"   - base_models: {type(self.base_models).__name__} (empty: {len(self.base_models) == 0})")
        tprint_debug(f"   - ensemble_metrics: {type(self.ensemble_metrics).__name__} (empty: {len(self.ensemble_metrics) == 0})")
        
        # Log system resources
        tprint_debug("🖥️ [NAS_TAS_ENSEMBLE] System resources at initialization:")
        try:
            memory_info = psutil.virtual_memory()
            tprint_debug(f"   - Total memory: {memory_info.total / (1024**3):.2f} GB")
            tprint_debug(f"   - Available memory: {memory_info.available / (1024**3):.2f} GB")
            tprint_debug(f"   - Memory usage: {memory_info.percent:.1f}%")
            tprint_debug(f"   - CPU count: {psutil.cpu_count()}")
            tprint_debug(f"   - CPU usage: {psutil.cpu_percent(interval=1):.1f}%")
        except Exception as e:
            tprint_warning(f"⚠️ [NAS_TAS_ENSEMBLE] Could not retrieve system resources: {e}")
        
        # Validate ML libraries availability
        tprint_debug("🔍 [NAS_TAS_ENSEMBLE] Validating ML libraries availability...")
        if not ML_LIBRARIES_AVAILABLE:
            tprint_error("❌ [NAS_TAS_ENSEMBLE] Critical ML libraries not available - component may not function properly")
            tprint_warning("⚠️ [NAS_TAS_ENSEMBLE] Some ensemble training features may be limited")
        else:
            tprint_success("✅ [NAS_TAS_ENSEMBLE] All required ML libraries are available")
        
        if not ENSEMBLE_AVAILABLE:
            tprint_warning("⚠️ [NAS_TAS_ENSEMBLE] Advanced ensemble classes not available - using basic sklearn ensembles")
        else:
            tprint_success("✅ [NAS_TAS_ENSEMBLE] Advanced ensemble classes are available")
        
        # Calculate initialization time
        init_duration = time.perf_counter() - init_start_time
        tprint_performance("NAS-TAS Ensemble Training Component initialization", init_duration)
        
        tprint_success("✅ [NAS_TAS_ENSEMBLE] NAS-TAS Ensemble Training Component initialized successfully", color="green", bold=True)
        tprint_debug(f"📊 [NAS_TAS_ENSEMBLE] Initialization completed in {init_duration:.3f}s")
    
    def get_required_artifacts(self) -> List[str]:
        """Get list of required artifacts this component must produce."""
        tprint("📋 [NAS_TAS_ENSEMBLE] Getting required artifacts", color="cyan")
        required_artifacts = ['nas_tas_ensemble_training_result']
        tprint(f"✅ [NAS_TAS_ENSEMBLE] Required artifacts: {required_artifacts}", color="green")
        return required_artifacts
    
    async def execute(self, data: pd.DataFrame, pipeline_state: Dict[str, Any]) -> ComponentResult:
        """
        Execute NAS ensemble training.
        
        Args:
            data: Market data DataFrame
            pipeline_state: Pipeline state containing features, targets, and regime labels
            
        Returns:
            ComponentResult with training results
        """
        execution_start_time = time.perf_counter()
        start_time = datetime.now()
        
        tprint("🚀 [NAS_TAS_ENSEMBLE] Starting NAS ensemble training execution", color="cyan", bold=True)
        tprint_debug(f"📊 [NAS_TAS_ENSEMBLE] Execution started at: {start_time.isoformat()}")
        
        # Log system resources at start
        try:
            memory_info = psutil.virtual_memory()
            tprint_debug(f"🖥️ [NAS_TAS_ENSEMBLE] System resources at execution start:")
            tprint_debug(f"   - Available memory: {memory_info.available / (1024**3):.2f} GB")
            tprint_debug(f"   - Memory usage: {memory_info.percent:.1f}%")
            tprint_debug(f"   - CPU usage: {psutil.cpu_percent(interval=0.1):.1f}%")
        except Exception as e:
            tprint_warning(f"⚠️ [NAS_TAS_ENSEMBLE] Could not retrieve system resources: {e}")
        
        # Log input parameters
        tprint_debug(f"📥 [NAS_TAS_ENSEMBLE] Input parameters:")
        tprint_debug(f"   - Data type: {type(data).__name__}")
        tprint_debug(f"   - Data shape: {data.shape if hasattr(data, 'shape') else 'No shape attribute'}")
        tprint_debug(f"   - Pipeline state type: {type(pipeline_state).__name__}")
        tprint_debug(f"   - Pipeline state keys: {list(pipeline_state.keys()) if pipeline_state else 'Empty'}")
        
        try:
            # Extract required data from pipeline state with detailed logging
            tprint("📊 [NAS_TAS_ENSEMBLE] Extracting data from pipeline state", color="yellow")
            tprint_debug("🔍 [NAS_TAS_ENSEMBLE] Extracting features...")
            X = pipeline_state.get('features')
            tprint_debug(f"   - Features type: {type(X).__name__}")
            tprint_debug(f"   - Features shape: {X.shape if hasattr(X, 'shape') else 'No shape'}")
            tprint_debug(f"   - Features is None: {X is None}")
            
            tprint_debug("🔍 [NAS_TAS_ENSEMBLE] Extracting targets...")
            y = pipeline_state.get('targets')
            tprint_debug(f"   - Targets type: {type(y).__name__}")
            tprint_debug(f"   - Targets shape: {y.shape if hasattr(y, 'shape') else 'No shape'}")
            tprint_debug(f"   - Targets is None: {y is None}")
            
            # Extract regime labels from pipeline state artifacts
            tprint_debug("🔍 [NAS_TAS_ENSEMBLE] Extracting artifacts...")
            artifacts = pipeline_state.get('artifacts', {})
            tprint_debug(f"   - Artifacts type: {type(artifacts).__name__}")
            tprint_debug(f"   - Artifacts keys: {list(artifacts.keys()) if artifacts else 'Empty'}")
            
            tprint_debug("🔍 [NAS_TAS_ENSEMBLE] Extracting NAS-TAS clustering result...")
            nas_tas_clustering_result = artifacts.get('nas_tas_clustering_result', {})
            tprint_debug(f"   - NAS-TAS clustering result type: {type(nas_tas_clustering_result).__name__}")
            tprint_debug(f"   - NAS-TAS clustering result keys: {list(nas_tas_clustering_result.keys()) if nas_tas_clustering_result else 'Empty'}")
            
            regime_labels = nas_tas_clustering_result.get('regime_assignments')
            tprint_debug(f"   - Regime labels type: {type(regime_labels).__name__}")
            tprint_debug(f"   - Regime labels shape: {regime_labels.shape if hasattr(regime_labels, 'shape') else 'No shape'}")
            tprint_debug(f"   - Regime labels is None: {regime_labels is None}")
            
            feature_names = pipeline_state.get('feature_names', [])
            tprint_debug(f"   - Feature names type: {type(feature_names).__name__}")
            tprint_debug(f"   - Feature names count: {len(feature_names) if feature_names else 0}")
            
            nas_models = pipeline_state.get('nas_models', {})
            tprint_debug(f"   - NAS models type: {type(nas_models).__name__}")
            tprint_debug(f"   - NAS models count: {len(nas_models) if nas_models else 0}")
            
            # Validate required data with detailed checks
            tprint("🔍 [NAS_TAS_ENSEMBLE] Validating required data", color="yellow")
            
            validation_errors = []
            
            if X is None:
                validation_errors.append("Features (X) is None")
            elif hasattr(X, 'shape') and X.shape[0] == 0:
                validation_errors.append("Features (X) is empty")
            elif hasattr(X, 'shape') and len(X.shape) != 2:
                validation_errors.append(f"Features (X) has invalid shape: {X.shape}")
            
            if y is None:
                validation_errors.append("Targets (y) is None")
            elif hasattr(y, 'shape') and y.shape[0] == 0:
                validation_errors.append("Targets (y) is empty")
            elif hasattr(y, 'shape') and len(y.shape) != 1:
                validation_errors.append(f"Targets (y) has invalid shape: {y.shape}")
            
            if validation_errors:
                error_msg = f"Data validation failed: {'; '.join(validation_errors)}"
                tprint_error(f"❌ [NAS_TAS_ENSEMBLE] {error_msg}")
                tprint_debug(f"🔍 [NAS_TAS_ENSEMBLE] Validation details:")
                for error in validation_errors:
                    tprint_debug(f"   - {error}")
                
                return ComponentResult(
                    success=False,
                    artifacts={},
                    error_message=error_msg,
                    metadata={'component_type': 'nas_tas_ensemble_training', 'validation_errors': validation_errors}
                )
            
            tprint_success("✅ [NAS_TAS_ENSEMBLE] Data validation passed")
            
            if regime_labels is None:
                tprint_warning("⚠️ [NAS_TAS_ENSEMBLE] No regime labels found, using targets as regime labels")
                regime_labels = y
                tprint_debug(f"   - Using targets as regime labels: {type(regime_labels).__name__}")
            else:
                tprint_success("✅ [NAS_TAS_ENSEMBLE] Regime labels found")
            
            # Log data shapes and types
            tprint_debug(f"📊 [NAS_TAS_ENSEMBLE] Data summary:")
            tprint_debug(f"   - X shape: {X.shape if hasattr(X, 'shape') else 'No shape'}")
            tprint_debug(f"   - y shape: {y.shape if hasattr(y, 'shape') else 'No shape'}")
            tprint_debug(f"   - regime_labels shape: {regime_labels.shape if hasattr(regime_labels, 'shape') else 'No shape'}")
            tprint_debug(f"   - X dtype: {X.dtype if hasattr(X, 'dtype') else 'No dtype'}")
            tprint_debug(f"   - y dtype: {y.dtype if hasattr(y, 'dtype') else 'No dtype'}")
            tprint_debug(f"   - regime_labels dtype: {regime_labels.dtype if hasattr(regime_labels, 'dtype') else 'No dtype'}")
            
            # Check for data consistency
            if hasattr(X, 'shape') and hasattr(y, 'shape') and X.shape[0] != y.shape[0]:
                tprint_error(f"❌ [NAS_TAS_ENSEMBLE] Data inconsistency: X has {X.shape[0]} samples, y has {y.shape[0]} samples")
                return ComponentResult(
                    success=False,
                    artifacts={},
                    error_message=f"Data inconsistency: X has {X.shape[0]} samples, y has {y.shape[0]} samples",
                    metadata={'component_type': 'nas_tas_ensemble_training'}
                )
            
            # Prepare data for ensemble training
            tprint("🔧 [NAS_TAS_ENSEMBLE] Preparing data for ensemble training", color="yellow")
            with tprint_timer("Data preparation"):
                X_processed, y_processed, regime_labels_processed = self._prepare_data(X, y, regime_labels)
            
            tprint_success(f"✅ [NAS_TAS_ENSEMBLE] Data prepared - X: {X_processed.shape}, y: {y_processed.shape}")
            
            # Train base models if not provided
            tprint("🏋️ [NAS_TAS_ENSEMBLE] Training base models", color="yellow")
            if not nas_models:
                tprint_info("📝 [NAS_TAS_ENSEMBLE] No pre-trained NAS models found, training base models")
                with tprint_timer("Base model training"):
                    base_models = self._train_base_models(X_processed, y_processed, regime_labels_processed)
            else:
                tprint_info("📝 [NAS_TAS_ENSEMBLE] Using pre-trained NAS models")
                base_models = nas_models
                tprint_debug(f"   - Pre-trained models count: {len(base_models)}")
                tprint_debug(f"   - Pre-trained model types: {[type(model).__name__ for model in base_models.values()]}")
            
            # Train voting ensemble
            tprint("🗳️ [NAS_TAS_ENSEMBLE] Training voting ensemble", color="yellow")
            with tprint_timer("Voting ensemble training"):
                voting_ensemble = self._train_voting_ensemble(X_processed, y_processed, base_models)
            
            # Train stacking ensemble
            tprint("📚 [NAS_TAS_ENSEMBLE] Training stacking ensemble", color="yellow")
            with tprint_timer("Stacking ensemble training"):
                stacking_ensemble = self._train_stacking_ensemble(X_processed, y_processed, base_models)
            
            # Evaluate ensembles
            tprint("📊 [NAS_TAS_ENSEMBLE] Evaluating ensemble performance", color="yellow")
            with tprint_timer("Ensemble evaluation"):
                ensemble_metrics = self._evaluate_ensembles(X_processed, y_processed, voting_ensemble, stacking_ensemble)
            
            # Create comprehensive results
            tprint("📦 [NAS_TAS_ENSEMBLE] Creating comprehensive results", color="yellow")
            
            # Calculate execution time
            execution_duration = time.perf_counter() - execution_start_time
            total_execution_time = (datetime.now() - start_time).total_seconds()
            
            # Log final system resources
            try:
                memory_info = psutil.virtual_memory()
                tprint_debug(f"🖥️ [NAS_TAS_ENSEMBLE] System resources at execution end:")
                tprint_debug(f"   - Available memory: {memory_info.available / (1024**3):.2f} GB")
                tprint_debug(f"   - Memory usage: {memory_info.percent:.1f}%")
            except Exception as e:
                tprint_warning(f"⚠️ [NAS_TAS_ENSEMBLE] Could not retrieve final system resources: {e}")
            
            results = {
                'nas_tas_ensemble_training_result': {
                    'voting_ensemble': voting_ensemble,
                    'stacking_ensemble': stacking_ensemble,
                    'base_models': base_models,
                    'ensemble_metrics': ensemble_metrics,
                    'training_time': total_execution_time,
                    'success': True,
                    'metadata': {
                        'component_type': 'nas_tas_ensemble_training',
                        'data_shape': X_processed.shape,
                        'n_regimes': len(np.unique(regime_labels_processed)) if regime_labels_processed is not None else 0,
                        'feature_names': feature_names,
                        'timestamp': datetime.now().isoformat(),
                        'execution_duration_seconds': execution_duration,
                        'base_models_count': len(base_models),
                        'voting_ensemble_available': voting_ensemble is not None,
                        'stacking_ensemble_available': stacking_ensemble is not None
                    }
                }
            }
            
            # Log structured results
            tprint_structured({
                "execution_summary": {
                    "success": True,
                    "execution_time_seconds": execution_duration,
                    "data_shape": X_processed.shape,
                    "base_models_trained": len(base_models),
                    "ensembles_created": {
                        "voting": voting_ensemble is not None,
                        "stacking": stacking_ensemble is not None
                    }
                }
            }, LogLevel.INFO)
            
            tprint_success("✅ [NAS_TAS_ENSEMBLE] NAS ensemble training completed successfully", color="green", bold=True)
            tprint_performance("Total NAS ensemble training execution", execution_duration)
            
            return ComponentResult(
                success=True,
                artifacts=results,
                metadata={'component_type': 'nas_tas_ensemble_training', 'execution_time': execution_duration}
            )
            
        except Exception as e:
            execution_duration = time.perf_counter() - execution_start_time
            error_msg = f"NAS ensemble training failed: {e}"
            tprint_error(f"❌ [NAS_TAS_ENSEMBLE] {error_msg}", color="red", bold=True)
            tprint_debug(f"🔍 [NAS_TAS_ENSEMBLE] Error details: {traceback.format_exc()}")
            tprint_performance("Failed NAS ensemble training execution", execution_duration)
            
            # Log error to system logger
            self.logger.error(f"NAS ensemble training failed: {e}", exc_info=True)
            
            return ComponentResult(
                success=False,
                artifacts={},
                error_message=str(e),
                metadata={'component_type': 'nas_tas_ensemble_training', 'execution_time': execution_duration}
            )
    
    def _prepare_data(self, X: np.ndarray, y: np.ndarray, regime_labels: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for ensemble training."""
        prep_start_time = time.perf_counter()
        tprint("🔧 [NAS_TAS_ENSEMBLE] Preparing data for ensemble training", color="yellow")
        
        # Log input data characteristics
        tprint_debug(f"📊 [NAS_TAS_ENSEMBLE] Input data characteristics:")
        tprint_debug(f"   - X type: {type(X).__name__}")
        tprint_debug(f"   - X shape: {X.shape if hasattr(X, 'shape') else 'No shape'}")
        tprint_debug(f"   - X dtype: {X.dtype if hasattr(X, 'dtype') else 'No dtype'}")
        tprint_debug(f"   - y type: {type(y).__name__}")
        tprint_debug(f"   - y shape: {y.shape if hasattr(y, 'shape') else 'No shape'}")
        tprint_debug(f"   - y dtype: {y.dtype if hasattr(y, 'dtype') else 'No dtype'}")
        tprint_debug(f"   - regime_labels type: {type(regime_labels).__name__}")
        tprint_debug(f"   - regime_labels shape: {regime_labels.shape if hasattr(regime_labels, 'shape') else 'No shape'}")
        tprint_debug(f"   - regime_labels dtype: {regime_labels.dtype if hasattr(regime_labels, 'dtype') else 'No dtype'}")
        
        # Handle missing values with detailed logging
        tprint("🧹 [NAS_TAS_ENSEMBLE] Handling missing values", color="blue")
        
        # Process X
        tprint_debug("🔍 [NAS_TAS_ENSEMBLE] Processing features (X)...")
        if isinstance(X, pd.DataFrame):
            tprint_debug(f"   - Converting DataFrame to numpy array")
            tprint_debug(f"   - DataFrame shape: {X.shape}")
            tprint_debug(f"   - DataFrame columns: {list(X.columns) if hasattr(X, 'columns') else 'No columns'}")
            tprint_debug(f"   - Missing values in DataFrame: {X.isnull().sum().sum()}")
            X = X.fillna(0).values
            tprint_debug(f"   - After fillna: {X.shape}")
        elif isinstance(X, list):
            tprint_debug(f"   - Converting list to numpy array")
            tprint_debug(f"   - List length: {len(X)}")
            X = np.array(X)
            tprint_debug(f"   - After conversion: {X.shape}")
        elif isinstance(X, np.ndarray):
            tprint_debug(f"   - X is already numpy array")
            tprint_debug(f"   - Missing values in array: {np.isnan(X).sum() if X.dtype in [np.float32, np.float64] else 'Not applicable'}")
        else:
            tprint_warning(f"   - Unexpected X type: {type(X)}")
        
        # Process y
        tprint_debug("🔍 [NAS_TAS_ENSEMBLE] Processing targets (y)...")
        if isinstance(y, pd.Series):
            tprint_debug(f"   - Converting Series to numpy array")
            tprint_debug(f"   - Series shape: {y.shape}")
            tprint_debug(f"   - Missing values in Series: {y.isnull().sum()}")
            y = y.values
            tprint_debug(f"   - After conversion: {y.shape}")
        elif isinstance(y, list):
            tprint_debug(f"   - Converting list to numpy array")
            tprint_debug(f"   - List length: {len(y)}")
            y = np.array(y)
            tprint_debug(f"   - After conversion: {y.shape}")
        elif isinstance(y, np.ndarray):
            tprint_debug(f"   - y is already numpy array")
            tprint_debug(f"   - Missing values in array: {np.isnan(y).sum() if y.dtype in [np.float32, np.float64] else 'Not applicable'}")
        else:
            tprint_warning(f"   - Unexpected y type: {type(y)}")
        
        # Process regime_labels
        tprint_debug("🔍 [NAS_TAS_ENSEMBLE] Processing regime labels...")
        if regime_labels is not None:
            if isinstance(regime_labels, pd.Series):
                tprint_debug(f"   - Converting Series to numpy array")
                tprint_debug(f"   - Series shape: {regime_labels.shape}")
                tprint_debug(f"   - Missing values in Series: {regime_labels.isnull().sum()}")
                regime_labels = regime_labels.values
                tprint_debug(f"   - After conversion: {regime_labels.shape}")
            elif isinstance(regime_labels, list):
                tprint_debug(f"   - Converting list to numpy array")
                tprint_debug(f"   - List length: {len(regime_labels)}")
                regime_labels = np.array(regime_labels)
                tprint_debug(f"   - After conversion: {regime_labels.shape}")
            elif isinstance(regime_labels, np.ndarray):
                tprint_debug(f"   - regime_labels is already numpy array")
                tprint_debug(f"   - Missing values in array: {np.isnan(regime_labels).sum() if regime_labels.dtype in [np.float32, np.float64] else 'Not applicable'}")
            else:
                tprint_warning(f"   - Unexpected regime_labels type: {type(regime_labels)}")
        else:
            tprint_debug(f"   - regime_labels is None")
        
        # Log data after initial processing
        tprint_debug(f"📊 [NAS_TAS_ENSEMBLE] Data after initial processing:")
        tprint_debug(f"   - X shape: {X.shape}")
        tprint_debug(f"   - X dtype: {X.dtype}")
        tprint_debug(f"   - y shape: {y.shape}")
        tprint_debug(f"   - y dtype: {y.dtype}")
        tprint_debug(f"   - regime_labels shape: {regime_labels.shape if regime_labels is not None else 'None'}")
        tprint_debug(f"   - regime_labels dtype: {regime_labels.dtype if regime_labels is not None else 'None'}")
        
        # Ensure all arrays have the same length
        tprint("📏 [NAS_TAS_ENSEMBLE] Ensuring consistent array lengths", color="blue")
        tprint_debug(f"🔍 [NAS_TAS_ENSEMBLE] Array lengths before adjustment:")
        tprint_debug(f"   - X length: {len(X)}")
        tprint_debug(f"   - y length: {len(y)}")
        tprint_debug(f"   - regime_labels length: {len(regime_labels) if regime_labels is not None else 'None'}")
        
        min_length = min(len(X), len(y))
        if regime_labels is not None:
            min_length = min(min_length, len(regime_labels))
        
        tprint_debug(f"   - Minimum length determined: {min_length}")
        
        # Check if truncation is needed
        if len(X) != min_length:
            tprint_debug(f"   - Truncating X from {len(X)} to {min_length}")
        if len(y) != min_length:
            tprint_debug(f"   - Truncating y from {len(y)} to {min_length}")
        if regime_labels is not None and len(regime_labels) != min_length:
            tprint_debug(f"   - Truncating regime_labels from {len(regime_labels)} to {min_length}")
        
        X = X[:min_length]
        y = y[:min_length]
        if regime_labels is not None:
            regime_labels = regime_labels[:min_length]
        
        # Log final data characteristics
        tprint_debug(f"📊 [NAS_TAS_ENSEMBLE] Final data characteristics:")
        tprint_debug(f"   - X shape: {X.shape}")
        tprint_debug(f"   - X dtype: {X.dtype}")
        tprint_debug(f"   - X memory usage: {X.nbytes / (1024**2):.2f} MB")
        tprint_debug(f"   - y shape: {y.shape}")
        tprint_debug(f"   - y dtype: {y.dtype}")
        tprint_debug(f"   - y memory usage: {y.nbytes / (1024**2):.2f} MB")
        tprint_debug(f"   - regime_labels shape: {regime_labels.shape if regime_labels is not None else 'None'}")
        tprint_debug(f"   - regime_labels dtype: {regime_labels.dtype if regime_labels is not None else 'None'}")
        if regime_labels is not None:
            tprint_debug(f"   - regime_labels memory usage: {regime_labels.nbytes / (1024**2):.2f} MB")
        
        # Log unique values in targets and regime labels
        if hasattr(y, 'shape') and y.shape[0] > 0:
            unique_y = np.unique(y)
            tprint_debug(f"   - Unique values in y: {len(unique_y)} (range: {unique_y.min()} to {unique_y.max()})")
        
        if regime_labels is not None and hasattr(regime_labels, 'shape') and regime_labels.shape[0] > 0:
            unique_regimes = np.unique(regime_labels)
            tprint_debug(f"   - Unique values in regime_labels: {len(unique_regimes)} (range: {unique_regimes.min()} to {unique_regimes.max()})")
        
        # Calculate preparation time
        prep_duration = time.perf_counter() - prep_start_time
        tprint_performance("Data preparation", prep_duration)
        
        tprint_success(f"✅ [NAS_TAS_ENSEMBLE] Data prepared - X: {X.shape}, y: {y.shape}, regime_labels: {regime_labels.shape if regime_labels is not None else 'None'}", color="green")
        return X, y, regime_labels
    
    def _train_base_models(self, X: np.ndarray, y: np.ndarray, regime_labels: np.ndarray) -> Dict[str, Any]:
        """Train base models for ensemble."""
        training_start_time = time.perf_counter()
        tprint("🏋️ [NAS_TAS_ENSEMBLE] Training base models", color="yellow")
        
        # Log training parameters
        tprint_debug(f"📊 [NAS_TAS_ENSEMBLE] Base model training parameters:")
        tprint_debug(f"   - X shape: {X.shape}")
        tprint_debug(f"   - y shape: {y.shape}")
        tprint_debug(f"   - regime_labels shape: {regime_labels.shape if regime_labels is not None else 'None'}")
        tprint_debug(f"   - Ensemble config: {self.ensemble_config}")
        
        # Log data characteristics
        tprint_debug(f"📊 [NAS_TAS_ENSEMBLE] Data characteristics for training:")
        tprint_debug(f"   - X dtype: {X.dtype}")
        tprint_debug(f"   - X memory usage: {X.nbytes / (1024**2):.2f} MB")
        tprint_debug(f"   - y dtype: {y.dtype}")
        tprint_debug(f"   - y memory usage: {y.nbytes / (1024**2):.2f} MB")
        if regime_labels is not None:
            tprint_debug(f"   - regime_labels dtype: {regime_labels.dtype}")
            tprint_debug(f"   - regime_labels memory usage: {regime_labels.nbytes / (1024**2):.2f} MB")
        
        # Log unique values
        unique_y = np.unique(y)
        tprint_debug(f"   - Unique target values: {len(unique_y)} (range: {unique_y.min()} to {unique_y.max()})")
        if regime_labels is not None:
            unique_regimes = np.unique(regime_labels)
            tprint_debug(f"   - Unique regime values: {len(unique_regimes)} (range: {unique_regimes.min()} to {unique_regimes.max()})")
        
        base_models = {}
        training_results = {}
        
        # LightGBM Classifier
        tprint("🌲 [NAS_TAS_ENSEMBLE] Training LightGBM classifier", color="blue")
        lgb_start_time = time.perf_counter()
        try:
            tprint_debug("🔧 [NAS_TAS_ENSEMBLE] Creating LightGBM classifier...")
            lgb_model = LGBMClassifier(
                n_estimators=self.ensemble_config['n_estimators'],
                max_depth=self.ensemble_config['max_depth'],
                learning_rate=self.ensemble_config['learning_rate'],
                random_state=self.ensemble_config['random_state'],
                n_jobs=self.ensemble_config['n_jobs'],
                verbose=-1
            )
            
            # Log model parameters
            tprint_structured({
                "model_type": "LightGBM",
                "parameters": {
                    "n_estimators": self.ensemble_config['n_estimators'],
                    "max_depth": self.ensemble_config['max_depth'],
                    "learning_rate": self.ensemble_config['learning_rate'],
                    "random_state": self.ensemble_config['random_state'],
                    "n_jobs": self.ensemble_config['n_jobs']
                }
            }, LogLevel.DEBUG)
            
            tprint_debug("🏋️ [NAS_TAS_ENSEMBLE] Fitting LightGBM model...")
            lgb_model.fit(X, y)
            
            lgb_duration = time.perf_counter() - lgb_start_time
            base_models['lightgbm'] = lgb_model
            training_results['lightgbm'] = {
                'success': True,
                'training_time': lgb_duration,
                'model_type': 'LGBMClassifier'
            }
            
            tprint_success("✅ [NAS_TAS_ENSEMBLE] LightGBM trained successfully")
            tprint_performance("LightGBM training", lgb_duration)
            
        except Exception as e:
            lgb_duration = time.perf_counter() - lgb_start_time
            training_results['lightgbm'] = {
                'success': False,
                'error': str(e),
                'training_time': lgb_duration,
                'model_type': 'LGBMClassifier'
            }
            tprint_error(f"❌ [NAS_TAS_ENSEMBLE] LightGBM training failed: {e}")
            tprint_debug(f"🔍 [NAS_TAS_ENSEMBLE] LightGBM error details: {traceback.format_exc()}")
            tprint_performance("Failed LightGBM training", lgb_duration)
        
        # XGBoost Classifier
        tprint("🚀 [NAS_TAS_ENSEMBLE] Training XGBoost classifier", color="blue")
        xgb_start_time = time.perf_counter()
        try:
            tprint_debug("🔧 [NAS_TAS_ENSEMBLE] Creating XGBoost classifier...")
            xgb_model = XGBClassifier(
                n_estimators=self.ensemble_config['n_estimators'],
                max_depth=self.ensemble_config['max_depth'],
                learning_rate=self.ensemble_config['learning_rate'],
                random_state=self.ensemble_config['random_state'],
                n_jobs=self.ensemble_config['n_jobs'],
                verbosity=0
            )
            
            # Log model parameters
            tprint_structured({
                "model_type": "XGBoost",
                "parameters": {
                    "n_estimators": self.ensemble_config['n_estimators'],
                    "max_depth": self.ensemble_config['max_depth'],
                    "learning_rate": self.ensemble_config['learning_rate'],
                    "random_state": self.ensemble_config['random_state'],
                    "n_jobs": self.ensemble_config['n_jobs']
                }
            }, LogLevel.DEBUG)
            
            tprint_debug("🏋️ [NAS_TAS_ENSEMBLE] Fitting XGBoost model...")
            xgb_model.fit(X, y)
            
            xgb_duration = time.perf_counter() - xgb_start_time
            base_models['xgboost'] = xgb_model
            training_results['xgboost'] = {
                'success': True,
                'training_time': xgb_duration,
                'model_type': 'XGBClassifier'
            }
            
            tprint_success("✅ [NAS_TAS_ENSEMBLE] XGBoost trained successfully")
            tprint_performance("XGBoost training", xgb_duration)
            
        except Exception as e:
            xgb_duration = time.perf_counter() - xgb_start_time
            training_results['xgboost'] = {
                'success': False,
                'error': str(e),
                'training_time': xgb_duration,
                'model_type': 'XGBClassifier'
            }
            tprint_error(f"❌ [NAS_TAS_ENSEMBLE] XGBoost training failed: {e}")
            tprint_debug(f"🔍 [NAS_TAS_ENSEMBLE] XGBoost error details: {traceback.format_exc()}")
            tprint_performance("Failed XGBoost training", xgb_duration)
        
        # Random Forest Classifier
        tprint("🌳 [NAS_TAS_ENSEMBLE] Training Random Forest classifier", color="blue")
        rf_start_time = time.perf_counter()
        try:
            tprint_debug("🔧 [NAS_TAS_ENSEMBLE] Creating Random Forest classifier...")
            rf_model = RandomForestClassifier(
                n_estimators=self.ensemble_config['n_estimators'],
                max_depth=self.ensemble_config['max_depth'],
                random_state=self.ensemble_config['random_state'],
                n_jobs=self.ensemble_config['n_jobs']
            )
            
            # Log model parameters
            tprint_structured({
                "model_type": "RandomForest",
                "parameters": {
                    "n_estimators": self.ensemble_config['n_estimators'],
                    "max_depth": self.ensemble_config['max_depth'],
                    "random_state": self.ensemble_config['random_state'],
                    "n_jobs": self.ensemble_config['n_jobs']
                }
            }, LogLevel.DEBUG)
            
            tprint_debug("🏋️ [NAS_TAS_ENSEMBLE] Fitting Random Forest model...")
            rf_model.fit(X, y)
            
            rf_duration = time.perf_counter() - rf_start_time
            base_models['random_forest'] = rf_model
            training_results['random_forest'] = {
                'success': True,
                'training_time': rf_duration,
                'model_type': 'RandomForestClassifier'
            }
            
            tprint_success("✅ [NAS_TAS_ENSEMBLE] Random Forest trained successfully")
            tprint_performance("Random Forest training", rf_duration)
            
        except Exception as e:
            rf_duration = time.perf_counter() - rf_start_time
            training_results['random_forest'] = {
                'success': False,
                'error': str(e),
                'training_time': rf_duration,
                'model_type': 'RandomForestClassifier'
            }
            tprint_error(f"❌ [NAS_TAS_ENSEMBLE] Random Forest training failed: {e}")
            tprint_debug(f"🔍 [NAS_TAS_ENSEMBLE] Random Forest error details: {traceback.format_exc()}")
            tprint_performance("Failed Random Forest training", rf_duration)
        
        # Logistic Regression
        tprint("📈 [NAS_TAS_ENSEMBLE] Training Logistic Regression", color="blue")
        lr_start_time = time.perf_counter()
        try:
            tprint_debug("🔧 [NAS_TAS_ENSEMBLE] Creating Logistic Regression classifier...")
            lr_model = LogisticRegression(
                random_state=self.ensemble_config['random_state'],
                max_iter=1000,
                n_jobs=self.ensemble_config['n_jobs']
            )
            
            # Log model parameters
            tprint_structured({
                "model_type": "LogisticRegression",
                "parameters": {
                    "random_state": self.ensemble_config['random_state'],
                    "max_iter": 1000,
                    "n_jobs": self.ensemble_config['n_jobs']
                }
            }, LogLevel.DEBUG)
            
            tprint_debug("🏋️ [NAS_TAS_ENSEMBLE] Fitting Logistic Regression model...")
            lr_model.fit(X, y)
            
            lr_duration = time.perf_counter() - lr_start_time
            base_models['logistic_regression'] = lr_model
            training_results['logistic_regression'] = {
                'success': True,
                'training_time': lr_duration,
                'model_type': 'LogisticRegression'
            }
            
            tprint_success("✅ [NAS_TAS_ENSEMBLE] Logistic Regression trained successfully")
            tprint_performance("Logistic Regression training", lr_duration)
            
        except Exception as e:
            lr_duration = time.perf_counter() - lr_start_time
            training_results['logistic_regression'] = {
                'success': False,
                'error': str(e),
                'training_time': lr_duration,
                'model_type': 'LogisticRegression'
            }
            tprint_error(f"❌ [NAS_TAS_ENSEMBLE] Logistic Regression training failed: {e}")
            tprint_debug(f"🔍 [NAS_TAS_ENSEMBLE] Logistic Regression error details: {traceback.format_exc()}")
            tprint_performance("Failed Logistic Regression training", lr_duration)
        
        # Calculate total training time
        total_training_duration = time.perf_counter() - training_start_time
        
        # Log training summary
        successful_models = [name for name, result in training_results.items() if result['success']]
        failed_models = [name for name, result in training_results.items() if not result['success']]
        
        tprint_debug(f"📊 [NAS_TAS_ENSEMBLE] Base model training summary:")
        tprint_debug(f"   - Total models attempted: {len(training_results)}")
        tprint_debug(f"   - Successful models: {len(successful_models)} ({', '.join(successful_models)})")
        tprint_debug(f"   - Failed models: {len(failed_models)} ({', '.join(failed_models)})")
        tprint_debug(f"   - Total training time: {total_training_duration:.3f}s")
        
        # Log structured training results
        tprint_structured({
            "base_model_training_summary": {
                "total_models": len(training_results),
                "successful_models": len(successful_models),
                "failed_models": len(failed_models),
                "total_training_time": total_training_duration,
                "results": training_results
            }
        }, LogLevel.INFO)
        
        tprint_success(f"✅ [NAS_TAS_ENSEMBLE] Base models training completed - {len(base_models)} models trained", color="green")
        tprint_performance("Total base model training", total_training_duration)
        
        return base_models
    
    def _train_voting_ensemble(self, X: np.ndarray, y: np.ndarray, base_models: Dict[str, Any]) -> Any:
        """Train voting ensemble."""
        voting_start_time = time.perf_counter()
        tprint("🗳️ [NAS_TAS_ENSEMBLE] Training voting ensemble", color="yellow")
        
        # Log input parameters
        tprint_debug(f"📊 [NAS_TAS_ENSEMBLE] Voting ensemble training parameters:")
        tprint_debug(f"   - X shape: {X.shape}")
        tprint_debug(f"   - y shape: {y.shape}")
        tprint_debug(f"   - Base models count: {len(base_models)}")
        tprint_debug(f"   - Base model types: {[type(model).__name__ for model in base_models.values()]}")
        
        if not base_models:
            tprint_error("❌ [NAS_TAS_ENSEMBLE] No base models available for voting ensemble")
            tprint_debug("🔍 [NAS_TAS_ENSEMBLE] Base models dictionary is empty")
            return None
        
        try:
            # Log base models details
            tprint_debug("🔍 [NAS_TAS_ENSEMBLE] Base models for voting ensemble:")
            for name, model in base_models.items():
                tprint_debug(f"   - {name}: {type(model).__name__}")
                if hasattr(model, 'get_params'):
                    try:
                        params = model.get_params()
                        tprint_debug(f"     Parameters: {list(params.keys())}")
                    except Exception as e:
                        tprint_debug(f"     Could not get parameters: {e}")
            
            # Create voting classifier
            tprint_debug("🔧 [NAS_TAS_ENSEMBLE] Creating voting classifier...")
            estimators = [(name, model) for name, model in base_models.items()]
            tprint_debug(f"   - Estimators: {[name for name, _ in estimators]}")
            
            voting_ensemble = VotingClassifier(estimators=estimators, voting='soft')
            tprint_debug(f"   - Voting strategy: soft")
            tprint_debug(f"   - Number of estimators: {len(estimators)}")
            
            # Log voting classifier parameters
            tprint_structured({
                "voting_ensemble_creation": {
                    "estimators": [name for name, _ in estimators],
                    "voting": "soft",
                    "n_estimators": len(estimators)
                }
            }, LogLevel.DEBUG)
            
            # Train the ensemble
            tprint("🏋️ [NAS_TAS_ENSEMBLE] Training voting classifier", color="blue")
            tprint_debug("🔧 [NAS_TAS_ENSEMBLE] Fitting voting ensemble...")
            voting_ensemble.fit(X, y)
            
            voting_duration = time.perf_counter() - voting_start_time
            tprint_success("✅ [NAS_TAS_ENSEMBLE] Voting ensemble trained successfully")
            tprint_performance("Voting ensemble training", voting_duration)
            
            # Log ensemble details
            tprint_debug(f"📊 [NAS_TAS_ENSEMBLE] Voting ensemble details:")
            tprint_debug(f"   - Type: {type(voting_ensemble).__name__}")
            tprint_debug(f"   - Voting: {voting_ensemble.voting}")
            tprint_debug(f"   - Estimators: {len(voting_ensemble.estimators_)}")
            tprint_debug(f"   - Training time: {voting_duration:.3f}s")
            
            return voting_ensemble
            
        except Exception as e:
            voting_duration = time.perf_counter() - voting_start_time
            tprint_error(f"❌ [NAS_TAS_ENSEMBLE] Voting ensemble training failed: {e}")
            tprint_debug(f"🔍 [NAS_TAS_ENSEMBLE] Voting ensemble error details: {traceback.format_exc()}")
            tprint_performance("Failed voting ensemble training", voting_duration)
            return None
    
    def _train_stacking_ensemble(self, X: np.ndarray, y: np.ndarray, base_models: Dict[str, Any]) -> Any:
        """Train stacking ensemble."""
        stacking_start_time = time.perf_counter()
        tprint("📚 [NAS_TAS_ENSEMBLE] Training stacking ensemble", color="yellow")
        
        # Log input parameters
        tprint_debug(f"📊 [NAS_TAS_ENSEMBLE] Stacking ensemble training parameters:")
        tprint_debug(f"   - X shape: {X.shape}")
        tprint_debug(f"   - y shape: {y.shape}")
        tprint_debug(f"   - Base models count: {len(base_models)}")
        tprint_debug(f"   - Base model types: {[type(model).__name__ for model in base_models.values()]}")
        
        if not base_models:
            tprint_error("❌ [NAS_TAS_ENSEMBLE] No base models available for stacking ensemble")
            tprint_debug("🔍 [NAS_TAS_ENSEMBLE] Base models dictionary is empty")
            return None
        
        try:
            # Log base models details
            tprint_debug("🔍 [NAS_TAS_ENSEMBLE] Base models for stacking ensemble:")
            for name, model in base_models.items():
                tprint_debug(f"   - {name}: {type(model).__name__}")
                if hasattr(model, 'get_params'):
                    try:
                        params = model.get_params()
                        tprint_debug(f"     Parameters: {list(params.keys())}")
                    except Exception as e:
                        tprint_debug(f"     Could not get parameters: {e}")
            
            # Create stacking classifier with meta-learner
            tprint_debug("🔧 [NAS_TAS_ENSEMBLE] Creating stacking classifier...")
            estimators = [(name, model) for name, model in base_models.items()]
            tprint_debug(f"   - Estimators: {[name for name, _ in estimators]}")
            
            # Create meta-learner
            tprint_debug("🔧 [NAS_TAS_ENSEMBLE] Creating meta-learner...")
            meta_learner = LogisticRegression(random_state=self.ensemble_config['random_state'])
            tprint_debug(f"   - Meta-learner: {type(meta_learner).__name__}")
            tprint_debug(f"   - Meta-learner parameters: {meta_learner.get_params()}")
            
            # Create stacking classifier
            stacking_ensemble = StackingClassifier(
                estimators=estimators,
                final_estimator=meta_learner,
                cv=5,
                n_jobs=self.ensemble_config['n_jobs']
            )
            tprint_debug(f"   - Cross-validation folds: 5")
            tprint_debug(f"   - Number of estimators: {len(estimators)}")
            tprint_debug(f"   - n_jobs: {self.ensemble_config['n_jobs']}")
            
            # Log stacking classifier parameters
            tprint_structured({
                "stacking_ensemble_creation": {
                    "estimators": [name for name, _ in estimators],
                    "meta_learner": type(meta_learner).__name__,
                    "cv": 5,
                    "n_jobs": self.ensemble_config['n_jobs'],
                    "n_estimators": len(estimators)
                }
            }, LogLevel.DEBUG)
            
            # Train the ensemble
            tprint("🏋️ [NAS_TAS_ENSEMBLE] Training stacking classifier", color="blue")
            tprint_debug("🔧 [NAS_TAS_ENSEMBLE] Fitting stacking ensemble...")
            tprint_debug("   - This may take longer due to cross-validation...")
            stacking_ensemble.fit(X, y)
            
            stacking_duration = time.perf_counter() - stacking_start_time
            tprint_success("✅ [NAS_TAS_ENSEMBLE] Stacking ensemble trained successfully")
            tprint_performance("Stacking ensemble training", stacking_duration)
            
            # Log ensemble details
            tprint_debug(f"📊 [NAS_TAS_ENSEMBLE] Stacking ensemble details:")
            tprint_debug(f"   - Type: {type(stacking_ensemble).__name__}")
            tprint_debug(f"   - Estimators: {len(stacking_ensemble.estimators_)}")
            tprint_debug(f"   - Final estimator: {type(stacking_ensemble.final_estimator_).__name__}")
            tprint_debug(f"   - Training time: {stacking_duration:.3f}s")
            
            # Log meta-learner details
            if hasattr(stacking_ensemble, 'final_estimator_'):
                final_estimator = stacking_ensemble.final_estimator_
                tprint_debug(f"   - Final estimator type: {type(final_estimator).__name__}")
                if hasattr(final_estimator, 'coef_'):
                    tprint_debug(f"   - Final estimator coefficients shape: {final_estimator.coef_.shape}")
                if hasattr(final_estimator, 'intercept_'):
                    tprint_debug(f"   - Final estimator intercept: {final_estimator.intercept_}")
            
            return stacking_ensemble
            
        except Exception as e:
            stacking_duration = time.perf_counter() - stacking_start_time
            tprint_error(f"❌ [NAS_TAS_ENSEMBLE] Stacking ensemble training failed: {e}")
            tprint_debug(f"🔍 [NAS_TAS_ENSEMBLE] Stacking ensemble error details: {traceback.format_exc()}")
            tprint_performance("Failed stacking ensemble training", stacking_duration)
            return None
    
    def _evaluate_ensembles(self, X: np.ndarray, y: np.ndarray, voting_ensemble: Any, stacking_ensemble: Any) -> Dict[str, Any]:
        """Evaluate ensemble performance."""
        eval_start_time = time.perf_counter()
        tprint("📊 [NAS_TAS_ENSEMBLE] Evaluating ensemble performance", color="yellow")
        
        # Log evaluation parameters
        tprint_debug(f"📊 [NAS_TAS_ENSEMBLE] Evaluation parameters:")
        tprint_debug(f"   - X shape: {X.shape}")
        tprint_debug(f"   - y shape: {y.shape}")
        tprint_debug(f"   - voting_ensemble available: {voting_ensemble is not None}")
        tprint_debug(f"   - stacking_ensemble available: {stacking_ensemble is not None}")
        
        # Log data characteristics for evaluation
        tprint_debug(f"📊 [NAS_TAS_ENSEMBLE] Data characteristics for evaluation:")
        tprint_debug(f"   - X dtype: {X.dtype}")
        tprint_debug(f"   - y dtype: {y.dtype}")
        unique_y = np.unique(y)
        tprint_debug(f"   - Unique target values: {len(unique_y)} (range: {unique_y.min()} to {unique_y.max()})")
        
        metrics = {}
        evaluation_results = {}
        
        # Evaluate voting ensemble
        if voting_ensemble is not None:
            tprint("🗳️ [NAS_TAS_ENSEMBLE] Evaluating voting ensemble", color="blue")
            voting_eval_start = time.perf_counter()
            try:
                tprint_debug("🔧 [NAS_TAS_ENSEMBLE] Making predictions with voting ensemble...")
                y_pred_voting = voting_ensemble.predict(X)
                tprint_debug(f"   - Predictions shape: {y_pred_voting.shape}")
                tprint_debug(f"   - Predictions dtype: {y_pred_voting.dtype}")
                
                tprint_debug("🔧 [NAS_TAS_ENSEMBLE] Calculating accuracy...")
                voting_accuracy = accuracy_score(y, y_pred_voting)
                tprint_debug(f"   - Accuracy: {voting_accuracy:.6f}")
                
                tprint_debug("🔧 [NAS_TAS_ENSEMBLE] Generating classification report...")
                voting_classification_report = classification_report(y, y_pred_voting, output_dict=True)
                tprint_debug(f"   - Classification report generated")
                
                # Log detailed metrics
                tprint_debug(f"📊 [NAS_TAS_ENSEMBLE] Voting ensemble detailed metrics:")
                tprint_debug(f"   - Overall accuracy: {voting_accuracy:.6f}")
                if 'macro avg' in voting_classification_report:
                    tprint_debug(f"   - Macro avg precision: {voting_classification_report['macro avg']['precision']:.6f}")
                    tprint_debug(f"   - Macro avg recall: {voting_classification_report['macro avg']['recall']:.6f}")
                    tprint_debug(f"   - Macro avg f1-score: {voting_classification_report['macro avg']['f1-score']:.6f}")
                if 'weighted avg' in voting_classification_report:
                    tprint_debug(f"   - Weighted avg precision: {voting_classification_report['weighted avg']['precision']:.6f}")
                    tprint_debug(f"   - Weighted avg recall: {voting_classification_report['weighted avg']['recall']:.6f}")
                    tprint_debug(f"   - Weighted avg f1-score: {voting_classification_report['weighted avg']['f1-score']:.6f}")
                
                voting_eval_duration = time.perf_counter() - voting_eval_start
                metrics['voting_ensemble'] = {
                    'accuracy': voting_accuracy,
                    'classification_report': voting_classification_report,
                    'evaluation_time': voting_eval_duration
                }
                evaluation_results['voting_ensemble'] = {
                    'success': True,
                    'accuracy': voting_accuracy,
                    'evaluation_time': voting_eval_duration
                }
                
                tprint_success(f"✅ [NAS_TAS_ENSEMBLE] Voting ensemble accuracy: {voting_accuracy:.4f}")
                tprint_performance("Voting ensemble evaluation", voting_eval_duration)
                
            except Exception as e:
                voting_eval_duration = time.perf_counter() - voting_eval_start
                tprint_error(f"❌ [NAS_TAS_ENSEMBLE] Voting ensemble evaluation failed: {e}")
                tprint_debug(f"🔍 [NAS_TAS_ENSEMBLE] Voting ensemble evaluation error details: {traceback.format_exc()}")
                tprint_performance("Failed voting ensemble evaluation", voting_eval_duration)
                
                metrics['voting_ensemble'] = {'error': str(e), 'evaluation_time': voting_eval_duration}
                evaluation_results['voting_ensemble'] = {
                    'success': False,
                    'error': str(e),
                    'evaluation_time': voting_eval_duration
                }
        else:
            tprint_warning("⚠️ [NAS_TAS_ENSEMBLE] Voting ensemble is None - skipping evaluation")
            evaluation_results['voting_ensemble'] = {'success': False, 'error': 'Voting ensemble is None'}
        
        # Evaluate stacking ensemble
        if stacking_ensemble is not None:
            tprint("📚 [NAS_TAS_ENSEMBLE] Evaluating stacking ensemble", color="blue")
            stacking_eval_start = time.perf_counter()
            try:
                tprint_debug("🔧 [NAS_TAS_ENSEMBLE] Making predictions with stacking ensemble...")
                y_pred_stacking = stacking_ensemble.predict(X)
                tprint_debug(f"   - Predictions shape: {y_pred_stacking.shape}")
                tprint_debug(f"   - Predictions dtype: {y_pred_stacking.dtype}")
                
                tprint_debug("🔧 [NAS_TAS_ENSEMBLE] Calculating accuracy...")
                stacking_accuracy = accuracy_score(y, y_pred_stacking)
                tprint_debug(f"   - Accuracy: {stacking_accuracy:.6f}")
                
                tprint_debug("🔧 [NAS_TAS_ENSEMBLE] Generating classification report...")
                stacking_classification_report = classification_report(y, y_pred_stacking, output_dict=True)
                tprint_debug(f"   - Classification report generated")
                
                # Log detailed metrics
                tprint_debug(f"📊 [NAS_TAS_ENSEMBLE] Stacking ensemble detailed metrics:")
                tprint_debug(f"   - Overall accuracy: {stacking_accuracy:.6f}")
                if 'macro avg' in stacking_classification_report:
                    tprint_debug(f"   - Macro avg precision: {stacking_classification_report['macro avg']['precision']:.6f}")
                    tprint_debug(f"   - Macro avg recall: {stacking_classification_report['macro avg']['recall']:.6f}")
                    tprint_debug(f"   - Macro avg f1-score: {stacking_classification_report['macro avg']['f1-score']:.6f}")
                if 'weighted avg' in stacking_classification_report:
                    tprint_debug(f"   - Weighted avg precision: {stacking_classification_report['weighted avg']['precision']:.6f}")
                    tprint_debug(f"   - Weighted avg recall: {stacking_classification_report['weighted avg']['recall']:.6f}")
                    tprint_debug(f"   - Weighted avg f1-score: {stacking_classification_report['weighted avg']['f1-score']:.6f}")
                
                stacking_eval_duration = time.perf_counter() - stacking_eval_start
                metrics['stacking_ensemble'] = {
                    'accuracy': stacking_accuracy,
                    'classification_report': stacking_classification_report,
                    'evaluation_time': stacking_eval_duration
                }
                evaluation_results['stacking_ensemble'] = {
                    'success': True,
                    'accuracy': stacking_accuracy,
                    'evaluation_time': stacking_eval_duration
                }
                
                tprint_success(f"✅ [NAS_TAS_ENSEMBLE] Stacking ensemble accuracy: {stacking_accuracy:.4f}")
                tprint_performance("Stacking ensemble evaluation", stacking_eval_duration)
                
            except Exception as e:
                stacking_eval_duration = time.perf_counter() - stacking_eval_start
                tprint_error(f"❌ [NAS_TAS_ENSEMBLE] Stacking ensemble evaluation failed: {e}")
                tprint_debug(f"🔍 [NAS_TAS_ENSEMBLE] Stacking ensemble evaluation error details: {traceback.format_exc()}")
                tprint_performance("Failed stacking ensemble evaluation", stacking_eval_duration)
                
                metrics['stacking_ensemble'] = {'error': str(e), 'evaluation_time': stacking_eval_duration}
                evaluation_results['stacking_ensemble'] = {
                    'success': False,
                    'error': str(e),
                    'evaluation_time': stacking_eval_duration
                }
        else:
            tprint_warning("⚠️ [NAS_TAS_ENSEMBLE] Stacking ensemble is None - skipping evaluation")
            evaluation_results['stacking_ensemble'] = {'success': False, 'error': 'Stacking ensemble is None'}
        
        # Calculate total evaluation time
        total_eval_duration = time.perf_counter() - eval_start_time
        
        # Log evaluation summary
        successful_evaluations = [name for name, result in evaluation_results.items() if result['success']]
        failed_evaluations = [name for name, result in evaluation_results.items() if not result['success']]
        
        tprint_debug(f"📊 [NAS_TAS_ENSEMBLE] Evaluation summary:")
        tprint_debug(f"   - Total evaluations attempted: {len(evaluation_results)}")
        tprint_debug(f"   - Successful evaluations: {len(successful_evaluations)} ({', '.join(successful_evaluations)})")
        tprint_debug(f"   - Failed evaluations: {len(failed_evaluations)} ({', '.join(failed_evaluations)})")
        tprint_debug(f"   - Total evaluation time: {total_eval_duration:.3f}s")
        
        # Log structured evaluation results
        tprint_structured({
            "ensemble_evaluation_summary": {
                "total_evaluations": len(evaluation_results),
                "successful_evaluations": len(successful_evaluations),
                "failed_evaluations": len(failed_evaluations),
                "total_evaluation_time": total_eval_duration,
                "results": evaluation_results
            }
        }, LogLevel.INFO)
        
        tprint_success("✅ [NAS_TAS_ENSEMBLE] Ensemble evaluation completed")
        tprint_performance("Total ensemble evaluation", total_eval_duration)
        
        return metrics
