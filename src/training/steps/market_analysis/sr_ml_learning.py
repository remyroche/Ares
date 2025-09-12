"""SR ML Learning Stage: ML-based learning for SR clusters with comprehensive model training."""

import asyncio
import sys
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable
try:
    from collections.abc import Iterable
except ImportError:
    from typing import Iterable
import time
import json
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, precision_recall_curve,
    roc_curve, log_loss, matthews_corrcoef, cohen_kappa_score
)
import joblib
import traceback
import logging
import random
import gc

# Core imports
try:
    from src.training.base_step import BaseStep
except ImportError:
    # Fallback BaseStep class
    class BaseStep:
        def __init__(self, config):
            self.config = config
        
        async def execute(self, data):
            pass
        
        def validate_config(self):
            pass
        
        def get_status(self):
            return {}

from src.utils.logger import system_logger

# Initialize logger early to avoid usage before definition
logger = system_logger.getChild('SRMLLearning')

# Required utility modules - Simplified imports
from src.utils.common_operations import (
    safe_json_load, safe_json_dump,
    ensure_directory, create_fallback_logger, create_fallback_decorator,
    get_current_datetime, format_datetime, create_empty_dataframe,
    safe_fillna, get_logger, setup_basic_logging,
    validate_dataframe, optimize_dataframe_dtypes,
    safe_log_metric, safe_log_params, safe_log_artifact
)

# Core decorators and errors
from src.core.decorators import handles_errors, error_boundary, converts_errors
from src.core.errors import (
    AppError, ValidationError, DataIntegrityError, 
    NotFoundError, BusinessRuleError
)

# Pipeline standards and utilities
from src.utils.pipeline_standards import PipelineStandards
from src.utils.monitoring_utils import (
    global_monitor, function_tracker, logging_patterns
)
from src.utils.comprehensive_function_logger import (
    log_step_functions, log_important_calls, log_all_calls, 
    log_internal_call, log_step_progress, log_data_operation
)

# ML Common utilities
try:
    from src.utils.ml_common import (
        ModelExplainabilityManager,
        ModelExplanationResult,
        MemoryEfficientTraining,
        HyperparameterOptimization,
        ModelRegistry,
        ParallelProcessingCoordinator
    )
    ML_COMMON_AVAILABLE = True
    logger.info('✅ ML Common utilities available')
except ImportError as e:
    ML_COMMON_AVAILABLE = False
    logger.warning(f'⚠️ ML Common utilities not available: {e}')

# M1 Optimization Utilities - Integrated via Common Operations
try:
    from src.utils.common_operations import (
        integrate_with_m1_optimizers, get_m1_gpu_manager, get_m1_memory_optimizer,
        get_m1_cpu_optimizer, cleanup_m1_optimizers, memory_checkpoint, gpu_context,
        optimize_memory, get_memory_usage
    )

    # Initialize M1 integration through common operations
    m1_integration_result = integrate_with_m1_optimizers()
    M1_GPU_AVAILABLE = m1_integration_result.get('gpu_manager', False)
    M1_MEMORY_AVAILABLE = m1_integration_result.get('memory_optimizer', False)
    M1_CPU_AVAILABLE = m1_integration_result.get('cpu_optimizer', False)
    M1_BATCH_AVAILABLE = M1_CPU_AVAILABLE  # Batch processor available if CPU optimizer is

    integration_status = m1_integration_result.get('integration_status', 'unknown')
    if integration_status == 'success':
        logger.info("✅ Complete M1 utilities integration successful")
    elif integration_status == 'partial':
        logger.info("⚠️ Partial M1 utilities integration - some components available")
    else:
        logger.warning("❌ M1 utilities integration failed")

except ImportError as e:
    M1_GPU_AVAILABLE = False
    M1_MEMORY_AVAILABLE = False
    M1_CPU_AVAILABLE = False
    M1_BATCH_AVAILABLE = False
    logger.warning(f"M1 utilities integration not available: {e}")
except Exception as e:
    M1_GPU_AVAILABLE = False
    M1_MEMORY_AVAILABLE = False
    M1_CPU_AVAILABLE = False
    M1_BATCH_AVAILABLE = False
    logger.error(f"Unexpected error in M1 utilities integration: {e}")

# Utility functions for memory management and validation
def get_memory_usage():
    try:
        import psutil
        return psutil.Process().memory_info().rss
    except ImportError:
        return 0

def format_bytes(bytes_val):
    return f"{bytes_val / 1024 / 1024:.1f} MB"

def memory_checkpoint(name):
    pass

def optimize_dataframe_dtypes(df):
    return df

def validate_dataframe(df):
    return True

# Import standardized math validation utilities
from src.utils.math_validation import validate_finite, safe_divide


class SRMLLearningStep(BaseStep):
    """SR ML Learning Stage: ML-based learning for SR clusters with comprehensive model training."""

    @log_important_calls
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize SR ML learning step."""
        super().__init__(config)
        self.logger = system_logger.getChild('SRMLLearningStep')
        self.standards = PipelineStandards(self.logger)
        self.sr_optimization_config = config.get('sr_optimization', {
            'min_touches': 2, 
            'tolerance_pct': 0.5, 
            'lookback_periods': 100
        })

        # Adjust configuration for LIGHT mode
        training_mode = os.environ.get('LIGHT_TRAINING_MODE', '')
        if training_mode == '1' or config.get('training_mode') == 'light':
            self.sr_optimization_config['lookback_periods'] = 10
            self.logger.info('💡 LIGHT mode: Adjusted lookback_periods to 10 (was 100)')
        
        # ML configuration
        self.ml_config = config.get('ml_learning', {
            'chunk_size': 10000,
            'test_size': 0.2,
            'random_state': 42,
            'n_jobs': -1
        })
        
        # Initialize ML Model Configurations
        self.ml_model_configs = {
            'RandomForestClassifier': {
                'class': RandomForestClassifier,
                'hyperparameters': {
                    'n_estimators': {'type': 'int', 'low': 50, 'high': 300, 'step': 10},
                    'max_depth': {'type': 'int', 'low': 3, 'high': 20},
                    'min_samples_split': {'type': 'int', 'low': 2, 'high': 20},
                    'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 10},
                    'max_features': {'type': 'categorical', 'choices': ['sqrt', 'log2', None]},
                    'bootstrap': {'type': 'categorical', 'choices': [True, False]},
                    'criterion': {'type': 'categorical', 'choices': ['gini', 'entropy']}
                },
                'default_params': {
                    'n_estimators': 100,
                    'max_depth': None,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1,
                    'max_features': 'sqrt',
                    'bootstrap': True,
                    'criterion': 'gini',
                    'random_state': 42,
                    'n_jobs': -1
                }
            },
            'LogisticRegression': {
                'class': LogisticRegression,
                'hyperparameters': {
                    'C': {'type': 'float', 'low': 1e-3, 'high': 10.0, 'log': True},
                    'penalty': {'type': 'categorical', 'choices': ['l2', 'l1']},
                    'solver': {'type': 'categorical', 'choices': ['lbfgs', 'liblinear']},
                    'max_iter': {'type': 'int', 'low': 1000, 'high': 5000, 'step': 500}
                },
                'default_params': {
                    'C': 1.0,
                    'penalty': 'l2',
                    'solver': 'lbfgs',
                    'max_iter': 1000,
                    'random_state': 42,
                    'n_jobs': -1
                }
            },
            'HistGradientBoostingClassifier': {
                'class': HistGradientBoostingClassifier,
                'hyperparameters': {
                    'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
                    'max_iter': {'type': 'int', 'low': 50, 'high': 300, 'step': 10},
                    'max_depth': {'type': 'int', 'low': 3, 'high': 15},
                    'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 20},
                    'l2_regularization': {'type': 'float', 'low': 0.0, 'high': 1.0}
                },
                'default_params': {
                    'learning_rate': 0.1,
                    'max_iter': 100,
                    'max_depth': None,
                    'min_samples_leaf': 20,
                    'l2_regularization': 0.0,
                    'random_state': 42
                }
            }
        }
        
        # Initialize automatic memory management
        try:
            from src.utils.hardware.memory_optimization import get_memory_manager, MemoryContext as memory_context
            self.memory_manager = get_memory_manager()
            self.memory_manager.start_monitoring()
            self.logger.info("🧠 Memory management initialized")
        except Exception as e:
            self.logger.warning(f"Memory manager initialization failed: {e}")
            # Fallback memory manager
            class FallbackMemoryManager:
                def start_monitoring(self):
                    pass
                def stop_monitoring(self):
                    pass
            self.memory_manager = FallbackMemoryManager()

        # Initialize ML Common components if available
        if ML_COMMON_AVAILABLE:
            try:
                self.memory_optimizer = MemoryEfficientTraining()
                self.hyperparameter_optimizer = HyperparameterOptimization()
                self.model_registry = ModelRegistry()
                self.parallel_coordinator = ParallelProcessingCoordinator()
                self.logger.info("✅ ML Common components initialized")
            except Exception as e:
                self.logger.warning(f"ML Common components initialization failed: {e}")
                self.memory_optimizer = None
                self.hyperparameter_optimizer = None
                self.model_registry = None
                self.parallel_coordinator = None
        else:
            self.memory_optimizer = None
            self.hyperparameter_optimizer = None
            self.model_registry = None
            self.parallel_coordinator = None
        
        # Initialize explainability manager
        if ML_COMMON_AVAILABLE:
            try:
                explainability_config = {
                    'enable_auto_explanations': True,
                    'enable_explanation_caching': True,
                    'auto_explain_on_training': True,
                    'explanations': {
                        'enable_shap': True,
                        'enable_lime': True,
                        'shap_sample_size': 50,
                        'lime_sample_size': 10
                    }
                }
                self.explainability_manager = ModelExplainabilityManager(
                    config=explainability_config,
                    model_registry=self.model_registry
                )
                self.logger.info("✅ Model explainability manager initialized")
            except Exception as e:
                self.logger.warning(f"Model explainability manager initialization failed: {e}")
                self.explainability_manager = None
        else:
            self.explainability_manager = None

    async def execute(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the SR ML learning stage."""
        self.logger.info('🤖 Starting SR ML Learning Stage execution')
        start_time = time.time()

        try:
            # Validate required data exists
            required_keys = ['dataframe', 'sr_levels']
            missing_keys = [key for key in required_keys if key not in pipeline_state]
            if missing_keys:
                raise ValueError(f"Missing required pipeline state keys: {missing_keys}")

            # Get features data
            features_data = pipeline_state.get('dataframe')
            if features_data is None or features_data.empty:
                raise ValueError("No features data found in pipeline state or dataframe is empty")

            # Get SR levels from pipeline state
            sr_levels = pipeline_state.get('sr_levels')
            if sr_levels is None or len(sr_levels) == 0:
                raise ValueError("No SR levels found in pipeline state")

            # Get clustered levels from previous stage or pipeline state
            clustered_levels = pipeline_state.get('clustered_levels')
            if clustered_levels is None or len(clustered_levels) == 0:
                self.logger.warning("No clustered levels found, using SR levels directly")
                clustered_levels = sr_levels

            # Validate data quality
            self._validate_data_quality(features_data, sr_levels, clustered_levels)

            self.logger.info(f'📊 Clustered levels loaded: {len(clustered_levels)} clusters')
            self.logger.info(f'📊 Features data loaded: {features_data.shape[0]:,} rows, {features_data.shape[1]} columns')

            # Train ML models
            ml_results = await self._train_ml_models(features_data, clustered_levels)
            
            execution_time = time.time() - start_time
            self.logger.info(f'✅ SR ML Learning completed in {execution_time:.2f} seconds')

            return {
                'success': True,
                'ml_results': ml_results,
                'execution_time': execution_time,
                'stage': 'sr_ml_learning'
            }

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f'❌ SR ML Learning failed: {e}')
            return {
                'success': False,
                'error': str(e),
                'execution_time': execution_time,
                'stage': 'sr_ml_learning'
            }

    async def _train_ml_models(self, features_data: pd.DataFrame, clustered_levels: Dict[str, Any]) -> Dict[str, Any]:
        """Train ML models for SR cluster prediction."""
        self.logger.info('🤖 ===== STARTING ML MODEL TRAINING =====')
        training_start_time = time.time()

        try:
            # Extract clusters
            clusters = clustered_levels.get('clusters', [])
            if not clusters:
                self.logger.error('❌ No clusters provided for ML training.')
                raise ValueError("No clusters provided for ML training.")

            self.logger.info(f'📊 Training on {len(clusters)} clusters')

            # Prepare features and labels
            self.logger.info('🔧 Preparing features and labels for ML training...')
            X, y, feature_names = self._prepare_ml_features(features_data, clusters)
            
            if X is None or len(X) == 0:
                self.logger.error('❌ No valid features prepared for ML training.')
                raise ValueError("No valid features prepared for ML training.")

            self.logger.info(f'📊 Prepared features: {X.shape[0]} samples, {X.shape[1]} features')
            self.logger.info(f'📊 Feature names: {feature_names[:10]}...' if len(feature_names) > 10 else f'📊 Feature names: {feature_names}')

            # Split data
            self.logger.info('🔧 Splitting data for training and testing...')
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=self.ml_config.get('test_size', 0.2),
                random_state=self.ml_config.get('random_state', 42),
                stratify=y if len(np.unique(y)) > 1 else None
            )

            self.logger.info(f'📊 Data split: {len(X_train)} training samples, {len(X_test)} test samples')

            # Train models
            self.logger.info('🤖 Training ML models...')
            model_results = {}
            
            for model_name, model_config in self.ml_model_configs.items():
                self.logger.info(f'🔄 Training {model_name}...')
                model_start_time = time.time()
                
                try:
                    # Create model with default parameters
                    model_class = model_config['class']
                    default_params = model_config['default_params'].copy()
                    
                    # Adjust parameters for light mode
                    if os.environ.get('LIGHT_TRAINING_MODE') == '1':
                        if 'n_estimators' in default_params:
                            default_params['n_estimators'] = min(default_params['n_estimators'], 50)
                        if 'max_iter' in default_params:
                            default_params['max_iter'] = min(default_params['max_iter'], 100)
                    
                    model = model_class(**default_params)
                    
                    # Train model
                    model.fit(X_train, y_train)
                    
                    # Make predictions
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
                    
                    # Calculate metrics
                    metrics = self._calculate_model_metrics(y_test, y_pred, y_pred_proba)
                    
                    # Feature importance
                    feature_importance = None
                    if hasattr(model, 'feature_importances_'):
                        feature_importance = model.feature_importances_
                    elif hasattr(model, 'coef_'):
                        feature_importance = np.abs(model.coef_[0]) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
                    
                    # Generate model explanations if explainability manager is available
                    model_explanation = None
                    if self.explainability_manager is not None:
                        try:
                            self.logger.info(f'🧠 Generating explanations for {model_name}...')
                            explanation_start_time = time.time()
                            
                            # Use smaller sample for explanations to avoid memory issues
                            explanation_sample_size = min(50, len(X_test))
                            if explanation_sample_size < len(X_test):
                                test_indices = np.random.choice(len(X_test), explanation_sample_size, replace=False)
                                X_test_sample = X_test[test_indices]
                            else:
                                X_test_sample = X_test
                            
                            model_explanation = self.explainability_manager.explain_model(
                                model=model,
                                X_train=X_train[:100],  # Use small sample for background
                                X_test=X_test_sample,
                                model_id=f"sr_ml_{model_name}",
                                model_type=type(model).__name__,
                                feature_names=feature_names,
                                cache_key=f"sr_ml_{model_name}_{hash(X_train.tobytes())}"
                            )
                            
                            explanation_time = time.time() - explanation_start_time
                            self.logger.info(f'✅ Model explanations generated in {explanation_time:.2f} seconds')
                            self.logger.info(f'   • Explanation confidence: {model_explanation.explanation_confidence:.3f}')
                            
                        except Exception as explanation_error:
                            self.logger.warning(f'⚠️ Model explanations failed for {model_name}: {explanation_error}')
                            model_explanation = None
                    
                    model_time = time.time() - model_start_time
                    
                    model_results[model_name] = {
                        'model': model,
                        'metrics': metrics,
                        'feature_importance': feature_importance,
                        'model_explanation': model_explanation,
                        'training_time': model_time,
                        'feature_names': feature_names,
                        'model_params': default_params
                    }
                    
                    self.logger.info(f'✅ {model_name} trained in {model_time:.2f} seconds')
                    self.logger.info(f'   • Accuracy: {metrics.get("accuracy", 0):.3f}')
                    self.logger.info(f'   • F1 Score: {metrics.get("f1_score", 0):.3f}')
                    
                except Exception as model_error:
                    model_time = time.time() - model_start_time
                    self.logger.error(f'❌ {model_name} training failed after {model_time:.2f} seconds: {model_error}')
                    model_results[model_name] = {
                        'error': str(model_error),
                        'training_time': model_time
                    }

            # Find best model
            best_model_name = self._find_best_model(model_results)
            
            total_training_time = time.time() - training_start_time
            
            # Log explainability summary
            explanations_generated = sum(1 for result in model_results.values() 
                                       if result.get('model_explanation') is not None)
            total_models = len(model_results)
            
            self.logger.info('🤖 ===== ML MODEL TRAINING COMPLETED =====')
            self.logger.info(f'✅ Total training time: {total_training_time:.2f} seconds')
            self.logger.info(f'🏆 Best model: {best_model_name}')
            self.logger.info(f'🧠 Model explanations: {explanations_generated}/{total_models} models explained')
            
            if self.explainability_manager is not None:
                cache_stats = self.explainability_manager.get_cache_stats()
                self.logger.info(f'📊 Explanation cache: {cache_stats["cache_size"]} entries, hit rate: {cache_stats["hit_rate"]:.3f}')
            
            return {
                'model_results': model_results,
                'best_model': best_model_name,
                'training_time': total_training_time,
                'feature_names': feature_names,
                'data_shape': X.shape,
                'train_test_split': {
                    'train_samples': len(X_train),
                    'test_samples': len(X_test),
                    'test_size': self.ml_config.get('test_size', 0.2)
                }
            }

        except Exception as e:
            training_time = time.time() - training_start_time
            self.logger.error(f'❌ ML training failed after {training_time:.2f} seconds: {e}')
            self.logger.error(f'❌ Error details: {traceback.format_exc()}')
            raise

    def _prepare_ml_features(self, features_data: pd.DataFrame, clusters: List[Any]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare features and labels for ML training."""
        self.logger.info('🔧 Preparing ML features from clustered SR levels...')
        
        try:
            # Extract features from clusters
            cluster_features = []
            cluster_labels = []
            
            for i, cluster in enumerate(clusters):
                if isinstance(cluster, dict):
                    levels = cluster.get('levels', [])
                    center_price = cluster.get('center_price', 0)
                    cluster_strength = cluster.get('cluster_strength', 0.5)
                    cluster_type = cluster.get('cluster_type', 'support')
                else:
                    # Handle object-based clusters
                    levels = getattr(cluster, 'levels', [])
                    center_price = getattr(cluster, 'center_price', 0)
                    cluster_strength = getattr(cluster, 'cluster_strength', 0.5)
                    cluster_type = getattr(cluster, 'cluster_type', 'support')
                
                # Create feature vector for this cluster
                cluster_feature = [
                    center_price,
                    cluster_strength,
                    len(levels),
                    float(cluster_type.lower() == 'support'),  # 1 for support, 0 for resistance
                ]
                
                # Add level-specific features
                if levels:
                    prices = [level.get('price', 0) if isinstance(level, dict) else getattr(level, 'price', 0) for level in levels]
                    strengths = [level.get('strength', 0.5) if isinstance(level, dict) else getattr(level, 'strength', 0.5) for level in levels]
                    
                    cluster_feature.extend([
                        np.mean(prices) if prices else 0,
                        np.std(prices) if len(prices) > 1 else 0,
                        np.mean(strengths) if strengths else 0,
                        np.std(strengths) if len(strengths) > 1 else 0,
                        max(prices) - min(prices) if len(prices) > 1 else 0
                    ])
                else:
                    cluster_feature.extend([0, 0, 0, 0, 0])
                
                cluster_features.append(cluster_feature)
                
                # Create label (binary classification: support vs resistance)
                cluster_labels.append(1 if cluster_type.lower() == 'support' else 0)
            
            # Convert to numpy arrays
            X = np.array(cluster_features)
            y = np.array(cluster_labels)
            
            # Create feature names
            feature_names = [
                'center_price', 'cluster_strength', 'level_count', 'is_support',
                'mean_price', 'price_std', 'mean_strength', 'strength_std', 'price_range'
            ]
            
            self.logger.info(f'✅ Prepared {len(cluster_features)} cluster features with {len(feature_names)} features each')
            
            return X, y, feature_names
            
        except Exception as e:
            self.logger.error(f'❌ Feature preparation failed: {e}')
            return None, None, []

    def _calculate_model_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate comprehensive model metrics."""
        try:
            metrics = {}
            
            # Basic classification metrics
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            # Additional metrics
            try:
                metrics['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)
            except:
                metrics['matthews_corrcoef'] = 0.0
            
            try:
                metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
            except:
                metrics['cohen_kappa'] = 0.0
            
            # Probability-based metrics
            if y_pred_proba is not None and len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
                try:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
                except:
                    metrics['roc_auc'] = 0.0
                
                try:
                    metrics['log_loss'] = log_loss(y_true, y_pred_proba)
                except:
                    metrics['log_loss'] = 0.0
            else:
                metrics['roc_auc'] = 0.0
                metrics['log_loss'] = 0.0
            
            return metrics
            
        except Exception as e:
            self.logger.warning(f'⚠️ Error calculating metrics: {e}')
            return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}

    def _validate_data_quality(self, data: pd.DataFrame, sr_levels: List[Any], clustered_levels: List[Any]) -> None:
        """Validate data quality for ML training."""
        self.logger.info('🔍 Validating data quality for ML training...')
        
        # Validate dataframe
        if data is None or data.empty:
            raise ValueError("Dataframe is None or empty")
        
        if data.shape[0] < 10:
            raise ValueError(f"Insufficient data for ML training: {data.shape[0]} rows (minimum 10 required)")
        
        # Validate SR levels
        if not sr_levels or len(sr_levels) == 0:
            raise ValueError("No SR levels provided for ML training")
        
        # Validate clustered levels
        if not clustered_levels or len(clustered_levels) == 0:
            raise ValueError("No clustered levels provided for ML training")
        
        # Check for sufficient clusters
        if len(clustered_levels) < 5:
            self.logger.warning(f"Low number of clusters: {len(clustered_levels)} (recommended minimum 5)")
        
        # Validate cluster quality
        valid_clusters = 0
        for cluster in clustered_levels:
            if self._is_high_quality_cluster(cluster):
                valid_clusters += 1
        
        if valid_clusters < 3:
            raise ValueError(f"Insufficient high-quality clusters: {valid_clusters} (minimum 3 required)")
        
        self.logger.info(f'✅ Data quality validation passed: {valid_clusters}/{len(clustered_levels)} high-quality clusters')

    def _is_high_quality_cluster(self, cluster: Any) -> bool:
        """Check if cluster meets quality standards for ML training."""
        try:
            if isinstance(cluster, dict):
                levels = cluster.get('levels', [])
                center_price = cluster.get('center_price', 0)
                cluster_strength = cluster.get('cluster_strength', 0)
            else:
                levels = getattr(cluster, 'levels', [])
                center_price = getattr(cluster, 'center_price', 0)
                cluster_strength = getattr(cluster, 'cluster_strength', 0)
            
            # Check basic requirements
            if not levels or len(levels) < 2:
                return False
            
            if center_price <= 0:
                return False
            
            if cluster_strength < 0 or cluster_strength > 1:
                return False
            
            # Check level quality within cluster
            valid_levels = 0
            for level in levels:
                if isinstance(level, dict):
                    price = level.get('price', 0)
                    strength = level.get('strength', 0)
                else:
                    price = getattr(level, 'price', 0)
                    strength = getattr(level, 'strength', 0)
                
                if price > 0 and 0 <= strength <= 1:
                    valid_levels += 1
            
            # At least 50% of levels should be valid
            return valid_levels >= len(levels) * 0.5
            
        except Exception as e:
            self.logger.warning(f"Error validating cluster quality: {e}")
            return False

    def _find_best_model(self, model_results: Dict[str, Any]) -> str:
        """Find the best performing model based on F1 score."""
        best_model = None
        best_f1_score = -1
        
        for model_name, results in model_results.items():
            if 'error' not in results and 'metrics' in results:
                f1_score = results['metrics'].get('f1_score', 0)
                if f1_score > best_f1_score:
                    best_f1_score = f1_score
                    best_model = model_name
        
        return best_model if best_model else list(model_results.keys())[0] if model_results else 'None'