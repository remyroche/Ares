"""
TAS Trainer - Tree Architecture Search Training System

Comprehensive training system for tree-based architectures with advanced
optimization, hardware acceleration, and tree-specific training methods.
"""

import logging
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
from datetime import datetime

# Import shared utilities
try:
    from src.utils.common_operations import (
        memory_checkpoint, gpu_context, optimize_memory, get_memory_usage,
        safe_json_dump, safe_json_load, ensure_directory
    )
    from src.utils.math_validation import MathValidation
    from src.utils.serialization_utils import UniversalSerializer
    from src.utils.tprint import (
        tprint, tprint_info, tprint_debug, tprint_warning, tprint_error,
        tprint_success, tprint_progress, tprint_performance
    )
    from src.utils.hardware.m1_gpu_utils import get_m1_gpu_manager
    from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
    from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer
    SHARED_UTILS_AVAILABLE = True
except ImportError:
    SHARED_UTILS_AVAILABLE = False
    def tprint(*args, **kwargs): print(*args)
    def tprint_info(*args, **kwargs): print("INFO:", *args)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args)
    def tprint_error(*args, **kwargs): print("ERROR:", *args)

# Import tree-specific libraries
try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
    import xgboost as xgb
    import lightgbm as lgb
    TREE_LIBS_AVAILABLE = True
except ImportError:
    TREE_LIBS_AVAILABLE = False
    tprint_warning("Tree libraries not available, using fallback implementations")

logger = logging.getLogger(__name__)

@dataclass
class TASTrainingConfig:
    """Configuration for TAS training."""
    
    # Tree model parameters
    model_type: str = "random_forest"  # random_forest, xgboost, lightgbm, decision_tree
    n_estimators: int = 100
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: Union[str, int, float] = "sqrt"
    random_state: int = 42
    
    # Training parameters
    validation_split: float = 0.2
    cv_folds: int = 5
    enable_cross_validation: bool = True
    enable_early_stopping: bool = True
    early_stopping_rounds: int = 10
    
    # Hardware optimization
    enable_m1_optimization: bool = True
    enable_parallel_processing: bool = True
    n_jobs: int = -1
    memory_limit_gb: Optional[float] = None
    
    # Performance monitoring
    verbose: bool = True
    log_level: str = "INFO"
    save_training_history: bool = True
    save_best_model: bool = True
    
    # Output settings
    output_dir: str = "tas_training_results"
    model_save_path: Optional[str] = None

@dataclass
class TASTrainingResult:
    """Result from TAS training."""
    
    # Training results
    success: bool
    best_model: Optional[Any] = None
    best_score: float = 0.0
    training_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Performance metrics
    train_score: float = 0.0
    validation_score: float = 0.0
    test_score: Optional[float] = None
    cross_validation_scores: Optional[List[float]] = None
    
    # Tree-specific metrics
    feature_importance: Optional[np.ndarray] = None
    tree_depth: Optional[int] = None
    n_leaves: Optional[int] = None
    model_complexity: int = 0
    
    # Training metadata
    execution_time: float = 0.0
    n_estimators_used: int = 0
    convergence_achieved: bool = False
    early_stopping_triggered: bool = False
    
    # Hardware optimization
    m1_optimization_used: bool = False
    memory_usage_mb: float = 0.0
    cpu_utilization: float = 0.0
    
    # Error handling
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

class TASTrainer:
    """
    Tree Architecture Search Trainer.
    
    Comprehensive training system for tree-based architectures with advanced
    optimization, hardware acceleration, and tree-specific training methods.
    """
    
    def __init__(self, config: Optional[TASTrainingConfig] = None):
        """Initialize TAS trainer."""
        self.config = config or TASTrainingConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize hardware optimizations
        self._init_hardware_optimizations()
        
        # Initialize utilities
        self._init_utilities()
        
        # Training state
        self.training_history = []
        self.best_model = None
        self.best_score = -np.inf
        
        tprint_success("🚀 TAS Trainer initialized")
        tprint_info(f"   → Model type: {self.config.model_type}")
        tprint_info(f"   → M1 optimization: {'enabled' if self.config.enable_m1_optimization else 'disabled'}")
        tprint_info(f"   → Parallel processing: {'enabled' if self.config.enable_parallel_processing else 'disabled'}")
    
    def _init_hardware_optimizations(self):
        """Initialize hardware optimizations."""
        if not self.config.enable_m1_optimization or not SHARED_UTILS_AVAILABLE:
            self.gpu_manager = None
            self.memory_optimizer = None
            self.cpu_optimizer = None
            return
        
        try:
            self.gpu_manager = get_m1_gpu_manager()
            self.memory_optimizer = get_m1_memory_optimizer(self.config.memory_limit_gb)
            self.cpu_optimizer = get_m1_cpu_optimizer()
            
            if self.memory_optimizer and hasattr(self.memory_optimizer, 'start_monitoring'):
                self.memory_optimizer.start_monitoring()
            
            tprint_success("✅ M1 hardware optimization initialized")
            
        except Exception as e:
            tprint_warning(f"⚠️ Hardware optimization initialization failed: {e}")
            self.gpu_manager = None
            self.memory_optimizer = None
            self.cpu_optimizer = None
    
    def _init_utilities(self):
        """Initialize utility components."""
        if SHARED_UTILS_AVAILABLE:
            self.math_validator = MathValidation()
            self.serializer = UniversalSerializer()
        else:
            self.math_validator = None
            self.serializer = None
    
    def train(self, 
              X: Union[np.ndarray, pd.DataFrame], 
              y: Union[np.ndarray, pd.Series],
              X_test: Optional[Union[np.ndarray, pd.DataFrame]] = None,
              y_test: Optional[Union[np.ndarray, pd.Series]] = None) -> TASTrainingResult:
        """
        Train a tree-based model.
        
        Args:
            X: Training features
            y: Training targets
            X_test: Optional test features
            y_test: Optional test targets
            
        Returns:
            TASTrainingResult with training results
        """
        start_time = time.time()
        tprint_info("🌳 Starting TAS training")
        
        try:
            # Validate inputs
            self._validate_inputs(X, y)
            
            # Preprocess data
            X_processed, y_processed = self._preprocess_data(X, y)
            
            # Split data for validation
            if X_test is None or y_test is None:
                X_train, X_val, y_train, y_val = train_test_split(
                    X_processed, y_processed, 
                    test_size=self.config.validation_split,
                    random_state=self.config.random_state
                )
            else:
                X_train, y_train = X_processed, y_processed
                X_val, y_val = self._preprocess_data(X_test, y_test)
            
            # Create model
            model = self._create_model()
            
            # Train model with hardware optimization
            with self._get_hardware_context():
                model, training_history = self._train_model(model, X_train, y_train, X_val, y_val)
            
            # Evaluate model
            train_score = self._evaluate_model(model, X_train, y_train)
            val_score = self._evaluate_model(model, X_val, y_val)
            test_score = None
            if X_test is not None and y_test is not None:
                test_score = self._evaluate_model(model, X_val, y_val)
            
            # Cross-validation if enabled
            cv_scores = None
            if self.config.enable_cross_validation:
                cv_scores = self._cross_validate_model(model, X_processed, y_processed)
            
            # Calculate tree-specific metrics
            feature_importance = self._get_feature_importance(model)
            tree_depth = self._get_tree_depth(model)
            n_leaves = self._get_n_leaves(model)
            model_complexity = self._calculate_model_complexity(model)
            
            # Get performance metrics
            memory_usage = self._get_memory_usage()
            cpu_utilization = self._get_cpu_utilization()
            
            # Create result
            result = TASTrainingResult(
                success=True,
                best_model=model,
                best_score=val_score,
                training_history=training_history,
                train_score=train_score,
                validation_score=val_score,
                test_score=test_score,
                cross_validation_scores=cv_scores,
                feature_importance=feature_importance,
                tree_depth=tree_depth,
                n_leaves=n_leaves,
                model_complexity=model_complexity,
                execution_time=time.time() - start_time,
                n_estimators_used=self.config.n_estimators,
                convergence_achieved=self._check_convergence(training_history),
                early_stopping_triggered=self._check_early_stopping(training_history),
                m1_optimization_used=self.config.enable_m1_optimization,
                memory_usage_mb=memory_usage,
                cpu_utilization=cpu_utilization
            )
            
            # Store results
            self.best_model = model
            self.best_score = val_score
            self.training_history = training_history
            
            # Save model if configured
            if self.config.save_best_model:
                self._save_model(model, result)
            
            tprint_success(f"✅ TAS training completed in {result.execution_time:.2f}s")
            tprint_info(f"   → Best score: {result.best_score:.4f}")
            tprint_info(f"   → Model complexity: {result.model_complexity}")
            tprint_info(f"   → Tree depth: {result.tree_depth}")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            tprint_error(f"❌ TAS training failed: {e}")
            
            return TASTrainingResult(
                success=False,
                execution_time=execution_time,
                error_message=str(e)
            )
    
    def _validate_inputs(self, X, y):
        """Validate input data."""
        if not TREE_LIBS_AVAILABLE:
            raise ImportError("Tree libraries not available")
        
        if X is None or y is None:
            raise ValueError("X and y cannot be None")
        
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        
        if len(X) == 0:
            raise ValueError("X and y cannot be empty")
    
    def _preprocess_data(self, X, y):
        """Preprocess input data."""
        # Convert to numpy arrays
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = np.array(X)
        
        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = np.array(y)
        
        # Handle missing values
        X_array = np.nan_to_num(X_array, nan=0.0, posinf=0.0, neginf=0.0)
        y_array = np.nan_to_num(y_array, nan=0.0, posinf=0.0, neginf=0.0)
        
        return X_array, y_array
    
    def _create_model(self):
        """Create tree model based on configuration."""
        model_type = self.config.model_type.lower()
        
        if model_type == "random_forest":
            # Determine if classification or regression
            # For now, assume classification - in practice, this should be determined from data
            return RandomForestClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                min_samples_split=self.config.min_samples_split,
                min_samples_leaf=self.config.min_samples_leaf,
                max_features=self.config.max_features,
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs if self.config.enable_parallel_processing else 1
            )
        
        elif model_type == "xgboost":
            return xgb.XGBClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs if self.config.enable_parallel_processing else 1
            )
        
        elif model_type == "lightgbm":
            return lgb.LGBMClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs if self.config.enable_parallel_processing else 1
            )
        
        elif model_type == "decision_tree":
            return DecisionTreeClassifier(
                max_depth=self.config.max_depth,
                min_samples_split=self.config.min_samples_split,
                min_samples_leaf=self.config.min_samples_leaf,
                max_features=self.config.max_features,
                random_state=self.config.random_state
            )
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _train_model(self, model, X_train, y_train, X_val, y_val):
        """Train the model with monitoring."""
        training_history = []
        
        # Simple training loop (in practice, you'd want more sophisticated training)
        start_time = time.time()
        
        # Fit model
        model.fit(X_train, y_train)
        
        # Evaluate on validation set
        val_score = self._evaluate_model(model, X_val, y_val)
        
        training_history.append({
            'epoch': 1,
            'train_score': self._evaluate_model(model, X_train, y_train),
            'val_score': val_score,
            'time': time.time() - start_time
        })
        
        return model, training_history
    
    def _evaluate_model(self, model, X, y):
        """Evaluate model performance."""
        try:
            y_pred = model.predict(X)
            
            # Use appropriate metric based on problem type
            if hasattr(model, 'predict_proba'):
                # Classification
                return accuracy_score(y, y_pred)
            else:
                # Regression
                return r2_score(y, y_pred)
                
        except Exception as e:
            tprint_warning(f"⚠️ Model evaluation failed: {e}")
            return 0.0
    
    def _cross_validate_model(self, model, X, y):
        """Perform cross-validation."""
        try:
            cv_scores = cross_val_score(model, X, y, cv=self.config.cv_folds)
            return cv_scores.tolist()
        except Exception as e:
            tprint_warning(f"⚠️ Cross-validation failed: {e}")
            return None
    
    def _get_feature_importance(self, model):
        """Get feature importance from model."""
        try:
            if hasattr(model, 'feature_importances_'):
                return model.feature_importances_
            else:
                return None
        except Exception:
            return None
    
    def _get_tree_depth(self, model):
        """Get maximum tree depth."""
        try:
            if hasattr(model, 'max_depth'):
                return model.max_depth
            elif hasattr(model, 'estimators_'):
                # For ensemble models, get average depth
                depths = [tree.tree_.max_depth for tree in model.estimators_]
                return int(np.mean(depths))
            else:
                return None
        except Exception:
            return None
    
    def _get_n_leaves(self, model):
        """Get number of leaves."""
        try:
            if hasattr(model, 'tree_'):
                return model.tree_.n_leaves
            elif hasattr(model, 'estimators_'):
                # For ensemble models, get total leaves
                total_leaves = sum(tree.tree_.n_leaves for tree in model.estimators_)
                return total_leaves
            else:
                return None
        except Exception:
            return None
    
    def _calculate_model_complexity(self, model):
        """Calculate model complexity."""
        try:
            if hasattr(model, 'n_estimators'):
                return model.n_estimators
            else:
                return 1
        except Exception:
            return 1
    
    def _check_convergence(self, training_history):
        """Check if training has converged."""
        if len(training_history) < 2:
            return False
        
        recent_scores = [h['val_score'] for h in training_history[-3:]]
        return abs(recent_scores[-1] - recent_scores[0]) < 0.01
    
    def _check_early_stopping(self, training_history):
        """Check if early stopping was triggered."""
        if len(training_history) < self.config.early_stopping_rounds:
            return False
        
        recent_scores = [h['val_score'] for h in training_history[-self.config.early_stopping_rounds:]]
        return max(recent_scores) != recent_scores[-1]
    
    def _get_memory_usage(self):
        """Get memory usage."""
        try:
            if self.memory_optimizer and hasattr(self.memory_optimizer, 'get_memory_usage'):
                return self.memory_optimizer.get_memory_usage() / (1024 * 1024)  # Convert to MB
            else:
                return 0.0
        except Exception:
            return 0.0
    
    def _get_cpu_utilization(self):
        """Get CPU utilization."""
        try:
            if self.cpu_optimizer and hasattr(self.cpu_optimizer, 'get_cpu_utilization'):
                return self.cpu_optimizer.get_cpu_utilization()
            else:
                return 0.0
        except Exception:
            return 0.0
    
    def _get_hardware_context(self):
        """Get hardware optimization context."""
        if self.config.enable_m1_optimization and SHARED_UTILS_AVAILABLE:
            return memory_checkpoint("tas_training")
        else:
            from contextlib import contextmanager
            @contextmanager
            def dummy_context():
                yield
            return dummy_context()
    
    def _save_model(self, model, result):
        """Save trained model."""
        try:
            if self.config.model_save_path:
                save_path = Path(self.config.model_save_path)
            else:
                save_path = Path(self.config.output_dir) / f"tas_model_{int(time.time())}.pkl"
            
            ensure_directory(save_path.parent)
            
            if self.serializer:
                self.serializer.save(model, str(save_path))
            else:
                import pickle
                with open(save_path, 'wb') as f:
                    pickle.dump(model, f)
            
            tprint_success(f"💾 Model saved to {save_path}")
            
        except Exception as e:
            tprint_warning(f"⚠️ Model saving failed: {e}")
    
    def get_training_summary(self):
        """Get training summary."""
        return {
            'best_score': self.best_score,
            'best_model': self.best_model is not None,
            'training_history_length': len(self.training_history),
            'config': self.config.__dict__,
            'hardware_optimization': self.config.enable_m1_optimization
        }