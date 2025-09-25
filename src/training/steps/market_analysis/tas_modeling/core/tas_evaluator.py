"""
TAS Evaluator - Tree Architecture Search Evaluation System

Comprehensive evaluation system for tree-based architectures with tree-specific
metrics, feature importance analysis, and tree structure evaluation.
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
    SHARED_UTILS_AVAILABLE = True
except ImportError:
    SHARED_UTILS_AVAILABLE = False
    def tprint(*args, **kwargs): print(*args)
    def tprint_info(*args, **kwargs): print("INFO:", *args)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args)
    def tprint_error(*args, **kwargs): print("ERROR:", *args)

# Import tree-specific libraries
try:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        mean_squared_error, mean_absolute_error, r2_score,
        classification_report, confusion_matrix
    )
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    TREE_LIBS_AVAILABLE = True
except ImportError:
    TREE_LIBS_AVAILABLE = False
    tprint_warning("Tree libraries not available, using fallback implementations")

logger = logging.getLogger(__name__)

@dataclass
class TASEvaluationConfig:
    """Configuration for TAS evaluation."""
    
    # Evaluation parameters
    evaluation_metrics: List[str] = field(default_factory=lambda: [
        'accuracy', 'precision', 'recall', 'f1_score', 'feature_importance'
    ])
    cross_validation_folds: int = 5
    enable_cross_validation: bool = True
    enable_feature_importance: bool = True
    enable_tree_analysis: bool = True
    
    # Tree-specific analysis
    analyze_tree_depth: bool = True
    analyze_tree_structure: bool = True
    analyze_leaf_purity: bool = True
    analyze_feature_usage: bool = True
    
    # Performance monitoring
    verbose: bool = True
    log_level: str = "INFO"
    save_evaluation_results: bool = True
    
    # Output settings
    output_dir: str = "tas_evaluation_results"
    results_format: str = "json"  # json, pickle

@dataclass
class TASEvaluationResult:
    """Result from TAS evaluation."""
    
    # Basic metrics
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    
    # Regression metrics (if applicable)
    mse: float = 0.0
    mae: float = 0.0
    r2_score: float = 0.0
    
    # Tree-specific metrics
    feature_importance: Optional[np.ndarray] = None
    tree_depth: Optional[int] = None
    n_leaves: Optional[int] = None
    leaf_purity: Optional[float] = None
    feature_usage: Optional[Dict[str, int]] = None
    
    # Cross-validation results
    cv_scores: Optional[List[float]] = None
    cv_mean: Optional[float] = None
    cv_std: Optional[float] = None
    
    # Tree structure analysis
    tree_structure: Optional[Dict[str, Any]] = None
    node_analysis: Optional[Dict[str, Any]] = None
    
    # Performance metrics
    evaluation_time: float = 0.0
    memory_usage_mb: float = 0.0
    
    # Metadata
    model_type: str = ""
    n_features: int = 0
    n_samples: int = 0
    success: bool = True
    error_message: Optional[str] = None

class TASEvaluator:
    """
    Tree Architecture Search Evaluator.
    
    Comprehensive evaluation system for tree-based architectures with tree-specific
    metrics, feature importance analysis, and tree structure evaluation.
    """
    
    def __init__(self, config: Optional[TASEvaluationConfig] = None):
        """Initialize TAS evaluator."""
        self.config = config or TASEvaluationConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize utilities
        self._init_utilities()
        
        # Evaluation state
        self.evaluation_history = []
        
        tprint_success("🚀 TAS Evaluator initialized")
        tprint_info(f"   → Evaluation metrics: {self.config.evaluation_metrics}")
        tprint_info(f"   → Cross-validation: {'enabled' if self.config.enable_cross_validation else 'disabled'}")
        tprint_info(f"   → Tree analysis: {'enabled' if self.config.enable_tree_analysis else 'disabled'}")
    
    def _init_utilities(self):
        """Initialize utility components."""
        if SHARED_UTILS_AVAILABLE:
            self.math_validator = MathValidation()
            self.serializer = UniversalSerializer()
        else:
            self.math_validator = None
            self.serializer = None
    
    def evaluate(self, 
                 model: Any,
                 X: Union[np.ndarray, pd.DataFrame], 
                 y: Union[np.ndarray, pd.Series],
                 X_test: Optional[Union[np.ndarray, pd.DataFrame]] = None,
                 y_test: Optional[Union[np.ndarray, pd.Series]] = None) -> TASEvaluationResult:
        """
        Evaluate a tree-based model.
        
        Args:
            model: Trained tree model
            X: Training features
            y: Training targets
            X_test: Optional test features
            y_test: Optional test targets
            
        Returns:
            TASEvaluationResult with evaluation results
        """
        start_time = time.time()
        tprint_info("🔍 Starting TAS evaluation")
        
        try:
            # Validate inputs
            self._validate_inputs(model, X, y)
            
            # Preprocess data
            X_processed, y_processed = self._preprocess_data(X, y)
            
            # Determine test data
            if X_test is not None and y_test is not None:
                X_test_processed, y_test_processed = self._preprocess_data(X_test, y_test)
            else:
                X_test_processed, y_test_processed = X_processed, y_processed
            
            # Basic evaluation metrics
            basic_metrics = self._evaluate_basic_metrics(model, X_test_processed, y_test_processed)
            
            # Tree-specific metrics
            tree_metrics = self._evaluate_tree_metrics(model, X_processed, y_processed)
            
            # Cross-validation if enabled
            cv_results = None
            if self.config.enable_cross_validation:
                cv_results = self._cross_validate_model(model, X_processed, y_processed)
            
            # Tree structure analysis
            tree_analysis = None
            if self.config.enable_tree_analysis:
                tree_analysis = self._analyze_tree_structure(model, X_processed, y_processed)
            
            # Get performance metrics
            memory_usage = self._get_memory_usage()
            
            # Create result
            result = TASEvaluationResult(
                # Basic metrics
                accuracy=basic_metrics.get('accuracy', 0.0),
                precision=basic_metrics.get('precision', 0.0),
                recall=basic_metrics.get('recall', 0.0),
                f1_score=basic_metrics.get('f1_score', 0.0),
                mse=basic_metrics.get('mse', 0.0),
                mae=basic_metrics.get('mae', 0.0),
                r2_score=basic_metrics.get('r2_score', 0.0),
                
                # Tree-specific metrics
                feature_importance=tree_metrics.get('feature_importance'),
                tree_depth=tree_metrics.get('tree_depth'),
                n_leaves=tree_metrics.get('n_leaves'),
                leaf_purity=tree_metrics.get('leaf_purity'),
                feature_usage=tree_metrics.get('feature_usage'),
                
                # Cross-validation results
                cv_scores=cv_results.get('scores') if cv_results else None,
                cv_mean=cv_results.get('mean') if cv_results else None,
                cv_std=cv_results.get('std') if cv_results else None,
                
                # Tree structure analysis
                tree_structure=tree_analysis.get('structure') if tree_analysis else None,
                node_analysis=tree_analysis.get('nodes') if tree_analysis else None,
                
                # Performance metrics
                evaluation_time=time.time() - start_time,
                memory_usage_mb=memory_usage,
                
                # Metadata
                model_type=type(model).__name__,
                n_features=X_processed.shape[1],
                n_samples=X_processed.shape[0],
                success=True
            )
            
            # Store evaluation history
            self.evaluation_history.append(result)
            
            # Save results if configured
            if self.config.save_evaluation_results:
                self._save_evaluation_results(result)
            
            tprint_success(f"✅ TAS evaluation completed in {result.evaluation_time:.2f}s")
            tprint_info(f"   → Accuracy: {result.accuracy:.4f}")
            tprint_info(f"   → F1 Score: {result.f1_score:.4f}")
            tprint_info(f"   → Tree Depth: {result.tree_depth}")
            tprint_info(f"   → Feature Importance: {result.feature_importance is not None}")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            tprint_error(f"❌ TAS evaluation failed: {e}")
            
            return TASEvaluationResult(
                success=False,
                evaluation_time=execution_time,
                error_message=str(e)
            )
    
    def _validate_inputs(self, model, X, y):
        """Validate input data."""
        if not TREE_LIBS_AVAILABLE:
            raise ImportError("Tree libraries not available")
        
        if model is None:
            raise ValueError("Model cannot be None")
        
        if X is None or y is None:
            raise ValueError("X and y cannot be None")
        
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
    
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
    
    def _evaluate_basic_metrics(self, model, X, y):
        """Evaluate basic performance metrics."""
        try:
            y_pred = model.predict(X)
            
            # Determine if classification or regression
            is_classification = hasattr(model, 'predict_proba') or hasattr(model, 'classes_')
            
            metrics = {}
            
            if is_classification:
                # Classification metrics
                metrics['accuracy'] = accuracy_score(y, y_pred)
                metrics['precision'] = precision_score(y, y_pred, average='weighted', zero_division=0)
                metrics['recall'] = recall_score(y, y_pred, average='weighted', zero_division=0)
                metrics['f1_score'] = f1_score(y, y_pred, average='weighted', zero_division=0)
            else:
                # Regression metrics
                metrics['mse'] = mean_squared_error(y, y_pred)
                metrics['mae'] = mean_absolute_error(y, y_pred)
                metrics['r2_score'] = r2_score(y, y_pred)
                # For regression, use R² as accuracy equivalent
                metrics['accuracy'] = max(0.0, metrics['r2_score'])
                metrics['precision'] = 0.0
                metrics['recall'] = 0.0
                metrics['f1_score'] = 0.0
            
            return metrics
            
        except Exception as e:
            tprint_warning(f"⚠️ Basic metrics evaluation failed: {e}")
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'mse': 0.0,
                'mae': 0.0,
                'r2_score': 0.0
            }
    
    def _evaluate_tree_metrics(self, model, X, y):
        """Evaluate tree-specific metrics."""
        try:
            metrics = {}
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                metrics['feature_importance'] = model.feature_importances_
            else:
                metrics['feature_importance'] = None
            
            # Tree depth
            if hasattr(model, 'max_depth'):
                metrics['tree_depth'] = model.max_depth
            elif hasattr(model, 'estimators_'):
                # For ensemble models, get average depth
                depths = [tree.tree_.max_depth for tree in model.estimators_]
                metrics['tree_depth'] = int(np.mean(depths))
            else:
                metrics['tree_depth'] = None
            
            # Number of leaves
            if hasattr(model, 'tree_'):
                metrics['n_leaves'] = model.tree_.n_leaves
            elif hasattr(model, 'estimators_'):
                # For ensemble models, get total leaves
                total_leaves = sum(tree.tree_.n_leaves for tree in model.estimators_)
                metrics['n_leaves'] = total_leaves
            else:
                metrics['n_leaves'] = None
            
            # Leaf purity
            metrics['leaf_purity'] = self._calculate_leaf_purity(model, X, y)
            
            # Feature usage
            metrics['feature_usage'] = self._analyze_feature_usage(model, X)
            
            return metrics
            
        except Exception as e:
            tprint_warning(f"⚠️ Tree metrics evaluation failed: {e}")
            return {
                'feature_importance': None,
                'tree_depth': None,
                'n_leaves': None,
                'leaf_purity': None,
                'feature_usage': None
            }
    
    def _calculate_leaf_purity(self, model, X, y):
        """Calculate average leaf purity."""
        try:
            if hasattr(model, 'tree_'):
                # Single tree
                leaf_ids = model.apply(X)
                unique_leaves = np.unique(leaf_ids)
                
                purities = []
                for leaf_id in unique_leaves:
                    mask = leaf_ids == leaf_id
                    leaf_labels = y[mask]
                    if len(leaf_labels) > 0:
                        # Calculate purity as the proportion of the most common class
                        unique, counts = np.unique(leaf_labels, return_counts=True)
                        purity = np.max(counts) / len(leaf_labels)
                        purities.append(purity)
                
                return np.mean(purities) if purities else 0.0
                
            elif hasattr(model, 'estimators_'):
                # Ensemble model
                all_purities = []
                for tree in model.estimators_:
                    leaf_ids = tree.apply(X)
                    unique_leaves = np.unique(leaf_ids)
                    
                    purities = []
                    for leaf_id in unique_leaves:
                        mask = leaf_ids == leaf_id
                        leaf_labels = y[mask]
                        if len(leaf_labels) > 0:
                            unique, counts = np.unique(leaf_labels, return_counts=True)
                            purity = np.max(counts) / len(leaf_labels)
                            purities.append(purity)
                    
                    if purities:
                        all_purities.append(np.mean(purities))
                
                return np.mean(all_purities) if all_purities else 0.0
            else:
                return None
                
        except Exception as e:
            tprint_warning(f"⚠️ Leaf purity calculation failed: {e}")
            return None
    
    def _analyze_feature_usage(self, model, X):
        """Analyze feature usage in the tree."""
        try:
            if hasattr(model, 'tree_'):
                # Single tree
                feature_usage = {}
                tree = model.tree_
                
                for i in range(tree.node_count):
                    if tree.children_left[i] != tree.children_right[i]:  # Not a leaf
                        feature = tree.feature[i]
                        if feature >= 0:
                            feature_usage[f'feature_{feature}'] = feature_usage.get(f'feature_{feature}', 0) + 1
                
                return feature_usage
                
            elif hasattr(model, 'estimators_'):
                # Ensemble model
                all_feature_usage = {}
                
                for tree in model.estimators_:
                    tree_feature_usage = {}
                    for i in range(tree.tree_.node_count):
                        if tree.tree_.children_left[i] != tree.tree_.children_right[i]:  # Not a leaf
                            feature = tree.tree_.feature[i]
                            if feature >= 0:
                                tree_feature_usage[f'feature_{feature}'] = tree_feature_usage.get(f'feature_{feature}', 0) + 1
                    
                    # Merge with overall usage
                    for feature, count in tree_feature_usage.items():
                        all_feature_usage[feature] = all_feature_usage.get(feature, 0) + count
                
                return all_feature_usage
            else:
                return None
                
        except Exception as e:
            tprint_warning(f"⚠️ Feature usage analysis failed: {e}")
            return None
    
    def _cross_validate_model(self, model, X, y):
        """Perform cross-validation."""
        try:
            # Determine if classification or regression
            is_classification = hasattr(model, 'predict_proba') or hasattr(model, 'classes_')
            
            if is_classification:
                cv = StratifiedKFold(n_splits=self.config.cross_validation_folds, shuffle=True, random_state=42)
                scoring = 'accuracy'
            else:
                from sklearn.model_selection import KFold
                cv = KFold(n_splits=self.config.cross_validation_folds, shuffle=True, random_state=42)
                scoring = 'r2'
            
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
            
            return {
                'scores': cv_scores.tolist(),
                'mean': float(np.mean(cv_scores)),
                'std': float(np.std(cv_scores))
            }
            
        except Exception as e:
            tprint_warning(f"⚠️ Cross-validation failed: {e}")
            return None
    
    def _analyze_tree_structure(self, model, X, y):
        """Analyze tree structure in detail."""
        try:
            structure = {}
            node_analysis = {}
            
            if hasattr(model, 'tree_'):
                # Single tree analysis
                tree = model.tree_
                structure['n_nodes'] = tree.node_count
                structure['n_leaves'] = tree.n_leaves
                structure['max_depth'] = tree.max_depth
                
                # Node analysis
                for i in range(tree.node_count):
                    node_info = {
                        'feature': tree.feature[i],
                        'threshold': tree.threshold[i],
                        'n_samples': tree.n_node_samples[i],
                        'is_leaf': tree.children_left[i] == tree.children_right[i]
                    }
                    node_analysis[f'node_{i}'] = node_info
                
            elif hasattr(model, 'estimators_'):
                # Ensemble model analysis
                structure['n_estimators'] = len(model.estimators_)
                structure['n_nodes'] = sum(tree.tree_.node_count for tree in model.estimators_)
                structure['n_leaves'] = sum(tree.tree_.n_leaves for tree in model.estimators_)
                structure['max_depth'] = max(tree.tree_.max_depth for tree in model.estimators_)
                
                # Average node analysis across trees
                all_nodes = []
                for tree in model.estimators_:
                    for i in range(tree.tree_.node_count):
                        all_nodes.append({
                            'feature': tree.tree_.feature[i],
                            'threshold': tree.tree_.threshold[i],
                            'n_samples': tree.tree_.n_node_samples[i],
                            'is_leaf': tree.tree_.children_left[i] == tree.tree_.children_right[i]
                        })
                
                # Aggregate statistics
                node_analysis = {
                    'avg_n_samples': np.mean([n['n_samples'] for n in all_nodes]),
                    'n_leaf_nodes': sum(1 for n in all_nodes if n['is_leaf']),
                    'n_internal_nodes': sum(1 for n in all_nodes if not n['is_leaf'])
                }
            
            return {
                'structure': structure,
                'nodes': node_analysis
            }
            
        except Exception as e:
            tprint_warning(f"⚠️ Tree structure analysis failed: {e}")
            return None
    
    def _get_memory_usage(self):
        """Get memory usage."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)  # Convert to MB
        except Exception:
            return 0.0
    
    def _save_evaluation_results(self, result):
        """Save evaluation results."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"tas_evaluation_{timestamp}.{self.config.results_format}"
            filepath = Path(self.config.output_dir) / filename
            
            ensure_directory(filepath.parent)
            
            # Prepare data for serialization
            result_data = {
                'accuracy': result.accuracy,
                'precision': result.precision,
                'recall': result.recall,
                'f1_score': result.f1_score,
                'mse': result.mse,
                'mae': result.mae,
                'r2_score': result.r2_score,
                'feature_importance': result.feature_importance.tolist() if result.feature_importance is not None else None,
                'tree_depth': result.tree_depth,
                'n_leaves': result.n_leaves,
                'leaf_purity': result.leaf_purity,
                'feature_usage': result.feature_usage,
                'cv_scores': result.cv_scores,
                'cv_mean': result.cv_mean,
                'cv_std': result.cv_std,
                'tree_structure': result.tree_structure,
                'node_analysis': result.node_analysis,
                'evaluation_time': result.evaluation_time,
                'memory_usage_mb': result.memory_usage_mb,
                'model_type': result.model_type,
                'n_features': result.n_features,
                'n_samples': result.n_samples,
                'success': result.success,
                'error_message': result.error_message
            }
            
            if self.config.results_format == 'json':
                safe_json_dump(result_data, filepath)
            elif self.config.results_format == 'pickle':
                import pickle
                with open(filepath, 'wb') as f:
                    pickle.dump(result_data, f)
            
            tprint_success(f"💾 Evaluation results saved to {filepath}")
            
        except Exception as e:
            tprint_warning(f"⚠️ Failed to save evaluation results: {e}")
    
    def get_evaluation_summary(self):
        """Get evaluation summary."""
        if not self.evaluation_history:
            return {'message': 'No evaluation results available'}
        
        latest_result = self.evaluation_history[-1]
        
        return {
            'total_evaluations': len(self.evaluation_history),
            'latest_result': {
                'accuracy': latest_result.accuracy,
                'f1_score': latest_result.f1_score,
                'tree_depth': latest_result.tree_depth,
                'n_leaves': latest_result.n_leaves,
                'evaluation_time': latest_result.evaluation_time,
                'success': latest_result.success
            },
            'config': self.config.__dict__
        }