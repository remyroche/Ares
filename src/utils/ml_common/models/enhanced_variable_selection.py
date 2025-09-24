"""
Enhanced Variable Selection for CVLSA

This module implements advanced variable selection with:
1. Parallel variable selection methods
2. Adaptive method selection based on data characteristics
3. Feature importance integration across methods
4. Incremental selection for dynamic feature management
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import logging
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from sklearn.feature_selection import (
    VarianceThreshold, SelectKBest, SelectPercentile, RFE, RFECV,
    mutual_info_regression, f_regression, chi2
)
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
import threading
from queue import Queue, Empty

# Import existing utilities
from src.utils.matrix_operations.enhanced_operations import get_enhanced_matrix_operations
from src.utils.hardware.m1_gpu_utils import get_m1_gpu_manager
from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer

logger = logging.getLogger(__name__)

@dataclass
class SelectionMethod:
    """Represents a variable selection method."""
    name: str
    method: Callable
    parameters: Dict[str, Any] = field(default_factory=dict)
    performance_score: float = 0.0
    execution_time: float = 0.0
    selected_features: List[int] = field(default_factory=list)
    feature_importance: np.ndarray = field(default_factory=lambda: np.array([]))
    
    def execute(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None) -> Tuple[List[int], np.ndarray]:
        """Execute the selection method."""
        start_time = time.time()
        
        try:
            # Execute the method
            if hasattr(self.method, 'fit_transform'):
                # Methods that return transformed data
                X_selected = self.method.fit_transform(X, y)
                selected_indices = list(range(X_selected.shape[1]))
                importance_scores = np.ones(X_selected.shape[1])
            else:
                # Methods that return selected indices
                self.method.fit(X, y)
                selected_indices = self.method.get_support(indices=True).tolist()
                importance_scores = getattr(self.method, 'scores_', np.ones(len(selected_indices)))
            
            self.execution_time = time.time() - start_time
            self.selected_features = selected_indices
            self.feature_importance = importance_scores
            
            logger.debug(f"✅ {self.name} completed in {self.execution_time:.3f}s, selected {len(selected_indices)} features")
            
            return selected_indices, importance_scores
            
        except Exception as e:
            logger.warning(f"❌ {self.name} failed: {e}")
            self.execution_time = time.time() - start_time
            return [], np.array([])

@dataclass
class VariableSelectionConfig:
    """Configuration for variable selection."""
    # Parallel processing
    use_parallel: bool = True
    max_workers: int = 4
    use_multiprocessing: bool = False
    
    # Selection methods
    methods: List[str] = field(default_factory=lambda: [
        'variance_threshold', 'mutual_info', 'f_regression', 'lasso', 'random_forest', 'rfe'
    ])
    
    # Adaptive selection
    adaptive_method_selection: bool = True
    performance_threshold: float = 0.1
    stability_threshold: float = 0.8
    
    # Feature importance integration
    importance_weighting: str = 'weighted_average'  # 'weighted_average', 'consensus', 'majority_vote'
    consensus_threshold: float = 0.5
    
    # Incremental selection
    enable_incremental: bool = True
    incremental_batch_size: int = 10
    max_features: int = 100
    
    # Performance monitoring
    track_performance: bool = True
    performance_metrics: List[str] = field(default_factory=lambda: ['mse', 'r2', 'mae'])

class EnhancedVariableSelector:
    """Enhanced variable selector with parallel processing and adaptive methods."""
    
    def __init__(self, config: Optional[VariableSelectionConfig] = None):
        self.config = config or VariableSelectionConfig()
        
        # Selection methods registry
        self.selection_methods: Dict[str, SelectionMethod] = {}
        self._init_selection_methods()
        
        # Performance tracking
        self.performance_history: List[Dict[str, Any]] = []
        self.feature_importance_history: List[Dict[str, Any]] = []
        
        # Resource monitoring
        self._init_resource_monitoring()
        
        # Thread safety
        self._lock = threading.Lock()
        
        logger.info("🔍 Enhanced Variable Selector initialized")
    
    def _init_resource_monitoring(self):
        """Initialize resource monitoring."""
        try:
            self.memory_optimizer = get_m1_memory_optimizer()
            self.gpu_manager = get_m1_gpu_manager()
            self.matrix_ops = get_enhanced_matrix_operations()
        except Exception as e:
            logger.warning(f"Resource monitoring not available: {e}")
            self.memory_optimizer = None
            self.gpu_manager = None
            self.matrix_ops = None
    
    def _init_selection_methods(self):
        """Initialize available selection methods."""
        # Variance threshold
        self.selection_methods['variance_threshold'] = SelectionMethod(
            name='Variance Threshold',
            method=VarianceThreshold(threshold=0.01),
            parameters={'threshold': 0.01}
        )
        
        # Mutual information
        self.selection_methods['mutual_info'] = SelectionMethod(
            name='Mutual Information',
            method=SelectKBest(score_func=mutual_info_regression, k='all'),
            parameters={'k': 'all'}
        )
        
        # F-regression
        self.selection_methods['f_regression'] = SelectionMethod(
            name='F-Regression',
            method=SelectKBest(score_func=f_regression, k='all'),
            parameters={'k': 'all'}
        )
        
        # Lasso regularization
        self.selection_methods['lasso'] = SelectionMethod(
            name='Lasso Regularization',
            method=LassoCV(cv=5, random_state=42),
            parameters={'cv': 5}
        )
        
        # Random Forest importance
        self.selection_methods['random_forest'] = SelectionMethod(
            name='Random Forest Importance',
            method=RandomForestRegressor(n_estimators=100, random_state=42),
            parameters={'n_estimators': 100}
        )
        
        # Recursive Feature Elimination
        self.selection_methods['rfe'] = SelectionMethod(
            name='Recursive Feature Elimination',
            method=RFE(estimator=RandomForestRegressor(n_estimators=50), n_features_to_select=0.5),
            parameters={'n_features_to_select': 0.5}
        )
        
        # Extra Trees importance
        self.selection_methods['extra_trees'] = SelectionMethod(
            name='Extra Trees Importance',
            method=ExtraTreesRegressor(n_estimators=100, random_state=42),
            parameters={'n_estimators': 100}
        )
        
        logger.info(f"📋 Initialized {len(self.selection_methods)} selection methods")
    
    def analyze_data_characteristics(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Analyze data characteristics to guide method selection."""
        characteristics = {
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'feature_density': X.shape[1] / X.shape[0] if X.shape[0] > 0 else 0,
            'target_variance': np.var(y),
            'feature_variance_mean': np.mean(np.var(X, axis=0)),
            'feature_variance_std': np.std(np.var(X, axis=0)),
            'correlation_strength': self._calculate_correlation_strength(X),
            'linearity_score': self._calculate_linearity_score(X, y),
            'noise_level': self._estimate_noise_level(X, y)
        }
        
        logger.info("📊 Data characteristics analyzed:")
        for key, value in characteristics.items():
            logger.info(f"   {key}: {value:.4f}")
        
        return characteristics
    
    def _calculate_correlation_strength(self, X: np.ndarray) -> float:
        """Calculate average correlation strength between features."""
        try:
            correlation_matrix = np.corrcoef(X.T)
            # Remove diagonal (self-correlation)
            mask = ~np.eye(correlation_matrix.shape[0], dtype=bool)
            correlations = correlation_matrix[mask]
            return np.mean(np.abs(correlations))
        except Exception:
            return 0.0
    
    def _calculate_linearity_score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate linearity score between features and target."""
        try:
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import r2_score
            
            # Fit linear model
            lr = LinearRegression()
            lr.fit(X, y)
            y_pred = lr.predict(X)
            
            # Calculate R² as linearity score
            r2 = r2_score(y, y_pred)
            return max(0, r2)  # Ensure non-negative
        except Exception:
            return 0.0
    
    def _estimate_noise_level(self, X: np.ndarray, y: np.ndarray) -> float:
        """Estimate noise level in the data."""
        try:
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import mean_squared_error
            
            # Fit linear model
            lr = LinearRegression()
            lr.fit(X, y)
            y_pred = lr.predict(X)
            
            # Calculate residual variance as noise estimate
            mse = mean_squared_error(y, y_pred)
            return mse / (np.var(y) + 1e-8)  # Normalize by target variance
        except Exception:
            return 0.5  # Default moderate noise
    
    def select_adaptive_methods(self, characteristics: Dict[str, Any]) -> List[str]:
        """Select methods based on data characteristics."""
        if not self.config.adaptive_method_selection:
            return self.config.methods
        
        selected_methods = []
        
        # High dimensionality -> use efficient methods
        if characteristics['feature_density'] > 0.1:
            selected_methods.extend(['variance_threshold', 'lasso', 'random_forest'])
        
        # High correlation -> use methods that handle multicollinearity
        if characteristics['correlation_strength'] > 0.5:
            selected_methods.extend(['lasso', 'rfe'])
        
        # High linearity -> use linear methods
        if characteristics['linearity_score'] > 0.7:
            selected_methods.extend(['f_regression', 'lasso'])
        
        # High noise -> use robust methods
        if characteristics['noise_level'] > 0.3:
            selected_methods.extend(['random_forest', 'extra_trees'])
        
        # Always include mutual information for non-linear relationships
        selected_methods.append('mutual_info')
        
        # Remove duplicates and ensure methods exist
        selected_methods = list(set(selected_methods))
        available_methods = [m for m in selected_methods if m in self.selection_methods]
        
        if not available_methods:
            available_methods = ['variance_threshold', 'mutual_info', 'random_forest']
        
        logger.info(f"🎯 Adaptive method selection: {available_methods}")
        return available_methods
    
    def execute_parallel_selection(self, X: np.ndarray, y: np.ndarray,
                                 feature_names: Optional[List[str]] = None,
                                 methods: Optional[List[str]] = None) -> Dict[str, SelectionMethod]:
        """Execute variable selection methods in parallel."""
        if methods is None:
            methods = self.config.methods
        
        logger.info(f"🚀 Executing parallel variable selection with {len(methods)} methods")
        
        # Prepare methods for parallel execution
        method_tasks = []
        for method_name in methods:
            if method_name in self.selection_methods:
                method_tasks.append((method_name, self.selection_methods[method_name]))
            else:
                logger.warning(f"Method {method_name} not found")
        
        results = {}
        
        if self.config.use_parallel and len(method_tasks) > 1:
            # Parallel execution
            if self.config.use_multiprocessing:
                with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
                    futures = {
                        executor.submit(self._execute_single_method, method_name, method, X, y, feature_names): method_name
                        for method_name, method in method_tasks
                    }
                    
                    for future in futures:
                        method_name = futures[future]
                        try:
                            selected_indices, importance_scores = future.result(timeout=300)  # 5 minute timeout
                            self.selection_methods[method_name].selected_features = selected_indices
                            self.selection_methods[method_name].feature_importance = importance_scores
                            results[method_name] = self.selection_methods[method_name]
                        except Exception as e:
                            logger.error(f"Method {method_name} failed: {e}")
            else:
                with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                    futures = {
                        executor.submit(self._execute_single_method, method_name, method, X, y, feature_names): method_name
                        for method_name, method in method_tasks
                    }
                    
                    for future in futures:
                        method_name = futures[future]
                        try:
                            selected_indices, importance_scores = future.result(timeout=300)
                            self.selection_methods[method_name].selected_features = selected_indices
                            self.selection_methods[method_name].feature_importance = importance_scores
                            results[method_name] = self.selection_methods[method_name]
                        except Exception as e:
                            logger.error(f"Method {method_name} failed: {e}")
        else:
            # Sequential execution
            for method_name, method in method_tasks:
                try:
                    selected_indices, importance_scores = self._execute_single_method(
                        method_name, method, X, y, feature_names
                    )
                    self.selection_methods[method_name].selected_features = selected_indices
                    self.selection_methods[method_name].feature_importance = importance_scores
                    results[method_name] = self.selection_methods[method_name]
                except Exception as e:
                    logger.error(f"Method {method_name} failed: {e}")
        
        logger.info(f"✅ Parallel selection completed: {len(results)} methods succeeded")
        return results
    
    def _execute_single_method(self, method_name: str, method: SelectionMethod,
                              X: np.ndarray, y: np.ndarray,
                              feature_names: Optional[List[str]] = None) -> Tuple[List[int], np.ndarray]:
        """Execute a single selection method."""
        return method.execute(X, y, feature_names)
    
    def integrate_feature_importance(self, results: Dict[str, SelectionMethod]) -> np.ndarray:
        """Integrate feature importance across different methods."""
        if not results:
            return np.array([])
        
        n_features = max(len(method.selected_features) for method in results.values())
        if n_features == 0:
            return np.array([])
        
        # Initialize importance matrix
        importance_matrix = np.zeros((len(results), n_features))
        method_weights = np.zeros(len(results))
        
        # Fill importance matrix
        for i, (method_name, method) in enumerate(results.items()):
            if len(method.selected_features) > 0:
                # Normalize importance scores
                if len(method.feature_importance) > 0:
                    importance_scores = method.feature_importance
                    if np.max(importance_scores) > 0:
                        importance_scores = importance_scores / np.max(importance_scores)
                else:
                    importance_scores = np.ones(len(method.selected_features))
                
                # Set importance for selected features
                for j, feature_idx in enumerate(method.selected_features):
                    if feature_idx < n_features:
                        importance_matrix[i, feature_idx] = importance_scores[j] if j < len(importance_scores) else 1.0
                
                # Calculate method weight based on performance
                method_weights[i] = 1.0 / (method.execution_time + 1e-8)  # Weight by speed
        
        # Normalize method weights
        if np.sum(method_weights) > 0:
            method_weights = method_weights / np.sum(method_weights)
        else:
            method_weights = np.ones(len(results)) / len(results)
        
        # Calculate integrated importance
        if self.config.importance_weighting == 'weighted_average':
            integrated_importance = np.average(importance_matrix, axis=0, weights=method_weights)
        elif self.config.importance_weighting == 'consensus':
            # Features selected by multiple methods get higher importance
            consensus_matrix = (importance_matrix > 0).astype(float)
            integrated_importance = np.average(consensus_matrix, axis=0, weights=method_weights)
        elif self.config.importance_weighting == 'majority_vote':
            # Binary importance based on majority vote
            consensus_matrix = (importance_matrix > 0).astype(float)
            integrated_importance = np.mean(consensus_matrix, axis=0)
        else:
            integrated_importance = np.mean(importance_matrix, axis=0)
        
        logger.info(f"🔗 Integrated feature importance using {self.config.importance_weighting}")
        logger.info(f"   Selected features: {np.sum(integrated_importance > 0)}")
        
        return integrated_importance
    
    def select_features_incremental(self, X: np.ndarray, y: np.ndarray,
                                  feature_names: Optional[List[str]] = None,
                                  initial_features: Optional[List[int]] = None) -> List[int]:
        """Incremental feature selection for dynamic feature management."""
        if not self.config.enable_incremental:
            return self._select_features_standard(X, y, feature_names)
        
        logger.info("🔄 Starting incremental feature selection...")
        
        # Initialize with existing features or empty set
        selected_features = set(initial_features) if initial_features else set()
        remaining_features = set(range(X.shape[1])) - selected_features
        
        # Performance tracking
        performance_history = []
        best_performance = float('-inf')
        stagnation_count = 0
        
        while remaining_features and len(selected_features) < self.config.max_features:
            # Evaluate adding each remaining feature
            feature_scores = {}
            
            for feature_idx in list(remaining_features)[:self.config.incremental_batch_size]:
                try:
                    # Create feature set with this feature added
                    test_features = list(selected_features) + [feature_idx]
                    X_test = X[:, test_features]
                    
                    # Quick performance evaluation
                    score = self._evaluate_feature_set(X_test, y)
                    feature_scores[feature_idx] = score
                    
                except Exception as e:
                    logger.debug(f"Failed to evaluate feature {feature_idx}: {e}")
                    continue
            
            if not feature_scores:
                break
            
            # Select best feature
            best_feature = max(feature_scores, key=feature_scores.get)
            best_score = feature_scores[best_feature]
            
            # Check for improvement
            if best_score > best_performance + self.config.performance_threshold:
                selected_features.add(best_feature)
                remaining_features.remove(best_feature)
                best_performance = best_score
                stagnation_count = 0
                
                performance_history.append({
                    'iteration': len(performance_history),
                    'feature_added': best_feature,
                    'performance': best_score,
                    'total_features': len(selected_features)
                })
                
                logger.debug(f"   Added feature {best_feature}, performance: {best_score:.4f}")
            else:
                stagnation_count += 1
                if stagnation_count >= 5:  # Stop if no improvement for 5 iterations
                    logger.info("🛑 No improvement found, stopping incremental selection")
                    break
        
        logger.info(f"✅ Incremental selection completed: {len(selected_features)} features selected")
        return list(selected_features)
    
    def _evaluate_feature_set(self, X: np.ndarray, y: np.ndarray) -> float:
        """Quick evaluation of a feature set."""
        try:
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import r2_score
            from sklearn.model_selection import cross_val_score
            
            # Use cross-validation for robust evaluation
            lr = LinearRegression()
            scores = cross_val_score(lr, X, y, cv=3, scoring='r2')
            return np.mean(scores)
        except Exception:
            return 0.0
    
    def _select_features_standard(self, X: np.ndarray, y: np.ndarray,
                                feature_names: Optional[List[str]] = None) -> List[int]:
        """Standard feature selection without incremental approach."""
        # Analyze data characteristics
        characteristics = self.analyze_data_characteristics(X, y)
        
        # Select adaptive methods
        methods = self.select_adaptive_methods(characteristics)
        
        # Execute parallel selection
        results = self.execute_parallel_selection(X, y, feature_names, methods)
        
        # Integrate feature importance
        integrated_importance = self.integrate_feature_importance(results)
        
        # Select features based on importance
        if len(integrated_importance) > 0:
            # Select top features
            n_features_to_select = min(self.config.max_features, len(integrated_importance))
            selected_indices = np.argsort(integrated_importance)[-n_features_to_select:]
            selected_features = selected_indices[integrated_importance[selected_indices] > 0].tolist()
        else:
            selected_features = []
        
        return selected_features
    
    def select_features(self, X: np.ndarray, y: np.ndarray,
                       feature_names: Optional[List[str]] = None,
                       incremental: Optional[bool] = None) -> Tuple[List[int], Dict[str, Any]]:
        """Main feature selection method."""
        logger.info("🔍 Starting enhanced variable selection...")
        
        start_time = time.time()
        
        # Determine if incremental selection should be used
        use_incremental = incremental if incremental is not None else self.config.enable_incremental
        
        # Resource monitoring
        if self.memory_optimizer:
            with self.memory_optimizer.memory_checkpoint("variable_selection"):
                if use_incremental:
                    selected_features = self.select_features_incremental(X, y, feature_names)
                else:
                    selected_features = self._select_features_standard(X, y, feature_names)
        else:
            if use_incremental:
                selected_features = self.select_features_incremental(X, y, feature_names)
            else:
                selected_features = self._select_features_standard(X, y, feature_names)
        
        # Prepare results
        selection_time = time.time() - start_time
        results = {
            'selected_features': selected_features,
            'selection_time': selection_time,
            'n_features_selected': len(selected_features),
            'n_features_original': X.shape[1],
            'reduction_ratio': len(selected_features) / X.shape[1] if X.shape[1] > 0 else 0,
            'methods_used': list(self.selection_methods.keys()),
            'performance_history': self.performance_history
        }
        
        # Store performance history
        if self.config.track_performance:
            self.performance_history.append({
                'timestamp': time.time(),
                'selection_time': selection_time,
                'n_features_selected': len(selected_features),
                'reduction_ratio': results['reduction_ratio']
            })
        
        logger.info(f"✅ Variable selection completed in {selection_time:.2f}s")
        logger.info(f"   Selected {len(selected_features)}/{X.shape[1]} features ({results['reduction_ratio']:.1%})")
        
        return selected_features, results
    
    def get_selection_analytics(self) -> Dict[str, Any]:
        """Get comprehensive analytics about the selection process."""
        analytics = {
            'total_selections': len(self.performance_history),
            'average_selection_time': np.mean([h['selection_time'] for h in self.performance_history]) if self.performance_history else 0,
            'average_reduction_ratio': np.mean([h['reduction_ratio'] for h in self.performance_history]) if self.performance_history else 0,
            'method_performance': {
                name: {
                    'execution_time': method.execution_time,
                    'features_selected': len(method.selected_features),
                    'performance_score': method.performance_score
                }
                for name, method in self.selection_methods.items()
            },
            'performance_history': self.performance_history
        }
        
        return analytics


# Factory functions
def create_enhanced_variable_selector(config: Optional[VariableSelectionConfig] = None) -> EnhancedVariableSelector:
    """Create enhanced variable selector."""
    return EnhancedVariableSelector(config)