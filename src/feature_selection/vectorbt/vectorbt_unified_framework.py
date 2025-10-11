"""
VectorBT Unified Feature Selection Framework

This module provides a unified framework for all VectorBT-optimized feature selection
methods with consistent API and performance monitoring.
"""

import numpy as np
import pandas as pd
import time
import logging
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from enum import Enum

# VectorBT imports
try:
    import vectorbt as vbt
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None

# Import VectorBT selectors
from .vectorbt_feature_selector import VectorBTFeatureSelector
from .vectorbt_correlation_filter import VectorBTCorrelationFilter
from .vectorbt_mutual_information import VectorBTMutualInformation
from .vectorbt_stability_selection import VectorBTStabilitySelection
from .vectorbt_mrmr_selector import VectorBTMRMRSelector
from .vectorbt_regularization import VectorBTRegularizationSelector
from .vectorbt_rfe_selector import VectorBTRFESelector
from .vectorbt_config import VectorBTFeatureSelectionConfig

# Import utilities
from src.utils.tprint import tprint, tprint_success, tprint_warning, tprint_performance, tprint_debug
from src.utils.math_validation import validate_numeric_array, validate_finite

logger = logging.getLogger(__name__)


class FeatureSelectionMethod(Enum):
    """Available feature selection methods."""
    COMPREHENSIVE = "comprehensive"
    CORRELATION = "correlation"
    MUTUAL_INFORMATION = "mutual_information"
    STABILITY_SELECTION = "stability_selection"
    MRMR = "mrmr"
    LASSO = "lasso"
    ELASTICNET = "elasticnet"
    RFE = "rfe"
    ADAPTIVE = "adaptive"


@dataclass
class SelectionResult:
    """Result of feature selection."""
    success: bool
    selected_features: List[str]
    selected_indices: List[int]
    feature_scores: Dict[str, float]
    n_selected: int
    n_total: int
    method: str
    execution_time: float
    performance_stats: Dict[str, Any]
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class VectorBTUnifiedFramework:
    """
    Unified VectorBT feature selection framework.
    
    This class provides:
    - Consistent API across all VectorBT-optimized methods
    - Automatic method selection based on data characteristics
    - Performance monitoring and benchmarking
    - Memory-efficient processing for large datasets
    - Financial data optimization
    """
    
    def __init__(self, config: Optional[VectorBTFeatureSelectionConfig] = None):
        """Initialize VectorBT unified framework."""
        self.config = config or VectorBTFeatureSelectionConfig()
        self.logger = logger.getChild('VectorBTUnifiedFramework')
        
        # Check VectorBT availability
        if not VECTORBT_AVAILABLE:
            raise ImportError("VectorBT is required but not available. Please install vectorbt.")
        
        # Initialize selectors
        self._initialize_selectors()
        
        # Performance tracking
        self.performance_stats = {
            'total_selections': 0,
            'successful_selections': 0,
            'failed_selections': 0,
            'total_time': 0.0,
            'methods_used': {},
            'avg_execution_time': 0.0,
            'memory_saved_mb': 0.0
        }
        
        tprint_success("🚀 VectorBTUnifiedFramework initialized")
    
    def _initialize_selectors(self):
        """Initialize all VectorBT selectors."""
        try:
            self.correlation_filter = VectorBTCorrelationFilter(self.config)
            self.mutual_information = VectorBTMutualInformation(self.config)
            self.stability_selection = VectorBTStabilitySelection(self.config)
            self.mrmr_selector = VectorBTMRMRSelector(self.config)
            self.regularization_selector = VectorBTRegularizationSelector(self.config)
            self.rfe_selector = VectorBTRFESelector(self.config)
            self.feature_selector = VectorBTFeatureSelector(self.config)
            
            tprint_debug("✅ All VectorBT selectors initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize selectors: {e}")
            raise
    
    def _validate_inputs(self, X: np.ndarray, y: np.ndarray, 
                        feature_names: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Validate and prepare inputs."""
        # Validate X
        X = validate_numeric_array(X, name="Feature matrix X")
        if not validate_finite(X):
            raise ValueError("Feature matrix X contains non-finite values")
        
        # Validate y
        y = validate_numeric_array(y, name="Target variable y")
        if not validate_finite(y):
            raise ValueError("Target variable y contains non-finite values")
        
        # Check dimensions
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y must have same number of samples: {X.shape[0]} vs {y.shape[0]}")
        
        # Prepare feature names
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        elif len(feature_names) != X.shape[1]:
            raise ValueError(f"Feature names length {len(feature_names)} doesn't match X shape[1] {X.shape[1]}")
        
        return X, y, feature_names
    
    def _select_method_automatically(self, X: np.ndarray, y: np.ndarray, 
                                   k: int) -> FeatureSelectionMethod:
        """Automatically select the best method based on data characteristics."""
        try:
            n_samples, n_features = X.shape
            
            # Decision tree for method selection
            if n_features <= 50:
                # Small feature set - use comprehensive method
                return FeatureSelectionMethod.COMPREHENSIVE
            elif n_features <= 200:
                # Medium feature set - use mRMR
                return FeatureSelectionMethod.MRMR
            elif n_features <= 1000:
                # Large feature set - use stability selection
                return FeatureSelectionMethod.STABILITY_SELECTION
            else:
                # Very large feature set - use regularization
                return FeatureSelectionMethod.ELASTICNET
            
        except Exception as e:
            self.logger.warning(f"Automatic method selection failed: {e}")
            return FeatureSelectionMethod.COMPREHENSIVE
    
    def select_features(self, X: np.ndarray, y: np.ndarray, 
                       method: Union[str, FeatureSelectionMethod] = 'auto',
                       k: int = None, feature_names: Optional[List[str]] = None,
                       **kwargs) -> SelectionResult:
        """
        Select features using VectorBT-optimized methods.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target variable (n_samples,)
            method: Feature selection method ('auto', 'comprehensive', 'correlation', etc.)
            k: Number of features to select
            feature_names: Optional list of feature names
            **kwargs: Additional method-specific parameters
            
        Returns:
            SelectionResult with selection results
        """
        start_time = time.time()
        
        try:
            # Validate inputs
            X, y, feature_names = self._validate_inputs(X, y, feature_names)
            
            # Determine method
            if method == 'auto':
                method = self._select_method_automatically(X, y, k or 50)
            elif isinstance(method, str):
                try:
                    method = FeatureSelectionMethod(method)
                except ValueError:
                    raise ValueError(f"Unknown method: {method}")
            
            # Set default k if not provided
            if k is None:
                k = min(50, X.shape[1] // 2)
            
            tprint(f"🚀 Starting VectorBT {method.value} selection with {X.shape[1]} features, target: {k}")
            
            # Execute method
            result = self._execute_method(method, X, y, k, feature_names, **kwargs)
            
            # Update performance stats
            execution_time = time.time() - start_time
            self.performance_stats['total_selections'] += 1
            self.performance_stats['total_time'] += execution_time
            
            if result['success']:
                self.performance_stats['successful_selections'] += 1
                self.performance_stats['methods_used'][method.value] = \
                    self.performance_stats['methods_used'].get(method.value, 0) + 1
            else:
                self.performance_stats['failed_selections'] += 1
            
            # Calculate average execution time
            self.performance_stats['avg_execution_time'] = \
                self.performance_stats['total_time'] / self.performance_stats['total_selections']
            
            # Create result object
            selection_result = SelectionResult(
                success=result['success'],
                selected_features=result.get('selected_features', []),
                selected_indices=result.get('selected_indices', []),
                feature_scores=result.get('feature_scores', {}),
                n_selected=result.get('n_selected', 0),
                n_total=result.get('n_total', X.shape[1]),
                method=result.get('method', method.value),
                execution_time=execution_time,
                performance_stats=result.get('performance_stats', {}),
                error=result.get('error'),
                metadata=result.get('metadata', {})
            )
            
            if result['success']:
                tprint_success(f"✅ VectorBT {method.value} completed: {selection_result.n_selected}/{selection_result.n_total} features "
                             f"in {execution_time:.3f}s")
            else:
                tprint_warning(f"⚠️ VectorBT {method.value} failed: {result.get('error', 'Unknown error')}")
            
            return selection_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.performance_stats['total_selections'] += 1
            self.performance_stats['failed_selections'] += 1
            self.performance_stats['total_time'] += execution_time
            
            self.logger.error(f"Feature selection failed: {e}")
            
            return SelectionResult(
                success=False,
                selected_features=[],
                selected_indices=[],
                feature_scores={},
                n_selected=0,
                n_total=X.shape[1] if 'X' in locals() else 0,
                method=method.value if isinstance(method, FeatureSelectionMethod) else str(method),
                execution_time=execution_time,
                performance_stats={},
                error=str(e)
            )
    
    def _execute_method(self, method: FeatureSelectionMethod, X: np.ndarray, y: np.ndarray,
                       k: int, feature_names: List[str], **kwargs) -> Dict[str, Any]:
        """Execute the specified feature selection method."""
        try:
            if method == FeatureSelectionMethod.COMPREHENSIVE:
                return self.feature_selector.comprehensive_feature_selection(
                    X, y, feature_names, method='comprehensive', **kwargs
                )
            elif method == FeatureSelectionMethod.CORRELATION:
                return self.correlation_filter.filter_features(X, **kwargs)
            elif method == FeatureSelectionMethod.MUTUAL_INFORMATION:
                return self.mutual_information.select_features(X, y, k, feature_names, **kwargs)
            elif method == FeatureSelectionMethod.STABILITY_SELECTION:
                return self.stability_selection.select_features(X, y, k=k, feature_names=feature_names, **kwargs)
            elif method == FeatureSelectionMethod.MRMR:
                return self.mrmr_selector.select_features(X, y, k, feature_names, **kwargs)
            elif method == FeatureSelectionMethod.LASSO:
                return self.regularization_selector.select_features_lasso(X, y, k, feature_names, **kwargs)
            elif method == FeatureSelectionMethod.ELASTICNET:
                return self.regularization_selector.select_features_elasticnet(X, y, k, feature_names, **kwargs)
            elif method == FeatureSelectionMethod.RFE:
                return self.rfe_selector.select_features(X, y, k, feature_names, **kwargs)
            elif method == FeatureSelectionMethod.ADAPTIVE:
                return self._execute_adaptive_selection(X, y, k, feature_names, **kwargs)
            else:
                raise ValueError(f"Unknown method: {method}")
                
        except Exception as e:
            self.logger.error(f"Method execution failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'method': method.value
            }
    
    def _execute_adaptive_selection(self, X: np.ndarray, y: np.ndarray, k: int,
                                   feature_names: List[str], **kwargs) -> Dict[str, Any]:
        """Execute adaptive feature selection using multiple methods."""
        try:
            # Try multiple methods and select the best one
            methods_to_try = [
                FeatureSelectionMethod.MRMR,
                FeatureSelectionMethod.STABILITY_SELECTION,
                FeatureSelectionMethod.ELASTICNET
            ]
            
            results = []
            for method in methods_to_try:
                try:
                    result = self._execute_method(method, X, y, k, feature_names, **kwargs)
                    if result['success']:
                        results.append((method, result))
                except Exception as e:
                    self.logger.warning(f"Method {method.value} failed: {e}")
                    continue
            
            if not results:
                return {
                    'success': False,
                    'error': 'All adaptive methods failed',
                    'method': 'adaptive'
                }
            
            # Select best result based on number of features selected
            best_method, best_result = min(results, key=lambda x: abs(x[1]['n_selected'] - k))
            
            best_result['method'] = f'adaptive_{best_method.value}'
            best_result['metadata'] = {
                'methods_tried': [m.value for m, _ in results],
                'best_method': best_method.value
            }
            
            return best_result
            
        except Exception as e:
            self.logger.error(f"Adaptive selection failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'method': 'adaptive'
            }
    
    def benchmark_methods(self, X: np.ndarray, y: np.ndarray, k: int = 50,
                         feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Benchmark all VectorBT methods on the given dataset.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target variable (n_samples,)
            k: Number of features to select
            feature_names: Optional list of feature names
            
        Returns:
            Dictionary with benchmark results
        """
        try:
            # Validate inputs
            X, y, feature_names = self._validate_inputs(X, y, feature_names)
            
            tprint(f"🚀 Starting VectorBT method benchmarking with {X.shape[1]} features, target: {k}")
            
            # Test all methods
            methods_to_test = [
                FeatureSelectionMethod.COMPREHENSIVE,
                FeatureSelectionMethod.CORRELATION,
                FeatureSelectionMethod.MUTUAL_INFORMATION,
                FeatureSelectionMethod.STABILITY_SELECTION,
                FeatureSelectionMethod.MRMR,
                FeatureSelectionMethod.LASSO,
                FeatureSelectionMethod.ELASTICNET,
                FeatureSelectionMethod.RFE
            ]
            
            benchmark_results = {}
            
            for method in methods_to_test:
                try:
                    tprint_debug(f"📊 Testing {method.value}...")
                    
                    start_time = time.time()
                    result = self.select_features(X, y, method, k, feature_names)
                    execution_time = time.time() - start_time
                    
                    benchmark_results[method.value] = {
                        'success': result.success,
                        'n_selected': result.n_selected,
                        'execution_time': execution_time,
                        'error': result.error,
                        'performance_stats': result.performance_stats
                    }
                    
                except Exception as e:
                    benchmark_results[method.value] = {
                        'success': False,
                        'n_selected': 0,
                        'execution_time': 0.0,
                        'error': str(e),
                        'performance_stats': {}
                    }
            
            # Find best method
            successful_results = {k: v for k, v in benchmark_results.items() if v['success']}
            
            if successful_results:
                best_method = min(successful_results.items(), 
                                key=lambda x: abs(x[1]['n_selected'] - k))
                best_method_name = best_method[0]
                best_method_result = best_method[1]
            else:
                best_method_name = None
                best_method_result = None
            
            tprint_success(f"✅ Benchmarking completed. Best method: {best_method_name}")
            
            return {
                'success': True,
                'benchmark_results': benchmark_results,
                'best_method': best_method_name,
                'best_method_result': best_method_result,
                'n_methods_tested': len(methods_to_test),
                'n_successful': len(successful_results)
            }
            
        except Exception as e:
            self.logger.error(f"Benchmarking failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'benchmark_results': {}
            }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = self.performance_stats.copy()
        
        if stats['total_selections'] > 0:
            stats['success_rate'] = stats['successful_selections'] / stats['total_selections']
        else:
            stats['success_rate'] = 0.0
        
        # Add individual selector stats
        stats['selector_stats'] = {
            'correlation_filter': self.correlation_filter.get_performance_stats(),
            'mutual_information': self.mutual_information.get_performance_stats(),
            'stability_selection': self.stability_selection.get_performance_stats(),
            'mrmr_selector': self.mrmr_selector.get_performance_stats(),
            'regularization_selector': self.regularization_selector.get_performance_stats(),
            'rfe_selector': self.rfe_selector.get_performance_stats(),
            'feature_selector': self.feature_selector.get_performance_stats()
        }
        
        tprint_performance(f"📊 VectorBT Unified Framework Stats: {stats['total_selections']} total selections, "
                         f"{stats['success_rate']:.2%} success rate, "
                         f"{stats['avg_execution_time']:.3f}s avg execution time")
        
        return stats


def create_vectorbt_unified_framework(config: Optional[VectorBTFeatureSelectionConfig] = None) -> VectorBTUnifiedFramework:
    """Create a VectorBT unified framework."""
    return VectorBTUnifiedFramework(config)