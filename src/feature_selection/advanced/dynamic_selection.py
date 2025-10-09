"""
Dynamic Feature Selection with Advanced Control

This module implements dynamic feature selection with multiple target
specifications, elbow method, and statistical thresholding.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from scipy import stats
from scipy.stats import permutation_test

# Import project utilities
from src.utils.tprint import tprint, tprint_success, tprint_warning, tprint_performance, tprint_debug
from src.utils.hardware.m1_cpu_optimizer import M1CPUOptimizer
from src.utils.hardware.unified_hardware_manager import UnifiedHardwareManager, HardwareConfig

logger = logging.getLogger(__name__)

class DynamicFeatureSelector:
    """Dynamic feature selector with advanced control options."""
    
    def __init__(self, config, hardware_manager: Optional[UnifiedHardwareManager] = None):
        """Initialize dynamic feature selector."""
        self.config = config
        self.hardware_manager = hardware_manager
        self.logger = logger.getChild('DynamicFeatureSelector')
        
        # Selection tracking
        self.selection_history = []
        self.performance_curves = {}
        self.threshold_history = {}
        
        # Performance tracking
        self.performance_stats = {
            'total_selections': 0,
            'elbow_detections': 0,
            'statistical_thresholds': 0,
            'avg_selection_time': 0.0
        }
        
        tprint_success("🔧 DynamicFeatureSelector initialized")
    
    def determine_target_count(self, X: np.ndarray, y: np.ndarray,
                             target_features: Optional[int] = None,
                             target_percentage: Optional[float] = None,
                             target_performance_threshold: Optional[float] = None,
                             feature_importance: Optional[np.ndarray] = None,
                             feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Determine target feature count using various methods."""
        if not self.config.enable_dynamic_selection:
            # Use default target
            default_target = int(X.shape[1] * self.config.default_target_value)
            return {
                'target_count': default_target,
                'method': 'default',
                'target_value': self.config.default_target_value
            }
        
        tprint_debug("🔧 Determining target feature count")
        
        start_time = time.time()
        
        try:
            # Determine target count based on input parameters
            if target_features is not None:
                target_count = min(target_features, X.shape[1])
                method = 'absolute'
                target_value = target_count
            elif target_percentage is not None:
                target_count = int(X.shape[1] * target_percentage)
                method = 'percentage'
                target_value = target_percentage
            elif target_performance_threshold is not None:
                target_count = self._determine_by_performance_threshold(
                    X, y, target_performance_threshold, feature_importance
                )
                method = 'performance_threshold'
                target_value = target_performance_threshold
            else:
                # Use default method
                if self.config.default_target_type == 'absolute':
                    target_count = int(self.config.default_target_value)
                    method = 'default_absolute'
                    target_value = target_count
                elif self.config.default_target_type == 'percentage':
                    target_count = int(X.shape[1] * self.config.default_target_value)
                    method = 'default_percentage'
                    target_value = self.config.default_target_value
                else:
                    # Use elbow method as default
                    target_count = self._determine_by_elbow_method(X, y, feature_importance)
                    method = 'elbow_method'
                    target_value = target_count
            
            # Apply constraints
            target_count = max(1, min(target_count, X.shape[1]))
            
            # Calculate additional information
            additional_info = self._calculate_additional_info(
                X, y, target_count, feature_importance, feature_names
            )
            
            # Update statistics
            end_time = time.time()
            execution_time = end_time - start_time
            self.performance_stats['total_selections'] += 1
            self.performance_stats['avg_selection_time'] = (
                (self.performance_stats['avg_selection_time'] * (self.performance_stats['total_selections'] - 1) + 
                 execution_time) / self.performance_stats['total_selections']
            )
            
            result = {
                'target_count': target_count,
                'method': method,
                'target_value': target_value,
                'execution_time': execution_time,
                **additional_info
            }
            
            # Store selection history
            self.selection_history.append(result)
            
            tprint_success(f"✅ Target count determined: {target_count} features using {method}")
            return result
            
        except Exception as e:
            self.logger.error(f"Target count determination failed: {e}")
            return {
                'target_count': max(1, X.shape[1] // 2),
                'method': 'fallback',
                'target_value': 0.5,
                'error': str(e)
            }
    
    def _determine_by_performance_threshold(self, X: np.ndarray, y: np.ndarray,
                                          threshold: float,
                                          feature_importance: Optional[np.ndarray] = None) -> int:
        """Determine target count based on performance threshold."""
        tprint_debug("🔧 Determining target count by performance threshold")
        
        try:
            if feature_importance is None:
                # Calculate feature importance using RandomForest
                from sklearn.ensemble import RandomForestRegressor
                rf = RandomForestRegressor(n_estimators=50, random_state=42)
                rf.fit(X, y)
                feature_importance = rf.feature_importances_
            
            # Sort features by importance
            sorted_indices = np.argsort(feature_importance)[::-1]
            
            # Find cutoff where performance drops below threshold
            target_count = X.shape[1]
            for i in range(1, X.shape[1]):
                # Calculate performance with top i features
                X_subset = X[:, sorted_indices[:i]]
                performance = self._calculate_subset_performance(X_subset, y)
                
                if performance < threshold:
                    target_count = i - 1
                    break
            
            return max(1, target_count)
            
        except Exception as e:
            self.logger.warning(f"Performance threshold determination failed: {e}")
            return max(1, X.shape[1] // 2)
    
    def _determine_by_elbow_method(self, X: np.ndarray, y: np.ndarray,
                                 feature_importance: Optional[np.ndarray] = None) -> int:
        """Determine target count using elbow method."""
        if not self.config.elbow_method.enable_elbow_method:
            return max(1, X.shape[1] // 2)
        
        tprint_debug("🔧 Determining target count by elbow method")
        
        try:
            if feature_importance is None:
                # Calculate feature importance using RandomForest
                from sklearn.ensemble import RandomForestRegressor
                rf = RandomForestRegressor(n_estimators=50, random_state=42)
                rf.fit(X, y)
                feature_importance = rf.feature_importances_
            
            # Sort features by importance
            sorted_indices = np.argsort(feature_importance)[::-1]
            
            # Calculate performance curve
            performance_curve = self._calculate_performance_curve(X, y, sorted_indices)
            
            # Find elbow point
            elbow_point = self._find_elbow_point(performance_curve)
            
            # Store performance curve
            self.performance_curves[len(self.selection_history)] = performance_curve
            
            self.performance_stats['elbow_detections'] += 1
            
            return max(1, min(elbow_point, X.shape[1]))
            
        except Exception as e:
            self.logger.warning(f"Elbow method determination failed: {e}")
            return max(1, X.shape[1] // 2)
    
    def _calculate_performance_curve(self, X: np.ndarray, y: np.ndarray,
                                   sorted_indices: np.ndarray) -> List[float]:
        """Calculate performance curve for different feature counts."""
        performance_curve = []
        
        min_features = self.config.elbow_method.min_features
        max_features = min(self.config.elbow_method.max_features, X.shape[1])
        step_size = self.config.elbow_method.step_size
        
        for n_features in range(min_features, max_features + 1, step_size):
            X_subset = X[:, sorted_indices[:n_features]]
            performance = self._calculate_subset_performance(X_subset, y)
            performance_curve.append(performance)
        
        return performance_curve
    
    def _calculate_subset_performance(self, X_subset: np.ndarray, y: np.ndarray) -> float:
        """Calculate performance for a feature subset."""
        try:
            # Use cross-validation for robust performance estimation
            from sklearn.linear_model import LinearRegression
            from sklearn.model_selection import cross_val_score
            
            model = LinearRegression()
            scores = cross_val_score(model, X_subset, y, cv=3, scoring='r2')
            return np.mean(scores)
            
        except Exception as e:
            self.logger.warning(f"Subset performance calculation failed: {e}")
            return 0.0
    
    def _find_elbow_point(self, performance_curve: List[float]) -> int:
        """Find elbow point in performance curve."""
        if len(performance_curve) < 3:
            return len(performance_curve)
        
        try:
            if self.config.elbow_method.elbow_detection_method == 'curvature':
                return self._find_curvature_elbow(performance_curve)
            elif self.config.elbow_method.elbow_detection_method == 'knee':
                return self._find_knee_point(performance_curve)
            else:  # elbow
                return self._find_elbow_point_simple(performance_curve)
                
        except Exception as e:
            self.logger.warning(f"Elbow point detection failed: {e}")
            return len(performance_curve) // 2
    
    def _find_curvature_elbow(self, performance_curve: List[float]) -> int:
        """Find elbow point using curvature method."""
        # Calculate second derivative (curvature)
        if len(performance_curve) < 3:
            return len(performance_curve)
        
        # First derivative
        first_deriv = np.diff(performance_curve)
        
        # Second derivative
        second_deriv = np.diff(first_deriv)
        
        # Find point with maximum curvature
        if len(second_deriv) > 0:
            elbow_idx = np.argmax(second_deriv) + 2  # +2 because of double differentiation
        else:
            elbow_idx = len(performance_curve) // 2
        
        return min(elbow_idx, len(performance_curve))
    
    def _find_knee_point(self, performance_curve: List[float]) -> int:
        """Find knee point in performance curve."""
        if len(performance_curve) < 3:
            return len(performance_curve)
        
        # Calculate distance from line connecting first and last points
        n_points = len(performance_curve)
        x = np.arange(n_points)
        y = np.array(performance_curve)
        
        # Line from first to last point
        x1, y1 = 0, y[0]
        x2, y2 = n_points - 1, y[-1]
        
        # Calculate distance from each point to the line
        distances = []
        for i in range(n_points):
            # Distance from point to line
            distance = abs((y2 - y1) * x[i] - (x2 - x1) * y[i] + x2 * y1 - y2 * x1) / np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
            distances.append(distance)
        
        # Find point with maximum distance
        knee_idx = np.argmax(distances)
        return min(knee_idx + 1, n_points)
    
    def _find_elbow_point_simple(self, performance_curve: List[float]) -> int:
        """Find elbow point using simple method."""
        if len(performance_curve) < 3:
            return len(performance_curve)
        
        # Calculate improvement rate
        improvements = np.diff(performance_curve)
        
        # Find point where improvement rate drops significantly
        if len(improvements) > 1:
            improvement_ratio = improvements[1:] / (improvements[:-1] + 1e-8)
            elbow_idx = np.argmin(improvement_ratio) + 2
        else:
            elbow_idx = len(performance_curve) // 2
        
        return min(elbow_idx, len(performance_curve))
    
    def calculate_statistical_threshold(self, feature_importance: np.ndarray,
                                      feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Calculate statistical significance threshold for feature importance."""
        if not self.config.statistical_thresholding.enable_statistical_thresholding:
            return {'threshold': 0.0, 'significant_features': [], 'method': 'disabled'}
        
        tprint_debug("🔧 Calculating statistical threshold")
        
        try:
            if self.config.statistical_thresholding.test_method == 'permutation':
                return self._calculate_permutation_threshold(feature_importance, feature_names)
            elif self.config.statistical_thresholding.test_method == 'bootstrap':
                return self._calculate_bootstrap_threshold(feature_importance, feature_names)
            else:  # t_test
                return self._calculate_t_test_threshold(feature_importance, feature_names)
                
        except Exception as e:
            self.logger.error(f"Statistical threshold calculation failed: {e}")
            return {'threshold': 0.0, 'significant_features': [], 'method': 'error', 'error': str(e)}
    
    def _calculate_permutation_threshold(self, feature_importance: np.ndarray,
                                       feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Calculate threshold using permutation test."""
        try:
            n_permutations = self.config.statistical_thresholding.n_permutations
            significance_level = self.config.statistical_thresholding.significance_level
            
            # Perform permutation test
            def statistic(x, y):
                return np.mean(x) - np.mean(y)
            
            # Create null distribution by permuting importance values
            null_distribution = []
            for _ in range(n_permutations):
                permuted_importance = np.random.permutation(feature_importance)
                null_distribution.append(np.mean(permuted_importance))
            
            null_distribution = np.array(null_distribution)
            
            # Calculate threshold
            threshold = np.percentile(null_distribution, (1 - significance_level) * 100)
            
            # Find significant features
            significant_indices = np.where(feature_importance > threshold)[0]
            significant_features = [feature_names[i] for i in significant_indices] if feature_names else significant_indices.tolist()
            
            self.performance_stats['statistical_thresholds'] += 1
            
            return {
                'threshold': float(threshold),
                'significant_features': significant_features,
                'n_significant': len(significant_features),
                'method': 'permutation',
                'significance_level': significance_level,
                'n_permutations': n_permutations
            }
            
        except Exception as e:
            self.logger.warning(f"Permutation threshold calculation failed: {e}")
            return {'threshold': 0.0, 'significant_features': [], 'method': 'permutation', 'error': str(e)}
    
    def _calculate_bootstrap_threshold(self, feature_importance: np.ndarray,
                                     feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Calculate threshold using bootstrap method."""
        try:
            n_bootstrap = self.config.statistical_thresholding.n_permutations
            significance_level = self.config.statistical_thresholding.significance_level
            
            # Bootstrap sampling
            bootstrap_means = []
            for _ in range(n_bootstrap):
                bootstrap_sample = np.random.choice(feature_importance, size=len(feature_importance), replace=True)
                bootstrap_means.append(np.mean(bootstrap_sample))
            
            bootstrap_means = np.array(bootstrap_means)
            
            # Calculate confidence interval
            lower_bound = np.percentile(bootstrap_means, significance_level * 100)
            upper_bound = np.percentile(bootstrap_means, (1 - significance_level) * 100)
            
            # Use upper bound as threshold
            threshold = upper_bound
            
            # Find significant features
            significant_indices = np.where(feature_importance > threshold)[0]
            significant_features = [feature_names[i] for i in significant_indices] if feature_names else significant_indices.tolist()
            
            self.performance_stats['statistical_thresholds'] += 1
            
            return {
                'threshold': float(threshold),
                'significant_features': significant_features,
                'n_significant': len(significant_features),
                'method': 'bootstrap',
                'significance_level': significance_level,
                'n_bootstrap': n_bootstrap,
                'confidence_interval': [float(lower_bound), float(upper_bound)]
            }
            
        except Exception as e:
            self.logger.warning(f"Bootstrap threshold calculation failed: {e}")
            return {'threshold': 0.0, 'significant_features': [], 'method': 'bootstrap', 'error': str(e)}
    
    def _calculate_t_test_threshold(self, feature_importance: np.ndarray,
                                  feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Calculate threshold using t-test method."""
        try:
            significance_level = self.config.statistical_thresholding.significance_level
            
            # Calculate t-statistic threshold
            mean_importance = np.mean(feature_importance)
            std_importance = np.std(feature_importance)
            n_features = len(feature_importance)
            
            # Calculate t-value for given significance level
            t_value = stats.t.ppf(1 - significance_level/2, n_features - 1)
            
            # Calculate threshold
            threshold = mean_importance + t_value * (std_importance / np.sqrt(n_features))
            
            # Find significant features
            significant_indices = np.where(feature_importance > threshold)[0]
            significant_features = [feature_names[i] for i in significant_indices] if feature_names else significant_indices.tolist()
            
            self.performance_stats['statistical_thresholds'] += 1
            
            return {
                'threshold': float(threshold),
                'significant_features': significant_features,
                'n_significant': len(significant_features),
                'method': 't_test',
                'significance_level': significance_level,
                't_value': float(t_value),
                'mean_importance': float(mean_importance),
                'std_importance': float(std_importance)
            }
            
        except Exception as e:
            self.logger.warning(f"T-test threshold calculation failed: {e}")
            return {'threshold': 0.0, 'significant_features': [], 'method': 't_test', 'error': str(e)}
    
    def _calculate_additional_info(self, X: np.ndarray, y: np.ndarray,
                                 target_count: int,
                                 feature_importance: Optional[np.ndarray] = None,
                                 feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Calculate additional information about the selection."""
        additional_info = {
            'total_features': X.shape[1],
            'selection_ratio': target_count / X.shape[1],
            'reduction_ratio': 1 - (target_count / X.shape[1])
        }
        
        # Add statistical threshold information
        if feature_importance is not None and self.config.statistical_thresholding.enable_statistical_thresholding:
            threshold_info = self.calculate_statistical_threshold(feature_importance, feature_names)
            additional_info['statistical_threshold'] = threshold_info
        
        # Add elbow method information
        if self.config.elbow_method.enable_elbow_method and len(self.performance_curves) > 0:
            latest_curve = list(self.performance_curves.values())[-1]
            additional_info['performance_curve'] = latest_curve
            additional_info['performance_curve_length'] = len(latest_curve)
        
        return additional_info
    
    def get_selection_statistics(self) -> Dict[str, Any]:
        """Get selection statistics."""
        stats = self.performance_stats.copy()
        
        # Add selection history insights
        if self.selection_history:
            stats['selection_history_size'] = len(self.selection_history)
            stats['avg_target_count'] = np.mean([s['target_count'] for s in self.selection_history])
            stats['target_count_std'] = np.std([s['target_count'] for s in self.selection_history])
            
            # Method usage statistics
            methods = [s['method'] for s in self.selection_history]
            method_counts = {}
            for method in methods:
                method_counts[method] = method_counts.get(method, 0) + 1
            stats['method_usage'] = method_counts
        
        return stats
    
    def get_selection_insights(self) -> Dict[str, Any]:
        """Get insights about selection behavior."""
        insights = {
            'total_selections': self.performance_stats['total_selections'],
            'avg_selection_time': self.performance_stats['avg_selection_time'],
            'elbow_detections': self.performance_stats['elbow_detections'],
            'statistical_thresholds': self.performance_stats['statistical_thresholds'],
            'selection_trends': {}
        }
        
        if self.selection_history:
            # Analyze selection trends
            target_counts = [s['target_count'] for s in self.selection_history]
            if len(target_counts) > 1:
                insights['selection_trends']['avg_target_count'] = float(np.mean(target_counts))
                insights['selection_trends']['target_count_std'] = float(np.std(target_counts))
                insights['selection_trends']['trend'] = 'increasing' if target_counts[-1] > target_counts[0] else 'decreasing' if target_counts[-1] < target_counts[0] else 'stable'
            
            # Method usage analysis
            methods = [s['method'] for s in self.selection_history]
            method_counts = {}
            for method in methods:
                method_counts[method] = method_counts.get(method, 0) + 1
            insights['method_usage'] = method_counts
            insights['most_used_method'] = max(method_counts.items(), key=lambda x: x[1])[0] if method_counts else 'none'
        
        return insights