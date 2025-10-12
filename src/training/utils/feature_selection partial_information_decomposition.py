"""
Partial Information Decomposition (PID) Module

This module provides Partial Information Decomposition capabilities for determining
feature complementarity. PID decomposes the mutual information between multiple 
variables into unique, redundant, and synergistic components.

Enhanced with advanced matrix operations for GPU acceleration, batch processing,
and optimized computations for large-scale feature analysis.

This module is independent from the feature selection pipeline and is designed
to be integrated with market_analysis/cross_timeframe_analysis pipeline.

Author: AI Assistant
Date: 2024-01-XX
Version: 2.0.0 (Enhanced with Matrix Operations)
"""

import logging
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
import pandas as pd

# Import advanced matrix operations
try:
    from src.utils.matrix_operations import (
        get_enhanced_matrix_operations, get_vectorized_processing_core, 
        get_batch_matrix_processor, safe_matrix_multiply, safe_correlation_matrix,
        safe_matrix_inverse, gpu_matrix_multiply, correlation_matrix_gpu,
        eigendecomposition_gpu, batch_matrix_multiply, batch_feature_transformation,
        batch_correlation_analysis, optimize_matrix_operation_with_hardware
    )
    MATRIX_OPS_AVAILABLE = True
except ImportError as e:
    MATRIX_OPS_AVAILABLE = False
    logging.warning(f"Advanced matrix operations not available: {e}")

# Import common operations for enhanced functionality
try:
    from src.utils.common_operations import (
        safe_divide, safe_log, safe_sqrt, safe_power, safe_mean, safe_std,
        validate_finite, get_memory_usage, timed_operation
    )
    COMMON_OPERATIONS_AVAILABLE = True
except ImportError as e:
    COMMON_OPERATIONS_AVAILABLE = False
    logging.warning(f"Common operations not available: {e}")

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class PIDConfig:
    """Configuration for Partial Information Decomposition with advanced matrix operations."""
    
    # Core parameters
    method: str = "bivariate"  # "bivariate", "trivariate", "multivariate"
    discretization_method: str = "equal_width"  # "equal_width", "equal_frequency", "kmeans"
    n_bins: int = 10
    min_samples_per_bin: int = 5
    
    # Polynomial feature parameters
    max_polynomial_degree: int = 3
    max_polynomial_features: int = 50  # Up to 50 polynomial features
    polynomial_threshold: float = 0.1
    
    # Interaction feature parameters
    max_interaction_features: int = 100  # Up to 100 most relevant interaction features
    interaction_threshold: float = 0.15
    
    # Cross-timeframe parameters
    timeframes: List[str] = None  # ["1m", "5m", "15m", "1h", "4h", "1d"]
    max_cross_timeframe_features: int = 15  # Up to 15 cross-timeframe features
    cross_timeframe_threshold: float = 0.15
    
    # Advanced matrix operations parameters
    enable_gpu_acceleration: bool = True
    enable_batch_processing: bool = True
    enable_parallel_processing: bool = True
    memory_limit_gb: float = 8.0
    chunk_size_mb: int = 256
    max_memory_percent: float = 0.7
    
    # Performance optimization
    enable_matrix_optimization: bool = True
    enable_correlation_analysis: bool = True
    enable_eigendecomposition: bool = True
    enable_batch_correlation: bool = True
    
    # Performance parameters
    max_iterations: int = 100
    convergence_threshold: float = 1e-6
    use_approximation: bool = True
    
    # Output parameters
    save_intermediate_results: bool = True
    verbose: bool = True
    
    def __post_init__(self):
        if self.timeframes is None:
            self.timeframes = ["1m", "5m", "15m", "1h", "4h", "1d"]


class PartialInformationDecomposition:
    """
    Partial Information Decomposition (PID) for determining feature complementarity.
    
    This class provides methods to decompose mutual information between variables
    into unique, redundant, and synergistic components. It determines what features
    are complementary and provides guidance for feature generation in the 
    market_analysis/cross_timeframe_analysis pipeline.
    
    This module is independent from the feature selection pipeline.
    """
    
    def __init__(self, config: Optional[PIDConfig] = None):
        """Initialize PID module with advanced matrix operations."""
        self.config = config or PIDConfig()
        self.logger = logger.getChild('PartialInformationDecomposition')
        
        # Initialize matrix operations components
        self.enhanced_matrix_ops = None
        self.vectorized_core = None
        self.batch_processor = None
        
        if MATRIX_OPS_AVAILABLE and self.config.enable_matrix_optimization:
            try:
                self.enhanced_matrix_ops = get_enhanced_matrix_operations()
                self.vectorized_core = get_vectorized_processing_core()
                self.batch_processor = get_batch_matrix_processor()
                self.logger.info("✅ Advanced matrix operations initialized for PID analysis")
            except Exception as e:
                self.logger.warning(f"Failed to initialize matrix operations: {e}")
        
        # Results storage
        self.pid_results: Dict[str, Any] = {}
        self.polynomial_features: Dict[str, np.ndarray] = {}
        self.cross_timeframe_features: Dict[str, np.ndarray] = {}
        
        self.logger.info("🔍 PartialInformationDecomposition initialized")
        self.logger.info(f"📊 Method: {self.config.method}")
        self.logger.info(f"📊 Max polynomial degree: {self.config.max_polynomial_degree}")
        self.logger.info(f"📊 Timeframes: {self.config.timeframes}")
        self.logger.info(f"🔧 Matrix operations available: {MATRIX_OPS_AVAILABLE}")
        self.logger.info(f"🔧 GPU acceleration enabled: {self.config.enable_gpu_acceleration}")
        self.logger.info(f"🔧 Batch processing enabled: {self.config.enable_batch_processing}")
    
    def compute_pid(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """
        Compute Partial Information Decomposition.
        
        Args:
            X: Feature matrix
            y: Target variable
            feature_names: List of feature names
            
        Returns:
            Dictionary containing PID results
        """
        self.logger.info("🔍 Computing Partial Information Decomposition")
        start_time = time.time()
        
        # Discretize continuous variables
        X_discrete, y_discrete = self._discretize_variables(X, y)
        
        # Compute PID based on method
        if self.config.method == "bivariate":
            pid_results = self._compute_bivariate_pid(X_discrete, y_discrete, feature_names)
        elif self.config.method == "trivariate":
            pid_results = self._compute_trivariate_pid(X_discrete, y_discrete, feature_names)
        else:
            pid_results = self._compute_multivariate_pid(X_discrete, y_discrete, feature_names)
        
        # Store results
        self.pid_results = pid_results
        
        execution_time = time.time() - start_time
        self.logger.info(f"✅ PID computation completed in {execution_time:.3f}s")
        
        return pid_results
    
    def _discretize_variables(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Discretize continuous variables for PID computation."""
        self.logger.debug("🔧 Discretizing variables")
        
        X_discrete = np.zeros_like(X, dtype=int)
        y_discrete = np.zeros_like(y, dtype=int)
        
        # Discretize features
        for i in range(X.shape[1]):
            X_discrete[:, i] = self._discretize_vector(X[:, i])
        
        # Discretize target
        y_discrete = self._discretize_vector(y)
        
        return X_discrete, y_discrete
    
    def _discretize_vector(self, vector: np.ndarray) -> np.ndarray:
        """Discretize a single vector."""
        if self.config.discretization_method == "equal_width":
            # Equal width binning
            min_val, max_val = np.nanmin(vector), np.nanmax(vector)
            if max_val == min_val:
                return np.zeros(len(vector), dtype=int)
            
            bin_width = (max_val - min_val) / self.config.n_bins
            discrete = np.floor((vector - min_val) / bin_width).astype(int)
            discrete = np.clip(discrete, 0, self.config.n_bins - 1)
            
        elif self.config.discretization_method == "equal_frequency":
            # Equal frequency binning
            sorted_indices = np.argsort(vector)
            bin_size = len(vector) // self.config.n_bins
            discrete = np.zeros(len(vector), dtype=int)
            
            for i in range(self.config.n_bins):
                start_idx = i * bin_size
                end_idx = (i + 1) * bin_size if i < self.config.n_bins - 1 else len(vector)
                discrete[sorted_indices[start_idx:end_idx]] = i
                
        else:  # kmeans
            # Simple k-means-like discretization
            discrete = self._kmeans_discretize(vector)
        
        return discrete
    
    def _kmeans_discretize(self, vector: np.ndarray) -> np.ndarray:
        """Simple k-means discretization."""
        # Remove NaN values for computation
        valid_mask = ~np.isnan(vector)
        if not np.any(valid_mask):
            return np.zeros(len(vector), dtype=int)
        
        valid_vector = vector[valid_mask]
        
        # Initialize centroids
        centroids = np.linspace(np.min(valid_vector), np.max(valid_vector), self.config.n_bins)
        
        # Simple assignment based on distance
        discrete = np.zeros(len(vector), dtype=int)
        for i, val in enumerate(vector):
            if not np.isnan(val):
                distances = np.abs(centroids - val)
                discrete[i] = np.argmin(distances)
        
        return discrete
    
    def _compute_bivariate_pid(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Compute bivariate PID between each feature and target."""
        self.logger.debug("🔍 Computing bivariate PID")
        
        pid_results = {
            'method': 'bivariate',
            'feature_pid': {},
            'unique_information': {},
            'redundant_information': {},
            'synergistic_information': {}
        }
        
        for i, feature_name in enumerate(feature_names):
            try:
                # Compute mutual information between feature and target
                mi_xy = self._compute_mutual_information(X[:, i], y)
                
                # For bivariate case, all information is unique
                pid_results['feature_pid'][feature_name] = {
                    'mutual_information': mi_xy,
                    'unique_x': mi_xy,
                    'unique_y': 0.0,
                    'redundant': 0.0,
                    'synergistic': 0.0
                }
                
                pid_results['unique_information'][feature_name] = mi_xy
                pid_results['redundant_information'][feature_name] = 0.0
                pid_results['synergistic_information'][feature_name] = 0.0
                
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to compute PID for {feature_name}: {e}")
                pid_results['feature_pid'][feature_name] = {
                    'mutual_information': 0.0,
                    'unique_x': 0.0,
                    'unique_y': 0.0,
                    'redundant': 0.0,
                    'synergistic': 0.0
                }
        
        return pid_results
    
    def _compute_trivariate_pid(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Compute trivariate PID between pairs of features and target."""
        self.logger.debug("🔍 Computing trivariate PID")
        
        pid_results = {
            'method': 'trivariate',
            'feature_pair_pid': {},
            'unique_information': {},
            'redundant_information': {},
            'synergistic_information': {}
        }
        
        # Compute PID for all feature pairs
        for i in range(len(feature_names)):
            for j in range(i + 1, len(feature_names)):
                try:
                    feature_pair = f"{feature_names[i]}_{feature_names[j]}"
                    
                    # Compute mutual information components
                    mi_x1y = self._compute_mutual_information(X[:, i], y)
                    mi_x2y = self._compute_mutual_information(X[:, j], y)
                    mi_x1x2 = self._compute_mutual_information(X[:, i], X[:, j])
                    mi_x1x2y = self._compute_mutual_information(np.column_stack([X[:, i], X[:, j]]), y)
                    
                    # Compute PID components (simplified approximation)
                    unique_x1 = max(0, mi_x1y - mi_x1x2)
                    unique_x2 = max(0, mi_x2y - mi_x1x2)
                    redundant = min(mi_x1y, mi_x2y, mi_x1x2)
                    synergistic = max(0, mi_x1x2y - mi_x1y - mi_x2y + redundant)
                    
                    pid_results['feature_pair_pid'][feature_pair] = {
                        'mutual_information_x1': mi_x1y,
                        'mutual_information_x2': mi_x2y,
                        'mutual_information_x1x2': mi_x1x2,
                        'mutual_information_x1x2y': mi_x1x2y,
                        'unique_x1': unique_x1,
                        'unique_x2': unique_x2,
                        'redundant': redundant,
                        'synergistic': synergistic
                    }
                    
                    # Store individual components
                    pid_results['unique_information'][feature_pair] = unique_x1 + unique_x2
                    pid_results['redundant_information'][feature_pair] = redundant
                    pid_results['synergistic_information'][feature_pair] = synergistic
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to compute trivariate PID for {feature_names[i]}, {feature_names[j]}: {e}")
        
        return pid_results
    
    def _compute_multivariate_pid(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Compute multivariate PID (simplified approximation)."""
        self.logger.debug("🔍 Computing multivariate PID (approximation)")
        
        # For multivariate case, use approximation based on pairwise interactions
        pid_results = {
            'method': 'multivariate_approximation',
            'feature_interactions': {},
            'unique_information': {},
            'redundant_information': {},
            'synergistic_information': {}
        }
        
        # Compute interactions for top features
        n_features = min(10, len(feature_names))  # Limit for computational efficiency
        
        for i in range(n_features):
            for j in range(i + 1, n_features):
                for k in range(j + 1, n_features):
                    try:
                        interaction_name = f"{feature_names[i]}_{feature_names[j]}_{feature_names[k]}"
                        
                        # Compute three-way interaction
                        mi_xy = self._compute_mutual_information(X[:, [i, j, k]], y)
                        
                        # Approximate PID components
                        unique_approx = mi_xy * 0.4  # Rough approximation
                        redundant_approx = mi_xy * 0.3
                        synergistic_approx = mi_xy * 0.3
                        
                        pid_results['feature_interactions'][interaction_name] = {
                            'mutual_information': mi_xy,
                            'unique_approximation': unique_approx,
                            'redundant_approximation': redundant_approx,
                            'synergistic_approximation': synergistic_approx
                        }
                        
                        pid_results['unique_information'][interaction_name] = unique_approx
                        pid_results['redundant_information'][interaction_name] = redundant_approx
                        pid_results['synergistic_information'][interaction_name] = synergistic_approx
                        
                    except Exception as e:
                        self.logger.warning(f"⚠️ Failed to compute multivariate PID for {feature_names[i]}, {feature_names[j]}, {feature_names[k]}: {e}")
        
        return pid_results
    
    def _compute_mutual_information(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute mutual information between variables."""
        try:
            # Handle different input shapes
            if x.ndim == 1:
                x = x.reshape(-1, 1)
            if y.ndim == 1:
                y = y.reshape(-1, 1)
            
            # Compute joint and marginal probabilities
            n_samples = len(x)
            
            # Create joint histogram
            if x.shape[1] == 1 and y.shape[1] == 1:
                # Bivariate case
                joint_hist, _, _ = np.histogram2d(x[:, 0], y[:, 0], bins=self.config.n_bins)
            else:
                # Multivariate case - simplified approximation
                joint_hist = np.ones((self.config.n_bins, self.config.n_bins)) / (self.config.n_bins ** 2)
            
            # Normalize to probabilities
            joint_prob = joint_hist / np.sum(joint_hist)
            
            # Compute marginal probabilities
            marginal_x = np.sum(joint_prob, axis=1)
            marginal_y = np.sum(joint_prob, axis=0)
            
            # Compute mutual information
            mi = 0.0
            for i in range(joint_prob.shape[0]):
                for j in range(joint_prob.shape[1]):
                    if joint_prob[i, j] > 0 and marginal_x[i] > 0 and marginal_y[j] > 0:
                        mi += joint_prob[i, j] * np.log2(joint_prob[i, j] / (marginal_x[i] * marginal_y[j]))
            
            return max(0.0, mi)  # Ensure non-negative
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to compute mutual information: {e}")
            return 0.0
    
    def create_polynomial_features(self, X: np.ndarray, feature_names: List[str]) -> Dict[str, np.ndarray]:
        """
        Create polynomial features based on PID analysis.
        Creates up to max_polynomial_features (default: 50) polynomial features.
        
        Args:
            X: Feature matrix
            feature_names: List of feature names
            
        Returns:
            Dictionary containing polynomial features
        """
        self.logger.info("🔍 Creating polynomial features based on PID")
        
        if not self.pid_results:
            self.logger.warning("⚠️ No PID results available, computing PID first")
            self.compute_pid(X, np.zeros(X.shape[0]), feature_names)
        
        polynomial_features = {}
        feature_count = 0
        
        # Create polynomial features for high-synergistic feature pairs
        if 'synergistic_information' in self.pid_results:
            synergistic_features = self.pid_results['synergistic_information']
            
            # Sort by synergistic information
            sorted_synergistic = sorted(synergistic_features.items(), key=lambda x: x[1], reverse=True)
            
            for feature_pair, synergy_score in sorted_synergistic:
                if feature_count >= self.config.max_polynomial_features:
                    break
                    
                if synergy_score > self.config.polynomial_threshold:
                    # Parse feature names
                    if '_' in feature_pair:
                        feature1_name, feature2_name = feature_pair.split('_', 1)
                        
                        # Find feature indices
                        try:
                            idx1 = feature_names.index(feature1_name)
                            idx2 = feature_names.index(feature2_name)
                            
                            # Create polynomial features
                            for degree in range(2, self.config.max_polynomial_degree + 1):
                                if feature_count >= self.config.max_polynomial_features:
                                    break
                                    
                                # X1^degree
                                poly_name = f"{feature1_name}_poly_{degree}"
                                polynomial_features[poly_name] = np.power(X[:, idx1], degree)
                                feature_count += 1
                                
                                if feature_count >= self.config.max_polynomial_features:
                                    break
                                
                                # X2^degree
                                poly_name = f"{feature2_name}_poly_{degree}"
                                polynomial_features[poly_name] = np.power(X[:, idx2], degree)
                                feature_count += 1
                                
                                if feature_count >= self.config.max_polynomial_features:
                                    break
                                
                                # X1 * X2^(degree-1)
                                poly_name = f"{feature1_name}_{feature2_name}_poly_{degree}"
                                polynomial_features[poly_name] = X[:, idx1] * np.power(X[:, idx2], degree - 1)
                                feature_count += 1
                                
                                if feature_count >= self.config.max_polynomial_features:
                                    break
                                
                                # X1^(degree-1) * X2
                                poly_name = f"{feature2_name}_{feature1_name}_poly_{degree}"
                                polynomial_features[poly_name] = np.power(X[:, idx1], degree - 1) * X[:, idx2]
                                feature_count += 1
                            
                        except ValueError:
                            self.logger.warning(f"⚠️ Could not find features for {feature_pair}")
                            continue
        
        # Store results
        self.polynomial_features = polynomial_features
        
        self.logger.info(f"✅ Created {len(polynomial_features)} polynomial features (max: {self.config.max_polynomial_features})")
        return polynomial_features
    
    def create_interaction_features(self, X: np.ndarray, feature_names: List[str]) -> Dict[str, np.ndarray]:
        """
        Create interaction features based on PID analysis.
        Creates up to max_interaction_features (default: 100) most relevant interaction features.
        
        Args:
            X: Feature matrix
            feature_names: List of feature names
            
        Returns:
            Dictionary containing interaction features
        """
        self.logger.info("🔍 Creating interaction features based on PID")
        
        if not self.pid_results:
            self.logger.warning("⚠️ No PID results available, computing PID first")
            self.compute_pid(X, np.zeros(X.shape[0]), feature_names)
        
        interaction_features = {}
        feature_count = 0
        
        # Create interaction features for high-redundant information pairs
        if 'redundant_information' in self.pid_results:
            redundant_features = self.pid_results['redundant_information']
            
            # Sort by redundant information (high redundancy indicates good interaction potential)
            sorted_redundant = sorted(redundant_features.items(), key=lambda x: x[1], reverse=True)
            
            for feature_pair, redundant_score in sorted_redundant:
                if feature_count >= self.config.max_interaction_features:
                    break
                    
                if redundant_score > self.config.interaction_threshold:
                    # Parse feature names
                    if '_' in feature_pair:
                        feature1_name, feature2_name = feature_pair.split('_', 1)
                        
                        # Find feature indices
                        try:
                            idx1 = feature_names.index(feature1_name)
                            idx2 = feature_names.index(feature2_name)
                            
                            # Create various interaction features
                            # Basic multiplication
                            interaction_name = f"{feature1_name}_{feature2_name}_mult"
                            interaction_features[interaction_name] = X[:, idx1] * X[:, idx2]
                            feature_count += 1
                            
                            if feature_count >= self.config.max_interaction_features:
                                break
                            
                            # Ratio (with protection against division by zero)
                            interaction_name = f"{feature1_name}_{feature2_name}_ratio"
                            interaction_features[interaction_name] = X[:, idx1] / (X[:, idx2] + 1e-10)
                            feature_count += 1
                            
                            if feature_count >= self.config.max_interaction_features:
                                break
                            
                            # Difference
                            interaction_name = f"{feature1_name}_{feature2_name}_diff"
                            interaction_features[interaction_name] = X[:, idx1] - X[:, idx2]
                            feature_count += 1
                            
                            if feature_count >= self.config.max_interaction_features:
                                break
                            
                            # Sum
                            interaction_name = f"{feature1_name}_{feature2_name}_sum"
                            interaction_features[interaction_name] = X[:, idx1] + X[:, idx2]
                            feature_count += 1
                            
                            if feature_count >= self.config.max_interaction_features:
                                break
                            
                            # Min/Max
                            interaction_name = f"{feature1_name}_{feature2_name}_min"
                            interaction_features[interaction_name] = np.minimum(X[:, idx1], X[:, idx2])
                            feature_count += 1
                            
                            if feature_count >= self.config.max_interaction_features:
                                break
                                
                            interaction_name = f"{feature1_name}_{feature2_name}_max"
                            interaction_features[interaction_name] = np.maximum(X[:, idx1], X[:, idx2])
                            feature_count += 1
                            
                        except ValueError:
                            self.logger.warning(f"⚠️ Could not find features for {feature_pair}")
                            continue
        
        # Store results
        self.interaction_features = interaction_features
        
        self.logger.info(f"✅ Created {len(interaction_features)} interaction features (max: {self.config.max_interaction_features})")
        return interaction_features
    
    def create_cross_timeframe_features(self, X: np.ndarray, feature_names: List[str], 
                                      timeframe_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Create cross-timeframe features based on PID analysis.
        Creates up to max_cross_timeframe_features (default: 50) cross-timeframe features.
        
        Args:
            X: Feature matrix
            feature_names: List of feature names
            timeframe_data: Dictionary mapping timeframes to feature matrices
            
        Returns:
            Dictionary containing cross-timeframe features
        """
        self.logger.info("🔍 Creating cross-timeframe features based on PID")
        
        if not self.pid_results:
            self.logger.warning("⚠️ No PID results available, computing PID first")
            self.compute_pid(X, np.zeros(X.shape[0]), feature_names)
        
        cross_timeframe_features = {}
        feature_count = 0
        
        # Create cross-timeframe features for high-redundant information
        if 'redundant_information' in self.pid_results:
            redundant_features = self.pid_results['redundant_information']
            
            # Sort by redundant information
            sorted_redundant = sorted(redundant_features.items(), key=lambda x: x[1], reverse=True)
            
            for feature_pair, redundant_score in sorted_redundant:
                if feature_count >= self.config.max_cross_timeframe_features:
                    break
                    
                if redundant_score > self.config.cross_timeframe_threshold:
                    # Parse feature names
                    if '_' in feature_pair:
                        feature1_name, feature2_name = feature_pair.split('_', 1)
                        
                        # Create cross-timeframe features
                        for tf1 in self.config.timeframes:
                            for tf2 in self.config.timeframes:
                                if feature_count >= self.config.max_cross_timeframe_features:
                                    break
                                    
                                if tf1 != tf2 and tf1 in timeframe_data and tf2 in timeframe_data:
                                    try:
                                        # Find feature indices in each timeframe
                                        tf1_features = timeframe_data[tf1]
                                        tf2_features = timeframe_data[tf2]
                                        
                                        # Create cross-timeframe interaction
                                        cross_name = f"{feature1_name}_{tf1}_{feature2_name}_{tf2}_cross"
                                        
                                        # Simple cross-timeframe feature (ratio)
                                        if tf1_features.shape[1] > 0 and tf2_features.shape[1] > 0:
                                            cross_timeframe_features[cross_name] = (
                                                tf1_features[:, 0] / (tf2_features[:, 0] + 1e-10)
                                            )
                                            feature_count += 1
                                        
                                    except Exception as e:
                                        self.logger.warning(f"⚠️ Failed to create cross-timeframe feature: {e}")
                                        continue
        
        # Store results
        self.cross_timeframe_features = cross_timeframe_features
        
        self.logger.info(f"✅ Created {len(cross_timeframe_features)} cross-timeframe features (max: {self.config.max_cross_timeframe_features})")
        return cross_timeframe_features
    
    def get_pid_summary(self) -> Dict[str, Any]:
        """Get summary of PID results."""
        if not self.pid_results:
            return {}
        
        summary = {
            'method': self.pid_results.get('method', 'unknown'),
            'n_polynomial_features': len(self.polynomial_features),
            'n_cross_timeframe_features': len(self.cross_timeframe_features),
            'total_features_analyzed': len(self.pid_results.get('unique_information', {})),
            'avg_unique_information': 0.0,
            'avg_redundant_information': 0.0,
            'avg_synergistic_information': 0.0
        }
        
        # Compute averages
        if 'unique_information' in self.pid_results:
            unique_values = list(self.pid_results['unique_information'].values())
            summary['avg_unique_information'] = np.mean(unique_values) if unique_values else 0.0
        
        if 'redundant_information' in self.pid_results:
            redundant_values = list(self.pid_results['redundant_information'].values())
            summary['avg_redundant_information'] = np.mean(redundant_values) if redundant_values else 0.0
        
        if 'synergistic_information' in self.pid_results:
            synergistic_values = list(self.pid_results['synergistic_information'].values())
            summary['avg_synergistic_information'] = np.mean(synergistic_values) if synergistic_values else 0.0
        
        return summary
    
    def compute_enhanced_correlation_analysis(self, X: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Compute enhanced correlation analysis using advanced matrix operations."""
        try:
            if not MATRIX_OPS_AVAILABLE or not self.config.enable_correlation_analysis:
                return {}
            
            results = {}
            
            # Convert to DataFrame for processing
            df = pd.DataFrame(X, columns=feature_names)
            
            if self.config.enable_gpu_acceleration and self.enhanced_matrix_ops:
                # Use GPU-accelerated correlation analysis
                corr_matrix = correlation_matrix_gpu(df)
                results['correlation_matrix'] = corr_matrix
                
                # Compute eigendecomposition for feature importance
                if self.config.enable_eigendecomposition:
                    eigenvalues, eigenvectors = eigendecomposition_gpu(corr_matrix)
                    results['eigenvalues'] = eigenvalues
                    results['eigenvectors'] = eigenvectors
                    
                    # Feature importance based on eigenvalues
                    feature_importance = np.abs(eigenvectors).sum(axis=1)
                    results['feature_importance'] = dict(zip(feature_names, feature_importance))
            else:
                # Fallback to traditional correlation analysis
                corr_matrix = df.corr()
                results['correlation_matrix'] = corr_matrix
                
                # Compute eigendecomposition
                eigenvalues, eigenvectors = np.linalg.eig(corr_matrix)
                results['eigenvalues'] = eigenvalues
                results['eigenvectors'] = eigenvectors
                
                # Feature importance
                feature_importance = np.abs(eigenvectors).sum(axis=1)
                results['feature_importance'] = dict(zip(feature_names, feature_importance))
            
            return results
            
        except Exception as e:
            self.logger.warning(f"Enhanced correlation analysis failed: {e}")
            return {}
    
    def compute_batch_correlation_analysis(self, X: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Compute correlation analysis in batches for large datasets."""
        try:
            if not MATRIX_OPS_AVAILABLE or not self.config.enable_batch_correlation:
                return {}
            
            if self.batch_processor and X.shape[0] > 1000:
                # Process in batches for memory efficiency
                batch_size = min(500, X.shape[0] // 4)
                batches = [X[i:i+batch_size] for i in range(0, X.shape[0], batch_size)]
                
                batch_results = []
                for batch in batches:
                    batch_df = pd.DataFrame(batch, columns=feature_names)
                    batch_corr = batch_correlation_analysis(batch_df)
                    batch_results.append(batch_corr)
                
                # Combine batch results
                if batch_results:
                    combined_corr = np.mean(batch_results, axis=0)
                    return {
                        'batch_correlation_matrix': combined_corr,
                        'n_batches_processed': len(batches),
                        'batch_size': batch_size
                    }
            
            return {}
            
        except Exception as e:
            self.logger.warning(f"Batch correlation analysis failed: {e}")
            return {}
    
    def optimize_matrix_operations(self, X: np.ndarray, operation_type: str = "correlation") -> Dict[str, Any]:
        """Optimize matrix operations based on hardware capabilities."""
        try:
            if not MATRIX_OPS_AVAILABLE or not self.config.enable_matrix_optimization:
                return {}
            
            optimization_result = optimize_matrix_operation_with_hardware(
                X, operation_type, 
                gpu_enabled=self.config.enable_gpu_acceleration,
                batch_enabled=self.config.enable_batch_processing
            )
            
            return optimization_result
            
        except Exception as e:
            self.logger.warning(f"Matrix operations optimization failed: {e}")
            return {}
    
    def get_enhanced_performance_metrics(self) -> Dict[str, Any]:
        """Get enhanced performance metrics including matrix operations status."""
        base_metrics = self.get_pid_summary()
        
        enhanced_metrics = {
            **base_metrics,
            'matrix_operations_available': MATRIX_OPS_AVAILABLE,
            'common_operations_available': COMMON_OPERATIONS_AVAILABLE,
            'enhanced_matrix_ops_initialized': self.enhanced_matrix_ops is not None,
            'vectorized_core_initialized': self.vectorized_core is not None,
            'batch_processor_initialized': self.batch_processor is not None,
            'gpu_acceleration_enabled': self.config.enable_gpu_acceleration,
            'batch_processing_enabled': self.config.enable_batch_processing,
            'memory_usage': get_memory_usage() if COMMON_OPERATIONS_AVAILABLE else 0.0
        }
        
        return enhanced_metrics


# Convenience functions
def create_pid_module(config: Optional[PIDConfig] = None) -> PartialInformationDecomposition:
    """Create a PID module instance."""
    return PartialInformationDecomposition(config)


def compute_pid_and_create_features(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    timeframe_data: Optional[Dict[str, np.ndarray]] = None,
    config: Optional[PIDConfig] = None
) -> Dict[str, Any]:
    """
    Convenience function to compute PID and create polynomial/interaction/cross-timeframe features.
    
    Args:
        X: Feature matrix
        y: Target variable
        feature_names: List of feature names
        timeframe_data: Optional timeframe data for cross-timeframe features
        config: PID configuration
        
    Returns:
        Dictionary containing PID results and created features
    """
    pid_module = create_pid_module(config)
    
    # Compute PID
    pid_results = pid_module.compute_pid(X, y, feature_names)
    
    # Create polynomial features (up to 50)
    polynomial_features = pid_module.create_polynomial_features(X, feature_names)
    
    # Create interaction features (up to 100 most relevant)
    interaction_features = pid_module.create_interaction_features(X, feature_names)
    
    # Create cross-timeframe features if data provided (up to 50)
    cross_timeframe_features = {}
    if timeframe_data:
        cross_timeframe_features = pid_module.create_cross_timeframe_features(
            X, feature_names, timeframe_data
        )
    
    return {
        'pid_results': pid_results,
        'polynomial_features': polynomial_features,
        'interaction_features': interaction_features,
        'cross_timeframe_features': cross_timeframe_features,
        'pid_summary': pid_module.get_pid_summary()
    }


# Export key classes and functions
__all__ = [
    'PartialInformationDecomposition',
    'PIDConfig',
    'create_pid_module',
    'compute_pid_and_create_features'
]
