"""
NAS Clusterer - Comprehensive Neural Architecture Search Clustering System

This module provides advanced clustering capabilities for market regime detection
and analysis using various clustering algorithms, optimization techniques, and
hardware acceleration. Integrates with shared utilities for comprehensive functionality.

Key Features:
- Multiple clustering algorithms (KMeans, DBSCAN, Agglomerative, GMM)
- Bayesian TPE and Grid Search optimization
- M1 hardware acceleration (GPU, Memory, CPU)
- Advanced ML utilities (CV, lookahead, HPO)
- Comprehensive validation and error handling
- Serialization and persistence capabilities
- Real-time performance monitoring
"""

import logging
import time
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import asyncio
import concurrent.futures
from contextlib import contextmanager
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.model_selection import cross_val_score, TimeSeriesSplit

# Import shared utilities
try:
    from src.utils.common_operations import (
        safe_dataframe_operation, validate_dataframe_columns, 
        safe_convert_dtypes, calculate_data_quality_metrics,
        safe_merge_dataframes, create_summary_statistics,
        optimize_dataframe_dtypes, safe_timestamp_conversion,
        get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer,
        integrate_with_m1_optimizers, memory_checkpoint, gpu_context,
        safe_json_dump, safe_json_load, ensure_directory
    )
    from src.utils.math_validation import (
        safe_divide, safe_log, safe_sqrt, safe_power, validate_finite,
        validate_positive, validate_range, safe_correlation, safe_covariance,
        safe_mean, safe_std, validate_numeric_array, MathValidation
    )
    from src.utils.serialization_utils import UniversalSerializer
    from src.utils.tprint import (
        tprint, tprint_info, tprint_debug, tprint_warning, tprint_error,
        tprint_success, tprint_progress, tprint_performance, tprint_structured
    )
    from src.utils.hardware.m1_gpu_utils import (
        is_m1_available, is_mps_available, optimize_dataframe_for_m1,
        create_m1_optimized_array, get_m1_gpu_manager
    )
    from src.utils.hardware.m1_memory_optimizer import (
        get_m1_memory_optimizer, optimize_dataframe_memory
    )
    from src.utils.hardware.m1_cpu_optimizer import (
        get_m1_cpu_optimizer
    )
    SHARED_UTILS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"⚠️ Shared utilities not available: {e}")
    SHARED_UTILS_AVAILABLE = False
    # Fallback implementations
    def tprint(*args, **kwargs): print(*args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_progress(*args, **kwargs): print("PROGRESS:", *args, **kwargs)
    def tprint_performance(*args, **kwargs): print("PERFORMANCE:", *args, **kwargs)

# Import ML utilities
try:
    from src.utils.nas_tas.bayesian_tpe_optimizer import (
        BayesianTPEOptimizer, BayesianTPEConfig, OptimizationResult
    )
    from src.utils.ml_common.validation import CrossValidator
    from src.utils.ml_common.feature_selection import FeatureSelector
    ML_COMMON_AVAILABLE = True
except ImportError:
    ML_COMMON_AVAILABLE = False
    tprint_warning("ML common utilities not available, using fallback implementations")

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class NASClusteringConfig:
    """Configuration for NAS Clustering system."""
    
    # Clustering parameters
    clustering_methods: List[str] = field(default_factory=lambda: [
        'kmeans', 'dbscan', 'agglomerative', 'gmm'
    ])
    n_clusters_range: Tuple[int, int] = (2, 20)
    max_clusters: int = 20
    
    # Algorithm-specific parameters
    dbscan_eps_range: Tuple[float, float] = (0.1, 2.0)
    dbscan_min_samples_range: Tuple[int, int] = (5, 50)
    gmm_covariance_types: List[str] = field(default_factory=lambda: [
        'full', 'tied', 'diag', 'spherical'
    ])
    
    # Optimization parameters
    enable_optimization: bool = True
    optimization_method: str = 'bayesian_tpe'  # 'grid_search', 'bayesian_tpe'
    n_trials: int = 50
    optimization_timeout: int = 3600  # 1 hour
    early_stopping_patience: int = 10
    
    # Validation parameters
    cv_folds: int = 5
    validation_strategy: str = 'time_series'  # 'time_series', 'kfold'
    enable_cross_validation: bool = True
    
    # Feature engineering
    enable_feature_selection: bool = True
    max_features: int = 100
    feature_selection_methods: List[str] = field(default_factory=lambda: [
        'variance', 'correlation', 'mutual_info'
    ])
    
    # Dimensionality reduction
    enable_dimensionality_reduction: bool = True
    reduction_methods: List[str] = field(default_factory=lambda: [
        'pca', 'ica', 'truncated_svd'
    ])
    n_components_range: Tuple[int, int] = (2, 50)
    
    # Hardware optimization
    enable_m1_optimization: bool = True
    enable_parallel_processing: bool = True
    max_workers: int = 4
    memory_limit_gb: Optional[float] = None
    
    # Performance monitoring
    enable_performance_monitoring: bool = True
    log_level: str = 'INFO'
    verbose: bool = True
    
    # Output settings
    save_results: bool = True
    output_dir: str = "nas_clustering_results"
    results_format: str = 'json'  # 'json', 'pickle', 'parquet'

@dataclass
class ClusteringResult:
    """Result of clustering analysis."""
    
    cluster_labels: np.ndarray
    cluster_centers: Optional[np.ndarray] = None
    cluster_probabilities: Optional[np.ndarray] = None
    silhouette_score: Optional[float] = None
    calinski_harabasz_score: Optional[float] = None
    davies_bouldin_score: Optional[float] = None
    n_clusters: int = 0
    algorithm_used: str = ""
    optimization_time: float = 0.0
    feature_importance: Optional[np.ndarray] = None
    cluster_characteristics: Optional[Dict[str, Any]] = None
    validation_metrics: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error_message: Optional[str] = None

class NASClusterer:
    """
    Neural Architecture Search Clustering System
    
    A comprehensive clustering system that integrates multiple clustering algorithms
    with advanced optimization, hardware acceleration, and ML utilities.
    """
    
    def __init__(self, config: Optional[NASClusteringConfig] = None):
        """Initialize NAS Clusterer."""
        self.config = config or NASClusteringConfig()
        self.logger = self._setup_logging()
        
        # Initialize shared utilities
        self._init_shared_utilities()
        
        # Initialize hardware optimizations
        self._init_hardware_optimizations()
        
        # Initialize ML utilities
        self._init_ml_utilities()
        
        # Initialize serialization
        self.serializer = UniversalSerializer() if SHARED_UTILS_AVAILABLE else None
        
        # State variables
        self.is_fitted = False
        self.scaler = None
        self.cluster_model = None
        self.feature_names = None
        self.last_clustering_time = None
        self.results = []
        
        tprint_success("🚀 NAS Clusterer initialized successfully")
        tprint_info(f"   → Clustering methods: {self.config.clustering_methods}")
        tprint_info(f"   → Optimization method: {self.config.optimization_method}")
        tprint_info(f"   → M1 optimization: {'enabled' if self.config.enable_m1_optimization else 'disabled'}")
        tprint_info(f"   → Parallel processing: {'enabled' if self.config.enable_parallel_processing else 'disabled'}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger(f"{__name__}.NASClusterer")
        logger.setLevel(getattr(logging, self.config.log_level.upper()))
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def _init_shared_utilities(self):
        """Initialize shared utilities."""
        if not SHARED_UTILS_AVAILABLE:
            self.logger.warning("⚠️ Shared utilities not available, using fallback implementations")
            self.shared_utils = None
            return
        
        try:
            # Initialize math validation
            self.math_validator = MathValidation()
            
            self.shared_utils = True
            self.logger.info("✅ Shared utilities initialized")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Shared utilities initialization failed: {e}")
            self.shared_utils = None
    
    def _init_hardware_optimizations(self):
        """Initialize hardware optimizations."""
        if not self.config.enable_m1_optimization:
            self.gpu_manager = None
            self.memory_optimizer = None
            self.cpu_optimizer = None
            return
        
        try:
            if SHARED_UTILS_AVAILABLE:
                self.gpu_manager = get_m1_gpu_manager()
                self.memory_optimizer = get_m1_memory_optimizer(self.config.memory_limit_gb)
                self.cpu_optimizer = get_m1_cpu_optimizer()
                
                # Start memory monitoring if available
                if self.memory_optimizer and hasattr(self.memory_optimizer, 'start_monitoring'):
                    self.memory_optimizer.start_monitoring()
                
                # Integrate with M1 optimizers
                integration_result = integrate_with_m1_optimizers()
                if integration_result.get('success', False):
                    tprint_success("✅ M1 hardware optimization integrated")
                else:
                    tprint_warning("⚠️ M1 hardware optimization integration failed")
                
                self.logger.info("✅ Hardware optimizations initialized")
            else:
                self.gpu_manager = None
                self.memory_optimizer = None
                self.cpu_optimizer = None
                self.logger.warning("⚠️ Hardware optimizations not available")
                
        except Exception as e:
            self.logger.warning(f"⚠️ Hardware optimization initialization failed: {e}")
            self.gpu_manager = None
            self.memory_optimizer = None
            self.cpu_optimizer = None
    
    def _init_ml_utilities(self):
        """Initialize ML utilities."""
        if not ML_COMMON_AVAILABLE:
            self.optimizer = None
            self.cross_validator = None
            self.feature_selector = None
            return
        
        try:
            if self.config.enable_optimization and self.config.optimization_method == 'bayesian_tpe':
                # Initialize Bayesian TPE optimizer
                tpe_config = BayesianTPEConfig(
                    n_trials=self.config.n_trials,
                    timeout_seconds=self.config.optimization_timeout,
                    enable_parallel=self.config.enable_parallel_processing,
                    max_workers=self.config.max_workers
                )
                self.optimizer = BayesianTPEOptimizer(tpe_config)
                tprint_success("✅ Bayesian TPE optimizer initialized")
            else:
                self.optimizer = None
                tprint_info("ℹ️ Optimization disabled or not available")
            
            # Initialize other ML utilities
            self.cross_validator = CrossValidator()
            self.feature_selector = FeatureSelector()
            
        except Exception as e:
            self.logger.warning(f"⚠️ ML utilities initialization failed: {e}")
            self.optimizer = None
            self.cross_validator = None
            self.feature_selector = None
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[np.ndarray] = None) -> 'NASClusterer':
        """
        Fit the clustering model to data.
        
        Args:
            X: Input features (DataFrame or numpy array)
            y: Optional target values (for supervised feature selection)
            
        Returns:
            Self for method chaining
        """
        tprint_info("🔍 Starting NAS clustering fitting")
        start_time = time.time()
        
        try:
            # Validate and preprocess data
            X_processed = self._preprocess_data(X)
            
            # Apply feature selection if enabled
            if self.config.enable_feature_selection and self.feature_selector:
                X_processed, self.feature_names = self._apply_feature_selection(X_processed, y)
            else:
                self.feature_names = list(range(X_processed.shape[1])) if hasattr(X_processed, 'shape') else []
            
            # Apply dimensionality reduction if enabled
            if self.config.enable_dimensionality_reduction:
                X_processed, reduction_info = self._apply_dimensionality_reduction(X_processed)
            else:
                reduction_info = None
            
            # Optimize clustering parameters if enabled
            if self.config.enable_optimization and self.optimizer:
                best_params = self._optimize_clustering_parameters(X_processed)
            else:
                best_params = self._get_default_parameters()
            
            # Fit clustering model with best parameters
            cluster_result = self._fit_clustering_model(X_processed, best_params)
            
            # Store results
            self.cluster_model = cluster_result['model']
            self.scaler = cluster_result['scaler']
            self.is_fitted = True
            self.last_clustering_time = time.time()
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(X_processed, cluster_result['labels'])
            
            # Store result
            result = ClusteringResult(
                cluster_labels=cluster_result['labels'],
                cluster_centers=cluster_result.get('centers'),
                cluster_probabilities=cluster_result.get('probabilities'),
                silhouette_score=performance_metrics.get('silhouette_score'),
                calinski_harabasz_score=performance_metrics.get('calinski_harabasz_score'),
                davies_bouldin_score=performance_metrics.get('davies_bouldin_score'),
                n_clusters=len(np.unique(cluster_result['labels'])),
                algorithm_used=best_params.get('algorithm', 'kmeans'),
                optimization_time=time.time() - start_time,
                feature_importance=cluster_result.get('feature_importance'),
                cluster_characteristics=self._analyze_cluster_characteristics(X_processed, cluster_result['labels']),
                validation_metrics=performance_metrics,
                metadata={
                    'reduction_info': reduction_info,
                    'best_params': best_params,
                    'data_shape': X_processed.shape,
                    'feature_names': self.feature_names
                },
                success=True
            )
            
            self.results.append(result)
            
            tprint_performance("NAS clustering fitting", time.time() - start_time)
            tprint_success(f"✅ Clustering completed: {result.n_clusters} clusters detected")
            tprint_info(f"   → Silhouette score: {result.silhouette_score:.3f}")
            tprint_info(f"   → Algorithm used: {result.algorithm_used}")
            
            return self
            
        except Exception as e:
            tprint_error(f"❌ Clustering fitting failed: {e}")
            error_result = ClusteringResult(
                cluster_labels=np.array([]),
                success=False,
                error_message=str(e),
                optimization_time=time.time() - start_time
            )
            self.results.append(error_result)
            raise
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> ClusteringResult:
        """
        Predict cluster labels for new data.
        
        Args:
            X: New data to cluster
            
        Returns:
            ClusteringResult with predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        tprint_info("🔮 Making clustering predictions")
        start_time = time.time()
        
        try:
            # Preprocess data using same transformations as training
            X_processed = self._preprocess_data(X)
            
            # Apply same feature selection and dimensionality reduction
            if self.feature_names and len(self.feature_names) < X_processed.shape[1]:
                X_processed = X_processed[:, :len(self.feature_names)]
            
            # Scale data using fitted scaler
            if self.scaler:
                X_processed = self.scaler.transform(X_processed)
            
            # Predict clusters
            if hasattr(self.cluster_model, 'predict'):
                cluster_labels = self.cluster_model.predict(X_processed)
            else:
                cluster_labels = self.cluster_model.fit_predict(X_processed)
            
            # Get cluster probabilities if available
            cluster_probabilities = None
            if hasattr(self.cluster_model, 'predict_proba'):
                cluster_probabilities = self.cluster_model.predict_proba(X_processed)
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(X_processed, cluster_labels)
            
            result = ClusteringResult(
                cluster_labels=cluster_labels,
                cluster_probabilities=cluster_probabilities,
                silhouette_score=performance_metrics.get('silhouette_score'),
                calinski_harabasz_score=performance_metrics.get('calinski_harabasz_score'),
                davies_bouldin_score=performance_metrics.get('davies_bouldin_score'),
                n_clusters=len(np.unique(cluster_labels)),
                algorithm_used=getattr(self.cluster_model, '__class__', {}).get('__name__', 'unknown'),
                optimization_time=time.time() - start_time,
                cluster_characteristics=self._analyze_cluster_characteristics(X_processed, cluster_labels),
                validation_metrics=performance_metrics,
                metadata={
                    'data_shape': X_processed.shape,
                    'prediction_time': time.time()
                },
                success=True
            )
            
            tprint_performance("Clustering prediction", time.time() - start_time)
            tprint_success(f"✅ Prediction completed: {result.n_clusters} clusters predicted")
            
            return result
            
        except Exception as e:
            tprint_error(f"❌ Clustering prediction failed: {e}")
            return ClusteringResult(
                cluster_labels=np.array([]),
                success=False,
                error_message=str(e),
                optimization_time=time.time() - start_time
            )
    
    def _preprocess_data(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Preprocess input data."""
        try:
            # Convert to numpy array if needed
            if isinstance(X, pd.DataFrame):
                X_array = X.values
                # Optimize DataFrame memory if available
                if self.memory_optimizer and hasattr(self.memory_optimizer, 'optimize_dataframe_memory'):
                    X = self.memory_optimizer.optimize_dataframe_memory(X)
            else:
                X_array = np.array(X)
            
            # Validate data
            if self.math_validator:
                try:
                    X_array = self.math_validator.validate_numeric_array(X_array, "input_data")
                except Exception as e:
                    tprint_warning(f"Data validation warning: {e}")
            
            # Handle missing values
            X_array = np.nan_to_num(X_array, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Optimize for M1 if available
            if is_m1_available() and SHARED_UTILS_AVAILABLE:
                X_array = create_m1_optimized_array(X_array)
            
            return X_array
            
        except Exception as e:
            tprint_error(f"❌ Data preprocessing failed: {e}")
            raise
    
    def _apply_feature_selection(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, List[str]]:
        """Apply feature selection to reduce dimensionality."""
        if not self.feature_selector or X.shape[1] <= self.config.max_features:
            return X, list(range(X.shape[1]))
        
        try:
            tprint_info(f"🔧 Applying feature selection: {X.shape[1]} → {self.config.max_features} features")
            
            # Use variance-based selection as fallback
            from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
            
            # Remove low variance features
            var_selector = VarianceThreshold(threshold=0.01)
            X_var = var_selector.fit_transform(X)
            
            # Select top features
            if X_var.shape[1] > self.config.max_features:
                if y is not None:
                    selector = SelectKBest(f_classif, k=self.config.max_features)
                    X_selected = selector.fit_transform(X_var, y)
                    selected_features = selector.get_support(indices=True)
                else:
                    # Use variance for unsupervised selection
                    selector = SelectKBest(k=self.config.max_features)
                    X_selected = selector.fit_transform(X_var)
                    selected_features = selector.get_support(indices=True)
            else:
                X_selected = X_var
                selected_features = list(range(X_var.shape[1]))
            
            feature_names = [f"feature_{i}" for i in selected_features]
            
            tprint_success(f"✅ Feature selection completed: {X.shape[1]} → {X_selected.shape[1]} features")
            return X_selected, feature_names
            
        except Exception as e:
            tprint_warning(f"⚠️ Feature selection failed: {e}")
            return X, list(range(X.shape[1]))
    
    def _apply_dimensionality_reduction(self, X: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply dimensionality reduction to features."""
        if X.shape[1] <= self.config.n_components_range[1]:
            return X, None
        
        try:
            tprint_info(f"🔧 Applying dimensionality reduction: {X.shape[1]} features")
            
            reduction_info = {}
            reduced_X = X
            
            for method in self.config.reduction_methods:
                try:
                    if method == 'pca':
                        n_components = min(self.config.n_components_range[1], X.shape[1])
                        reducer = PCA(n_components=n_components, random_state=42)
                        reduced_X = reducer.fit_transform(X)
                        reduction_info['pca'] = {
                            'n_components': n_components,
                            'explained_variance_ratio': reducer.explained_variance_ratio_.tolist()
                        }
                        break
                    
                    elif method == 'ica':
                        n_components = min(self.config.n_components_range[1], X.shape[1])
                        reducer = FastICA(n_components=n_components, random_state=42)
                        reduced_X = reducer.fit_transform(X)
                        reduction_info['ica'] = {'n_components': n_components}
                        break
                    
                    elif method == 'truncated_svd':
                        n_components = min(self.config.n_components_range[1], X.shape[1])
                        reducer = TruncatedSVD(n_components=n_components, random_state=42)
                        reduced_X = reducer.fit_transform(X)
                        reduction_info['truncated_svd'] = {
                            'n_components': n_components,
                            'explained_variance_ratio': reducer.explained_variance_ratio_.tolist()
                        }
                        break
                        
                except Exception as e:
                    tprint_warning(f"   → {method}: Reduction failed - {e}")
                    continue
            
            tprint_success(f"✅ Dimensionality reduction completed: {X.shape[1]} → {reduced_X.shape[1]} features")
            return reduced_X, reduction_info
            
        except Exception as e:
            tprint_warning(f"⚠️ Dimensionality reduction failed: {e}")
            return X, None
    
    def _optimize_clustering_parameters(self, X: np.ndarray) -> Dict[str, Any]:
        """Optimize clustering parameters using Bayesian TPE."""
        if not self.optimizer:
            return self._get_default_parameters()
        
        try:
            tprint_info("🔧 Optimizing clustering parameters")
            
            # Define search space
            search_space = self._create_search_space()
            
            # Define objective function
            def objective(params):
                try:
                    # Fit clustering model with given parameters
                    cluster_result = self._fit_clustering_model(X, params)
                    
                    # Calculate silhouette score as objective
                    if len(np.unique(cluster_result['labels'])) > 1:
                        score = silhouette_score(X, cluster_result['labels'])
                    else:
                        score = -1.0  # Penalty for single cluster
                    
                    return score
                    
                except Exception as e:
                    tprint_debug(f"Objective function failed: {e}")
                    return -np.inf
            
            # Run optimization
            result = self.optimizer.optimize(objective, search_space)
            
            if result.success:
                tprint_success(f"✅ Parameter optimization completed: best score = {result.best_score:.4f}")
                return result.best_params
            else:
                tprint_warning("⚠️ Parameter optimization failed, using default parameters")
                return self._get_default_parameters()
                
        except Exception as e:
            tprint_warning(f"⚠️ Parameter optimization failed: {e}")
            return self._get_default_parameters()
    
    def _create_search_space(self) -> Dict[str, Any]:
        """Create search space for optimization."""
        search_space = {
            'algorithm': {
                'type': 'categorical',
                'choices': self.config.clustering_methods
            },
            'n_clusters': {
                'type': 'int',
                'low': self.config.n_clusters_range[0],
                'high': min(self.config.n_clusters_range[1], X.shape[0] // 2)
            }
        }
        
        # Add algorithm-specific parameters
        if 'dbscan' in self.config.clustering_methods:
            search_space['eps'] = {
                'type': 'float',
                'low': self.config.dbscan_eps_range[0],
                'high': self.config.dbscan_eps_range[1]
            }
            search_space['min_samples'] = {
                'type': 'int',
                'low': self.config.dbscan_min_samples_range[0],
                'high': self.config.dbscan_min_samples_range[1]
            }
        
        if 'gmm' in self.config.clustering_methods:
            search_space['covariance_type'] = {
                'type': 'categorical',
                'choices': self.config.gmm_covariance_types
            }
        
        return search_space
    
    def _get_default_parameters(self) -> Dict[str, Any]:
        """Get default clustering parameters."""
        return {
            'algorithm': 'kmeans',
            'n_clusters': min(3, self.config.n_clusters_range[1]),
            'random_state': 42
        }
    
    def _fit_clustering_model(self, X: np.ndarray, params: Dict[str, Any]) -> Dict[str, Any]:
        """Fit clustering model with given parameters."""
        algorithm = params.get('algorithm', 'kmeans')
        n_clusters = params.get('n_clusters', 3)
        
        # Scale data
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit clustering model
        if algorithm == 'kmeans':
            model = KMeans(
                n_clusters=n_clusters,
                random_state=params.get('random_state', 42),
                n_init=10
            )
            labels = model.fit_predict(X_scaled)
            centers = model.cluster_centers_
            
        elif algorithm == 'dbscan':
            model = DBSCAN(
                eps=params.get('eps', 0.5),
                min_samples=params.get('min_samples', 5)
            )
            labels = model.fit_predict(X_scaled)
            centers = None
            
        elif algorithm == 'agglomerative':
            model = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage='ward'
            )
            labels = model.fit_predict(X_scaled)
            centers = None
            
        elif algorithm == 'gmm':
            model = GaussianMixture(
                n_components=n_clusters,
                covariance_type=params.get('covariance_type', 'full'),
                random_state=params.get('random_state', 42)
            )
            labels = model.fit_predict(X_scaled)
            centers = model.means_
            probabilities = model.predict_proba(X_scaled)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        result = {
            'model': model,
            'scaler': scaler,
            'labels': labels,
            'centers': centers,
            'algorithm': algorithm
        }
        
        if algorithm == 'gmm':
            result['probabilities'] = probabilities
        
        return result
    
    def _calculate_performance_metrics(self, X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Calculate clustering performance metrics."""
        try:
            metrics = {}
            
            if len(np.unique(labels)) > 1:
                metrics['silhouette_score'] = silhouette_score(X, labels)
                metrics['calinski_harabasz_score'] = calinski_harabasz_score(X, labels)
                metrics['davies_bouldin_score'] = davies_bouldin_score(X, labels)
            else:
                metrics['silhouette_score'] = -1.0
                metrics['calinski_harabasz_score'] = 0.0
                metrics['davies_bouldin_score'] = np.inf
            
            return metrics
            
        except Exception as e:
            tprint_warning(f"⚠️ Performance metrics calculation failed: {e}")
            return {
                'silhouette_score': -1.0,
                'calinski_harabasz_score': 0.0,
                'davies_bouldin_score': np.inf
            }
    
    def _analyze_cluster_characteristics(self, X: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """Analyze characteristics of each cluster."""
        try:
            characteristics = {}
            unique_labels = np.unique(labels)
            
            for label in unique_labels:
                mask = labels == label
                cluster_data = X[mask]
                
                if len(cluster_data) == 0:
                    continue
                
                cluster_stats = {
                    'count': len(cluster_data),
                    'percentage': len(cluster_data) / len(X) * 100,
                    'mean': np.mean(cluster_data, axis=0).tolist(),
                    'std': np.std(cluster_data, axis=0).tolist(),
                    'min': np.min(cluster_data, axis=0).tolist(),
                    'max': np.max(cluster_data, axis=0).tolist()
                }
                
                characteristics[str(label)] = cluster_stats
            
            return characteristics
            
        except Exception as e:
            tprint_warning(f"⚠️ Cluster characteristics analysis failed: {e}")
            return {}
    
    def save_results(self, filepath: str) -> bool:
        """Save clustering results to file."""
        try:
            if not self.results:
                tprint_warning("⚠️ No results to save")
                return False
            
            # Ensure directory exists
            ensure_directory(Path(filepath).parent)
            
            # Prepare data for serialization
            save_data = {
                'results': [
                    {
                        'cluster_labels': result.cluster_labels.tolist(),
                        'cluster_centers': result.cluster_centers.tolist() if result.cluster_centers is not None else None,
                        'cluster_probabilities': result.cluster_probabilities.tolist() if result.cluster_probabilities is not None else None,
                        'silhouette_score': result.silhouette_score,
                        'calinski_harabasz_score': result.calinski_harabasz_score,
                        'davies_bouldin_score': result.davies_bouldin_score,
                        'n_clusters': result.n_clusters,
                        'algorithm_used': result.algorithm_used,
                        'optimization_time': result.optimization_time,
                        'cluster_characteristics': result.cluster_characteristics,
                        'validation_metrics': result.validation_metrics,
                        'metadata': result.metadata,
                        'success': result.success,
                        'error_message': result.error_message
                    }
                    for result in self.results
                ],
                'config': self.config.__dict__,
                'model_info': {
                    'is_fitted': self.is_fitted,
                    'last_clustering_time': self.last_clustering_time,
                    'feature_names': self.feature_names
                }
            }
            
            # Save using appropriate format
            if self.config.results_format == 'json':
                success = safe_json_dump(save_data, filepath)
            elif self.config.results_format == 'pickle':
                with open(filepath, 'wb') as f:
                    pickle.dump(save_data, f)
                success = True
            else:
                tprint_error(f"❌ Unknown results format: {self.config.results_format}")
                return False
            
            if success:
                tprint_success(f"💾 Results saved to {filepath}")
            else:
                tprint_error("❌ Failed to save results")
            
            return success
            
        except Exception as e:
            tprint_error(f"❌ Error saving results: {e}")
            return False
    
    def load_results(self, filepath: str) -> bool:
        """Load clustering results from file."""
        try:
            if self.config.results_format == 'json':
                load_data = safe_json_load(filepath)
            elif self.config.results_format == 'pickle':
                with open(filepath, 'rb') as f:
                    load_data = pickle.load(f)
            else:
                tprint_error(f"❌ Unknown results format: {self.config.results_format}")
                return False
            
            if not load_data:
                tprint_error("❌ Failed to load results")
                return False
            
            # Reconstruct results
            self.results = []
            for result_data in load_data['results']:
                result = ClusteringResult(
                    cluster_labels=np.array(result_data['cluster_labels']),
                    cluster_centers=np.array(result_data['cluster_centers']) if result_data['cluster_centers'] else None,
                    cluster_probabilities=np.array(result_data['cluster_probabilities']) if result_data['cluster_probabilities'] else None,
                    silhouette_score=result_data['silhouette_score'],
                    calinski_harabasz_score=result_data['calinski_harabasz_score'],
                    davies_bouldin_score=result_data['davies_bouldin_score'],
                    n_clusters=result_data['n_clusters'],
                    algorithm_used=result_data['algorithm_used'],
                    optimization_time=result_data['optimization_time'],
                    cluster_characteristics=result_data['cluster_characteristics'],
                    validation_metrics=result_data['validation_metrics'],
                    metadata=result_data['metadata'],
                    success=result_data['success'],
                    error_message=result_data['error_message']
                )
                self.results.append(result)
            
            # Restore model state
            self.is_fitted = load_data['model_info']['is_fitted']
            self.last_clustering_time = load_data['model_info']['last_clustering_time']
            self.feature_names = load_data['model_info']['feature_names']
            
            tprint_success(f"✅ Results loaded from {filepath}")
            return True
            
        except Exception as e:
            tprint_error(f"❌ Error loading results: {e}")
            return False
    
    def get_clustering_summary(self) -> Dict[str, Any]:
        """Get summary of clustering results."""
        if not self.results:
            return {'message': 'No clustering results available'}
        
        latest_result = self.results[-1]
        
        summary = {
            'total_results': len(self.results),
            'latest_result': {
                'n_clusters': latest_result.n_clusters,
                'algorithm_used': latest_result.algorithm_used,
                'silhouette_score': latest_result.silhouette_score,
                'calinski_harabasz_score': latest_result.calinski_harabasz_score,
                'davies_bouldin_score': latest_result.davies_bouldin_score,
                'optimization_time': latest_result.optimization_time,
                'success': latest_result.success
            },
            'model_info': {
                'is_fitted': self.is_fitted,
                'feature_names': self.feature_names,
                'last_clustering_time': self.last_clustering_time
            }
        }
        
        return summary
    
    def cleanup(self):
        """Cleanup resources and stop monitoring."""
        try:
            if self.memory_optimizer and hasattr(self.memory_optimizer, 'stop_monitoring'):
                self.memory_optimizer.stop_monitoring()
            
            if SHARED_UTILS_AVAILABLE:
                from src.utils.common_operations import cleanup_m1_optimizers
                cleanup_m1_optimizers()
            
            tprint_success("🧹 NAS Clusterer cleanup completed")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Cleanup failed: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()


# Convenience functions
def create_nas_clusterer(config: Optional[NASClusteringConfig] = None) -> NASClusterer:
    """Create a NAS Clusterer instance."""
    return NASClusterer(config)


def cluster_data(X: Union[np.ndarray, pd.DataFrame], 
                config: Optional[NASClusteringConfig] = None,
                **kwargs) -> ClusteringResult:
    """Convenience function for clustering data."""
    clusterer = NASClusterer(config)
    try:
        clusterer.fit(X)
        return clusterer.predict(X)
    finally:
        clusterer.cleanup()


# Export main classes and functions
__all__ = [
    'NASClusterer',
    'NASClusteringConfig', 
    'ClusteringResult',
    'create_nas_clusterer',
    'cluster_data'
]
