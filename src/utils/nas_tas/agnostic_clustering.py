"""
Agnostic Clustering Module

Agnostic clustering system that can be used by both NAS and TAS components
with architecture-specific adaptations.
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

# Import clustering libraries
try:
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.decomposition import PCA, FastICA, TruncatedSVD
    from sklearn.feature_selection import SelectKBest, f_classif, f_regression
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    CLUSTERING_LIBS_AVAILABLE = True
except ImportError:
    CLUSTERING_LIBS_AVAILABLE = False
    tprint_warning("Clustering libraries not available, using fallback implementations")

logger = logging.getLogger(__name__)

@dataclass
class AgnosticClusteringConfig:
    """Configuration for agnostic clustering."""
    
    # Clustering parameters
    clustering_algorithm: str = "kmeans"  # kmeans, dbscan, agglomerative, gmm
    n_clusters: int = 5
    max_clusters: int = 20
    min_clusters: int = 2
    
    # DBSCAN parameters
    eps: float = 0.5
    min_samples: int = 5
    
    # Agglomerative parameters
    linkage: str = "ward"  # ward, complete, average, single
    
    # GMM parameters
    covariance_type: str = "full"  # full, tied, diag, spherical
    
    # Feature processing
    enable_feature_selection: bool = True
    n_features: Optional[int] = None
    feature_selection_method: str = "f_classif"  # f_classif, f_regression, mutual_info_classif
    
    # Dimensionality reduction
    enable_dimensionality_reduction: bool = True
    reduction_method: str = "pca"  # pca, ica, svd
    n_components: Optional[int] = None
    explained_variance_threshold: float = 0.95
    
    # Preprocessing
    enable_scaling: bool = True
    scaling_method: str = "standard"  # standard, minmax
    
    # Optimization
    enable_parameter_optimization: bool = True
    optimization_method: str = "bayesian_tpe"  # bayesian_tpe, grid_search, random_search
    n_trials: int = 50
    
    # Hardware optimization
    enable_m1_optimization: bool = True
    enable_parallel_processing: bool = True
    n_jobs: int = -1
    memory_limit_gb: Optional[float] = None
    
    # Performance monitoring
    verbose: bool = True
    log_level: str = "INFO"
    save_clustering_results: bool = True
    
    # Output settings
    output_dir: str = "agnostic_clustering_results"
    results_format: str = "json"

@dataclass
class AgnosticClusteringResult:
    """Result from agnostic clustering."""
    
    # Clustering results
    success: bool
    cluster_labels: Optional[np.ndarray] = None
    n_clusters: int = 0
    cluster_centers: Optional[np.ndarray] = None
    
    # Performance metrics
    silhouette_score: float = 0.0
    calinski_harabasz_score: float = 0.0
    davies_bouldin_score: float = 0.0
    inertia: float = 0.0
    
    # Feature analysis
    feature_importance: Optional[np.ndarray] = None
    selected_features: Optional[List[int]] = None
    dimensionality_reduction_applied: bool = False
    n_components_used: int = 0
    
    # Clustering analysis
    cluster_sizes: Optional[Dict[int, int]] = None
    cluster_characteristics: Optional[Dict[str, Any]] = None
    outlier_analysis: Optional[Dict[str, Any]] = None
    
    # Performance metrics
    clustering_time: float = 0.0
    memory_usage_mb: float = 0.0
    n_samples: int = 0
    n_features_original: int = 0
    n_features_used: int = 0
    
    # Error handling
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

class AgnosticClusterer:
    """
    Agnostic Clustering System.
    
    Clustering system that can be used by both NAS and TAS components
    with architecture-specific adaptations.
    """
    
    def __init__(self, config: Optional[AgnosticClusteringConfig] = None):
        """Initialize agnostic clusterer."""
        self.config = config or AgnosticClusteringConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize hardware optimizations
        self._init_hardware_optimizations()
        
        # Initialize utilities
        self._init_utilities()
        
        # Clustering state
        self.clustering_history = []
        self.best_model = None
        self.best_score = -np.inf
        
        tprint_success("🚀 Agnostic Clusterer initialized")
        tprint_info(f"   → Algorithm: {self.config.clustering_algorithm}")
        tprint_info(f"   → Feature selection: {'enabled' if self.config.enable_feature_selection else 'disabled'}")
        tprint_info(f"   → Dimensionality reduction: {'enabled' if self.config.enable_dimensionality_reduction else 'disabled'}")
        tprint_info(f"   → M1 optimization: {'enabled' if self.config.enable_m1_optimization else 'disabled'}")
    
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
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[Union[np.ndarray, pd.Series]] = None) -> AgnosticClusteringResult:
        """
        Fit clustering model to data.
        
        Args:
            X: Features to cluster
            y: Optional target variable for supervised feature selection
            
        Returns:
            AgnosticClusteringResult with clustering results
        """
        start_time = time.time()
        tprint_info("🔍 Starting agnostic clustering")
        
        try:
            # Validate inputs
            self._validate_inputs(X, y)
            
            # Preprocess data
            X_processed, y_processed = self._preprocess_data(X, y)
            
            # Apply feature selection if enabled
            if self.config.enable_feature_selection:
                X_processed, selected_features = self._apply_feature_selection(X_processed, y_processed)
            else:
                selected_features = None
            
            # Apply dimensionality reduction if enabled
            if self.config.enable_dimensionality_reduction:
                X_processed, n_components = self._apply_dimensionality_reduction(X_processed)
            else:
                n_components = X_processed.shape[1]
            
            # Optimize clustering parameters if enabled
            if self.config.enable_parameter_optimization:
                best_params = self._optimize_clustering_parameters(X_processed, y_processed)
            else:
                best_params = self._get_default_parameters()
            
            # Create and fit clustering model
            with self._get_hardware_context():
                model = self._create_clustering_model(best_params)
                cluster_labels = model.fit_predict(X_processed)
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(X_processed, cluster_labels)
            
            # Analyze clustering results
            cluster_analysis = self._analyze_clusters(X_processed, cluster_labels)
            
            # Get performance metrics
            memory_usage = self._get_memory_usage()
            
            # Create result
            result = AgnosticClusteringResult(
                success=True,
                cluster_labels=cluster_labels,
                n_clusters=len(np.unique(cluster_labels)),
                cluster_centers=self._get_cluster_centers(model, X_processed, cluster_labels),
                silhouette_score=performance_metrics.get('silhouette_score', 0.0),
                calinski_harabasz_score=performance_metrics.get('calinski_harabasz_score', 0.0),
                davies_bouldin_score=performance_metrics.get('davies_bouldin_score', 0.0),
                inertia=performance_metrics.get('inertia', 0.0),
                feature_importance=self._get_feature_importance(X_processed, cluster_labels),
                selected_features=selected_features,
                dimensionality_reduction_applied=self.config.enable_dimensionality_reduction,
                n_components_used=n_components,
                cluster_sizes=cluster_analysis.get('cluster_sizes'),
                cluster_characteristics=cluster_analysis.get('characteristics'),
                outlier_analysis=cluster_analysis.get('outliers'),
                clustering_time=time.time() - start_time,
                memory_usage_mb=memory_usage,
                n_samples=X_processed.shape[0],
                n_features_original=X.shape[1] if hasattr(X, 'shape') else len(X[0]),
                n_features_used=X_processed.shape[1]
            )
            
            # Store results
            self.best_model = model
            self.best_score = result.silhouette_score
            self.clustering_history.append(result)
            
            # Save results if configured
            if self.config.save_clustering_results:
                self._save_clustering_results(result)
            
            tprint_success(f"✅ Agnostic clustering completed in {result.clustering_time:.2f}s")
            tprint_info(f"   → Clusters found: {result.n_clusters}")
            tprint_info(f"   → Silhouette score: {result.silhouette_score:.4f}")
            tprint_info(f"   → Features used: {result.n_features_used}")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            tprint_error(f"❌ Agnostic clustering failed: {e}")
            
            return AgnosticClusteringResult(
                success=False,
                clustering_time=execution_time,
                error_message=str(e)
            )
    
    def _validate_inputs(self, X, y):
        """Validate input data."""
        if not CLUSTERING_LIBS_AVAILABLE:
            raise ImportError("Clustering libraries not available")
        
        if X is None:
            raise ValueError("X cannot be None")
        
        if len(X) == 0:
            raise ValueError("X cannot be empty")
    
    def _preprocess_data(self, X, y):
        """Preprocess input data."""
        # Convert to numpy arrays
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = np.array(X)
        
        y_array = None
        if y is not None:
            if isinstance(y, pd.Series):
                y_array = y.values
            else:
                y_array = np.array(y)
        
        # Handle missing values
        X_array = np.nan_to_num(X_array, nan=0.0, posinf=0.0, neginf=0.0)
        if y_array is not None:
            y_array = np.nan_to_num(y_array, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Apply scaling if enabled
        if self.config.enable_scaling:
            X_array = self._apply_scaling(X_array)
        
        return X_array, y_array
    
    def _apply_scaling(self, X):
        """Apply scaling to data."""
        try:
            if self.config.scaling_method == "standard":
                scaler = StandardScaler()
            elif self.config.scaling_method == "minmax":
                scaler = MinMaxScaler()
            else:
                return X
            
            return scaler.fit_transform(X)
            
        except Exception as e:
            tprint_warning(f"⚠️ Scaling failed: {e}")
            return X
    
    def _apply_feature_selection(self, X, y):
        """Apply feature selection."""
        try:
            if y is None:
                tprint_warning("⚠️ No target variable provided, skipping feature selection")
                return X, None
            
            # Determine number of features to select
            n_features = self.config.n_features
            if n_features is None:
                n_features = min(X.shape[1], 50)  # Default to 50 features
            
            # Select feature selection method
            if self.config.feature_selection_method == "f_classif":
                selector = SelectKBest(f_classif, k=n_features)
            elif self.config.feature_selection_method == "f_regression":
                selector = SelectKBest(f_regression, k=n_features)
            else:
                tprint_warning(f"⚠️ Unknown feature selection method: {self.config.feature_selection_method}")
                return X, None
            
            # Apply feature selection
            X_selected = selector.fit_transform(X, y)
            selected_features = selector.get_support(indices=True)
            
            tprint_info(f"✅ Feature selection: {X.shape[1]} → {X_selected.shape[1]} features")
            
            return X_selected, selected_features.tolist()
            
        except Exception as e:
            tprint_warning(f"⚠️ Feature selection failed: {e}")
            return X, None
    
    def _apply_dimensionality_reduction(self, X):
        """Apply dimensionality reduction."""
        try:
            # Determine number of components
            n_components = self.config.n_components
            if n_components is None:
                n_components = min(X.shape[1], 10)  # Default to 10 components
            
            # Select reduction method
            if self.config.reduction_method == "pca":
                reducer = PCA(n_components=n_components)
            elif self.config.reduction_method == "ica":
                reducer = FastICA(n_components=n_components, random_state=42)
            elif self.config.reduction_method == "svd":
                reducer = TruncatedSVD(n_components=n_components, random_state=42)
            else:
                tprint_warning(f"⚠️ Unknown reduction method: {self.config.reduction_method}")
                return X, X.shape[1]
            
            # Apply dimensionality reduction
            X_reduced = reducer.fit_transform(X)
            
            # Check explained variance for PCA
            if self.config.reduction_method == "pca" and hasattr(reducer, 'explained_variance_ratio_'):
                explained_variance = np.sum(reducer.explained_variance_ratio_)
                tprint_info(f"✅ Dimensionality reduction: {X.shape[1]} → {X_reduced.shape[1]} components")
                tprint_info(f"   → Explained variance: {explained_variance:.4f}")
            else:
                tprint_info(f"✅ Dimensionality reduction: {X.shape[1]} → {X_reduced.shape[1]} components")
            
            return X_reduced, n_components
            
        except Exception as e:
            tprint_warning(f"⚠️ Dimensionality reduction failed: {e}")
            return X, X.shape[1]
    
    def _optimize_clustering_parameters(self, X, y):
        """Optimize clustering parameters."""
        try:
            if self.config.optimization_method == "bayesian_tpe":
                return self._optimize_with_bayesian_tpe(X, y)
            elif self.config.optimization_method == "grid_search":
                return self._optimize_with_grid_search(X, y)
            elif self.config.optimization_method == "random_search":
                return self._optimize_with_random_search(X, y)
            else:
                tprint_warning(f"⚠️ Unknown optimization method: {self.config.optimization_method}")
                return self._get_default_parameters()
                
        except Exception as e:
            tprint_warning(f"⚠️ Parameter optimization failed: {e}")
            return self._get_default_parameters()
    
    def _optimize_with_bayesian_tpe(self, X, y):
        """Optimize parameters using Bayesian TPE."""
        try:
            from src.utils.bayesian_tpe import BayesianTPEOptimizer, BayesianTPEConfig
            
            # Define search space
            search_space = {
                'n_clusters': (self.config.min_clusters, self.config.max_clusters),
                'eps': (0.1, 2.0) if self.config.clustering_algorithm == "dbscan" else None,
                'min_samples': (2, 20) if self.config.clustering_algorithm == "dbscan" else None
            }
            
            # Remove None values
            search_space = {k: v for k, v in search_space.items() if v is not None}
            
            # Create optimizer
            config = BayesianTPEConfig(
                n_trials=self.config.n_trials,
                random_state=42
            )
            optimizer = BayesianTPEOptimizer(config)
            
            # Define objective function
            def objective(params):
                try:
                    model = self._create_clustering_model(params)
                    cluster_labels = model.fit_predict(X)
                    
                    if len(np.unique(cluster_labels)) < 2:
                        return -1.0  # Penalize single cluster
                    
                    score = silhouette_score(X, cluster_labels)
                    return score
                    
                except Exception:
                    return -1.0
            
            # Optimize
            best_params = optimizer.optimize(objective, search_space)
            
            tprint_info(f"✅ Bayesian TPE optimization completed")
            tprint_info(f"   → Best parameters: {best_params}")
            
            return best_params
            
        except Exception as e:
            tprint_warning(f"⚠️ Bayesian TPE optimization failed: {e}")
            return self._get_default_parameters()
    
    def _optimize_with_grid_search(self, X, y):
        """Optimize parameters using grid search."""
        try:
            best_score = -1.0
            best_params = self._get_default_parameters()
            
            # Define parameter grid
            if self.config.clustering_algorithm == "kmeans":
                param_grid = {
                    'n_clusters': range(self.config.min_clusters, min(self.config.max_clusters + 1, 11))
                }
            elif self.config.clustering_algorithm == "dbscan":
                param_grid = {
                    'eps': np.linspace(0.1, 2.0, 10),
                    'min_samples': range(2, 21, 2)
                }
            else:
                return self._get_default_parameters()
            
            # Grid search
            for params in self._generate_parameter_combinations(param_grid):
                try:
                    model = self._create_clustering_model(params)
                    cluster_labels = model.fit_predict(X)
                    
                    if len(np.unique(cluster_labels)) < 2:
                        continue
                    
                    score = silhouette_score(X, cluster_labels)
                    if score > best_score:
                        best_score = score
                        best_params = params
                        
                except Exception:
                    continue
            
            tprint_info(f"✅ Grid search optimization completed")
            tprint_info(f"   → Best parameters: {best_params}")
            
            return best_params
            
        except Exception as e:
            tprint_warning(f"⚠️ Grid search optimization failed: {e}")
            return self._get_default_parameters()
    
    def _optimize_with_random_search(self, X, y):
        """Optimize parameters using random search."""
        try:
            best_score = -1.0
            best_params = self._get_default_parameters()
            
            # Random search
            for _ in range(self.config.n_trials):
                try:
                    params = self._generate_random_parameters()
                    model = self._create_clustering_model(params)
                    cluster_labels = model.fit_predict(X)
                    
                    if len(np.unique(cluster_labels)) < 2:
                        continue
                    
                    score = silhouette_score(X, cluster_labels)
                    if score > best_score:
                        best_score = score
                        best_params = params
                        
                except Exception:
                    continue
            
            tprint_info(f"✅ Random search optimization completed")
            tprint_info(f"   → Best parameters: {best_params}")
            
            return best_params
            
        except Exception as e:
            tprint_warning(f"⚠️ Random search optimization failed: {e}")
            return self._get_default_parameters()
    
    def _generate_parameter_combinations(self, param_grid):
        """Generate parameter combinations for grid search."""
        import itertools
        
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        
        for combination in itertools.product(*values):
            yield dict(zip(keys, combination))
    
    def _generate_random_parameters(self):
        """Generate random parameters for random search."""
        params = {}
        
        if self.config.clustering_algorithm == "kmeans":
            params['n_clusters'] = np.random.randint(self.config.min_clusters, self.config.max_clusters + 1)
        elif self.config.clustering_algorithm == "dbscan":
            params['eps'] = np.random.uniform(0.1, 2.0)
            params['min_samples'] = np.random.randint(2, 21)
        elif self.config.clustering_algorithm == "agglomerative":
            params['n_clusters'] = np.random.randint(self.config.min_clusters, self.config.max_clusters + 1)
        elif self.config.clustering_algorithm == "gmm":
            params['n_components'] = np.random.randint(self.config.min_clusters, self.config.max_clusters + 1)
        
        return params
    
    def _get_default_parameters(self):
        """Get default parameters for clustering algorithm."""
        if self.config.clustering_algorithm == "kmeans":
            return {'n_clusters': self.config.n_clusters}
        elif self.config.clustering_algorithm == "dbscan":
            return {'eps': self.config.eps, 'min_samples': self.config.min_samples}
        elif self.config.clustering_algorithm == "agglomerative":
            return {'n_clusters': self.config.n_clusters, 'linkage': self.config.linkage}
        elif self.config.clustering_algorithm == "gmm":
            return {'n_components': self.config.n_clusters, 'covariance_type': self.config.covariance_type}
        else:
            return {}
    
    def _create_clustering_model(self, params):
        """Create clustering model with given parameters."""
        algorithm = self.config.clustering_algorithm.lower()
        
        if algorithm == "kmeans":
            return KMeans(
                n_clusters=params.get('n_clusters', self.config.n_clusters),
                random_state=42,
                n_jobs=self.config.n_jobs if self.config.enable_parallel_processing else 1
            )
        elif algorithm == "dbscan":
            return DBSCAN(
                eps=params.get('eps', self.config.eps),
                min_samples=params.get('min_samples', self.config.min_samples),
                n_jobs=self.config.n_jobs if self.config.enable_parallel_processing else 1
            )
        elif algorithm == "agglomerative":
            return AgglomerativeClustering(
                n_clusters=params.get('n_clusters', self.config.n_clusters),
                linkage=params.get('linkage', self.config.linkage)
            )
        elif algorithm == "gmm":
            return GaussianMixture(
                n_components=params.get('n_components', self.config.n_clusters),
                covariance_type=params.get('covariance_type', self.config.covariance_type),
                random_state=42
            )
        else:
            raise ValueError(f"Unknown clustering algorithm: {algorithm}")
    
    def _calculate_performance_metrics(self, X, cluster_labels):
        """Calculate clustering performance metrics."""
        try:
            metrics = {}
            
            # Silhouette score
            if len(np.unique(cluster_labels)) > 1:
                metrics['silhouette_score'] = silhouette_score(X, cluster_labels)
            else:
                metrics['silhouette_score'] = 0.0
            
            # Calinski-Harabasz score
            if len(np.unique(cluster_labels)) > 1:
                metrics['calinski_harabasz_score'] = calinski_harabasz_score(X, cluster_labels)
            else:
                metrics['calinski_harabasz_score'] = 0.0
            
            # Davies-Bouldin score
            if len(np.unique(cluster_labels)) > 1:
                metrics['davies_bouldin_score'] = davies_bouldin_score(X, cluster_labels)
            else:
                metrics['davies_bouldin_score'] = 0.0
            
            # Inertia (for KMeans-like algorithms)
            if hasattr(self.best_model, 'inertia_'):
                metrics['inertia'] = self.best_model.inertia_
            else:
                metrics['inertia'] = 0.0
            
            return metrics
            
        except Exception as e:
            tprint_warning(f"⚠️ Performance metrics calculation failed: {e}")
            return {
                'silhouette_score': 0.0,
                'calinski_harabasz_score': 0.0,
                'davies_bouldin_score': 0.0,
                'inertia': 0.0
            }
    
    def _analyze_clusters(self, X, cluster_labels):
        """Analyze clustering results."""
        try:
            analysis = {}
            
            # Cluster sizes
            unique_labels, counts = np.unique(cluster_labels, return_counts=True)
            analysis['cluster_sizes'] = {int(label): int(count) for label, count in zip(unique_labels, counts)}
            
            # Cluster characteristics
            characteristics = {}
            for label in unique_labels:
                mask = cluster_labels == label
                cluster_data = X[mask]
                
                characteristics[f'cluster_{label}'] = {
                    'size': int(np.sum(mask)),
                    'mean': cluster_data.mean(axis=0).tolist(),
                    'std': cluster_data.std(axis=0).tolist(),
                    'min': cluster_data.min(axis=0).tolist(),
                    'max': cluster_data.max(axis=0).tolist()
                }
            
            analysis['characteristics'] = characteristics
            
            # Outlier analysis (for DBSCAN)
            if self.config.clustering_algorithm == "dbscan":
                n_outliers = np.sum(cluster_labels == -1)
                analysis['outliers'] = {
                    'n_outliers': int(n_outliers),
                    'outlier_ratio': float(n_outliers / len(cluster_labels))
                }
            else:
                analysis['outliers'] = None
            
            return analysis
            
        except Exception as e:
            tprint_warning(f"⚠️ Cluster analysis failed: {e}")
            return {}
    
    def _get_cluster_centers(self, model, X, cluster_labels):
        """Get cluster centers."""
        try:
            if hasattr(model, 'cluster_centers_'):
                return model.cluster_centers_
            else:
                # Calculate centers manually
                unique_labels = np.unique(cluster_labels)
                centers = []
                for label in unique_labels:
                    if label != -1:  # Skip noise points
                        mask = cluster_labels == label
                        center = X[mask].mean(axis=0)
                        centers.append(center)
                return np.array(centers) if centers else None
                
        except Exception as e:
            tprint_warning(f"⚠️ Cluster centers calculation failed: {e}")
            return None
    
    def _get_feature_importance(self, X, cluster_labels):
        """Get feature importance for clustering."""
        try:
            # Simple feature importance based on variance within clusters
            unique_labels = np.unique(cluster_labels)
            if len(unique_labels) < 2:
                return None
            
            feature_importance = []
            for feature_idx in range(X.shape[1]):
                feature_values = X[:, feature_idx]
                within_cluster_variance = 0.0
                
                for label in unique_labels:
                    if label != -1:  # Skip noise points
                        mask = cluster_labels == label
                        cluster_values = feature_values[mask]
                        if len(cluster_values) > 1:
                            within_cluster_variance += np.var(cluster_values)
                
                # Higher variance means more important for clustering
                feature_importance.append(within_cluster_variance)
            
            # Normalize importance scores
            feature_importance = np.array(feature_importance)
            if np.sum(feature_importance) > 0:
                feature_importance = feature_importance / np.sum(feature_importance)
            
            return feature_importance
            
        except Exception as e:
            tprint_warning(f"⚠️ Feature importance calculation failed: {e}")
            return None
    
    def _get_memory_usage(self):
        """Get memory usage."""
        try:
            if self.memory_optimizer and hasattr(self.memory_optimizer, 'get_memory_usage'):
                return self.memory_optimizer.get_memory_usage() / (1024 * 1024)  # Convert to MB
            else:
                return 0.0
        except Exception:
            return 0.0
    
    def _get_hardware_context(self):
        """Get hardware optimization context."""
        if self.config.enable_m1_optimization and SHARED_UTILS_AVAILABLE:
            return memory_checkpoint("agnostic_clustering")
        else:
            from contextlib import contextmanager
            @contextmanager
            def dummy_context():
                yield
            return dummy_context()
    
    def _save_clustering_results(self, result):
        """Save clustering results."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"agnostic_clustering_{timestamp}.{self.config.results_format}"
            filepath = Path(self.config.output_dir) / filename
            
            ensure_directory(filepath.parent)
            
            # Prepare data for serialization
            result_data = {
                'success': result.success,
                'cluster_labels': result.cluster_labels.tolist() if result.cluster_labels is not None else None,
                'n_clusters': result.n_clusters,
                'cluster_centers': result.cluster_centers.tolist() if result.cluster_centers is not None else None,
                'silhouette_score': result.silhouette_score,
                'calinski_harabasz_score': result.calinski_harabasz_score,
                'davies_bouldin_score': result.davies_bouldin_score,
                'inertia': result.inertia,
                'feature_importance': result.feature_importance.tolist() if result.feature_importance is not None else None,
                'selected_features': result.selected_features,
                'dimensionality_reduction_applied': result.dimensionality_reduction_applied,
                'n_components_used': result.n_components_used,
                'cluster_sizes': result.cluster_sizes,
                'cluster_characteristics': result.cluster_characteristics,
                'outlier_analysis': result.outlier_analysis,
                'clustering_time': result.clustering_time,
                'memory_usage_mb': result.memory_usage_mb,
                'n_samples': result.n_samples,
                'n_features_original': result.n_features_original,
                'n_features_used': result.n_features_used,
                'error_message': result.error_message,
                'warnings': result.warnings
            }
            
            if self.config.results_format == 'json':
                safe_json_dump(result_data, filepath)
            elif self.config.results_format == 'pickle':
                import pickle
                with open(filepath, 'wb') as f:
                    pickle.dump(result_data, f)
            
            tprint_success(f"💾 Clustering results saved to {filepath}")
            
        except Exception as e:
            tprint_warning(f"⚠️ Failed to save clustering results: {e}")
    
    def get_clustering_summary(self):
        """Get clustering summary."""
        if not self.clustering_history:
            return {'message': 'No clustering results available'}
        
        latest_result = self.clustering_history[-1]
        
        return {
            'total_clusterings': len(self.clustering_history),
            'latest_result': {
                'n_clusters': latest_result.n_clusters,
                'silhouette_score': latest_result.silhouette_score,
                'clustering_time': latest_result.clustering_time,
                'success': latest_result.success
            },
            'config': self.config.__dict__
        }

# Factory functions for creating architecture-specific clusterers
def create_nas_clusterer(config: Optional[AgnosticClusteringConfig] = None) -> AgnosticClusterer:
    """Create a NAS-specific clusterer."""
    if config is None:
        config = AgnosticClusteringConfig()
    
    # NAS-specific adaptations
    config.clustering_algorithm = "kmeans"  # Default for NAS
    config.enable_feature_selection = True
    config.feature_selection_method = "f_classif"
    config.enable_dimensionality_reduction = True
    config.reduction_method = "pca"
    
    return AgnosticClusterer(config)

def create_tas_clusterer(config: Optional[AgnosticClusteringConfig] = None) -> AgnosticClusterer:
    """Create a TAS-specific clusterer."""
    if config is None:
        config = AgnosticClusteringConfig()
    
    # TAS-specific adaptations
    config.clustering_algorithm = "dbscan"  # Default for TAS
    config.enable_feature_selection = True
    config.feature_selection_method = "f_regression"
    config.enable_dimensionality_reduction = True
    config.reduction_method = "ica"
    
    return AgnosticClusterer(config)