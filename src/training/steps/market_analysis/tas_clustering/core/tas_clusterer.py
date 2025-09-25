"""
TAS Clusterer - Tree Architecture Search Clustering System

TAS-specific clustering system using agnostic clustering with tree-specific
adaptations for feature extraction and clustering analysis.
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

# Import agnostic clustering
from src.utils.nas_tas.agnostic_clustering import (
    AgnosticClusterer, AgnosticClusteringConfig, AgnosticClusteringResult,
    create_tas_clusterer
)

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
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
    import xgboost as xgb
    import lightgbm as lgb
    TREE_LIBS_AVAILABLE = True
except ImportError:
    TREE_LIBS_AVAILABLE = False
    tprint_warning("Tree libraries not available, using fallback implementations")

logger = logging.getLogger(__name__)

@dataclass
class TASClusteringConfig:
    """Configuration for TAS clustering."""
    
    # Clustering parameters
    clustering_algorithm: str = "dbscan"  # dbscan, kmeans, agglomerative, gmm
    n_clusters: int = 5
    max_clusters: int = 20
    min_clusters: int = 2
    
    # DBSCAN parameters (default for TAS)
    eps: float = 0.5
    min_samples: int = 5
    
    # Tree-specific feature extraction
    enable_tree_feature_extraction: bool = True
    tree_feature_types: List[str] = field(default_factory=lambda: [
        'feature_importance', 'tree_depth', 'leaf_purity', 'node_splits'
    ])
    tree_ensemble_size: int = 10
    
    # Tree-specific clustering
    enable_tree_structure_clustering: bool = True
    tree_structure_metrics: List[str] = field(default_factory=lambda: [
        'depth_distribution', 'leaf_distribution', 'split_criteria'
    ])
    
    # Feature processing
    enable_feature_selection: bool = True
    n_features: Optional[int] = None
    feature_selection_method: str = "f_regression"  # TAS-specific
    
    # Dimensionality reduction
    enable_dimensionality_reduction: bool = True
    reduction_method: str = "ica"  # TAS-specific
    n_components: Optional[int] = None
    
    # Optimization
    enable_parameter_optimization: bool = True
    optimization_method: str = "bayesian_tpe"
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
    output_dir: str = "tas_clustering_results"
    results_format: str = "json"

@dataclass
class TASClusteringResult:
    """Result from TAS clustering."""
    
    # Basic clustering results
    success: bool
    cluster_labels: Optional[np.ndarray] = None
    n_clusters: int = 0
    cluster_centers: Optional[np.ndarray] = None
    
    # Performance metrics
    silhouette_score: float = 0.0
    calinski_harabasz_score: float = 0.0
    davies_bouldin_score: float = 0.0
    inertia: float = 0.0
    
    # Tree-specific results
    tree_feature_importance: Optional[np.ndarray] = None
    tree_structure_analysis: Optional[Dict[str, Any]] = None
    tree_ensemble_analysis: Optional[Dict[str, Any]] = None
    
    # Feature analysis
    feature_importance: Optional[np.ndarray] = None
    selected_features: Optional[List[int]] = None
    tree_features_used: Optional[List[str]] = None
    
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

class TASClusterer:
    """
    Tree Architecture Search Clusterer.
    
    TAS-specific clustering system using agnostic clustering with tree-specific
    adaptations for feature extraction and clustering analysis.
    """
    
    def __init__(self, config: Optional[TASClusteringConfig] = None):
        """Initialize TAS clusterer."""
        self.config = config or TASClusteringConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize agnostic clusterer with TAS-specific config
        self._init_agnostic_clusterer()
        
        # Initialize utilities
        self._init_utilities()
        
        # Clustering state
        self.clustering_history = []
        self.tree_models = []
        
        tprint_success("🚀 TAS Clusterer initialized")
        tprint_info(f"   → Algorithm: {self.config.clustering_algorithm}")
        tprint_info(f"   → Tree feature extraction: {'enabled' if self.config.enable_tree_feature_extraction else 'disabled'}")
        tprint_info(f"   → Tree structure clustering: {'enabled' if self.config.enable_tree_structure_clustering else 'disabled'}")
    
    def _init_agnostic_clusterer(self):
        """Initialize agnostic clusterer with TAS-specific configuration."""
        # Convert TAS config to agnostic config
        agnostic_config = AgnosticClusteringConfig(
            clustering_algorithm=self.config.clustering_algorithm,
            n_clusters=self.config.n_clusters,
            max_clusters=self.config.max_clusters,
            min_clusters=self.config.min_clusters,
            eps=self.config.eps,
            min_samples=self.config.min_samples,
            enable_feature_selection=self.config.enable_feature_selection,
            n_features=self.config.n_features,
            feature_selection_method=self.config.feature_selection_method,
            enable_dimensionality_reduction=self.config.enable_dimensionality_reduction,
            reduction_method=self.config.reduction_method,
            n_components=self.config.n_components,
            enable_parameter_optimization=self.config.enable_parameter_optimization,
            optimization_method=self.config.optimization_method,
            n_trials=self.config.n_trials,
            enable_m1_optimization=self.config.enable_m1_optimization,
            enable_parallel_processing=self.config.enable_parallel_processing,
            n_jobs=self.config.n_jobs,
            memory_limit_gb=self.config.memory_limit_gb,
            verbose=self.config.verbose,
            log_level=self.config.log_level,
            save_clustering_results=self.config.save_clustering_results,
            output_dir=self.config.output_dir,
            results_format=self.config.results_format
        )
        
        # Create TAS-specific agnostic clusterer
        self.agnostic_clusterer = create_tas_clusterer(agnostic_config)
    
    def _init_utilities(self):
        """Initialize utility components."""
        if SHARED_UTILS_AVAILABLE:
            self.math_validator = MathValidation()
            self.serializer = UniversalSerializer()
        else:
            self.math_validator = None
            self.serializer = None
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[Union[np.ndarray, pd.Series]] = None) -> TASClusteringResult:
        """
        Fit TAS clustering model to data.
        
        Args:
            X: Features to cluster
            y: Optional target variable for supervised feature selection
            
        Returns:
            TASClusteringResult with clustering results
        """
        start_time = time.time()
        tprint_info("🌳 Starting TAS clustering")
        
        try:
            # Validate inputs
            self._validate_inputs(X, y)
            
            # Extract tree-specific features if enabled
            if self.config.enable_tree_feature_extraction:
                X_tree_features = self._extract_tree_features(X, y)
                X_combined = self._combine_features(X, X_tree_features)
            else:
                X_combined = X
            
            # Apply tree structure clustering if enabled
            if self.config.enable_tree_structure_clustering:
                X_structure_features = self._extract_tree_structure_features(X, y)
                X_combined = self._combine_features(X_combined, X_structure_features)
            
            # Use agnostic clusterer for main clustering
            agnostic_result = self.agnostic_clusterer.fit(X_combined, y)
            
            # Extract tree-specific analysis
            tree_analysis = self._analyze_tree_clustering(X, y, agnostic_result.cluster_labels)
            
            # Create TAS-specific result
            result = TASClusteringResult(
                success=agnostic_result.success,
                cluster_labels=agnostic_result.cluster_labels,
                n_clusters=agnostic_result.n_clusters,
                cluster_centers=agnostic_result.cluster_centers,
                silhouette_score=agnostic_result.silhouette_score,
                calinski_harabasz_score=agnostic_result.calinski_harabasz_score,
                davies_bouldin_score=agnostic_result.davies_bouldin_score,
                inertia=agnostic_result.inertia,
                tree_feature_importance=tree_analysis.get('tree_feature_importance'),
                tree_structure_analysis=tree_analysis.get('tree_structure_analysis'),
                tree_ensemble_analysis=tree_analysis.get('tree_ensemble_analysis'),
                feature_importance=agnostic_result.feature_importance,
                selected_features=agnostic_result.selected_features,
                tree_features_used=self.config.tree_feature_types if self.config.enable_tree_feature_extraction else None,
                cluster_sizes=agnostic_result.cluster_sizes,
                cluster_characteristics=agnostic_result.cluster_characteristics,
                outlier_analysis=agnostic_result.outlier_analysis,
                clustering_time=time.time() - start_time,
                memory_usage_mb=agnostic_result.memory_usage_mb,
                n_samples=agnostic_result.n_samples,
                n_features_original=agnostic_result.n_features_original,
                n_features_used=agnostic_result.n_features_used,
                error_message=agnostic_result.error_message,
                warnings=agnostic_result.warnings
            )
            
            # Store results
            self.clustering_history.append(result)
            
            tprint_success(f"✅ TAS clustering completed in {result.clustering_time:.2f}s")
            tprint_info(f"   → Clusters found: {result.n_clusters}")
            tprint_info(f"   → Silhouette score: {result.silhouette_score:.4f}")
            tprint_info(f"   → Tree features used: {len(self.config.tree_feature_types) if self.config.enable_tree_feature_extraction else 0}")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            tprint_error(f"❌ TAS clustering failed: {e}")
            
            return TASClusteringResult(
                success=False,
                clustering_time=execution_time,
                error_message=str(e)
            )
    
    def _validate_inputs(self, X, y):
        """Validate input data."""
        if not TREE_LIBS_AVAILABLE:
            raise ImportError("Tree libraries not available")
        
        if X is None:
            raise ValueError("X cannot be None")
        
        if len(X) == 0:
            raise ValueError("X cannot be empty")
    
    def _extract_tree_features(self, X, y):
        """Extract tree-specific features."""
        try:
            tree_features = []
            
            # Create tree ensemble for feature extraction
            if y is not None:
                # Determine if classification or regression
                is_classification = len(np.unique(y)) < len(y) * 0.1  # Simple heuristic
                
                if is_classification:
                    tree_model = RandomForestClassifier(
                        n_estimators=self.config.tree_ensemble_size,
                        random_state=42,
                        n_jobs=self.config.n_jobs if self.config.enable_parallel_processing else 1
                    )
                else:
                    tree_model = RandomForestRegressor(
                        n_estimators=self.config.tree_ensemble_size,
                        random_state=42,
                        n_jobs=self.config.n_jobs if self.config.enable_parallel_processing else 1
                    )
                
                # Fit tree model
                tree_model.fit(X, y)
                self.tree_models.append(tree_model)
                
                # Extract features based on configuration
                for feature_type in self.config.tree_feature_types:
                    if feature_type == 'feature_importance':
                        if hasattr(tree_model, 'feature_importances_'):
                            tree_features.append(tree_model.feature_importances_)
                    
                    elif feature_type == 'tree_depth':
                        depths = [tree.tree_.max_depth for tree in tree_model.estimators_]
                        tree_features.append(depths)
                    
                    elif feature_type == 'leaf_purity':
                        purities = self._calculate_leaf_purities(tree_model, X, y)
                        tree_features.append(purities)
                    
                    elif feature_type == 'node_splits':
                        splits = self._calculate_node_splits(tree_model, X)
                        tree_features.append(splits)
            
            # Combine tree features
            if tree_features:
                tree_features_array = np.column_stack(tree_features)
                tprint_info(f"✅ Tree features extracted: {tree_features_array.shape}")
                return tree_features_array
            else:
                tprint_warning("⚠️ No tree features extracted")
                return np.array([]).reshape(X.shape[0], 0)
                
        except Exception as e:
            tprint_warning(f"⚠️ Tree feature extraction failed: {e}")
            return np.array([]).reshape(X.shape[0], 0)
    
    def _extract_tree_structure_features(self, X, y):
        """Extract tree structure features."""
        try:
            structure_features = []
            
            # Extract structure metrics
            for metric in self.config.tree_structure_metrics:
                if metric == 'depth_distribution':
                    if self.tree_models:
                        depths = [tree.tree_.max_depth for tree in self.tree_models[-1].estimators_]
                        structure_features.append(depths)
                
                elif metric == 'leaf_distribution':
                    if self.tree_models:
                        leaves = [tree.tree_.n_leaves for tree in self.tree_models[-1].estimators_]
                        structure_features.append(leaves)
                
                elif metric == 'split_criteria':
                    if self.tree_models:
                        criteria = self._extract_split_criteria(self.tree_models[-1], X)
                        structure_features.append(criteria)
            
            # Combine structure features
            if structure_features:
                structure_features_array = np.column_stack(structure_features)
                tprint_info(f"✅ Tree structure features extracted: {structure_features_array.shape}")
                return structure_features_array
            else:
                tprint_warning("⚠️ No tree structure features extracted")
                return np.array([]).reshape(X.shape[0], 0)
                
        except Exception as e:
            tprint_warning(f"⚠️ Tree structure feature extraction failed: {e}")
            return np.array([]).reshape(X.shape[0], 0)
    
    def _calculate_leaf_purities(self, tree_model, X, y):
        """Calculate leaf purities for each sample."""
        try:
            purities = []
            
            for tree in tree_model.estimators_:
                leaf_ids = tree.apply(X)
                tree_purities = []
                
                for leaf_id in np.unique(leaf_ids):
                    mask = leaf_ids == leaf_id
                    leaf_labels = y[mask]
                    
                    if len(leaf_labels) > 0:
                        # Calculate purity as the proportion of the most common class
                        unique, counts = np.unique(leaf_labels, return_counts=True)
                        purity = np.max(counts) / len(leaf_labels)
                        tree_purities.append(purity)
                
                purities.append(np.mean(tree_purities) if tree_purities else 0.0)
            
            return purities
            
        except Exception as e:
            tprint_warning(f"⚠️ Leaf purity calculation failed: {e}")
            return [0.0] * len(tree_model.estimators_)
    
    def _calculate_node_splits(self, tree_model, X):
        """Calculate node split information."""
        try:
            splits = []
            
            for tree in tree_model.estimators_:
                tree_splits = []
                
                for i in range(tree.tree_.node_count):
                    if tree.tree_.children_left[i] != tree.tree_.children_right[i]:  # Not a leaf
                        feature = tree.tree_.feature[i]
                        threshold = tree.tree_.threshold[i]
                        tree_splits.append((feature, threshold))
                
                splits.append(len(tree_splits))  # Number of splits
            
            return splits
            
        except Exception as e:
            tprint_warning(f"⚠️ Node split calculation failed: {e}")
            return [0] * len(tree_model.estimators_)
    
    def _extract_split_criteria(self, tree_model, X):
        """Extract split criteria information."""
        try:
            criteria = []
            
            for tree in tree_model.estimators_:
                tree_criteria = []
                
                for i in range(tree.tree_.node_count):
                    if tree.tree_.children_left[i] != tree.tree_.children_right[i]:  # Not a leaf
                        feature = tree.tree_.feature[i]
                        tree_criteria.append(feature)
                
                criteria.append(len(set(tree_criteria)))  # Number of unique features used
            
            return criteria
            
        except Exception as e:
            tprint_warning(f"⚠️ Split criteria extraction failed: {e}")
            return [0] * len(tree_model.estimators_)
    
    def _combine_features(self, X, additional_features):
        """Combine original features with additional features."""
        try:
            if additional_features.size == 0:
                return X
            
            # Convert to numpy arrays
            if isinstance(X, pd.DataFrame):
                X_array = X.values
            else:
                X_array = np.array(X)
            
            # Combine features
            combined = np.column_stack([X_array, additional_features])
            
            tprint_info(f"✅ Features combined: {X_array.shape} + {additional_features.shape} = {combined.shape}")
            return combined
            
        except Exception as e:
            tprint_warning(f"⚠️ Feature combination failed: {e}")
            return X
    
    def _analyze_tree_clustering(self, X, y, cluster_labels):
        """Analyze tree-specific clustering results."""
        try:
            analysis = {}
            
            # Tree feature importance analysis
            if self.tree_models and hasattr(self.tree_models[-1], 'feature_importances_'):
                analysis['tree_feature_importance'] = self.tree_models[-1].feature_importances_
            
            # Tree structure analysis
            structure_analysis = {}
            if self.tree_models:
                tree_model = self.tree_models[-1]
                
                # Analyze tree structure per cluster
                for cluster_id in np.unique(cluster_labels):
                    if cluster_id != -1:  # Skip noise points
                        mask = cluster_labels == cluster_id
                        cluster_data = X[mask]
                        
                        if len(cluster_data) > 0:
                            # Analyze tree structure for this cluster
                            cluster_structure = {
                                'n_samples': int(np.sum(mask)),
                                'avg_tree_depth': np.mean([tree.tree_.max_depth for tree in tree_model.estimators_]),
                                'avg_n_leaves': np.mean([tree.tree_.n_leaves for tree in tree_model.estimators_]),
                                'feature_usage': self._analyze_feature_usage(tree_model, cluster_data)
                            }
                            structure_analysis[f'cluster_{cluster_id}'] = cluster_structure
            
            analysis['tree_structure_analysis'] = structure_analysis
            
            # Tree ensemble analysis
            ensemble_analysis = {}
            if self.tree_models:
                tree_model = self.tree_models[-1]
                ensemble_analysis = {
                    'n_estimators': len(tree_model.estimators_),
                    'avg_depth': np.mean([tree.tree_.max_depth for tree in tree_model.estimators_]),
                    'avg_leaves': np.mean([tree.tree_.n_leaves for tree in tree_model.estimators_]),
                    'depth_std': np.std([tree.tree_.max_depth for tree in tree_model.estimators_]),
                    'leaves_std': np.std([tree.tree_.n_leaves for tree in tree_model.estimators_])
                }
            
            analysis['tree_ensemble_analysis'] = ensemble_analysis
            
            return analysis
            
        except Exception as e:
            tprint_warning(f"⚠️ Tree clustering analysis failed: {e}")
            return {}
    
    def _analyze_feature_usage(self, tree_model, X):
        """Analyze feature usage in tree model."""
        try:
            feature_usage = {}
            
            for tree in tree_model.estimators_:
                for i in range(tree.tree_.node_count):
                    if tree.tree_.children_left[i] != tree.tree_.children_right[i]:  # Not a leaf
                        feature = tree.tree_.feature[i]
                        if feature >= 0:
                            feature_usage[f'feature_{feature}'] = feature_usage.get(f'feature_{feature}', 0) + 1
            
            return feature_usage
            
        except Exception as e:
            tprint_warning(f"⚠️ Feature usage analysis failed: {e}")
            return {}
    
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
                'success': latest_result.success,
                'tree_features_used': latest_result.tree_features_used
            },
            'config': self.config.__dict__,
            'tree_models_count': len(self.tree_models)
        }