"""
NAS Clusterer - Neural Architecture Search Clustering System

This module provides comprehensive NAS clustering functionality for market analysis,
including architecture search, clustering, optimization, and evaluation capabilities.

Key Features:
- Multi-objective architecture search
- Clustering-based architecture grouping
- Performance optimization with Bayesian TPE
- Hardware-aware optimization (M1 GPU/CPU)
- Comprehensive evaluation metrics
- Visualization and reporting
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path

# Import shared utilities
from src.utils.common_operations import (
    safe_dataframe_operation, validate_dataframe_columns, 
    calculate_data_quality_metrics, get_m1_gpu_manager, get_m1_memory_optimizer
)
from src.utils.common_utilities import safe_dataframe_operation as safe_df_op
from src.utils.math_validation import safe_divide, safe_log, safe_sqrt, validate_finite
from src.utils.serialization_utils import save_object, load_object
from src.utils.tprint import tprint

# Import ML utilities
from src.utils.ml_common.optimization.bayesian_tpe_optimizer import BayesianTPEOptimizer
from src.utils.ml_common.common_operations import create_ml_pipeline

# Import hardware utilities
try:
    from src.utils.hardware.m1_gpu_utils import is_m1_available, is_mps_available, get_m1_gpu_manager
    from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
    from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer
    HARDWARE_UTILS_AVAILABLE = True
except ImportError:
    HARDWARE_UTILS_AVAILABLE = False
    def is_m1_available(): return False
    def is_mps_available(): return False
    def get_m1_gpu_manager(): return None
    def get_m1_memory_optimizer(): return None
    def get_m1_cpu_optimizer(): return None

# Import matrix operations
try:
    from src.utils.matrix_operations import MatrixOperations
    MATRIX_OPS_AVAILABLE = True
except ImportError:
    MATRIX_OPS_AVAILABLE = False
    MatrixOperations = None

# Setup logging
logger = logging.getLogger(__name__)

@dataclass
class ArchitectureConfig:
    """Configuration for neural architecture."""
    input_dim: int
    output_dim: int
    hidden_dims: List[int]
    activation: str = 'relu'
    dropout: float = 0.1
    batch_norm: bool = True
    learning_rate: float = 0.001
    optimizer: str = 'adam'
    loss_function: str = 'mse'
    
@dataclass
class ClusteringConfig:
    """Configuration for clustering parameters."""
    n_clusters: int = 5
    algorithm: str = 'kmeans'  # kmeans, hierarchical, dbscan
    distance_metric: str = 'euclidean'
    linkage: str = 'ward'  # for hierarchical clustering
    min_samples: int = 5  # for DBSCAN
    eps: float = 0.5  # for DBSCAN
    
@dataclass
class OptimizationConfig:
    """Configuration for optimization parameters."""
    n_trials: int = 100
    timeout: int = 3600  # seconds
    early_stopping_rounds: int = 10
    use_bayesian: bool = True
    use_grid_search: bool = True
    parallel_trials: int = 4
    memory_limit: Optional[int] = None

class NASClusterer:
    """
    Neural Architecture Search Clustering System.
    
    This class provides comprehensive NAS clustering functionality including:
    - Architecture search and generation
    - Clustering of architectures
    - Performance optimization
    - Hardware-aware optimization
    - Evaluation and reporting
    """
    
    def __init__(self, 
                 architecture_config: Optional[ArchitectureConfig] = None,
                 clustering_config: Optional[ClusteringConfig] = None,
                 optimization_config: Optional[OptimizationConfig] = None,
                 use_hardware_optimization: bool = True,
                 verbose: bool = True):
        """
        Initialize NAS Clusterer.
        
        Args:
            architecture_config: Configuration for neural architectures
            clustering_config: Configuration for clustering parameters
            optimization_config: Configuration for optimization parameters
            use_hardware_optimization: Whether to use hardware-specific optimizations
            verbose: Whether to enable verbose logging
        """
        self.architecture_config = architecture_config or ArchitectureConfig(
            input_dim=10, output_dim=1, hidden_dims=[64, 32]
        )
        self.clustering_config = clustering_config or ClusteringConfig()
        self.optimization_config = optimization_config or OptimizationConfig()
        self.use_hardware_optimization = use_hardware_optimization
        self.verbose = verbose
        
        # Initialize hardware managers
        self.gpu_manager = None
        self.memory_optimizer = None
        self.cpu_optimizer = None
        
        if self.use_hardware_optimization and HARDWARE_UTILS_AVAILABLE:
            self._initialize_hardware_managers()
        
        # Initialize optimization components
        self.optimizer = None
        self.matrix_ops = None
        
        if MATRIX_OPS_AVAILABLE:
            self.matrix_ops = MatrixOperations()
        
        # Storage for results
        self.architectures = []
        self.clusters = {}
        self.optimization_results = {}
        self.evaluation_metrics = {}
        
        # Initialize logging
        if self.verbose:
            tprint("🚀 NAS Clusterer initialized successfully")
            if self.use_hardware_optimization:
                tprint(f"🔧 Hardware optimization: {'Enabled' if HARDWARE_UTILS_AVAILABLE else 'Disabled'}")
    
    def _initialize_hardware_managers(self):
        """Initialize hardware-specific managers."""
        try:
            if is_m1_available():
                self.gpu_manager = get_m1_gpu_manager()
                self.memory_optimizer = get_m1_memory_optimizer()
                self.cpu_optimizer = get_m1_cpu_optimizer()
                
                if self.verbose:
                    tprint("🍎 M1 hardware optimization enabled")
            else:
                if self.verbose:
                    tprint("⚠️ M1 hardware not detected, using standard optimization")
        except Exception as e:
            logger.warning(f"Failed to initialize hardware managers: {e}")
    
    def generate_architectures(self, n_architectures: int = 100) -> List[Dict[str, Any]]:
        """
        Generate a set of neural architectures for clustering.
        
        Args:
            n_architectures: Number of architectures to generate
            
        Returns:
            List of architecture dictionaries
        """
        if self.verbose:
            tprint(f"🏗️ Generating {n_architectures} architectures...")
        
        architectures = []
        
        for i in range(n_architectures):
            # Generate random architecture parameters
            n_layers = np.random.randint(2, 8)
            hidden_dims = []
            
            for _ in range(n_layers):
                dim = np.random.choice([32, 64, 128, 256, 512])
                hidden_dims.append(dim)
            
            # Random activation function
            activation = np.random.choice(['relu', 'tanh', 'sigmoid', 'leaky_relu'])
            
            # Random dropout rate
            dropout = np.random.uniform(0.0, 0.5)
            
            # Random learning rate
            learning_rate = 10 ** np.random.uniform(-4, -2)
            
            architecture = {
                'id': i,
                'input_dim': self.architecture_config.input_dim,
                'output_dim': self.architecture_config.output_dim,
                'hidden_dims': hidden_dims,
                'activation': activation,
                'dropout': dropout,
                'batch_norm': np.random.choice([True, False]),
                'learning_rate': learning_rate,
                'optimizer': np.random.choice(['adam', 'sgd', 'rmsprop']),
                'loss_function': np.random.choice(['mse', 'mae', 'huber']),
                'complexity': sum(hidden_dims),  # Simple complexity metric
                'n_layers': n_layers
            }
            
            architectures.append(architecture)
        
        self.architectures = architectures
        
        if self.verbose:
            tprint(f"✅ Generated {len(architectures)} architectures")
        
        return architectures
    
    def cluster_architectures(self, architectures: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Cluster architectures based on their characteristics.
        
        Args:
            architectures: List of architectures to cluster (uses self.architectures if None)
            
        Returns:
            Dictionary containing clustering results
        """
        if architectures is None:
            architectures = self.architectures
        
        if not architectures:
            raise ValueError("No architectures provided for clustering")
        
        if self.verbose:
            tprint(f"🔍 Clustering {len(architectures)} architectures...")
        
        # Extract features for clustering
        features = self._extract_architecture_features(architectures)
        
        # Perform clustering
        if self.clustering_config.algorithm == 'kmeans':
            cluster_labels = self._kmeans_clustering(features)
        elif self.clustering_config.algorithm == 'hierarchical':
            cluster_labels = self._hierarchical_clustering(features)
        elif self.clustering_config.algorithm == 'dbscan':
            cluster_labels = self._dbscan_clustering(features)
        else:
            raise ValueError(f"Unsupported clustering algorithm: {self.clustering_config.algorithm}")
        
        # Organize results
        clusters = {}
        for i, (arch, label) in enumerate(zip(architectures, cluster_labels)):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(arch)
        
        # Calculate cluster statistics
        cluster_stats = self._calculate_cluster_statistics(clusters)
        
        self.clusters = {
            'clusters': clusters,
            'labels': cluster_labels,
            'features': features,
            'statistics': cluster_stats,
            'algorithm': self.clustering_config.algorithm,
            'n_clusters': len(set(cluster_labels))
        }
        
        if self.verbose:
            tprint(f"✅ Clustered into {len(set(cluster_labels))} clusters")
            for cluster_id, cluster_archs in clusters.items():
                tprint(f"   Cluster {cluster_id}: {len(cluster_archs)} architectures")
        
        return self.clusters
    
    def _extract_architecture_features(self, architectures: List[Dict[str, Any]]) -> np.ndarray:
        """Extract numerical features from architectures for clustering."""
        features = []
        
        for arch in architectures:
            feature_vector = [
                arch['n_layers'],
                arch['complexity'],
                arch['dropout'],
                np.log(arch['learning_rate']),
                1 if arch['batch_norm'] else 0,
                len(arch['hidden_dims']),
                np.mean(arch['hidden_dims']),
                np.std(arch['hidden_dims']),
                np.max(arch['hidden_dims']),
                np.min(arch['hidden_dims'])
            ]
            features.append(feature_vector)
        
        return np.array(features)
    
    def _kmeans_clustering(self, features: np.ndarray) -> np.ndarray:
        """Perform K-means clustering."""
        from sklearn.cluster import KMeans
        
        kmeans = KMeans(
            n_clusters=self.clustering_config.n_clusters,
            random_state=42,
            n_init=10
        )
        return kmeans.fit_predict(features)
    
    def _hierarchical_clustering(self, features: np.ndarray) -> np.ndarray:
        """Perform hierarchical clustering."""
        from sklearn.cluster import AgglomerativeClustering
        
        clustering = AgglomerativeClustering(
            n_clusters=self.clustering_config.n_clusters,
            linkage=self.clustering_config.linkage
        )
        return clustering.fit_predict(features)
    
    def _dbscan_clustering(self, features: np.ndarray) -> np.ndarray:
        """Perform DBSCAN clustering."""
        from sklearn.cluster import DBSCAN
        
        clustering = DBSCAN(
            eps=self.clustering_config.eps,
            min_samples=self.clustering_config.min_samples
        )
        return clustering.fit_predict(features)
    
    def _calculate_cluster_statistics(self, clusters: Dict[int, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Calculate statistics for each cluster."""
        stats = {}
        
        for cluster_id, cluster_archs in clusters.items():
            if not cluster_archs:
                continue
                
            complexities = [arch['complexity'] for arch in cluster_archs]
            n_layers = [arch['n_layers'] for arch in cluster_archs]
            dropouts = [arch['dropout'] for arch in cluster_archs]
            learning_rates = [arch['learning_rate'] for arch in cluster_archs]
            
            stats[cluster_id] = {
                'size': len(cluster_archs),
                'avg_complexity': np.mean(complexities),
                'std_complexity': np.std(complexities),
                'avg_layers': np.mean(n_layers),
                'avg_dropout': np.mean(dropouts),
                'avg_learning_rate': np.mean(learning_rates),
                'complexity_range': [np.min(complexities), np.max(complexities)]
            }
        
        return stats
    
    def optimize_architectures(self, 
                             objective_function: Callable,
                             search_space: Dict[str, Any],
                             data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Optimize architectures using Bayesian TPE optimization.
        
        Args:
            objective_function: Function to optimize
            search_space: Search space definition
            data: Optional data for optimization
            
        Returns:
            Optimization results
        """
        if self.verbose:
            tprint("🎯 Starting architecture optimization...")
        
        # Initialize optimizer
        if self.optimizer is None:
            self.optimizer = BayesianTPEOptimizer(
                config=self.optimization_config,
                use_hardware_optimization=self.use_hardware_optimization
            )
        
        # Run optimization
        try:
            results = self.optimizer.optimize(
                objective_function=objective_function,
                search_space=search_space,
                data=data
            )
            
            self.optimization_results = results
            
            if self.verbose:
                tprint(f"✅ Optimization completed with best score: {results.get('best_score', 'N/A')}")
            
            return results
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            raise
    
    def evaluate_clusters(self, 
                         test_data: Optional[pd.DataFrame] = None,
                         metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Evaluate the quality of clusters.
        
        Args:
            test_data: Test data for evaluation
            metrics: List of metrics to calculate
            
        Returns:
            Evaluation results
        """
        if not self.clusters:
            raise ValueError("No clusters available for evaluation. Run cluster_architectures() first.")
        
        if self.verbose:
            tprint("📊 Evaluating cluster quality...")
        
        if metrics is None:
            metrics = ['silhouette_score', 'calinski_harabasz_score', 'davies_bouldin_score']
        
        evaluation_results = {}
        
        # Calculate clustering metrics
        if 'silhouette_score' in metrics:
            from sklearn.metrics import silhouette_score
            evaluation_results['silhouette_score'] = silhouette_score(
                self.clusters['features'], 
                self.clusters['labels']
            )
        
        if 'calinski_harabasz_score' in metrics:
            from sklearn.metrics import calinski_harabasz_score
            evaluation_results['calinski_harabasz_score'] = calinski_harabasz_score(
                self.clusters['features'], 
                self.clusters['labels']
            )
        
        if 'davies_bouldin_score' in metrics:
            from sklearn.metrics import davies_bouldin_score
            evaluation_results['davies_bouldin_score'] = davies_bouldin_score(
                self.clusters['features'], 
                self.clusters['labels']
            )
        
        # Calculate cluster balance
        cluster_sizes = [len(cluster) for cluster in self.clusters['clusters'].values()]
        evaluation_results['cluster_balance'] = {
            'min_size': min(cluster_sizes),
            'max_size': max(cluster_sizes),
            'std_size': np.std(cluster_sizes),
            'balance_ratio': min(cluster_sizes) / max(cluster_sizes) if max(cluster_sizes) > 0 else 0
        }
        
        self.evaluation_metrics = evaluation_results
        
        if self.verbose:
            tprint("✅ Cluster evaluation completed")
            for metric, value in evaluation_results.items():
                if isinstance(value, dict):
                    tprint(f"   {metric}: {value}")
                else:
                    tprint(f"   {metric}: {value:.4f}")
        
        return evaluation_results
    
    def get_best_architectures(self, n_best: int = 5) -> List[Dict[str, Any]]:
        """
        Get the best architectures from each cluster.
        
        Args:
            n_best: Number of best architectures to return per cluster
            
        Returns:
            List of best architectures
        """
        if not self.clusters:
            raise ValueError("No clusters available. Run cluster_architectures() first.")
        
        best_architectures = []
        
        for cluster_id, cluster_archs in self.clusters['clusters'].items():
            # Sort by complexity (assuming lower complexity is better for this example)
            sorted_archs = sorted(cluster_archs, key=lambda x: x['complexity'])
            best_architectures.extend(sorted_archs[:n_best])
        
        return best_architectures
    
    def save_results(self, filepath: str) -> None:
        """Save clustering results to file."""
        results = {
            'architectures': self.architectures,
            'clusters': self.clusters,
            'optimization_results': self.optimization_results,
            'evaluation_metrics': self.evaluation_metrics,
            'config': {
                'architecture': self.architecture_config.__dict__,
                'clustering': self.clustering_config.__dict__,
                'optimization': self.optimization_config.__dict__
            }
        }
        
        save_object(results, filepath)
        
        if self.verbose:
            tprint(f"💾 Results saved to {filepath}")
    
    def load_results(self, filepath: str) -> None:
        """Load clustering results from file."""
        results = load_object(filepath)
        
        self.architectures = results.get('architectures', [])
        self.clusters = results.get('clusters', {})
        self.optimization_results = results.get('optimization_results', {})
        self.evaluation_metrics = results.get('evaluation_metrics', {})
        
        if self.verbose:
            tprint(f"📁 Results loaded from {filepath}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the clustering results."""
        if not self.clusters:
            return {"error": "No clustering results available"}
        
        summary = {
            'total_architectures': len(self.architectures),
            'n_clusters': self.clusters.get('n_clusters', 0),
            'cluster_sizes': {str(k): len(v) for k, v in self.clusters.get('clusters', {}).items()},
            'evaluation_metrics': self.evaluation_metrics,
            'optimization_completed': bool(self.optimization_results),
            'hardware_optimization': self.use_hardware_optimization
        }
        
        return summary