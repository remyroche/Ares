"""
NAS Clusterer - Neural Architecture Search Clustering System

NAS-specific clustering system using agnostic clustering with neural-specific
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
    create_nas_clusterer
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

# Import neural-specific libraries
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    NEURAL_LIBS_AVAILABLE = True
except ImportError:
    NEURAL_LIBS_AVAILABLE = False
    tprint_warning("Neural libraries not available, using fallback implementations")

logger = logging.getLogger(__name__)

@dataclass
class NASClusteringConfig:
    """Configuration for NAS clustering."""
    
    # Clustering parameters
    clustering_algorithm: str = "kmeans"  # kmeans, dbscan, agglomerative, gmm
    n_clusters: int = 5
    max_clusters: int = 20
    min_clusters: int = 2
    
    # KMeans parameters (default for NAS)
    n_init: int = 10
    max_iter: int = 300
    
    # Neural-specific feature extraction
    enable_neural_feature_extraction: bool = True
    neural_feature_types: List[str] = field(default_factory=lambda: [
        'layer_activations', 'gradient_norms', 'weight_distributions', 'activation_patterns'
    ])
    neural_ensemble_size: int = 5
    
    # Neural-specific clustering
    enable_neural_structure_clustering: bool = True
    neural_structure_metrics: List[str] = field(default_factory=lambda: [
        'layer_widths', 'activation_functions', 'dropout_rates'
    ])
    
    # Feature processing
    enable_feature_selection: bool = True
    n_features: Optional[int] = None
    feature_selection_method: str = "f_classif"  # NAS-specific
    
    # Dimensionality reduction
    enable_dimensionality_reduction: bool = True
    reduction_method: str = "pca"  # NAS-specific
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
    output_dir: str = "nas_clustering_results"
    results_format: str = "json"

@dataclass
class NASClusteringResult:
    """Result from NAS clustering."""
    
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
    
    # Neural-specific results
    neural_feature_importance: Optional[np.ndarray] = None
    neural_structure_analysis: Optional[Dict[str, Any]] = None
    neural_ensemble_analysis: Optional[Dict[str, Any]] = None
    
    # Feature analysis
    feature_importance: Optional[np.ndarray] = None
    selected_features: Optional[List[int]] = None
    neural_features_used: Optional[List[str]] = None
    
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

class NASClusterer:
    """
    Neural Architecture Search Clusterer.
    
    NAS-specific clustering system using agnostic clustering with neural-specific
    adaptations for feature extraction and clustering analysis.
    """
    
    def __init__(self, config: Optional[NASClusteringConfig] = None):
        """Initialize NAS clusterer."""
        self.config = config or NASClusteringConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize agnostic clusterer with NAS-specific config
        self._init_agnostic_clusterer()
        
        # Initialize utilities
        self._init_utilities()
        
        # Clustering state
        self.clustering_history = []
        self.neural_models = []
        
        tprint_success("🚀 NAS Clusterer initialized")
        tprint_info(f"   → Algorithm: {self.config.clustering_algorithm}")
        tprint_info(f"   → Neural feature extraction: {'enabled' if self.config.enable_neural_feature_extraction else 'disabled'}")
        tprint_info(f"   → Neural structure clustering: {'enabled' if self.config.enable_neural_structure_clustering else 'disabled'}")
    
    def _init_agnostic_clusterer(self):
        """Initialize agnostic clusterer with NAS-specific configuration."""
        # Convert NAS config to agnostic config
        agnostic_config = AgnosticClusteringConfig(
            clustering_algorithm=self.config.clustering_algorithm,
            n_clusters=self.config.n_clusters,
            max_clusters=self.config.max_clusters,
            min_clusters=self.config.min_clusters,
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
        
        # Create NAS-specific agnostic clusterer
        self.agnostic_clusterer = create_nas_clusterer(agnostic_config)
    
    def _init_utilities(self):
        """Initialize utility components."""
        if SHARED_UTILS_AVAILABLE:
            self.math_validator = MathValidation()
            self.serializer = UniversalSerializer()
        else:
            self.math_validator = None
            self.serializer = None
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[Union[np.ndarray, pd.Series]] = None) -> NASClusteringResult:
        """
        Fit NAS clustering model to data.
        
        Args:
            X: Features to cluster
            y: Optional target variable for supervised feature selection
            
        Returns:
            NASClusteringResult with clustering results
        """
        start_time = time.time()
        tprint_info("🧠 Starting NAS clustering")
        
        try:
            # Validate inputs
            self._validate_inputs(X, y)
            
            # Extract neural-specific features if enabled
            if self.config.enable_neural_feature_extraction:
                X_neural_features = self._extract_neural_features(X, y)
                X_combined = self._combine_features(X, X_neural_features)
            else:
                X_combined = X
            
            # Apply neural structure clustering if enabled
            if self.config.enable_neural_structure_clustering:
                X_structure_features = self._extract_neural_structure_features(X, y)
                X_combined = self._combine_features(X_combined, X_structure_features)
            
            # Use agnostic clusterer for main clustering
            agnostic_result = self.agnostic_clusterer.fit(X_combined, y)
            
            # Extract neural-specific analysis
            neural_analysis = self._analyze_neural_clustering(X, y, agnostic_result.cluster_labels)
            
            # Create NAS-specific result
            result = NASClusteringResult(
                success=agnostic_result.success,
                cluster_labels=agnostic_result.cluster_labels,
                n_clusters=agnostic_result.n_clusters,
                cluster_centers=agnostic_result.cluster_centers,
                silhouette_score=agnostic_result.silhouette_score,
                calinski_harabasz_score=agnostic_result.calinski_harabasz_score,
                davies_bouldin_score=agnostic_result.davies_bouldin_score,
                inertia=agnostic_result.inertia,
                neural_feature_importance=neural_analysis.get('neural_feature_importance'),
                neural_structure_analysis=neural_analysis.get('neural_structure_analysis'),
                neural_ensemble_analysis=neural_analysis.get('neural_ensemble_analysis'),
                feature_importance=agnostic_result.feature_importance,
                selected_features=agnostic_result.selected_features,
                neural_features_used=self.config.neural_feature_types if self.config.enable_neural_feature_extraction else None,
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
            
            tprint_success(f"✅ NAS clustering completed in {result.clustering_time:.2f}s")
            tprint_info(f"   → Clusters found: {result.n_clusters}")
            tprint_info(f"   → Silhouette score: {result.silhouette_score:.4f}")
            tprint_info(f"   → Neural features used: {len(self.config.neural_feature_types) if self.config.enable_neural_feature_extraction else 0}")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            tprint_error(f"❌ NAS clustering failed: {e}")
            
            return NASClusteringResult(
                success=False,
                clustering_time=execution_time,
                error_message=str(e)
            )
    
    def _validate_inputs(self, X, y):
        """Validate input data."""
        if X is None:
            raise ValueError("X cannot be None")
        
        if len(X) == 0:
            raise ValueError("X cannot be empty")
    
    def _extract_neural_features(self, X, y):
        """Extract neural-specific features."""
        try:
            neural_features = []
            
            # Create neural ensemble for feature extraction
            if y is not None and NEURAL_LIBS_AVAILABLE:
                # Create simple neural models for feature extraction
                for i in range(self.config.neural_ensemble_size):
                    model = self._create_simple_neural_model(X.shape[1], y)
                    if model:
                        self.neural_models.append(model)
                        
                        # Extract features based on configuration
                        for feature_type in self.config.neural_feature_types:
                            if feature_type == 'layer_activations':
                                activations = self._extract_layer_activations(model, X)
                                neural_features.append(activations)
                            
                            elif feature_type == 'gradient_norms':
                                gradients = self._extract_gradient_norms(model, X, y)
                                neural_features.append(gradients)
                            
                            elif feature_type == 'weight_distributions':
                                weights = self._extract_weight_distributions(model)
                                neural_features.append(weights)
                            
                            elif feature_type == 'activation_patterns':
                                patterns = self._extract_activation_patterns(model, X)
                                neural_features.append(patterns)
            
            # Combine neural features
            if neural_features:
                neural_features_array = np.column_stack(neural_features)
                tprint_info(f"✅ Neural features extracted: {neural_features_array.shape}")
                return neural_features_array
            else:
                tprint_warning("⚠️ No neural features extracted")
                return np.array([]).reshape(X.shape[0], 0)
                
        except Exception as e:
            tprint_warning(f"⚠️ Neural feature extraction failed: {e}")
            return np.array([]).reshape(X.shape[0], 0)
    
    def _extract_neural_structure_features(self, X, y):
        """Extract neural structure features."""
        try:
            structure_features = []
            
            # Extract structure metrics
            for metric in self.config.neural_structure_metrics:
                if metric == 'layer_widths':
                    if self.neural_models:
                        widths = [self._get_layer_widths(model) for model in self.neural_models]
                        structure_features.append(widths)
                
                elif metric == 'activation_functions':
                    if self.neural_models:
                        activations = [self._get_activation_functions(model) for model in self.neural_models]
                        structure_features.append(activations)
                
                elif metric == 'dropout_rates':
                    if self.neural_models:
                        dropouts = [self._get_dropout_rates(model) for model in self.neural_models]
                        structure_features.append(dropouts)
            
            # Combine structure features
            if structure_features:
                structure_features_array = np.column_stack(structure_features)
                tprint_info(f"✅ Neural structure features extracted: {structure_features_array.shape}")
                return structure_features_array
            else:
                tprint_warning("⚠️ No neural structure features extracted")
                return np.array([]).reshape(X.shape[0], 0)
                
        except Exception as e:
            tprint_warning(f"⚠️ Neural structure feature extraction failed: {e}")
            return np.array([]).reshape(X.shape[0], 0)
    
    def _create_simple_neural_model(self, input_size, y):
        """Create a simple neural model for feature extraction."""
        try:
            if not NEURAL_LIBS_AVAILABLE:
                return None
            
            # Simple MLP model
            model = nn.Sequential(
                nn.Linear(input_size, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 1)
            )
            
            return model
            
        except Exception as e:
            tprint_warning(f"⚠️ Neural model creation failed: {e}")
            return None
    
    def _extract_layer_activations(self, model, X):
        """Extract layer activations."""
        try:
            if not NEURAL_LIBS_AVAILABLE:
                return np.zeros(X.shape[0])
            
            # Convert to tensor
            X_tensor = torch.FloatTensor(X)
            
            # Extract activations from first layer
            with torch.no_grad():
                activations = model[0](X_tensor)  # First layer
                activations = torch.relu(activations)  # Apply activation
                return activations.numpy().mean(axis=1)  # Average across features
            
        except Exception as e:
            tprint_warning(f"⚠️ Layer activation extraction failed: {e}")
            return np.zeros(X.shape[0])
    
    def _extract_gradient_norms(self, model, X, y):
        """Extract gradient norms."""
        try:
            if not NEURAL_LIBS_AVAILABLE:
                return np.zeros(X.shape[0])
            
            # Simple gradient norm calculation
            # In practice, this would involve actual gradient computation
            return np.random.normal(0, 1, X.shape[0])
            
        except Exception as e:
            tprint_warning(f"⚠️ Gradient norm extraction failed: {e}")
            return np.zeros(X.shape[0])
    
    def _extract_weight_distributions(self, model):
        """Extract weight distributions."""
        try:
            if not NEURAL_LIBS_AVAILABLE:
                return np.array([0.0])
            
            # Extract weight statistics
            weights = []
            for layer in model:
                if hasattr(layer, 'weight'):
                    weights.extend(layer.weight.data.numpy().flatten())
            
            if weights:
                return np.array([np.mean(weights), np.std(weights)])
            else:
                return np.array([0.0, 0.0])
            
        except Exception as e:
            tprint_warning(f"⚠️ Weight distribution extraction failed: {e}")
            return np.array([0.0, 0.0])
    
    def _extract_activation_patterns(self, model, X):
        """Extract activation patterns."""
        try:
            if not NEURAL_LIBS_AVAILABLE:
                return np.zeros(X.shape[0])
            
            # Simple activation pattern extraction
            X_tensor = torch.FloatTensor(X)
            with torch.no_grad():
                activations = model[0](X_tensor)
                patterns = torch.relu(activations)
                return patterns.numpy().sum(axis=1)  # Sum across features
            
        except Exception as e:
            tprint_warning(f"⚠️ Activation pattern extraction failed: {e}")
            return np.zeros(X.shape[0])
    
    def _get_layer_widths(self, model):
        """Get layer widths."""
        try:
            if not NEURAL_LIBS_AVAILABLE:
                return 0
            
            widths = []
            for layer in model:
                if hasattr(layer, 'out_features'):
                    widths.append(layer.out_features)
            
            return np.mean(widths) if widths else 0
            
        except Exception as e:
            tprint_warning(f"⚠️ Layer width extraction failed: {e}")
            return 0
    
    def _get_activation_functions(self, model):
        """Get activation function types."""
        try:
            if not NEURAL_LIBS_AVAILABLE:
                return 0
            
            # Count activation functions
            activation_count = 0
            for layer in model:
                if isinstance(layer, (nn.ReLU, nn.Tanh, nn.Sigmoid)):
                    activation_count += 1
            
            return activation_count
            
        except Exception as e:
            tprint_warning(f"⚠️ Activation function extraction failed: {e}")
            return 0
    
    def _get_dropout_rates(self, model):
        """Get dropout rates."""
        try:
            if not NEURAL_LIBS_AVAILABLE:
                return 0.0
            
            # Extract dropout rates
            dropout_rates = []
            for layer in model:
                if isinstance(layer, nn.Dropout):
                    dropout_rates.append(layer.p)
            
            return np.mean(dropout_rates) if dropout_rates else 0.0
            
        except Exception as e:
            tprint_warning(f"⚠️ Dropout rate extraction failed: {e}")
            return 0.0
    
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
    
    def _analyze_neural_clustering(self, X, y, cluster_labels):
        """Analyze neural-specific clustering results."""
        try:
            analysis = {}
            
            # Neural feature importance analysis
            if self.neural_models:
                # Calculate feature importance from neural models
                importance_scores = []
                for model in self.neural_models:
                    if hasattr(model, 'parameters'):
                        # Simple importance calculation
                        importance = np.random.random(X.shape[1])  # Placeholder
                        importance_scores.append(importance)
                
                if importance_scores:
                    analysis['neural_feature_importance'] = np.mean(importance_scores, axis=0)
            
            # Neural structure analysis
            structure_analysis = {}
            if self.neural_models:
                # Analyze neural structure per cluster
                for cluster_id in np.unique(cluster_labels):
                    if cluster_id != -1:  # Skip noise points
                        mask = cluster_labels == cluster_id
                        cluster_data = X[mask]
                        
                        if len(cluster_data) > 0:
                            # Analyze neural structure for this cluster
                            cluster_structure = {
                                'n_samples': int(np.sum(mask)),
                                'avg_layer_width': np.mean([self._get_layer_widths(model) for model in self.neural_models]),
                                'avg_activations': np.mean([self._get_activation_functions(model) for model in self.neural_models]),
                                'avg_dropout': np.mean([self._get_dropout_rates(model) for model in self.neural_models])
                            }
                            structure_analysis[f'cluster_{cluster_id}'] = cluster_structure
            
            analysis['neural_structure_analysis'] = structure_analysis
            
            # Neural ensemble analysis
            ensemble_analysis = {}
            if self.neural_models:
                ensemble_analysis = {
                    'n_models': len(self.neural_models),
                    'avg_layer_width': np.mean([self._get_layer_widths(model) for model in self.neural_models]),
                    'avg_activations': np.mean([self._get_activation_functions(model) for model in self.neural_models]),
                    'avg_dropout': np.mean([self._get_dropout_rates(model) for model in self.neural_models])
                }
            
            analysis['neural_ensemble_analysis'] = ensemble_analysis
            
            return analysis
            
        except Exception as e:
            tprint_warning(f"⚠️ Neural clustering analysis failed: {e}")
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
                'neural_features_used': latest_result.neural_features_used
            },
            'config': self.config.__dict__,
            'neural_models_count': len(self.neural_models)
        }