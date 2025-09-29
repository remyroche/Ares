"""
NAS Config

Implementation for NAS clustering configuration.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum


class ArchitectureType(Enum):
    """Types of neural architectures."""
    FEEDFORWARD = "feedforward"
    CONVOLUTIONAL = "convolutional"
    RECURRENT = "recurrent"
    TRANSFORMER = "transformer"
    HYBRID = "hybrid"


class ClusteringMethod(Enum):
    """Clustering methods."""
    KMEANS = "kmeans"
    DBSCAN = "dbscan"
    AGGLOMERATIVE = "agglomerative"
    SPECTRAL = "spectral"
    GAUSSIAN_MIXTURE = "gaussian_mixture"


@dataclass
class NASClusteringConfig:
    """Configuration for NAS clustering."""
    # Architecture settings
    architecture_types: List[ArchitectureType]
    max_layers: int = 10
    min_layers: int = 2
    layer_widths: List[int] = None
    activations: List[str] = None
    
    # Clustering settings
    clustering_method: ClusteringMethod = ClusteringMethod.KMEANS
    n_clusters: Optional[int] = None
    eps: float = 0.5
    min_samples: int = 5
    linkage: str = "ward"
    
    # Feature extraction settings
    feature_extractors: List[str] = None
    normalize_features: bool = True
    dimensionality_reduction: bool = False
    n_components: int = 10
    
    # Optimization settings
    optimization_objectives: List[str] = None
    optimization_weights: List[float] = None
    max_iterations: int = 100
    convergence_threshold: float = 1e-4
    
    # Evaluation settings
    validation_split: float = 0.2
    test_split: float = 0.1
    cross_validation_folds: int = 5
    
    def __post_init__(self):
        """Initialize default values."""
        if self.layer_widths is None:
            self.layer_widths = [32, 64, 128, 256, 512]
        
        if self.activations is None:
            self.activations = ['relu', 'tanh', 'sigmoid', 'swish']
        
        if self.feature_extractors is None:
            self.feature_extractors = ['layer_count', 'parameter_count', 'activation_diversity']
        
        if self.optimization_objectives is None:
            self.optimization_objectives = ['accuracy', 'efficiency', 'complexity']
        
        if self.optimization_weights is None:
            self.optimization_weights = [0.4, 0.3, 0.3]
    
    def validate(self) -> bool:
        """Validate configuration."""
        # Check architecture types
        if not self.architecture_types:
            raise ValueError("At least one architecture type must be specified")
        
        # Check layer constraints
        if self.max_layers < self.min_layers:
            raise ValueError("max_layers must be >= min_layers")
        
        # Check clustering method
        if self.clustering_method == ClusteringMethod.KMEANS and self.n_clusters is None:
            self.n_clusters = 3
        
        # Check optimization objectives and weights
        if len(self.optimization_objectives) != len(self.optimization_weights):
            raise ValueError("Number of objectives must match number of weights")
        
        # Check validation splits
        if self.validation_split + self.test_split >= 1.0:
            raise ValueError("validation_split + test_split must be < 1.0")
        
        return True
    
    def get_architecture_space(self) -> Dict:
        """Get architecture search space."""
        return {
            'architecture_types': [t.value for t in self.architecture_types],
            'max_layers': self.max_layers,
            'min_layers': self.min_layers,
            'layer_widths': self.layer_widths,
            'activations': self.activations
        }
    
    def get_clustering_params(self) -> Dict:
        """Get clustering parameters."""
        params = {
            'method': self.clustering_method.value,
            'eps': self.eps,
            'min_samples': self.min_samples,
            'linkage': self.linkage
        }
        
        if self.n_clusters is not None:
            params['n_clusters'] = self.n_clusters
        
        return params
    
    def get_feature_extraction_params(self) -> Dict:
        """Get feature extraction parameters."""
        return {
            'extractors': self.feature_extractors,
            'normalize': self.normalize_features,
            'dimensionality_reduction': self.dimensionality_reduction,
            'n_components': self.n_components
        }
    
    def get_optimization_params(self) -> Dict:
        """Get optimization parameters."""
        return {
            'objectives': self.optimization_objectives,
            'weights': self.optimization_weights,
            'max_iterations': self.max_iterations,
            'convergence_threshold': self.convergence_threshold
        }
    
    def get_evaluation_params(self) -> Dict:
        """Get evaluation parameters."""
        return {
            'validation_split': self.validation_split,
            'test_split': self.test_split,
            'cross_validation_folds': self.cross_validation_folds
        }


class NASArchitectureType:
    """Neural Architecture Search Architecture Type."""
    
    def __init__(self, name: str, description: str = "", 
                 default_layers: int = 3, default_width: int = 64):
        """Initialize architecture type.
        
        Args:
            name: Name of the architecture type
            description: Description of the architecture type
            default_layers: Default number of layers
            default_width: Default layer width
        """
        self.name = name
        self.description = description
        self.default_layers = default_layers
        self.default_width = default_width
        self.valid_activations = ['relu', 'tanh', 'sigmoid', 'swish']
        self.valid_layer_types = ['dense', 'conv2d', 'lstm', 'gru', 'attention']
    
    def create_architecture(self, layers: Optional[int] = None, 
                           width: Optional[int] = None) -> Dict:
        """Create a default architecture of this type.
        
        Args:
            layers: Number of layers (uses default if None)
            width: Layer width (uses default if None)
            
        Returns:
            Architecture specification
        """
        num_layers = layers or self.default_layers
        layer_width = width or self.default_width
        
        architecture = {
            'type': self.name,
            'description': self.description,
            'layers': []
        }
        
        for i in range(num_layers):
            layer = {
                'type': 'dense',  # Default layer type
                'width': layer_width,
                'activation': 'relu',
                'dropout': 0.0
            }
            architecture['layers'].append(layer)
        
        return architecture
    
    def validate_architecture(self, architecture: Dict) -> bool:
        """Validate architecture against this type.
        
        Args:
            architecture: Architecture to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(architecture, dict):
            return False
        
        if 'layers' not in architecture:
            return False
        
        layers = architecture['layers']
        if not isinstance(layers, list):
            return False
        
        for layer in layers:
            if not isinstance(layer, dict):
                return False
            
            # Check required fields
            if 'type' not in layer or 'width' not in layer:
                return False
            
            # Check layer type
            if layer['type'] not in self.valid_layer_types:
                return False
            
            # Check width is positive
            if layer['width'] <= 0:
                return False
            
            # Check activation if present
            if 'activation' in layer:
                if layer['activation'] not in self.valid_activations:
                    return False
        
        return True
    
    def get_complexity_score(self, architecture: Dict) -> float:
        """Calculate complexity score for architecture.
        
        Args:
            architecture: Architecture to score
            
        Returns:
            Complexity score (higher = more complex)
        """
        if not self.validate_architecture(architecture):
            return 0.0
        
        layers = architecture.get('layers', [])
        if not layers:
            return 0.0
        
        # Calculate complexity based on layers and parameters
        num_layers = len(layers)
        total_params = sum(layer.get('width', 0) for layer in layers)
        
        # Complexity score
        complexity = num_layers * 0.1 + total_params / 1000
        
        return complexity
    
    def get_parameter_count(self, architecture: Dict) -> int:
        """Calculate total parameter count for architecture.
        
        Args:
            architecture: Architecture to analyze
            
        Returns:
            Total parameter count
        """
        if not self.validate_architecture(architecture):
            return 0
        
        layers = architecture.get('layers', [])
        total_params = 0
        
        for i, layer in enumerate(layers):
            width = layer.get('width', 0)
            if i == 0:
                # Input layer
                total_params += width * 10  # Assume 10 input features
            else:
                # Hidden layers
                prev_width = layers[i-1].get('width', 0)
                total_params += prev_width * width + width  # weights + bias
        
        return total_params
    
    def __str__(self) -> str:
        """String representation."""
        return f"NASArchitectureType(name='{self.name}', description='{self.description}')"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"NASArchitectureType(name='{self.name}', description='{self.description}', "
                f"default_layers={self.default_layers}, default_width={self.default_width})")
