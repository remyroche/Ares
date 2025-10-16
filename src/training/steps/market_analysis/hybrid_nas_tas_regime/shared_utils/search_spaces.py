"""
Comprehensive Search Space Definitions for NAS and TAS Systems

This module provides systematic search space definitions for both Neural Architecture
Search (NAS) and Tree Architecture Search (TAS), including layer types, activation
functions, connections, and architectural constraints.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error,
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)

logger = logging.getLogger(__name__)

class LayerType(Enum):
    """Neural network layer types for NAS."""
    # Basic layers
    LINEAR = "linear"
    CONV1D = "conv1d"
    CONV2D = "conv2d"
    LSTM = "lstm"
    GRU = "gru"

    # Advanced layers
    ATTENTION = "attention"
    TRANSFORMER_BLOCK = "transformer_block"
    RESIDUAL_BLOCK = "residual_block"
    DENSE_BLOCK = "dense_block"
    INCEPTION_BLOCK = "inception_block"

    # Specialized layers
    DROPOUT = "dropout"
    BATCH_NORM = "batch_norm"
    LAYER_NORM = "layer_norm"
    ACTIVATION = "activation"

    # Tree-specific layers
    DECISION_TREE = "decision_tree"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"

class ActivationFunction(Enum):
    """Activation functions for neural networks."""
    RELU = "relu"
    LEAKY_RELU = "leaky_relu"
    ELU = "elu"
    SELU = "selu"
    GELU = "gelu"
    SWISH = "swish"
    MISH = "mish"
    TANH = "tanh"
    SIGMOID = "sigmoid"
    SOFTMAX = "softmax"
    NONE = "none"

class ConnectionType(Enum):
    """Connection types between layers."""
    DENSE = "dense"
    SPARSE = "sparse"
    RESIDUAL = "residual"
    DENSE_RESIDUAL = "dense_residual"
    SKIP = "skip"
    HIGHWAY = "highway"
    CONCATENATE = "concatenate"

@dataclass
class LayerSpecification:
    """Specification for a neural network layer."""
    layer_type: LayerType
    hidden_size: int
    activation: Optional[ActivationFunction] = None
    dropout_rate: float = 0.0
    kernel_size: Optional[int] = None
    stride: int = 1
    padding: Union[str, int] = 'valid'
    use_bias: bool = True
    batch_norm: bool = False
    layer_norm: bool = False
    residual: bool = False
    parameters: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TreeSpecification:
    """Specification for a tree-based model."""
    tree_type: LayerType
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: Optional[Union[str, int, float]] = None
    criterion: str = 'gini'
    splitter: str = 'best'
    max_leaf_nodes: Optional[int] = None
    min_impurity_decrease: float = 0.0
    ccp_alpha: float = 0.0
    bootstrap: bool = False
    n_estimators: int = 1
    learning_rate: float = 0.1
    subsample: float = 1.0
    colsample_bytree: float = 1.0
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0

@dataclass
class ArchitectureConstraints:
    """Constraints for valid architectures."""
    max_layers: int = 20
    min_layers: int = 2
    max_hidden_size: int = 2048
    min_hidden_size: int = 8
    max_parameters: int = 10000000  # 10M parameters
    min_parameters: int = 100
    allowed_layer_types: List[LayerType] = field(default_factory=list)
    allowed_activations: List[ActivationFunction] = field(default_factory=list)
    max_dropout_rate: float = 0.8
    min_dropout_rate: float = 0.0
    max_connections_per_layer: int = 5
    min_connections_per_layer: int = 1
    allow_residual_connections: bool = True
    allow_skip_connections: bool = True
    max_residual_depth: int = 5
    enforce_gradient_flow: bool = True
    max_memory_usage_mb: int = 4096  # 4GB
    max_training_time_seconds: int = 3600  # 1 hour

@dataclass
class NeuralArchitecture:
    """Complete neural architecture specification."""
    layers: List[LayerSpecification]
    connections: List[Tuple[int, int, ConnectionType]]  # (from_layer, to_layer, connection_type)
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    parameters: Dict[str, Any] = field(default_factory=dict)
    estimated_complexity: float = 0.0
    estimated_memory_usage: float = 0.0
    estimated_training_time: float = 0.0

    def calculate_complexity(self) -> float:
        """Calculate architecture complexity score."""
        complexity = 0.0

        # Layer count complexity
        complexity += len(self.layers) * 0.1

        # Parameter count complexity
        total_params = sum(layer.hidden_size * layer.hidden_size for layer in self.layers)
        complexity += min(total_params / 1000000, 1.0)  # Normalize to max 1.0

        # Connection complexity
        complexity += len(self.connections) * 0.05

        # Residual connection complexity
        residual_connections = sum(1 for conn in self.connections if conn[2] == ConnectionType.RESIDUAL)
        complexity += residual_connections * 0.1

        return min(complexity, 5.0)  # Cap at 5.0

@dataclass
class TreeArchitecture:
    """Complete tree architecture specification."""
    trees: List[TreeSpecification]
    ensemble_method: str = 'voting'  # voting, averaging, stacking
    feature_preprocessing: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    estimated_complexity: float = 0.0
    estimated_memory_usage: float = 0.0
    estimated_training_time: float = 0.0

    def calculate_complexity(self) -> float:
        """Calculate tree architecture complexity score."""
        complexity = 0.0

        # Tree count complexity
        complexity += len(self.trees) * 0.1

        # Average depth complexity
        avg_depth = sum(tree.max_depth or 10 for tree in self.trees) / len(self.trees)
        complexity += min(avg_depth / 20, 1.0)

        # Ensemble complexity
        if self.ensemble_method == 'stacking':
            complexity += 0.5
        elif self.ensemble_method == 'averaging':
            complexity += 0.2

        # Feature preprocessing complexity
        complexity += len(self.feature_preprocessing) * 0.05

        return min(complexity, 5.0)

class NeuralSearchSpace:
    """Comprehensive search space for neural architectures."""

    def __init__(self, constraints: Optional[ArchitectureConstraints] = None):
        """Initialize neural search space."""
        self.constraints = constraints or ArchitectureConstraints()

        # Define layer specifications
        self.layer_specs = self._define_layer_specifications()

        # Define activation functions
        self.activation_specs = self._define_activation_specifications()

        # Define connection specifications
        self.connection_specs = self._define_connection_specifications()

        logger.info("✅ Neural Search Space initialized")
        logger.info(f"   Layer types: {len(self.layer_specs)}")
        logger.info(f"   Activation functions: {len(self.activation_specs)}")
        logger.info(f"   Connection types: {len(self.connection_specs)}")

    def _define_layer_specifications(self) -> Dict[str, Dict[str, Any]]:
        """Define layer specifications and their parameter ranges."""
        return {
            LayerType.LINEAR.value: {
                'hidden_size': {'type': 'discrete', 'choices': [16, 32, 64, 128, 256, 512, 1024]},
                'activation': {'type': 'discrete', 'choices': [act.value for act in ActivationFunction]},
                'dropout_rate': {'type': 'continuous', 'min': 0.0, 'max': 0.5},
                'use_bias': {'type': 'discrete', 'choices': [True, False]},
                'batch_norm': {'type': 'discrete', 'choices': [True, False]},
                'layer_norm': {'type': 'discrete', 'choices': [True, False]}
            },
            LayerType.CONV1D.value: {
                'hidden_size': {'type': 'discrete', 'choices': [32, 64, 128, 256]},
                'kernel_size': {'type': 'discrete', 'choices': [3, 5, 7, 9]},
                'stride': {'type': 'discrete', 'choices': [1, 2]},
                'padding': {'type': 'discrete', 'choices': ['valid', 'same', 'causal']},
                'activation': {'type': 'discrete', 'choices': [act.value for act in ActivationFunction]},
                'dropout_rate': {'type': 'continuous', 'min': 0.0, 'max': 0.3},
                'use_bias': {'type': 'discrete', 'choices': [True, False]}
            },
            LayerType.LSTM.value: {
                'hidden_size': {'type': 'discrete', 'choices': [32, 64, 128, 256]},
                'dropout_rate': {'type': 'continuous', 'min': 0.0, 'max': 0.3},
                'recurrent_dropout': {'type': 'continuous', 'min': 0.0, 'max': 0.3},
                'return_sequences': {'type': 'discrete', 'choices': [True, False]},
                'go_backwards': {'type': 'discrete', 'choices': [True, False]}
            },
            LayerType.ATTENTION.value: {
                'hidden_size': {'type': 'discrete', 'choices': [64, 128, 256]},
                'num_heads': {'type': 'discrete', 'choices': [4, 8, 16]},
                'dropout_rate': {'type': 'continuous', 'min': 0.0, 'max': 0.2},
                'use_mask': {'type': 'discrete', 'choices': [True, False]}
            },
            LayerType.RESIDUAL_BLOCK.value: {
                'hidden_size': {'type': 'discrete', 'choices': [64, 128, 256, 512]},
                'num_layers': {'type': 'discrete', 'choices': [2, 3, 4]},
                'dropout_rate': {'type': 'continuous', 'min': 0.0, 'max': 0.2}
            },
            LayerType.DROPOUT.value: {
                'dropout_rate': {'type': 'continuous', 'min': 0.0, 'max': 0.8}
            },
            LayerType.BATCH_NORM.value: {
                'momentum': {'type': 'continuous', 'min': 0.1, 'max': 0.99},
                'epsilon': {'type': 'continuous', 'min': 1e-5, 'max': 1e-3}
            }
        }

    def _define_activation_specifications(self) -> Dict[str, Dict[str, Any]]:
        """Define activation function specifications."""
        return {
            'relu': {'alpha': {'type': 'continuous', 'min': 0.0, 'max': 0.3}},
            'leaky_relu': {'alpha': {'type': 'continuous', 'min': 0.01, 'max': 0.3}},
            'elu': {'alpha': {'type': 'continuous', 'min': 0.1, 'max': 2.0}},
            'selu': {},  # No parameters
            'gelu': {},  # No parameters
            'swish': {},  # No parameters
            'mish': {},  # No parameters
            'tanh': {},  # No parameters
            'sigmoid': {},  # No parameters
            'softmax': {'axis': {'type': 'discrete', 'choices': [-1, 0, 1]}}
        }

    def _define_connection_specifications(self) -> Dict[str, Dict[str, Any]]:
        """Define connection specifications."""
        return {
            ConnectionType.DENSE.value: {},
            ConnectionType.SPARSE.value: {
                'sparsity': {'type': 'continuous', 'min': 0.1, 'max': 0.9}
            },
            ConnectionType.RESIDUAL.value: {
                'scale': {'type': 'continuous', 'min': 0.1, 'max': 1.0}
            },
            ConnectionType.SKIP.value: {
                'skip_layers': {'type': 'integer', 'min': 1, 'max': 5}
            },
            ConnectionType.HIGHWAY.value: {
                'carry_bias': {'type': 'continuous', 'min': -2.0, 'max': 2.0}
            }
        }

    def sample_random_architecture(self) -> NeuralArchitecture:
        """Sample a random architecture from the search space."""
        # Sample number of layers
        n_layers = np.random.randint(self.constraints.min_layers, self.constraints.max_layers + 1)

        layers = []
        connections = []

        # Input shape (assuming time series data)
        current_shape = (None, 30)  # (timesteps, features)

        for i in range(n_layers):
            # Sample layer type
            available_layers = [lt.value for lt in LayerType if lt.value in self.layer_specs]
            layer_type = np.random.choice(available_layers)

            # Sample layer parameters
            layer_params = {}
            for param_name, param_spec in self.layer_specs[layer_type].items():
                if param_spec['type'] == 'discrete':
                    layer_params[param_name] = np.random.choice(param_spec['choices'])
                elif param_spec['type'] == 'continuous':
                    layer_params[param_name] = np.random.uniform(param_spec['min'], param_spec['max'])
                elif param_spec['type'] == 'integer':
                    layer_params[param_name] = np.random.randint(param_spec['min'], param_spec['max'] + 1)

            # Create layer specification
            layer_spec = LayerSpecification(
                layer_type=LayerType(layer_type),
                hidden_size=layer_params.get('hidden_size', 64),
                activation=ActivationFunction(layer_params.get('activation', 'relu')) if 'activation' in layer_params else None,
                dropout_rate=layer_params.get('dropout_rate', 0.0),
                kernel_size=layer_params.get('kernel_size'),
                stride=layer_params.get('stride', 1),
                padding=layer_params.get('padding', 'valid'),
                use_bias=layer_params.get('use_bias', True),
                batch_norm=layer_params.get('batch_norm', False),
                layer_norm=layer_params.get('layer_norm', False),
                residual=layer_params.get('residual', False),
                parameters=layer_params
            )
            layers.append(layer_spec)

            # Add connections
            if i > 0:  # Not the first layer
                # Dense connection from previous layer
                connections.append((i-1, i, ConnectionType.DENSE))

                # Possibly add residual connection
                if (self.constraints.allow_residual_connections and
                    i >= 2 and np.random.random() < 0.3):
                    skip_from = max(0, i - np.random.randint(2, min(i, self.constraints.max_residual_depth) + 1))
                    connections.append((skip_from, i, ConnectionType.RESIDUAL))

        # Output shape (assuming regression or binary classification)
        output_shape = (1,)

        architecture = NeuralArchitecture(
            layers=layers,
            connections=connections,
            input_shape=current_shape,
            output_shape=output_shape
        )

        architecture.estimated_complexity = architecture.calculate_complexity()
        return architecture

    def validate_architecture(self, architecture: NeuralArchitecture) -> Tuple[bool, List[str]]:
        """Validate if an architecture meets the constraints."""
        errors = []

        # Check layer count
        if len(architecture.layers) < self.constraints.min_layers:
            errors.append(f"Too few layers: {len(architecture.layers)} < {self.constraints.min_layers}")
        if len(architecture.layers) > self.constraints.max_layers:
            errors.append(f"Too many layers: {len(architecture.layers)} > {self.constraints.max_layers}")

        # Check parameter count
        total_params = sum(layer.hidden_size * (layer.hidden_size if hasattr(layer, 'hidden_size') else 1)
                          for layer in architecture.layers)
        if total_params < self.constraints.min_parameters:
            errors.append(f"Too few parameters: {total_params} < {self.constraints.min_parameters}")
        if total_params > self.constraints.max_parameters:
            errors.append(f"Too many parameters: {total_params} > {self.constraints.max_parameters}")

        # Check dropout rates
        for layer in architecture.layers:
            if layer.dropout_rate < self.constraints.min_dropout_rate:
                errors.append(f"Dropout rate too low: {layer.dropout_rate} < {self.constraints.min_dropout_rate}")
            if layer.dropout_rate > self.constraints.max_dropout_rate:
                errors.append(f"Dropout rate too high: {layer.dropout_rate} > {self.constraints.max_dropout_rate}")

        # Check complexity
        if architecture.estimated_complexity > 5.0:
            errors.append(f"Architecture too complex: {architecture.estimated_complexity:.2f}")

        return len(errors) == 0, errors

class TreeSearchSpace:
    """Comprehensive search space for tree architectures."""

    def __init__(self, constraints: Optional[ArchitectureConstraints] = None):
        """Initialize tree search space."""
        self.constraints = constraints or ArchitectureConstraints()

        # Define tree specifications
        self.tree_specs = self._define_tree_specifications()

        # Define ensemble specifications
        self.ensemble_specs = self._define_ensemble_specifications()

        logger.info("✅ Tree Search Space initialized")
        logger.info(f"   Tree types: {len(self.tree_specs)}")
        logger.info(f"   Ensemble methods: {len(self.ensemble_specs)}")

    def _define_tree_specifications(self) -> Dict[str, Dict[str, Any]]:
        """Define tree specifications and their parameter ranges."""
        return {
            LayerType.DECISION_TREE.value: {
                'max_depth': {'type': 'discrete', 'choices': [3, 5, 7, 10, 15, None]},
                'min_samples_split': {'type': 'discrete', 'choices': [2, 5, 10, 20]},
                'min_samples_leaf': {'type': 'discrete', 'choices': [1, 2, 5, 10]},
                'max_features': {'type': 'discrete', 'choices': ['sqrt', 'log2', None, 0.3, 0.5, 0.7]},
                'criterion': {'type': 'discrete', 'choices': ['gini', 'entropy']},
                'splitter': {'type': 'discrete', 'choices': ['best', 'random']},
                'min_impurity_decrease': {'type': 'continuous', 'min': 0.0, 'max': 0.1}
            },
            LayerType.RANDOM_FOREST.value: {
                'n_estimators': {'type': 'discrete', 'choices': [10, 50, 100, 200]},
                'max_depth': {'type': 'discrete', 'choices': [5, 10, 15, 20, None]},
                'min_samples_split': {'type': 'discrete', 'choices': [2, 5, 10]},
                'min_samples_leaf': {'type': 'discrete', 'choices': [1, 2, 4]},
                'max_features': {'type': 'discrete', 'choices': ['sqrt', 'log2', 0.3, 0.5]},
                'bootstrap': {'type': 'discrete', 'choices': [True, False]},
                'criterion': {'type': 'discrete', 'choices': ['gini', 'entropy']}
            },
            LayerType.GRADIENT_BOOSTING.value: {
                'n_estimators': {'type': 'discrete', 'choices': [50, 100, 200, 500]},
                'learning_rate': {'type': 'continuous', 'min': 0.01, 'max': 0.3},
                'max_depth': {'type': 'discrete', 'choices': [3, 5, 7, 10]},
                'subsample': {'type': 'continuous', 'min': 0.5, 'max': 1.0},
                'min_samples_split': {'type': 'discrete', 'choices': [2, 5, 10]},
                'min_samples_leaf': {'type': 'discrete', 'choices': [1, 2, 5]},
                'max_features': {'type': 'discrete', 'choices': ['sqrt', 'log2', 0.5, 0.7]}
            },
            LayerType.XGBOOST.value: {
                'n_estimators': {'type': 'discrete', 'choices': [100, 200, 500, 1000]},
                'learning_rate': {'type': 'continuous', 'min': 0.01, 'max': 0.3},
                'max_depth': {'type': 'discrete', 'choices': [3, 5, 7, 9]},
                'subsample': {'type': 'continuous', 'min': 0.5, 'max': 1.0},
                'colsample_bytree': {'type': 'continuous', 'min': 0.3, 'max': 1.0},
                'colsample_bylevel': {'type': 'continuous', 'min': 0.3, 'max': 1.0},
                'reg_alpha': {'type': 'continuous', 'min': 0.0, 'max': 1.0},
                'reg_lambda': {'type': 'continuous', 'min': 0.1, 'max': 2.0}
            }
        }

    def _define_ensemble_specifications(self) -> Dict[str, Dict[str, Any]]:
        """Define ensemble specifications."""
        return {
            'voting': {
                'voting': {'type': 'discrete', 'choices': ['hard', 'soft']},
                'weights': {'type': 'continuous', 'min': 0.1, 'max': 1.0}
            },
            'averaging': {
                'weights': {'type': 'continuous', 'min': 0.1, 'max': 1.0}
            },
            'stacking': {
                'meta_learner': {'type': 'discrete', 'choices': ['linear', 'ridge', 'lasso', 'elastic_net']},
                'cv_folds': {'type': 'discrete', 'choices': [3, 5, 7, 10]}
            }
        }

    def sample_random_architecture(self) -> TreeArchitecture:
        """Sample a random tree architecture from the search space."""
        # Sample number of trees
        n_trees = np.random.randint(1, 10)

        trees = []
        for i in range(n_trees):
            # Sample tree type
            tree_types = [lt.value for lt in LayerType if lt.value in self.tree_specs]
            tree_type = np.random.choice(tree_types)

            # Sample tree parameters
            tree_params = {}
            for param_name, param_spec in self.tree_specs[tree_type].items():
                if param_spec['type'] == 'discrete':
                    tree_params[param_name] = np.random.choice(param_spec['choices'])
                elif param_spec['type'] == 'continuous':
                    tree_params[param_name] = np.random.uniform(param_spec['min'], param_spec['max'])
                elif param_spec['type'] == 'integer':
                    tree_params[param_name] = np.random.randint(param_spec['min'], param_spec['max'] + 1)

            # Create tree specification
            tree_spec = TreeSpecification(
                tree_type=LayerType(tree_type),
                max_depth=tree_params.get('max_depth'),
                min_samples_split=tree_params.get('min_samples_split', 2),
                min_samples_leaf=tree_params.get('min_samples_leaf', 1),
                max_features=tree_params.get('max_features'),
                criterion=tree_params.get('criterion', 'gini'),
                splitter=tree_params.get('splitter', 'best'),
                max_leaf_nodes=None,
                min_impurity_decrease=tree_params.get('min_impurity_decrease', 0.0),
                ccp_alpha=0.0,
                bootstrap=tree_params.get('bootstrap', False),
                n_estimators=tree_params.get('n_estimators', 1),
                learning_rate=tree_params.get('learning_rate', 0.1),
                subsample=tree_params.get('subsample', 1.0),
                colsample_bytree=tree_params.get('colsample_bytree', 1.0),
                reg_alpha=tree_params.get('reg_alpha', 0.0),
                reg_lambda=tree_params.get('reg_lambda', 1.0)
            )
            trees.append(tree_spec)

        # Sample ensemble method
        ensemble_methods = list(self.ensemble_specs.keys())
        ensemble_method = np.random.choice(ensemble_methods)

        # Sample feature preprocessing
        feature_preprocessing = np.random.choice(
            [['scaling', 'encoding'], ['scaling'], ['encoding'], []],
            p=[0.4, 0.3, 0.2, 0.1]
        )

        architecture = TreeArchitecture(
            trees=trees,
            ensemble_method=ensemble_method,
            feature_preprocessing=feature_preprocessing
        )

        architecture.estimated_complexity = architecture.calculate_complexity()
        return architecture

    def validate_architecture(self, architecture: TreeArchitecture) -> Tuple[bool, List[str]]:
        """Validate if a tree architecture meets the constraints."""
        errors = []

        # Check tree count
        if len(architecture.trees) < 1:
            errors.append("At least one tree required")
        if len(architecture.trees) > 20:
            errors.append(f"Too many trees: {len(architecture.trees)} > 20")

        # Check complexity
        if architecture.estimated_complexity > 5.0:
            errors.append(f"Architecture too complex: {architecture.estimated_complexity:.2f}")

        # Validate individual trees
        for i, tree in enumerate(architecture.trees):
            if tree.max_depth is not None and tree.max_depth > 30:
                errors.append(f"Tree {i} depth too high: {tree.max_depth}")

            if tree.n_estimators > 1000:
                errors.append(f"Tree {i} has too many estimators: {tree.n_estimators}")

        return len(errors) == 0, errors

def create_neural_search_space(constraints: Optional[ArchitectureConstraints] = None) -> NeuralSearchSpace:
    """Create a neural search space instance."""
    return NeuralSearchSpace(constraints)

def create_tree_search_space(constraints: Optional[ArchitectureConstraints] = None) -> TreeSearchSpace:
    """Create a tree search space instance."""
    return TreeSearchSpace(constraints)
