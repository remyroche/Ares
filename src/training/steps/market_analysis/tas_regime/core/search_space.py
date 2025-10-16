"""
Tree Search Space Classes

Classes for defining and managing the search space for tree architectures.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import numpy as np
import random
from enum import Enum

from .tas_config import TreeModelType, OptimizationObjective
from .tree_architecture import TreeArchitectureCandidate

class SearchSpaceType(Enum):
    """Types of search spaces for TAS."""
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    CATEGORICAL = "categorical"
    MIXED = "mixed"

@dataclass
class ParameterRange:
    """Range definition for a parameter."""

    name: str
    param_type: SearchSpaceType
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    choices: Optional[List[Any]] = None
    step: Optional[Union[int, float]] = None
    log_scale: bool = False

    def sample(self) -> Any:
        """Sample a value from the parameter range."""
        if self.param_type == SearchSpaceType.CONTINUOUS:
            if self.log_scale:
                min_log = np.log(self.min_value) if self.min_value else 0
                max_log = np.log(self.max_value) if self.max_value else 1
                return np.exp(random.uniform(min_log, max_log))
            else:
                return random.uniform(self.min_value, self.max_value)

        elif self.param_type == SearchSpaceType.DISCRETE:
            if self.step:
                return random.randrange(
                    int(self.min_value),
                    int(self.max_value) + 1,
                    int(self.step)
                )
            else:
                return random.randint(self.min_value, self.max_value)

        elif self.param_type == SearchSpaceType.CATEGORICAL:
            return random.choice(self.choices)

        else:
            raise ValueError(f"Unknown parameter type: {self.param_type}")

    def is_valid(self, value: Any) -> bool:
        """Check if a value is valid for this parameter range."""
        if self.param_type == SearchSpaceType.CONTINUOUS:
            return self.min_value <= value <= self.max_value

        elif self.param_type == SearchSpaceType.DISCRETE:
            return self.min_value <= value <= self.max_value and isinstance(value, int)

        elif self.param_type == SearchSpaceType.CATEGORICAL:
            return value in self.choices

        else:
            return False

@dataclass
class TreeSearchSpace:
    """Search space for tree architectures."""

    # Model type choices
    model_types: List[TreeModelType] = field(default_factory=lambda: [
        TreeModelType.RANDOM_FOREST,
        TreeModelType.XGBOOST,
        TreeModelType.LIGHTGBM,
        TreeModelType.EXTRA_TREES,
        TreeModelType.GRADIENT_BOOSTING,
        TreeModelType.NGBOOST,
        TreeModelType.QUANTILE_GBDT,
        TreeModelType.DART,
        TreeModelType.DEEPGBM,
        TreeModelType.NODE
    ])

    # Parameter ranges
    parameter_ranges: Dict[str, ParameterRange] = field(default_factory=dict)

    # Constraints
    constraints: List[Callable[[TreeArchitectureCandidate], bool]] = field(default_factory=list)

    # Search space metadata
    space_type: SearchSpaceType = SearchSpaceType.MIXED
    dimensionality: int = 0

    def __post_init__(self):
        """Initialize search space."""
        self._initialize_parameter_ranges()
        self._calculate_dimensionality()

    def _initialize_parameter_ranges(self):
        """Initialize default parameter ranges."""
        if not self.parameter_ranges:
            self.parameter_ranges = {
                'n_trees': ParameterRange(
                    name='n_trees',
                    param_type=SearchSpaceType.DISCRETE,
                    min_value=10,
                    max_value=1000
                ),
                'max_depth': ParameterRange(
                    name='max_depth',
                    param_type=SearchSpaceType.DISCRETE,
                    min_value=1,
                    max_value=20
                ),
                'min_samples_split': ParameterRange(
                    name='min_samples_split',
                    param_type=SearchSpaceType.DISCRETE,
                    min_value=2,
                    max_value=100
                ),
                'min_samples_leaf': ParameterRange(
                    name='min_samples_leaf',
                    param_type=SearchSpaceType.DISCRETE,
                    min_value=1,
                    max_value=50
                ),
                'max_features': ParameterRange(
                    name='max_features',
                    param_type=SearchSpaceType.CATEGORICAL,
                    choices=['auto', 'sqrt', 'log2', 0.5, 0.7, 0.9]
                ),
                'learning_rate': ParameterRange(
                    name='learning_rate',
                    param_type=SearchSpaceType.CONTINUOUS,
                    min_value=0.01,
                    max_value=1.0,
                    log_scale=True
                ),
                'subsample': ParameterRange(
                    name='subsample',
                    param_type=SearchSpaceType.CONTINUOUS,
                    min_value=0.1,
                    max_value=1.0
                ),
                'colsample_bytree': ParameterRange(
                    name='colsample_bytree',
                    param_type=SearchSpaceType.CONTINUOUS,
                    min_value=0.1,
                    max_value=1.0
                ),
                'reg_alpha': ParameterRange(
                    name='reg_alpha',
                    param_type=SearchSpaceType.CONTINUOUS,
                    min_value=0.0,
                    max_value=10.0,
                    log_scale=True
                ),
                'reg_lambda': ParameterRange(
                    name='reg_lambda',
                    param_type=SearchSpaceType.CONTINUOUS,
                    min_value=0.0,
                    max_value=10.0,
                    log_scale=True
                ),
                # NGBoost specific parameters
                'base_learner': ParameterRange(
                    name='base_learner',
                    param_type=SearchSpaceType.CATEGORICAL,
                    choices=['random_forest', 'extra_trees']  # decision_tree removed
                ),
                'natural_gradient': ParameterRange(
                    name='natural_gradient',
                    param_type=SearchSpaceType.CATEGORICAL,
                    choices=[True, False]
                ),
                'expected_information': ParameterRange(
                    name='expected_information',
                    param_type=SearchSpaceType.CATEGORICAL,
                    choices=[True, False]
                ),
                # DART specific parameters
                'dart_drop_rate': ParameterRange(
                    name='dart_drop_rate',
                    param_type=SearchSpaceType.CONTINUOUS,
                    min_value=0.0,
                    max_value=0.5
                ),
                'dart_skip_drop': ParameterRange(
                    name='dart_skip_drop',
                    param_type=SearchSpaceType.CONTINUOUS,
                    min_value=0.0,
                    max_value=1.0
                ),
                # DeepGBM specific parameters
                'num_layers': ParameterRange(
                    name='num_layers',
                    param_type=SearchSpaceType.DISCRETE,
                    min_value=2,
                    max_value=10
                ),
                'layer_size': ParameterRange(
                    name='layer_size',
                    param_type=SearchSpaceType.DISCRETE,
                    min_value=10,
                    max_value=100
                ),
                # Quantile GBDT specific parameters
                'quantile_alpha': ParameterRange(
                    name='quantile_alpha',
                    param_type=SearchSpaceType.CONTINUOUS,
                    min_value=0.1,
                    max_value=0.9
                ),
                'quantile_loss': ParameterRange(
                    name='quantile_loss',
                    param_type=SearchSpaceType.CATEGORICAL,
                    choices=['pinball', 'huber']
                )
            }

    def _calculate_dimensionality(self):
        """Calculate search space dimensionality."""
        self.dimensionality = len(self.parameter_ranges) + 1  # +1 for model_type

    def sample_architecture(self) -> TreeArchitectureCandidate:
        """Sample a random architecture from the search space."""
        # Sample model type
        model_type = random.choice(self.model_types)

        # Sample parameters
        hyperparams = {}
        for param_name, param_range in self.parameter_ranges.items():
            hyperparams[param_name] = param_range.sample()

        # Create architecture candidate
        architecture = TreeArchitectureCandidate(
            model_type=model_type,
            **hyperparams
        )

        # Apply constraints
        for constraint in self.constraints:
            if not constraint(architecture):
                # If constraint fails, try sampling again
                return self.sample_architecture()

        return architecture

    def sample_architectures(self, n: int) -> List[TreeArchitectureCandidate]:
        """Sample multiple architectures from the search space."""
        architectures = []
        for _ in range(n):
            architecture = self.sample_architecture()
            architectures.append(architecture)
        return architectures

    def is_valid_architecture(self, architecture: TreeArchitectureCandidate) -> bool:
        """Check if an architecture is valid in the search space."""
        # Check model type
        if architecture.model_type not in self.model_types:
            return False

        # Check parameter ranges
        for param_name, param_range in self.parameter_ranges.items():
            param_value = getattr(architecture, param_name, None)
            if param_value is not None and not param_range.is_valid(param_value):
                return False

        # Check constraints
        for constraint in self.constraints:
            if not constraint(architecture):
                return False

        return True

    def get_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Get parameter bounds for optimization algorithms."""
        bounds = {}

        for param_name, param_range in self.parameter_ranges.items():
            if param_range.param_type == SearchSpaceType.CONTINUOUS:
                bounds[param_name] = (param_range.min_value, param_range.max_value)
            elif param_range.param_type == SearchSpaceType.DISCRETE:
                bounds[param_name] = (float(param_range.min_value), float(param_range.max_value))

        return bounds

    def get_categorical_parameters(self) -> Dict[str, List[Any]]:
        """Get categorical parameters for optimization algorithms."""
        categorical = {}

        for param_name, param_range in self.parameter_ranges.items():
            if param_range.param_type == SearchSpaceType.CATEGORICAL:
                categorical[param_name] = param_range.choices

        return categorical

    def add_constraint(self, constraint: Callable[[TreeArchitectureCandidate], bool]):
        """Add a constraint to the search space."""
        self.constraints.append(constraint)

    def remove_constraint(self, constraint: Callable[[TreeArchitectureCandidate], bool]):
        """Remove a constraint from the search space."""
        if constraint in self.constraints:
            self.constraints.remove(constraint)

    def get_space_info(self) -> Dict[str, Any]:
        """Get information about the search space."""
        return {
            'model_types': [t.value for t in self.model_types],
            'n_parameters': len(self.parameter_ranges),
            'dimensionality': self.dimensionality,
            'space_type': self.space_type.value,
            'n_constraints': len(self.constraints),
            'parameter_names': list(self.parameter_ranges.keys()),
            'continuous_parameters': [
                name for name, param in self.parameter_ranges.items()
                if param.param_type == SearchSpaceType.CONTINUOUS
            ],
            'discrete_parameters': [
                name for name, param in self.parameter_ranges.items()
                if param.param_type == SearchSpaceType.DISCRETE
            ],
            'categorical_parameters': [
                name for name, param in self.parameter_ranges.items()
                if param.param_type == SearchSpaceType.CATEGORICAL
            ]
        }

@dataclass
class TreeArchitectureSpace:
    """Extended architecture space with advanced features."""

    # Base search space
    base_space: TreeSearchSpace

    # Advanced features
    enable_feature_selection: bool = True
    enable_ensemble_methods: bool = True
    enable_regularization: bool = True

    # Feature selection options
    feature_selection_methods: List[str] = field(default_factory=lambda: [
        'auto', 'sqrt', 'log2', 'none'
    ])

    # Ensemble options
    ensemble_methods: List[str] = field(default_factory=lambda: [
        'voting', 'stacking', 'bagging', 'boosting'
    ])

    # Regularization options
    regularization_methods: List[str] = field(default_factory=lambda: [
        'l1', 'l2', 'elastic_net', 'dropout'
    ])

    def sample_advanced_architecture(self) -> TreeArchitectureCandidate:
        """Sample an advanced architecture with additional features."""
        # Start with base architecture
        architecture = self.base_space.sample_architecture()

        # Add feature selection if enabled
        if self.enable_feature_selection:
            architecture.max_features = random.choice(self.feature_selection_methods)

        # Add regularization if enabled
        if self.enable_regularization:
            reg_method = random.choice(self.regularization_methods)
            if reg_method == 'l1':
                architecture.reg_alpha = random.uniform(0.0, 1.0)
            elif reg_method == 'l2':
                architecture.reg_lambda = random.uniform(0.0, 1.0)
            elif reg_method == 'elastic_net':
                architecture.reg_alpha = random.uniform(0.0, 1.0)
                architecture.reg_lambda = random.uniform(0.0, 1.0)

        return architecture

    def get_advanced_space_info(self) -> Dict[str, Any]:
        """Get information about the advanced search space."""
        base_info = self.base_space.get_space_info()
        base_info.update({
            'enable_feature_selection': self.enable_feature_selection,
            'enable_ensemble_methods': self.enable_ensemble_methods,
            'enable_regularization': self.enable_regularization,
            'feature_selection_methods': self.feature_selection_methods,
            'ensemble_methods': self.ensemble_methods,
            'regularization_methods': self.regularization_methods
        })
        return base_info

# Predefined search spaces
def create_quick_search_space() -> TreeSearchSpace:
    """Create a quick search space for fast exploration."""
    space = TreeSearchSpace()

    # Reduce parameter ranges for quick search
    space.parameter_ranges['n_trees'].max_value = 100
    space.parameter_ranges['max_depth'].max_value = 10
    space.parameter_ranges['min_samples_split'].max_value = 20
    space.parameter_ranges['min_samples_leaf'].max_value = 10

    return space

def create_comprehensive_search_space() -> TreeSearchSpace:
    """Create a comprehensive search space for thorough exploration."""
    space = TreeSearchSpace()

    # Expand parameter ranges for comprehensive search
    space.parameter_ranges['n_trees'].max_value = 2000
    space.parameter_ranges['max_depth'].max_value = 30
    space.parameter_ranges['min_samples_split'].max_value = 200
    space.parameter_ranges['min_samples_leaf'].max_value = 100

    # Add more model types
    space.model_types.extend([
        TreeModelType.GRADIENT_BOOSTING,
        TreeModelType.ADABOOST,
        TreeModelType.BAGGING
    ])

    return space

def create_regime_aware_search_space() -> TreeSearchSpace:
    """Create a regime-aware search space."""
    space = TreeSearchSpace()

    # Add regime-specific constraints
    def regime_constraint(arch: TreeArchitectureCandidate) -> bool:
        # Ensure architectures are robust to regime changes
        return arch.n_trees >= 50 and arch.max_depth >= 3

    space.add_constraint(regime_constraint)

    return space

def create_real_time_search_space() -> TreeSearchSpace:
    """Create a real-time search space for fast adaptation."""
    space = TreeSearchSpace()

    # Limit complexity for real-time performance
    space.parameter_ranges['n_trees'].max_value = 200
    space.parameter_ranges['max_depth'].max_value = 8

    # Add real-time constraints
    def real_time_constraint(arch: TreeArchitectureCandidate) -> bool:
        # Ensure architectures can be trained quickly
        complexity = arch.n_trees * (2 ** arch.max_depth)
        return complexity <= 10000

    space.add_constraint(real_time_constraint)

    return space
