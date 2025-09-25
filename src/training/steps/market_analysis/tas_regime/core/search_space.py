"""
Tree Search Space Classes

Classes for defining and managing the search space for tree architectures.

This module now uses the consolidated search space implementation from src/utils/nas_tas/.
"""

# Import the consolidated search space implementation
from src.utils.nas_tas.search_space import (
    SearchSpace, SearchSpaceConfig, ParameterRange, SearchSpaceType, OptimizationStrategy,
    create_tree_search_space
)

# Import TAS-specific classes
from .tas_config import TreeModelType, OptimizationObjective
from .tree_architecture import TreeArchitectureCandidate


# TAS-specific wrapper classes that use the consolidated implementation
@dataclass
class TreeSearchSpace:
    """Search space for tree architectures using consolidated implementation."""
    
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
    
    # Constraints
    constraints: List[Callable[[TreeArchitectureCandidate], bool]] = field(default_factory=list)
    
    # Search space metadata
    space_type: SearchSpaceType = SearchSpaceType.MIXED
    dimensionality: int = 0
    
    def __post_init__(self):
        """Initialize search space."""
        # Create the consolidated search space
        self._search_space = create_tree_search_space()
        self._calculate_dimensionality()
    
    def _calculate_dimensionality(self):
        """Calculate search space dimensionality."""
        self.dimensionality = len(self._search_space.parameters) + 1  # +1 for model_type
    
    def sample_architecture(self) -> TreeArchitectureCandidate:
        """Sample a random architecture from the search space."""
        import random
        
        # Sample model type
        model_type = random.choice(self.model_types)
        
        # Sample parameters using consolidated search space
        params = self._search_space.sample_parameters(1)[0]
        
        # Create architecture candidate
        architecture = TreeArchitectureCandidate(
            model_type=model_type,
            **params
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
        
        # Check constraints
        for constraint in self.constraints:
            if not constraint(architecture):
                return False
        
        return True
    
    def get_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Get parameter bounds for optimization algorithms."""
        bounds = {}
        
        for name, param in self._search_space.parameters.items():
            if param.param_type == SearchSpaceType.CONTINUOUS:
                bounds[name] = (param.min_value, param.max_value)
            elif param.param_type == SearchSpaceType.DISCRETE:
                bounds[name] = (float(param.min_value), float(param.max_value))
        
        return bounds
    
    def get_categorical_parameters(self) -> Dict[str, List[Any]]:
        """Get categorical parameters for optimization algorithms."""
        categorical = {}
        
        for name, param in self._search_space.parameters.items():
            if param.param_type == SearchSpaceType.CATEGORICAL:
                categorical[name] = param.choices
        
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
            'n_parameters': len(self._search_space.parameters),
            'dimensionality': self.dimensionality,
            'space_type': self.space_type.value,
            'n_constraints': len(self.constraints),
            'parameter_names': list(self._search_space.parameters.keys()),
            'continuous_parameters': [
                name for name, param in self._search_space.parameters.items()
                if param.param_type == SearchSpaceType.CONTINUOUS
            ],
            'discrete_parameters': [
                name for name, param in self._search_space.parameters.items()
                if param.param_type == SearchSpaceType.DISCRETE
            ],
            'categorical_parameters': [
                name for name, param in self._search_space.parameters.items()
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
        import random
        
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
    space._search_space.parameters['n_estimators'].max_value = 100
    space._search_space.parameters['max_depth'].max_value = 10
    space._search_space.parameters['min_samples_split'].max_value = 20
    space._search_space.parameters['min_samples_leaf'].max_value = 10
    
    return space


def create_comprehensive_search_space() -> TreeSearchSpace:
    """Create a comprehensive search space for thorough exploration."""
    space = TreeSearchSpace()
    
    # Expand parameter ranges for comprehensive search
    space._search_space.parameters['n_estimators'].max_value = 2000
    space._search_space.parameters['max_depth'].max_value = 30
    space._search_space.parameters['min_samples_split'].max_value = 200
    space._search_space.parameters['min_samples_leaf'].max_value = 100
    
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
        return arch.n_estimators >= 50 and arch.max_depth >= 3
    
    space.add_constraint(regime_constraint)
    
    return space


def create_real_time_search_space() -> TreeSearchSpace:
    """Create a real-time search space for fast adaptation."""
    space = TreeSearchSpace()
    
    # Limit complexity for real-time performance
    space._search_space.parameters['n_estimators'].max_value = 200
    space._search_space.parameters['max_depth'].max_value = 8
    
    # Add real-time constraints
    def real_time_constraint(arch: TreeArchitectureCandidate) -> bool:
        # Ensure architectures can be trained quickly
        complexity = arch.n_estimators * (2 ** arch.max_depth)
        return complexity <= 10000
    
    space.add_constraint(real_time_constraint)
    
    return space