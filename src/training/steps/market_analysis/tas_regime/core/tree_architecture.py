"""
Tree Architecture Classes

Classes for representing and managing tree-based architectures in TAS.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import json
import random
from datetime import datetime
from enum import Enum

# Import tprint for comprehensive logging
from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error,
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)

from ..core.tas_config import TreeModelType, OptimizationObjective

class ArchitectureStatus(Enum):
    """Status of tree architecture."""
    PENDING = "pending"
    EVALUATING = "evaluating"
    EVALUATED = "evaluated"
    FAILED = "failed"
    OPTIMIZED = "optimized"

@dataclass
class TreeArchitectureCandidate:
    """Candidate tree architecture for TAS."""

    # Architecture parameters
    model_type: TreeModelType = TreeModelType.RANDOM_FOREST
    n_trees: int = 100
    max_depth: int = 10
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: Union[int, float, str] = "auto"

    # Hyperparameters
    learning_rate: Optional[float] = None
    subsample: Optional[float] = None
    colsample_bytree: Optional[float] = None
    reg_alpha: Optional[float] = None
    reg_lambda: Optional[float] = None

    # NGBoost specific parameters
    base_learner: Optional[str] = None
    natural_gradient: Optional[bool] = None
    expected_information: Optional[bool] = None

    # DART specific parameters
    dart_drop_rate: Optional[float] = None
    dart_skip_drop: Optional[float] = None

    # DeepGBM specific parameters
    num_layers: Optional[int] = None
    layer_size: Optional[int] = None

    # Quantile GBDT specific parameters
    quantile_alpha: Optional[float] = None
    quantile_loss: Optional[str] = None

    # Performance metrics
    scores: Dict[str, float] = field(default_factory=dict)
    overall_score: float = 0.0

    # Architecture metadata
    status: ArchitectureStatus = ArchitectureStatus.PENDING
    evaluation_time: float = 0.0
    memory_usage: float = 0.0

    # Search metadata
    search_iteration: int = 0
    search_strategy: str = "unknown"
    parent_architectures: List[int] = field(default_factory=list)

    # Advanced features
    feature_importance: Optional[np.ndarray] = None
    uncertainty_estimates: Optional[Dict[str, float]] = None
    robustness_scores: Optional[Dict[str, float]] = None

    def __post_init__(self):
        """Post-initialization processing."""
        if self.learning_rate is None:
            self.learning_rate = self._get_default_learning_rate()
        if self.subsample is None:
            self.subsample = self._get_default_subsample()
        if self.colsample_bytree is None:
            self.colsample_bytree = self._get_default_colsample_bytree()
        if self.base_learner is None:
            self.base_learner = self._get_default_base_learner()
        if self.natural_gradient is None:
            self.natural_gradient = self._get_default_natural_gradient()
        if self.dart_drop_rate is None:
            self.dart_drop_rate = self._get_default_dart_drop_rate()
        if self.num_layers is None:
            self.num_layers = self._get_default_num_layers()
        if self.quantile_alpha is None:
            self.quantile_alpha = self._get_default_quantile_alpha()

    def _get_default_learning_rate(self) -> float:
        """Get default learning rate based on model type."""
        defaults = {
            TreeModelType.XGBOOST: 0.1,
            TreeModelType.LIGHTGBM: 0.1,
            TreeModelType.GRADIENT_BOOSTING: 0.1,
            TreeModelType.ADABOOST: 1.0
        }
        return defaults.get(self.model_type, 0.1)

    def _get_default_subsample(self) -> float:
        """Get default subsample based on model type."""
        defaults = {
            TreeModelType.XGBOOST: 1.0,
            TreeModelType.LIGHTGBM: 1.0,
            TreeModelType.GRADIENT_BOOSTING: 1.0,
            TreeModelType.RANDOM_FOREST: 1.0,
            TreeModelType.EXTRA_TREES: 1.0,
            TreeModelType.BAGGING: 1.0
        }
        return defaults.get(self.model_type, 1.0)

    def _get_default_colsample_bytree(self) -> float:
        """Get default colsample_bytree based on model type."""
        defaults = {
            TreeModelType.XGBOOST: 1.0,
            TreeModelType.LIGHTGBM: 1.0,
            TreeModelType.GRADIENT_BOOSTING: 1.0,
            TreeModelType.DART: 1.0,
            TreeModelType.DEEPGBM: 1.0
        }
        return defaults.get(self.model_type, 1.0)

    def _get_default_base_learner(self) -> str:
        """Get default base learner for NGBoost."""
        return "random_forest"  # Use RandomForest instead of decision_tree

    def _get_default_natural_gradient(self) -> bool:
        """Get default natural gradient setting."""
        return True

    def _get_default_dart_drop_rate(self) -> float:
        """Get default DART drop rate."""
        return 0.1

    def _get_default_num_layers(self) -> int:
        """Get default number of layers for DeepGBM."""
        return 3

    def _get_default_quantile_alpha(self) -> float:
        """Get default quantile alpha."""
        return 0.5

    def to_dict(self) -> Dict[str, Any]:
        """Convert architecture to dictionary."""
        return {
            'model_type': self.model_type.value,
            'n_trees': self.n_trees,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'max_features': self.max_features,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'base_learner': self.base_learner,
            'natural_gradient': self.natural_gradient,
            'expected_information': self.expected_information,
            'dart_drop_rate': self.dart_drop_rate,
            'dart_skip_drop': self.dart_skip_drop,
            'num_layers': self.num_layers,
            'layer_size': self.layer_size,
            'quantile_alpha': self.quantile_alpha,
            'quantile_loss': self.quantile_loss,
            'scores': self.scores,
            'overall_score': self.overall_score,
            'status': self.status.value,
            'evaluation_time': self.evaluation_time,
            'memory_usage': self.memory_usage,
            'search_iteration': self.search_iteration,
            'search_strategy': self.search_strategy,
            'parent_architectures': self.parent_architectures,
            'feature_importance': self.feature_importance.tolist() if self.feature_importance is not None else None,
            'uncertainty_estimates': self.uncertainty_estimates,
            'robustness_scores': self.robustness_scores
        }

    @classmethod
    def from_dict(cls, arch_dict: Dict[str, Any]) -> 'TreeArchitectureCandidate':
        """Create architecture from dictionary."""
        # Convert string values back to enums
        if 'model_type' in arch_dict:
            arch_dict['model_type'] = TreeModelType(arch_dict['model_type'])
        if 'status' in arch_dict:
            arch_dict['status'] = ArchitectureStatus(arch_dict['status'])

        # Convert feature importance back to numpy array
        if 'feature_importance' in arch_dict and arch_dict['feature_importance'] is not None:
            arch_dict['feature_importance'] = np.array(arch_dict['feature_importance'])

        return cls(**arch_dict)

    def get_hyperparameters(self) -> Dict[str, Any]:
        """Get hyperparameters for the architecture."""
        hyperparams = {
            'n_trees': self.n_trees,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'max_features': self.max_features
        }

        # Add model-specific hyperparameters
        if self.learning_rate is not None:
            hyperparams['learning_rate'] = self.learning_rate
        if self.subsample is not None:
            hyperparams['subsample'] = self.subsample
        if self.colsample_bytree is not None:
            hyperparams['colsample_bytree'] = self.colsample_bytree
        if self.reg_alpha is not None:
            hyperparams['reg_alpha'] = self.reg_alpha
        if self.reg_lambda is not None:
            hyperparams['reg_lambda'] = self.reg_lambda
        if self.base_learner is not None:
            hyperparams['base_learner'] = self.base_learner
        if self.natural_gradient is not None:
            hyperparams['natural_gradient'] = self.natural_gradient
        if self.expected_information is not None:
            hyperparams['expected_information'] = self.expected_information
        if self.dart_drop_rate is not None:
            hyperparams['dart_drop_rate'] = self.dart_drop_rate
        if self.dart_skip_drop is not None:
            hyperparams['dart_skip_drop'] = self.dart_skip_drop
        if self.num_layers is not None:
            hyperparams['num_layers'] = self.num_layers
        if self.layer_size is not None:
            hyperparams['layer_size'] = self.layer_size
        if self.quantile_alpha is not None:
            hyperparams['quantile_alpha'] = self.quantile_alpha
        if self.quantile_loss is not None:
            hyperparams['quantile_loss'] = self.quantile_loss

        return hyperparams

    def get_complexity_score(self) -> float:
        """Calculate complexity score for the architecture."""
        # Base complexity from number of trees and depth
        tree_complexity = self.n_trees * (2 ** self.max_depth)

        # Feature complexity
        if isinstance(self.max_features, (int, float)):
            feature_complexity = self.max_features
        else:
            feature_complexity = 1.0  # Default for "auto", "sqrt", "log2"

        # Regularization complexity
        reg_complexity = 1.0
        if self.reg_alpha is not None and self.reg_alpha > 0:
            reg_complexity += self.reg_alpha
        if self.reg_lambda is not None and self.reg_lambda > 0:
            reg_complexity += self.reg_lambda

        # Combined complexity score
        complexity = tree_complexity * feature_complexity * reg_complexity

        # Normalize to [0, 1] range
        return min(complexity / 1000000, 1.0)

    def get_efficiency_score(self) -> float:
        """Calculate efficiency score for the architecture."""
        # Efficiency based on performance per complexity
        if self.overall_score <= 0:
            return 0.0

        complexity = self.get_complexity_score()
        if complexity <= 0:
            return 0.0

        efficiency = self.overall_score / complexity
        return min(efficiency, 1.0)

    def is_valid(self) -> bool:
        """Check if architecture is valid."""
        # Check basic constraints
        if self.n_trees <= 0 or self.max_depth <= 0:
            return False

        if self.min_samples_split < 2 or self.min_samples_leaf < 1:
            return False

        if self.min_samples_split < 2 * self.min_samples_leaf:
            return False

        # Check model-specific constraints
        if self.model_type in [TreeModelType.XGBOOST, TreeModelType.LIGHTGBM]:
            if self.learning_rate is not None and (self.learning_rate <= 0 or self.learning_rate > 1):
                return False

        if self.subsample is not None and (self.subsample <= 0 or self.subsample > 1):
            return False

        if self.colsample_bytree is not None and (self.colsample_bytree <= 0 or self.colsample_bytree > 1):
            return False

        return True

    def mutate(self, mutation_rate: float = 0.1) -> 'TreeArchitectureCandidate':
        """Create a mutated version of the architecture."""
        tprint_debug(f"🧬 Mutating architecture with rate: {mutation_rate}")
        import random

        # Create a copy
        tprint_debug("📋 Creating base copy of architecture...")
        mutated = TreeArchitectureCandidate(
            model_type=self.model_type,
            n_trees=self.n_trees,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            parent_architectures=self.parent_architectures + [id(self)]
        )

        # Mutate parameters with given probability
        mutations_applied = []

        if random.random() < mutation_rate:
            old_value = mutated.n_trees
            mutated.n_trees = max(1, int(self.n_trees * random.uniform(0.8, 1.2)))
            mutations_applied.append(f"n_trees: {old_value} -> {mutated.n_trees}")

        if random.random() < mutation_rate:
            old_value = mutated.max_depth
            mutated.max_depth = max(1, int(self.max_depth * random.uniform(0.8, 1.2)))
            mutations_applied.append(f"max_depth: {old_value} -> {mutated.max_depth}")

        if random.random() < mutation_rate:
            old_value = mutated.min_samples_split
            mutated.min_samples_split = max(2, int(self.min_samples_split * random.uniform(0.8, 1.2)))
            mutations_applied.append(f"min_samples_split: {old_value} -> {mutated.min_samples_split}")

        if random.random() < mutation_rate:
            old_value = mutated.min_samples_leaf
            mutated.min_samples_leaf = max(1, int(self.min_samples_leaf * random.uniform(0.8, 1.2)))
            mutations_applied.append(f"min_samples_leaf: {old_value} -> {mutated.min_samples_leaf}")

        if random.random() < mutation_rate and self.learning_rate is not None:
            old_value = mutated.learning_rate
            mutated.learning_rate = max(0.01, min(1.0, self.learning_rate * random.uniform(0.8, 1.2)))
            mutations_applied.append(f"learning_rate: {old_value} -> {mutated.learning_rate}")

        if random.random() < mutation_rate and self.subsample is not None:
            old_value = mutated.subsample
            mutated.subsample = max(0.1, min(1.0, self.subsample * random.uniform(0.8, 1.2)))
            mutations_applied.append(f"subsample: {old_value} -> {mutated.subsample}")

        if random.random() < mutation_rate and self.colsample_bytree is not None:
            old_value = mutated.colsample_bytree
            mutated.colsample_bytree = max(0.1, min(1.0, self.colsample_bytree * random.uniform(0.8, 1.2)))
            mutations_applied.append(f"colsample_bytree: {old_value} -> {mutated.colsample_bytree}")

        if mutations_applied:
            tprint_debug(f"✅ Applied {len(mutations_applied)} mutations: {', '.join(mutations_applied)}")
        else:
            tprint_debug("🚫 No mutations applied")

        return mutated

    def crossover(self, other: 'TreeArchitectureCandidate') -> 'TreeArchitectureCandidate':
        """Create a crossover between two architectures."""
        tprint_debug("🔀 Performing crossover between two architectures")

        # Create offspring
        tprint_debug("🧬 Creating offspring from parent architectures...")
        offspring = TreeArchitectureCandidate(
            model_type=random.choice([self.model_type, other.model_type]),
            n_trees=random.choice([self.n_trees, other.n_trees]),
            max_depth=random.choice([self.max_depth, other.max_depth]),
            min_samples_split=random.choice([self.min_samples_split, other.min_samples_split]),
            min_samples_leaf=random.choice([self.min_samples_leaf, other.min_samples_leaf]),
            max_features=random.choice([self.max_features, other.max_features]),
            learning_rate=random.choice([self.learning_rate, other.learning_rate]),
            subsample=random.choice([self.subsample, other.subsample]),
            colsample_bytree=random.choice([self.colsample_bytree, other.colsample_bytree]),
            reg_alpha=random.choice([self.reg_alpha, other.reg_alpha]),
            reg_lambda=random.choice([self.reg_lambda, other.reg_lambda]),
            parent_architectures=[id(self), id(other)]
        )

        tprint_debug(f"✅ Offspring created with model_type: {offspring.model_type.value}, n_trees: {offspring.n_trees}")
        return offspring

    def __str__(self) -> str:
        """String representation of the architecture."""
        return f"TreeArchitecture({self.model_type.value}, trees={self.n_trees}, depth={self.max_depth}, score={self.overall_score:.4f})"

    def __repr__(self) -> str:
        """Detailed string representation of the architecture."""
        return f"TreeArchitectureCandidate(model_type={self.model_type.value}, n_trees={self.n_trees}, max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, min_samples_leaf={self.min_samples_leaf}, max_features={self.max_features}, overall_score={self.overall_score:.4f}, status={self.status.value})"

@dataclass
class TreeArchitecture:
    """Complete tree architecture with additional metadata."""

    # Core architecture
    candidate: TreeArchitectureCandidate

    # Additional metadata
    creation_time: str = field(default_factory=lambda: datetime.now().isoformat())
    optimization_history: List[Dict[str, Any]] = field(default_factory=list)

    # Performance tracking
    training_scores: Dict[str, float] = field(default_factory=dict)
    validation_scores: Dict[str, float] = field(default_factory=dict)
    test_scores: Optional[Dict[str, float]] = None

    # Architecture analysis
    feature_importance: Optional[np.ndarray] = None
    tree_analysis: Optional[Dict[str, Any]] = None
    complexity_analysis: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert architecture to dictionary."""
        tprint_debug("🔄 Converting TreeArchitecture to dictionary")
        result = {
            'candidate': self.candidate.to_dict(),
            'creation_time': self.creation_time,
            'optimization_history': self.optimization_history,
            'training_scores': self.training_scores,
            'validation_scores': self.validation_scores,
            'test_scores': self.test_scores,
            'feature_importance': self.feature_importance.tolist() if self.feature_importance is not None else None,
            'tree_analysis': self.tree_analysis,
            'complexity_analysis': self.complexity_analysis
        }
        tprint_debug(f"✅ TreeArchitecture converted to dictionary with {len(result)} keys")
        return result

    @classmethod
    def from_dict(cls, arch_dict: Dict[str, Any]) -> 'TreeArchitecture':
        """Create architecture from dictionary."""
        tprint_debug("🔄 Creating TreeArchitecture from dictionary")

        # Reconstruct candidate
        tprint_debug("🧩 Reconstructing TreeArchitectureCandidate")
        candidate = TreeArchitectureCandidate.from_dict(arch_dict['candidate'])

        # Convert feature importance back to numpy array
        tprint_debug("🔢 Converting feature importance to numpy array")
        feature_importance = None
        if arch_dict.get('feature_importance') is not None:
            feature_importance = np.array(arch_dict['feature_importance'])

        tprint_debug("🏗️ Building TreeArchitecture instance")
        result = cls(
            candidate=candidate,
            creation_time=arch_dict.get('creation_time', datetime.now().isoformat()),
            optimization_history=arch_dict.get('optimization_history', []),
            training_scores=arch_dict.get('training_scores', {}),
            validation_scores=arch_dict.get('validation_scores', {}),
            test_scores=arch_dict.get('test_scores'),
            feature_importance=feature_importance,
            tree_analysis=arch_dict.get('tree_analysis'),
            complexity_analysis=arch_dict.get('complexity_analysis')
        )
        tprint_debug(f"✅ TreeArchitecture created from dictionary with model_type: {result.candidate.model_type.value}")
        return result

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary of the architecture."""
        tprint_debug("📊 Generating TreeArchitecture performance summary")
        summary = {
            'overall_score': self.candidate.overall_score,
            'training_scores': self.training_scores,
            'validation_scores': self.validation_scores,
            'test_scores': self.test_scores,
            'complexity_score': self.candidate.get_complexity_score(),
            'efficiency_score': self.candidate.get_efficiency_score(),
            'is_valid': self.candidate.is_valid()
        }
        tprint_debug(f"✅ Performance summary generated with overall_score: {self.candidate.overall_score}")
        return summary

    def update_scores(self, scores: Dict[str, float], dataset_type: str = "validation"):
        """Update scores for the architecture."""
        tprint_debug(f"📊 Updating scores for {dataset_type} dataset: {scores}")

        if dataset_type == "training":
            self.training_scores.update(scores)
            tprint_debug(f"✅ Updated training scores: {self.training_scores}")
        elif dataset_type == "validation":
            self.validation_scores.update(scores)
            tprint_debug(f"✅ Updated validation scores: {self.validation_scores}")
        elif dataset_type == "test":
            if self.test_scores is None:
                self.test_scores = {}
            self.test_scores.update(scores)
            tprint_debug(f"✅ Updated test scores: {self.test_scores}")

        # Update overall score
        if "overall_score" in scores:
            old_score = self.candidate.overall_score
            self.candidate.overall_score = scores["overall_score"]
            tprint_debug(f"📈 Overall score updated: {old_score} -> {self.candidate.overall_score}")

    def add_optimization_step(self, step_info: Dict[str, Any]):
        """Add optimization step to history."""
        tprint_debug(f"📝 Adding optimization step: {step_info}")
        step_info['timestamp'] = datetime.now().isoformat()
        self.optimization_history.append(step_info)
        tprint_debug(f"✅ Optimization history now has {len(self.optimization_history)} steps")

    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get optimization summary."""
        if not self.optimization_history:
            return {}

        return {
            'n_steps': len(self.optimization_history),
            'first_step': self.optimization_history[0],
            'last_step': self.optimization_history[-1],
            'improvement': self.optimization_history[-1].get('score', 0) - self.optimization_history[0].get('score', 0)
        }
