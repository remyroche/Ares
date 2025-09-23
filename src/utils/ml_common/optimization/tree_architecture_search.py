"""
Tree Architecture Search (TAS) for ML Common

This module provides comprehensive Tree Architecture Search capabilities
specifically designed for tree-based models like Random Forest, XGBoost, etc.

Key Features:
- Evolutionary architecture search for tree structures
- Multi-objective optimization (accuracy + efficiency + interpretability)
- Tree-specific search spaces (depth, width, splitting strategies)
- Integration with existing tree-based models
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
import logging
import time
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class TreeArchitectureConfig:
    """Configuration for tree architecture search."""

    # Search space for tree structure
    min_depth: int = 3
    max_depth: int = 15
    min_trees: int = 10
    max_trees: int = 500
    min_features_per_split: int = 2
    max_features_per_split: int = 20

    # Splitting strategies
    splitting_strategies: List[str] = field(default_factory=lambda: [
        'gini', 'entropy', 'log_loss', 'friedman_mse', 'squared_error',
        'xgb_gbtree', 'xgb_gblinear', 'xgb_dart',
        'lgb_gbdt', 'lgb_rf', 'lgb_goss', 'lgb_dart'
    ])
    feature_selection_methods: List[str] = field(default_factory=lambda: [
        'sqrt', 'log2', 'auto', 'custom_ratio'
    ])

    # Multi-objective optimization
    objectives: List[str] = field(default_factory=lambda: ['accuracy', 'efficiency', 'interpretability'])
    objective_weights: List[float] = field(default_factory=lambda: [0.4, 0.3, 0.3])

    # Search parameters
    n_trials: int = 30
    population_size: int = 20
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8


@dataclass
class TreeArchitectureCandidate:
    """A candidate tree architecture."""

    # Tree structure parameters
    n_trees: int
    max_depth: int
    min_samples_split: int
    min_samples_leaf: int
    max_features: Union[str, int, float]
    splitting_strategy: str
    bootstrap: bool = True

    # Performance metrics
    accuracy: float = 0.0
    efficiency_score: float = 0.0
    interpretability_score: float = 0.0
    overall_score: float = 0.0

    # Training info
    training_time: float = 0.0
    model_size_mb: float = 0.0
    feature_importance_stability: float = 0.0

    # Metadata
    trial_number: int = 0
    created_at: datetime = field(default_factory=datetime.now)


class TreeArchitectureSearch:
    """Main Tree Architecture Search implementation."""

    def __init__(self, config: TreeArchitectureConfig):
        self.config = config
        self.logger = logger.getChild('TreeArchitectureSearch')
        self.candidates = []

        self.logger.info("✅ Tree Architecture Search initialized")

    def search(self,
               X_train: np.ndarray,
               y_train: np.ndarray,
               X_val: Optional[np.ndarray] = None,
               y_val: Optional[np.ndarray] = None) -> TreeArchitectureCandidate:
        """
        Perform tree architecture search.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)

        Returns:
            Best tree architecture candidate
        """
        self.logger.info("🚀 Starting Tree Architecture Search...")

        try:
            # Initialize search
            best_candidate = self._evolutionary_search(X_train, y_train, X_val, y_val)

            self.logger.info(f"✅ TAS completed: Best architecture has score {best_candidate.overall_score:.4f}")
            return best_candidate

        except Exception as e:
            self.logger.error(f"Tree Architecture Search failed: {e}")
            raise

    def _evolutionary_search(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray) -> TreeArchitectureCandidate:
        """Perform evolutionary search for tree architectures."""
        # Initialize population
        population = self._initialize_population()

        best_candidate = None
        best_score = -np.inf

        for generation in range(self.config.n_trials):
            # Evaluate population
            for candidate in population:
                if candidate.accuracy == 0.0:  # Not yet evaluated
                    self._evaluate_candidate(candidate, X_train, y_train, X_val, y_val)

            # Update best candidate
            for candidate in population:
                if candidate.overall_score > best_score:
                    best_score = candidate.overall_score
                    best_candidate = candidate

            # Evolve population
            population = self._evolve_population(population)

            self.logger.debug(f"Generation {generation}: Best score {best_score:.4f}")

        return best_candidate

    def _initialize_population(self) -> List[TreeArchitectureCandidate]:
        """Initialize population with random architectures."""
        population = []

        for i in range(self.config.population_size):
            candidate = TreeArchitectureCandidate(
                n_trees=np.random.randint(self.config.min_trees, self.config.max_trees + 1),
                max_depth=np.random.randint(self.config.min_depth, self.config.max_depth + 1),
                min_samples_split=np.random.randint(2, 20),
                min_samples_leaf=np.random.randint(1, 10),
                max_features=np.random.choice(self.config.feature_selection_methods),
                splitting_strategy=np.random.choice(self.config.splitting_strategies),
                trial_number=i
            )
            population.append(candidate)

        return population

    def _evaluate_candidate(self, candidate: TreeArchitectureCandidate,
                          X_train: np.ndarray, y_train: np.ndarray,
                          X_val: np.ndarray, y_val: np.ndarray) -> None:
        """Evaluate a single tree architecture candidate."""
        try:
            # Create model with candidate parameters
            model = self._create_model_from_candidate(candidate, y_train)

            # Train model
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time

            # Evaluate on validation set
            val_pred = model.predict(X_val)
            if len(y_val.shape) > 1:
                accuracy = np.mean(np.argmax(val_pred, axis=1) == np.argmax(y_val, axis=1))
            else:
                accuracy = model.score(X_val, y_val)

            # Calculate efficiency score (inverse of training time and model size)
            model_size = len(model.get_params()) * 4 / (1024 * 1024)  # Rough estimate
            efficiency_score = 1.0 / (1.0 + training_time + model_size)

            # Calculate interpretability score (based on tree simplicity)
            interpretability_score = self._calculate_interpretability_score(candidate)

            # Calculate overall score
            overall_score = (
                self.config.objective_weights[0] * accuracy +
                self.config.objective_weights[1] * efficiency_score +
                self.config.objective_weights[2] * interpretability_score
            )

            # Update candidate
            candidate.accuracy = accuracy
            candidate.efficiency_score = efficiency_score
            candidate.interpretability_score = interpretability_score
            candidate.overall_score = overall_score
            candidate.training_time = training_time
            candidate.model_size_mb = model_size

        except Exception as e:
            self.logger.warning(f"Candidate evaluation failed: {e}")
            candidate.overall_score = 0.0

    def _create_model_from_candidate(self, candidate: TreeArchitectureCandidate, y_train: np.ndarray = None):
        """Create a tree-based model from architecture candidate."""
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
        from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier

        # Try to import XGBoost and LightGBM
        try:
            import xgboost as xgb
            XGB_AVAILABLE = True
        except ImportError:
            XGB_AVAILABLE = False

        try:
            import lightgbm as lgb
            LGB_AVAILABLE = True
        except ImportError:
            LGB_AVAILABLE = False

        # Determine if it's a classification or regression problem
        n_classes = len(np.unique(y_train)) if y_train is not None and len(y_train.shape) == 1 else 2

        # Choose model type based on splitting strategy
        if candidate.splitting_strategy.startswith('xgb_'):
            # XGBoost
            if not XGB_AVAILABLE:
                raise ImportError("XGBoost not available")

            if n_classes > 2:
                model = xgb.XGBClassifier(
                    n_estimators=candidate.n_trees,
                    max_depth=candidate.max_depth,
                    min_child_weight=candidate.min_samples_leaf,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42
                )
            else:
                model = xgb.XGBRegressor(
                    n_estimators=candidate.n_trees,
                    max_depth=candidate.max_depth,
                    min_child_weight=candidate.min_samples_leaf,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42
                )

        elif candidate.splitting_strategy.startswith('lgb_'):
            # LightGBM
            if not LGB_AVAILABLE:
                raise ImportError("LightGBM not available")

            if n_classes > 2:
                model = lgb.LGBMClassifier(
                    n_estimators=candidate.n_trees,
                    max_depth=candidate.max_depth,
                    min_child_samples=candidate.min_samples_leaf,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    verbosity=-1
                )
            else:
                model = lgb.LGBMRegressor(
                    n_estimators=candidate.n_trees,
                    max_depth=candidate.max_depth,
                    min_child_samples=candidate.min_samples_leaf,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    verbosity=-1
                )

        elif candidate.splitting_strategy in ['gini', 'entropy', 'log_loss']:
            # Random Forest / Extra Trees
            if candidate.splitting_strategy == 'gini':
                # Random Forest
                if n_classes > 2:
                    model = RandomForestClassifier(
                        n_estimators=candidate.n_trees,
                        max_depth=candidate.max_depth,
                        min_samples_split=candidate.min_samples_split,
                        min_samples_leaf=candidate.min_samples_leaf,
                        max_features=candidate.max_features,
                        criterion='gini',
                        bootstrap=candidate.bootstrap,
                        random_state=42
                    )
                else:
                    model = RandomForestRegressor(
                        n_estimators=candidate.n_trees,
                        max_depth=candidate.max_depth,
                        min_samples_split=candidate.min_samples_split,
                        min_samples_leaf=candidate.min_samples_leaf,
                        max_features=candidate.max_features,
                        criterion='squared_error',
                        bootstrap=candidate.bootstrap,
                        random_state=42
                    )
            else:
                # Extra Trees
                if n_classes > 2:
                    model = ExtraTreesClassifier(
                        n_estimators=candidate.n_trees,
                        max_depth=candidate.max_depth,
                        min_samples_split=candidate.min_samples_split,
                        min_samples_leaf=candidate.min_samples_leaf,
                        max_features=candidate.max_features,
                        criterion=candidate.splitting_strategy,
                        bootstrap=candidate.bootstrap,
                        random_state=42
                    )
                else:
                    model = ExtraTreesRegressor(
                        n_estimators=candidate.n_trees,
                        max_depth=candidate.max_depth,
                        min_samples_split=candidate.min_samples_split,
                        min_samples_leaf=candidate.min_samples_leaf,
                        max_features=candidate.max_features,
                        criterion='squared_error',
                        bootstrap=candidate.bootstrap,
                        random_state=42
                    )

        else:
            # HistGradientBoosting
            if n_classes > 2:
                model = HistGradientBoostingClassifier(
                    max_iter=candidate.n_trees,
                    max_depth=candidate.max_depth,
                    min_samples_leaf=candidate.min_samples_leaf,
                    random_state=42
                )
            else:
                model = HistGradientBoostingRegressor(
                    max_iter=candidate.n_trees,
                    max_depth=candidate.max_depth,
                    min_samples_leaf=candidate.min_samples_leaf,
                    random_state=42
                )

        return model

    def _calculate_interpretability_score(self, candidate: TreeArchitectureCandidate) -> float:
        """Calculate interpretability score based on tree complexity."""
        # Simpler trees (fewer trees, less depth) are more interpretable
        complexity_penalty = candidate.n_trees * 0.001 + candidate.max_depth * 0.1
        return 1.0 / (1.0 + complexity_penalty)

    def _evolve_population(self, population: List[TreeArchitectureCandidate]) -> List[TreeArchitectureCandidate]:
        """Evolve population using genetic operators."""
        # Sort by fitness
        population.sort(key=lambda x: x.overall_score, reverse=True)

        # Keep best candidates (elitism)
        new_population = population[:int(self.config.population_size * 0.2)]

        # Generate new candidates through crossover and mutation
        while len(new_population) < self.config.population_size:
            if np.random.random() < self.config.crossover_rate:
                # Crossover
                parent1 = np.random.choice(population[:10])  # Choose from best
                parent2 = np.random.choice(population[:10])

                child = self._crossover(parent1, parent2)
                new_population.append(child)
            else:
                # Mutation
                parent = np.random.choice(population[:10])
                child = self._mutate(parent)
                new_population.append(child)

        return new_population

    def _crossover(self, parent1: TreeArchitectureCandidate, parent2: TreeArchitectureCandidate) -> TreeArchitectureCandidate:
        """Perform crossover between two parent architectures."""
        child = TreeArchitectureCandidate(
            n_trees=(parent1.n_trees + parent2.n_trees) // 2,
            max_depth=(parent1.max_depth + parent2.max_depth) // 2,
            min_samples_split=(parent1.min_samples_split + parent2.min_samples_split) // 2,
            min_samples_leaf=(parent1.min_samples_leaf + parent2.min_samples_leaf) // 2,
            max_features=np.random.choice([parent1.max_features, parent2.max_features]),
            splitting_strategy=np.random.choice([parent1.splitting_strategy, parent2.splitting_strategy]),
            bootstrap=np.random.choice([parent1.bootstrap, parent2.bootstrap]),
            trial_number=len(self.candidates)
        )

        self.candidates.append(child)
        return child

    def _mutate(self, parent: TreeArchitectureCandidate) -> TreeArchitectureCandidate:
        """Mutate a parent architecture."""
        mutated = TreeArchitectureCandidate(
            n_trees=max(self.config.min_trees, min(self.config.max_trees,
                  parent.n_trees + np.random.randint(-20, 20))),
            max_depth=max(self.config.min_depth, min(self.config.max_depth,
                    parent.max_depth + np.random.randint(-3, 3))),
            min_samples_split=max(2, parent.min_samples_split + np.random.randint(-3, 3)),
            min_samples_leaf=max(1, parent.min_samples_leaf + np.random.randint(-2, 2)),
            max_features=np.random.choice(self.config.feature_selection_methods),
            splitting_strategy=np.random.choice(self.config.splitting_strategies),
            bootstrap=not parent.bootstrap if np.random.random() < 0.1 else parent.bootstrap,
            trial_number=len(self.candidates)
        )

        self.candidates.append(mutated)
        return mutated


# Convenience function
def search_tree_architecture(X_train: np.ndarray,
                            y_train: np.ndarray,
                            X_val: Optional[np.ndarray] = None,
                            y_val: Optional[np.ndarray] = None,
                            config: Optional[TreeArchitectureConfig] = None) -> TreeArchitectureCandidate:
    """
    Convenience function to perform tree architecture search.

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features (optional)
        y_val: Validation labels (optional)
        config: Tree architecture search configuration

    Returns:
        Best tree architecture candidate
    """
    if config is None:
        config = TreeArchitectureConfig()

    tas = TreeArchitectureSearch(config)
    return tas.search(X_train, y_train, X_val, y_val)