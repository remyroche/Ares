"""
Unified Ensemble Search Space for NAS and TAS Systems

This module provides a comprehensive ensemble search space that can be used by both
NAS and TAS systems to explore ensemble combinations, weighting strategies, and
ensemble optimization techniques.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from datetime import datetime
import random
from collections import defaultdict
from src.utils.tprint import (tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, tprint_success, tprint_progress, tprint_performance, tprint_timer)
from .architecture_encoders import BaseArchitectureEncoder, UnifiedArchitectureEncoder
from .performance_estimators import UnifiedPerformanceEstimator
from .constraint_systems import UnifiedConstraintValidator

logger = logging.getLogger(__name__)

class EnsembleMethod(Enum):
    """Types of ensemble methods."""
    VOTING = "voting"
    WEIGHTED_VOTING = "weighted_voting"
    STACKING = "stacking"
    BAGGING = "bagging"
    BOOSTING = "boosting"
    ADAPTIVE_WEIGHTING = "adaptive_weighting"
    DYNAMIC_WEIGHTING = "dynamic_weighting"
    UNCERTAINTY_WEIGHTING = "uncertainty_weighting"
    REGIME_WEIGHTING = "regime_weighting"
    HYBRID_ENSEMBLE = "hybrid_ensemble"

class EnsembleCombinationStrategy(Enum):
    """Strategies for combining different architecture types."""
    NEURAL_ONLY = "neural_only"
    TREE_ONLY = "tree_only"
    HYBRID_MIXED = "hybrid_mixed"
    ADAPTIVE_SELECTION = "adaptive_selection"
    PERFORMANCE_BASED = "performance_based"
    REGIME_BASED = "regime_based"

@dataclass
class EnsembleSearchSpaceConfig:
    """Configuration for ensemble search space."""

    # Ensemble composition
    min_models: int = 2
    max_models: int = 8
    min_neural_models: int = 1
    max_neural_models: int = 5
    min_tree_models: int = 1
    max_tree_models: int = 5

    # Ensemble methods
    allowed_ensemble_methods: List[EnsembleMethod] = field(default_factory=lambda: [
        EnsembleMethod.VOTING,
        EnsembleMethod.WEIGHTED_VOTING,
        EnsembleMethod.STACKING,
        EnsembleMethod.ADAPTIVE_WEIGHTING,
        EnsembleMethod.DYNAMIC_WEIGHTING,
        EnsembleMethod.UNCERTAINTY_WEIGHTING,
        EnsembleMethod.HYBRID_ENSEMBLE
    ])

    # Combination strategies
    allowed_combination_strategies: List[EnsembleCombinationStrategy] = field(default_factory=lambda: [
        EnsembleCombinationStrategy.HYBRID_MIXED,
        EnsembleCombinationStrategy.ADAPTIVE_SELECTION,
        EnsembleCombinationStrategy.PERFORMANCE_BASED,
        EnsembleCombinationStrategy.REGIME_BASED
    ])

    # Weight constraints
    min_weight: float = 0.05
    max_weight: float = 0.8
    weight_sum_constraint: bool = True

    # Diversity requirements
    min_diversity_threshold: float = 0.3
    max_correlation_threshold: float = 0.8

    # Performance constraints
    min_individual_performance: float = 0.6
    min_ensemble_performance: float = 0.7

    # Optimization parameters
    enable_weight_optimization: bool = True
    enable_architecture_selection: bool = True
    enable_dynamic_adaptation: bool = True

@dataclass
class EnsembleArchitecture:
    """Represents an ensemble architecture."""
    ensemble_id: str
    models: List[Dict[str, Any]]
    ensemble_method: EnsembleMethod
    combination_strategy: EnsembleCombinationStrategy
    weights: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate ensemble architecture."""
        if len(self.models) != len(self.weights):
            raise ValueError("Number of models must match number of weights")

        if abs(sum(self.weights) - 1.0) > 1e-6:
            if self.metadata.get('weight_sum_constraint', True):
                # Normalize weights
                total_weight = sum(self.weights)
                self.weights = [w / total_weight for w in self.weights]

@dataclass
class EnsembleSearchResult:
    """Result from ensemble search."""
    best_ensemble: EnsembleArchitecture
    ensemble_score: float
    individual_scores: List[float]
    ensemble_diversity: float
    ensemble_robustness: float
    search_history: List[Dict[str, Any]]
    execution_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class UnifiedEnsembleSearchSpace:
    """Unified ensemble search space for NAS and TAS systems."""

    def __init__(self,
                 config: EnsembleSearchSpaceConfig,
                 nas_models: List[Dict[str, Any]] = None,
                 tas_models: List[Dict[str, Any]] = None,
                 performance_estimator: UnifiedPerformanceEstimator = None,
                 constraint_validator: UnifiedConstraintValidator = None):
        """Initialize the unified ensemble search space.

        Args:
            config: Ensemble search space configuration
            nas_models: Available NAS models for ensemble
            tas_models: Available TAS models for ensemble
            performance_estimator: Performance estimator for ensemble evaluation
            constraint_validator: Constraint validator for ensemble validation
        """
        self.config = config
        self.nas_models = nas_models or []
        self.tas_models = tas_models or []
        self.performance_estimator = performance_estimator
        self.constraint_validator = constraint_validator

        self.logger = logging.getLogger(self.__class__.__name__)

        # Available models
        self.all_models = self.nas_models + self.tas_models

        # Search state
        self.search_history = []
        self.ensemble_cache = {}

        self.logger.info("✅ Unified Ensemble Search Space initialized")
        self.logger.info(f"   NAS Models: {len(self.nas_models)}")
        self.logger.info(f"   TAS Models: {len(self.tas_models)}")
        self.logger.info(f"   Ensemble Methods: {len(config.allowed_ensemble_methods)}")
        self.logger.info(f"   Combination Strategies: {len(config.allowed_combination_strategies)}")

    def sample_ensemble_architecture(self) -> EnsembleArchitecture:
        """Sample a random ensemble architecture from the search space."""
        try:
            # Determine ensemble size
            n_models = random.randint(self.config.min_models, self.config.max_models)

            # Select models based on combination strategy
            combination_strategy = random.choice(self.config.allowed_combination_strategies)
            selected_models = self._select_models_for_ensemble(n_models, combination_strategy)

            # Select ensemble method
            ensemble_method = random.choice(self.config.allowed_ensemble_methods)

            # Generate weights
            weights = self._generate_ensemble_weights(len(selected_models), ensemble_method)

            # Create ensemble architecture
            ensemble_id = f"ensemble_{len(self.search_history)}_{int(time.time())}"
            ensemble = EnsembleArchitecture(
                ensemble_id=ensemble_id,
                models=selected_models,
                ensemble_method=ensemble_method,
                combination_strategy=combination_strategy,
                weights=weights,
                metadata={
                    'sampling_time': datetime.now(),
                    'search_space_config': self.config.__dict__
                }
            )

            # Validate ensemble
            if self.constraint_validator:
                validation_result = self._validate_ensemble(ensemble)
                if not validation_result.get('is_valid', True):
                    # Try to fix or resample
                    return self.sample_ensemble_architecture()

            return ensemble

        except Exception as e:
            self.logger.warning(f"Ensemble sampling failed: {e}")
            return self._create_fallback_ensemble()

    def _select_models_for_ensemble(self, n_models: int, strategy: EnsembleCombinationStrategy) -> List[Dict[str, Any]]:
        """Select models for ensemble based on strategy."""
        if len(self.all_models) < n_models:
            return self.all_models.copy()

        if strategy == EnsembleCombinationStrategy.NEURAL_ONLY:
            available_models = [m for m in self.all_models if m.get('type') == 'neural']
            if len(available_models) < n_models:
                available_models = self.nas_models
        elif strategy == EnsembleCombinationStrategy.TREE_ONLY:
            available_models = [m for m in self.all_models if m.get('type') == 'tree']
            if len(available_models) < n_models:
                available_models = self.tas_models
        else:
            available_models = self.all_models

        # Select models with diversity
        selected_models = []
        remaining_models = available_models.copy()

        for i in range(n_models):
            if not remaining_models:
                break

            if i == 0 or len(selected_models) < self.config.min_models:
                # First model or need to reach minimum
                model = random.choice(remaining_models)
            else:
                # Select model with diversity
                model = self._select_diverse_model(selected_models, remaining_models)

            selected_models.append(model)
            remaining_models.remove(model)

        return selected_models

    def _select_diverse_model(self, selected_models: List[Dict[str, Any]],
                            remaining_models: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Select a model that adds diversity to the ensemble."""
        if not remaining_models:
            return random.choice(remaining_models)

        # Calculate diversity scores
        diversity_scores = []
        for model in remaining_models:
            diversity_score = self._calculate_model_diversity(model, selected_models)
            diversity_scores.append(diversity_score)

        # Weighted selection favoring diversity
        weights = np.array(diversity_scores)
        weights = weights / np.sum(weights) if np.sum(weights) > 0 else np.ones_like(weights) / len(weights)

        selected_idx = np.random.choice(len(remaining_models), p=weights)
        return remaining_models[selected_idx]

    def _calculate_model_diversity(self, model: Dict[str, Any],
                                 selected_models: List[Dict[str, Any]]) -> float:
        """Calculate diversity score for a model relative to selected models."""
        if not selected_models:
            return 1.0

        diversity_scores = []
        for selected_model in selected_models:
            # Calculate diversity based on architecture type
            if model.get('type') != selected_model.get('type'):
                diversity_scores.append(1.0)
            else:
                # Calculate diversity based on architecture features
                diversity_score = self._calculate_architecture_diversity(model, selected_model)
                diversity_scores.append(diversity_score)

        return np.mean(diversity_scores) if diversity_scores else 0.0

    def _calculate_architecture_diversity(self, model1: Dict[str, Any],
                                        model2: Dict[str, Any]) -> float:
        """Calculate diversity between two architectures."""
        try:
            # Extract architecture features
            features1 = self._extract_architecture_features(model1)
            features2 = self._extract_architecture_features(model2)

            # Calculate cosine similarity
            if len(features1) != len(features2):
                return 1.0  # Different feature dimensions = high diversity

            dot_product = np.dot(features1, features2)
            norm1 = np.linalg.norm(features1)
            norm2 = np.linalg.norm(features2)

            if norm1 == 0 or norm2 == 0:
                return 1.0

            similarity = dot_product / (norm1 * norm2)
            diversity = 1.0 - similarity

            return max(0.0, min(1.0, diversity))

        except Exception as e:
            self.logger.warning(f"Architecture diversity calculation failed: {e}")
            return 0.5  # Default moderate diversity

    def _extract_architecture_features(self, model: Dict[str, Any]) -> np.ndarray:
        """Extract numerical features from architecture for diversity calculation."""
        features = []

        # Basic architecture features
        features.append(model.get('n_layers', 0))
        features.append(model.get('n_parameters', 0))
        features.append(model.get('complexity_score', 0.0))

        # Architecture type encoding
        arch_type = model.get('type', 'unknown')
        if arch_type == 'neural':
            features.extend([1, 0, 0])  # Neural encoding
        elif arch_type == 'tree':
            features.extend([0, 1, 0])  # Tree encoding
        else:
            features.extend([0, 0, 1])  # Unknown encoding

        # Performance features
        features.append(model.get('performance_score', 0.0))
        features.append(model.get('training_time', 0.0))

        return np.array(features, dtype=np.float32)

    def _generate_ensemble_weights(self, n_models: int,
                                 ensemble_method: EnsembleMethod) -> List[float]:
        """Generate weights for ensemble based on method."""
        if ensemble_method in [EnsembleMethod.VOTING, EnsembleMethod.BAGGING]:
            # Equal weights
            return [1.0 / n_models] * n_models

        elif ensemble_method in [EnsembleMethod.WEIGHTED_VOTING,
                               EnsembleMethod.ADAPTIVE_WEIGHTING,
                               EnsembleMethod.DYNAMIC_WEIGHTING,
                               EnsembleMethod.UNCERTAINTY_WEIGHTING]:
            # Random weights with constraints
            weights = []
            for _ in range(n_models):
                weight = random.uniform(self.config.min_weight, self.config.max_weight)
                weights.append(weight)

            # Normalize weights
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]

            return weights

        elif ensemble_method == EnsembleMethod.STACKING:
            # Stacking typically uses meta-learner weights
            # For now, use uniform weights for base models
            return [1.0 / n_models] * n_models

        else:
            # Default to uniform weights
            return [1.0 / n_models] * n_models

    def _validate_ensemble(self, ensemble: EnsembleArchitecture) -> Dict[str, Any]:
        """Validate ensemble architecture against constraints."""
        violations = []

        # Check model count constraints
        if len(ensemble.models) < self.config.min_models:
            violations.append(f"Too few models: {len(ensemble.models)} < {self.config.min_models}")

        if len(ensemble.models) > self.config.max_models:
            violations.append(f"Too many models: {len(ensemble.models)} > {self.config.max_models}")

        # Check weight constraints
        if self.config.weight_sum_constraint:
            weight_sum = sum(ensemble.weights)
            if abs(weight_sum - 1.0) > 1e-6:
                violations.append(f"Weights don't sum to 1.0: {weight_sum}")

        for i, weight in enumerate(ensemble.weights):
            if weight < self.config.min_weight:
                violations.append(f"Model {i} weight too low: {weight} < {self.config.min_weight}")
            if weight > self.config.max_weight:
                violations.append(f"Model {i} weight too high: {weight} > {self.config.max_weight}")

        # Check diversity constraints
        ensemble_diversity = self._calculate_ensemble_diversity(ensemble)
        if ensemble_diversity < self.config.min_diversity_threshold:
            violations.append(f"Ensemble diversity too low: {ensemble_diversity} < {self.config.min_diversity_threshold}")

        # Check performance constraints
        if self.performance_estimator:
            try:
                ensemble_score = self._estimate_ensemble_performance(ensemble)
                if ensemble_score < self.config.min_ensemble_performance:
                    violations.append(f"Ensemble performance too low: {ensemble_score} < {self.config.min_ensemble_performance}")
            except Exception as e:
                violations.append(f"Performance estimation failed: {e}")

        return {
            'is_valid': len(violations) == 0,
            'violations': violations,
            'ensemble_diversity': ensemble_diversity
        }

    def _calculate_ensemble_diversity(self, ensemble: EnsembleArchitecture) -> float:
        """Calculate diversity of ensemble models."""
        if len(ensemble.models) < 2:
            return 1.0

        diversity_scores = []
        for i in range(len(ensemble.models)):
            for j in range(i + 1, len(ensemble.models)):
                diversity = self._calculate_architecture_diversity(
                    ensemble.models[i], ensemble.models[j]
                )
                diversity_scores.append(diversity)

        return np.mean(diversity_scores) if diversity_scores else 0.0

    def _estimate_ensemble_performance(self, ensemble: EnsembleArchitecture) -> float:
        """Estimate ensemble performance using performance estimator."""
        if not self.performance_estimator:
            return 0.5  # Default moderate performance

        try:
            # Estimate individual model performances
            individual_scores = []
            for model in ensemble.models:
                prediction = self.performance_estimator.predict_performance(model)
                individual_scores.append(prediction.predicted_performance)

            # Weighted average of individual performances
            weighted_score = sum(score * weight for score, weight in zip(individual_scores, ensemble.weights))

            # Add ensemble bonus (diversity bonus)
            ensemble_diversity = self._calculate_ensemble_diversity(ensemble)
            diversity_bonus = ensemble_diversity * 0.1  # 10% bonus for high diversity

            final_score = weighted_score + diversity_bonus
            return min(1.0, max(0.0, final_score))

        except Exception as e:
            self.logger.warning(f"Ensemble performance estimation failed: {e}")
            return 0.5

    def _create_fallback_ensemble(self) -> EnsembleArchitecture:
        """Create a fallback ensemble when sampling fails."""
        fallback_models = self.all_models[:self.config.min_models] if self.all_models else []

        if not fallback_models:
            # Create dummy models if no models available
            fallback_models = [
                {'type': 'neural', 'n_layers': 3, 'performance_score': 0.5},
                {'type': 'tree', 'n_trees': 50, 'performance_score': 0.5}
            ]

        return EnsembleArchitecture(
            ensemble_id=f"fallback_ensemble_{int(time.time())}",
            models=fallback_models,
            ensemble_method=EnsembleMethod.WEIGHTED_VOTING,
            combination_strategy=EnsembleCombinationStrategy.HYBRID_MIXED,
            weights=[1.0 / len(fallback_models)] * len(fallback_models),
            metadata={'is_fallback': True}
        )

    def search_ensemble_space(self,
                            objective_function: Callable[[EnsembleArchitecture], float],
                            max_iterations: int = 100,
                            search_strategy: str = "random") -> EnsembleSearchResult:
        """Search the ensemble space for optimal ensemble architectures."""
        start_time = time.time()
        self.logger.info(f"🔍 Starting ensemble space search with {search_strategy} strategy")

        best_ensemble = None
        best_score = -np.inf
        search_history = []

        try:
            for iteration in range(max_iterations):
                # Sample ensemble architecture
                ensemble = self.sample_ensemble_architecture()

                # Evaluate ensemble
                try:
                    score = objective_function(ensemble)

                    # Update best ensemble
                    if score > best_score:
                        best_score = score
                        best_ensemble = ensemble

                    # Record search history
                    search_history.append({
                        'iteration': iteration,
                        'ensemble_id': ensemble.ensemble_id,
                        'score': score,
                        'ensemble_method': ensemble.ensemble_method.value,
                        'combination_strategy': ensemble.combination_strategy.value,
                        'n_models': len(ensemble.models),
                        'ensemble_diversity': self._calculate_ensemble_diversity(ensemble)
                    })

                    if iteration % 10 == 0:
                        self.logger.info(f"   Iteration {iteration}: Best Score = {best_score:.4f}")

                except Exception as e:
                    self.logger.warning(f"Ensemble evaluation failed at iteration {iteration}: {e}")
                    continue

            execution_time = time.time() - start_time

            # Calculate ensemble metrics
            individual_scores = []
            if best_ensemble and self.performance_estimator:
                for model in best_ensemble.models:
                    try:
                        prediction = self.performance_estimator.predict_performance(model)
                        individual_scores.append(prediction.predicted_performance)
                    except:
                        individual_scores.append(0.5)

            ensemble_diversity = self._calculate_ensemble_diversity(best_ensemble) if best_ensemble else 0.0
            ensemble_robustness = self._calculate_ensemble_robustness(best_ensemble) if best_ensemble else 0.0

            result = EnsembleSearchResult(
                best_ensemble=best_ensemble,
                ensemble_score=best_score,
                individual_scores=individual_scores,
                ensemble_diversity=ensemble_diversity,
                ensemble_robustness=ensemble_robustness,
                search_history=search_history,
                execution_time=execution_time,
                metadata={
                    'search_strategy': search_strategy,
                    'max_iterations': max_iterations,
                    'total_models_available': len(self.all_models)
                }
            )

            self.logger.info(f"✅ Ensemble search completed in {execution_time:.2f}s")
            self.logger.info(f"   Best Score: {best_score:.4f}")
            self.logger.info(f"   Ensemble Diversity: {ensemble_diversity:.4f}")

            return result

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ Ensemble search failed: {e}")

            return EnsembleSearchResult(
                best_ensemble=self._create_fallback_ensemble(),
                ensemble_score=0.0,
                individual_scores=[],
                ensemble_diversity=0.0,
                ensemble_robustness=0.0,
                search_history=search_history,
                execution_time=execution_time,
                metadata={'error': str(e)}
            )

    def _calculate_ensemble_robustness(self, ensemble: EnsembleArchitecture) -> float:
        """Calculate robustness of ensemble architecture."""
        if not ensemble or len(ensemble.models) < 2:
            return 0.0

        try:
            # Robustness based on model diversity and weight distribution
            diversity = self._calculate_ensemble_diversity(ensemble)

            # Weight distribution entropy (higher entropy = more robust)
            weights = np.array(ensemble.weights)
            weights = weights / np.sum(weights)  # Normalize
            entropy = -np.sum(weights * np.log(weights + 1e-10))
            max_entropy = np.log(len(weights))
            weight_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

            # Combined robustness score
            robustness = (diversity + weight_entropy) / 2.0

            return min(1.0, max(0.0, robustness))

        except Exception as e:
            self.logger.warning(f"Ensemble robustness calculation failed: {e}")
            return 0.5

    def get_search_space_info(self) -> Dict[str, Any]:
        """Get information about the ensemble search space."""
        return {
            'total_models': len(self.all_models),
            'nas_models': len(self.nas_models),
            'tas_models': len(self.tas_models),
            'ensemble_methods': [method.value for method in self.config.allowed_ensemble_methods],
            'combination_strategies': [strategy.value for strategy in self.config.allowed_combination_strategies],
            'min_models': self.config.min_models,
            'max_models': self.config.max_models,
            'search_history_length': len(self.search_history),
            'ensemble_cache_size': len(self.ensemble_cache)
        }

def create_unified_ensemble_search_space(
    nas_models: List[Dict[str, Any]] = None,
    tas_models: List[Dict[str, Any]] = None,
    config: EnsembleSearchSpaceConfig = None,
    performance_estimator: UnifiedPerformanceEstimator = None,
    constraint_validator: UnifiedConstraintValidator = None
) -> UnifiedEnsembleSearchSpace:
    """Create a unified ensemble search space instance."""
    if config is None:
        config = EnsembleSearchSpaceConfig()

    return UnifiedEnsembleSearchSpace(
        config=config,
        nas_models=nas_models,
        tas_models=tas_models,
        performance_estimator=performance_estimator,
        constraint_validator=constraint_validator
    )

def quick_ensemble_search(
    nas_models: List[Dict[str, Any]],
    tas_models: List[Dict[str, Any]],
    objective_function: Callable[[EnsembleArchitecture], float],
    max_iterations: int = 50
) -> EnsembleSearchResult:
    """Quick ensemble search with default settings."""
    config = EnsembleSearchSpaceConfig(
        max_models=5,
        allowed_ensemble_methods=[EnsembleMethod.WEIGHTED_VOTING, EnsembleMethod.ADAPTIVE_WEIGHTING],
        allowed_combination_strategies=[EnsembleCombinationStrategy.HYBRID_MIXED]
    )

    search_space = create_unified_ensemble_search_space(
        nas_models=nas_models,
        tas_models=tas_models,
        config=config
    )

    return search_space.search_ensemble_space(objective_function, max_iterations)
