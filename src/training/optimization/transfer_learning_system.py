#!/usr/bin/env python3
"""
Transfer Learning System for Surrogate Optimization

This module provides transfer learning capabilities for surrogate optimization:
- Knowledge transfer between similar problems
- Pre-trained model adaptation
- Meta-learning for optimization
- Problem similarity detection
- Warm-start strategies
"""

import numpy as np
import time
import json
import os
import pickle
import hashlib

# ML libraries
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.ensemble import RandomForestRegressor

# Utilities
from src.utils.logger import system_logger


@dataclass
class ProblemSignature:
    """Signature of an optimization problem for similarity detection."""
    problem_id: str
    dimensionality: int
    parameter_bounds: List[Tuple[float, float]]
    objective_type: str  # "minimization", "maximization"
    constraint_count: int
    noise_level: float
    complexity_score: float
    feature_vector: np.ndarray
    metadata: Dict[str, Any]


@dataclass
class TransferKnowledge:
    """Knowledge transferred from previous optimization problems."""
    source_problem_id: str
    target_problem_id: str
    similarity_score: float
    transferred_models: Dict[str, Any]
    transferred_hyperparameters: Dict[str, Any]
    transferred_strategies: Dict[str, Any]
    adaptation_weights: Dict[str, float]
    transfer_timestamp: float
    transfer_effectiveness: float


@dataclass
class OptimizationHistory:
    """Complete history of an optimization problem."""
    problem_id: str
    problem_signature: ProblemSignature
    parameter_space: Dict[str, Any]
    objective_function: str  # Function signature/hash
    optimization_results: Dict[str, Any]
    surrogate_models: Dict[str, Any]
    best_parameters: Dict[str, Any]
    best_score: float
    convergence_history: List[float]
    training_time: float
    completion_timestamp: float


class ProblemSimilarityDetector:
    """Detects similarity between optimization problems."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild("ProblemSimilarityDetector")

    def calculate_similarity(
        self,
        problem1: ProblemSignature,
        problem2: ProblemSignature
    ) -> float:
        """Calculate similarity between two problems."""

        # Feature-based similarity
        feature_similarity = self._calculate_feature_similarity(
            problem1.feature_vector, problem2.feature_vector
        )

        # Structural similarity
        structural_similarity = self._calculate_structural_similarity(problem1, problem2)

        # Domain similarity
        domain_similarity = self._calculate_domain_similarity(problem1, problem2)

        # Weighted combination
        weights = self.config.get('similarity_weights', {
            'feature': 0.4,
            'structural': 0.4,
            'domain': 0.2
        })

        total_similarity = (
            weights['feature'] * feature_similarity +
            weights['structural'] * structural_similarity +
            weights['domain'] * domain_similarity
        )

        return total_similarity

    def _calculate_feature_similarity(
        self,
        features1: np.ndarray,
        features2: np.ndarray
    ) -> float:
        """Calculate similarity based on feature vectors."""
        try:
            # Cosine similarity
            cosine_sim = cosine_similarity(
                features1.reshape(1, -1),
                features2.reshape(1, -1)
            )[0, 0]

            # Euclidean distance (normalized)
            euclidean_dist = euclidean_distances(
                features1.reshape(1, -1),
                features2.reshape(1, -1)
            )[0, 0]

            # Normalize Euclidean distance
            max_possible_dist = np.sqrt(len(features1))
            normalized_euclidean = 1.0 - (euclidean_dist / max_possible_dist)

            # Combine similarities
            return 0.7 * cosine_sim + 0.3 * normalized_euclidean

        except Exception as e:
            self.logger.warning(f"Error calculating feature similarity: {e}")
            return 0.0

    def _calculate_structural_similarity(
        self,
        problem1: ProblemSignature,
        problem2: ProblemSignature
    ) -> float:
        """Calculate structural similarity between problems."""
        similarities = []

        # Dimensionality similarity
        dim_similarity = 1.0 - abs(problem1.dimensionality - problem2.dimensionality) / max(
            problem1.dimensionality, problem2.dimensionality, 1
        )
        similarities.append(dim_similarity)

        # Constraint similarity
        constraint_similarity = 1.0 - abs(problem1.constraint_count - problem2.constraint_count) / max(
            problem1.constraint_count, problem2.constraint_count, 1
        )
        similarities.append(constraint_similarity)

        # Complexity similarity
        complexity_similarity = 1.0 - abs(problem1.complexity_score - problem2.complexity_score)
        similarities.append(complexity_similarity)

        # Objective type similarity
        objective_similarity = 1.0 if problem1.objective_type == problem2.objective_type else 0.0
        similarities.append(objective_similarity)

        return np.mean(similarities)

    def _calculate_domain_similarity(
        self,
        problem1: ProblemSignature,
        problem2: ProblemSignature
    ) -> float:
        """Calculate domain similarity between problems."""
        # Extract domain information from metadata
        domain1 = problem1.metadata.get('domain', 'unknown')
        domain2 = problem2.metadata.get('domain', 'unknown')

        if domain1 == domain2:
            return 1.0
        elif domain1 in ['unknown', 'general'] or domain2 in ['unknown', 'general']:
            return 0.5
        else:
            return 0.0


class KnowledgeTransferManager:
    """Manages knowledge transfer between optimization problems."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild("KnowledgeTransferManager")

        # Storage for optimization history
        self.optimization_history: List[OptimizationHistory] = []
        self.transfer_knowledge: List[TransferKnowledge] = []

        # Similarity detector
        self.similarity_detector = ProblemSimilarityDetector(config)

        # Storage paths
        self.history_file = config.get('history_file', 'optimization_history.pkl')
        self.transfer_file = config.get('transfer_file', 'transfer_knowledge.pkl')

        # Load existing history
        self._load_history()

    def _load_history(self) -> None:
        """Load optimization history from disk."""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'rb') as f:
                    self.optimization_history = pickle.load(f)
                self.logger.info(f"Loaded {len(self.optimization_history)} optimization histories")

            if os.path.exists(self.transfer_file):
                with open(self.transfer_file, 'rb') as f:
                    self.transfer_knowledge = pickle.load(f)
                self.logger.info(f"Loaded {len(self.transfer_knowledge)} transfer knowledge records")

        except Exception as e:
            self.logger.warning(f"Error loading history: {e}")

    def _save_history(self) -> None:
        """Save optimization history to disk."""
        try:
            with open(self.history_file, 'wb') as f:
                pickle.dump(self.optimization_history, f)

            with open(self.transfer_file, 'wb') as f:
                pickle.dump(self.transfer_knowledge, f)

        except Exception as e:
            self.logger.error(f"Error saving history: {e}")

    def add_optimization_history(self, history: OptimizationHistory) -> None:
        """Add optimization history to the knowledge base."""
        self.optimization_history.append(history)
        self._save_history()
        self.logger.info(f"Added optimization history for problem {history.problem_id}")

    def find_similar_problems(
        self,
        target_problem: ProblemSignature,
        similarity_threshold: float = 0.7,
        max_results: int = 5
    ) -> List[Tuple[OptimizationHistory, float]]:
        """Find problems similar to the target problem."""
        similarities = []

        for history in self.optimization_history:
            similarity = self.similarity_detector.calculate_similarity(
                target_problem, history.problem_signature
            )

            if similarity >= similarity_threshold:
                similarities.append((history, similarity))

        # Sort by similarity and return top results
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:max_results]

    def transfer_knowledge(
        self,
        target_problem: ProblemSignature,
        source_problems: List[OptimizationHistory],
        similarity_scores: List[float]
    ) -> TransferKnowledge:
        """Transfer knowledge from source problems to target problem."""

        # Combine knowledge from multiple source problems
        transferred_models = {}
        transferred_hyperparameters = {}
        transferred_strategies = {}
        adaptation_weights = {}

        total_weight = sum(similarity_scores)

        for i, (source_problem, similarity) in enumerate(zip(source_problems, similarity_scores)):
            weight = similarity / total_weight

            # Transfer surrogate models
            for model_name, model in source_problem.surrogate_models.items():
                if model_name not in transferred_models:
                    transferred_models[model_name] = []
                transferred_models[model_name].append((model, weight))

            # Transfer hyperparameters
            for param_name, param_value in source_problem.optimization_results.get('hyperparameters', {}).items():
                if param_name not in transferred_hyperparameters:
                    transferred_hyperparameters[param_name] = []
                transferred_hyperparameters[param_name].append((param_value, weight))

            # Transfer strategies
            for strategy_name, strategy_value in source_problem.optimization_results.get('strategies', {}).items():
                if strategy_name not in transferred_strategies:
                    transferred_strategies[strategy_name] = []
                transferred_strategies[strategy_name].append((strategy_value, weight))

            # Store adaptation weights
            adaptation_weights[f"source_{i}"] = weight

        # Create transfer knowledge record
        transfer_knowledge = TransferKnowledge(
            source_problem_id=",".join([p.problem_id for p in source_problems]),
            target_problem_id=target_problem.problem_id,
            similarity_score=np.mean(similarity_scores),
            transferred_models=transferred_models,
            transferred_hyperparameters=transferred_hyperparameters,
            transferred_strategies=transferred_strategies,
            adaptation_weights=adaptation_weights,
            transfer_timestamp=time.time(),
            transfer_effectiveness=0.0  # Will be updated after optimization
        )

        self.transfer_knowledge.append(transfer_knowledge)
        self._save_history()

        return transfer_knowledge

    def adapt_transferred_knowledge(
        self,
        transfer_knowledge: TransferKnowledge,
        target_parameter_space: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Adapt transferred knowledge to the target problem."""

        adapted_knowledge = {}

        # Adapt hyperparameters
        adapted_hyperparameters = {}
        for param_name, weighted_values in transfer_knowledge.transferred_hyperparameters.items():
            # Weighted average of hyperparameters
            adapted_value = sum(value * weight for value, weight in weighted_values)
            adapted_hyperparameters[param_name] = adapted_value

        adapted_knowledge['hyperparameters'] = adapted_hyperparameters

        # Adapt strategies
        adapted_strategies = {}
        for strategy_name, weighted_values in transfer_knowledge.transferred_strategies.items():
            # For categorical strategies, use weighted voting
            if isinstance(weighted_values[0][0], str):
                # Count weighted votes
                votes = {}
                for value, weight in weighted_values:
                    votes[value] = votes.get(value, 0) + weight

                # Select strategy with highest weighted vote
                adapted_strategies[strategy_name] = max(votes.items(), key=lambda x: x[1])[0]
            else:
                # For numerical strategies, use weighted average
                adapted_value = sum(value * weight for value, weight in weighted_values)
                adapted_strategies[strategy_name] = adapted_value

        adapted_knowledge['strategies'] = adapted_strategies

        # Adapt surrogate models
        adapted_models = {}
        for model_name, weighted_models in transfer_knowledge.transferred_models.items():
            # For now, use the model with highest weight
            best_model, best_weight = max(weighted_models, key=lambda x: x[1])
            adapted_models[model_name] = best_model

        adapted_knowledge['models'] = adapted_models

        return adapted_knowledge

    def update_transfer_effectiveness(
        self,
        transfer_knowledge: TransferKnowledge,
        optimization_performance: Dict[str, Any]
    ) -> None:
        """Update the effectiveness of a knowledge transfer."""

        # Calculate effectiveness based on performance improvement
        baseline_performance = optimization_performance.get('baseline_performance', 0.0)
        transfer_performance = optimization_performance.get('transfer_performance', 0.0)

        if baseline_performance > 0:
            effectiveness = (transfer_performance - baseline_performance) / baseline_performance
        else:
            effectiveness = 0.0

        # Update the transfer knowledge
        transfer_knowledge.transfer_effectiveness = max(0.0, min(1.0, effectiveness))

        self._save_history()
        self.logger.info(f"Updated transfer effectiveness: {effectiveness:.3f}")


class MetaLearner:
    """Meta-learning system for optimization strategy selection."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild("MetaLearner")

        # Meta-models for different aspects
        self.strategy_selector = RandomForestRegressor(n_estimators=100, random_state=42)
        self.hyperparameter_predictor = RandomForestRegressor(n_estimators=100, random_state=42)
        self.performance_predictor = RandomForestRegressor(n_estimators=100, random_state=42)

        # Training data
        self.training_data = []
        self.is_trained = False

    def extract_problem_features(self, problem_signature: ProblemSignature) -> np.ndarray:
        """Extract features for meta-learning."""
        features = [
            problem_signature.dimensionality,
            problem_signature.constraint_count,
            problem_signature.noise_level,
            problem_signature.complexity_score,
            len(problem_signature.parameter_bounds),
            # Add more features as needed
        ]

        # Add feature vector
        features.extend(problem_signature.feature_vector)

        return np.array(features)

    def add_training_example(
        self,
        problem_signature: ProblemSignature,
        strategy_used: str,
        hyperparameters: Dict[str, Any],
        performance: float
    ) -> None:
        """Add a training example for meta-learning."""
        features = self.extract_problem_features(problem_signature)

        training_example = {
            'features': features,
            'strategy': strategy_used,
            'hyperparameters': hyperparameters,
            'performance': performance
        }

        self.training_data.append(training_example)

    def train_meta_models(self) -> None:
        """Train meta-learning models."""
        if len(self.training_data) < 10:
            self.logger.warning("Insufficient training data for meta-learning")
            return

        # Prepare training data
        X = np.array([example['features'] for example in self.training_data])

        # Strategy labels (convert to numerical)
        strategy_labels = [example['strategy'] for example in self.training_data]
        unique_strategies = list(set(strategy_labels))
        strategy_mapping = {strategy: i for i, strategy in enumerate(unique_strategies)}
        y_strategy = np.array([strategy_mapping[strategy] for strategy in strategy_labels])

        # Hyperparameter targets (use key hyperparameters)
        key_hyperparams = ['learning_rate', 'exploration_balance', 'uncertainty_threshold']
        y_hyperparams = []

        for example in self.training_data:
            hyperparam_vector = []
            for param in key_hyperparams:
                value = example['hyperparameters'].get(param, 0.0)
                hyperparam_vector.append(value)
            y_hyperparams.append(hyperparam_vector)

        y_hyperparams = np.array(y_hyperparams)

        # Performance targets
        y_performance = np.array([example['performance'] for example in self.training_data])

        # Train models
        try:
            self.strategy_selector.fit(X, y_strategy)
            self.hyperparameter_predictor.fit(X, y_hyperparams)
            self.performance_predictor.fit(X, y_performance)

            self.is_trained = True
            self.logger.info("Meta-learning models trained successfully")

        except Exception as e:
            self.logger.error(f"Error training meta-models: {e}")

    def predict_optimal_strategy(
        self,
        problem_signature: ProblemSignature
    ) -> Tuple[str, Dict[str, Any], float]:
        """Predict optimal strategy and hyperparameters for a problem."""
        if not self.is_trained:
            return "default", {}, 0.0

        features = self.extract_problem_features(problem_signature)
        features = features.reshape(1, -1)

        # Predict strategy
        strategy_idx = self.strategy_selector.predict(features)[0]
        strategy = list(set([example['strategy'] for example in self.training_data]))[strategy_idx]

        # Predict hyperparameters
        hyperparam_vector = self.hyperparameter_predictor.predict(features)[0]
        key_hyperparams = ['learning_rate', 'exploration_balance', 'uncertainty_threshold']
        hyperparameters = dict(zip(key_hyperparams, hyperparam_vector))

        # Predict expected performance
        expected_performance = self.performance_predictor.predict(features)[0]

        return strategy, hyperparameters, expected_performance


class TransferLearningOptimizer:
    """Main transfer learning optimizer that combines all components."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild("TransferLearningOptimizer")

        # Initialize components
        self.knowledge_manager = KnowledgeTransferManager(config)
        self.meta_learner = MetaLearner(config)

        # Transfer learning settings
        self.enable_transfer = config.get('enable_transfer_learning', True)
        self.similarity_threshold = config.get('similarity_threshold', 0.7)
        self.max_source_problems = config.get('max_source_problems', 3)

    def optimize_with_transfer(
        self,
        objective_function: Callable,
        parameter_space: Dict[str, Any],
        problem_metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Optimize with transfer learning capabilities."""

        # Create problem signature
        problem_signature = self._create_problem_signature(
            objective_function, parameter_space, problem_metadata
        )

        # Find similar problems
        similar_problems = []
        if self.enable_transfer:
            similar_problems_with_scores = self.knowledge_manager.find_similar_problems(
                problem_signature,
                self.similarity_threshold,
                self.max_source_problems
            )

            if similar_problems_with_scores:
                similar_problems, similarity_scores = zip(*similar_problems_with_scores)
                similar_problems = list(similar_problems)
                similarity_scores = list(similarity_scores)

                self.logger.info(f"Found {len(similar_problems)} similar problems")

                # Transfer knowledge
                transfer_knowledge = self.knowledge_manager.transfer_knowledge(
                    problem_signature, similar_problems, similarity_scores
                )

                # Adapt transferred knowledge
                adapted_knowledge = self.knowledge_manager.adapt_transferred_knowledge(
                    transfer_knowledge, parameter_space
                )

                # Use transferred knowledge for warm start
                optimization_config = self._create_optimization_config_with_transfer(
                    adapted_knowledge, problem_signature
                )
            else:
                self.logger.info("No similar problems found, using default configuration")
                optimization_config = self._create_default_optimization_config(problem_signature)
        else:
            optimization_config = self._create_default_optimization_config(problem_signature)

        # Run optimization
        optimization_results = self._run_optimization(
            objective_function, parameter_space, optimization_config
        )

        # Update transfer effectiveness if transfer was used
        if similar_problems and self.enable_transfer:
            self.knowledge_manager.update_transfer_effectiveness(
                transfer_knowledge, optimization_results
            )

        # Add to training data for meta-learning
        self.meta_learner.add_training_example(
            problem_signature,
            optimization_config.get('strategy', 'default'),
            optimization_config.get('hyperparameters', {}),
            optimization_results.get('best_score', 0.0)
        )

        # Save optimization history
        optimization_history = OptimizationHistory(
            problem_id=problem_signature.problem_id,
            problem_signature=problem_signature,
            parameter_space=parameter_space,
            objective_function=self._hash_function(objective_function),
            optimization_results=optimization_results,
            surrogate_models=optimization_results.get('surrogate_models', {}),
            best_parameters=optimization_results.get('best_parameters', {}),
            best_score=optimization_results.get('best_score', 0.0),
            convergence_history=optimization_results.get('convergence_history', []),
            training_time=optimization_results.get('training_time', 0.0),
            completion_timestamp=time.time()
        )

        self.knowledge_manager.add_optimization_history(optimization_history)

        return optimization_results

    def _create_problem_signature(
        self,
        objective_function: Callable,
        parameter_space: Dict[str, Any],
        problem_metadata: Dict[str, Any] = None
    ) -> ProblemSignature:
        """Create a signature for the optimization problem."""

        # Generate problem ID
        problem_id = self._generate_problem_id(objective_function, parameter_space)

        # Extract basic characteristics
        dimensionality = len(parameter_space)
        parameter_bounds = self._extract_bounds(parameter_space)
        constraint_count = self._count_constraints(parameter_space)

        # Create feature vector
        feature_vector = self._create_feature_vector(parameter_space)

        # Estimate complexity
        complexity_score = self._estimate_complexity(dimensionality, constraint_count)

        # Create signature
        return ProblemSignature(
            problem_id=problem_id,
            dimensionality=dimensionality,
            parameter_bounds=parameter_bounds,
            objective_type="minimization",  # Default, could be detected
            constraint_count=constraint_count,
            noise_level=0.0,  # Would need to be estimated
            complexity_score=complexity_score,
            feature_vector=feature_vector,
            metadata=problem_metadata or {}
        )

    def _generate_problem_id(
        self,
        objective_function: Callable,
        parameter_space: Dict[str, Any]
    ) -> str:
        """Generate a unique ID for the problem."""
        # Create a hash of the function and parameter space
        function_str = str(objective_function.__name__)
        param_str = json.dumps(parameter_space, sort_keys=True)

        combined_str = function_str + param_str
        return hashlib.md5(combined_str.encode()).hexdigest()[:8]

    def _extract_bounds(self, parameter_space: Dict[str, Any]) -> List[Tuple[float, float]]:
        """Extract parameter bounds."""
        bounds = []
        for param_config in parameter_space.values():
            if isinstance(param_config, dict):
                if 'min' in param_config and 'max' in param_config:
                    bounds.append((param_config['min'], param_config['max']))
            elif isinstance(param_config, (list, tuple)) and len(param_config) == 2:
                bounds.append(tuple(param_config))
        return bounds

    def _count_constraints(self, parameter_space: Dict[str, Any]) -> int:
        """Count the number of constraints in the parameter space."""
        constraint_count = 0
        for param_name, param_config in parameter_space.items():
            if isinstance(param_config, dict):
                if 'constraints' in param_config:
                    constraint_count += len(param_config['constraints'])
        return constraint_count

    def _create_feature_vector(self, parameter_space: Dict[str, Any]) -> np.ndarray:
        """Create a feature vector representing the parameter space."""
        features = []

        for param_name, param_config in parameter_space.items():
            if isinstance(param_config, dict):
                if 'min' in param_config and 'max' in param_config:
                    # Continuous parameter
                    features.extend([
                        param_config['min'],
                        param_config['max'],
                        param_config['max'] - param_config['min']
                    ])
                elif 'choices' in param_config:
                    # Discrete parameter
                    features.extend([
                        len(param_config['choices']),
                        min(param_config['choices']),
                        max(param_config['choices'])
                    ])
            elif isinstance(param_config, (list, tuple)) and len(param_config) == 2:
                # Simple bounds
                features.extend([
                    param_config[0],
                    param_config[1],
                    param_config[1] - param_config[0]
                ])

        return np.array(features)

    def _estimate_complexity(self, dimensionality: int, constraint_count: int) -> float:
        """Estimate problem complexity."""
        complexity = 0.0

        # Dimensionality contribution
        complexity += min(dimensionality / 10.0, 1.0) * 0.6

        # Constraint contribution
        complexity += min(constraint_count / 5.0, 1.0) * 0.4

        return complexity

    def _hash_function(self, func: Callable) -> str:
        """Create a hash of the function."""
        return hashlib.md5(str(func.__name__).encode()).hexdigest()[:8]

    def _create_optimization_config_with_transfer(
        self,
        adapted_knowledge: Dict[str, Any],
        problem_signature: ProblemSignature
    ) -> Dict[str, Any]:
        """Create optimization configuration using transferred knowledge."""
        config = {
            'strategy': adapted_knowledge.get('strategies', {}).get('strategy', 'default'),
            'hyperparameters': adapted_knowledge.get('hyperparameters', {}),
            'surrogate_models': adapted_knowledge.get('models', {}),
            'warm_start': True,
            'transfer_learning': True
        }

        return config

    def _create_default_optimization_config(
        self,
        problem_signature: ProblemSignature
    ) -> Dict[str, Any]:
        """Create default optimization configuration."""
        config = {
            'strategy': 'default',
            'hyperparameters': {
                'learning_rate': 0.1,
                'exploration_balance': 0.3,
                'uncertainty_threshold': 0.1
            },
            'warm_start': False,
            'transfer_learning': False
        }

        return config

    def _run_optimization(
        self,
        objective_function: Callable,
        parameter_space: Dict[str, Any],
        optimization_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run the actual optimization."""
        # This would integrate with your existing surrogate optimizer
        # For now, return a mock result
        return {
            'best_parameters': {},
            'best_score': 0.0,
            'convergence_history': [],
            'training_time': 0.0,
            'surrogate_models': {},
            'strategy_used': optimization_config.get('strategy', 'default'),
            'hyperparameters_used': optimization_config.get('hyperparameters', {})
        }