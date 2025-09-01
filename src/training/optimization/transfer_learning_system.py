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
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.ensemble import RandomForestRegressor
from dataclasses import dataclass
from src.utils.logger import system_logger
from src.utils.error_handler import handle_errors


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

    def calculate_similarity(self, problem1: ProblemSignature, problem2: ProblemSignature) -> float:
        """Calculate overall similarity between two optimization problems."""
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

    def _calculate_feature_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Calculate similarity based on feature vectors."""
        try:
            # Ensure vectors have same shape
            if features1.shape != features2.shape:
                # Pad or truncate to match
                min_length = min(len(features1), len(features2))
                features1 = features1[:min_length]
                features2 = features2[:min_length]
            
            # Use cosine similarity for feature vectors
            similarity = cosine_similarity(features1.reshape(1, -1), features2.reshape(1, -1))[0, 0]
            return max(0.0, similarity)  # Ensure non-negative
        except Exception as e:
            self.logger.warning(f"Error calculating feature similarity: {e}")
            return 0.0

    def _calculate_structural_similarity(self, problem1: ProblemSignature, problem2: ProblemSignature) -> float:
        """Calculate similarity based on structural properties."""
        similarities = []
        
        # Dimensionality similarity
        dim_similarity = 1.0 - abs(problem1.dimensionality - problem2.dimensionality) / max(problem1.dimensionality, problem2.dimensionality, 1)
        similarities.append(dim_similarity)
        
        # Constraint count similarity
        constraint_similarity = 1.0 - abs(problem1.constraint_count - problem2.constraint_count) / max(problem1.constraint_count, problem2.constraint_count, 1)
        similarities.append(constraint_similarity)
        
        # Objective type similarity
        objective_similarity = 1.0 if problem1.objective_type == problem2.objective_type else 0.0
        similarities.append(objective_similarity)
        
        # Complexity similarity
        complexity_diff = abs(problem1.complexity_score - problem2.complexity_score)
        complexity_similarity = 1.0 / (1.0 + complexity_diff)
        similarities.append(complexity_similarity)
        
        return np.mean(similarities)

    def _calculate_domain_similarity(self, problem1: ProblemSignature, problem2: ProblemSignature) -> float:
        """Calculate similarity based on domain information."""
        # Extract domain information from metadata
        domain1 = problem1.metadata.get('domain', 'unknown')
        domain2 = problem2.metadata.get('domain', 'unknown')
        
        if domain1 == domain2:
            return 1.0
        elif domain1 == 'unknown' or domain2 == 'unknown':
            return 0.5  # Neutral similarity for unknown domains
        else:
            return 0.0  # Different domains


class KnowledgeTransferManager:
    """Manages knowledge transfer between optimization problems."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild("KnowledgeTransferManager")
        self.history_file = config.get('history_file', 'optimization_history.pkl')
        self.optimization_history: List[OptimizationHistory] = []
        
        # Load existing history
        self._load_history()

    def _load_history(self) -> None:
        """Load optimization history from file."""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'rb') as f:
                    self.optimization_history = pickle.load(f)
                self.logger.info(f"Loaded {len(self.optimization_history)} optimization histories")
        except Exception as e:
            self.logger.warning(f"Error loading history: {e}")
            self.optimization_history = []

    def _save_history(self) -> None:
        """Save optimization history to file."""
        try:
            with open(self.history_file, 'wb') as f:
                pickle.dump(self.optimization_history, f)
        except Exception as e:
            self.logger.error(f"Error saving history: {e}")

    def add_optimization_history(self, history: OptimizationHistory) -> None:
        """Add a new optimization history."""
        self.optimization_history.append(history)
        self._save_history()
        self.logger.info(f"Added optimization history for problem {history.problem_id}")

    def find_similar_problems(self, target_signature: ProblemSignature, max_results: int = 5) -> List[Tuple[str, float]]:
        """Find problems similar to the target problem."""
        similarities = []
        detector = ProblemSimilarityDetector(self.config)
        
        for history in self.optimization_history:
            similarity = detector.calculate_similarity(target_signature, history.problem_signature)
            similarities.append((history.problem_id, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:max_results]

    def transfer_knowledge(self, target_signature: ProblemSignature, source_problem_ids: List[str]) -> TransferKnowledge:
        """Transfer knowledge from source problems to target problem."""
        # Combine knowledge from multiple source problems
        transferred_models = {}
        transferred_hyperparameters = {}
        transferred_strategies = {}
        adaptation_weights = {}
        
        total_similarity = 0.0
        
        for source_id in source_problem_ids:
            # Find the source history
            source_history = next((h for h in self.optimization_history if h.problem_id == source_id), None)
            if source_history is None:
                continue
            
            # Calculate similarity
            detector = ProblemSimilarityDetector(self.config)
            similarity = detector.calculate_similarity(target_signature, source_history.problem_signature)
            
            # Weight the transferred knowledge by similarity
            weight = similarity
            total_similarity += weight
            
            # Aggregate models, hyperparameters, and strategies
            for model_name, model in source_history.surrogate_models.items():
                if model_name not in transferred_models:
                    transferred_models[model_name] = []
                transferred_models[model_name].append((model, weight))
            
            for param_name, param_value in source_history.optimization_results.get('hyperparameters', {}).items():
                if param_name not in transferred_hyperparameters:
                    transferred_hyperparameters[param_name] = []
                transferred_hyperparameters[param_name].append((param_value, weight))
            
            for strategy_name, strategy_config in source_history.optimization_results.get('strategies', {}).items():
                if strategy_name not in transferred_strategies:
                    transferred_strategies[strategy_name] = []
                transferred_strategies[strategy_name].append((strategy_config, weight))
        
        # Normalize weights
        if total_similarity > 0:
            for model_name in transferred_models:
                transferred_models[model_name] = [(model, weight / total_similarity) 
                                                for model, weight in transferred_models[model_name]]
            for param_name in transferred_hyperparameters:
                transferred_hyperparameters[param_name] = [(value, weight / total_similarity) 
                                                         for value, weight in transferred_hyperparameters[param_name]]
            for strategy_name in transferred_strategies:
                transferred_strategies[strategy_name] = [(config, weight / total_similarity) 
                                                       for config, weight in transferred_strategies[strategy_name]]
        
        return TransferKnowledge(
            source_problem_id=",".join(source_problem_ids),
            target_problem_id=target_signature.problem_id,
            similarity_score=total_similarity / len(source_problem_ids) if source_problem_ids else 0.0,
            transferred_models=transferred_models,
            transferred_hyperparameters=transferred_hyperparameters,
            transferred_strategies=transferred_strategies,
            adaptation_weights=adaptation_weights,
            transfer_timestamp=time.time(),
            transfer_effectiveness=0.0  # Will be updated after optimization
        )

    def adapt_transferred_knowledge(self, transferred_knowledge: TransferKnowledge, 
                                  target_problem: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt transferred knowledge to the target problem."""
        adapted_knowledge = {}
        
        # Adapt hyperparameters
        adapted_hyperparameters = {}
        for param_name, weighted_values in transferred_knowledge.transferred_hyperparameters.items():
            if weighted_values:
                # Weighted average of hyperparameter values
                total_weight = sum(weight for _, weight in weighted_values)
                if total_weight > 0:
                    adapted_value = sum(value * weight for value, weight in weighted_values) / total_weight
                    adapted_hyperparameters[param_name] = adapted_value
        
        adapted_knowledge['hyperparameters'] = adapted_hyperparameters
        
        # Adapt strategies
        adapted_strategies = {}
        for strategy_name, weighted_configs in transferred_knowledge.transferred_strategies.items():
            if weighted_configs:
                # Select the most similar strategy configuration
                best_config, best_weight = max(weighted_configs, key=lambda x: x[1])
                adapted_strategies[strategy_name] = best_config
        
        adapted_knowledge['strategies'] = adapted_strategies
        
        # Adapt models (simplified - in practice would need more sophisticated adaptation)
        adapted_knowledge['models'] = transferred_knowledge.transferred_models
        
        return adapted_knowledge

    def update_transfer_effectiveness(self, transfer_knowledge: TransferKnowledge, 
                                    optimization_performance: Dict[str, Any]) -> None:
        """Update the effectiveness of a knowledge transfer."""
        # Calculate effectiveness based on performance improvement
        baseline_performance = optimization_performance.get('baseline_performance', 0.0)
        transfer_performance = optimization_performance.get('transfer_performance', 0.0)
        
        if baseline_performance > 0:
            improvement = (transfer_performance - baseline_performance) / baseline_performance
            transfer_knowledge.transfer_effectiveness = max(0.0, improvement)
        else:
            transfer_knowledge.transfer_effectiveness = 0.0


class MetaLearner:
    """Meta-learning system for optimization strategy selection."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild("MetaLearner")
        self.training_data = []
        self.meta_models = {}
        self.is_trained = False

    def extract_problem_features(self, problem_signature: ProblemSignature) -> np.ndarray:
        """Extract features from problem signature for meta-learning."""
        features = [
            problem_signature.dimensionality,
            problem_signature.constraint_count,
            problem_signature.noise_level,
            problem_signature.complexity_score,
            len(problem_signature.parameter_bounds),
            # Add more features as needed
        ]
        return np.array(features)

    def add_training_example(self, problem_signature: ProblemSignature, 
                           strategy_used: str, performance: float) -> None:
        """Add a training example for meta-learning."""
        features = self.extract_problem_features(problem_signature)
        training_example = {
            'features': features,
            'strategy': strategy_used,
            'performance': performance
        }
        self.training_data.append(training_example)

    def train_meta_models(self) -> None:
        """Train meta-models for strategy prediction."""
        if len(self.training_data) < 10:
            self.logger.warning("Insufficient training data for meta-learning")
            return
        
        try:
            # Prepare training data
            X = np.array([example['features'] for example in self.training_data])
            y_strategy = [example['strategy'] for example in self.training_data]
            y_performance = [example['performance'] for example in self.training_data]
            
            # Train strategy classifier
            from sklearn.ensemble import RandomForestClassifier
            strategy_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
            strategy_classifier.fit(X, y_strategy)
            self.meta_models['strategy_classifier'] = strategy_classifier
            
            # Train performance predictor
            performance_predictor = RandomForestRegressor(n_estimators=100, random_state=42)
            performance_predictor.fit(X, y_performance)
            self.meta_models['performance_predictor'] = performance_predictor
            
            self.is_trained = True
            self.logger.info("Meta-models trained successfully")
            
        except Exception as e:
            self.logger.error(f"Error training meta-models: {e}")

    def predict_optimal_strategy(self, problem_signature: ProblemSignature) -> Tuple[str, Dict[str, Any], float]:
        """Predict the optimal strategy for a given problem."""
        if not self.is_trained:
            return "default", {}, 0.0
        
        try:
            features = self.extract_problem_features(problem_signature)
            features = features.reshape(1, -1)
            
            # Predict strategy
            strategy_classifier = self.meta_models['strategy_classifier']
            predicted_strategy = strategy_classifier.predict(features)[0]
            
            # Predict performance
            performance_predictor = self.meta_models['performance_predictor']
            predicted_performance = performance_predictor.predict(features)[0]
            
            # Get strategy confidence
            strategy_proba = strategy_classifier.predict_proba(features)[0]
            confidence = max(strategy_proba)
            
            strategy_config = self._get_strategy_config(predicted_strategy)
            
            return predicted_strategy, strategy_config, confidence
            
        except Exception as e:
            self.logger.error(f"Error predicting optimal strategy: {e}")
            return "default", {}, 0.0

    def _get_strategy_config(self, strategy_name: str) -> Dict[str, Any]:
        """Get configuration for a specific strategy."""
        # This would contain strategy-specific configurations
        strategy_configs = {
            'default': {'max_iterations': 100, 'tolerance': 1e-6},
            'aggressive': {'max_iterations': 200, 'tolerance': 1e-8},
            'conservative': {'max_iterations': 50, 'tolerance': 1e-4},
        }
        return strategy_configs.get(strategy_name, strategy_configs['default'])


class TransferLearningOptimizer:
    """Main transfer learning optimizer that combines all components."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild("TransferLearningOptimizer")
        self.similarity_detector = ProblemSimilarityDetector(config)
        self.knowledge_manager = KnowledgeTransferManager(config)
        self.meta_learner = MetaLearner(config)
        self.similarity_threshold = config.get('similarity_threshold', 0.7)
        self.max_source_problems = config.get('max_source_problems', 3)

    def optimize_with_transfer(self, objective_function, parameter_space: Dict[str, Any], 
                             **kwargs) -> Dict[str, Any]:
        """Optimize using transfer learning."""
        # Create problem signature
        problem_signature = self._create_problem_signature(objective_function, parameter_space)
        
        # Find similar problems
        similar_problems = self.knowledge_manager.find_similar_problems(
            problem_signature, self.max_source_problems
        )
        
        # Filter by similarity threshold
        source_problem_ids = [
            problem_id for problem_id, similarity in similar_problems 
            if similarity >= self.similarity_threshold
        ]
        
        if source_problem_ids:
            # Transfer knowledge
            transferred_knowledge = self.knowledge_manager.transfer_knowledge(
                problem_signature, source_problem_ids
            )
            
            # Adapt knowledge to current problem
            adapted_knowledge = self.knowledge_manager.adapt_transferred_knowledge(
                transferred_knowledge, {'objective_function': objective_function, 'parameter_space': parameter_space}
            )
            
            # Get meta-learning prediction
            predicted_strategy, strategy_config, confidence = self.meta_learner.predict_optimal_strategy(
                problem_signature
            )
            
            # Create optimization configuration
            optimization_config = self._create_optimization_config_with_transfer(
                adapted_knowledge, predicted_strategy, strategy_config
            )
            
            self.logger.info(f"Using transfer learning with {len(source_problem_ids)} source problems")
        else:
            # No similar problems found, use default configuration
            optimization_config = self._create_default_optimization_config()
            self.logger.info("No similar problems found, using default optimization")
        
        # Run optimization
        optimization_results = self._run_optimization(objective_function, parameter_space, optimization_config)
        
        # Update meta-learner with results
        if 'best_score' in optimization_results:
            self.meta_learner.add_training_example(
                problem_signature, 
                optimization_config.get('strategy', 'default'),
                optimization_results['best_score']
            )
        
        return optimization_results

    def _create_problem_signature(self, objective_function, parameter_space: Dict[str, Any]) -> ProblemSignature:
        """Create a problem signature for the current optimization problem."""
        # Generate problem ID
        problem_id = self._generate_problem_id(objective_function, parameter_space)
        
        # Extract bounds
        bounds = self._extract_bounds(parameter_space)
        
        # Count constraints
        constraint_count = self._count_constraints(parameter_space)
        
        # Create feature vector
        feature_vector = self._create_feature_vector(parameter_space, bounds)
        
        # Estimate complexity
        complexity_score = self._estimate_complexity(objective_function, parameter_space)
        
        # Create metadata
        metadata = {
            'function_name': objective_function.__name__,
            'parameter_count': len(parameter_space),
            'domain': 'unknown'  # Could be inferred from function name or parameters
        }
        
        return ProblemSignature(
            problem_id=problem_id,
            dimensionality=len(parameter_space),
            parameter_bounds=bounds,
            objective_type="minimization",  # Default assumption
            constraint_count=constraint_count,
            noise_level=0.0,  # Could be estimated from function evaluation
            complexity_score=complexity_score,
            feature_vector=feature_vector,
            metadata=metadata
        )

    def _generate_problem_id(self, objective_function, parameter_space: Dict[str, Any]) -> str:
        """Generate a unique ID for the optimization problem."""
        # Create a hash of the function and parameter space
        function_str = str(objective_function.__name__)
        param_str = str(sorted(parameter_space.items()))
        combined_str = function_str + param_str
        return hashlib.md5(combined_str.encode()).hexdigest()[:8]

    def _extract_bounds(self, parameter_space: Dict[str, Any]) -> List[Tuple[float, float]]:
        """Extract parameter bounds from parameter space."""
        bounds = []
        for param_config in parameter_space.values():
            if isinstance(param_config, dict) and 'bounds' in param_config:
                bounds.append(tuple(param_config['bounds']))
        return bounds

    def _count_constraints(self, parameter_space: Dict[str, Any]) -> int:
        """Count the number of constraints in the parameter space."""
        constraint_count = 0
        for param_name, param_config in parameter_space.items():
            if isinstance(param_config, dict) and 'constraints' in param_config:
                constraint_count += len(param_config['constraints'])
        return constraint_count

    def _create_feature_vector(self, parameter_space: Dict[str, Any], bounds: List[Tuple[float, float]]) -> np.ndarray:
        """Create a feature vector representing the problem."""
        features = []
        
        # Number of parameters
        features.append(len(parameter_space))
        
        # Average bound width
        if bounds:
            bound_widths = [abs(b[1] - b[0]) for b in bounds]
            features.append(np.mean(bound_widths))
            features.append(np.std(bound_widths))
        else:
            features.extend([0.0, 0.0])
        
        # Parameter types (continuous vs discrete)
        continuous_count = sum(1 for config in parameter_space.values() 
                             if isinstance(config, dict) and config.get('type') == 'continuous')
        features.append(continuous_count / len(parameter_space) if parameter_space else 0.0)
        
        return np.array(features)

    def _estimate_complexity(self, objective_function, parameter_space: Dict[str, Any]) -> float:
        """Estimate the complexity of the optimization problem."""
        complexity = 0.0
        
        # Base complexity from number of parameters
        complexity += len(parameter_space) * 0.1
        
        # Add complexity based on function name (heuristic)
        func_name = objective_function.__name__.lower()
        if 'complex' in func_name or 'difficult' in func_name:
            complexity += 1.0
        elif 'simple' in func_name or 'easy' in func_name:
            complexity -= 0.5
        
        return max(0.1, complexity)

    def _create_optimization_config_with_transfer(self, adapted_knowledge: Dict[str, Any], 
                                                predicted_strategy: str, 
                                                strategy_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create optimization configuration using transferred knowledge."""
        config = {
            'strategy': adapted_knowledge.get('strategies', {}).get('strategy', predicted_strategy),
            'hyperparameters': adapted_knowledge.get('hyperparameters', {}),
            'initial_models': adapted_knowledge.get('models', {}),
            'max_iterations': strategy_config.get('max_iterations', 100),
            'tolerance': strategy_config.get('tolerance', 1e-6),
        }
        return config

    def _create_default_optimization_config(self) -> Dict[str, Any]:
        """Create default optimization configuration."""
        config = {
            'strategy': 'default',
            'hyperparameters': {},
            'initial_models': {},
            'max_iterations': 100,
            'tolerance': 1e-6,
        }
        return config

    def _run_optimization(self, objective_function, parameter_space: Dict[str, Any], 
                         optimization_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run the actual optimization."""
        # This would integrate with your existing surrogate optimizer
        # For now, return a mock result
        return {
            'best_parameters': {name: 0.0 for name in parameter_space.keys()},
            'best_score': 0.0,
            'convergence_history': [1.0, 0.8, 0.6, 0.4, 0.2],
            'training_time': 1.0,
            'strategy_used': optimization_config.get('strategy', 'default')
        }