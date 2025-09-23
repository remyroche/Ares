"""
Markov State Model (MSM) NAS Integration

This module provides NAS optimization for Markov State Models,
replacing HMM-specific functionality with MSM-based approaches.
"""

import numpy as np
import torch
import torch.nn as nn
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from sklearn.base import BaseEstimator

from ..core.nas_search import NASArchitectureSearch, NASSearchConfig
from ..core.nas_model import NASModel
from ..core.nas_trainer import NASTrainer, TrainingConfig
from ..core.nas_evaluator import NASEvaluator, EvaluationConfig
from ..search.search_space import SearchSpace, ArchitectureConfig

logger = logging.getLogger(__name__)

@dataclass
class MSM_NAS_Config:
    """Configuration for MSM NAS optimization."""
    n_states_min: int = 3
    n_states_max: int = 15
    lag_time_min: int = 1
    lag_time_max: int = 10
    connectivity_threshold: float = 0.1
    ergodic_cutoff: float = 1e-6
    use_bayesian_optimization: bool = True
    n_optimization_trials: int = 20

class MSM_NAS_Optimizer:
    """
    NAS Optimization for Markov State Models

    Optimizes neural architectures for MSM-based regime detection,
    focusing on state identification and transition modeling without
    traditional HMM assumptions.
    """

    def __init__(self, config: MSM_NAS_Config):
        """Initialize MSM NAS optimizer.

        Args:
            config: MSM NAS configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize NAS components
        self.nas_search = NASArchitectureSearch(NASSearchConfig(
            max_iterations=config.n_optimization_trials,
            search_strategy="random",
            primary_metric="accuracy"
        ))

        self.search_space = SearchSpace()

        # MSM-specific components
        self.state_encoder = None
        self.transition_modeler = None

        self.logger.info("🔍 MSM NAS Optimizer initialized")

    def optimize_msm_architecture(self,
                                 market_data: np.ndarray,
                                 n_states: int = 5,
                                 n_iterations: int = 20) -> Dict[str, Any]:
        """
        Optimize neural architecture for MSM-based regime detection.

        Args:
            market_data: Market data for training
            n_states: Number of MSM states
            n_iterations: Number of optimization iterations

        Returns:
            Dictionary with optimization results
        """
        logger.info(f"🚀 Optimizing MSM architecture for {n_states} states")

        try:
            # Configure for MSM problem
            search_config = NASSearchConfig(
                max_iterations=n_iterations,
                search_strategy="random",
                primary_metric="msm_score"
            )

            # Update NAS search
            self.nas_search = NASArchitectureSearch(search_config)

            # Perform architecture search
            search_result = self.nas_search.search(
                train_data=(market_data, self._create_msm_labels(market_data, n_states)),
                validation_data=(market_data, self._create_msm_labels(market_data, n_states)),
                problem_type="msm_regime_detection",
                input_shape=market_data.shape
            )

            # Train best architecture
            best_model = NASModel.create_from_config(search_result.best_architecture, "msm_regime_detection")

            # Create MSM-specific training data
            X_train, y_train = self._prepare_msm_training_data(market_data, n_states)

            trainer_config = TrainingConfig(epochs=30, batch_size=64)
            trainer = NASTrainer(trainer_config)
            training_result = trainer.train(best_model, X_train, X_train, "msm_regime_detection")  # Self-supervised

            # Evaluate
            evaluator_config = EvaluationConfig(batch_size=64)
            evaluator = NASEvaluator(evaluator_config)
            evaluation_result = evaluator.evaluate_architecture(
                training_result.model, X_train, X_train, problem_type="msm_regime_detection"
            )

            results = {
                'search_result': search_result,
                'training_result': training_result,
                'evaluation_result': evaluation_result,
                'best_architecture': search_result.best_architecture,
                'best_score': search_result.best_score,
                'n_states': n_states,
                'msm_score': evaluation_result.accuracy,
                'model_type': 'MSM_NAS'
            }

            logger.info(f"✅ MSM architecture optimization completed with score: {evaluation_result.accuracy:.4f}")
            return results

        except Exception as e:
            logger.error(f"❌ MSM architecture optimization failed: {e}")
            raise

    def _create_msm_labels(self, market_data: np.ndarray, n_states: int) -> np.ndarray:
        """
        Create MSM state labels from market data.

        Args:
            market_data: Market data
            n_states: Number of states

        Returns:
            State labels
        """
        # Simple MSM state assignment based on price movement
        if len(market_data.shape) > 1:
            # Use closing prices or main feature
            prices = market_data[:, 0] if market_data.shape[1] > 0 else market_data.flatten()
        else:
            prices = market_data

        # Calculate returns
        returns = np.diff(prices) / prices[:-1]

        # Create states based on return quantiles
        quantiles = np.quantile(np.abs(returns), np.linspace(0, 1, n_states))
        quantiles = np.concatenate([[quantiles[0] - 1e-6], quantiles])  # Add lower bound

        # Assign states
        state_labels = np.digitize(np.abs(returns), quantiles) - 1
        state_labels = np.clip(state_labels, 0, n_states - 1)

        # Add sign information (positive/negative movement)
        positive_mask = returns > 0
        state_labels = state_labels * 2 + positive_mask.astype(int)

        # Ensure we don't exceed n_states - 1
        state_labels = np.clip(state_labels, 0, n_states - 1)

        return state_labels.astype(np.int64)

    def _prepare_msm_training_data(self, market_data: np.ndarray, n_states: int) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
        """
        Prepare training data for MSM.

        Args:
            market_data: Market data
            n_states: Number of states

        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        # Create state labels
        state_labels = self._create_msm_labels(market_data, n_states)

        # Create input sequences (sliding windows)
        sequence_length = min(20, len(market_data) - 1)  # Max sequence length

        if len(market_data) <= sequence_length:
            # Pad if too short
            padded_data = np.pad(market_data, ((0, sequence_length - len(market_data) + 1), (0, 0)), mode='edge')
            X = padded_data[:sequence_length].reshape(1, sequence_length, -1)
            y = state_labels[:1]
        else:
            # Create sequences
            X = []
            y = []

            for i in range(len(market_data) - sequence_length):
                seq = market_data[i:i+sequence_length]
                X.append(seq)
                y.append(state_labels[i+sequence_length-1])  # Predict next state

            X = np.array(X)
            y = np.array(y)

        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)

        # Create datasets
        from torch.utils.data import TensorDataset
        dataset = TensorDataset(X_tensor, y_tensor)

        # Split into train/val (80/20)
        n_train = int(0.8 * len(dataset))
        train_dataset = torch.utils.data.Subset(dataset, range(n_train))
        val_dataset = torch.utils.data.Subset(dataset, range(n_train, len(dataset)))

        return train_dataset, val_dataset

class MSM_Ensemble_NAS:
    """
    MSM Ensemble NAS for Complementary Models

    Uses NAS to find complementary models that work well together,
    optimizing for ensemble performance rather than individual performance.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize MSM ensemble NAS.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # NAS components
        self.nas_search = NASArchitectureSearch(NASSearchConfig(
            max_iterations=config.get('n_iterations', 30),
            search_strategy="random"
        ))

        # Ensemble state
        self.ensemble_models = []
        self.ensemble_weights = []
        self.complementarity_matrix = None

        self.logger.info("🎭 MSM Ensemble NAS initialized")

    def find_complementary_models(self,
                                 market_data: np.ndarray,
                                 n_models: int = 3,
                                 n_states: int = 5) -> Dict[str, Any]:
        """
        Find complementary models using NAS optimization.

        Args:
            market_data: Market data for training
            n_models: Number of complementary models to find
            n_states: Number of MSM states

        Returns:
            Dictionary with complementary models and analysis
        """
        logger.info(f"🔍 Finding {n_models} complementary MSM models")

        try:
            complementary_models = []
            diversity_scores = []

            # Find first model (best individual performance)
            logger.info("🏆 Finding first model (best individual performance)...")
            first_model_result = self._find_best_individual_model(market_data, n_states)
            complementary_models.append(first_model_result)
            diversity_scores.append(1.0)  # First model has max diversity

            # Find complementary models
            for i in range(1, n_models):
                logger.info(f"🔍 Finding model {i+1}/{n_models} (maximizing complementarity)...")

                complementary_result = self._find_complementary_model(
                    market_data, complementary_models, n_states
                )

                complementary_models.append(complementary_result)
                diversity_score = self._calculate_model_diversity(
                    complementary_result['model'],
                    [m['model'] for m in complementary_models[:-1]]
                )
                diversity_scores.append(diversity_score)

            # Optimize ensemble weights
            logger.info("⚖️ Optimizing ensemble weights...")
            ensemble_weights = self._optimize_ensemble_weights(
                complementary_models, market_data, n_states
            )

            # Create complementarity analysis
            complementarity_analysis = self._analyze_complementarity(
                complementary_models, market_data, n_states
            )

            results = {
                'complementary_models': complementary_models,
                'diversity_scores': diversity_scores,
                'ensemble_weights': ensemble_weights,
                'complementarity_analysis': complementarity_analysis,
                'individual_performance': [m['score'] for m in complementary_models],
                'ensemble_performance': self._evaluate_ensemble(complementary_models, market_data, n_states),
                'n_models': n_models,
                'n_states': n_states,
                'optimization_method': 'complementarity_optimization'
            }

            logger.info(f"✅ Found {n_models} complementary models")
            logger.info(f"📊 Individual scores: {[r['score']:.4f for r in complementary_models]}")
            logger.info(f"🎯 Ensemble performance: {results['ensemble_performance']:.4f}")

            return results

        except Exception as e:
            logger.error(f"❌ Complementary model search failed: {e}")
            raise

    def _find_best_individual_model(self, market_data: np.ndarray, n_states: int) -> Dict[str, Any]:
        """Find the best individual model.

        Args:
            market_data: Market data
            n_states: Number of states

        Returns:
            Dictionary with best model results
        """
        # Use standard NAS search for best individual performance
        search_result = self.nas_search.search(
            train_data=(market_data, self._create_msm_labels(market_data, n_states)),
            validation_data=(market_data, self._create_msm_labels(market_data, n_states)),
            problem_type="msm_regime_detection"
        )

        return {
            'model': search_result.best_architecture,
            'score': search_result.best_score,
            'search_result': search_result,
            'model_type': 'individual_best'
        }

    def _find_complementary_model(self,
                                 market_data: np.ndarray,
                                 existing_models: List[Dict[str, Any]],
                                 n_states: int) -> Dict[str, Any]:
        """
        Find a model that complements existing models.

        Args:
            market_data: Market data
            existing_models: List of existing models
            n_states: Number of states

        Returns:
            Dictionary with complementary model results
        """
        # Create custom evaluation function that rewards complementarity
        def complementary_evaluation(model, X, y):
            # Evaluate individual performance
            individual_score = self._evaluate_individual_performance(model, X, y)

            # Calculate complementarity bonus
            complementarity_bonus = 0.0
            for existing_model in existing_models:
                existing_arch = existing_model['model']
                diversity = self._calculate_architecture_diversity(model, existing_arch)
                complementarity_bonus += diversity * 0.1  # Small bonus for diversity

            return individual_score + complementarity_bonus

        # Perform search with complementary evaluation
        search_result = self.nas_search.search(
            train_data=(market_data, self._create_msm_labels(market_data, n_states)),
            validation_data=(market_data, self._create_msm_labels(market_data, n_states)),
            problem_type="complementary_msm"
        )

        return {
            'model': search_result.best_architecture,
            'score': search_result.best_score,
            'search_result': search_result,
            'model_type': 'complementary',
            'complementarity_score': complementary_evaluation(search_result.best_architecture, market_data, None)
        }

    def _evaluate_individual_performance(self, architecture: ArchitectureConfig,
                                       X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate individual model performance.

        Args:
            architecture: Architecture to evaluate
            X: Feature matrix
            y: Target labels

        Returns:
            Performance score
        """
        # This would integrate with the actual model evaluation
        # For now, return a placeholder based on architecture complexity
        complexity = architecture.calculate_complexity()
        return 0.5 + (1.0 / (1.0 + complexity)) * 0.3  # Placeholder

    def _calculate_architecture_diversity(self, arch1: ArchitectureConfig, arch2: ArchitectureConfig) -> float:
        """
        Calculate diversity between two architectures.

        Args:
            arch1: First architecture
            arch2: Second architecture

        Returns:
            Diversity score (0-1, higher = more diverse)
        """
        diversity = 0.0
        factors = 0

        # Hidden dimensions diversity
        if arch1.hidden_dims != arch2.hidden_dims:
            diversity += 1.0
        factors += 1

        # Activation diversity
        if arch1.activation != arch2.activation:
            diversity += 1.0
        factors += 1

        # Dropout diversity
        dropout_diff = abs(arch1.dropout_rate - arch2.dropout_rate)
        diversity += min(dropout_diff * 2, 1.0)  # Scale to 0-1
        factors += 1

        # Architecture features
        features = ['batch_norm', 'use_residual', 'use_attention', 'use_lstm', 'use_convolution']
        for feature in features:
            val1 = getattr(arch1, feature, False)
            val2 = getattr(arch2, feature, False)
            if val1 != val2:
                diversity += 1.0
            factors += 1

        return diversity / factors if factors > 0 else 0.0

    def _calculate_model_diversity(self, model: ArchitectureConfig, other_models: List[ArchitectureConfig]) -> float:
        """
        Calculate average diversity of a model compared to other models.

        Args:
            model: Model to evaluate
            other_models: List of other models

        Returns:
            Average diversity score
        """
        if not other_models:
            return 1.0

        total_diversity = 0.0
        for other_model in other_models:
            diversity = self._calculate_architecture_diversity(model, other_model)
            total_diversity += diversity

        return total_diversity / len(other_models)

    def _optimize_ensemble_weights(self,
                                  models: List[Dict[str, Any]],
                                  market_data: np.ndarray,
                                  n_states: int) -> np.ndarray:
        """
        Optimize ensemble weights for complementary models.

        Args:
            models: List of model results
            market_data: Market data
            n_states: Number of states

        Returns:
            Optimized ensemble weights
        """
        # Simple weight optimization based on individual performance
        scores = [model['score'] for model in models]

        # Normalize scores to weights
        total_score = sum(scores)
        weights = [score / total_score for score in scores]

        return np.array(weights)

    def _analyze_complementarity(self,
                                models: List[Dict[str, Any]],
                                market_data: np.ndarray,
                                n_states: int) -> Dict[str, Any]:
        """
        Analyze complementarity of the model ensemble.

        Args:
            models: List of model results
            market_data: Market data
            n_states: Number of states

        Returns:
            Complementarity analysis
        """
        n_models = len(models)

        # Calculate pairwise diversity
        diversity_matrix = np.zeros((n_models, n_models))
        for i in range(n_models):
            for j in range(i+1, n_models):
                diversity = self._calculate_architecture_diversity(
                    models[i]['model'], models[j]['model']
                )
                diversity_matrix[i, j] = diversity
                diversity_matrix[j, i] = diversity

        # Calculate average diversity per model
        avg_diversities = np.mean(diversity_matrix, axis=1)

        # Calculate overall complementarity score
        overall_complementarity = np.mean(diversity_matrix)

        return {
            'diversity_matrix': diversity_matrix,
            'average_diversities': avg_diversities,
            'overall_complementarity': overall_complementarity,
            'diversity_variance': np.var(avg_diversities),
            'min_diversity': np.min(diversity_matrix[diversity_matrix > 0]) if np.any(diversity_matrix > 0) else 0.0,
            'max_diversity': np.max(diversity_matrix)
        }

    def _evaluate_ensemble(self,
                          models: List[Dict[str, Any]],
                          market_data: np.ndarray,
                          n_states: int) -> float:
        """
        Evaluate ensemble performance.

        Args:
            models: List of model results
            market_data: Market data
            n_states: Number of states

        Returns:
            Ensemble performance score
        """
        # Simple ensemble evaluation
        individual_scores = [model['score'] for model in models]
        weights = self.ensemble_weights if hasattr(self, 'ensemble_weights') else None

        if weights is None:
            weights = np.ones(len(models)) / len(models)

        # Weighted average of individual scores
        ensemble_score = np.sum(np.array(individual_scores) * weights)

        return ensemble_score


# Convenience functions
def optimize_msm_architecture(market_data: np.ndarray,
                            n_states: int = 5,
                            config: Optional[MSM_NAS_Config] = None) -> Dict[str, Any]:
    """
    Convenience function for MSM architecture optimization.

    Args:
        market_data: Market data for training
        n_states: Number of MSM states
        config: MSM NAS configuration

    Returns:
        Dictionary with optimization results
    """
    if config is None:
        config = MSM_NAS_Config()

    optimizer = MSM_NAS_Optimizer(config)
    return optimizer.optimize_msm_architecture(market_data, n_states)


def find_complementary_msm_models(market_data: np.ndarray,
                                 n_models: int = 3,
                                 n_states: int = 5,
                                 config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Convenience function for finding complementary MSM models.

    Args:
        market_data: Market data for training
        n_models: Number of complementary models
        n_states: Number of MSM states
        config: Configuration dictionary

    Returns:
        Dictionary with complementary models
    """
    if config is None:
        config = {'n_iterations': 30}

    ensemble_nas = MSM_Ensemble_NAS(config)
    return ensemble_nas.find_complementary_models(market_data, n_models, n_states)