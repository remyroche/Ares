"""
Bayesian Optimization Search Strategy

This module implements Bayesian optimization for neural architecture search,
using Gaussian processes to model the architecture performance landscape.
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from scipy.optimize import minimize
from scipy.stats import norm
import torch

from ..search.search_space import SearchSpace, ArchitectureConfig
from ..utils.nas_utils import NASUtils
from ..utils.logging_utils import NASLogger

logger = logging.getLogger(__name__)

@dataclass
class BayesianSearchConfig:
    """Configuration for Bayesian search."""
    max_samples: int = 100
    acquisition_function: str = "ei"  # "ei", "ucb", "pi"
    exploration_weight: float = 1.0  # Higher = more exploration
    noise_level: float = 0.01
    n_candidates: int = 100  # Number of candidates for acquisition optimization
    n_restarts_optimizer: int = 10
    random_state: int = 42

class BayesianSearch:
    """
    Bayesian Optimization Search Strategy

    Uses Gaussian processes to model the architecture performance landscape
    and intelligently selects promising architectures to evaluate.
    """

    def __init__(self, config: BayesianSearchConfig):
        """Initialize Bayesian search.

        Args:
            config: Bayesian search configuration
        """
        self.config = config
        self.logger = NASLogger.get_logger(self.__class__.__name__)

        # Initialize components
        self.search_space = SearchSpace()
        self.nas_utils = NASUtils()

        # Search state
        self.X_train = []  # Architecture encodings
        self.y_train = []  # Architecture scores
        self.searched_architectures = []

        # GP model parameters
        self.length_scale = 1.0
        self.signal_variance = 1.0
        self.noise_variance = config.noise_level

        # Acquisition function
        self.acquisition_functions = {
            'ei': self._expected_improvement,
            'ucb': self._upper_confidence_bound,
            'pi': self._probability_of_improvement
        }

        self.logger.info("🧠 Bayesian search initialized")

    def generate_architecture(self, iteration: int) -> Optional[ArchitectureConfig]:
        """
        Generate architecture using Bayesian optimization.

        Args:
            iteration: Current search iteration

        Returns:
            Optimized architecture configuration or None
        """
        try:
            if iteration <= 5:  # Initial random samples
                return self._generate_random_architecture()
            else:
                return self._generate_bayesian_architecture()

        except Exception as e:
            self.logger.error(f"❌ Failed to generate architecture at iteration {iteration}: {e}")
            return None

    def _generate_random_architecture(self) -> ArchitectureConfig:
        """Generate a random architecture for initial sampling.

        Returns:
            Random architecture configuration
        """
        input_dim = 100  # Default for market data
        output_dim = 5   # Default for regime detection

        architecture = self.search_space.generate_random_architecture(
            input_dim=input_dim,
            output_dim=output_dim,
            problem_type="regime_detection"
        )

        return architecture

    def _generate_bayesian_architecture(self) -> ArchitectureConfig:
        """Generate architecture using Bayesian optimization.

        Returns:
            Bayesian-optimized architecture configuration
        """
        # Encode current architectures
        X = np.array([self._encode_architecture(arch) for arch in self.searched_architectures])

        if len(X) < 2:
            return self._generate_random_architecture()

        # Optimize acquisition function
        best_candidate = self._optimize_acquisition(X)

        # Convert candidate back to architecture
        architecture = self._decode_architecture(best_candidate)

        self.logger.debug(f"🧠 Generated Bayesian architecture: {architecture.name}")
        return architecture

    def _encode_architecture(self, architecture: ArchitectureConfig) -> np.ndarray:
        """Encode architecture to numerical vector.

        Args:
            architecture: Architecture configuration

        Returns:
            Numerical encoding
        """
        encoding = []

        # Hidden dimensions (pad or truncate to fixed length)
        hidden_dims = architecture.hidden_dims[:5] + [0] * max(0, 5 - len(architecture.hidden_dims))
        encoding.extend(hidden_dims)

        # Activation function (one-hot)
        activations = ['relu', 'tanh', 'sigmoid', 'leaky_relu', 'elu', 'gelu', 'swish']
        activation_idx = activations.index(architecture.activation) if architecture.activation in activations else 0
        activation_onehot = [1 if i == activation_idx else 0 for i in range(len(activations))]
        encoding.extend(activation_onehot)

        # Dropout rate (normalized)
        encoding.append(architecture.dropout_rate)

        # Boolean features
        encoding.extend([
            1 if architecture.batch_norm else 0,
            1 if architecture.use_residual else 0,
            1 if architecture.use_attention else 0,
            1 if architecture.use_lstm else 0,
            1 if architecture.use_convolution else 0,
            architecture.complexity_score,
            architecture.estimated_params / 1000000  # Normalize parameters
        ])

        return np.array(encoding)

    def _decode_architecture(self, encoding: np.ndarray) -> ArchitectureConfig:
        """Decode numerical vector back to architecture.

        Args:
            encoding: Numerical encoding

        Returns:
            Architecture configuration
        """
        # Extract components
        hidden_dims = [int(x) for x in encoding[:5] if x > 0][:3]  # Max 3 hidden layers

        # Activation (argmax)
        activations = ['relu', 'tanh', 'sigmoid', 'leaky_relu', 'elu', 'gelu', 'swish']
        activation_start = 5
        activation_end = activation_start + len(activations)
        activation_idx = np.argmax(encoding[activation_start:activation_end])
        activation = activations[activation_idx]

        # Dropout rate
        dropout_rate = float(encoding[activation_end])

        # Boolean features
        batch_norm = bool(encoding[activation_end + 1])
        use_residual = bool(encoding[activation_end + 2])
        use_attention = bool(encoding[activation_end + 3])
        use_lstm = bool(encoding[activation_end + 4])
        use_convolution = bool(encoding[activation_end + 5])

        # Create architecture name
        dims_str = '_'.join(map(str, hidden_dims))
        name = f"bayes_{dims_str}_{activation}_d{dropout_rate:.2f}"

        # Create configuration
        config = ArchitectureConfig(
            name=name,
            input_dim=100,  # Default
            output_dim=5,   # Default
            hidden_dims=hidden_dims,
            activation=activation,
            dropout_rate=dropout_rate,
            batch_norm=batch_norm,
            use_residual=use_residual,
            problem_type="regime_detection",
            use_attention=use_attention,
            use_lstm=use_lstm,
            use_convolution=use_convolution
        )

        return config

    def _optimize_acquisition(self, X: np.ndarray) -> np.ndarray:
        """Optimize acquisition function to find next architecture.

        Args:
            X: Current architecture encodings

        Returns:
            Optimal architecture encoding
        """
        # Define bounds for optimization
        bounds = self._get_architecture_bounds()

        # Initial candidates
        n_candidates = self.config.n_candidates
        candidates = np.random.uniform(bounds[:, 0], bounds[:, 1], (n_candidates, bounds.shape[0]))

        # Optimize acquisition function
        best_candidate = None
        best_acquisition_value = float('-inf')

        for candidate in candidates:
            # Ensure candidate is within bounds
            candidate = np.clip(candidate, bounds[:, 0], bounds[:, 1])

            # Calculate acquisition value
            acquisition_value = self._calculate_acquisition(X, candidate)

            if acquisition_value > best_acquisition_value:
                best_acquisition_value = acquisition_value
                best_candidate = candidate.copy()

        # Local optimization around best candidate
        if best_candidate is not None:
            result = minimize(
                lambda x: -self._calculate_acquisition(X, x),  # Negative for minimization
                best_candidate,
                bounds=bounds,
                method='L-BFGS-B',
                options={'maxiter': self.config.n_restarts_optimizer}
            )

            if result.success:
                best_candidate = result.x

        return best_candidate if best_candidate is not None else candidates[0]

    def _get_architecture_bounds(self) -> np.ndarray:
        """Get bounds for architecture optimization.

        Returns:
            Bounds array of shape (n_features, 2)
        """
        bounds = []

        # Hidden dimensions (5 positions, can be 0)
        for _ in range(5):
            bounds.append([0, 1024])  # 0 to 1024

        # Activation function (one-hot, 7 options)
        for _ in range(7):
            bounds.append([0, 1])  # 0 to 1

        # Dropout rate
        bounds.append([0.0, 0.5])

        # Boolean features (6 booleans)
        for _ in range(6):
            bounds.append([0, 1])

        # Complexity and parameters (normalized)
        bounds.append([0, 10])   # Complexity
        bounds.append([0, 10])   # Parameters (in millions)

        return np.array(bounds)

    def _calculate_acquisition(self, X: np.ndarray, candidate: np.ndarray) -> float:
        """Calculate acquisition function value.

        Args:
            X: Current training data
            candidate: Candidate architecture encoding

        Returns:
            Acquisition function value
        """
        if len(X) < 2:
            return 0.0

        acquisition_fn = self.acquisition_functions.get(self.config.acquisition_function, self._expected_improvement)
        return acquisition_fn(X, candidate)

    def _expected_improvement(self, X: np.ndarray, candidate: np.ndarray) -> float:
        """Calculate Expected Improvement acquisition function.

        Args:
            X: Current training data
            candidate: Candidate architecture encoding

        Returns:
            Expected improvement
        """
        # Predict mean and std for candidate
        mu, sigma = self._predict_gp(X, candidate)

        if sigma == 0:
            return 0.0

        # Current best
        y_best = np.max(self.y_train)

        # Calculate improvement
        improvement = mu - y_best - 0.01  # xi parameter for exploration
        z = improvement / sigma
        ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)

        return ei * self.config.exploration_weight

    def _upper_confidence_bound(self, X: np.ndarray, candidate: np.ndarray) -> float:
        """Calculate Upper Confidence Bound acquisition function.

        Args:
            X: Current training data
            candidate: Candidate architecture encoding

        Returns:
            Upper confidence bound
        """
        mu, sigma = self._predict_gp(X, candidate)
        beta = self.config.exploration_weight
        ucb = mu + beta * sigma

        return ucb

    def _probability_of_improvement(self, X: np.ndarray, candidate: np.ndarray) -> float:
        """Calculate Probability of Improvement acquisition function.

        Args:
            X: Current training data
            candidate: Candidate architecture encoding

        Returns:
            Probability of improvement
        """
        mu, sigma = self._predict_gp(X, candidate)

        if sigma == 0:
            return 0.0

        y_best = np.max(self.y_train)
        z = (mu - y_best) / sigma
        pi = norm.cdf(z)

        return pi * self.config.exploration_weight

    def _predict_gp(self, X: np.ndarray, candidate: np.ndarray) -> Tuple[float, float]:
        """Predict Gaussian process mean and standard deviation.

        Args:
            X: Training data
            candidate: Candidate point

        Returns:
            Tuple of (mean, standard deviation)
        """
        # Simple GP implementation using RBF kernel
        # In practice, you might want to use a proper GP library like GPy or sklearn's GaussianProcessRegressor

        if len(X) == 0:
            return 0.0, 1.0

        # Calculate kernel matrix
        K = self._rbf_kernel(X, X)
        K_s = self._rbf_kernel(X, candidate.reshape(1, -1))
        K_ss = self._rbf_kernel(candidate.reshape(1, -1), candidate.reshape(1, -1))

        # Add noise
        K += np.eye(len(X)) * self.noise_variance

        # Cholesky decomposition
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            # Add jitter for numerical stability
            K += np.eye(len(X)) * 1e-6
            L = np.linalg.cholesky(K)

        # Solve for alpha
        alpha = np.linalg.solve(L, self.y_train)
        alpha = np.linalg.solve(L.T, alpha)

        # Predict mean
        mu = K_s.T @ alpha

        # Predict variance
        v = np.linalg.solve(L, K_s.T)
        var = K_ss - v.T @ v

        return float(mu[0]), float(np.sqrt(var[0][0]))

    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """RBF (Gaussian) kernel.

        Args:
            X1: First set of points
            X2: Second set of points

        Returns:
            Kernel matrix
        """
        X1 = X1 / self.length_scale
        X2 = X2 / self.length_scale

        dist_sq = np.sum(X1**2, axis=1, keepdims=True) + np.sum(X2**2, axis=1) - 2 * X1 @ X2.T
        K = self.signal_variance * np.exp(-0.5 * dist_sq)

        return K

    def update_observations(self, architecture: ArchitectureConfig, score: float):
        """Update observations with new architecture and score.

        Args:
            architecture: Evaluated architecture
            score: Architecture score
        """
        encoding = self._encode_architecture(architecture)
        self.X_train.append(encoding)
        self.y_train.append(score)
        self.searched_architectures.append(architecture)

        self.logger.debug(f"📊 Added observation: {architecture.name} -> {score:.4f}")

    def get_search_statistics(self) -> Dict[str, Any]:
        """Get search statistics.

        Returns:
            Dictionary with search statistics
        """
        return {
            'n_observations': len(self.X_train),
            'acquisition_function': self.config.acquisition_function,
            'exploration_weight': self.config.exploration_weight,
            'best_score': np.max(self.y_train) if self.y_train else None,
            'mean_score': np.mean(self.y_train) if self.y_train else None,
            'std_score': np.std(self.y_train) if self.y_train else None
        }

    def reset_search(self):
        """Reset search state."""
        self.X_train = []
        self.y_train = []
        self.searched_architectures = []
        self.logger.info("🔄 Bayesian search reset")