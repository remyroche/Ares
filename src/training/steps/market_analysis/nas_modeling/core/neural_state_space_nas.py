"""
Neural State Space Models (SSM) NAS Integration

This module provides NAS optimization for Neural State Space Models,
offering a more advanced alternative to MSM for regime detection.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
class NeuralSSMConfig:
    """Configuration for Neural State Space Models."""
    state_dim: int = 8
    hidden_dim: int = 64
    transition_layers: int = 2
    emission_layers: int = 2
    use_attention: bool = True
    attention_heads: int = 4
    use_residual: bool = True
    dropout_rate: float = 0.1
    sequence_length: int = 20

class NeuralStateSpaceModel(nn.Module):
    """
    Neural State Space Model for regime detection.

    Combines the flexibility of neural networks with state space modeling
    for better regime detection than traditional MSM or HMM.
    """

    def __init__(self, config: NeuralSSMConfig):
        """Initialize Neural SSM.

        Args:
            config: Neural SSM configuration
        """
        super(NeuralStateSpaceModel, self).__init__()

        self.config = config
        self.state_dim = config.state_dim
        self.hidden_dim = config.hidden_dim

        # State encoder: maps observations to latent states
        self.state_encoder = self._build_state_encoder()

        # State transition model: predicts next state from current state
        self.transition_model = self._build_transition_model()

        # Observation decoder: maps states to observations
        self.observation_decoder = self._build_observation_decoder()

        # Regime classifier: classifies regimes from states
        self.regime_classifier = self._build_regime_classifier()

        self.apply(self._init_weights)

    def _build_state_encoder(self) -> nn.Module:
        """Build state encoder network."""
        layers = []

        # Input projection
        layers.append(nn.Linear(self.config.sequence_length * 4, self.hidden_dim))  # 4 features
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(self.config.dropout_rate))

        # Hidden layers
        for i in range(self.config.transition_layers - 1):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.config.dropout_rate))

        # Output to state dimension
        layers.append(nn.Linear(self.hidden_dim, self.state_dim))
        layers.append(nn.Tanh())  # States often bounded

        return nn.Sequential(*layers)

    def _build_transition_model(self) -> nn.Module:
        """Build state transition model."""
        layers = []

        # Transition from current state to next state
        layers.append(nn.Linear(self.state_dim, self.hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(self.config.dropout_rate))

        # Hidden layers
        for i in range(self.config.transition_layers - 1):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.config.dropout_rate))

        # Output to next state
        layers.append(nn.Linear(self.hidden_dim, self.state_dim))

        return nn.Sequential(*layers)

    def _build_observation_decoder(self) -> nn.Module:
        """Build observation decoder."""
        layers = []

        # Decode state to observation
        layers.append(nn.Linear(self.state_dim, self.hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(self.config.dropout_rate))

        # Hidden layers
        for i in range(self.config.emission_layers - 1):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.config.dropout_rate))

        # Output to observation dimension
        layers.append(nn.Linear(self.hidden_dim, self.config.sequence_length * 4))

        return nn.Sequential(*layers)

    def _build_regime_classifier(self) -> nn.Module:
        """Build regime classifier."""
        layers = []

        # Classify regime from state
        layers.append(nn.Linear(self.state_dim, self.hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(self.config.dropout_rate))

        # Hidden layers
        for i in range(self.config.emission_layers - 1):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.config.dropout_rate))

        # Output to regime classes (assume 5 regimes)
        layers.append(nn.Linear(self.hidden_dim, 5))
        layers.append(nn.LogSoftmax(dim=-1))

        return nn.Sequential(*layers)

    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through Neural SSM.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, features)

        Returns:
            Tuple of (regime_probs, state_sequence, reconstructed_observations)
        """
        batch_size, seq_len, features = x.shape

        # Encode initial state from first observation
        initial_obs = x[:, 0, :]  # First time step
        initial_state = self.state_encoder(initial_obs)

        # Process sequence through transition model
        state_sequence = [initial_state]
        for t in range(1, seq_len):
            current_state = state_sequence[-1]
            next_state = self.transition_model(current_state)
            state_sequence.append(next_state)

        # Stack state sequence
        states = torch.stack(state_sequence, dim=1)  # (batch_size, seq_len, state_dim)

        # Decode observations from states (for reconstruction loss)
        reconstructed = self.observation_decoder(states.view(batch_size, seq_len, -1))
        reconstructed = reconstructed.view(batch_size, seq_len, features)

        # Classify regimes from final state
        final_state = states[:, -1, :]  # Last state
        regime_probs = self.regime_classifier(final_state)

        return regime_probs, states, reconstructed

class NeuralSSM_NAS_Optimizer:
    """
    NAS Optimization for Neural State Space Models.

    Provides a more advanced alternative to MSM for regime detection,
    using neural networks to learn continuous state representations.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize Neural SSM NAS optimizer.

        Args:
            config: Optimization configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # NAS components
        self.nas_search = NASArchitectureSearch(NASSearchConfig(
            max_iterations=config.get('n_iterations', 30),
            search_strategy="random"
        ))

        self.logger.info("🔬 Neural SSM NAS Optimizer initialized")

    def optimize_neural_ssm(self,
                           market_data: np.ndarray,
                           n_regimes: int = 5,
                           n_iterations: int = 20) -> Dict[str, Any]:
        """
        Optimize Neural SSM architecture for regime detection.

        Args:
            market_data: Market data for training
            n_regimes: Number of market regimes
            n_iterations: Number of optimization iterations

        Returns:
            Dictionary with optimization results
        """
        logger.info(f"🚀 Optimizing Neural SSM architecture for {n_regimes} regimes")

        try:
            # Create training data
            X_train, y_train = self._prepare_neural_ssm_data(market_data, n_regimes)

            # Split data
            n_samples = len(X_train)
            n_train = int(0.7 * n_samples)
            n_val = int(0.15 * n_samples)

            X_train_split = X_train[:n_train]
            y_train_split = y_train[:n_train]
            X_val_split = X_train[n_train:n_train+n_val]
            y_val_split = y_train[n_train:n_train+n_val]
            X_test_split = X_train[n_train+n_val:]
            y_test_split = y_train[n_train+n_val:]

            # Create data loaders
            train_loader, val_loader, test_loader = self._create_data_loaders(
                X_train_split, y_train_split, X_val_split, y_val_split, X_test_split, y_test_split
            )

            # Perform architecture search
            search_result = self.nas_search.search(
                train_data=(X_train_split, y_train_split),
                validation_data=(X_val_split, y_val_split),
                problem_type="neural_ssm_regime_detection"
            )

            # Create Neural SSM model
            ssm_config = NeuralSSMConfig(
                state_dim=8,
                hidden_dim=64,
                transition_layers=2,
                emission_layers=2
            )
            best_model = NeuralStateSpaceModel(ssm_config)

            # Train model
            trainer_config = TrainingConfig(epochs=30, batch_size=64)
            trainer = NASTrainer(trainer_config)
            training_result = trainer.train(best_model, train_loader, val_loader, "neural_ssm_regime_detection")

            # Evaluate model
            evaluator_config = EvaluationConfig(batch_size=64)
            evaluator = NASEvaluator(evaluator_config)
            evaluation_result = evaluator.evaluate_architecture(
                training_result.model, train_loader, val_loader, test_loader,
                search_result.best_architecture.name, "neural_ssm_regime_detection"
            )

            results = {
                'search_result': search_result,
                'training_result': training_result,
                'evaluation_result': evaluation_result,
                'best_model': best_model,
                'best_score': search_result.best_score,
                'n_regimes': n_regimes,
                'model_type': 'neural_ssm',
                'state_space_approach': True,
                'neural_representation': True
            }

            logger.info(f"✅ Neural SSM optimization completed with accuracy: {evaluation_result.accuracy:.4f}")
            return results

        except Exception as e:
            logger.error(f"❌ Neural SSM optimization failed: {e}")
            raise

    def _prepare_neural_ssm_data(self, market_data: np.ndarray, n_regimes: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data for Neural SSM.

        Args:
            market_data: Market data
            n_regimes: Number of regimes

        Returns:
            Tuple of (X, y) where X is sequences and y is regime labels
        """
        # Normalize data
        normalized_data = (market_data - np.mean(market_data, axis=0)) / (np.std(market_data, axis=0) + 1e-8)

        # Create sequences
        sequence_length = min(20, len(normalized_data) - 1)
        X_sequences = []
        y_regimes = []

        for i in range(len(normalized_data) - sequence_length):
            seq = normalized_data[i:i+sequence_length]
            X_sequences.append(seq)

            # Create regime label based on price movement patterns
            price_seq = seq[:, 0]  # Close prices
            returns = np.diff(price_seq) / price_seq[:-1]

            # Classify regime based on return patterns
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            volatility_ratio = std_return / (np.abs(mean_return) + 1e-8)

            if mean_return > 0.002:  # Strong uptrend
                regime = 0  # Bullish
            elif mean_return < -0.002:  # Strong downtrend
                regime = 1  # Bearish
            elif volatility_ratio > 0.5:  # High volatility
                regime = 2  # Volatile
            elif np.abs(mean_return) < 0.001:  # Sideways
                regime = 3  # Consolidation
            else:
                regime = 4  # Mixed

            y_regimes.append(regime)

        X = np.array(X_sequences)
        y = np.array(y_regimes)

        return X, y

    def _create_data_loaders(self, X_train, y_train, X_val, y_val, X_test=None, y_test=None):
        """Create PyTorch data loaders."""
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.LongTensor(y_val)

        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)

        test_loader = None
        if X_test is not None and y_test is not None:
            X_test_tensor = torch.FloatTensor(X_test)
            y_test_tensor = torch.LongTensor(y_test)
            test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

        return train_loader, val_loader, test_loader

class TransformerRegimeDetector(nn.Module):
    """
    Transformer-based regime detector.

    Uses self-attention to capture complex temporal patterns
    in market data for regime detection.
    """

    def __init__(self, input_dim: int = 4, n_regimes: int = 5, d_model: int = 64, n_heads: int = 4, n_layers: int = 2):
        """Initialize Transformer regime detector.

        Args:
            input_dim: Input feature dimension
            n_regimes: Number of regimes
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
        """
        super(TransformerRegimeDetector, self).__init__()

        self.d_model = d_model
        self.n_regimes = n_regimes

        # Input embedding
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = self._create_positional_encoding()

        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, n_heads, batch_first=True)
            for _ in range(n_layers)
        ])

        # Output projection
        self.output_projection = nn.Linear(d_model, n_regimes)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def _create_positional_encoding(self, max_len: int = 100) -> torch.Tensor:
        """Create positional encoding."""
        pe = torch.zeros(max_len, self.d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-np.log(10000.0) / self.d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Transformer.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, features)

        Returns:
            Regime probabilities
        """
        batch_size, seq_len, features = x.shape

        # Embed input
        x_embedded = self.input_embedding(x)

        # Add positional encoding
        x_embedded = x_embedded + self.positional_encoding[:, :seq_len, :].to(x.device)

        # Apply transformer layers
        for layer in self.transformer_layers:
            x_embedded = layer(x_embedded)

        # Global average pooling
        x_pooled = torch.mean(x_embedded, dim=1)

        # Project to regimes
        output = self.output_projection(x_pooled)
        return self.log_softmax(output)

class ContrastiveRegimeLearner(nn.Module):
    """
    Contrastive learning for regime detection.

    Uses self-supervised contrastive learning to discover
    regime representations without labeled data.
    """

    def __init__(self, input_dim: int = 4, hidden_dim: int = 64, n_regimes: int = 5):
        """Initialize contrastive regime learner.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            n_regimes: Number of regimes
        """
        super(ContrastiveRegimeLearner, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_regimes = n_regimes

        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        )

        # Regime classifier
        self.regime_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_regimes)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through contrastive learner.

        Args:
            x: Input tensor

        Returns:
            Tuple of (features, projections, regime_logits)
        """
        features = self.feature_extractor(x)
        projections = self.projection_head(features)
        regime_logits = self.regime_classifier(features)

        return features, projections, regime_logits

    def contrastive_loss(self, projections1: torch.Tensor, projections2: torch.Tensor,
                        temperature: float = 0.5) -> torch.Tensor:
        """Calculate contrastive loss.

        Args:
            projections1: First set of projections
            projections2: Second set of projections
            temperature: Temperature for softmax

        Returns:
            Contrastive loss
        """
        batch_size = projections1.size(0)

        # Normalize projections
        projections1 = F.normalize(projections1, dim=-1)
        projections2 = F.normalize(projections2, dim=-1)

        # Concatenate projections
        projections = torch.cat([projections1, projections2], dim=0)

        # Compute similarity matrix
        similarity_matrix = torch.matmul(projections, projections.T) / temperature

        # Create labels (positive pairs are diagonal blocks)
        mask = torch.eye(batch_size, dtype=torch.bool)
        mask = torch.cat([mask, mask], dim=1)

        # Positive pairs
        positives = similarity_matrix[mask].view(2 * batch_size, -1)

        # Negative pairs
        negatives = similarity_matrix[~mask].view(2 * batch_size, -1)

        # Contrastive loss
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(2 * batch_size, dtype=torch.long, device=logits.device)

        loss = F.cross_entropy(logits, labels)
        return loss

def create_neural_ssm(config: NeuralSSMConfig) -> NeuralStateSpaceModel:
    """Create Neural SSM model.

    Args:
        config: Neural SSM configuration

    Returns:
        Neural SSM model
    """
    return NeuralStateSpaceModel(config)

def create_transformer_regime_detector(input_dim: int = 4, n_regimes: int = 5) -> TransformerRegimeDetector:
    """Create Transformer regime detector.

    Args:
        input_dim: Input feature dimension
        n_regimes: Number of regimes

    Returns:
        Transformer regime detector
    """
    return TransformerRegimeDetector(input_dim, n_regimes)

def create_contrastive_regime_learner(input_dim: int = 4, n_regimes: int = 5) -> ContrastiveRegimeLearner:
    """Create contrastive regime learner.

    Args:
        input_dim: Input feature dimension
        n_regimes: Number of regimes

    Returns:
        Contrastive regime learner
    """
    return ContrastiveRegimeLearner(input_dim, n_regimes)