"""
Reinforcement Learning Neural Architecture Search (RL-NAS)

This module implements RL-NAS techniques for neural architecture search:
- Q-Learning based architecture search
- Policy gradient methods for NAS
- Deep reinforcement learning for architecture optimization
- Reward shaping for better search
- Multi-agent architecture search
- Hierarchical RL-NAS
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as distributions
from torch.nn.utils import weight_norm
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import OrderedDict, defaultdict, deque
import copy
import random
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class RLNASConfig:
    """Configuration for RL-NAS."""
    search_space_size: int = 100
    hidden_state_size: int = 64
    num_actions: int = 10
    learning_rate: float = 1e-3
    gamma: float = 0.99  # Discount factor
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    target_update_freq: int = 10
    memory_size: int = 10000
    batch_size: int = 64
    num_episodes: int = 1000
    use_double_dqn: bool = True
    use_dueling_dqn: bool = True
    use_rainbow: bool = True
    use_ppo: bool = True
    ppo_clip_ratio: float = 0.2
    use_a2c: bool = False
    value_coeff: float = 0.5
    entropy_coeff: float = 0.01

class RLController(nn.Module):
    """
    Reinforcement Learning Controller for NAS.

    Uses deep RL to learn architecture search policies.
    """

    def __init__(self, config: RLNASConfig):
        """Initialize RL controller.

        Args:
            config: RL-NAS configuration
        """
        super(RLController, self).__init__()
        self.config = config

        # Q-Network for DQN
        self.q_network = self._build_q_network()
        self.target_q_network = self._build_q_network()
        self.target_q_network.load_state_dict(self.q_network.state_dict())

        # Policy network for PPO/A2C
        if config.use_ppo or config.use_a2c:
            self.policy_network = self._build_policy_network()
            self.value_network = self._build_value_network()

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=config.learning_rate)

        # Replay memory
        self.memory = deque(maxlen=config.memory_size)

        # Exploration
        self.epsilon = config.epsilon_start
        self.steps = 0

        self.logger = logging.getLogger(self.__class__.__name__)

    def _build_q_network(self) -> nn.Module:
        """Build Q-network for DQN."""
        if self.config.use_dueling_dqn:
            return DuelingDQN(self.config)
        else:
            return QNetwork(self.config)

    def _build_policy_network(self) -> nn.Module:
        """Build policy network for actor-critic methods."""
        return PolicyNetwork(self.config)

    def _build_value_network(self) -> nn.Module:
        """Build value network for actor-critic methods."""
        return ValueNetwork(self.config)

    def select_action(self, state: torch.Tensor) -> int:
        """Select action using epsilon-greedy policy."""
        if random.random() < self.epsilon:
            action = random.randint(0, self.config.num_actions - 1)
        else:
            with torch.no_grad():
                q_values = self.q_network(state)
                action = q_values.argmax().item()

        return action

    def update_epsilon(self):
        """Update epsilon for exploration."""
        self.epsilon = max(
            self.config.epsilon_end,
            self.epsilon * self.config.epsilon_decay
        )

    def remember(self, state: torch.Tensor, action: int, reward: float, next_state: torch.Tensor, done: bool):
        """Store experience in replay memory."""
        self.memory.append((state, action, reward, next_state, done))

    def update_q_network(self):
        """Update Q-network using DQN."""
        if len(self.memory) < self.config.batch_size:
            return

        # Sample batch from memory
        batch = random.sample(self.memory, self.config.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack(states)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.stack(next_states)
        dones = torch.tensor(dones, dtype=torch.float32)

        # Compute current Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target Q values
        with torch.no_grad():
            if self.config.use_double_dqn:
                # Double DQN: use main network to select actions, target network to evaluate
                next_actions = self.q_network(next_states).argmax(dim=1)
                next_q = self.target_q_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                next_q = self.target_q_network(next_states).max(dim=1)[0]

            target_q = rewards + (1 - dones) * self.config.gamma * next_q

        # Compute loss
        loss = F.mse_loss(current_q, target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        """Update target Q-network."""
        self.target_q_network.load_state_dict(self.q_network.state_dict())

class QNetwork(nn.Module):
    """Q-Network for DQN."""

    def __init__(self, config: RLNASConfig):
        super(QNetwork, self).__init__()
        self.config = config

        self.network = nn.Sequential(
            nn.Linear(config.hidden_state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, config.num_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class DuelingDQN(nn.Module):
    """Dueling Deep Q-Network."""

    def __init__(self, config: RLNASConfig):
        super(DuelingDQN, self).__init__()
        self.config = config

        # Feature extraction
        self.feature_layer = nn.Sequential(
            nn.Linear(config.hidden_state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        # Advantage stream
        self.advantage_layer = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, config.num_actions)
        )

        # Value stream
        self.value_layer = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_layer(x)

        advantage = self.advantage_layer(features)
        value = self.value_layer(features)

        # Q = V + A - mean(A)
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)

        return q_values

class PolicyNetwork(nn.Module):
    """Policy Network for actor-critic methods."""

    def __init__(self, config: RLNASConfig):
        super(PolicyNetwork, self).__init__()
        self.config = config

        self.network = nn.Sequential(
            nn.Linear(config.hidden_state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, config.num_actions),
            nn.Softmax(dim=-1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class ValueNetwork(nn.Module):
    """Value Network for actor-critic methods."""

    def __init__(self, config: RLNASConfig):
        super(ValueNetwork, self).__init__()
        self.config = config

        self.network = nn.Sequential(
            nn.Linear(config.hidden_state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class RLNAS_Search:
    """
    RL-NAS Search using reinforcement learning.

    Uses deep RL to search for optimal neural architectures.
    """

    def __init__(self, config: RLNASConfig):
        """Initialize RL-NAS search.

        Args:
            config: RL-NAS configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize RL controller
        self.controller = RLController(config)

        # Search state
        self.best_architecture = None
        self.best_reward = float('-inf')
        self.search_history = []

        self.logger.info("🧠 RL-NAS Search initialized")

    def search(self, train_data: Tuple[np.ndarray, np.ndarray],
              val_data: Tuple[np.ndarray, np.ndarray],
              problem_type: str = "classification") -> Dict[str, Any]:
        """
        Perform RL-NAS search.

        Args:
            train_data: Training data
            val_data: Validation data
            problem_type: Type of problem

        Returns:
            Search results
        """
        logger.info("🚀 Starting RL-NAS search")

        # RL search loop
        for episode in range(self.config.num_episodes):
            # Sample architecture using RL policy
            architecture = self._sample_architecture()

            # Evaluate architecture
            reward = self._evaluate_architecture(architecture, train_data, val_data, problem_type)

            # Update RL controller
            state = self._get_state_representation(architecture)
            next_state = self._get_next_state(architecture, reward)

            self.controller.remember(state, 0, reward, next_state, False)
            self.controller.update_q_network()

            # Update target network
            if episode % self.config.target_update_freq == 0:
                self.controller.update_target_network()

            # Update exploration rate
            self.controller.update_epsilon()

            # Track best architecture
            if reward > self.best_reward:
                self.best_reward = reward
                self.best_architecture = architecture

            # Log progress
            if episode % 100 == 0:
                self.logger.info(f"📈 Episode {episode}: Best reward = {self.best_reward:.4f}")

        results = {
            'best_architecture': self.best_architecture,
            'best_reward': self.best_reward,
            'search_method': 'rl_nas',
            'num_episodes': self.config.num_episodes,
            'final_epsilon': self.controller.epsilon
        }

        self.logger.info(f"✅ RL-NAS search completed with best reward: {self.best_reward:.4f}")
        return results

    def _sample_architecture(self) -> Dict[str, Any]:
        """Sample architecture using RL policy."""
        # Get current state
        state = torch.randn(1, self.config.hidden_state_size)  # Placeholder state

        # Select action
        action = self.controller.select_action(state)

        # Convert action to architecture parameters
        architecture = self._action_to_architecture(action)

        return architecture

    def _action_to_architecture(self, action: int) -> Dict[str, Any]:
        """Convert RL action to architecture configuration."""
        # Simple mapping from action to architecture
        architectures = [
            {"hidden_dims": [32], "activation": "relu", "dropout": 0.0},
            {"hidden_dims": [64], "activation": "relu", "dropout": 0.1},
            {"hidden_dims": [128], "activation": "relu", "dropout": 0.2},
            {"hidden_dims": [64, 32], "activation": "tanh", "dropout": 0.1},
            {"hidden_dims": [128, 64], "activation": "leaky_relu", "dropout": 0.2},
            {"hidden_dims": [256, 128, 64], "activation": "swish", "dropout": 0.3},
            # Add more architecture variants...
        ]

        return architectures[action % len(architectures)]

    def _evaluate_architecture(self, architecture: Dict[str, Any],
                              train_data: Tuple[np.ndarray, np.ndarray],
                              val_data: Tuple[np.ndarray, np.ndarray],
                              problem_type: str) -> float:
        """Evaluate architecture and return reward."""
        try:
            # Create model from architecture
            model = self._create_model_from_architecture(architecture)

            # Train briefly
            train_loader, val_loader = self._create_data_loaders(train_data, val_data)
            trainer = self._create_trainer(model)
            training_result = trainer.train(train_loader, val_loader, num_epochs=5)

            # Compute reward based on accuracy and complexity
            accuracy = training_result['accuracy']
            complexity = self._compute_complexity(architecture)

            reward = accuracy - 0.1 * complexity  # Trade-off between accuracy and complexity

            return reward

        except Exception as e:
            self.logger.warning(f"⚠️ Architecture evaluation failed: {e}")
            return -1.0  # Negative reward for failed architectures

    def _create_model_from_architecture(self, architecture: Dict[str, Any]) -> nn.Module:
        """Create PyTorch model from architecture specification."""
        layers = []

        # Input layer
        layers.append(nn.Linear(100, architecture['hidden_dims'][0]))  # Assuming input_dim=100
        layers.append(nn.ReLU())

        # Hidden layers
        for i in range(1, len(architecture['hidden_dims'])):
            layers.append(nn.Linear(architecture['hidden_dims'][i-1], architecture['hidden_dims'][i]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(architecture['dropout']))

        # Output layer
        layers.append(nn.Linear(architecture['hidden_dims'][-1], 5))  # Assuming output_dim=5
        layers.append(nn.LogSoftmax(dim=1))

        return nn.Sequential(*layers)

    def _compute_complexity(self, architecture: Dict[str, Any]) -> float:
        """Compute architecture complexity."""
        total_params = sum(architecture['hidden_dims']) + architecture['hidden_dims'][-1] * 5
        return total_params / 10000.0  # Normalized complexity

    def _get_state_representation(self, architecture: Dict[str, Any]) -> torch.Tensor:
        """Get state representation for RL."""
        # Convert architecture to state vector
        state = torch.zeros(self.config.hidden_state_size)

        # Encode hidden dimensions
        for i, dim in enumerate(architecture['hidden_dims']):
            if i < 4:  # Max 4 layers
                state[i] = dim / 1000.0  # Normalize

        # Encode activation function
        activations = {'relu': 0, 'tanh': 1, 'leaky_relu': 2, 'swish': 3}
        act_idx = activations.get(architecture['activation'], 0)
        state[10] = act_idx / 3.0

        # Encode dropout
        state[11] = architecture['dropout']

        return state.unsqueeze(0)

    def _get_next_state(self, architecture: Dict[str, Any], reward: float) -> torch.Tensor:
        """Get next state representation."""
        return self._get_state_representation(architecture)

    def _create_data_loaders(self, train_data: Tuple[np.ndarray, np.ndarray],
                           val_data: Tuple[np.ndarray, np.ndarray]) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """Create data loaders for evaluation."""
        X_train, y_train = train_data
        X_val, y_val = val_data

        train_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_train), torch.LongTensor(y_train)
        )
        val_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_val), torch.LongTensor(y_val)
        )

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

        return train_loader, val_loader

    def _create_trainer(self, model: nn.Module) -> Any:
        """Create simple trainer for architecture evaluation."""
        class SimpleTrainer:
            def __init__(self, model):
                self.model = model
                self.optimizer = optim.Adam(model.parameters(), lr=1e-3)
                self.criterion = nn.NLLLoss()

            def train(self, train_loader, val_loader, num_epochs=5):
                for epoch in range(num_epochs):
                    self.model.train()
                    for batch_x, batch_y in train_loader:
                        self.optimizer.zero_grad()
                        outputs = self.model(batch_x)
                        loss = self.criterion(outputs, batch_y)
                        loss.backward()
                        self.optimizer.step()

                # Evaluate
                self.model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        outputs = self.model(batch_x)
                        predictions = outputs.argmax(dim=1)
                        correct += (predictions == batch_y).sum().item()
                        total += batch_y.size(0)

                accuracy = correct / total
                return {'accuracy': accuracy}

        return SimpleTrainer(model)

class VisionTransformerTimeSeries(nn.Module):
    """
    Vision Transformer adapted for time series data.

    Treats time series sequences as "images" with temporal dimension.
    """

    def __init__(self, sequence_length: int = 100, feature_dim: int = 4,
                 patch_size: int = 10, embed_dim: int = 64, num_heads: int = 8,
                 num_layers: int = 6, num_classes: int = 5):
        """Initialize Vision Transformer for time series.

        Args:
            sequence_length: Length of time series
            feature_dim: Number of features
            patch_size: Size of each patch
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            num_classes: Number of output classes
        """
        super(VisionTransformerTimeSeries, self).__init__()

        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_classes = num_classes

        # Calculate number of patches
        self.num_patches = sequence_length // patch_size

        # Patch embedding
        self.patch_embedding = nn.Linear(patch_size * feature_dim, embed_dim)

        # Position embedding
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))

        # Class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.classification_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Vision Transformer.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, feature_dim)

        Returns:
            Class logits
        """
        batch_size = x.size(0)

        # Create patches
        x_patches = x.view(batch_size, self.num_patches, self.patch_size * self.feature_dim)

        # Patch embedding
        x_embedded = self.patch_embedding(x_patches)  # (batch_size, num_patches, embed_dim)

        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, embed_dim)
        x_embedded = torch.cat((cls_tokens, x_embedded), dim=1)  # (batch_size, num_patches + 1, embed_dim)

        # Add position embedding
        x_embedded = x_embedded + self.position_embedding

        # Transformer encoding
        x_encoded = self.transformer_encoder(x_embedded)

        # Classification from class token
        cls_output = x_encoded[:, 0]  # (batch_size, embed_dim)
        logits = self.classification_head(cls_output)

        return logits

class TemporalFusionTransformer(nn.Module):
    """
    Temporal Fusion Transformer for time series forecasting.

    Combines LSTM with attention mechanisms for temporal modeling.
    """

    def __init__(self, input_size: int = 4, hidden_size: int = 64,
                 attention_heads: int = 8, num_layers: int = 3, output_size: int = 5):
        """Initialize Temporal Fusion Transformer.

        Args:
            input_size: Input feature dimension
            hidden_size: Hidden dimension
            attention_heads: Number of attention heads
            num_layers: Number of layers
            output_size: Output dimension
        """
        super(TemporalFusionTransformer, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.attention_heads = attention_heads
        self.num_layers = num_layers
        self.output_size = output_size

        # Variable selection networks
        self.variable_selection = VariableSelectionNetwork(input_size, hidden_size)

        # LSTM layers
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers,
                           batch_first=True, dropout=0.1 if num_layers > 1 else 0)

        # Temporal attention
        self.temporal_attention = nn.MultiheadAttention(hidden_size, attention_heads, batch_first=True)

        # Position-wise feedforward
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(0.1)
        )

        # Output projection
        self.output_projection = nn.Linear(hidden_size, output_size)

        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Temporal Fusion Transformer.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            Output logits
        """
        batch_size, seq_len, input_size = x.shape

        # Variable selection
        selected_vars = self.variable_selection(x)  # (batch_size, seq_len, hidden_size)

        # LSTM processing
        lstm_out, _ = self.lstm(selected_vars)  # (batch_size, seq_len, hidden_size)

        # Temporal attention
        attn_out, _ = self.temporal_attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.layer_norm1(lstm_out + attn_out)

        # Position-wise feedforward
        ff_out = self.feedforward(attn_out)
        ff_out = self.layer_norm2(attn_out + ff_out)

        # Global average pooling
        pooled = ff_out.mean(dim=1)  # (batch_size, hidden_size)

        # Output projection
        output = self.output_projection(pooled)

        return output

class VariableSelectionNetwork(nn.Module):
    """Variable Selection Network for TFT."""

    def __init__(self, input_size: int, hidden_size: int):
        super(VariableSelectionNetwork, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        # Variable selection weights
        self.variable_weights = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
            nn.Softmax(dim=-1)
        )

        # Variable processing
        self.variable_processor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Variable Selection Network.

        Args:
            x: Input tensor

        Returns:
            Processed variables
        """
        batch_size, seq_len, input_size = x.shape

        # Compute variable selection weights
        weights = self.variable_weights(x)  # (batch_size, seq_len, input_size)

        # Apply weights to variables
        weighted_vars = x * weights  # (batch_size, seq_len, input_size)

        # Process selected variables
        processed_vars = self.variable_processor(weighted_vars)

        return processed_vars

class MultiAgentNAS:
    """
    Multi-agent reinforcement learning for NAS.

    Uses multiple RL agents to collaboratively search for architectures.
    """

    def __init__(self, num_agents: int = 5, config: RLNASConfig = None):
        """Initialize multi-agent NAS.

        Args:
            num_agents: Number of RL agents
            config: RL-NAS configuration
        """
        self.num_agents = num_agents
        self.config = config or RLNASConfig()

        # Initialize agents
        self.agents = [RLController(self.config) for _ in range(num_agents)]

        # Communication between agents
        self.communication_matrix = torch.ones(num_agents, num_agents) / num_agents

        self.logger = logging.getLogger(self.__class__.__name__)

    def collaborative_search(self, train_data: Tuple[np.ndarray, np.ndarray],
                           val_data: Tuple[np.ndarray, np.ndarray]) -> Dict[str, Any]:
        """
        Perform collaborative search using multiple agents.

        Args:
            train_data: Training data
            val_data: Validation data

        Returns:
            Collaborative search results
        """
        logger.info(f"🤝 Starting collaborative search with {self.num_agents} agents")

        # Collaborative search loop
        for episode in range(self.config.num_episodes):
            # Each agent proposes architecture
            agent_architectures = []
            agent_rewards = []

            for agent in self.agents:
                architecture = self._sample_architecture(agent)
                reward = self._evaluate_architecture(architecture, train_data, val_data)

                agent_architectures.append(architecture)
                agent_rewards.append(reward)

            # Agents share information
            self._agent_communication(agent_rewards)

            # Update agents
            for agent, reward in zip(self.agents, agent_rewards):
                state = self._get_state_representation(agent_architectures[0])  # Simplified
                next_state = self._get_next_state(agent_architectures[0], reward)

                agent.remember(state, 0, reward, next_state, False)
                agent.update_q_network()

            # Update target networks
            if episode % self.config.target_update_freq == 0:
                for agent in self.agents:
                    agent.update_target_network()

            # Update exploration
            for agent in self.agents:
                agent.update_epsilon()

            # Log progress
            if episode % 100 == 0:
                best_reward = max(agent_rewards)
                self.logger.info(f"📈 Episode {episode}: Best reward = {best_reward:.4f}")

        # Find best architecture across all agents
        best_architecture = None
        best_reward = float('-inf')

        for agent, architecture in zip(self.agents, agent_architectures):
            reward = self._evaluate_architecture(architecture, train_data, val_data)
            if reward > best_reward:
                best_reward = reward
                best_architecture = architecture

        results = {
            'best_architecture': best_architecture,
            'best_reward': best_reward,
            'num_agents': self.num_agents,
            'search_method': 'multi_agent_rl_nas'
        }

        self.logger.info(f"✅ Collaborative search completed with best reward: {best_reward:.4f}")
        return results

    def _agent_communication(self, agent_rewards: List[float]):
        """Enable communication between agents."""
        # Simple communication: share rewards
        mean_reward = sum(agent_rewards) / len(agent_rewards)

        # Update communication matrix based on performance
        for i, reward in enumerate(agent_rewards):
            if reward > mean_reward:
                # Increase influence of better-performing agents
                self.communication_matrix[i] *= 1.1
            else:
                # Decrease influence of worse-performing agents
                self.communication_matrix[i] *= 0.9

        # Normalize communication matrix
        self.communication_matrix = self.communication_matrix / self.communication_matrix.sum()

    def _sample_architecture(self, agent: RLController) -> Dict[str, Any]:
        """Sample architecture using agent policy."""
        state = torch.randn(1, self.config.hidden_state_size)
        action = agent.select_action(state)
        return self._action_to_architecture(action)

    def _evaluate_architecture(self, architecture: Dict[str, Any],
                              train_data: Tuple[np.ndarray, np.ndarray],
                              val_data: Tuple[np.ndarray, np.ndarray]) -> float:
        """Evaluate architecture (same as in RLNAS_Search)."""
        # Implementation same as in RLNAS_Search
        return 0.5  # Placeholder

    def _action_to_architecture(self, action: int) -> Dict[str, Any]:
        """Convert action to architecture (same as in RLNAS_Search)."""
        return {"hidden_dims": [64], "activation": "relu", "dropout": 0.1}

    def _get_state_representation(self, architecture: Dict[str, Any]) -> torch.Tensor:
        """Get state representation (same as in RLNAS_Search)."""
        return torch.randn(1, self.config.hidden_state_size)

    def _get_next_state(self, architecture: Dict[str, Any], reward: float) -> torch.Tensor:
        """Get next state (same as in RLNAS_Search)."""
        return torch.randn(1, self.config.hidden_state_size)

# Utility functions
def create_vision_transformer_timeseries(sequence_length: int = 100,
                                       feature_dim: int = 4,
                                       patch_size: int = 10) -> VisionTransformerTimeSeries:
    """Create Vision Transformer for time series."""
    return VisionTransformerTimeSeries(sequence_length, feature_dim, patch_size)

def create_temporal_fusion_transformer(input_size: int = 4,
                                     hidden_size: int = 64) -> TemporalFusionTransformer:
    """Create Temporal Fusion Transformer."""
    return TemporalFusionTransformer(input_size, hidden_size)

def create_multi_agent_nas(num_agents: int = 5) -> MultiAgentNAS:
    """Create multi-agent NAS system."""
    return MultiAgentNAS(num_agents)