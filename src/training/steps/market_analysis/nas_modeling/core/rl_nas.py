"""
RL NAS

Implementation for Reinforcement Learning Neural Architecture Search.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import random

class RLAction(Enum):
    """RL actions for architecture search."""
    ADD_LAYER = "add_layer"
    REMOVE_LAYER = "remove_layer"
    MODIFY_LAYER = "modify_layer"
    CHANGE_ACTIVATION = "change_activation"
    CHANGE_WIDTH = "change_width"

@dataclass
class RLConfig:
    """Configuration for RL NAS."""
    state_dim: int
    action_dim: int
    learning_rate: float = 0.001
    epsilon: float = 0.1
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.01
    gamma: float = 0.95
    memory_size: int = 10000
    batch_size: int = 32

class RL_NAS_Optimizer:
    """Reinforcement Learning NAS Optimizer."""

    def __init__(self, config: RLConfig, search_space: Optional[Dict] = None):
        """Initialize RL NAS optimizer.

        Args:
            config: RL configuration
            search_space: Optional search space configuration
        """
        self.config = config
        self.search_space = search_space or self._get_default_search_space()
        self.q_table = self._initialize_q_table()
        self.memory = []
        self.epsilon = config.epsilon
        self.episode_rewards = []
        self.best_architecture = None
        self.best_reward = float('-inf')

    def _get_default_search_space(self) -> Dict:
        """Get default search space configuration."""
        return {
            'max_layers': 10,
            'min_layers': 2,
            'layer_widths': [32, 64, 128, 256, 512],
            'activations': ['relu', 'tanh', 'sigmoid', 'swish'],
            'max_width': 512,
            'min_width': 32
        }

    def _initialize_q_table(self) -> Dict:
        """Initialize Q-table for RL."""
        return {}

    def _get_state_key(self, architecture: Dict) -> str:
        """Convert architecture to state key."""
        return str(sorted(architecture.items()))

    def _get_available_actions(self, architecture: Dict) -> List[RLAction]:
        """Get available actions for current architecture."""
        actions = []

        num_layers = len(architecture.get('layers', []))

        if num_layers < self.search_space['max_layers']:
            actions.append(RLAction.ADD_LAYER)

        if num_layers > self.search_space['min_layers']:
            actions.append(RLAction.REMOVE_LAYER)

        if num_layers > 0:
            actions.extend([RLAction.MODIFY_LAYER, RLAction.CHANGE_ACTIVATION, RLAction.CHANGE_WIDTH])

        return actions

    def _select_action(self, architecture: Dict) -> RLAction:
        """Select action using epsilon-greedy policy."""
        available_actions = self._get_available_actions(architecture)

        if not available_actions:
            return RLAction.ADD_LAYER

        state_key = self._get_state_key(architecture)

        if random.random() < self.epsilon:
            return random.choice(available_actions)
        else:
            # Select action with highest Q-value
            best_action = None
            best_q_value = float('-inf')

            for action in available_actions:
                q_key = (state_key, action.value)
                q_value = self.q_table.get(q_key, 0.0)

                if q_value > best_q_value:
                    best_q_value = q_value
                    best_action = action

            return best_action if best_action else random.choice(available_actions)

    def _execute_action(self, architecture: Dict, action: RLAction) -> Dict:
        """Execute action on architecture."""
        new_architecture = architecture.copy()

        if 'layers' not in new_architecture:
            new_architecture['layers'] = []

        if action == RLAction.ADD_LAYER:
            width = random.choice(self.search_space['layer_widths'])
            activation = random.choice(self.search_space['activations'])
            new_architecture['layers'].append({
                'width': width,
                'activation': activation
            })

        elif action == RLAction.REMOVE_LAYER and new_architecture['layers']:
            new_architecture['layers'].pop()

        elif action == RLAction.MODIFY_LAYER and new_architecture['layers']:
            layer_idx = random.randint(0, len(new_architecture['layers']) - 1)
            new_architecture['layers'][layer_idx]['width'] = random.choice(self.search_space['layer_widths'])

        elif action == RLAction.CHANGE_ACTIVATION and new_architecture['layers']:
            layer_idx = random.randint(0, len(new_architecture['layers']) - 1)
            new_architecture['layers'][layer_idx]['activation'] = random.choice(self.search_space['activations'])

        elif action == RLAction.CHANGE_WIDTH and new_architecture['layers']:
            layer_idx = random.randint(0, len(new_architecture['layers']) - 1)
            new_architecture['layers'][layer_idx]['width'] = random.choice(self.search_space['layer_widths'])

        return new_architecture

    def _evaluate_architecture(self, architecture: Dict, data: np.ndarray,
                              target: np.ndarray) -> float:
        """Evaluate architecture performance."""
        try:
            # Calculate reward based on architecture properties
            reward = 0.0

            # Reward for reasonable architecture size
            num_layers = len(architecture.get('layers', []))
            if 2 <= num_layers <= 8:
                reward += 1.0

            # Penalty for too many parameters
            total_params = sum(layer.get('width', 0) for layer in architecture.get('layers', []))
            if total_params > 10000:  # Too many parameters
                reward -= 0.5

            # Reward for diversity in activations
            activations = [layer.get('activation', 'relu') for layer in architecture.get('layers', [])]
            unique_activations = len(set(activations))
            reward += unique_activations * 0.1

            # Add some randomness to simulate actual performance
            reward += random.random() * 0.5

            return reward
        except Exception:
            return -1.0

    def _update_q_table(self, state: str, action: RLAction, reward: float,
                       next_state: str, next_actions: List[RLAction]):
        """Update Q-table using Q-learning."""
        current_q_key = (state, action.value)
        current_q = self.q_table.get(current_q_key, 0.0)

        # Calculate max Q-value for next state
        max_next_q = 0.0
        if next_actions:
            for next_action in next_actions:
                next_q_key = (next_state, next_action.value)
                next_q = self.q_table.get(next_q_key, 0.0)
                max_next_q = max(max_next_q, next_q)

        # Q-learning update
        new_q = current_q + self.config.learning_rate * (
            reward + self.config.gamma * max_next_q - current_q
        )

        self.q_table[current_q_key] = new_q

    def optimize(self, data: np.ndarray, target: np.ndarray,
                 episodes: int = 1000) -> Dict:
        """Optimize architecture using RL.

        Args:
            data: Input data
            target: Target data
            episodes: Number of training episodes

        Returns:
            Dictionary containing optimization results
        """
        current_architecture = {'layers': []}

        for episode in range(episodes):
            episode_reward = 0.0
            steps = 0
            max_steps = 20

            while steps < max_steps:
                # Select action
                action = self._select_action(current_architecture)

                # Execute action
                next_architecture = self._execute_action(current_architecture, action)

                # Evaluate reward
                reward = self._evaluate_architecture(next_architecture, data, target)
                episode_reward += reward

                # Update Q-table
                current_state = self._get_state_key(current_architecture)
                next_state = self._get_state_key(next_architecture)
                next_actions = self._get_available_actions(next_architecture)

                self._update_q_table(current_state, action, reward, next_state, next_actions)

                # Update best architecture
                if reward > self.best_reward:
                    self.best_reward = reward
                    self.best_architecture = next_architecture.copy()

                current_architecture = next_architecture
                steps += 1

            # Decay epsilon
            self.epsilon = max(self.config.epsilon_min,
                             self.epsilon * self.config.epsilon_decay)

            self.episode_rewards.append(episode_reward)

        return {
            'best_architecture': self.best_architecture,
            'best_reward': self.best_reward,
            'episode_rewards': self.episode_rewards,
            'q_table_size': len(self.q_table)
        }

    def get_best_architecture(self) -> Optional[Dict]:
        """Get the best architecture found during optimization."""
        return self.best_architecture

    def get_q_table(self) -> Dict:
        """Get the Q-table."""
        return self.q_table

    def reset(self):
        """Reset the optimizer."""
        self.q_table = {}
        self.memory = []
        self.epsilon = self.config.epsilon
        self.episode_rewards = []
        self.best_architecture = None
        self.best_reward = float('-inf')
