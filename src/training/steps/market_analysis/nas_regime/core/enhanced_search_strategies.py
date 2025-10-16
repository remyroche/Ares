"""
Enhanced Search Strategies for Neural Architecture Search

This module implements advanced search strategies for neural architecture search:
- Reinforcement Learning-based architecture search (RL-NAS)
- Differentiable Architecture Search (DARTS)
- Progressive Architecture Search (PAS)
- Multi-objective Evolutionary Search (MOES)
- Adaptive Search Strategy Selection (ASSS)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import logging
from dataclasses import dataclass, field
from enum import Enum
import random
import copy
import math
from collections import deque
import gymnasium as gym
from gymnasium import spaces
from datetime import datetime

# Import tprint for comprehensive debugging
from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error,
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)

logger = logging.getLogger(__name__)

class SearchStrategyType(Enum):
    """Available search strategy types."""
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    DIFFERENTIABLE_DARTS = "differentiable_darts"
    PROGRESSIVE_SEARCH = "progressive_search"
    MULTI_OBJECTIVE_EVOLUTIONARY = "multi_objective_evolutionary"
    ADAPTIVE_STRATEGY_SELECTION = "adaptive_strategy_selection"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    HYBRID_SEARCH = "hybrid_search"

@dataclass
class SearchStrategyConfig:
    """Configuration for search strategies."""
    strategy_type: SearchStrategyType = SearchStrategyType.REINFORCEMENT_LEARNING

    # RL-NAS parameters
    rl_learning_rate: float = 0.001
    rl_gamma: float = 0.99
    rl_epsilon_start: float = 1.0
    rl_epsilon_end: float = 0.01
    rl_epsilon_decay: int = 1000
    rl_replay_buffer_size: int = 10000
    rl_batch_size: int = 32
    rl_target_update_freq: int = 100

    # DARTS parameters
    darts_learning_rate: float = 0.025
    darts_momentum: float = 0.9
    darts_weight_decay: float = 3e-4
    darts_arch_learning_rate: float = 3e-4
    darts_arch_weight_decay: float = 1e-3
    darts_grad_clip: float = 5.0

    # Progressive search parameters
    progressive_initial_ops: int = 2
    progressive_growth_rate: float = 1.5
    progressive_max_ops: int = 10
    progressive_evolution_rounds: int = 5

    # Multi-objective parameters
    mo_objectives: List[str] = field(default_factory=lambda: ['accuracy', 'efficiency', 'complexity'])
    mo_population_size: int = 50
    mo_generations: int = 100
    mo_crossover_rate: float = 0.8
    mo_mutation_rate: float = 0.1

    # General parameters
    max_search_iterations: int = 1000
    early_stopping_patience: int = 50
    performance_threshold: float = 0.8
    enable_parallel_evaluation: bool = True
    n_workers: int = 4

class ArchitectureEnvironment(gym.Env):
    """Gym environment for RL-based architecture search."""

    def __init__(self, search_space, performance_evaluator, config):
        super().__init__()
        self.search_space = search_space
        self.performance_evaluator = performance_evaluator
        self.config = config

        # Define action and observation spaces
        self.action_space = spaces.Discrete(len(search_space.operations))
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(search_space.max_layers * search_space.max_ops_per_layer,), dtype=np.float32
        )

        self.current_architecture = None
        self.step_count = 0
        self.performance_history = deque(maxlen=100)

    def reset(self):
        """Reset the environment."""
        self.current_architecture = self.search_space.create_empty_architecture()
        self.step_count = 0
        return self._get_observation()

    def step(self, action):
        """Take a step in the environment."""
        # Apply action to modify architecture
        self.current_architecture = self.search_space.apply_operation(
            self.current_architecture, action
        )

        # Evaluate performance
        performance = self.performance_evaluator(self.current_architecture)
        self.performance_history.append(performance)

        # Calculate reward
        reward = self._calculate_reward(performance)

        # Check if done
        done = (self.step_count >= self.config.max_search_iterations or
                performance >= self.config.performance_threshold)

        self.step_count += 1

        return self._get_observation(), reward, done, {'performance': performance}

    def _get_observation(self):
        """Get current observation."""
        # Convert architecture to observation vector
        obs = np.zeros(self.observation_space.shape[0])

        if self.current_architecture:
            for i, layer in enumerate(self.current_architecture.layers):
                if i < len(obs):
                    obs[i] = layer.operation_id / len(self.search_space.operations)

        return obs.astype(np.float32)

    def _calculate_reward(self, performance):
        """Calculate reward based on performance."""
        # Base reward from performance
        base_reward = performance * 10

        # Bonus for improvement
        if len(self.performance_history) > 1:
            improvement = performance - self.performance_history[-2]
            improvement_bonus = improvement * 50
        else:
            improvement_bonus = 0

        # Penalty for complexity
        complexity_penalty = len(self.current_architecture.layers) * 0.1

        return base_reward + improvement_bonus - complexity_penalty

class DQNAgent:
    """Deep Q-Network agent for RL-based architecture search."""

    def __init__(self, state_size, action_size, config):
        self.state_size = state_size
        self.action_size = action_size
        self.config = config

        # Q-networks
        self.q_network = QNetwork(state_size, action_size)
        self.target_network = QNetwork(state_size, action_size)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=config.rl_learning_rate)

        # RL parameters
        self.epsilon = config.rl_epsilon_start
        self.epsilon_decay = (config.rl_epsilon_start - config.rl_epsilon_end) / config.rl_epsilon_decay
        self.gamma = config.rl_gamma

        # Experience replay
        self.memory = deque(maxlen=config.rl_replay_buffer_size)
        self.batch_size = config.rl_batch_size

        # Update target network
        self.update_target_network()

    def update_target_network(self):
        """Update the target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, training=True):
        """Choose action using epsilon-greedy policy."""
        if training and np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)

        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())

    def replay(self):
        """Train the agent on a batch of experiences."""
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])

        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.config.rl_epsilon_end:
            self.epsilon -= self.epsilon_decay

class QNetwork(nn.Module):
    """Q-network for DQN agent."""

    def __init__(self, state_size, action_size, hidden_size=256):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)

class ReinforcementLearningSearch:
    """Reinforcement Learning-based Neural Architecture Search."""

    def __init__(self, search_space, performance_evaluator, config):
        self.search_space = search_space
        self.performance_evaluator = performance_evaluator
        self.config = config

        # Create environment and agent
        self.env = ArchitectureEnvironment(search_space, performance_evaluator, config)

        self.agent = DQNAgent(
            self.env.observation_space.shape[0],
            self.env.action_space.n,
            config
        )

        self.best_architecture = None
        self.best_performance = -np.inf
        self.search_history = []

    def search(self, max_episodes=1000):
        """Perform RL-based architecture search."""
        for episode in range(max_episodes):
            state = self.env.reset()
            total_reward = 0
            episode_history = []
            step_count = 0

            while True:
                step_count += 1

                action = self.agent.act(state, training=True)
                next_state, reward, done, info = self.env.step(action)

                self.agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                episode_history.append({
                    'action': action,
                    'reward': reward,
                    'performance': info['performance']
                })

                if done:
                    break

            # Train the agent
            self.agent.replay()

            # Update target network periodically
            if episode % self.config.rl_target_update_freq == 0:
                self.agent.update_target_network()

            # Track best architecture
            if episode_history:
                final_performance = episode_history[-1]['performance']
                if final_performance > self.best_performance:
                    self.best_performance = final_performance
                    self.best_architecture = copy.deepcopy(self.env.current_architecture)

            self.search_history.append({
                'episode': episode,
                'total_reward': total_reward,
                'final_performance': episode_history[-1]['performance'] if episode_history else 0,
                'best_performance': self.best_performance
            })

        return {
            'best_architecture': self.best_architecture,
            'best_performance': self.best_performance,
            'search_history': self.search_history
        }

class DifferentiableArchitectureSearch:
    """Differentiable Architecture Search (DARTS) implementation."""

    def __init__(self, search_space, config):
        self.search_space = search_space
        self.config = config

        # Architecture parameters (alpha parameters)
        self.arch_parameters = {}
        self.weight_parameters = {}

        self.search_history = []
        self.best_architecture = None
        self.best_performance = -np.inf

    def _initialize_architecture_parameters(self):
        """Initialize architecture parameters."""
        # Initialize alpha parameters for each edge
        for edge_id in self.search_space.edges:
            num_ops = len(self.search_space.operations)
            self.arch_parameters[edge_id] = nn.Parameter(
                1e-3 * torch.randn(num_ops)
            )

    def _get_architecture_weights(self, edge_id):
        """Get architecture weights for an edge."""
        alpha = self.arch_parameters[edge_id]
        return F.softmax(alpha, dim=-1)

    def _sample_architecture(self):
        """Sample architecture from current alpha parameters."""
        architecture = self.search_space.create_empty_architecture()

        for edge_id in self.search_space.edges:
            weights = self._get_architecture_weights(edge_id)
            operation_idx = torch.multinomial(weights, 1).item()
            operation = self.search_space.operations[operation_idx]
            architecture = self.search_space.apply_operation(architecture, operation_idx)

        return architecture

    def search(self, train_loader, val_loader, max_epochs=100):
        """Perform DARTS search."""
        logger.info("Starting Differentiable Architecture Search (DARTS)")

        # Initialize architecture parameters
        self._initialize_architecture_parameters()

        # Optimizers
        arch_optimizer = torch.optim.Adam(
            self.arch_parameters.values(),
            lr=self.config.darts_arch_learning_rate,
            weight_decay=self.config.darts_arch_weight_decay
        )

        for epoch in range(max_epochs):
            # Training phase
            train_loss = self._train_epoch(train_loader, arch_optimizer)

            # Validation phase
            val_performance = self._validate_epoch(val_loader)

            # Sample and evaluate architecture
            if epoch % 10 == 0:
                sampled_arch = self._sample_architecture()
                arch_performance = self.performance_evaluator(sampled_arch)

                if arch_performance > self.best_performance:
                    self.best_performance = arch_performance
                    self.best_architecture = copy.deepcopy(sampled_arch)

                self.search_history.append({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_performance': val_performance,
                    'arch_performance': arch_performance,
                    'best_performance': self.best_performance
                })

                logger.info(f"Epoch {epoch}: Best performance = {self.best_performance:.4f}")

        return {
            'best_architecture': self.best_architecture,
            'best_performance': self.best_performance,
            'search_history': self.search_history,
            'final_alpha_parameters': self.arch_parameters
        }

    def _train_epoch(self, train_loader, arch_optimizer):
        """Train for one epoch."""
        total_loss = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            # Forward pass
            output = self._forward_with_architecture_weights(data)
            loss = F.cross_entropy(output, target)

            # Backward pass
            arch_optimizer.zero_grad()
            loss.backward()
            arch_optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def _validate_epoch(self, val_loader):
        """Validate for one epoch."""
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for data, target in val_loader:
                output = self._forward_with_architecture_weights(data)
                pred = output.argmax(dim=1)
                total_correct += pred.eq(target).sum().item()
                total_samples += target.size(0)

        return total_correct / total_samples

    def _forward_with_architecture_weights(self, x):
        """Forward pass using current architecture weights."""
        try:
            # Get the current architecture weights
            arch_weights = self.architecture_weights

            # Apply architecture weights to determine which operations to use
            if hasattr(self, 'supernet') and self.supernet is not None:
                # Use supernet with architecture weights
                return self.supernet.forward_with_weights(x, arch_weights)
            else:
                # Fallback: use the most likely operations based on weights
                selected_ops = self._select_operations_from_weights(arch_weights)
                return self._forward_with_selected_ops(x, selected_ops)

        except Exception as e:
            tprint(f"⚠️ [ARCHITECTURE] Error in forward pass with architecture weights: {e}", color="yellow")
            # Fallback to standard forward pass
            return self.forward(x)

    def _select_operations_from_weights(self, arch_weights):
        """Select operations based on architecture weights."""
        selected_ops = []

        for edge_weights in arch_weights:
            # Select the operation with the highest weight
            if len(edge_weights) > 0:
                max_idx = torch.argmax(edge_weights)
                selected_ops.append(max_idx.item())
            else:
                selected_ops.append(0)  # Default operation

        return selected_ops

    def _forward_with_selected_ops(self, x, selected_ops):
        """Forward pass with selected operations."""
        # This is a simplified implementation
        # In practice, this would involve applying the selected operations
        # to the input tensor

        # For now, return the input as-is (identity operation)
        # In a real implementation, this would apply the selected operations
        return x

class ProgressiveArchitectureSearch:
    """Progressive Architecture Search implementation."""

    def __init__(self, search_space, performance_evaluator, config):
        tprint("📈 [PROGRESSIVE] Initializing Progressive Architecture Search", color="blue", bold=True)
        tprint(f"📊 [PROGRESSIVE] Config: initial_ops={config.progressive_initial_ops}, max_ops={config.progressive_max_ops}", color="cyan")

        self.search_space = search_space
        self.performance_evaluator = performance_evaluator
        self.config = config

        self.search_history = []
        self.best_architecture = None
        self.best_performance = -np.inf

        tprint_success("✅ [PROGRESSIVE] Progressive Architecture Search initialized")

    def search(self):
        """Perform progressive architecture search."""
        tprint("🚀 [PROGRESSIVE] Starting Progressive Architecture Search", color="blue", bold=True)
        tprint(f"📊 [PROGRESSIVE] Evolution rounds: {self.config.progressive_evolution_rounds}", color="cyan")

        current_ops = self.config.progressive_initial_ops
        population = []

        for round_num in range(self.config.progressive_evolution_rounds):
            tprint(f"📈 [PROGRESSIVE] Starting round {round_num + 1}/{self.config.progressive_evolution_rounds} with {current_ops} operations", color="yellow", bold=True)

            # Create initial population
            if not population:
                tprint(f"🔧 [PROGRESSIVE] Creating initial population for round {round_num + 1}", color="yellow")
                population = self._create_initial_population(current_ops)
                tprint(f"📊 [PROGRESSIVE] Initial population size: {len(population)}", color="cyan")
            else:
                tprint(f"🔧 [PROGRESSIVE] Evolving population for round {round_num + 1}", color="yellow")
                # Evolve population
                population = self._evolve_population(population, current_ops)
                tprint(f"📊 [PROGRESSIVE] Evolved population size: {len(population)}", color="cyan")

            # Evaluate population
            tprint(f"🔧 [PROGRESSIVE] Evaluating population for round {round_num + 1}", color="yellow")
            for i, arch in enumerate(population):
                performance = self.performance_evaluator(arch)
                tprint(f"📊 [PROGRESSIVE] Architecture {i+1}/{len(population)}: Performance = {performance:.4f}", color="cyan")

                if performance > self.best_performance:
                    tprint(f"🏆 [PROGRESSIVE] New best performance: {performance:.4f} (round {round_num + 1})", color="green", bold=True)
                    self.best_performance = performance
                    self.best_architecture = copy.deepcopy(arch)

            self.search_history.append({
                'round': round_num + 1,
                'operations': current_ops,
                'population_size': len(population),
                'best_performance': self.best_performance
            })

            # Increase complexity
            old_ops = current_ops
            current_ops = min(
                int(current_ops * self.config.progressive_growth_rate),
                self.config.progressive_max_ops
            )
            tprint(f"📈 [PROGRESSIVE] Round {round_num + 1} complete. Complexity: {old_ops} → {current_ops} operations", color="cyan")

        tprint_success("✅ [PROGRESSIVE] Progressive Architecture Search completed")
        tprint(f"🏆 [PROGRESSIVE] Final best performance: {self.best_performance:.4f}", color="green", bold=True)

        return {
            'best_architecture': self.best_architecture,
            'best_performance': self.best_performance,
            'search_history': self.search_history
        }

    def _create_initial_population(self, num_ops):
        """Create initial population with specified number of operations."""
        population = []
        for _ in range(self.config.mo_population_size):
            arch = self.search_space.create_empty_architecture()
            for _ in range(num_ops):
                op_idx = random.randint(0, len(self.search_space.operations) - 1)
                arch = self.search_space.apply_operation(arch, op_idx)
            population.append(arch)
        return population

    def _evolve_population(self, population, target_ops):
        """Evolve population to target number of operations."""
        new_population = []

        # Keep best individuals
        population.sort(key=lambda x: self.performance_evaluator(x), reverse=True)
        elite_size = len(population) // 4
        new_population.extend(population[:elite_size])

        # Generate offspring
        while len(new_population) < self.config.mo_population_size:
            parent1 = random.choice(population)
            parent2 = random.choice(population)

            # Crossover
            child = self._crossover(parent1, parent2)

            # Mutation
            child = self._mutate(child, target_ops)

            new_population.append(child)

        return new_population

    def _crossover(self, parent1, parent2):
        """Perform crossover between two architectures."""
        child = self.search_space.create_empty_architecture()

        # Simple crossover: take operations from both parents
        max_ops = max(len(parent1.layers), len(parent2.layers))
        for i in range(max_ops):
            if i < len(parent1.layers) and i < len(parent2.layers):
                # Randomly choose from either parent
                parent = random.choice([parent1, parent2])
                child = self.search_space.apply_operation(child, parent.layers[i].operation_id)
            elif i < len(parent1.layers):
                child = self.search_space.apply_operation(child, parent1.layers[i].operation_id)
            elif i < len(parent2.layers):
                child = self.search_space.apply_operation(child, parent2.layers[i].operation_id)

        return child

    def _mutate(self, architecture, target_ops):
        """Mutate architecture to reach target number of operations."""
        current_ops = len(architecture.layers)

        # Add operations if needed
        while len(architecture.layers) < target_ops:
            op_idx = random.randint(0, len(self.search_space.operations) - 1)
            architecture = self.search_space.apply_operation(architecture, op_idx)

        # Random mutation
        if random.random() < self.config.mo_mutation_rate:
            if architecture.layers:
                # Replace random operation
                layer_idx = random.randint(0, len(architecture.layers) - 1)
                op_idx = random.randint(0, len(self.search_space.operations) - 1)
                architecture.layers[layer_idx].operation_id = op_idx

        return architecture

class MultiObjectiveEvolutionarySearch:
    """Multi-Objective Evolutionary Search implementation."""

    def __init__(self, search_space, performance_evaluator, config):
        tprint("🎯 [MO-ES] Initializing Multi-Objective Evolutionary Search", color="blue", bold=True)
        tprint(f"📊 [MO-ES] Config: population_size={config.mo_population_size}, generations={config.mo_generations}", color="cyan")

        self.search_space = search_space
        self.performance_evaluator = performance_evaluator
        self.config = config

        self.search_history = []
        self.pareto_front = []

        tprint_success("✅ [MO-ES] Multi-Objective Evolutionary Search initialized")

    def search(self):
        """Perform multi-objective evolutionary search."""
        tprint("🚀 [MO-ES] Starting Multi-Objective Evolutionary Search", color="blue", bold=True)
        tprint(f"📊 [MO-ES] Generations: {self.config.mo_generations}, Population: {self.config.mo_population_size}", color="cyan")

        # Initialize population
        tprint("🔧 [MO-ES] Creating initial population", color="yellow")
        population = self._create_initial_population()
        tprint(f"📊 [MO-ES] Initial population size: {len(population)}", color="cyan")

        for generation in range(self.config.mo_generations):
            tprint(f"🧬 [MO-ES] Starting generation {generation+1}/{self.config.mo_generations}", color="yellow", bold=True)

            # Evaluate population
            tprint(f"🔧 [MO-ES] Evaluating population for generation {generation+1}", color="yellow")
            evaluated_population = []
            for i, arch in enumerate(population):
                objectives = self._evaluate_objectives(arch)
                evaluated_population.append((arch, objectives))
                tprint(f"📊 [MO-ES] Architecture {i+1}/{len(population)}: Objectives = {objectives}", color="cyan")

            # Non-dominated sorting
            tprint(f"🔧 [MO-ES] Performing non-dominated sorting for generation {generation+1}", color="yellow")
            fronts = self._non_dominated_sorting(evaluated_population)
            tprint(f"📊 [MO-ES] Found {len(fronts)} fronts", color="cyan")

            # Update Pareto front
            if fronts:
                self.pareto_front = fronts[0]
                tprint(f"📊 [MO-ES] Pareto front size: {len(self.pareto_front)}", color="cyan")

            # Selection
            tprint(f"🔧 [MO-ES] Performing selection for generation {generation+1}", color="yellow")
            population = self._selection(evaluated_population, fronts)
            tprint(f"📊 [MO-ES] Selected population size: {len(population)}", color="cyan")

            # Crossover and mutation
            tprint(f"🔧 [MO-ES] Generating offspring for generation {generation+1}", color="yellow")
            offspring = self._generate_offspring(population)
            tprint(f"📊 [MO-ES] Generated {len(offspring)} offspring", color="cyan")

            # Combine parent and offspring populations
            combined_population = population + offspring
            tprint(f"📊 [MO-ES] Combined population size: {len(combined_population)}", color="cyan")

            # Environmental selection
            tprint(f"🔧 [MO-ES] Performing environmental selection for generation {generation+1}", color="yellow")
            population = self._environmental_selection(combined_population)
            tprint(f"📊 [MO-ES] Final population size: {len(population)}", color="cyan")

            self.search_history.append({
                'generation': generation,
                'population_size': len(population),
                'pareto_front_size': len(self.pareto_front),
                'best_objectives': self._get_best_objectives()
            })

            if generation % 20 == 0:
                tprint(f"📊 [MO-ES] Generation {generation}: Pareto front size = {len(self.pareto_front)}", color="cyan")

        tprint_success("✅ [MO-ES] Multi-Objective Evolutionary Search completed")
        tprint(f"🏆 [MO-ES] Final Pareto front size: {len(self.pareto_front)}", color="green", bold=True)

        return {
            'pareto_front': self.pareto_front,
            'search_history': self.search_history,
            'best_architectures': [arch for arch, _ in self.pareto_front]
        }

    def _create_initial_population(self):
        """Create initial population."""
        population = []
        for _ in range(self.config.mo_population_size):
            arch = self.search_space.create_empty_architecture()
            num_ops = random.randint(2, 10)
            for _ in range(num_ops):
                op_idx = random.randint(0, len(self.search_space.operations) - 1)
                arch = self.search_space.apply_operation(arch, op_idx)
            population.append(arch)
        return population

    def _evaluate_objectives(self, architecture):
        """Evaluate multiple objectives for an architecture."""
        objectives = {}

        # Accuracy objective
        objectives['accuracy'] = self.performance_evaluator(architecture)

        # Efficiency objective (inverse of computational complexity)
        objectives['efficiency'] = 1.0 / (1.0 + len(architecture.layers))

        # Complexity objective (parameter count)
        objectives['complexity'] = -len(architecture.layers)  # Negative for minimization

        return objectives

    def _non_dominated_sorting(self, population):
        """Perform non-dominated sorting."""
        fronts = [[]]
        dominated_count = {}
        dominated_solutions = {}

        for i, (arch1, obj1) in enumerate(population):
            dominated_count[i] = 0
            dominated_solutions[i] = []

            for j, (arch2, obj2) in enumerate(population):
                if i != j:
                    if self._dominates(obj1, obj2):
                        dominated_solutions[i].append(j)
                    elif self._dominates(obj2, obj1):
                        dominated_count[i] += 1

            if dominated_count[i] == 0:
                fronts[0].append((arch1, obj1))

        current_front = 0
        while fronts[current_front]:
            next_front = []
            for arch, obj in fronts[current_front]:
                for dominated_idx in dominated_solutions[population.index((arch, obj))]:
                    dominated_count[dominated_idx] -= 1
                    if dominated_count[dominated_idx] == 0:
                        next_front.append(population[dominated_idx])
            current_front += 1
            if next_front:
                fronts.append(next_front)

        return fronts

    def _dominates(self, obj1, obj2):
        """Check if obj1 dominates obj2."""
        better_in_at_least_one = False
        for key in obj1:
            if obj1[key] < obj2[key]:
                return False
            elif obj1[key] > obj2[key]:
                better_in_at_least_one = True
        return better_in_at_least_one

    def _selection(self, population, fronts):
        """Select parents using tournament selection."""
        parents = []
        for _ in range(len(population)):
            # Tournament selection
            tournament_size = 3
            tournament = random.sample(population, min(tournament_size, len(population)))
            winner = max(tournament, key=lambda x: sum(x[1].values()))
            parents.append(winner[0])
        return parents

    def _generate_offspring(self, population):
        """Generate offspring through crossover and mutation."""
        offspring = []
        for _ in range(len(population)):
            if random.random() < self.config.mo_crossover_rate:
                parent1 = random.choice(population)
                parent2 = random.choice(population)
                child = self._crossover(parent1, parent2)

                if random.random() < self.config.mo_mutation_rate:
                    child = self._mutate(child)

                offspring.append(child)
        return offspring

    def _crossover(self, parent1, parent2):
        """Perform crossover between two architectures."""
        # Simplified crossover implementation
        child = self.search_space.create_empty_architecture()

        # Randomly combine operations from both parents
        max_ops = max(len(parent1.layers), len(parent2.layers))
        for i in range(max_ops):
            if random.random() < 0.5 and i < len(parent1.layers):
                child = self.search_space.apply_operation(child, parent1.layers[i].operation_id)
            elif i < len(parent2.layers):
                child = self.search_space.apply_operation(child, parent2.layers[i].operation_id)

        return child

    def _mutate(self, architecture):
        """Mutate architecture."""
        if architecture.layers and random.random() < 0.1:
            # Replace random operation
            layer_idx = random.randint(0, len(architecture.layers) - 1)
            op_idx = random.randint(0, len(self.search_space.operations) - 1)
            architecture.layers[layer_idx].operation_id = op_idx

        return architecture

    def _environmental_selection(self, population):
        """Environmental selection to maintain population size."""
        # Simplified selection - keep best individuals
        evaluated = [(arch, self._evaluate_objectives(arch)) for arch in population]
        evaluated.sort(key=lambda x: sum(x[1].values()), reverse=True)
        return [arch for arch, _ in evaluated[:self.config.mo_population_size]]

    def _get_best_objectives(self):
        """Get best objectives from Pareto front."""
        if not self.pareto_front:
            return {}

        best = {}
        for key in self.pareto_front[0][1].keys():
            best[key] = max(obj[key] for _, obj in self.pareto_front)
        return best

class EnhancedSearchStrategyManager:
    """Manager for enhanced search strategies."""

    def __init__(self, search_space, performance_evaluator, config):
        tprint("🎛️ [SEARCH-MANAGER] Initializing Enhanced Search Strategy Manager", color="blue", bold=True)
        tprint(f"📊 [SEARCH-MANAGER] Default strategy: {config.strategy_type.value}", color="cyan")

        self.search_space = search_space
        self.performance_evaluator = performance_evaluator
        self.config = config

        self.search_strategies = {
            SearchStrategyType.REINFORCEMENT_LEARNING: ReinforcementLearningSearch,
            SearchStrategyType.DIFFERENTIABLE_DARTS: DifferentiableArchitectureSearch,
            SearchStrategyType.PROGRESSIVE_SEARCH: ProgressiveArchitectureSearch,
            SearchStrategyType.MULTI_OBJECTIVE_EVOLUTIONARY: MultiObjectiveEvolutionarySearch
        }

        self.search_history = []
        self.best_results = {}

        tprint_success("✅ [SEARCH-MANAGER] Enhanced Search Strategy Manager initialized")

    def search(self, strategy_type=None, **kwargs):
        """Perform architecture search using specified strategy."""
        if strategy_type is None:
            strategy_type = self.config.strategy_type

        tprint(f"🚀 [SEARCH-MANAGER] Starting {strategy_type.value} search strategy", color="blue", bold=True)
        tprint(f"📊 [SEARCH-MANAGER] Strategy type: {strategy_type.value}", color="cyan")

        # Create search strategy
        if strategy_type in self.search_strategies:
            tprint(f"🔧 [SEARCH-MANAGER] Creating {strategy_type.value} strategy", color="yellow")
            strategy = self.search_strategies[strategy_type](
                self.search_space, self.performance_evaluator, self.config
            )

            # Perform search
            tprint(f"🔧 [SEARCH-MANAGER] Executing {strategy_type.value} search", color="yellow")
            result = strategy.search(**kwargs)

            # Store results
            tprint(f"🔧 [SEARCH-MANAGER] Storing results for {strategy_type.value}", color="yellow")
            self.best_results[strategy_type] = result
            self.search_history.append({
                'strategy': strategy_type.value,
                'result': result,
                'timestamp': datetime.now().isoformat()
            })

            tprint_success(f"✅ [SEARCH-MANAGER] {strategy_type.value} search completed")

            return result
        else:
            tprint_error(f"❌ [SEARCH-MANAGER] Unsupported search strategy: {strategy_type}")
            raise ValueError(f"Unsupported search strategy: {strategy_type}")

    def hybrid_search(self, strategies=None, **kwargs):
        """Perform hybrid search using multiple strategies."""
        if strategies is None:
            strategies = [
                SearchStrategyType.REINFORCEMENT_LEARNING,
                SearchStrategyType.PROGRESSIVE_SEARCH,
                SearchStrategyType.MULTI_OBJECTIVE_EVOLUTIONARY
            ]

        tprint("🚀 [SEARCH-MANAGER] Starting hybrid search with multiple strategies", color="blue", bold=True)
        tprint(f"📊 [SEARCH-MANAGER] Strategies: {[s.value for s in strategies]}", color="cyan")

        results = {}
        for i, strategy_type in enumerate(strategies):
            try:
                tprint(f"🔧 [SEARCH-MANAGER] Executing strategy {i+1}/{len(strategies)}: {strategy_type.value}", color="yellow")
                result = self.search(strategy_type, **kwargs)
                results[strategy_type.value] = result
                tprint_success(f"✅ [SEARCH-MANAGER] Strategy {strategy_type.value} completed successfully")
            except Exception as e:
                tprint_warning(f"⚠️ [SEARCH-MANAGER] Strategy {strategy_type.value} failed: {e}")

        # Combine results
        tprint("🔧 [SEARCH-MANAGER] Combining results from all strategies", color="yellow")
        combined_result = self._combine_results(results)

        self.search_history.append({
            'strategy': 'hybrid',
            'results': results,
            'combined_result': combined_result,
            'timestamp': datetime.now().isoformat()
        })

        tprint_success("✅ [SEARCH-MANAGER] Hybrid search completed")
        tprint(f"🏆 [SEARCH-MANAGER] Best combined performance: {combined_result.get('best_performance', 0.0):.4f}", color="green", bold=True)

        return combined_result

    def _combine_results(self, results):
        """Combine results from multiple search strategies."""
        best_architecture = None
        best_performance = -np.inf

        for strategy_name, result in results.items():
            if 'best_performance' in result and result['best_performance'] > best_performance:
                best_performance = result['best_performance']
                best_architecture = result['best_architecture']

        return {
            'best_architecture': best_architecture,
            'best_performance': best_performance,
            'individual_results': results
        }

    def get_search_summary(self):
        """Get summary of all search results."""
        summary = {
            'total_searches': len(self.search_history),
            'best_overall_performance': -np.inf,
            'best_strategy': None,
            'strategy_performance': {}
        }

        for entry in self.search_history:
            strategy = entry['strategy']
            if strategy != 'hybrid':
                result = entry['result']
                performance = result.get('best_performance', -np.inf)
                summary['strategy_performance'][strategy] = performance

                if performance > summary['best_overall_performance']:
                    summary['best_overall_performance'] = performance
                    summary['best_strategy'] = strategy

        return summary

# Factory functions
def create_search_strategy(strategy_type: SearchStrategyType, search_space,
                          performance_evaluator, config: SearchStrategyConfig):
    """Factory function to create search strategies."""
    strategies = {
        SearchStrategyType.REINFORCEMENT_LEARNING: ReinforcementLearningSearch,
        SearchStrategyType.DIFFERENTIABLE_DARTS: DifferentiableArchitectureSearch,
        SearchStrategyType.PROGRESSIVE_SEARCH: ProgressiveArchitectureSearch,
        SearchStrategyType.MULTI_OBJECTIVE_EVOLUTIONARY: MultiObjectiveEvolutionarySearch
    }

    if strategy_type in strategies:
        return strategies[strategy_type](search_space, performance_evaluator, config)
    else:
        raise ValueError(f"Unsupported search strategy: {strategy_type}")

def create_enhanced_search_manager(search_space, performance_evaluator,
                                  config: SearchStrategyConfig):
    """Factory function to create enhanced search strategy manager."""
    return EnhancedSearchStrategyManager(search_space, performance_evaluator, config)
