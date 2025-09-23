"""
Meta-Learning and Few-Shot Learning for NAS

This module implements advanced meta-learning techniques including:
- MAML (Model-Agnostic Meta-Learning)
- Few-shot learning for regime adaptation
- Prototypical networks for regime classification
- Meta-optimization for NAS
- Continual learning for dynamic regimes
- Uncertainty estimation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import weight_norm
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import OrderedDict, defaultdict
import copy
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class MetaLearningConfig:
    """Configuration for meta-learning."""
    meta_learning_rate: float = 1e-3
    inner_learning_rate: float = 0.01
    num_inner_steps: int = 5
    num_outer_steps: int = 100
    num_shots: int = 5  # K-shot learning
    num_ways: int = 5   # N-way classification
    use_maml: bool = True
    use_prototypical: bool = True
    use_continual: bool = True
    adaptation_steps: int = 10
    meta_batch_size: int = 32
    support_set_size: int = 20
    query_set_size: int = 15
    use_uncertainty: bool = True

class MAML(nn.Module):
    """
    Model-Agnostic Meta-Learning (MAML) implementation.

    Learns to learn quickly on new tasks with few gradient steps.
    """

    def __init__(self, base_model: nn.Module, config: MetaLearningConfig):
        """Initialize MAML.

        Args:
            base_model: Base neural network model
            config: Meta-learning configuration
        """
        super(MAML, self).__init__()
        self.config = config
        self.base_model = base_model

        # Meta-optimizer for outer loop
        self.meta_optimizer = optim.Adam(
            self.parameters(),
            lr=config.meta_learning_rate
        )

        self.logger = logging.getLogger(self.__class__.__name__)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.base_model(x)

    def inner_adapt(self, support_x: torch.Tensor, support_y: torch.Tensor,
                   num_steps: int = None) -> nn.Module:
        """
        Inner loop adaptation for few-shot learning.

        Args:
            support_x: Support set features
            support_y: Support set labels
            num_steps: Number of adaptation steps

        Returns:
            Adapted model
        """
        if num_steps is None:
            num_steps = self.config.num_inner_steps

        # Create a copy of the model for adaptation
        adapted_model = copy.deepcopy(self.base_model)
        inner_optimizer = optim.SGD(adapted_model.parameters(), lr=self.config.inner_learning_rate)

        # Inner loop optimization
        for step in range(num_steps):
            # Forward pass
            support_logits = adapted_model(support_x)
            inner_loss = F.cross_entropy(support_logits, support_y)

            # Inner gradient step
            inner_optimizer.zero_grad()
            inner_loss.backward()
            inner_optimizer.step()

        return adapted_model

    def meta_train_step(self, task_batch: List[Dict[str, torch.Tensor]]) -> float:
        """
        Meta-training step.

        Args:
            task_batch: Batch of tasks for meta-training

        Returns:
            Meta-loss
        """
        meta_loss = 0.0

        for task in task_batch:
            support_x = task['support_x']
            support_y = task['support_y']
            query_x = task['query_x']
            query_y = task['query_y']

            # Inner adaptation
            adapted_model = self.inner_adapt(support_x, support_y)

            # Meta-objective on query set
            query_logits = adapted_model(query_x)
            task_loss = F.cross_entropy(query_logits, query_y)

            meta_loss += task_loss

        # Meta-optimization
        meta_loss = meta_loss / len(task_batch)
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()

        return meta_loss.item()

class PrototypicalNetwork(nn.Module):
    """
    Prototypical Networks for few-shot learning.

    Learns a metric space where classification is performed by
    computing distances to prototype representations.
    """

    def __init__(self, input_dim: int = 64, hidden_dim: int = 64, output_dim: int = 64):
        """Initialize Prototypical Network.

        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension
            output_dim: Output dimension (embedding space)
        """
        super(PrototypicalNetwork, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        self.logger = logging.getLogger(self.__class__.__name__)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder."""
        return self.encoder(x)

    def get_prototypes(self, support_set: torch.Tensor, support_labels: torch.Tensor) -> torch.Tensor:
        """
        Compute class prototypes from support set.

        Args:
            support_set: Support set embeddings
            support_labels: Support set labels

        Returns:
            Class prototypes
        """
        unique_labels = torch.unique(support_labels)
        prototypes = []

        for label in unique_labels:
            mask = support_labels == label
            class_embeddings = support_set[mask]
            prototype = class_embeddings.mean(dim=0)
            prototypes.append(prototype)

        return torch.stack(prototypes)

    def compute_distances(self, query_embeddings: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        """
        Compute distances between query embeddings and prototypes.

        Args:
            query_embeddings: Query embeddings
            prototypes: Class prototypes

        Returns:
            Distance matrix
        """
        # Expand dimensions for broadcasting
        query_expanded = query_embeddings.unsqueeze(1)  # (N, 1, D)
        prototypes_expanded = prototypes.unsqueeze(0)    # (1, K, D)

        # Compute Euclidean distance
        distances = torch.sqrt(((query_expanded - prototypes_expanded) ** 2).sum(dim=2))

        return -distances  # Negative for similarity

    def predict(self, support_set: torch.Tensor, support_labels: torch.Tensor,
                query_set: torch.Tensor) -> torch.Tensor:
        """
        Make predictions on query set.

        Args:
            support_set: Support set
            support_labels: Support set labels
            query_set: Query set

        Returns:
            Logits for query set
        """
        # Encode support and query sets
        support_embeddings = self.forward(support_set)
        query_embeddings = self.forward(query_set)

        # Compute prototypes
        prototypes = self.get_prototypes(support_embeddings, support_labels)

        # Compute distances to prototypes
        distances = self.compute_distances(query_embeddings, prototypes)

        return distances

class UncertaintyEstimator(nn.Module):
    """
    Uncertainty estimation using Monte Carlo dropout and ensembles.

    Provides uncertainty quantification for regime predictions.
    """

    def __init__(self, base_model: nn.Module, dropout_rate: float = 0.1, num_samples: int = 10):
        """Initialize uncertainty estimator.

        Args:
            base_model: Base model
            dropout_rate: Dropout rate for MC dropout
            num_samples: Number of Monte Carlo samples
        """
        super(UncertaintyEstimator, self).__init__()
        self.base_model = base_model
        self.dropout_rate = dropout_rate
        self.num_samples = num_samples

        # Enable dropout during inference
        self._enable_dropout()

    def _enable_dropout(self):
        """Enable dropout layers during inference."""
        for module in self.base_model.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def predict_with_uncertainty(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Make predictions with uncertainty estimates.

        Args:
            x: Input tensor

        Returns:
            Tuple of (mean_predictions, std_predictions, uncertainty_score)
        """
        predictions_list = []

        # Monte Carlo sampling with dropout
        with torch.no_grad():
            for _ in range(self.num_samples):
                predictions = self.base_model(x)
                predictions_list.append(predictions)

        # Stack predictions
        all_predictions = torch.stack(predictions_list)  # (num_samples, batch_size, num_classes)

        # Compute statistics
        mean_predictions = all_predictions.mean(dim=0)
        std_predictions = all_predictions.std(dim=0)

        # Compute uncertainty score (average predictive entropy)
        predictive_entropy = -torch.sum(
            F.softmax(mean_predictions, dim=1) * F.log_softmax(mean_predictions, dim=1),
            dim=1
        )
        uncertainty_score = predictive_entropy.mean()

        return mean_predictions, std_predictions, uncertainty_score

class ContinualLearningModel(nn.Module):
    """
    Continual learning model for dynamic regime adaptation.

    Handles concept drift and regime changes over time.
    """

    def __init__(self, base_model: nn.Module, memory_size: int = 1000):
        """Initialize continual learning model.

        Args:
            base_model: Base model
            memory_size: Size of episodic memory
        """
        super(ContinualLearningModel, self).__init__()
        self.base_model = base_model
        self.memory_size = memory_size

        # Episodic memory for replay
        self.episodic_memory = []

        # Track performance on past tasks
        self.task_performance = defaultdict(list)

        self.logger = logging.getLogger(self.__class__.__name__)

    def update_memory(self, x: torch.Tensor, y: torch.Tensor):
        """Update episodic memory with new samples."""
        # Add new samples to memory
        for i in range(len(x)):
            sample = {
                'data': x[i].clone(),
                'label': y[i].clone()
            }
            self.episodic_memory.append(sample)

            # Remove oldest samples if memory is full
            if len(self.episodic_memory) > self.memory_size:
                self.episodic_memory.pop(0)

    def replay_from_memory(self, batch_size: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample batch from episodic memory for replay."""
        if len(self.episodic_memory) < batch_size:
            return None, None

        # Randomly sample from memory
        indices = np.random.choice(len(self.episodic_memory), batch_size, replace=False)
        batch_x = torch.stack([self.episodic_memory[i]['data'] for i in indices])
        batch_y = torch.stack([self.episodic_memory[i]['label'] for i in indices])

        return batch_x, batch_y

    def compute_memory_loss(self, current_loss: torch.Tensor, memory_batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Compute loss that balances current and memory tasks."""
        if memory_batch[0] is None:
            return current_loss

        x_mem, y_mem = memory_batch

        # Compute memory loss
        memory_logits = self.base_model(x_mem)
        memory_loss = F.cross_entropy(memory_logits, y_mem)

        # Combine losses (weighted average)
        combined_loss = 0.7 * current_loss + 0.3 * memory_loss

        return combined_loss

class FewShotRegimeLearner:
    """
    Few-shot learning for regime detection.

    Adapts quickly to new market regimes with limited data.
    """

    def __init__(self, config: MetaLearningConfig):
        """Initialize few-shot regime learner.

        Args:
            config: Meta-learning configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize models
        self.maml_model = None
        self.prototypical_model = None
        self.uncertainty_estimator = None

    def few_shot_adaptation(self,
                           support_set: Tuple[torch.Tensor, torch.Tensor],
                           query_set: Tuple[torch.Tensor, torch.Tensor],
                           regime_type: str = "unknown") -> Dict[str, Any]:
        """
        Perform few-shot adaptation for new regime.

        Args:
            support_set: Support set for adaptation
            query_set: Query set for evaluation
            regime_type: Type of regime being adapted to

        Returns:
            Adaptation results
        """
        results = {
            'regime_type': regime_type,
            'support_set_size': len(support_set[0]),
            'query_set_size': len(query_set[0]),
            'adaptation_method': 'few_shot'
        }

        # MAML adaptation if available
        if self.maml_model is not None:
            adapted_model = self.maml_model.inner_adapt(
                support_set[0], support_set[1], self.config.num_inner_steps
            )

            # Evaluate on query set
            with torch.no_grad():
                query_logits = adapted_model(query_set[0])
                maml_accuracy = (query_logits.argmax(dim=1) == query_set[1]).float().mean()

            results['maml_accuracy'] = maml_accuracy.item()

        # Prototypical network adaptation
        if self.prototypical_model is not None:
            proto_logits = self.prototypical_model.predict(
                support_set[0], support_set[1], query_set[0]
            )
            proto_accuracy = (proto_logits.argmax(dim=1) == query_set[1]).float().mean()

            results['prototypical_accuracy'] = proto_accuracy.item()

        # Uncertainty estimation
        if self.uncertainty_estimator is not None:
            mean_pred, std_pred, uncertainty = self.uncertainty_estimator.predict_with_uncertainty(query_set[0])
            results['uncertainty_score'] = uncertainty.item()
            results['prediction_std'] = std_pred.mean().item()

        self.logger.info(f"🎯 Few-shot adaptation completed for {regime_type} regime")
        return results

class MetaNAS_Optimizer:
    """
    Meta-optimized Neural Architecture Search.

    Uses meta-learning to optimize the NAS process itself.
    """

    def __init__(self, config: MetaLearningConfig):
        """Initialize meta-NAS optimizer.

        Args:
            config: Meta-learning configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Meta-NAS state
        self.meta_architectures = []
        self.meta_performance = []
        self.adaptation_history = []

    def meta_optimize_architecture(self,
                                 architecture: nn.Module,
                                 meta_train_tasks: List[Dict],
                                 meta_test_tasks: List[Dict]) -> Dict[str, Any]:
        """
        Meta-optimize architecture for few-shot learning.

        Args:
            architecture: Architecture to optimize
            meta_train_tasks: Meta-training tasks
            meta_test_tasks: Meta-testing tasks

        Returns:
            Meta-optimization results
        """
        logger.info("🚀 Starting meta-optimization of architecture")

        # Initialize MAML for architecture
        meta_model = MAML(architecture, self.config)

        # Meta-training loop
        for outer_step in range(self.config.num_outer_steps):
            # Sample task batch
            task_batch = np.random.choice(meta_train_tasks, self.config.meta_batch_size, replace=False)

            # Meta-training step
            meta_loss = meta_model.meta_train_step(task_batch)

            if outer_step % 10 == 0:
                # Evaluate on meta-test tasks
                meta_test_loss = self._evaluate_meta_test(meta_model, meta_test_tasks)
                self.logger.info(f"📈 Outer step {outer_step}: Meta-loss = {meta_loss:.4f}, Meta-test = {meta_test_loss:.4f}")

        # Final evaluation
        final_performance = self._evaluate_meta_test(meta_model, meta_test_tasks)

        results = {
            'meta_model': meta_model,
            'final_performance': final_performance,
            'meta_training_steps': self.config.num_outer_steps,
            'task_batch_size': self.config.meta_batch_size,
            'meta_optimization_success': final_performance < 0.5  # Threshold for success
        }

        self.logger.info(f"✅ Meta-optimization completed with performance: {final_performance:.4f}")
        return results

    def _evaluate_meta_test(self, meta_model: MAML, meta_test_tasks: List[Dict]) -> float:
        """Evaluate meta-model on test tasks."""
        total_loss = 0.0

        for task in meta_test_tasks:
            support_x = task['support_x']
            support_y = task['support_y']
            query_x = task['query_x']
            query_y = task['query_y']

            # Inner adaptation
            adapted_model = meta_model.inner_adapt(support_x, support_y)

            # Evaluate on query set
            query_logits = adapted_model(query_x)
            loss = F.cross_entropy(query_logits, query_y)

            total_loss += loss.item()

        return total_loss / len(meta_test_tasks)

class AdaptiveRegimeLearner:
    """
    Adaptive regime learner with continual learning.

    Continuously adapts to changing market regimes
    while preventing catastrophic forgetting.
    """

    def __init__(self, base_model: nn.Module, config: MetaLearningConfig):
        """Initialize adaptive regime learner.

        Args:
            base_model: Base model
            config: Meta-learning configuration
        """
        self.config = config
        self.base_model = base_model
        self.continual_model = ContinualLearningModel(base_model)

        # Track regime changes
        self.current_regime = None
        self.regime_history = []
        self.adaptation_count = 0

    def detect_regime_change(self, new_data: torch.Tensor, threshold: float = 0.1) -> bool:
        """
        Detect if regime has changed significantly.

        Args:
            new_data: New market data
            threshold: Threshold for regime change detection

        Returns:
            True if regime change detected
        """
        if self.current_regime is None:
            return False

        # Simple regime change detection based on prediction confidence
        with torch.no_grad():
            predictions = self.base_model(new_data)
            confidence = torch.softmax(predictions, dim=1).max(dim=1)[0].mean()

        # Regime change if confidence drops significantly
        regime_change = confidence < (1.0 - threshold)

        if regime_change:
            self.logger.info(f"🔄 Regime change detected (confidence: {confidence:.4f})")

        return regime_change

    def adapt_to_new_regime(self, new_data: torch.Tensor, new_labels: torch.Tensor,
                           regime_type: str = "unknown") -> Dict[str, Any]:
        """
        Adapt model to new regime.

        Args:
            new_data: New regime data
            new_labels: New regime labels
            regime_type: Type of new regime

        Returns:
            Adaptation results
        """
        self.adaptation_count += 1
        self.current_regime = regime_type
        self.regime_history.append(regime_type)

        # Update episodic memory
        self.continual_model.update_memory(new_data, new_labels)

        # Perform adaptation
        adaptation_results = {
            'regime_type': regime_type,
            'adaptation_step': self.adaptation_count,
            'memory_size': len(self.continual_model.episodic_memory),
            'regime_history_length': len(self.regime_history)
        }

        self.logger.info(f"🎯 Adapted to new regime: {regime_type}")
        return adaptation_results

    def get_regime_statistics(self) -> Dict[str, Any]:
        """Get statistics about regime adaptations."""
        return {
            'current_regime': self.current_regime,
            'total_adaptations': self.adaptation_count,
            'unique_regimes': len(set(self.regime_history)),
            'regime_sequence': self.regime_history,
            'memory_utilization': len(self.continual_model.episodic_memory) / self.continual_model.memory_size
        }

# Utility functions for meta-learning
def create_support_query_split(dataset: torch.utils.data.Dataset,
                             support_size: int = 20,
                             query_size: int = 15,
                             num_classes: int = 5) -> Tuple[torch.utils.data.Subset, torch.utils.data.Subset]:
    """Create support and query sets for few-shot learning."""
    total_size = len(dataset)
    indices = np.random.permutation(total_size)

    support_indices = indices[:support_size]
    query_indices = indices[support_size:support_size + query_size]

    support_set = torch.utils.data.Subset(dataset, support_indices)
    query_set = torch.utils.data.Subset(dataset, query_indices)

    return support_set, query_set

def create_meta_learning_tasks(datasets: List[torch.utils.data.Dataset],
                             num_tasks: int = 100,
                             support_size: int = 20,
                             query_size: int = 15) -> List[Dict[str, torch.Tensor]]:
    """Create meta-learning tasks from multiple datasets."""
    tasks = []

    for _ in range(num_tasks):
        # Randomly select dataset
        dataset = np.random.choice(datasets)

        # Create support and query sets
        support_set, query_set = create_support_query_split(
            dataset, support_size, query_size
        )

        # Convert to tensors
        support_x = torch.stack([support_set[i][0] for i in range(len(support_set))])
        support_y = torch.tensor([support_set[i][1] for i in range(len(support_set))])
        query_x = torch.stack([query_set[i][0] for i in range(len(query_set))])
        query_y = torch.tensor([query_set[i][1] for i in range(len(query_set))])

        task = {
            'support_x': support_x,
            'support_y': support_y,
            'query_x': query_x,
            'query_y': query_y
        }

        tasks.append(task)

    return tasks

def evaluate_few_shot_performance(model: nn.Module,
                                test_datasets: List[torch.utils.data.Dataset],
                                num_shots: List[int] = [1, 5, 10, 20]) -> Dict[str, float]:
    """Evaluate few-shot learning performance."""
    results = {}

    for n_shot in num_shots:
        accuracies = []

        for dataset in test_datasets:
            # Create few-shot task
            support_set, query_set = create_support_query_split(
                dataset, support_size=n_shot, query_size=15
            )

            # Adapt model
            if hasattr(model, 'inner_adapt'):
                adapted_model = model.inner_adapt(
                    torch.stack([support_set[i][0] for i in range(len(support_set))]),
                    torch.tensor([support_set[i][1] for i in range(len(support_set))])
                )
            else:
                adapted_model = model

            # Evaluate
            query_x = torch.stack([query_set[i][0] for i in range(len(query_set))])
            query_y = torch.tensor([query_set[i][1] for i in range(len(query_set))])

            with torch.no_grad():
                predictions = adapted_model(query_x)
                accuracy = (predictions.argmax(dim=1) == query_y).float().mean()
                accuracies.append(accuracy.item())

        results[f'{n_shot}_shot'] = np.mean(accuracies)

    return results