"""
Unified Meta-Learning Utilities

This module provides comprehensive meta-learning capabilities that can be shared
between TAS and NAS architectures, enabling rapid adaptation, few-shot learning,
and continual learning across different market regimes.
"""

import logging
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict, deque
import pickle
from pathlib import Path

from .unified_architecture_config import ArchitectureType, OptimizationObjective

logger = logging.getLogger(__name__)


class MetaLearningMethod(Enum):
    """Types of meta-learning methods."""
    MAML = "maml"  # Model-Agnostic Meta-Learning
    PROTONET = "protonet"  # Prototypical Networks
    META_SGD = "meta_sgd"  # Meta-SGD
    REPTILE = "reptile"  # Reptile
    FOMAML = "fomaml"  # First-Order MAML
    ADAPTIVE_META = "adaptive_meta"  # Adaptive Meta-Learning


class AdaptationType(Enum):
    """Types of adaptation scenarios."""
    FEW_SHOT = "few_shot"
    DOMAIN_ADAPTATION = "domain_adaptation"
    REGIME_ADAPTATION = "regime_adaptation"
    CONTINUAL_LEARNING = "continual_learning"
    TRANSFER_LEARNING = "transfer_learning"


@dataclass
class MetaTask:
    """Meta-learning task definition."""
    task_id: str
    support_set: Tuple[np.ndarray, np.ndarray]  # (X_support, y_support)
    query_set: Tuple[np.ndarray, np.ndarray]    # (X_query, y_query)
    regime_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AdaptationResult:
    """Result of meta-learning adaptation."""
    task_id: str
    adaptation_method: MetaLearningMethod
    initial_performance: Dict[str, float]
    adapted_performance: Dict[str, float]
    adaptation_steps: int
    adaptation_time: float
    convergence_achieved: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetaLearningConfig:
    """Configuration for meta-learning."""
    method: MetaLearningMethod = MetaLearningMethod.MAML
    adaptation_type: AdaptationType = AdaptationType.REGIME_ADAPTATION
    
    # Training parameters
    meta_learning_rate: float = 1e-3
    inner_learning_rate: float = 0.01
    num_inner_steps: int = 5
    num_outer_steps: int = 100
    meta_batch_size: int = 32
    
    # Few-shot learning parameters
    num_shots: int = 5
    num_ways: int = 5
    num_queries: int = 15
    
    # Adaptation parameters
    adaptation_threshold: float = 0.05
    max_adaptation_steps: int = 20
    patience: int = 5
    
    # Memory and continual learning
    memory_size: int = 1000
    forgetting_rate: float = 0.1
    replay_method: str = "random"  # "random", "importance", "diversity"
    
    # Hardware optimization
    enable_gpu: bool = True
    batch_size: int = 64
    num_workers: int = 4


class UnifiedMetaLearner:
    """Unified meta-learning system for TAS and NAS architectures."""
    
    def __init__(self, 
                 architecture_type: ArchitectureType,
                 config: MetaLearningConfig):
        """Initialize the unified meta-learner.
        
        Args:
            architecture_type: Type of architecture (TAS/NAS)
            config: Meta-learning configuration
        """
        self.architecture_type = architecture_type
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Meta-learning state
        self.meta_optimizer = None
        self.meta_model = None
        self.task_memory = deque(maxlen=config.memory_size)
        self.adaptation_history: List[AdaptationResult] = []
        
        # Performance tracking
        self.meta_performance_history: deque = deque(maxlen=1000)
        self.regime_adaptation_speed: Dict[str, float] = {}
        self.convergence_patterns: Dict[str, List[float]] = defaultdict(list)
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() and config.enable_gpu else "cpu")
        
        self.logger.info(f"✅ Unified Meta-Learner initialized for {architecture_type.value}")
        self.logger.info(f"   Method: {config.method.value}")
        self.logger.info(f"   Device: {self.device}")
        self.logger.info(f"   Meta Learning Rate: {config.meta_learning_rate}")
    
    def initialize_meta_model(self, base_model: Any):
        """Initialize the meta-model for adaptation.
        
        Args:
            base_model: Base model to be used for meta-learning
        """
        try:
            # Store reference to base model
            self.base_model = base_model
            
            # Create meta-model wrapper based on architecture type
            if self.architecture_type == ArchitectureType.TAS:
                self.meta_model = TASMetaModel(base_model, self.config)
            elif self.architecture_type == ArchitectureType.NAS:
                self.meta_model = NASMetaModel(base_model, self.config)
            else:
                self.meta_model = HybridMetaModel(base_model, self.config)
            
            # Initialize meta-optimizer
            self.meta_optimizer = optim.Adam(
                self.meta_model.parameters(),
                lr=self.config.meta_learning_rate
            )
            
            self.logger.info("✅ Meta-model initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize meta-model: {e}")
            raise
    
    def add_meta_task(self, task: MetaTask):
        """Add a meta-learning task to the task memory.
        
        Args:
            task: Meta-learning task to add
        """
        self.task_memory.append(task)
        self.logger.debug(f"📚 Added meta-task {task.task_id} to memory")
    
    def create_meta_tasks(self, 
                         data: Dict[str, Tuple[np.ndarray, np.ndarray]],
                         regime_labels: Optional[np.ndarray] = None,
                         num_tasks: int = None) -> List[MetaTask]:
        """Create meta-learning tasks from data.
        
        Args:
            data: Dictionary mapping regime_id to (X, y) data
            regime_labels: Optional regime labels for data
            num_tasks: Number of tasks to create (default: all possible)
            
        Returns:
            List of meta-learning tasks
        """
        tasks = []
        task_id = 0
        
        if regime_labels is not None:
            # Create regime-specific tasks
            unique_regimes = np.unique(regime_labels)
            
            for regime in unique_regimes:
                regime_mask = regime_labels == regime
                regime_data = {regime_id: (X[regime_mask], y[regime_mask]) 
                             for regime_id, (X, y) in data.items()}
                
                tasks.extend(self._create_regime_tasks(regime_data, regime, task_id))
                task_id += len(tasks)
        
        else:
            # Create general tasks
            for regime_id, (X, y) in data.items():
                tasks.extend(self._create_general_tasks(X, y, regime_id, task_id))
                task_id += len(tasks)
        
        # Limit number of tasks if specified
        if num_tasks is not None:
            tasks = tasks[:num_tasks]
        
        self.logger.info(f"✅ Created {len(tasks)} meta-learning tasks")
        return tasks
    
    def _create_regime_tasks(self, 
                           regime_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
                           regime_id: str,
                           start_task_id: int) -> List[MetaTask]:
        """Create regime-specific meta-learning tasks."""
        tasks = []
        
        for data_id, (X, y) in regime_data.items():
            if len(X) < self.config.num_shots + self.config.num_queries:
                continue
            
            # Create support and query sets
            support_indices = np.random.choice(
                len(X), size=self.config.num_shots, replace=False
            )
            query_indices = np.random.choice(
                np.setdiff1d(np.arange(len(X)), support_indices),
                size=self.config.num_queries, replace=False
            )
            
            task = MetaTask(
                task_id=f"regime_{regime_id}_{start_task_id + len(tasks)}",
                support_set=(X[support_indices], y[support_indices]),
                query_set=(X[query_indices], y[query_indices]),
                regime_id=regime_id,
                metadata={'data_id': data_id, 'regime_type': regime_id}
            )
            
            tasks.append(task)
        
        return tasks
    
    def _create_general_tasks(self, 
                            X: np.ndarray, 
                            y: np.ndarray,
                            data_id: str,
                            start_task_id: int) -> List[MetaTask]:
        """Create general meta-learning tasks."""
        tasks = []
        
        if len(X) < self.config.num_shots + self.config.num_queries:
            return tasks
        
        # Create multiple tasks from the same dataset
        num_tasks = min(10, len(X) // (self.config.num_shots + self.config.num_queries))
        
        for i in range(num_tasks):
            # Create support and query sets
            support_indices = np.random.choice(
                len(X), size=self.config.num_shots, replace=False
            )
            query_indices = np.random.choice(
                np.setdiff1d(np.arange(len(X)), support_indices),
                size=self.config.num_queries, replace=False
            )
            
            task = MetaTask(
                task_id=f"general_{data_id}_{start_task_id + i}",
                support_set=(X[support_indices], y[support_indices]),
                query_set=(X[query_indices], y[query_indices]),
                metadata={'data_id': data_id, 'task_type': 'general'}
            )
            
            tasks.append(task)
        
        return tasks
    
    def meta_train(self, tasks: List[MetaTask]) -> Dict[str, Any]:
        """Perform meta-training on a set of tasks.
        
        Args:
            tasks: List of meta-learning tasks
            
        Returns:
            Meta-training results
        """
        if self.meta_model is None:
            raise ValueError("Meta-model not initialized. Call initialize_meta_model first.")
        
        self.logger.info(f"🚀 Starting meta-training on {len(tasks)} tasks")
        start_time = time.time()
        
        meta_losses = []
        adaptation_speeds = []
        
        for outer_step in range(self.config.num_outer_steps):
            # Sample batch of tasks
            task_batch = np.random.choice(tasks, size=min(self.config.meta_batch_size, len(tasks)), replace=False)
            
            outer_loss = 0.0
            
            for task in task_batch:
                # Inner loop: adapt to task
                adapted_model = self._inner_loop(task)
                
                # Evaluate adapted model on query set
                query_loss = self._evaluate_on_query_set(adapted_model, task)
                outer_loss += query_loss
            
            # Outer loop: update meta-parameters
            outer_loss /= len(task_batch)
            meta_losses.append(outer_loss)
            
            self.meta_optimizer.zero_grad()
            outer_loss.backward()
            self.meta_optimizer.step()
            
            # Track adaptation speed
            adaptation_speed = self._calculate_adaptation_speed(task_batch)
            adaptation_speeds.append(adaptation_speed)
            
            if outer_step % 10 == 0:
                self.logger.info(f"   Meta-training step {outer_step}/{self.config.num_outer_steps}, "
                               f"loss: {outer_loss:.4f}, speed: {adaptation_speed:.4f}")
        
        training_time = time.time() - start_time
        
        # Store meta-training results
        meta_training_result = {
            'method': self.config.method.value,
            'architecture_type': self.architecture_type.value,
            'num_tasks': len(tasks),
            'num_outer_steps': self.config.num_outer_steps,
            'training_time': training_time,
            'final_meta_loss': meta_losses[-1] if meta_losses else 0.0,
            'avg_adaptation_speed': np.mean(adaptation_speeds) if adaptation_speeds else 0.0,
            'convergence_achieved': len(meta_losses) > 10 and np.std(meta_losses[-10:]) < 0.01,
            'meta_losses': meta_losses,
            'adaptation_speeds': adaptation_speeds
        }
        
        self.meta_performance_history.append(meta_training_result)
        
        self.logger.info(f"✅ Meta-training completed in {training_time:.2f}s")
        self.logger.info(f"   Final meta-loss: {meta_training_result['final_meta_loss']:.4f}")
        self.logger.info(f"   Avg adaptation speed: {meta_training_result['avg_adaptation_speed']:.4f}")
        
        return meta_training_result
    
    def _inner_loop(self, task: MetaTask):
        """Perform inner loop adaptation for a task.
        
        Args:
            task: Meta-learning task
            
        Returns:
            Adapted model
        """
        # Create a copy of the meta-model for adaptation
        adapted_model = self.meta_model.clone()
        
        # Inner optimizer for this task
        inner_optimizer = optim.SGD(
            adapted_model.parameters(),
            lr=self.config.inner_learning_rate
        )
        
        X_support, y_support = task.support_set
        
        # Inner loop adaptation
        for inner_step in range(self.config.num_inner_steps):
            inner_optimizer.zero_grad()
            
            # Forward pass on support set
            support_loss = adapted_model.compute_loss(X_support, y_support)
            
            # Backward pass
            support_loss.backward()
            inner_optimizer.step()
            
            # Early stopping if converged
            if support_loss.item() < self.config.adaptation_threshold:
                break
        
        return adapted_model
    
    def _evaluate_on_query_set(self, adapted_model, task: MetaTask) -> torch.Tensor:
        """Evaluate adapted model on query set.
        
        Args:
            adapted_model: Model adapted for the task
            task: Meta-learning task
            
        Returns:
            Query loss
        """
        X_query, y_query = task.query_set
        
        # Forward pass on query set
        query_loss = adapted_model.compute_loss(X_query, y_query)
        
        return query_loss
    
    def _calculate_adaptation_speed(self, tasks: List[MetaTask]) -> float:
        """Calculate adaptation speed for a batch of tasks."""
        speeds = []
        
        for task in tasks:
            # Measure adaptation time
            start_time = time.time()
            adapted_model = self._inner_loop(task)
            adaptation_time = time.time() - start_time
            
            # Calculate speed as inverse of adaptation time
            speed = 1.0 / (adaptation_time + 1e-8)
            speeds.append(speed)
        
        return np.mean(speeds)
    
    def adapt_to_new_regime(self, 
                          new_data: Tuple[np.ndarray, np.ndarray],
                          regime_id: str,
                          max_steps: int = None) -> AdaptationResult:
        """Adapt the meta-model to a new regime.
        
        Args:
            new_data: New regime data (X, y)
            regime_id: Identifier for the new regime
            max_steps: Maximum adaptation steps (default: config value)
            
        Returns:
            Adaptation result
        """
        if self.meta_model is None:
            raise ValueError("Meta-model not initialized.")
        
        self.logger.info(f"🔄 Adapting to new regime: {regime_id}")
        start_time = time.time()
        
        X_new, y_new = new_data
        max_steps = max_steps or self.config.max_adaptation_steps
        
        # Create adaptation task
        adaptation_task = MetaTask(
            task_id=f"adaptation_{regime_id}_{int(time.time())}",
            support_set=new_data,
            query_set=new_data,  # Use same data for query set in adaptation
            regime_id=regime_id,
            metadata={'adaptation_type': 'regime_adaptation'}
        )
        
        # Evaluate initial performance
        initial_performance = self._evaluate_performance(X_new, y_new)
        
        # Perform adaptation
        adapted_model = self._inner_loop(adaptation_task)
        
        # Evaluate adapted performance
        adapted_performance = self._evaluate_performance(X_new, y_new, adapted_model)
        
        adaptation_time = time.time() - start_time
        
        # Check convergence
        convergence_achieved = (
            abs(adapted_performance.get('loss', 0) - initial_performance.get('loss', 0)) < self.config.adaptation_threshold
        )
        
        # Create adaptation result
        result = AdaptationResult(
            task_id=adaptation_task.task_id,
            adaptation_method=self.config.method,
            initial_performance=initial_performance,
            adapted_performance=adapted_performance,
            adaptation_steps=self.config.num_inner_steps,
            adaptation_time=adaptation_time,
            convergence_achieved=convergence_achieved,
            metadata={
                'regime_id': regime_id,
                'data_size': len(X_new),
                'adaptation_type': 'regime_adaptation'
            }
        )
        
        self.adaptation_history.append(result)
        
        # Update regime adaptation speed
        self.regime_adaptation_speed[regime_id] = 1.0 / (adaptation_time + 1e-8)
        
        self.logger.info(f"✅ Adaptation completed in {adaptation_time:.2f}s")
        self.logger.info(f"   Initial performance: {initial_performance}")
        self.logger.info(f"   Adapted performance: {adapted_performance}")
        self.logger.info(f"   Convergence achieved: {convergence_achieved}")
        
        return result
    
    def _evaluate_performance(self, 
                            X: np.ndarray, 
                            y: np.ndarray, 
                            model: Any = None) -> Dict[str, float]:
        """Evaluate model performance on data."""
        if model is None:
            model = self.meta_model
        
        try:
            # Convert to appropriate format
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)
            
            # Forward pass
            with torch.no_grad():
                predictions = model(X_tensor)
                loss = model.compute_loss(X_tensor, y_tensor)
                
                # Calculate additional metrics
                if hasattr(model, 'calculate_metrics'):
                    metrics = model.calculate_metrics(predictions, y_tensor)
                else:
                    # Default metrics
                    metrics = {
                        'accuracy': torch.mean((torch.round(predictions) == y_tensor).float()).item(),
                        'mse': torch.mean((predictions - y_tensor) ** 2).item()
                    }
            
            return {
                'loss': loss.item(),
                **metrics
            }
            
        except Exception as e:
            self.logger.warning(f"Performance evaluation failed: {e}")
            return {'loss': float('inf'), 'accuracy': 0.0}
    
    def continual_learning_update(self, 
                                new_data: Tuple[np.ndarray, np.ndarray],
                                regime_id: str):
        """Update the meta-model using continual learning.
        
        Args:
            new_data: New data for continual learning
            regime_id: Regime identifier
        """
        self.logger.info(f"📚 Continual learning update for regime: {regime_id}")
        
        # Add new task to memory
        new_task = MetaTask(
            task_id=f"continual_{regime_id}_{int(time.time())}",
            support_set=new_data,
            query_set=new_data,
            regime_id=regime_id,
            metadata={'learning_type': 'continual'}
        )
        
        self.add_meta_task(new_task)
        
        # Perform replay if memory is full
        if len(self.task_memory) >= self.config.memory_size:
            self._perform_replay()
        
        # Update meta-model with new task
        self._update_meta_model_with_task(new_task)
    
    def _perform_replay(self):
        """Perform experience replay to prevent catastrophic forgetting."""
        if len(self.task_memory) < 2:
            return
        
        # Select tasks for replay based on method
        if self.config.replay_method == "random":
            replay_tasks = np.random.choice(
                list(self.task_memory), 
                size=min(10, len(self.task_memory) - 1), 
                replace=False
            ).tolist()
        elif self.config.replay_method == "importance":
            # Select most important tasks (simplified)
            replay_tasks = list(self.task_memory)[-10:]  # Most recent
        else:  # diversity
            # Select diverse tasks (simplified)
            replay_tasks = list(self.task_memory)[::max(1, len(self.task_memory) // 10)]
        
        # Perform replay training
        for task in replay_tasks:
            self._update_meta_model_with_task(task)
    
    def _update_meta_model_with_task(self, task: MetaTask):
        """Update meta-model with a specific task."""
        try:
            X_support, y_support = task.support_set
            
            # Single update step
            self.meta_optimizer.zero_grad()
            
            support_loss = self.meta_model.compute_loss(
                torch.tensor(X_support, dtype=torch.float32).to(self.device),
                torch.tensor(y_support, dtype=torch.float32).to(self.device)
            )
            
            support_loss.backward()
            self.meta_optimizer.step()
            
        except Exception as e:
            self.logger.warning(f"Task update failed: {e}")
    
    def get_adaptation_statistics(self) -> Dict[str, Any]:
        """Get statistics about adaptation performance."""
        if not self.adaptation_history:
            return {'error': 'No adaptation history available'}
        
        adaptation_times = [result.adaptation_time for result in self.adaptation_history]
        convergence_rates = [result.convergence_achieved for result in self.adaptation_history]
        
        # Calculate performance improvements
        improvements = []
        for result in self.adaptation_history:
            if 'loss' in result.initial_performance and 'loss' in result.adapted_performance:
                improvement = (result.initial_performance['loss'] - result.adapted_performance['loss']) / result.initial_performance['loss']
                improvements.append(improvement)
        
        statistics = {
            'total_adaptations': len(self.adaptation_history),
            'avg_adaptation_time': np.mean(adaptation_times),
            'convergence_rate': np.mean(convergence_rates),
            'avg_performance_improvement': np.mean(improvements) if improvements else 0.0,
            'regime_adaptation_speeds': self.regime_adaptation_speed,
            'memory_usage': len(self.task_memory),
            'meta_training_sessions': len(self.meta_performance_history)
        }
        
        return statistics
    
    def export_meta_model(self, filepath: str):
        """Export the meta-model to file."""
        try:
            output_path = Path(filepath)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save meta-model state
            torch.save({
                'meta_model_state_dict': self.meta_model.state_dict(),
                'meta_optimizer_state_dict': self.meta_optimizer.state_dict(),
                'config': self.config,
                'architecture_type': self.architecture_type.value,
                'adaptation_history': self.adaptation_history,
                'regime_adaptation_speeds': self.regime_adaptation_speeds
            }, output_path)
            
            self.logger.info(f"✅ Meta-model exported to {filepath}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to export meta-model: {e}")
            raise
    
    def import_meta_model(self, filepath: str):
        """Import meta-model from file."""
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            
            # Load meta-model state
            if self.meta_model is None:
                raise ValueError("Meta-model not initialized. Call initialize_meta_model first.")
            
            self.meta_model.load_state_dict(checkpoint['meta_model_state_dict'])
            self.meta_optimizer.load_state_dict(checkpoint['meta_optimizer_state_dict'])
            
            # Load additional data
            self.adaptation_history = checkpoint.get('adaptation_history', [])
            self.regime_adaptation_speeds = checkpoint.get('regime_adaptation_speeds', {})
            
            self.logger.info(f"✅ Meta-model imported from {filepath}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to import meta-model: {e}")
            raise


# Meta-model wrappers for different architectures
class BaseMetaModel(nn.Module):
    """Base meta-model wrapper."""
    
    def __init__(self, base_model: Any, config: MetaLearningConfig):
        super().__init__()
        self.base_model = base_model
        self.config = config
    
    def clone(self):
        """Create a clone of the meta-model."""
        try:
            # Create a new instance with the same configuration
            cloned_model = self.__class__(self.base_model, self.config)
            
            # Copy the current state if available
            if hasattr(self, 'state_dict'):
                cloned_model.load_state_dict(self.state_dict())
            
            return cloned_model
        except Exception as e:
            tprint(f"⚠️ [META_LEARNING] Error cloning meta-model: {e}", color="yellow")
            # Return a new instance as fallback
            return self.__class__(self.base_model, self.config)
    
    def compute_loss(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute loss for given inputs and targets."""
        try:
            # Forward pass through the base model
            if hasattr(self.base_model, 'forward'):
                predictions = self.base_model.forward(X)
            else:
                predictions = self.base_model(X)
            
            # Compute loss using the configured loss function
            if hasattr(self.config, 'loss_function'):
                loss_fn = self.config.loss_function
            else:
                # Default to MSE loss for regression, CrossEntropy for classification
                if len(y.shape) > 1 and y.shape[1] > 1:
                    loss_fn = torch.nn.CrossEntropyLoss()
                else:
                    loss_fn = torch.nn.MSELoss()
            
            loss = loss_fn(predictions, y)
            return loss
            
        except Exception as e:
            tprint(f"⚠️ [META_LEARNING] Error computing loss: {e}", color="yellow")
            # Return a default loss value
            return torch.tensor(0.0, requires_grad=True)


class TASMetaModel(BaseMetaModel):
    """Meta-model wrapper for TAS architectures."""
    
    def __init__(self, base_model: Any, config: MetaLearningConfig):
        super().__init__(base_model, config)
        # Initialize TAS-specific meta-parameters
        self.tree_weights = nn.Parameter(torch.randn(10))  # Example: 10 tree weights
        self.feature_weights = nn.Parameter(torch.randn(100))  # Example: 100 feature weights
    
    def clone(self):
        """Create a clone of the TAS meta-model."""
        cloned = TASMetaModel(self.base_model, self.config)
        cloned.load_state_dict(self.state_dict())
        return cloned
    
    def compute_loss(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute loss for TAS model."""
        # Simplified loss computation for TAS
        # In practice, this would interface with the actual TAS model
        predictions = self.forward(X)
        return nn.MSELoss()(predictions, y)
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass for TAS meta-model."""
        # Simplified forward pass
        # In practice, this would use the actual TAS model with meta-parameters
        return torch.sum(X * self.feature_weights[:X.size(1)], dim=1, keepdim=True)


class NASMetaModel(BaseMetaModel):
    """Meta-model wrapper for NAS architectures."""
    
    def __init__(self, base_model: Any, config: MetaLearningConfig):
        super().__init__(base_model, config)
        # Initialize NAS-specific meta-parameters
        self.layer_weights = nn.Parameter(torch.randn(20))  # Example: 20 layer weights
        self.activation_weights = nn.Parameter(torch.randn(10))  # Example: 10 activation weights
    
    def clone(self):
        """Create a clone of the NAS meta-model."""
        cloned = NASMetaModel(self.base_model, self.config)
        cloned.load_state_dict(self.state_dict())
        return cloned
    
    def compute_loss(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute loss for NAS model."""
        # Simplified loss computation for NAS
        predictions = self.forward(X)
        return nn.MSELoss()(predictions, y)
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass for NAS meta-model."""
        # Simplified forward pass
        # In practice, this would use the actual NAS model with meta-parameters
        return torch.sum(X * self.layer_weights[:X.size(1)], dim=1, keepdim=True)


class HybridMetaModel(BaseMetaModel):
    """Meta-model wrapper for hybrid TAS-NAS architectures."""
    
    def __init__(self, base_model: Any, config: MetaLearningConfig):
        super().__init__(base_model, config)
        # Initialize hybrid-specific meta-parameters
        self.tas_weight = nn.Parameter(torch.tensor(0.5))
        self.nas_weight = nn.Parameter(torch.tensor(0.5))
        self.fusion_weights = nn.Parameter(torch.randn(50))  # Example: 50 fusion weights
    
    def clone(self):
        """Create a clone of the hybrid meta-model."""
        cloned = HybridMetaModel(self.base_model, self.config)
        cloned.load_state_dict(self.state_dict())
        return cloned
    
    def compute_loss(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute loss for hybrid model."""
        predictions = self.forward(X)
        return nn.MSELoss()(predictions, y)
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass for hybrid meta-model."""
        # Simplified forward pass combining TAS and NAS
        tas_output = torch.sum(X * self.fusion_weights[:X.size(1)//2], dim=1, keepdim=True)
        nas_output = torch.sum(X * self.fusion_weights[X.size(1)//2:], dim=1, keepdim=True)
        
        # Weighted combination
        return self.tas_weight * tas_output + self.nas_weight * nas_output


# Convenience functions
def create_meta_learner(architecture_type: ArchitectureType,
                       method: MetaLearningMethod = MetaLearningMethod.MAML,
                       **kwargs) -> UnifiedMetaLearner:
    """Create a meta-learner with default settings."""
    config = MetaLearningConfig(method=method, **kwargs)
    return UnifiedMetaLearner(architecture_type=architecture_type, config=config)


def create_few_shot_learner(architecture_type: ArchitectureType) -> UnifiedMetaLearner:
    """Create a few-shot learning meta-learner."""
    config = MetaLearningConfig(
        method=MetaLearningMethod.PROTONET,
        adaptation_type=AdaptationType.FEW_SHOT,
        num_shots=5,
        num_ways=5
    )
    return UnifiedMetaLearner(architecture_type=architecture_type, config=config)


def create_continual_learner(architecture_type: ArchitectureType) -> UnifiedMetaLearner:
    """Create a continual learning meta-learner."""
    config = MetaLearningConfig(
        method=MetaLearningMethod.ADAPTIVE_META,
        adaptation_type=AdaptationType.CONTINUAL_LEARNING,
        memory_size=1000,
        forgetting_rate=0.1,
        replay_method="importance"
    )
    return UnifiedMetaLearner(architecture_type=architecture_type, config=config)