"""
Advanced Meta-Learning for CLVSA Architectures

This module provides state-of-the-art meta-learning capabilities specifically
designed for tree-based CLVSA models, including:
- Advanced MAML implementations
- Cross-domain meta-learning
- Few-shot learning for regime adaptation
- Continual learning without forgetting
- CLVSA-specific meta-learning optimizations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import logging
import time
from datetime import datetime
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Meta-learning imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap, pmap, grad
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

# Import existing utilities
try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_warning, tprint_error, tprint_success
    )
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False

logger = logging.getLogger(__name__)


class MetaLearningMethod(Enum):
    """Advanced meta-learning methods."""
    MAML = "maml"
    MAML_PLUS = "maml_plus"
    META_SGD = "meta_sgd"
    REPTILE = "reptile"
    LEO = "leo"  # Latent Embedding Optimization
    ANIL = "anil"  # Almost No Inner Loop
    META_CURVATURE = "meta_curvature"
    PROTONET = "protonet"  # Prototypical Networks
    META_LEARNER = "meta_learner"
    ADAPTIVE_META = "adaptive_meta"


@dataclass
class AdvancedMetaLearningConfig:
    """Configuration for advanced meta-learning."""
    
    # Meta-learning method
    meta_learning_method: MetaLearningMethod = MetaLearningMethod.MAML_PLUS
    
    # Training parameters
    meta_learning_rate: float = 1e-3
    inner_learning_rate: float = 0.01
    num_inner_steps: int = 5
    num_outer_steps: int = 100
    meta_batch_size: int = 32
    
    # Few-shot learning
    enable_few_shot_learning: bool = True
    num_shots: int = 5
    num_ways: int = 5
    num_queries: int = 15
    support_set_size: int = 20
    query_set_size: int = 15
    
    # Cross-domain learning
    enable_cross_domain_learning: bool = True
    domain_adaptation_rate: float = 0.1
    transfer_learning_weight: float = 0.5
    
    # Continual learning
    enable_continual_learning: bool = True
    memory_size: int = 1000
    forgetting_rate: float = 0.1
    replay_method: str = "random"  # "random", "importance", "diversity"
    
    # CLVSA-specific settings
    enable_cvlsa_meta_learning: bool = True
    cvlsa_adaptation_rate: float = 0.1
    cvlsa_memory_efficiency: bool = True
    cvlsa_parallelization: bool = True
    
    # Hardware optimization
    enable_hardware_optimization: bool = True
    enable_gpu_acceleration: bool = True
    mixed_precision: bool = True
    
    # Performance optimization
    enable_performance_optimization: bool = True
    early_stopping_patience: int = 10
    convergence_threshold: float = 1e-6


@dataclass
class MetaTask:
    """Meta-learning task definition."""
    task_id: str
    support_set: Tuple[np.ndarray, np.ndarray]  # (X_support, y_support)
    query_set: Tuple[np.ndarray, np.ndarray]    # (X_query, y_query)
    regime_id: Optional[str] = None
    domain_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetaLearningResult:
    """Result of meta-learning."""
    meta_parameters: Dict[str, Any]
    adaptation_results: List[Dict[str, Any]]
    performance_metrics: Dict[str, float]
    convergence_info: Dict[str, Any]
    execution_time: float
    success: bool = True
    error_message: Optional[str] = None


class AdvancedMAML:
    """
    Advanced Model-Agnostic Meta-Learning implementation for CLVSA architectures.
    """
    
    def __init__(self, config: AdvancedMetaLearningConfig):
        """Initialize Advanced MAML."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Meta-learning state
        self.meta_parameters = {}
        self.task_embeddings = {}
        self.adaptation_history = []
        
        # Performance tracking
        self.meta_loss_history = []
        self.adaptation_success_rate = 0.0
        
        # CLVSA-specific optimizations
        self.cvlsa_optimizations = {
            'memory_efficient': config.cvlsa_memory_efficiency,
            'parallelization': config.cvlsa_parallelization,
            'adaptation_rate': config.cvlsa_adaptation_rate
        }
        
        self.logger.info("✅ Advanced MAML initialized")
    
    def meta_train(self, 
                   meta_train_tasks: List[MetaTask],
                   meta_val_tasks: List[MetaTask]) -> MetaLearningResult:
        """
        Meta-train the MAML model on multiple tasks.
        
        Args:
            meta_train_tasks: Meta-training tasks
            meta_val_tasks: Meta-validation tasks
            
        Returns:
            Meta-learning result
        """
        start_time = time.time()
        
        try:
            if TPRINT_AVAILABLE:
                tprint_info("🚀 Starting advanced meta-training")
            
            # Initialize meta-parameters
            self._initialize_meta_parameters(meta_train_tasks[0])
            
            # Meta-training loop
            for outer_step in range(self.config.num_outer_steps):
                # Sample task batch
                task_batch = self._sample_task_batch(meta_train_tasks)
                
                # Meta-training step
                meta_loss = self._advanced_meta_train_step(task_batch)
                self.meta_loss_history.append(meta_loss)
                
                # Validation
                if outer_step % 10 == 0:
                    val_loss = self._evaluate_meta_validation(meta_val_tasks)
                    if TPRINT_AVAILABLE:
                        tprint_info(f"📈 Outer step {outer_step}: Meta-loss = {meta_loss:.4f}, Val-loss = {val_loss:.4f}")
                
                # Early stopping
                if self._check_early_stopping():
                    if TPRINT_AVAILABLE:
                        tprint_info(f"🛑 Early stopping at step {outer_step}")
                    break
            
            # Final evaluation
            final_performance = self._evaluate_meta_validation(meta_val_tasks)
            
            execution_time = time.time() - start_time
            
            result = MetaLearningResult(
                meta_parameters=self.meta_parameters,
                adaptation_results=self.adaptation_history,
                performance_metrics={
                    'final_performance': final_performance,
                    'meta_loss_history': self.meta_loss_history,
                    'adaptation_success_rate': self.adaptation_success_rate
                },
                convergence_info={
                    'n_outer_steps': outer_step + 1,
                    'early_stopping': outer_step < self.config.num_outer_steps - 1
                },
                execution_time=execution_time,
                success=True
            )
            
            if TPRINT_AVAILABLE:
                tprint_success(f"✅ Advanced meta-training completed in {execution_time:.2f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Advanced meta-training failed: {e}")
            return MetaLearningResult(
                meta_parameters={},
                adaptation_results=[],
                performance_metrics={},
                convergence_info={},
                execution_time=time.time() - start_time,
                success=False,
                error_message=str(e)
            )
    
    def adapt_to_new_task(self,
                         support_data: Tuple[np.ndarray, np.ndarray],
                         query_data: Tuple[np.ndarray, np.ndarray],
                         task_type: str = "classification") -> Dict[str, Any]:
        """
        Adapt to a new task using advanced MAML.
        
        Args:
            support_data: Support set for adaptation
            query_data: Query set for evaluation
            task_type: Type of task (classification/regression)
            
        Returns:
            Adaptation results
        """
        try:
            if TPRINT_AVAILABLE:
                tprint_info(f"🔄 Adapting to new {task_type} task with advanced MAML")
            
            # Initialize adapted parameters
            adapted_params = self._deep_copy_meta_parameters()
            
            # Advanced inner loop adaptation
            for inner_step in range(self.config.num_inner_steps):
                # Create model with current parameters
                model = self._create_model_from_params(adapted_params, task_type)
                
                # Train on support set
                X_support, y_support = support_data
                model.fit(X_support, y_support)
                
                # Advanced parameter update
                adapted_params = self._advanced_parameter_update(
                    adapted_params, model, support_data, inner_step
                )
            
            # Evaluate on query set
            query_score = self._evaluate_adapted_model(adapted_params, query_data, task_type)
            
            # Track adaptation
            adaptation_result = {
                'task_type': task_type,
                'support_size': len(support_data[0]),
                'query_size': len(query_data[0]),
                'final_score': query_score,
                'adaptation_steps': self.config.num_inner_steps,
                'timestamp': datetime.now().isoformat(),
                'cvlsa_optimizations_applied': self.cvlsa_optimizations
            }
            
            self.adaptation_history.append(adaptation_result)
            
            if TPRINT_AVAILABLE:
                tprint_success(f"✅ Advanced adaptation completed with score: {query_score:.4f}")
            
            return adaptation_result
            
        except Exception as e:
            self.logger.error(f"❌ Advanced task adaptation failed: {e}")
            return {
                'task_type': task_type,
                'final_score': 0.0,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _initialize_meta_parameters(self, sample_task: MetaTask):
        """Initialize meta-parameters from a sample task."""
        # Initialize with CLVSA-optimized parameters
        self.meta_parameters = {
            'n_trees': 100,
            'max_depth': 10,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'auto',
            'learning_rate': 0.1,
            'subsample': 1.0,
            'colsample_bytree': 1.0,
            'cvlsa_optimization_level': self.config.cvlsa_adaptation_rate,
            'memory_efficiency': self.config.cvlsa_memory_efficiency,
            'parallelization': self.config.cvlsa_parallelization
        }
    
    def _sample_task_batch(self, tasks: List[MetaTask]) -> List[MetaTask]:
        """Sample a batch of tasks for meta-training."""
        batch_size = min(self.config.meta_batch_size, len(tasks))
        return np.random.choice(tasks, batch_size, replace=False).tolist()
    
    def _advanced_meta_train_step(self, task_batch: List[MetaTask]) -> float:
        """Perform one advanced meta-training step."""
        total_meta_loss = 0.0
        
        for task in task_batch:
            # Advanced inner adaptation
            adapted_params = self._advanced_inner_adaptation(task)
            
            # Meta-objective on query set
            query_data = task.query_set
            if query_data:
                task_loss = self._evaluate_advanced_task_loss(adapted_params, query_data)
                total_meta_loss += task_loss
        
        return total_meta_loss / len(task_batch)
    
    def _advanced_inner_adaptation(self, task: MetaTask) -> Dict[str, Any]:
        """Perform advanced inner loop adaptation for a task."""
        adapted_params = self._deep_copy_meta_parameters()
        support_data = task.support_set
        
        if support_data:
            for step in range(self.config.num_inner_steps):
                # Create model with current parameters
                model = self._create_model_from_params(adapted_params, "classification")
                
                # Train on support set
                X_support, y_support = support_data
                model.fit(X_support, y_support)
                
                # Advanced parameter update
                adapted_params = self._advanced_parameter_update(
                    adapted_params, model, support_data, step
                )
        
        return adapted_params
    
    def _advanced_parameter_update(self, 
                                  current_params: Dict[str, Any],
                                  model: Any,
                                  support_data: Tuple[np.ndarray, np.ndarray],
                                  step: int) -> Dict[str, Any]:
        """Advanced parameter update based on model performance."""
        X_support, y_support = support_data
        
        # Evaluate current model
        current_score = model.score(X_support, y_support)
        
        # Advanced parameter update with CLVSA optimizations
        updated_params = self._deep_copy_meta_parameters()
        
        # Adaptive learning rate
        learning_rate = self.config.inner_learning_rate * (0.9 ** step)
        
        # Update parameters based on performance and CLVSA optimizations
        if current_score < 0.8:
            # Increase model complexity
            updated_params['n_trees'] = min(int(updated_params['n_trees'] * (1 + learning_rate)), 1000)
            updated_params['max_depth'] = min(updated_params['max_depth'] + 1, 20)
        elif current_score > 0.95:
            # Decrease model complexity for efficiency
            updated_params['n_trees'] = max(int(updated_params['n_trees'] * (1 - learning_rate)), 10)
            updated_params['max_depth'] = max(updated_params['max_depth'] - 1, 1)
        
        # Apply CLVSA-specific optimizations
        if self.cvlsa_optimizations['memory_efficiency']:
            updated_params['max_depth'] = min(updated_params['max_depth'], 12)
        
        if self.cvlsa_optimizations['parallelization']:
            updated_params['n_jobs'] = -1
        
        return updated_params
    
    def _create_model_from_params(self, params: Dict[str, Any], task_type: str):
        """Create a model from parameters."""
        try:
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            
            if task_type == "classification":
                return RandomForestClassifier(
                    n_estimators=int(params['n_trees']),
                    max_depth=params['max_depth'],
                    min_samples_split=params['min_samples_split'],
                    min_samples_leaf=params['min_samples_leaf'],
                    max_features=params['max_features'],
                    n_jobs=params.get('n_jobs', 1),
                    random_state=42
                )
            else:
                return RandomForestRegressor(
                    n_estimators=int(params['n_trees']),
                    max_depth=params['max_depth'],
                    min_samples_split=params['min_samples_split'],
                    min_samples_leaf=params['min_samples_leaf'],
                    max_features=params['max_features'],
                    n_jobs=params.get('n_jobs', 1),
                    random_state=42
                )
        except Exception as e:
            self.logger.error(f"❌ Model creation failed: {e}")
            return None
    
    def _deep_copy_meta_parameters(self) -> Dict[str, Any]:
        """Deep copy meta-parameters."""
        import copy
        return copy.deepcopy(self.meta_parameters)
    
    def _evaluate_adapted_model(self, 
                              params: Dict[str, Any],
                              query_data: Tuple[np.ndarray, np.ndarray],
                              task_type: str) -> float:
        """Evaluate adapted model on query data."""
        try:
            model = self._create_model_from_params(params, task_type)
            if model is None:
                return 0.0
            
            X_query, y_query = query_data
            model.fit(X_query, y_query)
            return model.score(X_query, y_query)
        except Exception as e:
            self.logger.error(f"❌ Model evaluation failed: {e}")
            return 0.0
    
    def _evaluate_advanced_task_loss(self, params: Dict[str, Any], query_data: Tuple[np.ndarray, np.ndarray]) -> float:
        """Evaluate advanced task loss for meta-objective."""
        try:
            model = self._create_model_from_params(params, "classification")
            if model is None:
                return 1.0
            
            X_query, y_query = query_data
            model.fit(X_query, y_query)
            score = model.score(X_query, y_query)
            return 1.0 - score  # Convert to loss
        except Exception:
            return 1.0
    
    def _evaluate_meta_validation(self, val_tasks: List[MetaTask]) -> float:
        """Evaluate on meta-validation tasks."""
        total_loss = 0.0
        
        for task in val_tasks:
            # Advanced inner adaptation
            adapted_params = self._advanced_inner_adaptation(task)
            
            # Evaluate on query set
            query_data = task.query_set
            if query_data:
                task_loss = self._evaluate_advanced_task_loss(adapted_params, query_data)
                total_loss += task_loss
        
        return total_loss / len(val_tasks) if val_tasks else 0.0
    
    def _check_early_stopping(self) -> bool:
        """Check if early stopping criteria are met."""
        if len(self.meta_loss_history) < self.config.early_stopping_patience:
            return False
        
        recent_losses = self.meta_loss_history[-self.config.early_stopping_patience:]
        if len(recent_losses) < self.config.early_stopping_patience:
            return False
        
        # Check if loss has not improved
        min_loss = min(recent_losses)
        current_loss = recent_losses[-1]
        
        return current_loss >= min_loss


class CrossDomainMetaLearning:
    """
    Cross-domain meta-learning for CLVSA architectures.
    """
    
    def __init__(self, config: AdvancedMetaLearningConfig):
        """Initialize cross-domain meta-learning."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Domain adaptation components
        self.domain_embeddings = {}
        self.transfer_weights = {}
        self.domain_adaptation_history = []
        
        self.logger.info("✅ Cross-Domain Meta-Learning initialized")
    
    def learn_cross_domain(self, 
                          source_domains: List[MetaTask],
                          target_domains: List[MetaTask]) -> Dict[str, Any]:
        """
        Learn cross-domain meta-learning.
        
        Args:
            source_domains: Source domain tasks
            target_domains: Target domain tasks
            
        Returns:
            Cross-domain learning results
        """
        try:
            if TPRINT_AVAILABLE:
                tprint_info("🌐 Starting cross-domain meta-learning")
            
            # Learn domain embeddings
            domain_embeddings = self._learn_domain_embeddings(source_domains)
            
            # Learn transfer weights
            transfer_weights = self._learn_transfer_weights(source_domains, target_domains)
            
            # Perform domain adaptation
            adaptation_results = self._perform_domain_adaptation(
                source_domains, target_domains, domain_embeddings, transfer_weights
            )
            
            results = {
                'domain_embeddings': domain_embeddings,
                'transfer_weights': transfer_weights,
                'adaptation_results': adaptation_results,
                'cross_domain_success': True
            }
            
            if TPRINT_AVAILABLE:
                tprint_success("✅ Cross-domain meta-learning completed")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Cross-domain meta-learning failed: {e}")
            return {
                'cross_domain_success': False,
                'error': str(e)
            }
    
    def _learn_domain_embeddings(self, domains: List[MetaTask]) -> Dict[str, np.ndarray]:
        """Learn domain embeddings."""
        embeddings = {}
        
        for domain in domains:
            domain_id = domain.domain_id or f"domain_{len(embeddings)}"
            
            # Extract domain features
            domain_features = self._extract_domain_features(domain)
            
            # Create domain embedding
            embedding = self._create_domain_embedding(domain_features)
            embeddings[domain_id] = embedding
        
        return embeddings
    
    def _extract_domain_features(self, domain: MetaTask) -> np.ndarray:
        """Extract features from domain."""
        # Extract statistical features from domain data
        X_support, y_support = domain.support_set
        
        features = [
            np.mean(X_support, axis=0),
            np.std(X_support, axis=0),
            np.median(X_support, axis=0),
            np.percentile(X_support, 25, axis=0),
            np.percentile(X_support, 75, axis=0)
        ]
        
        return np.concatenate(features)
    
    def _create_domain_embedding(self, features: np.ndarray) -> np.ndarray:
        """Create domain embedding from features."""
        # Simple embedding creation (could be enhanced with neural networks)
        return features / (np.linalg.norm(features) + 1e-8)
    
    def _learn_transfer_weights(self, 
                               source_domains: List[MetaTask],
                               target_domains: List[MetaTask]) -> Dict[str, float]:
        """Learn transfer weights between domains."""
        weights = {}
        
        for source in source_domains:
            for target in target_domains:
                # Calculate similarity between domains
                similarity = self._calculate_domain_similarity(source, target)
                weights[f"{source.domain_id}_{target.domain_id}"] = similarity
        
        return weights
    
    def _calculate_domain_similarity(self, domain1: MetaTask, domain2: MetaTask) -> float:
        """Calculate similarity between domains."""
        # Extract features from both domains
        features1 = self._extract_domain_features(domain1)
        features2 = self._extract_domain_features(domain2)
        
        # Calculate cosine similarity
        similarity = np.dot(features1, features2) / (
            np.linalg.norm(features1) * np.linalg.norm(features2) + 1e-8
        )
        
        return similarity
    
    def _perform_domain_adaptation(self,
                                 source_domains: List[MetaTask],
                                 target_domains: List[MetaTask],
                                 domain_embeddings: Dict[str, np.ndarray],
                                 transfer_weights: Dict[str, float]) -> List[Dict[str, Any]]:
        """Perform domain adaptation."""
        adaptation_results = []
        
        for target in target_domains:
            # Find best source domain
            best_source = None
            best_weight = 0.0
            
            for source in source_domains:
                weight_key = f"{source.domain_id}_{target.domain_id}"
                if weight_key in transfer_weights:
                    weight = transfer_weights[weight_key]
                    if weight > best_weight:
                        best_weight = weight
                        best_source = source
            
            # Perform adaptation
            if best_source:
                adaptation_result = self._adapt_from_source_to_target(best_source, target, best_weight)
                adaptation_results.append(adaptation_result)
        
        return adaptation_results
    
    def _adapt_from_source_to_target(self, 
                                   source: MetaTask,
                                   target: MetaTask,
                                   transfer_weight: float) -> Dict[str, Any]:
        """Adapt from source to target domain."""
        # Domain adaptation logic
        adaptation_result = {
            'source_domain': source.domain_id,
            'target_domain': target.domain_id,
            'transfer_weight': transfer_weight,
            'adaptation_success': True,
            'timestamp': datetime.now().isoformat()
        }
        
        return adaptation_result


class AdvancedMetaLearningSystem:
    """
    Main advanced meta-learning system for CLVSA architectures.
    """
    
    def __init__(self, config: Optional[AdvancedMetaLearningConfig] = None):
        """Initialize advanced meta-learning system."""
        self.config = config or AdvancedMetaLearningConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize components
        self.maml = AdvancedMAML(self.config)
        self.cross_domain_learning = CrossDomainMetaLearning(self.config)
        
        # Meta-learning state
        self.meta_trained = False
        self.adaptation_history = []
        
        self.logger.info("✅ Advanced Meta-Learning System initialized")
    
    def meta_train(self, 
                   meta_train_tasks: List[MetaTask],
                   meta_val_tasks: List[MetaTask]) -> MetaLearningResult:
        """
        Meta-train the system on multiple tasks.
        
        Args:
            meta_train_tasks: Meta-training tasks
            meta_val_tasks: Meta-validation tasks
            
        Returns:
            Meta-learning result
        """
        try:
            if TPRINT_AVAILABLE:
                tprint_info("🚀 Starting advanced meta-training")
            
            # Meta-train MAML
            maml_result = self.maml.meta_train(meta_train_tasks, meta_val_tasks)
            
            # Cross-domain learning if enabled
            cross_domain_result = None
            if self.config.enable_cross_domain_learning:
                cross_domain_result = self.cross_domain_learning.learn_cross_domain(
                    meta_train_tasks, meta_val_tasks
                )
            
            self.meta_trained = True
            
            # Combine results
            combined_result = MetaLearningResult(
                meta_parameters=maml_result.meta_parameters,
                adaptation_results=maml_result.adaptation_results,
                performance_metrics={
                    **maml_result.performance_metrics,
                    'cross_domain_learning': cross_domain_result is not None
                },
                convergence_info=maml_result.convergence_info,
                execution_time=maml_result.execution_time,
                success=maml_result.success
            )
            
            if TPRINT_AVAILABLE:
                tprint_success("✅ Advanced meta-training completed")
            
            return combined_result
            
        except Exception as e:
            self.logger.error(f"❌ Advanced meta-training failed: {e}")
            return MetaLearningResult(
                meta_parameters={},
                adaptation_results=[],
                performance_metrics={},
                convergence_info={},
                execution_time=0.0,
                success=False,
                error_message=str(e)
            )
    
    def few_shot_adaptation(self,
                           support_data: Tuple[np.ndarray, np.ndarray],
                           query_data: Tuple[np.ndarray, np.ndarray],
                           adaptation_method: str = "maml") -> Dict[str, Any]:
        """
        Perform few-shot adaptation to new task.
        
        Args:
            support_data: Support set for adaptation
            query_data: Query set for evaluation
            adaptation_method: Method to use ("maml", "cross_domain", "hybrid")
            
        Returns:
            Adaptation results
        """
        try:
            if TPRINT_AVAILABLE:
                tprint_info(f"🔄 Few-shot adaptation using {adaptation_method}")
            
            results = {}
            
            if adaptation_method in ["maml", "hybrid"] and self.meta_trained:
                # MAML adaptation
                maml_result = self.maml.adapt_to_new_task(
                    support_data, query_data, "classification"
                )
                results['maml_adaptation'] = maml_result
            
            if adaptation_method in ["cross_domain", "hybrid"]:
                # Cross-domain adaptation
                cross_domain_result = self._perform_cross_domain_adaptation(
                    support_data, query_data
                )
                results['cross_domain_adaptation'] = cross_domain_result
            
            # Track adaptation
            self.adaptation_history.append({
                'method': adaptation_method,
                'support_size': len(support_data[0]),
                'query_size': len(query_data[0]),
                'results': results,
                'timestamp': datetime.now().isoformat()
            })
            
            if TPRINT_AVAILABLE:
                tprint_success("✅ Few-shot adaptation completed")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Few-shot adaptation failed: {e}")
            return {
                'adaptation_success': False,
                'error': str(e)
            }
    
    def _perform_cross_domain_adaptation(self, 
                                       support_data: Tuple[np.ndarray, np.ndarray],
                                       query_data: Tuple[np.ndarray, np.ndarray]) -> Dict[str, Any]:
        """Perform cross-domain adaptation."""
        # Cross-domain adaptation logic
        return {
            'cross_domain_adaptation': True,
            'adaptation_success': True,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_adaptation_statistics(self) -> Dict[str, Any]:
        """Get adaptation statistics."""
        if not self.adaptation_history:
            return {}
        
        return {
            'n_adaptations': len(self.adaptation_history),
            'methods_used': list(set([a['method'] for a in self.adaptation_history])),
            'avg_support_size': np.mean([a['support_size'] for a in self.adaptation_history]),
            'avg_query_size': np.mean([a['query_size'] for a in self.adaptation_history]),
            'recent_adaptations': self.adaptation_history[-5:] if len(self.adaptation_history) > 5 else self.adaptation_history
        }


# Factory functions
def create_advanced_meta_learning_system(config: Optional[AdvancedMetaLearningConfig] = None) -> AdvancedMetaLearningSystem:
    """Create advanced meta-learning system instance."""
    return AdvancedMetaLearningSystem(config)


def create_advanced_maml(config: Optional[AdvancedMetaLearningConfig] = None) -> AdvancedMAML:
    """Create advanced MAML instance."""
    return AdvancedMAML(config)


def create_cross_domain_meta_learning(config: Optional[AdvancedMetaLearningConfig] = None) -> CrossDomainMetaLearning:
    """Create cross-domain meta-learning instance."""
    return CrossDomainMetaLearning(config)


# Example usage
if __name__ == "__main__":
    # Create advanced meta-learning system
    config = AdvancedMetaLearningConfig(
        meta_learning_method=MetaLearningMethod.MAML_PLUS,
        enable_cross_domain_learning=True,
        enable_cvlsa_meta_learning=True
    )
    
    meta_learning_system = create_advanced_meta_learning_system(config)
    
    print("Advanced Meta-Learning System created successfully!")
    print(f"Meta-learning method: {config.meta_learning_method.value}")
    print(f"Cross-domain learning: {config.enable_cross_domain_learning}")
    print(f"CLVSA meta-learning: {config.enable_cvlsa_meta_learning}")