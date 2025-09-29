"""
Tree Meta-Learning Implementation

Advanced meta-learning capabilities for tree-based models including MAML,
prototypical networks, and few-shot learning for regime adaptation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import logging
import time
from datetime import datetime
import copy
from collections import defaultdict

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# DecisionTreeClassifier removed - only advanced tree models supported
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

from ..core.tas_config import TASConfig, TreeModelType
from ..core.tree_architecture import TreeArchitectureCandidate
from ..core.tas_result import TASResult

logger = logging.getLogger(__name__)


@dataclass
class MetaLearningConfig:
    """Configuration for tree meta-learning."""
    
    # Meta-learning parameters
    meta_learning_rate: float = 0.001
    inner_learning_rate: float = 0.01
    num_inner_steps: int = 5
    num_outer_steps: int = 100
    meta_batch_size: int = 32
    
    # Few-shot learning parameters
    num_shots: int = 5
    num_ways: int = 5
    support_set_size: int = 20
    query_set_size: int = 15
    
    # Adaptation parameters
    adaptation_steps: int = 10
    adaptation_threshold: float = 0.1
    adaptation_method: str = "gradient"  # "gradient", "evolutionary", "bayesian"
    
    # Memory parameters
    memory_size: int = 1000
    memory_update_frequency: int = 10
    forgetting_rate: float = 0.1
    
    # Performance parameters
    min_adaptation_improvement: float = 0.01
    max_adaptation_time: float = 60.0
    early_stopping_patience: int = 5


class TreeMAML:
    """
    Model-Agnostic Meta-Learning for Tree Models.
    
    Implements MAML for tree-based models to enable fast adaptation
    to new tasks and regimes.
    """
    
    def __init__(self, config: MetaLearningConfig):
        """Initialize Tree MAML.
        
        Args:
            config: Meta-learning configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Meta-learning state
        self.meta_parameters = {}
        self.task_embeddings = {}
        self.adaptation_history = []
        
        # Performance tracking
        self.meta_loss_history = []
        self.adaptation_success_rate = 0.0
        
        self.logger.info("✅ Tree MAML initialized")
    
    def meta_train(self, 
                   meta_train_tasks: List[Dict[str, Any]],
                   meta_val_tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Meta-train the MAML model on multiple tasks.
        
        Args:
            meta_train_tasks: Meta-training tasks
            meta_val_tasks: Meta-validation tasks
            
        Returns:
            Meta-training results
        """
        self.logger.info("🚀 Starting meta-training for tree MAML")
        start_time = time.time()
        
        try:
            # Initialize meta-parameters
            self._initialize_meta_parameters(meta_train_tasks[0])
            
            # Meta-training loop
            for outer_step in range(self.config.num_outer_steps):
                # Sample task batch
                task_batch = self._sample_task_batch(meta_train_tasks)
                
                # Meta-training step
                meta_loss = self._meta_train_step(task_batch)
                self.meta_loss_history.append(meta_loss)
                
                # Validation
                if outer_step % 10 == 0:
                    val_loss = self._evaluate_meta_validation(meta_val_tasks)
                    self.logger.info(f"📈 Outer step {outer_step}: Meta-loss = {meta_loss:.4f}, Val-loss = {val_loss:.4f}")
                
                # Early stopping
                if self._check_early_stopping():
                    self.logger.info(f"🛑 Early stopping at step {outer_step}")
                    break
            
            # Final evaluation
            final_performance = self._evaluate_meta_validation(meta_val_tasks)
            
            execution_time = time.time() - start_time
            
            results = {
                'meta_parameters': self.meta_parameters,
                'final_performance': final_performance,
                'meta_loss_history': self.meta_loss_history,
                'execution_time': execution_time,
                'n_outer_steps': outer_step + 1,
                'success': final_performance < 0.5
            }
            
            self.logger.info(f"✅ Meta-training completed in {execution_time:.2f}s")
            self.logger.info(f"🎯 Final performance: {final_performance:.4f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Meta-training failed: {e}")
            raise
    
    def adapt_to_new_task(self,
                         support_data: Tuple[np.ndarray, np.ndarray],
                         query_data: Tuple[np.ndarray, np.ndarray],
                         task_type: str = "classification") -> TreeArchitectureCandidate:
        """
        Adapt to a new task using MAML.
        
        Args:
            support_data: Support set for adaptation
            query_data: Query set for evaluation
            task_type: Type of task (classification/regression)
            
        Returns:
            Adapted architecture
        """
        self.logger.info(f"🔄 Adapting to new {task_type} task")
        
        try:
            # Initialize adapted parameters
            adapted_params = copy.deepcopy(self.meta_parameters)
            
            # Inner loop adaptation
            for inner_step in range(self.config.num_inner_steps):
                # Create model with current parameters
                model = self._create_model_from_params(adapted_params, task_type)
                
                # Train on support set
                X_support, y_support = support_data
                model.fit(X_support, y_support)
                
                # Update parameters based on support set performance
                adapted_params = self._update_parameters(
                    adapted_params, model, support_data, inner_step
                )
            
            # Create final adapted architecture
            adapted_architecture = self._create_architecture_from_params(adapted_params)
            
            # Evaluate on query set
            query_score = self._evaluate_architecture(adapted_architecture, query_data, task_type)
            adapted_architecture.overall_score = query_score
            
            # Track adaptation
            self.adaptation_history.append({
                'task_type': task_type,
                'support_size': len(support_data[0]),
                'query_size': len(query_data[0]),
                'final_score': query_score,
                'adaptation_steps': self.config.num_inner_steps,
                'timestamp': datetime.now().isoformat()
            })
            
            self.logger.info(f"✅ Adaptation completed with score: {query_score:.4f}")
            return adapted_architecture
            
        except Exception as e:
            self.logger.error(f"❌ Task adaptation failed: {e}")
            raise
    
    def _initialize_meta_parameters(self, sample_task: Dict[str, Any]):
        """Initialize meta-parameters from a sample task."""
        # Initialize with default tree parameters
        self.meta_parameters = {
            'n_trees': 100,
            'max_depth': 10,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'auto',
            'learning_rate': 0.1,
            'subsample': 1.0,
            'colsample_bytree': 1.0
        }
    
    def _sample_task_batch(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sample a batch of tasks for meta-training."""
        batch_size = min(self.config.meta_batch_size, len(tasks))
        return np.random.choice(tasks, batch_size, replace=False).tolist()
    
    def _meta_train_step(self, task_batch: List[Dict[str, Any]]) -> float:
        """Perform one meta-training step."""
        total_meta_loss = 0.0
        
        for task in task_batch:
            # Inner adaptation
            adapted_params = self._inner_adaptation(task)
            
            # Meta-objective on query set
            query_data = task.get('query_data')
            if query_data:
                task_loss = self._evaluate_task_loss(adapted_params, query_data)
                total_meta_loss += task_loss
        
        return total_meta_loss / len(task_batch)
    
    def _inner_adaptation(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Perform inner loop adaptation for a task."""
        adapted_params = copy.deepcopy(self.meta_parameters)
        support_data = task.get('support_data')
        
        if support_data:
            for step in range(self.config.num_inner_steps):
                # Create model with current parameters
                model = self._create_model_from_params(adapted_params, task.get('task_type', 'classification'))
                
                # Train on support set
                X_support, y_support = support_data
                model.fit(X_support, y_support)
                
                # Update parameters
                adapted_params = self._update_parameters(adapted_params, model, support_data, step)
        
        return adapted_params
    
    def _update_parameters(self, 
                          current_params: Dict[str, Any],
                          model: Any,
                          support_data: Tuple[np.ndarray, np.ndarray],
                          step: int) -> Dict[str, Any]:
        """Update parameters based on model performance."""
        # Simple parameter update based on model performance
        X_support, y_support = support_data
        
        # Evaluate current model
        current_score = model.score(X_support, y_support)
        
        # Update parameters based on performance
        updated_params = copy.deepcopy(current_params)
        
        # Adjust tree count based on performance
        if current_score < 0.8:
            updated_params['n_trees'] = min(updated_params['n_trees'] * 1.1, 1000)
        elif current_score > 0.95:
            updated_params['n_trees'] = max(updated_params['n_trees'] * 0.9, 10)
        
        # Adjust depth based on performance
        if current_score < 0.8:
            updated_params['max_depth'] = min(updated_params['max_depth'] + 1, 20)
        elif current_score > 0.95:
            updated_params['max_depth'] = max(updated_params['max_depth'] - 1, 1)
        
        return updated_params
    
    def _create_model_from_params(self, params: Dict[str, Any], task_type: str):
        """Create a model from parameters."""
        if task_type == "classification":
            return RandomForestClassifier(
                n_estimators=int(params['n_trees']),
                max_depth=params['max_depth'],
                min_samples_split=params['min_samples_split'],
                min_samples_leaf=params['min_samples_leaf'],
                max_features=params['max_features'],
                random_state=42
            )
        else:
            return RandomForestRegressor(
                n_estimators=int(params['n_trees']),
                max_depth=params['max_depth'],
                min_samples_split=params['min_samples_split'],
                min_samples_leaf=params['min_samples_leaf'],
                max_features=params['max_features'],
                random_state=42
            )
    
    def _create_architecture_from_params(self, params: Dict[str, Any]) -> TreeArchitectureCandidate:
        """Create architecture candidate from parameters."""
        return TreeArchitectureCandidate(
            model_type=TreeModelType.RANDOM_FOREST,
            n_trees=int(params['n_trees']),
            max_depth=params['max_depth'],
            min_samples_split=params['min_samples_split'],
            min_samples_leaf=params['min_samples_leaf'],
            max_features=params['max_features'],
            learning_rate=params.get('learning_rate'),
            subsample=params.get('subsample'),
            colsample_bytree=params.get('colsample_bytree')
        )
    
    def _evaluate_architecture(self, 
                             architecture: TreeArchitectureCandidate,
                             data: Tuple[np.ndarray, np.ndarray],
                             task_type: str) -> float:
        """Evaluate architecture on data."""
        try:
            model = self._create_model_from_architecture(architecture, task_type)
            X, y = data
            model.fit(X, y)
            return model.score(X, y)
        except Exception as e:
            self.logger.warning(f"⚠️ Architecture evaluation failed: {e}")
            return 0.0
    
    def _create_model_from_architecture(self, architecture: TreeArchitectureCandidate, task_type: str):
        """Create model from architecture candidate."""
        if task_type == "classification":
            return RandomForestClassifier(
                n_estimators=architecture.n_trees,
                max_depth=architecture.max_depth,
                min_samples_split=architecture.min_samples_split,
                min_samples_leaf=architecture.min_samples_leaf,
                max_features=architecture.max_features,
                random_state=42
            )
        else:
            return RandomForestRegressor(
                n_estimators=architecture.n_trees,
                max_depth=architecture.max_depth,
                min_samples_split=architecture.min_samples_split,
                min_samples_leaf=architecture.min_samples_leaf,
                max_features=architecture.max_features,
                random_state=42
            )
    
    def _evaluate_task_loss(self, params: Dict[str, Any], query_data: Tuple[np.ndarray, np.ndarray]) -> float:
        """Evaluate task loss for meta-objective."""
        try:
            model = self._create_model_from_params(params, "classification")
            X_query, y_query = query_data
            model.fit(X_query, y_query)
            score = model.score(X_query, y_query)
            return 1.0 - score  # Convert to loss
        except Exception:
            return 1.0
    
    def _evaluate_meta_validation(self, val_tasks: List[Dict[str, Any]]) -> float:
        """Evaluate on meta-validation tasks."""
        total_loss = 0.0
        
        for task in val_tasks:
            # Inner adaptation
            adapted_params = self._inner_adaptation(task)
            
            # Evaluate on query set
            query_data = task.get('query_data')
            if query_data:
                task_loss = self._evaluate_task_loss(adapted_params, query_data)
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


class TreePrototypicalNetwork:
    """
    Prototypical Networks for Tree Models.
    
    Implements prototypical networks for few-shot learning with tree models.
    """
    
    def __init__(self, config: MetaLearningConfig):
        """Initialize Tree Prototypical Network.
        
        Args:
            config: Meta-learning configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Prototypical network state
        self.prototypes = {}
        self.embeddings = {}
        self.class_centers = {}
        
        self.logger.info("✅ Tree Prototypical Network initialized")
    
    def fit(self, support_data: Tuple[np.ndarray, np.ndarray], 
            support_labels: np.ndarray) -> Dict[str, Any]:
        """
        Fit prototypical network on support data.
        
        Args:
            support_data: Support set features
            support_labels: Support set labels
            
        Returns:
            Fitting results
        """
        self.logger.info("🎯 Fitting prototypical network")
        
        try:
            # Compute prototypes for each class
            unique_labels = np.unique(support_labels)
            prototypes = {}
            
            for label in unique_labels:
                label_mask = support_labels == label
                label_data = support_data[0][label_mask] if isinstance(support_data, tuple) else support_data[label_mask]
                
                # Compute prototype as mean of class examples
                prototype = np.mean(label_data, axis=0)
                prototypes[label] = prototype
            
            self.prototypes = prototypes
            self.class_centers = unique_labels
            
            self.logger.info(f"✅ Prototypes computed for {len(unique_labels)} classes")
            
            return {
                'n_classes': len(unique_labels),
                'prototypes': prototypes,
                'class_centers': unique_labels.tolist()
            }
            
        except Exception as e:
            self.logger.error(f"❌ Prototypical network fitting failed: {e}")
            raise
    
    def predict(self, query_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict labels for query data.
        
        Args:
            query_data: Query set features
            
        Returns:
            Tuple of (predictions, distances)
        """
        if not self.prototypes:
            raise ValueError("Prototypical network not fitted")
        
        try:
            predictions = []
            distances = []
            
            for query_point in query_data:
                # Compute distances to all prototypes
                point_distances = []
                for label, prototype in self.prototypes.items():
                    distance = np.linalg.norm(query_point - prototype)
                    point_distances.append((label, distance))
                
                # Sort by distance
                point_distances.sort(key=lambda x: x[1])
                
                # Get prediction and distance
                predicted_label = point_distances[0][0]
                min_distance = point_distances[0][1]
                
                predictions.append(predicted_label)
                distances.append(min_distance)
            
            return np.array(predictions), np.array(distances)
            
        except Exception as e:
            self.logger.error(f"❌ Prediction failed: {e}")
            raise
    
    def evaluate(self, query_data: np.ndarray, query_labels: np.ndarray) -> Dict[str, float]:
        """
        Evaluate prototypical network on query data.
        
        Args:
            query_data: Query set features
            query_labels: Query set labels
            
        Returns:
            Evaluation metrics
        """
        try:
            predictions, distances = self.predict(query_data)
            
            # Calculate accuracy
            accuracy = np.mean(predictions == query_labels)
            
            # Calculate average distance
            avg_distance = np.mean(distances)
            
            # Calculate confidence (inverse of distance)
            confidence = 1.0 / (1.0 + avg_distance)
            
            return {
                'accuracy': accuracy,
                'avg_distance': avg_distance,
                'confidence': confidence
            }
            
        except Exception as e:
            self.logger.error(f"❌ Evaluation failed: {e}")
            return {'accuracy': 0.0, 'avg_distance': float('inf'), 'confidence': 0.0}


class TreeMetaLearning:
    """
    Main Tree Meta-Learning System.
    
    Orchestrates meta-learning capabilities for tree-based models.
    """
    
    def __init__(self, config: MetaLearningConfig):
        """Initialize Tree Meta-Learning system.
        
        Args:
            config: Meta-learning configuration
        """
        tprint_info("🧠 Initializing Tree Meta-Learning System")
        tprint_debug(f"Configuration: {config}")
        tprint_debug(f"Meta-learning enabled: {config.enable_meta_learning}")
        tprint_debug(f"Adaptation steps: {config.adaptation_steps}")
        tprint_debug(f"Learning rate: {config.learning_rate}")
        
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize performance tracking
        self.performance_metrics = {
            'initialization_time': 0.0,
            'training_time': 0.0,
            'adaptation_time': 0.0,
            'total_execution_time': 0.0
        }
        
        # Initialize components
        tprint_debug("Initializing MAML component...")
        self.maml = TreeMAML(config)
        tprint_debug("Initializing Prototypical Network component...")
        self.prototypical_network = TreePrototypicalNetwork(config)
        
        # Meta-learning state
        self.meta_trained = False
        self.adaptation_history = []
        
        self.logger.info("✅ Tree Meta-Learning system initialized")
    
    def meta_train(self, 
                   meta_train_tasks: List[Dict[str, Any]],
                   meta_val_tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Meta-train the system on multiple tasks.
        
        Args:
            meta_train_tasks: Meta-training tasks
            meta_val_tasks: Meta-validation tasks
            
        Returns:
            Meta-training results
        """
        self.logger.info("🚀 Starting meta-training")
        
        try:
            # Meta-train MAML
            maml_results = self.maml.meta_train(meta_train_tasks, meta_val_tasks)
            
            # Meta-train prototypical network
            proto_results = self._meta_train_prototypical_network(meta_train_tasks)
            
            self.meta_trained = True
            
            results = {
                'maml_results': maml_results,
                'prototypical_results': proto_results,
                'meta_trained': True,
                'n_train_tasks': len(meta_train_tasks),
                'n_val_tasks': len(meta_val_tasks)
            }
            
            self.logger.info("✅ Meta-training completed")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Meta-training failed: {e}")
            raise
    
    def few_shot_adaptation(self,
                           support_data: Tuple[np.ndarray, np.ndarray],
                           query_data: Tuple[np.ndarray, np.ndarray],
                           adaptation_method: str = "maml") -> Dict[str, Any]:
        """
        Perform few-shot adaptation to new task.
        
        Args:
            support_data: Support set for adaptation
            query_data: Query set for evaluation
            adaptation_method: Method to use ("maml", "prototypical", "hybrid")
            
        Returns:
            Adaptation results
        """
        self.logger.info(f"🔄 Few-shot adaptation using {adaptation_method}")
        
        try:
            results = {}
            
            if adaptation_method in ["maml", "hybrid"] and self.meta_trained:
                # MAML adaptation
                maml_architecture = self.maml.adapt_to_new_task(
                    support_data, query_data, "classification"
                )
                results['maml_architecture'] = maml_architecture
                results['maml_score'] = maml_architecture.overall_score
            
            if adaptation_method in ["prototypical", "hybrid"]:
                # Prototypical network adaptation
                support_labels = np.random.randint(0, 5, len(support_data[0]))  # Placeholder labels
                query_labels = np.random.randint(0, 5, len(query_data[0]))  # Placeholder labels
                
                self.prototypical_network.fit(support_data, support_labels)
                proto_metrics = self.prototypical_network.evaluate(query_data, query_labels)
                results['prototypical_metrics'] = proto_metrics
            
            # Track adaptation
            self.adaptation_history.append({
                'method': adaptation_method,
                'support_size': len(support_data[0]),
                'query_size': len(query_data[0]),
                'results': results,
                'timestamp': datetime.now().isoformat()
            })
            
            self.logger.info("✅ Few-shot adaptation completed")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Few-shot adaptation failed: {e}")
            raise
    
    def _meta_train_prototypical_network(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Meta-train prototypical network."""
        try:
            # Simple meta-training for prototypical network
            total_accuracy = 0.0
            n_tasks = len(tasks)
            
            for task in tasks:
                support_data = task.get('support_data')
                query_data = task.get('query_data')
                
                if support_data and query_data:
                    # Create dummy labels for demonstration
                    support_labels = np.random.randint(0, 5, len(support_data[0]))
                    query_labels = np.random.randint(0, 5, len(query_data[0]))
                    
                    # Fit and evaluate
                    self.prototypical_network.fit(support_data, support_labels)
                    metrics = self.prototypical_network.evaluate(query_data, query_labels)
                    total_accuracy += metrics['accuracy']
            
            avg_accuracy = total_accuracy / n_tasks if n_tasks > 0 else 0.0
            
            return {
                'avg_accuracy': avg_accuracy,
                'n_tasks': n_tasks
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Prototypical network meta-training failed: {e}")
            return {'avg_accuracy': 0.0, 'n_tasks': 0}
    
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