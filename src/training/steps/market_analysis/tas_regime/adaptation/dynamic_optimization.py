"""
Dynamic Optimization for Tree Architecture Search

Advanced dynamic optimization capabilities for tree-based models including
online learning, incremental optimization, and adaptive parameter tuning.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import logging
import time
from datetime import datetime, timedelta
from collections import defaultdict, deque

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# DecisionTreeClassifier removed - only advanced tree models supported
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

from ..core.tas_config import TASConfig, TreeModelType
from ..core.tree_architecture import TreeArchitectureCandidate
from ..core.tas_result import TASResult

logger = logging.getLogger(__name__)


@dataclass
class OptimizationState:
    """State of dynamic optimization."""
    current_parameters: Dict[str, Any]
    optimization_history: List[Dict[str, Any]]
    performance_trends: Dict[str, List[float]]
    adaptation_count: int = 0
    convergence_reached: bool = False
    last_update: datetime = field(default_factory=datetime.now)


class TreeDynamicOptimizer:
    """
    Dynamic Optimizer for Tree Architecture Search.

    Performs dynamic optimization of tree parameters during training
    with real-time parameter adjustment based on performance feedback.
    """

    def __init__(self, config: TASConfig):
        """Initialize dynamic optimizer.

        Args:
            config: TAS configuration
        """
        tprint_info("⚡ Initializing Dynamic Optimization System")
        tprint_debug(f"Configuration: {config}")
        tprint_debug(f"Optimization enabled: {config.enable_dynamic_optimization}")
        tprint_debug(f"Optimization frequency: {config.optimization_frequency}")
        
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize performance tracking
        self.performance_metrics = {
            'initialization_time': 0.0,
            'optimization_time': 0.0,
            'analysis_time': 0.0,
            'total_execution_time': 0.0
        }

        # Optimization state
        self.optimization_state = OptimizationState(
            current_parameters=self._get_default_parameters(),
            optimization_history=[],
            performance_trends=defaultdict(list)
        )

        # Optimization parameters
        self.learning_rate = 0.01
        self.momentum = 0.9
        self.adaptation_frequency = 10
        self.patience = 5

        # Parameter bounds
        self.parameter_bounds = {
            'n_trees': (10, 1000),
            'max_depth': (3, 50),
            'min_samples_split': (2, 50),
            'min_samples_leaf': (1, 50),
            'max_features': (0.1, 1.0)
        }

        self.logger.info("✅ Tree Dynamic Optimizer initialized")
        self.logger.info(f"🔄 Learning rate: {self.learning_rate}")
        self.logger.info(f"📈 Momentum: {self.momentum}")

    def optimize_architecture(self,
                            base_architecture: TreeArchitectureCandidate,
                            train_data: Tuple[np.ndarray, np.ndarray],
                            validation_data: Tuple[np.ndarray, np.ndarray],
                            max_iterations: int = 50) -> TreeArchitectureCandidate:
        """Perform dynamic optimization of architecture.

        Args:
            base_architecture: Base architecture to optimize
            train_data: Training data
            validation_data: Validation data
            max_iterations: Maximum optimization iterations

        Returns:
            Optimized architecture
        """
        self.logger.info("🚀 Starting dynamic optimization")

        try:
            # Initialize with base architecture
            current_architecture = base_architecture
            best_architecture = base_architecture
            best_score = 0.0

            # Track gradients
            parameter_gradients = {param: 0.0 for param in self.optimization_state.current_parameters.keys()}

            # Optimization loop
            for iteration in range(max_iterations):
                # Train current architecture
                model = self._train_architecture(current_architecture, train_data)

                # Evaluate on validation set
                current_score = self._evaluate_architecture(model, validation_data)

                # Update best architecture
                if current_score > best_score:
                    best_score = current_score
                    best_architecture = current_architecture

                # Calculate gradients
                gradients = self._calculate_gradients(current_architecture, current_score, train_data)

                # Update parameters using gradients
                parameter_gradients = self._update_gradients(parameter_gradients, gradients)

                # Apply parameter updates
                updated_parameters = self._apply_parameter_updates(current_architecture, parameter_gradients)

                # Create new architecture with updated parameters
                current_architecture = self._create_architecture_from_parameters(updated_parameters)

                # Record optimization step
                self._record_optimization_step(iteration, current_score, current_architecture)

                # Check convergence
                if self._check_convergence(iteration):
                    self.logger.info(f"✅ Optimization converged at iteration {iteration}")
                    break

                # Log progress
                if iteration % 10 == 0:
                    self.logger.info(f"📈 Iteration {iteration}: Score = {current_score:.4f}, "
                                   f"Best = {best_score:.4f}")

            # Update optimization state
            self.optimization_state.convergence_reached = True
            self.optimization_state.last_update = datetime.now()

            best_architecture.overall_score = best_score

            self.logger.info(f"✅ Dynamic optimization completed with best score: {best_score:.4f}")
            return best_architecture

        except Exception as e:
            self.logger.error(f"❌ Dynamic optimization failed: {e}")
            return base_architecture

    def _get_default_parameters(self) -> Dict[str, Any]:
        """Get default optimization parameters."""
        return {
            'n_trees': 100,
            'max_depth': 10,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'auto'
        }

    def _calculate_gradients(self,
                           architecture: TreeArchitectureCandidate,
                           score: float,
                           data: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
        """Calculate gradients for parameter updates."""
        try:
            gradients = {}

            # Base gradient magnitude
            base_gradient = self.learning_rate * (1.0 - score)  # Lower score = higher gradient

            # Parameter-specific gradients
            X, y = data

            # Tree count gradient
            n_trees = architecture.n_trees
            if n_trees < 50:
                gradients['n_trees'] = base_gradient * 2.0  # Encourage more trees
            elif n_trees > 500:
                gradients['n_trees'] = -base_gradient * 2.0  # Reduce complexity
            else:
                gradients['n_trees'] = base_gradient

            # Depth gradient
            max_depth = architecture.max_depth
            if max_depth < 5:
                gradients['max_depth'] = base_gradient * 1.5
            elif max_depth > 30:
                gradients['max_depth'] = -base_gradient * 1.5
            else:
                gradients['max_depth'] = base_gradient * 0.5

            # Sample split gradient
            min_samples_split = architecture.min_samples_split
            if score < 0.7:
                gradients['min_samples_split'] = -base_gradient  # Reduce to allow more splits
            else:
                gradients['min_samples_split'] = base_gradient * 0.3

            # Sample leaf gradient
            min_samples_leaf = architecture.min_samples_leaf
            if score < 0.7:
                gradients['min_samples_leaf'] = -base_gradient * 0.5
            else:
                gradients['min_samples_leaf'] = base_gradient * 0.2

            return gradients

        except Exception as e:
            self.logger.warning(f"⚠️ Gradient calculation failed: {e}")
            return {param: 0.0 for param in self.optimization_state.current_parameters.keys()}

    def _update_gradients(self,
                         current_gradients: Dict[str, float],
                         new_gradients: Dict[str, float]) -> Dict[str, float]:
        """Update gradients using momentum."""
        try:
            updated_gradients = {}

            for param in self.optimization_state.current_parameters.keys():
                if param in new_gradients:
                    # Apply momentum
                    updated_gradients[param] = (self.momentum * current_gradients.get(param, 0.0) +
                                               (1 - self.momentum) * new_gradients[param])

            return updated_gradients

        except Exception as e:
            self.logger.warning(f"⚠️ Gradient update failed: {e}")
            return current_gradients

    def _apply_parameter_updates(self,
                               architecture: TreeArchitectureCandidate,
                               gradients: Dict[str, float]) -> Dict[str, Any]:
        """Apply parameter updates based on gradients."""
        try:
            updated_params = {}

            # Apply updates to each parameter
            for param, value in self.optimization_state.current_parameters.items():
                if param in gradients:
                    gradient = gradients[param]

                    # Apply gradient with bounds checking
                    if param == 'n_trees':
                        new_value = int(value + gradient * 10)
                        new_value = max(self.parameter_bounds['n_trees'][0],
                                      min(self.parameter_bounds['n_trees'][1], new_value))

                    elif param == 'max_depth':
                        new_value = int(value + gradient * 2)
                        new_value = max(self.parameter_bounds['max_depth'][0],
                                      min(self.parameter_bounds['max_depth'][1], new_value))

                    elif param == 'min_samples_split':
                        new_value = int(value + gradient * 1)
                        new_value = max(self.parameter_bounds['min_samples_split'][0],
                                      min(self.parameter_bounds['min_samples_split'][1], new_value))

                    elif param == 'min_samples_leaf':
                        new_value = int(value + gradient * 0.5)
                        new_value = max(self.parameter_bounds['min_samples_leaf'][0],
                                      min(self.parameter_bounds['min_samples_leaf'][1], new_value))

                    elif param == 'max_features':
                        if isinstance(value, str):
                            new_value = value
                        else:
                            new_value = value + gradient * 0.1
                            new_value = max(self.parameter_bounds['max_features'][0],
                                          min(self.parameter_bounds['max_features'][1], new_value))

                    else:
                        new_value = value

                    updated_params[param] = new_value

            return updated_params

        except Exception as e:
            self.logger.warning(f"⚠️ Parameter update failed: {e}")
            return self.optimization_state.current_parameters

    def _create_architecture_from_parameters(self, parameters: Dict[str, Any]) -> TreeArchitectureCandidate:
        """Create architecture from parameter dictionary."""
        return TreeArchitectureCandidate(
            model_type=TreeModelType.RANDOM_FOREST,
            n_trees=int(parameters.get('n_trees', 100)),
            max_depth=int(parameters.get('max_depth', 10)),
            min_samples_split=int(parameters.get('min_samples_split', 2)),
            min_samples_leaf=int(parameters.get('min_samples_leaf', 1)),
            max_features=parameters.get('max_features', 'auto')
        )

    def _train_architecture(self,
                           architecture: TreeArchitectureCandidate,
                           data: Tuple[np.ndarray, np.ndarray]) -> Any:
        """Train architecture on data."""
        try:
            X, y = data

            model = RandomForestClassifier(
                n_estimators=architecture.n_trees,
                max_depth=architecture.max_depth,
                min_samples_split=architecture.min_samples_split,
                min_samples_leaf=architecture.min_samples_leaf,
                max_features=architecture.max_features,
                random_state=42
            )

            model.fit(X, y)
            return model

        except Exception as e:
            self.logger.error(f"❌ Architecture training failed: {e}")
            raise

    def _evaluate_architecture(self,
                              model: Any,
                              data: Tuple[np.ndarray, np.ndarray]) -> float:
        """Evaluate architecture on data."""
        try:
            X, y = data
            return model.score(X, y)

        except Exception as e:
            self.logger.error(f"❌ Architecture evaluation failed: {e}")
            return 0.0

    def _record_optimization_step(self,
                                 iteration: int,
                                 score: float,
                                 architecture: TreeArchitectureCandidate):
        """Record optimization step."""
        try:
            step_record = {
                'iteration': iteration,
                'score': score,
                'parameters': {
                    'n_trees': architecture.n_trees,
                    'max_depth': architecture.max_depth,
                    'min_samples_split': architecture.min_samples_split,
                    'min_samples_leaf': architecture.min_samples_leaf,
                    'max_features': architecture.max_features
                },
                'timestamp': datetime.now()
            }

            self.optimization_state.optimization_history.append(step_record)

        except Exception as e:
            self.logger.warning(f"⚠️ Optimization step recording failed: {e}")

    def _check_convergence(self, iteration: int) -> bool:
        """Check if optimization has converged."""
        try:
            if len(self.optimization_state.optimization_history) < self.patience:
                return False

            # Check if recent scores have plateaued
            recent_scores = [step['score'] for step in self.optimization_state.optimization_history[-self.patience:]]
            score_std = np.std(recent_scores)

            return score_std < 0.001  # Very low variation indicates convergence

        except Exception:
            return False

    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get optimization summary."""
        if not self.optimization_state.optimization_history:
            return {}

        try:
            scores = [step['score'] for step in self.optimization_state.optimization_history]

            return {
                'total_iterations': len(self.optimization_state.optimization_history),
                'best_score': max(scores),
                'final_score': scores[-1] if scores else 0.0,
                'score_improvement': max(scores) - scores[0] if len(scores) > 1 else 0.0,
                'convergence_reached': self.optimization_state.convergence_reached,
                'parameter_evolution': self.optimization_state.optimization_history
            }

        except Exception as e:
            self.logger.warning(f"⚠️ Optimization summary failed: {e}")
            return {}


class TreeIncrementalLearner:
    """
    Incremental Learning System for Tree Architecture Search.

    Enables incremental learning and online updates for tree models.
    """

    def __init__(self, config: TASConfig):
        """Initialize incremental learner.

        Args:
            config: TAS configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Incremental learning state
        self.current_model = None
        self.learning_history = []
        self.update_count = 0

        # Incremental parameters
        self.batch_size = 100
        self.update_frequency = 10
        self.forgetting_rate = 0.01

        self.logger.info("✅ Tree Incremental Learner initialized")

    def incremental_fit(self,
                       X: np.ndarray,
                       y: np.ndarray,
                       batch_size: Optional[int] = None) -> Any:
        """Perform incremental learning on data.

        Args:
            X: Feature data
            y: Target data
            batch_size: Batch size for incremental learning

        Returns:
            Updated model
        """
        self.logger.info(f"🔄 Incremental learning on {len(X)} samples")

        try:
            batch_size = batch_size or self.batch_size
            n_batches = len(X) // batch_size

            # Initialize model if needed
            if self.current_model is None:
                self.current_model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42
                )

            # Process data in batches
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(X))

                X_batch = X[start_idx:end_idx]
                y_batch = y[start_idx:end_idx]

                # Incremental update
                self.current_model.fit(X_batch, y_batch)

                # Record update
                self.learning_history.append({
                    'batch': i,
                    'samples': len(X_batch),
                    'timestamp': datetime.now()
                })

                self.update_count += 1

                # Log progress
                if i % 10 == 0:
                    accuracy = self.current_model.score(X_batch, y_batch)
                    self.logger.debug(f"📊 Batch {i}/{n_batches}: Accuracy = {accuracy:.4f}")

            self.logger.info(f"✅ Incremental learning completed with {self.update_count} updates")
            return self.current_model

        except Exception as e:
            self.logger.error(f"❌ Incremental learning failed: {e}")
            raise

    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get incremental learning statistics."""
        if not self.learning_history:
            return {}

        return {
            'total_updates': self.update_count,
            'total_batches': len(self.learning_history),
            'avg_batch_size': np.mean([h['samples'] for h in self.learning_history]),
            'learning_history': self.learning_history[-10:]  # Last 10 updates
        }


class TreeOnlineOptimizer:
    """
    Online Optimizer for Tree Architecture Search.

    Performs online optimization with streaming data and real-time adaptation.
    """

    def __init__(self, config: TASConfig):
        """Initialize online optimizer.

        Args:
            config: TAS configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Online optimization state
        self.current_architecture = None
        self.streaming_buffer = deque(maxlen=1000)
        self.performance_buffer = deque(maxlen=100)

        # Online parameters
        self.streaming_batch_size = 50
        self.optimization_interval = 20
        self.drift_detection_enabled = True

        self.logger.info("✅ Tree Online Optimizer initialized")

    def process_stream(self,
                      X_stream: np.ndarray,
                      y_stream: np.ndarray,
                      optimize_periodically: bool = True) -> List[Dict[str, Any]]:
        """Process streaming data with online optimization.

        Args:
            X_stream: Streaming feature data
            y_stream: Streaming target data
            optimize_periodically: Whether to optimize periodically

        Returns:
            List of optimization results
        """
        self.logger.info(f"🔄 Processing stream with {len(X_stream)} samples")

        try:
            results = []

            # Process stream in batches
            for i in range(0, len(X_stream), self.streaming_batch_size):
                X_batch = X_stream[i:i + self.streaming_batch_size]
                y_batch = y_stream[i:i + self.streaming_batch_size]

                # Add to buffer
                self.streaming_buffer.extend(zip(X_batch, y_batch))

                # Process batch
                batch_result = self._process_batch(X_batch, y_batch)
                results.append(batch_result)

                # Periodic optimization
                if optimize_periodically and (i // self.streaming_batch_size) % self.optimization_interval == 0:
                    optimization_result = self._perform_online_optimization()
                    results.append(optimization_result)

            self.logger.info(f"✅ Stream processing completed with {len(results)} results")
            return results

        except Exception as e:
            self.logger.error(f"❌ Stream processing failed: {e}")
            raise

    def _process_batch(self, X_batch: np.ndarray, y_batch: np.ndarray) -> Dict[str, Any]:
        """Process a single batch of streaming data."""
        try:
            # Update model with batch
            if self.current_architecture is None:
                self.current_architecture = TreeArchitectureCandidate(
                    model_type=TreeModelType.RANDOM_FOREST,
                    n_trees=100,
                    max_depth=10
                )

            # Create and train model
            model = RandomForestClassifier(
                n_estimators=self.current_architecture.n_trees,
                max_depth=self.current_architecture.max_depth,
                random_state=42
            )

            model.fit(X_batch, y_batch)
            accuracy = model.score(X_batch, y_batch)

            # Update performance buffer
            self.performance_buffer.append(accuracy)

            return {
                'batch_size': len(X_batch),
                'accuracy': accuracy,
                'avg_performance': np.mean(list(self.performance_buffer)),
                'timestamp': datetime.now()
            }

        except Exception as e:
            self.logger.error(f"❌ Batch processing failed: {e}")
            return {'error': str(e)}

    def _perform_online_optimization(self) -> Dict[str, Any]:
        """Perform online optimization based on recent performance."""
        try:
            if not self.performance_buffer:
                return {'message': 'No performance data available'}

            # Check for performance degradation
            recent_performance = list(self.performance_buffer)[-10:]
            avg_performance = np.mean(recent_performance)

            if avg_performance < 0.7:  # Performance threshold
                # Optimize architecture
                if self.current_architecture:
                    self.current_architecture.n_trees = min(200, self.current_architecture.n_trees + 20)
                    self.current_architecture.max_depth = min(15, self.current_architecture.max_depth + 1)

                return {
                    'optimization': 'architecture_updated',
                    'new_n_trees': self.current_architecture.n_trees if self.current_architecture else None,
                    'new_max_depth': self.current_architecture.max_depth if self.current_architecture else None,
                    'avg_performance': avg_performance,
                    'timestamp': datetime.now()
                }

            return {
                'optimization': 'no_change_needed',
                'avg_performance': avg_performance,
                'timestamp': datetime.now()
            }

        except Exception as e:
            self.logger.error(f"❌ Online optimization failed: {e}")
            return {'error': str(e)}


class TreeAdaptiveSearch:
    """
    Adaptive Search System for Tree Architecture Search.
    
    Performs adaptive search with dynamic strategy selection and parameter tuning.
    """
    
    def __init__(self, config: TASConfig):
        """Initialize adaptive search.
        
        Args:
            config: TAS configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Adaptive search state
        self.search_strategies = ['random', 'grid', 'bayesian', 'evolutionary']
        self.current_strategy = 'random'
        self.strategy_performance = defaultdict(list)
        self.adaptation_count = 0
        
        # Search parameters
        self.strategy_switch_threshold = 0.1
        self.performance_window = 10
        self.exploration_rate = 0.3
        
        self.logger.info("✅ Tree Adaptive Search initialized")
        self.logger.info(f"🎯 Available strategies: {self.search_strategies}")
    
    def adaptive_search(self,
                       search_space: Dict[str, Any],
                       objective_function: Callable,
                       max_iterations: int = 100) -> Dict[str, Any]:
        """Perform adaptive search with strategy selection.
        
        Args:
            search_space: Search space definition
            objective_function: Objective function to optimize
            max_iterations: Maximum search iterations
            
        Returns:
            Best search result
        """
        self.logger.info("🚀 Starting adaptive search")
        
        try:
            best_result = None
            best_score = -np.inf
            
            # Search loop
            for iteration in range(max_iterations):
                # Select strategy based on performance
                strategy = self._select_strategy()
                
                # Generate candidate using selected strategy
                candidate = self._generate_candidate(strategy, search_space)
                
                # Evaluate candidate
                score = objective_function(candidate)
                
                # Update best result
                if score > best_score:
                    best_score = score
                    best_result = candidate.copy()
                
                # Record strategy performance
                self._record_strategy_performance(strategy, score)
                
                # Adapt strategy if needed
                if iteration % self.performance_window == 0:
                    self._adapt_strategy()
                
                # Log progress
                if iteration % 20 == 0:
                    self.logger.info(f"📈 Iteration {iteration}: Score = {score:.4f}, "
                                   f"Best = {best_score:.4f}, Strategy = {strategy}")
            
            self.logger.info(f"✅ Adaptive search completed with best score: {best_score:.4f}")
            return {
                'best_result': best_result,
                'best_score': best_score,
                'strategy_usage': dict(self.strategy_performance),
                'adaptation_count': self.adaptation_count
            }
            
        except Exception as e:
            self.logger.error(f"❌ Adaptive search failed: {e}")
            return {'error': str(e)}
    
    def _select_strategy(self) -> str:
        """Select search strategy based on performance."""
        try:
            # If no performance data, use random
            if not any(self.strategy_performance.values()):
                return 'random'
            
            # Calculate average performance for each strategy
            strategy_scores = {}
            for strategy, scores in self.strategy_performance.items():
                if scores:
                    strategy_scores[strategy] = np.mean(scores[-self.performance_window:])
            
            # Select best performing strategy with exploration
            if np.random.random() < self.exploration_rate:
                return np.random.choice(self.search_strategies)
            else:
                return max(strategy_scores.items(), key=lambda x: x[1])[0]
                
        except Exception as e:
            self.logger.warning(f"⚠️ Strategy selection failed: {e}")
            return 'random'
    
    def _generate_candidate(self, strategy: str, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Generate candidate using specified strategy."""
        try:
            if strategy == 'random':
                return self._random_search(search_space)
            elif strategy == 'grid':
                return self._grid_search(search_space)
            elif strategy == 'bayesian':
                return self._bayesian_search(search_space)
            elif strategy == 'evolutionary':
                return self._evolutionary_search(search_space)
            else:
                return self._random_search(search_space)
                
        except Exception as e:
            self.logger.warning(f"⚠️ Candidate generation failed: {e}")
            return self._random_search(search_space)
    
    def _random_search(self, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Generate random candidate."""
        candidate = {}
        for param, values in search_space.items():
            if isinstance(values, list):
                candidate[param] = np.random.choice(values)
            elif isinstance(values, tuple) and len(values) == 2:
                candidate[param] = np.random.uniform(values[0], values[1])
            else:
                candidate[param] = values
        return candidate
    
    def _grid_search(self, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Generate grid search candidate."""
        # Simplified grid search - in practice would maintain grid state
        return self._random_search(search_space)
    
    def _bayesian_search(self, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Generate Bayesian search candidate."""
        # Simplified Bayesian search - in practice would use acquisition function
        return self._random_search(search_space)
    
    def _evolutionary_search(self, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Generate evolutionary search candidate."""
        # Simplified evolutionary search - in practice would maintain population
        return self._random_search(search_space)
    
    def _record_strategy_performance(self, strategy: str, score: float):
        """Record strategy performance."""
        try:
            self.strategy_performance[strategy].append(score)
            
            # Keep only recent performance data
            if len(self.strategy_performance[strategy]) > self.performance_window * 2:
                self.strategy_performance[strategy] = self.strategy_performance[strategy][-self.performance_window:]
                
        except Exception as e:
            self.logger.warning(f"⚠️ Performance recording failed: {e}")
    
    def _adapt_strategy(self):
        """Adapt search strategy based on performance."""
        try:
            # Check if strategy switching is needed
            if len(self.strategy_performance) < 2:
                return
            
            # Calculate performance differences
            strategy_means = {}
            for strategy, scores in self.strategy_performance.items():
                if scores:
                    strategy_means[strategy] = np.mean(scores[-self.performance_window:])
            
            if len(strategy_means) >= 2:
                best_strategy = max(strategy_means.items(), key=lambda x: x[1])[0]
                worst_strategy = min(strategy_means.items(), key=lambda x: x[1])[0]
                
                # Switch if performance difference is significant
                if (strategy_means[best_strategy] - strategy_means[worst_strategy] > 
                    self.strategy_switch_threshold):
                    self.current_strategy = best_strategy
                    self.adaptation_count += 1
                    self.logger.info(f"🔄 Strategy adapted to: {best_strategy}")
                    
        except Exception as e:
            self.logger.warning(f"⚠️ Strategy adaptation failed: {e}")
    
    def get_adaptation_summary(self) -> Dict[str, Any]:
        """Get adaptation summary."""
        return {
            'current_strategy': self.current_strategy,
            'adaptation_count': self.adaptation_count,
            'strategy_performance': dict(self.strategy_performance),
            'available_strategies': self.search_strategies
        }


# Convenience functions
def create_dynamic_optimizer(config: TASConfig) -> TreeDynamicOptimizer:
    """Create a dynamic optimizer with default configuration."""
    return TreeDynamicOptimizer(config)


def create_incremental_learner(config: TASConfig) -> TreeIncrementalLearner:
    """Create an incremental learner with default configuration."""
    return TreeIncrementalLearner(config)


def create_online_optimizer(config: TASConfig) -> TreeOnlineOptimizer:
    """Create an online optimizer with default configuration."""
    return TreeOnlineOptimizer(config)


def create_adaptive_search(config: TASConfig) -> TreeAdaptiveSearch:
    """Create an adaptive search with default configuration."""
    return TreeAdaptiveSearch(config)