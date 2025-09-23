"""
Neural Architecture Search Engine

This module provides the main NAS architecture search functionality,
including search strategies, evaluation, and optimization.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import logging
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
import time
import json
from pathlib import Path

from ..search.search_space import SearchSpace, ArchitectureConfig
from ..search.random_search import RandomSearch
from ..search.bayesian_search import BayesianSearch
from ..search.evolutionary_search import EvolutionarySearch
from ..evaluation.nas_metrics import NASMetrics
from ..utils.nas_utils import NASUtils
from ..utils.logging_utils import NASLogger

logger = logging.getLogger(__name__)

@dataclass
class NASSearchConfig:
    """Configuration for NAS architecture search."""
    max_iterations: int = 100
    max_time_seconds: int = 3600  # 1 hour
    min_improvement_threshold: float = 0.001
    max_no_improvement_rounds: int = 10
    validation_split: float = 0.2
    test_split: float = 0.1
    random_seed: int = 42

    # Search strategy
    search_strategy: str = "random"  # "random", "bayesian", "evolutionary"

    # Evaluation
    primary_metric: str = "accuracy"
    minimize_metric: bool = False

    # Hardware
    use_gpu: bool = True
    batch_size: int = 32
    num_workers: int = 4

    # Output
    save_best_architecture: bool = True
    save_search_history: bool = True
    output_dir: str = "nas_results"

@dataclass
class SearchResult:
    """Result of NAS architecture search."""
    best_architecture: ArchitectureConfig
    best_score: float
    best_model_state: Dict[str, Any]
    search_history: List[Dict[str, Any]]
    convergence_history: List[float]
    execution_time: float
    n_evaluations: int
    metadata: Dict[str, Any] = field(default_factory=dict)

class NASArchitectureSearch:
    """
    Main Neural Architecture Search Engine

    This class orchestrates the neural architecture search process,
    managing search strategies, evaluation, and optimization.
    """

    def __init__(self, config: NASSearchConfig):
        """Initialize NAS architecture search.

        Args:
            config: NAS search configuration
        """
        self.config = config
        self.logger = NASLogger.get_logger(self.__class__.__name__)

        # Initialize components
        self.search_space = SearchSpace()
        self.nas_utils = NASUtils()
        self.nas_metrics = NASMetrics()

        # Initialize search strategies
        self.search_strategies = {
            'random': RandomSearch(config),
            'bayesian': BayesianSearch(config),
            'evolutionary': EvolutionarySearch(config)
        }

        self.current_strategy = self.search_strategies[config.search_strategy]

        # Search state
        self.search_history = []
        self.best_architecture = None
        self.best_score = float('-inf') if not config.minimize_metric else float('inf')
        self.convergence_history = []
        self.no_improvement_count = 0

        # Hardware setup
        self.device = torch.device('cuda' if config.use_gpu and torch.cuda.is_available() else 'cpu')
        self.logger.info(f"🔧 NAS initialized with device: {self.device}")

    def search(self,
               train_data: Tuple[np.ndarray, np.ndarray],
               validation_data: Tuple[np.ndarray, np.ndarray],
               test_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
               problem_type: str = "classification",
               input_shape: Optional[Tuple[int, ...]] = None) -> SearchResult:
        """
        Perform neural architecture search.

        Args:
            train_data: Training data (X, y)
            validation_data: Validation data (X, y)
            test_data: Optional test data (X, y)
            problem_type: Type of problem ("classification", "regression", "hmm")
            input_shape: Input shape for the architecture

        Returns:
            SearchResult with best architecture and search history
        """
        start_time = time.time()
        self.logger.info("🚀 Starting Neural Architecture Search")
        self.logger.info(f"📊 Problem type: {problem_type}")
        self.logger.info(f"🔍 Search strategy: {self.config.search_strategy}")
        self.logger.info(f"⏰ Max iterations: {self.config.max_iterations}")

        # Setup data
        X_train, y_train = train_data
        X_val, y_val = validation_data

        # Prepare datasets
        train_dataset = self._prepare_dataset(X_train, y_train)
        val_dataset = self._prepare_dataset(X_val, y_val)
        test_dataset = self._prepare_dataset(test_data[0], test_data[1]) if test_data else None

        # Initialize search
        iteration = 0
        best_score = float('-inf') if not self.config.minimize_metric else float('inf')

        try:
            while iteration < self.config.max_iterations:
                iteration += 1

                # Generate architecture
                architecture = self.current_strategy.generate_architecture(iteration)
                if architecture is None:
                    self.logger.warning(f"⚠️ No architecture generated at iteration {iteration}")
                    continue

                self.logger.info(f"🔬 Evaluating architecture {iteration}: {architecture.name}")

                # Train and evaluate architecture
                try:
                    score = self._evaluate_architecture(
                        architecture, train_dataset, val_dataset, problem_type
                    )

                    # Track search history
                    result_entry = {
                        'iteration': iteration,
                        'architecture': architecture.to_dict(),
                        'score': score,
                        'timestamp': datetime.now().isoformat()
                    }
                    self.search_history.append(result_entry)

                    # Check if this is the best architecture
                    is_better = self._is_better_score(score, best_score)
                    if is_better:
                        best_score = score
                        self.best_architecture = architecture
                        self.best_score = score
                        self.no_improvement_count = 0

                        self.logger.info(f"🎯 New best architecture! Score: {score:.4f}")
                        self.logger.info(f"🏗️ Architecture: {architecture.name}")

                        # Save best architecture
                        if self.config.save_best_architecture:
                            self._save_best_architecture(architecture, score)
                    else:
                        self.no_improvement_count += 1

                    # Track convergence
                    self.convergence_history.append(best_score)

                    # Check convergence criteria
                    if self._check_convergence(iteration):
                        self.logger.info(f"✅ Convergence reached at iteration {iteration}")
                        break

                except Exception as e:
                    self.logger.error(f"❌ Error evaluating architecture {iteration}: {e}")
                    continue

                # Progress logging
                if iteration % 10 == 0:
                    self.logger.info(f"📈 Progress: {iteration}/{self.config.max_iterations} | Best Score: {best_score:.4f}")

                # Time limit check
                if time.time() - start_time > self.config.max_time_seconds:
                    self.logger.warning(f"⏰ Time limit reached after {iteration} iterations")
                    break

        except KeyboardInterrupt:
            self.logger.info("🛑 Search interrupted by user")
        except Exception as e:
            self.logger.error(f"❌ Search failed: {e}")

        # Final evaluation on test set if available
        test_score = None
        if test_dataset and self.best_architecture:
            try:
                test_score = self._evaluate_architecture(
                    self.best_architecture, train_dataset, test_dataset, problem_type, is_test=True
                )
                self.logger.info(f"🧪 Test set performance: {test_score:.4f}")
            except Exception as e:
                self.logger.warning(f"⚠️ Could not evaluate on test set: {e}")

        execution_time = time.time() - start_time

        # Create search result
        result = SearchResult(
            best_architecture=self.best_architecture,
            best_score=self.best_score,
            best_model_state=self._get_best_model_state(),
            search_history=self.search_history,
            convergence_history=self.convergence_history,
            execution_time=execution_time,
            n_evaluations=len(self.search_history),
            metadata={
                'problem_type': problem_type,
                'input_shape': input_shape,
                'train_samples': len(X_train),
                'val_samples': len(X_val),
                'test_samples': len(test_data[0]) if test_data else 0,
                'test_score': test_score,
                'search_strategy': self.config.search_strategy,
                'converged': self._check_convergence(iteration),
                'final_iteration': iteration
            }
        )

        self.logger.info(f"✅ NAS completed in {execution_time:.2f}s")
        self.logger.info(f"🏆 Best architecture: {result.best_architecture.name if result.best_architecture else 'None'}")
        self.logger.info(f"🎯 Best score: {result.best_score:.4f}")

        return result

    def _evaluate_architecture(self,
                             architecture: ArchitectureConfig,
                             train_dataset: Any,
                             val_dataset: Any,
                             problem_type: str,
                             is_test: bool = False) -> float:
        """
        Evaluate a single architecture.

        Args:
            architecture: Architecture to evaluate
            train_dataset: Training dataset
            val_dataset: Validation/test dataset
            problem_type: Type of problem
            is_test: Whether this is test evaluation

        Returns:
            Evaluation score
        """
        try:
            # Create model from architecture
            model = self._create_model_from_architecture(architecture, problem_type)

            # Train model
            trainer = NASTrainer(self.config)
            trained_model = trainer.train(model, train_dataset, val_dataset)

            # Evaluate model
            evaluator = NASEvaluator(self.config)
            score = evaluator.evaluate(trained_model, val_dataset, problem_type, self.config.primary_metric)

            return score

        except Exception as e:
            self.logger.error(f"❌ Architecture evaluation failed: {e}")
            return float('-inf') if not self.config.minimize_metric else float('inf')

    def _create_model_from_architecture(self, architecture: ArchitectureConfig, problem_type: str) -> nn.Module:
        """
        Create PyTorch model from architecture configuration.

        Args:
            architecture: Architecture configuration
            problem_type: Type of problem

        Returns:
            PyTorch model
        """
        # This will be implemented in the NASModel class
        # For now, create a basic model based on architecture
        return NASModel.create_from_config(architecture, problem_type)

    def _is_better_score(self, score: float, best_score: float) -> bool:
        """Check if score is better than current best."""
        if not self.config.minimize_metric:
            return score > best_score
        else:
            return score < best_score

    def _check_convergence(self, iteration: int) -> bool:
        """
        Check if search should converge.

        Args:
            iteration: Current iteration

        Returns:
            True if convergence criteria met
        """
        # Check no improvement rounds
        if self.no_improvement_count >= self.config.max_no_improvement_rounds:
            self.logger.info(f"🛑 No improvement for {self.no_improvement_count} rounds")
            return True

        # Check minimum improvement threshold
        if len(self.convergence_history) >= 2:
            recent_scores = self.convergence_history[-5:]  # Last 5 scores
            if len(recent_scores) >= 2:
                improvement = abs(recent_scores[-1] - recent_scores[-2])
                if improvement < self.config.min_improvement_threshold:
                    self.logger.info(f"🛑 Improvement {improvement:.6f} below threshold {self.config.min_improvement_threshold}")
                    return True

        return False

    def _save_best_architecture(self, architecture: ArchitectureConfig, score: float):
        """Save best architecture to file."""
        try:
            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            architecture_file = output_dir / "best_architecture.json"
            with open(architecture_file, 'w') as f:
                json.dump({
                    'architecture': architecture.to_dict(),
                    'score': score,
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2)

            self.logger.info(f"💾 Best architecture saved to {architecture_file}")

        except Exception as e:
            self.logger.warning(f"⚠️ Could not save best architecture: {e}")

    def _get_best_model_state(self) -> Dict[str, Any]:
        """Get state dict of best model."""
        # This would be implemented when we have the actual model
        # For now return empty dict
        return {}

    def _prepare_dataset(self, X: np.ndarray, y: np.ndarray) -> Any:
        """
        Prepare dataset for training.

        Args:
            X: Feature matrix
            y: Target vector

        Returns:
            Dataset object
        """
        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y) if isinstance(y[0], int) else torch.FloatTensor(y)

        return torch.utils.data.TensorDataset(X_tensor, y_tensor)

    def save_search_results(self, result: SearchResult, output_path: str):
        """Save complete search results."""
        try:
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)

            # Save search result
            result_file = output_path / "nas_search_result.json"
            with open(result_file, 'w') as f:
                # Convert result to dict for JSON serialization
                result_dict = {
                    'best_architecture': result.best_architecture.to_dict() if result.best_architecture else None,
                    'best_score': result.best_score,
                    'search_history': result.search_history,
                    'convergence_history': result.convergence_history,
                    'execution_time': result.execution_time,
                    'n_evaluations': result.n_evaluations,
                    'metadata': result.metadata
                }
                json.dump(result_dict, f, indent=2, default=str)

            self.logger.info(f"💾 Search results saved to {result_file}")

        except Exception as e:
            self.logger.error(f"❌ Failed to save search results: {e}")

    def load_search_results(self, input_path: str) -> Optional[SearchResult]:
        """Load search results from file."""
        try:
            input_path = Path(input_path)
            result_file = input_path / "nas_search_result.json"

            if not result_file.exists():
                self.logger.warning(f"⚠️ Search result file not found: {result_file}")
                return None

            with open(result_file, 'r') as f:
                result_dict = json.load(f)

            # Reconstruct architecture
            best_architecture = None
            if result_dict['best_architecture']:
                best_architecture = ArchitectureConfig.from_dict(result_dict['best_architecture'])

            result = SearchResult(
                best_architecture=best_architecture,
                best_score=result_dict['best_score'],
                best_model_state={},
                search_history=result_dict['search_history'],
                convergence_history=result_dict['convergence_history'],
                execution_time=result_dict['execution_time'],
                n_evaluations=result_dict['n_evaluations'],
                metadata=result_dict['metadata']
            )

            self.logger.info(f"📁 Search results loaded from {result_file}")
            return result

        except Exception as e:
            self.logger.error(f"❌ Failed to load search results: {e}")
            return None