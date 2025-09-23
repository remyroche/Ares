"""
Neural Architecture Search Evaluator

This module provides evaluation functionality for NAS models,
including various metrics and evaluation strategies.
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix
import time

from ..utils.nas_utils import NASUtils
from ..utils.logging_utils import NASLogger

logger = logging.getLogger(__name__)

@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""
    batch_size: int = 32
    num_workers: int = 4
    use_gpu: bool = True
    mixed_precision: bool = True

    # Metrics to compute
    compute_confusion_matrix: bool = True
    compute_per_class_metrics: bool = True
    compute_predictions: bool = True

    # HMM-specific
    compute_hmm_metrics: bool = False
    compute_regime_stability: bool = False

@dataclass
class EvaluationResult:
    """Result of model evaluation."""
    loss: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    predictions: Optional[np.ndarray] = None
    targets: Optional[np.ndarray] = None
    confusion_matrix: Optional[np.ndarray] = None
    per_class_metrics: Optional[Dict[str, Any]] = None
    execution_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class NASEvaluator:
    """
    Neural Architecture Search Evaluator

    Handles evaluation of NAS models with comprehensive metrics
    for different problem types (classification, regression, HMM).
    """

    def __init__(self, config: EvaluationConfig):
        """Initialize NAS evaluator.

        Args:
            config: Evaluation configuration
        """
        self.config = config
        self.logger = NASLogger.get_logger(self.__class__.__name__)

        # Setup device
        self.device = torch.device('cuda' if config.use_gpu and torch.cuda.is_available() else 'cpu')

        # Metric functions
        self.classification_metrics = {
            'accuracy': accuracy_score,
            'precision_macro': lambda y_true, y_pred: precision_score(y_true, y_pred, average='macro'),
            'precision_micro': lambda y_true, y_pred: precision_score(y_true, y_pred, average='micro'),
            'recall_macro': lambda y_true, y_pred: recall_score(y_true, y_pred, average='macro'),
            'recall_micro': lambda y_true, y_pred: recall_score(y_true, y_pred, average='micro'),
            'f1_macro': lambda y_true, y_pred: f1_score(y_true, y_pred, average='macro'),
            'f1_micro': lambda y_true, y_pred: f1_score(y_true, y_pred, average='micro')
        }

        self.regression_metrics = {
            'mse': mean_squared_error,
            'rmse': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score,
            'mae': lambda y_true, y_pred: np.mean(np.abs(y_true - y_pred))
        }

    def evaluate(self,
                 model: nn.Module,
                 dataset: Dataset,
                 problem_type: str,
                 metric_name: str = "accuracy") -> float:
        """
        Evaluate a NAS model.

        Args:
            model: Model to evaluate
            dataset: Evaluation dataset
            problem_type: Type of problem
            metric_name: Name of primary metric to return

        Returns:
            Primary metric value
        """
        start_time = time.time()

        # Move model to device
        model = model.to(self.device)
        model.eval()

        # Create data loader
        data_loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )

        # Evaluate model
        result = self._evaluate_model(model, data_loader, problem_type)

        execution_time = time.time() - start_time
        result.execution_time = execution_time

        self.logger.info(f"📊 Evaluation completed in {execution_time:.2f}s")
        self.logger.info(f"🎯 {metric_name}: {result.__dict__.get(metric_name, result.accuracy):.4f}")

        # Return primary metric
        return result.__dict__.get(metric_name, result.accuracy)

    def _evaluate_model(self,
                       model: nn.Module,
                       data_loader: DataLoader,
                       problem_type: str) -> EvaluationResult:
        """Evaluate model on dataset.

        Args:
            model: Model to evaluate
            data_loader: Data loader
            problem_type: Type of problem

        Returns:
            EvaluationResult
        """
        all_predictions = []
        all_targets = []
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss() if problem_type == "classification" else nn.MSELoss()

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                data, target = data.to(self.device), target.to(self.device)

                # Forward pass
                output = model(data)
                loss = criterion(output, target)
                total_loss += loss.item()

                # Get predictions
                if isinstance(criterion, nn.CrossEntropyLoss):
                    predictions = output.argmax(dim=1).cpu().numpy()
                else:
                    predictions = output.squeeze().cpu().numpy()

                targets = target.cpu().numpy()

                all_predictions.extend(predictions)
                all_targets.extend(targets)

        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        avg_loss = total_loss / len(data_loader)

        # Calculate metrics based on problem type
        if problem_type == "classification":
            return self._evaluate_classification(all_predictions, all_targets, avg_loss)
        elif problem_type == "regression":
            return self._evaluate_regression(all_predictions, all_targets, avg_loss)
        elif problem_type == "hmm":
            return self._evaluate_hmm(all_predictions, all_targets, avg_loss)
        else:
            # Default to classification
            return self._evaluate_classification(all_predictions, all_targets, avg_loss)

    def _evaluate_classification(self,
                                predictions: np.ndarray,
                                targets: np.ndarray,
                                loss: float) -> EvaluationResult:
        """Evaluate classification model.

        Args:
            predictions: Model predictions
            targets: Ground truth labels
            loss: Average loss

        Returns:
            EvaluationResult for classification
        """
        # Basic metrics
        accuracy = accuracy_score(targets, predictions)
        precision_macro = precision_score(targets, predictions, average='macro', zero_division=0)
        precision_micro = precision_score(targets, predictions, average='micro', zero_division=0)
        recall_macro = recall_score(targets, predictions, average='macro', zero_division=0)
        recall_micro = recall_score(targets, predictions, average='micro', zero_division=0)
        f1_macro = f1_score(targets, predictions, average='macro', zero_division=0)
        f1_micro = f1_score(targets, predictions, average='micro', zero_division=0)

        # Confusion matrix
        confusion_matrix = None
        if self.config.compute_confusion_matrix:
            confusion_matrix = confusion_matrix(targets, predictions)

        # Per-class metrics
        per_class_metrics = None
        if self.config.compute_per_class_metrics:
            per_class_metrics = self._calculate_per_class_metrics(targets, predictions)

        return EvaluationResult(
            loss=loss,
            accuracy=accuracy,
            precision=precision_macro,
            recall=recall_macro,
            f1_score=f1_macro,
            predictions=predictions if self.config.compute_predictions else None,
            targets=targets if self.config.compute_predictions else None,
            confusion_matrix=confusion_matrix,
            per_class_metrics=per_class_metrics,
            metadata={
                'n_samples': len(targets),
                'n_classes': len(np.unique(targets)),
                'precision_micro': precision_micro,
                'recall_micro': recall_micro,
                'f1_micro': f1_micro
            }
        )

    def _evaluate_regression(self,
                            predictions: np.ndarray,
                            targets: np.ndarray,
                            loss: float) -> EvaluationResult:
        """Evaluate regression model.

        Args:
            predictions: Model predictions
            targets: Ground truth values
            loss: Average loss

        Returns:
            EvaluationResult for regression
        """
        mse = mean_squared_error(targets, predictions)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(targets - predictions))
        r2 = r2_score(targets, predictions)

        return EvaluationResult(
            loss=loss,
            accuracy=r2,  # Use R² as accuracy for regression
            precision=mae,  # Use MAE as precision
            recall=rmse,  # Use RMSE as recall
            f1_score=mse,  # Use MSE as F1
            predictions=predictions if self.config.compute_predictions else None,
            targets=targets if self.config.compute_predictions else None,
            metadata={
                'n_samples': len(targets),
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            }
        )

    def _evaluate_hmm(self,
                     predictions: np.ndarray,
                     targets: np.ndarray,
                     loss: float) -> EvaluationResult:
        """Evaluate HMM model.

        Args:
            predictions: Model predictions (state probabilities)
            targets: Ground truth states
            loss: Average loss

        Returns:
            EvaluationResult for HMM
        """
        # Basic classification metrics
        accuracy = accuracy_score(targets, predictions)
        precision_macro = precision_score(targets, predictions, average='macro', zero_division=0)
        recall_macro = recall_score(targets, predictions, average='macro', zero_division=0)
        f1_macro = f1_score(targets, predictions, average='macro', zero_division=0)

        # HMM-specific metrics
        hmm_metrics = {}
        if self.config.compute_hmm_metrics:
            hmm_metrics = self._calculate_hmm_metrics(predictions, targets)

        # Regime stability (if enabled)
        regime_stability = None
        if self.config.compute_regime_stability:
            regime_stability = self._calculate_regime_stability(predictions)

        return EvaluationResult(
            loss=loss,
            accuracy=accuracy,
            precision=precision_macro,
            recall=recall_macro,
            f1_score=f1_macro,
            predictions=predictions if self.config.compute_predictions else None,
            targets=targets if self.config.compute_predictions else None,
            metadata={
                'n_samples': len(targets),
                'n_states': len(np.unique(targets)),
                'hmm_metrics': hmm_metrics,
                'regime_stability': regime_stability
            }
        )

    def _calculate_per_class_metrics(self, targets: np.ndarray, predictions: np.ndarray) -> Dict[str, Any]:
        """Calculate per-class metrics.

        Args:
            targets: Ground truth labels
            predictions: Model predictions

        Returns:
            Dictionary with per-class metrics
        """
        n_classes = len(np.unique(targets))
        per_class = {}

        for class_idx in range(n_classes):
            class_mask = targets == class_idx
            if np.any(class_mask):
                class_true = targets[class_mask]
                class_pred = predictions[class_mask]

                per_class[f'class_{class_idx}'] = {
                    'support': len(class_true),
                    'accuracy': accuracy_score(class_true, class_pred),
                    'precision': precision_score(class_true, class_pred, average='macro', zero_division=0),
                    'recall': recall_score(class_true, class_pred, average='macro', zero_division=0),
                    'f1': f1_score(class_true, class_pred, average='macro', zero_division=0)
                }

        return per_class

    def _calculate_hmm_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, Any]:
        """Calculate HMM-specific metrics.

        Args:
            predictions: Model predictions
            targets: Ground truth labels

        Returns:
            Dictionary with HMM metrics
        """
        hmm_metrics = {}

        # State transition consistency
        transitions_true = self._calculate_transitions(targets)
        transitions_pred = self._calculate_transitions(predictions)

        # Transition accuracy
        transition_accuracy = np.mean([
            transitions_true[i, j] == transitions_pred[i, j]
            for i in range(transitions_true.shape[0])
            for j in range(transitions_true.shape[1])
        ])

        hmm_metrics['transition_accuracy'] = transition_accuracy

        # State persistence (how long states tend to persist)
        state_persistence_true = self._calculate_state_persistence(targets)
        state_persistence_pred = self._calculate_state_persistence(predictions)

        hmm_metrics['state_persistence_true'] = state_persistence_true
        hmm_metrics['state_persistence_pred'] = state_persistence_pred
        hmm_metrics['persistence_error'] = abs(state_persistence_true - state_persistence_pred)

        # State entropy (diversity of state usage)
        state_entropy_true = self._calculate_state_entropy(targets)
        state_entropy_pred = self._calculate_state_entropy(predictions)

        hmm_metrics['state_entropy_true'] = state_entropy_true
        hmm_metrics['state_entropy_pred'] = state_entropy_pred
        hmm_metrics['entropy_error'] = abs(state_entropy_true - state_entropy_pred)

        return hmm_metrics

    def _calculate_transitions(self, states: np.ndarray) -> np.ndarray:
        """Calculate state transition matrix.

        Args:
            states: State sequence

        Returns:
            Transition matrix
        """
        n_states = len(np.unique(states))
        transitions = np.zeros((n_states, n_states))

        for i in range(len(states) - 1):
            current_state = states[i]
            next_state = states[i + 1]
            transitions[current_state, next_state] += 1

        # Normalize rows
        row_sums = transitions.sum(axis=1)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        transitions = transitions / row_sums[:, np.newaxis]

        return transitions

    def _calculate_state_persistence(self, states: np.ndarray) -> float:
        """Calculate average state persistence.

        Args:
            states: State sequence

        Returns:
            Average persistence (in steps)
        """
        persistence_counts = []
        current_state = states[0]
        count = 1

        for state in states[1:]:
            if state == current_state:
                count += 1
            else:
                persistence_counts.append(count)
                current_state = state
                count = 1

        persistence_counts.append(count)

        return np.mean(persistence_counts)

    def _calculate_state_entropy(self, states: np.ndarray) -> float:
        """Calculate state entropy (diversity).

        Args:
            states: State sequence

        Returns:
            State entropy
        """
        unique_states, counts = np.unique(states, return_counts=True)
        probabilities = counts / len(states)
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
        return entropy

    def _calculate_regime_stability(self, regimes: np.ndarray) -> float:
        """Calculate regime stability score.

        Args:
            regimes: Regime sequence

        Returns:
            Regime stability score
        """
        # Calculate regime changes
        regime_changes = np.sum(np.diff(regimes) != 0)
        total_periods = len(regimes) - 1

        if total_periods == 0:
            return 1.0

        # Stability is inverse of change frequency
        stability = 1.0 - (regime_changes / total_periods)

        return max(0.0, min(1.0, stability))

    def get_all_metrics(self, result: EvaluationResult) -> Dict[str, float]:
        """Get all available metrics from evaluation result.

        Args:
            result: Evaluation result

        Returns:
            Dictionary with all metrics
        """
        metrics = {
            'loss': result.loss,
            'accuracy': result.accuracy,
            'precision': result.precision,
            'recall': result.recall,
            'f1_score': result.f1_score,
            'execution_time': result.execution_time
        }

        # Add metadata metrics
        if result.metadata:
            metrics.update(result.metadata)

        return metrics

    def compare_models(self,
                      models: List[nn.Module],
                      datasets: List[Dataset],
                      model_names: List[str],
                      problem_type: str,
                      metric_name: str = "accuracy") -> Dict[str, Any]:
        """Compare multiple models on the same datasets.

        Args:
            models: List of models to compare
            datasets: List of datasets for evaluation
            model_names: Names of models
            problem_type: Type of problem
            metric_name: Primary metric for comparison

        Returns:
            Dictionary with comparison results
        """
        comparison_results = {}

        for model, dataset, name in zip(models, datasets, model_names):
            self.logger.info(f"📊 Evaluating {name}")
            score = self.evaluate(model, dataset, problem_type, metric_name)
            comparison_results[name] = score

        # Rank models by primary metric
        sorted_models = sorted(comparison_results.items(), key=lambda x: x[1], reverse=True)

        return {
            'individual_scores': comparison_results,
            'ranking': sorted_models,
            'best_model': sorted_models[0][0] if sorted_models else None,
            'best_score': sorted_models[0][1] if sorted_models else 0.0
        }