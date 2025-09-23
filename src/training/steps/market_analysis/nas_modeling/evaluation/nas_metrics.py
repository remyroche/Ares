"""
NAS Evaluation Metrics

This module provides comprehensive metrics for evaluating
neural architectures in the context of market analysis.
"""

import numpy as np
import torch
import torch.nn as nn
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
import time

from ..utils.nas_utils import NASUtils
from ..utils.logging_utils import NASLogger

logger = logging.getLogger(__name__)

@dataclass
class NASMetricsConfig:
    """Configuration for NAS metrics."""
    primary_metric: str = "accuracy"
    minimize_metric: bool = False
    compute_complexity_metrics: bool = True
    compute_efficiency_metrics: bool = True
    compute_generalization_metrics: bool = True
    compute_stability_metrics: bool = True

@dataclass
class ArchitectureMetrics:
    """Comprehensive metrics for a single architecture."""
    # Performance metrics
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    loss: float = 0.0

    # Complexity metrics
    num_parameters: int = 0
    complexity_score: float = 0.0
    flops: int = 0
    memory_usage: int = 0  # in bytes

    # Efficiency metrics
    training_time: float = 0.0
    inference_time: float = 0.0
    parameters_per_second: float = 0.0

    # Generalization metrics
    train_accuracy: float = 0.0
    val_accuracy: float = 0.0
    test_accuracy: float = 0.0
    overfitting_gap: float = 0.0

    # Stability metrics
    parameter_variance: float = 0.0
    gradient_norm: float = 0.0
    eigenvalue_stability: float = 0.0

    # Problem-specific metrics
    regime_detection_accuracy: float = 0.0
    hmm_state_accuracy: float = 0.0
    transition_accuracy: float = 0.0

    # Metadata
    architecture_name: str = ""
    evaluated_at: str = ""

class NASMetrics:
    """
    NAS Evaluation Metrics

    Provides comprehensive evaluation metrics for neural architectures,
    including performance, complexity, efficiency, and generalization metrics.
    """

    def __init__(self, config: NASMetricsConfig):
        """Initialize NAS metrics.

        Args:
            config: Metrics configuration
        """
        self.config = config
        self.logger = NASLogger.get_logger(self.__class__.__name__)
        self.nas_utils = NASUtils()

        self.logger.info("📊 NAS metrics initialized")

    def evaluate_architecture(self,
                             model: nn.Module,
                             train_loader: torch.utils.data.DataLoader,
                             val_loader: torch.utils.data.DataLoader,
                             test_loader: Optional[torch.utils.data.DataLoader] = None,
                             architecture_name: str = "",
                             problem_type: str = "classification") -> ArchitectureMetrics:
        """
        Comprehensive evaluation of a neural architecture.

        Args:
            model: Model to evaluate
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Optional test data loader
            architecture_name: Name of the architecture
            problem_type: Type of problem

        Returns:
            ArchitectureMetrics with comprehensive evaluation
        """
        start_time = time.time()
        self.logger.info(f"🔬 Evaluating architecture: {architecture_name}")

        # Basic performance metrics
        metrics = ArchitectureMetrics(architecture_name=architecture_name)

        # Evaluate on different datasets
        train_metrics = self._evaluate_dataset(model, train_loader, problem_type, "train")
        val_metrics = self._evaluate_dataset(model, val_loader, problem_type, "validation")
        test_metrics = None
        if test_loader:
            test_metrics = self._evaluate_dataset(model, test_loader, problem_type, "test")

        # Fill basic metrics
        metrics.accuracy = val_metrics['accuracy']
        metrics.precision = val_metrics['precision']
        metrics.recall = val_metrics['recall']
        metrics.f1_score = val_metrics['f1_score']
        metrics.loss = val_metrics['loss']

        # Dataset-specific accuracies
        metrics.train_accuracy = train_metrics['accuracy']
        metrics.val_accuracy = val_metrics['accuracy']
        if test_metrics:
            metrics.test_accuracy = test_metrics['accuracy']

        # Overfitting gap
        metrics.overfitting_gap = abs(train_metrics['accuracy'] - val_metrics['accuracy'])

        # Complexity metrics
        if self.config.compute_complexity_metrics:
            metrics.num_parameters = self._count_parameters(model)
            metrics.complexity_score = self._calculate_complexity_score(model)
            metrics.flops = self._estimate_flops(model, next(iter(train_loader))[0].shape)
            metrics.memory_usage = self._estimate_memory_usage(model, next(iter(train_loader))[0].shape)

        # Efficiency metrics
        if self.config.compute_efficiency_metrics:
            metrics.training_time = self._measure_training_time(model, train_loader, problem_type)
            metrics.inference_time = self._measure_inference_time(model, val_loader)
            metrics.parameters_per_second = metrics.num_parameters / (metrics.training_time + 1e-8)

        # Generalization metrics
        if self.config.compute_generalization_metrics and test_metrics:
            metrics = self._add_generalization_metrics(metrics, train_metrics, val_metrics, test_metrics)

        # Problem-specific metrics
        metrics = self._add_problem_specific_metrics(metrics, model, val_loader, problem_type)

        metrics.evaluated_at = time.strftime("%Y-%m-%d %H:%M:%S")
        evaluation_time = time.time() - start_time

        self.logger.info(f"✅ Architecture evaluation completed in {evaluation_time:.2f}s")
        self.logger.info(f"🎯 Final accuracy: {metrics.accuracy:.4f}")
        self.logger.info(f"📊 Parameters: {metrics.num_parameters:,}")

        return metrics

    def _evaluate_dataset(self,
                         model: nn.Module,
                         data_loader: torch.utils.data.DataLoader,
                         problem_type: str,
                         dataset_name: str) -> Dict[str, float]:
        """Evaluate model on a specific dataset.

        Args:
            model: Model to evaluate
            data_loader: Data loader
            problem_type: Type of problem
            dataset_name: Name of dataset for logging

        Returns:
            Dictionary with metrics
        """
        model.eval()
        all_predictions = []
        all_targets = []
        total_loss = 0.0

        criterion = self._get_loss_function(problem_type)

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                output = model(data)
                loss = criterion(output, target)
                total_loss += loss.item()

                # Get predictions
                if problem_type == "classification":
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
            metrics = self._calculate_classification_metrics(all_targets, all_predictions, avg_loss)
        elif problem_type == "regression":
            metrics = self._calculate_regression_metrics(all_targets, all_predictions, avg_loss)
        elif problem_type == "hmm":
            metrics = self._calculate_hmm_metrics(all_targets, all_predictions, avg_loss)
        else:
            metrics = self._calculate_classification_metrics(all_targets, all_predictions, avg_loss)

        self.logger.debug(f"📊 {dataset_name} metrics: {metrics}")
        return metrics

    def _calculate_classification_metrics(self,
                                        targets: np.ndarray,
                                        predictions: np.ndarray,
                                        loss: float) -> Dict[str, float]:
        """Calculate classification metrics.

        Args:
            targets: Ground truth labels
            predictions: Model predictions
            loss: Average loss

        Returns:
            Dictionary with classification metrics
        """
        try:
            accuracy = accuracy_score(targets, predictions)
            precision = precision_score(targets, predictions, average='macro', zero_division=0)
            recall = recall_score(targets, predictions, average='macro', zero_division=0)
            f1 = f1_score(targets, predictions, average='macro', zero_division=0)

            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'loss': loss
            }
        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating classification metrics: {e}")
            return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'loss': loss}

    def _calculate_regression_metrics(self,
                                    targets: np.ndarray,
                                    predictions: np.ndarray,
                                    loss: float) -> Dict[str, float]:
        """Calculate regression metrics.

        Args:
            targets: Ground truth values
            predictions: Model predictions
            loss: Average loss

        Returns:
            Dictionary with regression metrics
        """
        try:
            mse = mean_squared_error(targets, predictions)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(targets, predictions)
            r2 = r2_score(targets, predictions)

            # Pearson and Spearman correlations
            pearson_corr, _ = pearsonr(targets, predictions)
            spearman_corr, _ = spearmanr(targets, predictions)

            return {
                'accuracy': r2,  # Use R² as accuracy for regression
                'precision': mae,  # Use MAE as precision
                'recall': rmse,  # Use RMSE as recall
                'f1_score': mse,  # Use MSE as F1
                'loss': loss,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'pearson_corr': pearson_corr,
                'spearman_corr': spearman_corr
            }
        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating regression metrics: {e}")
            return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'loss': loss}

    def _calculate_hmm_metrics(self,
                             targets: np.ndarray,
                             predictions: np.ndarray,
                             loss: float) -> Dict[str, float]:
        """Calculate HMM-specific metrics.

        Args:
            targets: Ground truth states
            predictions: Predicted states
            loss: Average loss

        Returns:
            Dictionary with HMM metrics
        """
        try:
            # Basic classification metrics
            accuracy = accuracy_score(targets, predictions)
            precision = precision_score(targets, predictions, average='macro', zero_division=0)
            recall = recall_score(targets, predictions, average='macro', zero_division=0)
            f1 = f1_score(targets, predictions, average='macro', zero_division=0)

            # HMM-specific metrics
            state_persistence = self._calculate_state_persistence(targets)
            transition_accuracy = self._calculate_transition_accuracy(targets, predictions)

            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'loss': loss,
                'state_persistence': state_persistence,
                'transition_accuracy': transition_accuracy
            }
        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating HMM metrics: {e}")
            return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'loss': loss}

    def _count_parameters(self, model: nn.Module) -> int:
        """Count number of trainable parameters.

        Args:
            model: PyTorch model

        Returns:
            Number of parameters
        """
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def _calculate_complexity_score(self, model: nn.Module) -> float:
        """Calculate architecture complexity score.

        Args:
            model: PyTorch model

        Returns:
            Complexity score
        """
        # Simple complexity measure based on parameter count and depth
        n_params = self._count_parameters(model)

        # Estimate depth (rough approximation)
        depth = 0
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.LSTM, nn.GRU)):
                depth += 1

        # Complexity score combines parameters and depth
        complexity = np.log(n_params + 1) + np.log(depth + 1)
        return complexity

    def _estimate_flops(self, model: nn.Module, input_shape: Tuple[int, ...]) -> int:
        """Estimate FLOPs for the model.

        Args:
            model: PyTorch model
            input_shape: Shape of input tensor

        Returns:
            Estimated FLOPs
        """
        # This is a simplified FLOP estimation
        # In practice, you might want to use libraries like ptflops or fvcore

        total_flops = 0

        # Create dummy input
        batch_size = input_shape[0]
        dummy_input = torch.randn(input_shape)

        def count_flops_hook(module, input, output):
            nonlocal total_flops

            if isinstance(module, nn.Linear):
                # Linear layer: batch_size * in_features * out_features
                total_flops += batch_size * input[0].shape[1] * output.shape[1]
            elif isinstance(module, (nn.Conv1d, nn.Conv2d)):
                # Convolution: complex calculation based on kernel size, etc.
                total_flops += output.numel() * module.kernel_size[0] ** len(module.kernel_size)

        # Register hooks
        hooks = []
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                hooks.append(module.register_forward_hook(count_flops_hook))

        # Forward pass
        model.eval()
        with torch.no_grad():
            _ = model(dummy_input)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        return total_flops

    def _estimate_memory_usage(self, model: nn.Module, input_shape: Tuple[int, ...]) -> int:
        """Estimate memory usage.

        Args:
            model: PyTorch model
            input_shape: Shape of input tensor

        Returns:
            Estimated memory usage in bytes
        """
        # Parameters memory
        param_memory = sum(p.numel() * p.element_size() for p in model.parameters())

        # Activation memory (rough estimate)
        batch_size = input_shape[0]
        activation_memory = 0

        def activation_hook(module, input, output):
            nonlocal activation_memory
            activation_memory += output.numel() * output.element_size()

        hooks = []
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.LSTM, nn.GRU)):
                hooks.append(module.register_forward_hook(activation_hook))

        dummy_input = torch.randn(input_shape)
        with torch.no_grad():
            _ = model(dummy_input)

        for hook in hooks:
            hook.remove()

        return param_memory + activation_memory * batch_size

    def _measure_training_time(self,
                              model: nn.Module,
                              train_loader: torch.utils.data.DataLoader,
                              problem_type: str) -> float:
        """Measure training time for one epoch.

        Args:
            model: Model to train
            train_loader: Training data loader
            problem_type: Type of problem

        Returns:
            Training time in seconds
        """
        model.train()
        criterion = self._get_loss_function(problem_type)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        start_time = time.time()

        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            break  # Only measure one batch

        return time.time() - start_time

    def _measure_inference_time(self,
                               model: nn.Module,
                               data_loader: torch.utils.data.DataLoader) -> float:
        """Measure inference time.

        Args:
            model: Model to evaluate
            data_loader: Data loader

        Returns:
            Inference time in seconds
        """
        model.eval()
        start_time = time.time()

        with torch.no_grad():
            for data, target in data_loader:
                _ = model(data)
                break  # Only measure one batch

        return time.time() - start_time

    def _add_generalization_metrics(self,
                                  metrics: ArchitectureMetrics,
                                  train_metrics: Dict[str, float],
                                  val_metrics: Dict[str, float],
                                  test_metrics: Dict[str, float]) -> ArchitectureMetrics:
        """Add generalization metrics.

        Args:
            metrics: Current metrics
            train_metrics: Training metrics
            val_metrics: Validation metrics
            test_metrics: Test metrics

        Returns:
            Updated metrics
        """
        # Train-validation gap
        metrics.overfitting_gap = abs(train_metrics['accuracy'] - val_metrics['accuracy'])

        # Validation-test gap
        if test_metrics:
            val_test_gap = abs(val_metrics['accuracy'] - test_metrics['accuracy'])
            metrics.metadata['val_test_gap'] = val_test_gap

        # Consistency score (lower gap = higher consistency)
        consistency_score = 1.0 / (1.0 + metrics.overfitting_gap)
        metrics.metadata['consistency_score'] = consistency_score

        return metrics

    def _add_problem_specific_metrics(self,
                                    metrics: ArchitectureMetrics,
                                    model: nn.Module,
                                    data_loader: torch.utils.data.DataLoader,
                                    problem_type: str) -> ArchitectureMetrics:
        """Add problem-specific metrics.

        Args:
            metrics: Current metrics
            model: Model
            data_loader: Data loader
            problem_type: Type of problem

        Returns:
            Updated metrics
        """
        if problem_type == "hmm":
            metrics.hmm_state_accuracy = self._calculate_hmm_state_accuracy(model, data_loader)
            metrics.transition_accuracy = self._calculate_transition_accuracy_from_model(model, data_loader)
        elif problem_type == "regime_detection":
            metrics.regime_detection_accuracy = self._calculate_regime_detection_accuracy(model, data_loader)

        return metrics

    def _calculate_hmm_state_accuracy(self, model: nn.Module, data_loader: torch.utils.data.DataLoader) -> float:
        """Calculate HMM state prediction accuracy.

        Args:
            model: HMM model
            data_loader: Data loader

        Returns:
            State accuracy
        """
        # This would require a model that outputs state probabilities
        # For now, return the general accuracy
        return metrics.accuracy

    def _calculate_transition_accuracy_from_model(self, model: nn.Module, data_loader: torch.utils.data.DataLoader) -> float:
        """Calculate transition accuracy from model.

        Args:
            model: HMM model
            data_loader: Data loader

        Returns:
            Transition accuracy
        """
        # This would require sequence data and transition predictions
        # For now, return a placeholder
        return 0.8  # Placeholder

    def _calculate_regime_detection_accuracy(self, model: nn.Module, data_loader: torch.utils.data.DataLoader) -> float:
        """Calculate regime detection accuracy.

        Args:
            model: Regime detection model
            data_loader: Data loader

        Returns:
            Regime detection accuracy
        """
        # Similar to HMM state accuracy
        return metrics.accuracy

    def _calculate_state_persistence(self, states: np.ndarray) -> float:
        """Calculate average state persistence.

        Args:
            states: State sequence

        Returns:
            Average persistence
        """
        if len(states) < 2:
            return 1.0

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

    def _get_loss_function(self, problem_type: str) -> nn.Module:
        """Get appropriate loss function.

        Args:
            problem_type: Type of problem

        Returns:
            Loss function
        """
        if problem_type == "classification":
            return nn.CrossEntropyLoss()
        elif problem_type == "regression":
            return nn.MSELoss()
        elif problem_type == "hmm":
            return nn.NLLLoss()
        else:
            return nn.CrossEntropyLoss()

    def compare_architectures(self,
                            architecture_metrics: List[ArchitectureMetrics],
                            primary_metric: str = "accuracy") -> Dict[str, Any]:
        """Compare multiple architectures.

        Args:
            architecture_metrics: List of architecture metrics
            primary_metric: Primary metric for comparison

        Returns:
            Comparison results
        """
        if not architecture_metrics:
            return {}

        # Sort by primary metric
        sorted_architectures = sorted(
            architecture_metrics,
            key=lambda x: getattr(x, primary_metric),
            reverse=not self.config.minimize_metric
        )

        # Find best architecture
        best_architecture = sorted_architectures[0]

        # Calculate averages
        avg_metrics = {}
        for attr in ['accuracy', 'precision', 'recall', 'f1_score', 'num_parameters', 'complexity_score']:
            values = [getattr(arch, attr) for arch in architecture_metrics]
            avg_metrics[f'avg_{attr}'] = np.mean(values)
            avg_metrics[f'std_{attr}'] = np.std(values)

        return {
            'best_architecture': best_architecture.architecture_name,
            'best_score': getattr(best_architecture, primary_metric),
            'sorted_architectures': [arch.architecture_name for arch in sorted_architectures],
            'average_metrics': avg_metrics,
            'total_architectures': len(architecture_metrics)
        }