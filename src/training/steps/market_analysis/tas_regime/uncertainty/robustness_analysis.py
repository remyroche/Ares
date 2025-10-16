"""
Robustness Analysis for TAS Tree Architecture

This module provides robustness analysis methods for tree architecture predictions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class RobustnessConfig:
    """Configuration for robustness analysis."""
    noise_level: float = 0.1
    n_perturbations: int = 100
    perturbation_method: str = 'gaussian'  # 'gaussian', 'uniform', 'adversarial'

class TreeRobustnessAnalyzer:
    """Robustness analyzer for tree architectures."""

    def __init__(self, config: RobustnessConfig):
        self.config = config
        self.robustness_metrics = {}

    def analyze_robustness(self, X: np.ndarray, y: np.ndarray, model) -> Dict[str, float]:
        """Analyze model robustness."""
        logger.info("Analyzing model robustness")

        # Get baseline predictions
        baseline_predictions = model.predict(X)

        # Generate perturbations
        perturbations = self._generate_perturbations(X)

        # Test robustness
        robustness_scores = []
        for perturbation in perturbations:
            perturbed_X = X + perturbation
            perturbed_predictions = model.predict(perturbed_X)

            # Calculate robustness score
            robustness = self._calculate_robustness_score(
                baseline_predictions, perturbed_predictions
            )
            robustness_scores.append(robustness)

        # Store robustness metrics
        self.robustness_metrics = {
            'mean_robustness': np.mean(robustness_scores),
            'std_robustness': np.std(robustness_scores),
            'min_robustness': np.min(robustness_scores),
            'max_robustness': np.max(robustness_scores),
            'robustness_scores': robustness_scores
        }

        return self.robustness_metrics

    def _generate_perturbations(self, X: np.ndarray) -> List[np.ndarray]:
        """Generate perturbations for robustness testing."""
        perturbations = []

        for _ in range(self.config.n_perturbations):
            if self.config.perturbation_method == 'gaussian':
                perturbation = np.random.normal(0, self.config.noise_level, X.shape)
            elif self.config.perturbation_method == 'uniform':
                perturbation = np.random.uniform(
                    -self.config.noise_level,
                    self.config.noise_level,
                    X.shape
                )
            elif self.config.perturbation_method == 'adversarial':
                # Simple adversarial perturbation
                perturbation = self.config.noise_level * np.sign(X)
            else:
                perturbation = np.zeros_like(X)

            perturbations.append(perturbation)

        return perturbations

    def _calculate_robustness_score(self, baseline: np.ndarray, perturbed: np.ndarray) -> float:
        """Calculate robustness score."""
        # Robustness is inversely related to prediction change
        prediction_change = np.mean(np.abs(perturbed - baseline))
        robustness = 1.0 / (1.0 + prediction_change)
        return robustness

    def get_robustness_report(self) -> Dict[str, Any]:
        """Get comprehensive robustness report."""
        if not self.robustness_metrics:
            return {'error': 'No robustness analysis performed'}

        return {
            'robustness_summary': {
                'mean_robustness': self.robustness_metrics['mean_robustness'],
                'std_robustness': self.robustness_metrics['std_robustness'],
                'min_robustness': self.robustness_metrics['min_robustness'],
                'max_robustness': self.robustness_metrics['max_robustness']
            },
            'robustness_distribution': {
                'percentiles': {
                    '25th': np.percentile(self.robustness_metrics['robustness_scores'], 25),
                    '50th': np.percentile(self.robustness_metrics['robustness_scores'], 50),
                    '75th': np.percentile(self.robustness_metrics['robustness_scores'], 75),
                    '90th': np.percentile(self.robustness_metrics['robustness_scores'], 90),
                    '95th': np.percentile(self.robustness_metrics['robustness_scores'], 95)
                }
            },
            'robustness_assessment': self._assess_robustness_level()
        }

    def _assess_robustness_level(self) -> str:
        """Assess overall robustness level."""
        if not self.robustness_metrics:
            return 'Unknown'

        mean_robustness = self.robustness_metrics['mean_robustness']

        if mean_robustness >= 0.8:
            return 'High'
        elif mean_robustness >= 0.6:
            return 'Medium'
        elif mean_robustness >= 0.4:
            return 'Low'
        else:
            return 'Very Low'

class TreeAdversarialTesting:
    """Adversarial testing for tree architectures."""

    def __init__(self, config: RobustnessConfig):
        self.config = config
        self.adversarial_results = {}

    def test_adversarial_robustness(self, X: np.ndarray, y: np.ndarray, model) -> Dict[str, Any]:
        """Test adversarial robustness."""
        logger.info("Testing adversarial robustness")

        # Generate adversarial examples
        adversarial_examples = self._generate_adversarial_examples(X, y, model)

        # Test model on adversarial examples
        adversarial_predictions = model.predict(adversarial_examples)
        baseline_predictions = model.predict(X)

        # Calculate adversarial robustness
        adversarial_robustness = self._calculate_adversarial_robustness(
            baseline_predictions, adversarial_predictions
        )

        # Store results
        self.adversarial_results = {
            'adversarial_examples': adversarial_examples,
            'adversarial_predictions': adversarial_predictions,
            'baseline_predictions': baseline_predictions,
            'adversarial_robustness': adversarial_robustness
        }

        return self.adversarial_results

    def _generate_adversarial_examples(self, X: np.ndarray, y: np.ndarray, model) -> np.ndarray:
        """Generate adversarial examples."""
        adversarial_examples = X.copy()

        # Simple adversarial perturbation
        for i in range(len(X)):
            # Add noise in the direction that would change the prediction
            perturbation = self.config.noise_level * np.random.randn(*X[i].shape)
            adversarial_examples[i] = X[i] + perturbation

        return adversarial_examples

    def _calculate_adversarial_robustness(self, baseline: np.ndarray, adversarial: np.ndarray) -> float:
        """Calculate adversarial robustness."""
        # Robustness is inversely related to prediction change
        prediction_change = np.mean(np.abs(adversarial - baseline))
        robustness = 1.0 / (1.0 + prediction_change)
        return robustness

    def get_adversarial_report(self) -> Dict[str, Any]:
        """Get adversarial testing report."""
        if not self.adversarial_results:
            return {'error': 'No adversarial testing performed'}

        return {
            'adversarial_robustness': self.adversarial_results['adversarial_robustness'],
            'prediction_change': np.mean(np.abs(
                self.adversarial_results['adversarial_predictions'] -
                self.adversarial_results['baseline_predictions']
            )),
            'adversarial_success_rate': self._calculate_adversarial_success_rate()
        }

    def _calculate_adversarial_success_rate(self) -> float:
        """Calculate adversarial success rate."""
        if not self.adversarial_results:
            return 0.0

        baseline = self.adversarial_results['baseline_predictions']
        adversarial = self.adversarial_results['adversarial_predictions']

        # Count cases where prediction changed significantly
        significant_changes = np.abs(adversarial - baseline) > 0.1
        success_rate = np.mean(significant_changes)

        return success_rate

class TreePerturbationAnalysis:
    """Perturbation analysis for tree architectures."""

    def __init__(self, config: RobustnessConfig):
        self.config = config
        self.perturbation_results = {}

    def analyze_perturbations(self, X: np.ndarray, y: np.ndarray, model) -> Dict[str, Any]:
        """Analyze model sensitivity to perturbations."""
        logger.info("Analyzing perturbation sensitivity")

        # Get baseline predictions
        baseline_predictions = model.predict(X)

        # Generate different types of perturbations
        perturbations = self._generate_perturbation_types(X)

        # Test sensitivity to each perturbation type
        sensitivity_results = {}
        for perturbation_type, perturbation in perturbations.items():
            perturbed_X = X + perturbation
            perturbed_predictions = model.predict(perturbed_X)

            # Calculate sensitivity
            sensitivity = self._calculate_sensitivity(
                baseline_predictions, perturbed_predictions
            )
            sensitivity_results[perturbation_type] = sensitivity

        # Store results
        self.perturbation_results = {
            'baseline_predictions': baseline_predictions,
            'sensitivity_results': sensitivity_results,
            'perturbations': perturbations
        }

        return self.perturbation_results

    def _generate_perturbation_types(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Generate different types of perturbations."""
        perturbations = {}

        # Gaussian noise
        perturbations['gaussian'] = np.random.normal(0, self.config.noise_level, X.shape)

        # Uniform noise
        perturbations['uniform'] = np.random.uniform(
            -self.config.noise_level,
            self.config.noise_level,
            X.shape
        )

        # Systematic bias
        perturbations['bias'] = self.config.noise_level * np.ones_like(X)

        # Feature-specific noise
        perturbations['feature_specific'] = np.zeros_like(X)
        for i in range(X.shape[1]):
            perturbations['feature_specific'][:, i] = np.random.normal(
                0, self.config.noise_level, X.shape[0]
            )

        return perturbations

    def _calculate_sensitivity(self, baseline: np.ndarray, perturbed: np.ndarray) -> float:
        """Calculate sensitivity to perturbations."""
        # Sensitivity is the change in predictions
        prediction_change = np.mean(np.abs(perturbed - baseline))
        return prediction_change

    def get_perturbation_report(self) -> Dict[str, Any]:
        """Get comprehensive perturbation analysis report."""
        if not self.perturbation_results:
            return {'error': 'No perturbation analysis performed'}

        return {
            'sensitivity_summary': {
                'mean_sensitivity': np.mean(list(self.perturbation_results['sensitivity_results'].values())),
                'max_sensitivity': np.max(list(self.perturbation_results['sensitivity_results'].values())),
                'min_sensitivity': np.min(list(self.perturbation_results['sensitivity_results'].values()))
            },
            'sensitivity_by_type': self.perturbation_results['sensitivity_results'],
            'perturbation_assessment': self._assess_perturbation_sensitivity()
        }

    def _assess_perturbation_sensitivity(self) -> str:
        """Assess overall perturbation sensitivity."""
        if not self.perturbation_results:
            return 'Unknown'

        sensitivities = list(self.perturbation_results['sensitivity_results'].values())
        mean_sensitivity = np.mean(sensitivities)

        if mean_sensitivity <= 0.1:
            return 'Low'
        elif mean_sensitivity <= 0.3:
            return 'Medium'
        elif mean_sensitivity <= 0.5:
            return 'High'
        else:
            return 'Very High'
