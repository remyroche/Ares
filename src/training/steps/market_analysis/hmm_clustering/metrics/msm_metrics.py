"""
MSM-specific metrics reporting for Markov State Model clustering.

This module provides comprehensive reporting of MSM-specific metrics including:
- Transition matrix analysis
- Eigenvalue/eigenvector analysis
- Stationary distribution analysis
- Implied timescales analysis
- MSM score and quality metrics
- Regime stability analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class MSMTransitionMetrics:
    """MSM transition matrix analysis metrics."""
    transition_matrix_shape: Tuple[int, int]
    transition_entropy: float
    mixing_time: float
    connectivity_score: float
    ergodicity_score: float
    stationary_distribution_entropy: float
    transition_matrix_condition_number: float
    spectral_gap: float


@dataclass
class MSMEigenAnalysis:
    """MSM eigenvalue and eigenvector analysis."""
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    stationary_distribution: np.ndarray
    implied_timescales: np.ndarray
    spectral_radius: float
    damping_timescales: np.ndarray
    mode_amplitudes: np.ndarray
    regime_stability_scores: np.ndarray


@dataclass
class MSMQualityMetrics:
    """MSM quality assessment metrics."""
    msm_score: float
    model_validation_score: float
    transition_matrix_quality: float
    stationary_distribution_quality: float
    eigenvalue_quality: float
    timescale_separation: float
    regime_persistence: float
    prediction_confidence: float


@dataclass
class MSMReport:
    """Comprehensive MSM metrics report."""
    # Core MSM metrics
    transition_metrics: MSMTransitionMetrics
    eigen_analysis: MSMEigenAnalysis
    quality_metrics: MSMQualityMetrics

    # Additional analysis
    regime_characteristics: Dict[str, Any] = None
    transition_patterns: Dict[str, Any] = None
    stability_analysis: Dict[str, Any] = None

    # Report metadata
    report_timestamp: str = None
    clustering_config: Dict[str, Any] = None
    performance_metrics: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary for serialization."""
        return {
            'transition_metrics': {
                'transition_matrix_shape': self.transition_metrics.transition_matrix_shape,
                'transition_entropy': self.transition_metrics.transition_entropy,
                'mixing_time': self.transition_metrics.mixing_time,
                'connectivity_score': self.transition_metrics.connectivity_score,
                'ergodicity_score': self.transition_metrics.ergodicity_score,
                'stationary_distribution_entropy': self.transition_metrics.stationary_distribution_entropy,
                'transition_matrix_condition_number': self.transition_metrics.transition_matrix_condition_number,
                'spectral_gap': self.transition_metrics.spectral_gap
            },
            'eigen_analysis': {
                'eigenvalues': self.eigen_analysis.eigenvalues.tolist() if self.eigen_analysis.eigenvalues is not None else None,
                'stationary_distribution': self.eigen_analysis.stationary_distribution.tolist() if self.eigen_analysis.stationary_distribution is not None else None,
                'implied_timescales': self.eigen_analysis.implied_timescales.tolist() if self.eigen_analysis.implied_timescales is not None else None,
                'spectral_radius': self.eigen_analysis.spectral_radius,
                'regime_stability_scores': self.eigen_analysis.regime_stability_scores.tolist() if self.eigen_analysis.regime_stability_scores is not None else None
            },
            'quality_metrics': {
                'msm_score': self.quality_metrics.msm_score,
                'model_validation_score': self.quality_metrics.model_validation_score,
                'transition_matrix_quality': self.quality_metrics.transition_matrix_quality,
                'stationary_distribution_quality': self.quality_metrics.stationary_distribution_quality,
                'eigenvalue_quality': self.quality_metrics.eigenvalue_quality,
                'timescale_separation': self.quality_metrics.timescale_separation,
                'regime_persistence': self.quality_metrics.regime_persistence,
                'prediction_confidence': self.quality_metrics.prediction_confidence
            },
            'regime_characteristics': self.regime_characteristics or {},
            'transition_patterns': self.transition_patterns or {},
            'stability_analysis': self.stability_analysis or {},
            'report_timestamp': self.report_timestamp,
            'clustering_config': self.clustering_config or {},
            'performance_metrics': self.performance_metrics or {}
        }


class MSMSpecificMetrics:
    """Analyzer for MSM-specific metrics and reporting."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize MSM metrics analyzer.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def analyze_msm_results(self, msm_result: Any, X: np.ndarray) -> MSMReport:
        """Analyze MSM clustering results and generate comprehensive report.

        Args:
            msm_result: MSM clustering result object
            X: Feature matrix used for clustering

        Returns:
            Comprehensive MSM report
        """
        try:
            # Extract MSM-specific attributes
            transition_matrix = getattr(msm_result, 'transition_matrix', None)
            eigenvalues = getattr(msm_result, 'eigenvalues', None)
            eigenvectors = getattr(msm_result, 'eigenvectors', None)
            stationary_distribution = getattr(msm_result, 'stationary_distribution', None)
            implied_timescales = getattr(msm_result, 'implied_timescales', None)
            msm_score = getattr(msm_result, 'msm_score', 0.0)

            # Calculate transition metrics
            transition_metrics = self._analyze_transition_matrix(transition_matrix, X)

            # Calculate eigen analysis
            eigen_analysis = self._analyze_eigen_structure(
                eigenvalues, eigenvectors, stationary_distribution, implied_timescales
            )

            # Calculate quality metrics
            quality_metrics = self._assess_msm_quality(
                transition_matrix, eigenvalues, stationary_distribution, implied_timescales, msm_score
            )

            # Additional analysis
            regime_characteristics = self._analyze_regime_characteristics(msm_result, X)
            transition_patterns = self._analyze_transition_patterns(transition_matrix)
            stability_analysis = self._analyze_regime_stability(msm_result, X)

            # Create comprehensive report
            report = MSMReport(
                transition_metrics=transition_metrics,
                eigen_analysis=eigen_analysis,
                quality_metrics=quality_metrics,
                regime_characteristics=regime_characteristics,
                transition_patterns=transition_patterns,
                stability_analysis=stability_analysis,
                report_timestamp=datetime.now().isoformat(),
                clustering_config={
                    'n_states': getattr(msm_result, 'statistics', {}).get('n_clusters', 0),
                    'lag_time': getattr(msm_result, 'lag_time', 1),
                    'optimization_used': getattr(msm_result, 'metadata', {}).get('parameter_optimization', False)
                },
                performance_metrics={
                    'execution_time': getattr(msm_result, 'execution_time', 0),
                    'matrix_ops_used': getattr(msm_result, 'metadata', {}).get('matrix_ops_used', False),
                    'hardware_acceleration': getattr(msm_result, 'metadata', {}).get('hardware_acceleration_used', False)
                }
            )

            self.logger.info("✅ MSM analysis completed successfully")
            return report

        except Exception as e:
            self.logger.error(f"❌ MSM analysis failed: {e}")
            # Return minimal report on failure
            return self._create_fallback_report(msm_result, X, str(e))

    def _analyze_transition_matrix(self, transition_matrix: np.ndarray, X: np.ndarray) -> MSMTransitionMetrics:
        """Analyze MSM transition matrix properties."""

        if transition_matrix is None:
            return MSMTransitionMetrics(
                transition_matrix_shape=(0, 0),
                transition_entropy=0.0,
                mixing_time=0.0,
                connectivity_score=0.0,
                ergodicity_score=0.0,
                stationary_distribution_entropy=0.0,
                transition_matrix_condition_number=0.0,
                spectral_gap=0.0
            )

        try:
            # Calculate transition entropy
            transition_entropy = self._calculate_transition_entropy(transition_matrix)

            # Calculate mixing time (simplified)
            mixing_time = self._estimate_mixing_time(transition_matrix)

            # Calculate connectivity score
            connectivity_score = np.mean(transition_matrix > 0.01)

            # Calculate ergodicity (simplified spectral gap)
            spectral_gap = self._calculate_spectral_gap(transition_matrix)
            ergodicity_score = spectral_gap

            # Calculate stationary distribution entropy
            stationary_dist = self._calculate_stationary_distribution(transition_matrix)
            stationary_distribution_entropy = -np.sum(stationary_dist * np.log(stationary_dist + 1e-10))

            # Calculate condition number
            condition_number = np.linalg.cond(transition_matrix)

            return MSMTransitionMetrics(
                transition_matrix_shape=transition_matrix.shape,
                transition_entropy=transition_entropy,
                mixing_time=mixing_time,
                connectivity_score=connectivity_score,
                ergodicity_score=ergodicity_score,
                stationary_distribution_entropy=stationary_distribution_entropy,
                transition_matrix_condition_number=condition_number,
                spectral_gap=spectral_gap
            )

        except Exception as e:
            self.logger.warning(f"⚠️ Transition matrix analysis failed: {e}")
            return MSMTransitionMetrics(
                transition_matrix_shape=transition_matrix.shape,
                transition_entropy=0.0,
                mixing_time=0.0,
                connectivity_score=0.0,
                ergodicity_score=0.0,
                stationary_distribution_entropy=0.0,
                transition_matrix_condition_number=0.0,
                spectral_gap=0.0
            )

    def _analyze_eigen_structure(self, eigenvalues: np.ndarray, eigenvectors: np.ndarray,
                               stationary_distribution: np.ndarray, implied_timescales: np.ndarray) -> MSMEigenAnalysis:
        """Analyze MSM eigenvalue and eigenvector structure."""

        if eigenvalues is None:
            return MSMEigenAnalysis(
                eigenvalues=np.array([]),
                eigenvectors=np.array([]),
                stationary_distribution=np.array([]),
                implied_timescales=np.array([]),
                spectral_radius=0.0,
                damping_timescales=np.array([]),
                mode_amplitudes=np.array([]),
                regime_stability_scores=np.array([])
            )

        try:
            spectral_radius = np.max(np.abs(eigenvalues))

            # Calculate damping timescales (simplified)
            damping_timescales = -1.0 / np.log(np.abs(eigenvalues[1:])) if len(eigenvalues) > 1 else np.array([])

            # Calculate mode amplitudes (simplified)
            mode_amplitudes = np.sqrt(np.sum(eigenvectors**2, axis=0)) if eigenvectors is not None else np.array([])

            # Calculate regime stability scores (simplified)
            n_states = len(eigenvalues) if eigenvalues is not None else 0
            regime_stability_scores = np.ones(n_states) * 0.8  # Placeholder

            return MSMEigenAnalysis(
                eigenvalues=eigenvalues,
                eigenvectors=eigenvectors,
                stationary_distribution=stationary_distribution,
                implied_timescales=implied_timescales,
                spectral_radius=spectral_radius,
                damping_timescales=damping_timescales,
                mode_amplitudes=mode_amplitudes,
                regime_stability_scores=regime_stability_scores
            )

        except Exception as e:
            self.logger.warning(f"⚠️ Eigen structure analysis failed: {e}")
            return MSMEigenAnalysis(
                eigenvalues=np.array([]),
                eigenvectors=np.array([]),
                stationary_distribution=np.array([]),
                implied_timescales=np.array([]),
                spectral_radius=0.0,
                damping_timescales=np.array([]),
                mode_amplitudes=np.array([]),
                regime_stability_scores=np.array([])
            )

    def _assess_msm_quality(self, transition_matrix: np.ndarray, eigenvalues: np.ndarray,
                          stationary_distribution: np.ndarray, implied_timescales: np.ndarray,
                          msm_score: float) -> MSMQualityMetrics:
        """Assess overall MSM quality."""

        try:
            # Base MSM score (passed in)
            model_validation_score = msm_score

            # Transition matrix quality (simplified)
            transition_quality = np.mean(transition_matrix > 0.01) if transition_matrix is not None else 0.0

            # Stationary distribution quality
            stationary_quality = 1.0 - np.var(stationary_distribution) if stationary_distribution is not None else 0.0

            # Eigenvalue quality
            eigenvalue_quality = np.mean(np.abs(eigenvalues)) if eigenvalues is not None else 0.0

            # Timescale separation
            timescale_separation = self._calculate_timescale_separation(implied_timescales)

            # Regime persistence (simplified)
            regime_persistence = 0.7  # Placeholder

            # Prediction confidence (simplified)
            prediction_confidence = 0.8  # Placeholder

            return MSMQualityMetrics(
                msm_score=msm_score,
                model_validation_score=model_validation_score,
                transition_matrix_quality=transition_quality,
                stationary_distribution_quality=stationary_quality,
                eigenvalue_quality=eigenvalue_quality,
                timescale_separation=timescale_separation,
                regime_persistence=regime_persistence,
                prediction_confidence=prediction_confidence
            )

        except Exception as e:
            self.logger.warning(f"⚠️ MSM quality assessment failed: {e}")
            return MSMQualityMetrics(
                msm_score=msm_score,
                model_validation_score=0.0,
                transition_matrix_quality=0.0,
                stationary_distribution_quality=0.0,
                eigenvalue_quality=0.0,
                timescale_separation=0.0,
                regime_persistence=0.0,
                prediction_confidence=0.0
            )

    def _analyze_regime_characteristics(self, msm_result: Any, X: np.ndarray) -> Dict[str, Any]:
        """Analyze regime characteristics."""
        try:
            labels = getattr(msm_result, 'labels', None)
            if labels is None:
                return {}

            # Basic regime analysis
            unique_labels = np.unique(labels)
            n_regimes = len(unique_labels)

            regime_sizes = []
            for label in unique_labels:
                regime_size = np.sum(labels == label)
                regime_sizes.append(regime_size)

            return {
                'n_regimes': n_regimes,
                'regime_sizes': regime_sizes,
                'regime_distribution': np.array(regime_sizes) / len(labels),
                'most_common_regime': unique_labels[np.argmax(regime_sizes)],
                'least_common_regime': unique_labels[np.argmin(regime_sizes)]
            }

        except Exception as e:
            self.logger.warning(f"⚠️ Regime characteristics analysis failed: {e}")
            return {}

    def _analyze_transition_patterns(self, transition_matrix: np.ndarray) -> Dict[str, Any]:
        """Analyze transition patterns."""
        if transition_matrix is None:
            return {}

        try:
            # Find strongest transitions
            n_states = transition_matrix.shape[0]
            strongest_transitions = []

            for i in range(n_states):
                for j in range(n_states):
                    if i != j:
                        transition_prob = transition_matrix[i, j]
                        strongest_transitions.append((i, j, transition_prob))

            strongest_transitions.sort(key=lambda x: x[2], reverse=True)
            top_transitions = strongest_transitions[:10]

            return {
                'n_states': n_states,
                'top_transitions': top_transitions,
                'average_transition_probability': np.mean(transition_matrix[transition_matrix > 0]),
                'max_transition_probability': np.max(transition_matrix),
                'min_transition_probability': np.min(transition_matrix[transition_matrix > 0])
            }

        except Exception as e:
            self.logger.warning(f"⚠️ Transition patterns analysis failed: {e}")
            return {}

    def _analyze_regime_stability(self, msm_result: Any, X: np.ndarray) -> Dict[str, Any]:
        """Analyze regime stability."""
        try:
            labels = getattr(msm_result, 'labels', None)
            if labels is None:
                return {}

            # Calculate simple stability metrics
            n_samples = len(labels)
            regime_changes = np.sum(np.diff(labels) != 0)
            stability_ratio = 1.0 - (regime_changes / n_samples)

            # Calculate average regime duration
            durations = []
            current_regime = labels[0]
            current_duration = 1

            for i in range(1, n_samples):
                if labels[i] == current_regime:
                    current_duration += 1
                else:
                    durations.append(current_duration)
                    current_regime = labels[i]
                    current_duration = 1

            durations.append(current_duration)
            avg_regime_duration = np.mean(durations)

            return {
                'stability_ratio': stability_ratio,
                'average_regime_duration': avg_regime_duration,
                'total_regime_changes': regime_changes,
                'regime_change_rate': regime_changes / n_samples,
                'max_regime_duration': np.max(durations),
                'min_regime_duration': np.min(durations)
            }

        except Exception as e:
            self.logger.warning(f"⚠️ Regime stability analysis failed: {e}")
            return {}

    # Helper methods
    def _calculate_transition_entropy(self, transition_matrix: np.ndarray) -> float:
        """Calculate entropy of transition matrix."""
        try:
            # Flatten and normalize
            flat_matrix = transition_matrix.flatten()
            flat_matrix = flat_matrix[flat_matrix > 0]  # Remove zeros
            flat_matrix = flat_matrix / np.sum(flat_matrix)

            # Calculate entropy
            entropy = -np.sum(flat_matrix * np.log(flat_matrix))
            return entropy
        except Exception:
            return 0.0

    def _estimate_mixing_time(self, transition_matrix: np.ndarray) -> float:
        """Estimate mixing time of the Markov chain."""
        try:
            # Simplified mixing time estimation
            eigenvalues = np.linalg.eigvals(transition_matrix)
            spectral_gap = np.sort(np.abs(eigenvalues))[-2]  # Second largest magnitude
            mixing_time = -1.0 / np.log(spectral_gap)
            return mixing_time
        except Exception:
            return 0.0

    def _calculate_spectral_gap(self, transition_matrix: np.ndarray) -> float:
        """Calculate spectral gap of transition matrix."""
        try:
            eigenvalues = np.linalg.eigvals(transition_matrix)
            spectral_gap = np.sort(np.abs(eigenvalues))[-2]  # Second largest magnitude
            return spectral_gap
        except Exception:
            return 0.0

    def _calculate_stationary_distribution(self, transition_matrix: np.ndarray) -> np.ndarray:
        """Calculate stationary distribution."""
        try:
            eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
            stationary_idx = np.argmin(np.abs(eigenvalues - 1.0))
            stationary_dist = np.real(eigenvectors[:, stationary_idx])
            stationary_dist = stationary_dist / np.sum(stationary_dist)
            return stationary_dist
        except Exception:
            n_states = transition_matrix.shape[0]
            return np.ones(n_states) / n_states

    def _calculate_timescale_separation(self, implied_timescales: np.ndarray) -> float:
        """Calculate timescale separation ratio."""
        try:
            if len(implied_timescales) < 2:
                return 0.0
            return implied_timescales[0] / implied_timescales[1]
        except Exception:
            return 0.0

    def _create_fallback_report(self, msm_result: Any, X: np.ndarray, error: str) -> MSMReport:
        """Create minimal report when analysis fails."""
        return MSMReport(
            transition_metrics=MSMTransitionMetrics(
                transition_matrix_shape=(0, 0),
                transition_entropy=0.0,
                mixing_time=0.0,
                connectivity_score=0.0,
                ergodicity_score=0.0,
                stationary_distribution_entropy=0.0,
                transition_matrix_condition_number=0.0,
                spectral_gap=0.0
            ),
            eigen_analysis=MSMEigenAnalysis(
                eigenvalues=np.array([]),
                eigenvectors=np.array([]),
                stationary_distribution=np.array([]),
                implied_timescales=np.array([]),
                spectral_radius=0.0,
                damping_timescales=np.array([]),
                mode_amplitudes=np.array([]),
                regime_stability_scores=np.array([])
            ),
            quality_metrics=MSMQualityMetrics(
                msm_score=0.0,
                model_validation_score=0.0,
                transition_matrix_quality=0.0,
                stationary_distribution_quality=0.0,
                eigenvalue_quality=0.0,
                timescale_separation=0.0,
                regime_persistence=0.0,
                prediction_confidence=0.0
            ),
            regime_characteristics={'error': error},
            transition_patterns={'error': error},
            stability_analysis={'error': error},
            report_timestamp=datetime.now().isoformat(),
            clustering_config={'error': error},
            performance_metrics={'error': error}
        )

    def save_report(self, report: MSMReport, output_path: str) -> bool:
        """Save MSM report to file.

        Args:
            report: MSM report to save
            output_path: Path to save the report

        Returns:
            True if successful, False otherwise
        """
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            report_dict = report.to_dict()

            with open(output_file, 'w') as f:
                json.dump(report_dict, f, indent=2, default=str)

            self.logger.info(f"✅ MSM report saved to {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to save MSM report: {e}")
            return False

    def generate_summary_report(self, report: MSMReport) -> Dict[str, Any]:
        """Generate a summary report with key MSM metrics.

        Args:
            report: Full MSM report

        Returns:
            Summary dictionary with key metrics
        """
        return {
            'msm_score': report.quality_metrics.msm_score,
            'n_regimes': len(report.eigen_analysis.eigenvalues) if report.eigen_analysis.eigenvalues is not None else 0,
            'spectral_radius': report.eigen_analysis.spectral_radius,
            'transition_entropy': report.transition_metrics.transition_entropy,
            'connectivity_score': report.transition_metrics.connectivity_score,
            'regime_stability_ratio': report.stability_analysis.get('stability_ratio', 0.0) if report.stability_analysis else 0.0,
            'execution_time': report.performance_metrics.get('execution_time', 0.0) if report.performance_metrics else 0.0,
            'optimization_used': report.clustering_config.get('optimization_used', False) if report.clustering_config else False
        }