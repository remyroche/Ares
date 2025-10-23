"""
Regime Probability Analyzer

This utility provides comprehensive analysis and reporting capabilities for probabilistic regime outputs
from both regime models training and regime ensemble training components.

Features:
- Comprehensive regime probability analysis
- Regime transition analysis
- Regime persistence analysis
- Uncertainty and confidence metrics
- Visualization support
- Export capabilities
"""

import numpy as np
import pandas as pd
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.utils.logger import system_logger
from src.utils.tprint import tprint

class RegimeProbabilityAnalyzer:
    """
    Comprehensive analyzer for probabilistic regime outputs.

    This class provides detailed analysis and reporting capabilities for regime probability
    predictions from both individual models and ensemble methods.
    """

    def __init__(self):
        """Initialize the Regime Probability Analyzer."""
        self.logger = system_logger.getChild('RegimeProbabilityAnalyzer')
        tprint("🔬 [REGIME_ANALYZER] Initializing Regime Probability Analyzer", color="cyan", bold=True)

        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

        tprint("✅ [REGIME_ANALYZER] Regime Probability Analyzer initialized successfully", color="green")

    def analyze_regime_predictions(
        self,
        prediction_result: Dict[str, Any],
        model_name: str = "Unknown Model"
    ) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of regime prediction results.

        Args:
            prediction_result: Dictionary containing regime prediction results
            model_name: Name of the model that generated the predictions

        Returns:
            Dictionary containing comprehensive analysis results
        """
        try:
            tprint(f"🔬 [REGIME_ANALYZER] Analyzing regime predictions from {model_name}", color="cyan", bold=True)

            # Extract basic information
            regime_labels = prediction_result.get('regime_labels', np.array([]))
            regime_probabilities = prediction_result.get('regime_probabilities', np.array([]))
            n_regimes = prediction_result.get('n_regimes', 0)

            if len(regime_labels) == 0 or len(regime_probabilities) == 0:
                tprint("❌ [REGIME_ANALYZER] No valid prediction data found", color="red")
                return {'error': 'No valid prediction data found'}

            # Perform comprehensive analysis
            analysis_results = {
                'model_name': model_name,
                'analysis_timestamp': datetime.now().isoformat(),
                'basic_statistics': self._analyze_basic_statistics(regime_labels, regime_probabilities, n_regimes),
                'probability_analysis': self._analyze_probability_distributions(regime_probabilities, n_regimes),
                'regime_characteristics': self._analyze_regime_characteristics(regime_labels, regime_probabilities, n_regimes),
                'uncertainty_analysis': self._analyze_uncertainty_metrics(regime_probabilities),
                'transition_analysis': prediction_result.get('regime_transitions', {}),
                'persistence_analysis': prediction_result.get('regime_persistence', {}),
                'ensemble_analysis': self._analyze_ensemble_predictions(prediction_result.get('ensemble_probabilities', {})),
                'quality_metrics': self._calculate_quality_metrics(regime_labels, regime_probabilities, n_regimes)
            }

            tprint("✅ [REGIME_ANALYZER] Comprehensive analysis completed", color="green")
            return analysis_results

        except Exception as e:
            tprint(f"❌ [REGIME_ANALYZER] Analysis failed: {e}", color="red")
            self.logger.error(f"Regime prediction analysis failed: {e}", exc_info=True)
            return {'error': str(e)}

    def _analyze_basic_statistics(
        self,
        regime_labels: np.ndarray,
        regime_probabilities: np.ndarray,
        n_regimes: int
    ) -> Dict[str, Any]:
        """Analyze basic statistics of regime predictions."""
        try:
            tprint("📊 [REGIME_ANALYZER] Analyzing basic statistics", color="blue")

            # Regime distribution
            regime_counts = np.bincount(regime_labels, minlength=n_regimes)
            regime_percentages = regime_counts / len(regime_labels) * 100

            # Probability statistics
            max_probs = np.max(regime_probabilities, axis=1)
            avg_probs = np.mean(regime_probabilities, axis=0)

            # Confidence distribution
            high_confidence = np.sum(max_probs >= 0.8)
            medium_confidence = np.sum((max_probs >= 0.5) & (max_probs < 0.8))
            low_confidence = np.sum(max_probs < 0.5)

            return {
                'total_samples': len(regime_labels),
                'n_regimes': n_regimes,
                'regime_distribution': {
                    'counts': regime_counts.tolist(),
                    'percentages': regime_percentages.tolist(),
                    'most_common_regime': int(np.argmax(regime_counts)),
                    'least_common_regime': int(np.argmin(regime_counts[regime_counts > 0]) if np.any(regime_counts > 0) else 0),
                    'regime_balance': float(np.std(regime_percentages))
                },
                'probability_statistics': {
                    'mean_max_probability': float(np.mean(max_probs)),
                    'std_max_probability': float(np.std(max_probs)),
                    'min_max_probability': float(np.min(max_probs)),
                    'max_max_probability': float(np.max(max_probs)),
                    'avg_regime_probabilities': avg_probs.tolist()
                },
                'confidence_distribution': {
                    'high_confidence': int(high_confidence),
                    'medium_confidence': int(medium_confidence),
                    'low_confidence': int(low_confidence),
                    'high_confidence_ratio': float(high_confidence / len(regime_labels)),
                    'low_confidence_ratio': float(low_confidence / len(regime_labels))
                }
            }

        except Exception as e:
            tprint(f"⚠️ [REGIME_ANALYZER] Basic statistics analysis failed: {e}", color="yellow")
            return {'error': str(e)}

    def _analyze_probability_distributions(
        self,
        regime_probabilities: np.ndarray,
        n_regimes: int
    ) -> Dict[str, Any]:
        """Analyze probability distributions for each regime."""
        try:
            tprint("📈 [REGIME_ANALYZER] Analyzing probability distributions", color="blue")

            regime_distributions = {}

            for regime in range(n_regimes):
                regime_probs = regime_probabilities[:, regime]

                # Calculate distribution statistics
                regime_distributions[f'regime_{regime}'] = {
                    'mean': float(np.mean(regime_probs)),
                    'std': float(np.std(regime_probs)),
                    'min': float(np.min(regime_probs)),
                    'max': float(np.max(regime_probs)),
                    'median': float(np.median(regime_probs)),
                    'q25': float(np.percentile(regime_probs, 25)),
                    'q75': float(np.percentile(regime_probs, 75)),
                    'skewness': self._calculate_skewness(regime_probs),
                    'kurtosis': self._calculate_kurtosis(regime_probs)
                }

            # Cross-regime correlations
            correlations = {}
            for i in range(n_regimes):
                for j in range(i + 1, n_regimes):
                    corr = np.corrcoef(regime_probabilities[:, i], regime_probabilities[:, j])[0, 1]
                    correlations[f'regime_{i}_vs_regime_{j}'] = float(corr) if not np.isnan(corr) else 0.0

            return {
                'regime_distributions': regime_distributions,
                'cross_regime_correlations': correlations,
                'overall_probability_entropy': float(np.mean(-np.sum(regime_probabilities * np.log(regime_probabilities + 1e-10), axis=1)))
            }

        except Exception as e:
            tprint(f"⚠️ [REGIME_ANALYZER] Probability distribution analysis failed: {e}", color="yellow")
            return {'error': str(e)}

    def _analyze_regime_characteristics(
        self,
        regime_labels: np.ndarray,
        regime_probabilities: np.ndarray,
        n_regimes: int
    ) -> Dict[str, Any]:
        """Analyze characteristics of each regime."""
        try:
            tprint("🎯 [REGIME_ANALYZER] Analyzing regime characteristics", color="blue")

            regime_characteristics = {}

            for regime in range(n_regimes):
                regime_mask = (regime_labels == regime)
                regime_probs = regime_probabilities[regime_mask, regime] if regime_mask.any() else np.array([])

                if len(regime_probs) > 0:
                    regime_characteristics[f'regime_{regime}'] = {
                        'sample_count': int(np.sum(regime_mask)),
                        'percentage': float(np.sum(regime_mask) / len(regime_labels) * 100),
                        'avg_confidence': float(np.mean(regime_probs)),
                        'confidence_std': float(np.std(regime_probs)),
                        'min_confidence': float(np.min(regime_probs)),
                        'max_confidence': float(np.max(regime_probs)),
                        'confidence_consistency': float(1.0 - np.std(regime_probs)) if len(regime_probs) > 1 else 1.0,
                        'dominance_score': float(np.mean(regime_probs - np.mean(regime_probabilities[regime_mask, :], axis=1))) if len(regime_probs) > 0 else 0.0
                    }
                else:
                    regime_characteristics[f'regime_{regime}'] = {
                        'sample_count': 0,
                        'percentage': 0.0,
                        'avg_confidence': 0.0,
                        'confidence_std': 0.0,
                        'min_confidence': 0.0,
                        'max_confidence': 0.0,
                        'confidence_consistency': 0.0,
                        'dominance_score': 0.0
                    }

            return regime_characteristics

        except Exception as e:
            tprint(f"⚠️ [REGIME_ANALYZER] Regime characteristics analysis failed: {e}", color="yellow")
            return {'error': str(e)}

    def _analyze_uncertainty_metrics(self, regime_probabilities: np.ndarray) -> Dict[str, Any]:
        """Analyze uncertainty metrics for regime predictions."""
        try:
            tprint("🎲 [REGIME_ANALYZER] Analyzing uncertainty metrics", color="blue")

            # Calculate entropy for each sample
            entropy = -np.sum(regime_probabilities * np.log(regime_probabilities + 1e-10), axis=1)

            # Calculate dominance (difference between top 2 probabilities)
            sorted_probs = np.sort(regime_probabilities, axis=1)
            dominance = sorted_probs[:, -1] - sorted_probs[:, -2] if regime_probabilities.shape[1] > 1 else np.ones(len(regime_probabilities))

            # Uncertainty categories
            very_low_uncertainty = np.sum(entropy < 0.2)
            low_uncertainty = np.sum((entropy >= 0.2) & (entropy < 0.5))
            medium_uncertainty = np.sum((entropy >= 0.5) & (entropy < 1.0))
            high_uncertainty = np.sum((entropy >= 1.0) & (entropy < 1.5))
            very_high_uncertainty = np.sum(entropy >= 1.5)

            return {
                'entropy_statistics': {
                    'mean': float(np.mean(entropy)),
                    'std': float(np.std(entropy)),
                    'min': float(np.min(entropy)),
                    'max': float(np.max(entropy)),
                    'median': float(np.median(entropy))
                },
                'dominance_statistics': {
                    'mean': float(np.mean(dominance)),
                    'std': float(np.std(dominance)),
                    'min': float(np.min(dominance)),
                    'max': float(np.max(dominance)),
                    'median': float(np.median(dominance))
                },
                'uncertainty_distribution': {
                    'very_low': int(very_low_uncertainty),
                    'low': int(low_uncertainty),
                    'medium': int(medium_uncertainty),
                    'high': int(high_uncertainty),
                    'very_high': int(very_high_uncertainty)
                },
                'uncertainty_ratios': {
                    'very_low': float(very_low_uncertainty / len(entropy)),
                    'low': float(low_uncertainty / len(entropy)),
                    'medium': float(medium_uncertainty / len(entropy)),
                    'high': float(high_uncertainty / len(entropy)),
                    'very_high': float(very_high_uncertainty / len(entropy))
                }
            }

        except Exception as e:
            tprint(f"⚠️ [REGIME_ANALYZER] Uncertainty analysis failed: {e}", color="yellow")
            return {'error': str(e)}

    def _analyze_ensemble_predictions(self, ensemble_probabilities: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Analyze ensemble prediction consistency and agreement."""
        try:
            tprint("🤝 [REGIME_ANALYZER] Analyzing ensemble predictions", color="blue")

            if not ensemble_probabilities:
                return {'error': 'No ensemble probabilities provided'}

            model_names = list(ensemble_probabilities.keys())
            n_models = len(model_names)

            if n_models < 2:
                return {'error': 'Need at least 2 models for ensemble analysis'}

            # Calculate agreement between models
            agreement_metrics = {}
            for i, model1 in enumerate(model_names):
                for j, model2 in enumerate(model_names[i+1:], i+1):
                    prob1 = ensemble_probabilities[model1]
                    prob2 = ensemble_probabilities[model2]

                    # Calculate correlation between probability vectors
                    if prob1.shape == prob2.shape:
                        # Flatten and calculate correlation
                        corr = np.corrcoef(prob1.flatten(), prob2.flatten())[0, 1]
                        agreement_metrics[f'{model1}_vs_{model2}'] = float(corr) if not np.isnan(corr) else 0.0

            # Calculate ensemble consensus
            all_probs = np.array(list(ensemble_probabilities.values()))
            ensemble_mean = np.mean(all_probs, axis=0)
            ensemble_std = np.std(all_probs, axis=0)

            return {
                'n_models': n_models,
                'model_names': model_names,
                'agreement_metrics': agreement_metrics,
                'ensemble_consensus': {
                    'mean_probabilities': ensemble_mean.tolist(),
                    'std_probabilities': ensemble_std.tolist(),
                    'consensus_strength': float(1.0 - np.mean(ensemble_std)),
                    'disagreement_level': float(np.mean(ensemble_std))
                }
            }

        except Exception as e:
            tprint(f"⚠️ [REGIME_ANALYZER] Ensemble analysis failed: {e}", color="yellow")
            return {'error': str(e)}

    def _calculate_quality_metrics(
        self,
        regime_labels: np.ndarray,
        regime_probabilities: np.ndarray,
        n_regimes: int
    ) -> Dict[str, Any]:
        """Calculate quality metrics for regime predictions."""
        try:
            tprint("⭐ [REGIME_ANALYZER] Calculating quality metrics", color="blue")

            # Prediction confidence
            max_probs = np.max(regime_probabilities, axis=1)
            avg_confidence = np.mean(max_probs)

            # Regime balance
            regime_counts = np.bincount(regime_labels, minlength=n_regimes)
            regime_balance = 1.0 - np.std(regime_counts / len(regime_labels))

            # Probability consistency
            entropy = -np.sum(regime_probabilities * np.log(regime_probabilities + 1e-10), axis=1)
            avg_entropy = np.mean(entropy)
            consistency_score = 1.0 - (avg_entropy / np.log(n_regimes)) if n_regimes > 1 else 1.0

            # Regime stability (how often the same regime is predicted)
            regime_changes = np.sum(regime_labels[1:] != regime_labels[:-1])
            stability_score = 1.0 - (regime_changes / (len(regime_labels) - 1)) if len(regime_labels) > 1 else 1.0

            return {
                'prediction_confidence': float(avg_confidence),
                'regime_balance': float(regime_balance),
                'consistency_score': float(consistency_score),
                'stability_score': float(stability_score),
                'overall_quality': float((avg_confidence + regime_balance + consistency_score + stability_score) / 4),
                'regime_changes': int(regime_changes),
                'change_rate': float(regime_changes / (len(regime_labels) - 1)) if len(regime_labels) > 1 else 0.0
            }

        except Exception as e:
            tprint(f"⚠️ [REGIME_ANALYZER] Quality metrics calculation failed: {e}", color="yellow")
            return {'error': str(e)}

    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        try:
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return 0.0
            return float(np.mean(((data - mean) / std) ** 3))
        except:
            return 0.0

    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data."""
        try:
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return 0.0
            return float(np.mean(((data - mean) / std) ** 4)) - 3.0
        except:
            return 0.0

    def generate_comprehensive_report(
        self,
        analysis_results: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate a comprehensive text report from analysis results.

        Args:
            analysis_results: Dictionary containing analysis results
            output_path: Optional path to save the report

        Returns:
            String containing the comprehensive report
        """
        try:
            tprint("📝 [REGIME_ANALYZER] Generating comprehensive report", color="cyan")

            model_name = analysis_results.get('model_name', 'Unknown Model')
            timestamp = analysis_results.get('analysis_timestamp', datetime.now().isoformat())

            report = []
            report.append("=" * 80)
            report.append(f"COMPREHENSIVE REGIME PROBABILITY ANALYSIS REPORT")
            report.append(f"Model: {model_name}")
            report.append(f"Generated: {timestamp}")
            report.append("=" * 80)
            report.append("")

            # Basic Statistics
            basic_stats = analysis_results.get('basic_statistics', {})
            if 'error' not in basic_stats:
                report.append("📊 BASIC STATISTICS")
                report.append("-" * 40)
                report.append(f"Total Samples: {basic_stats.get('total_samples', 'N/A')}")
                report.append(f"Number of Regimes: {basic_stats.get('n_regimes', 'N/A')}")

                regime_dist = basic_stats.get('regime_distribution', {})
                report.append(f"Most Common Regime: {regime_dist.get('most_common_regime', 'N/A')}")
                report.append(f"Regime Balance: {regime_dist.get('regime_balance', 0):.3f}")

                prob_stats = basic_stats.get('probability_statistics', {})
                report.append(f"Mean Max Probability: {prob_stats.get('mean_max_probability', 0):.3f}")
                report.append(f"Std Max Probability: {prob_stats.get('std_max_probability', 0):.3f}")
                report.append("")

            # Quality Metrics
            quality_metrics = analysis_results.get('quality_metrics', {})
            if 'error' not in quality_metrics:
                report.append("⭐ QUALITY METRICS")
                report.append("-" * 40)
                report.append(f"Prediction Confidence: {quality_metrics.get('prediction_confidence', 0):.3f}")
                report.append(f"Regime Balance: {quality_metrics.get('regime_balance', 0):.3f}")
                report.append(f"Consistency Score: {quality_metrics.get('consistency_score', 0):.3f}")
                report.append(f"Stability Score: {quality_metrics.get('stability_score', 0):.3f}")
                report.append(f"Overall Quality: {quality_metrics.get('overall_quality', 0):.3f}")
                report.append("")

            # Uncertainty Analysis
            uncertainty = analysis_results.get('uncertainty_analysis', {})
            if 'error' not in uncertainty:
                report.append("🎲 UNCERTAINTY ANALYSIS")
                report.append("-" * 40)
                entropy_stats = uncertainty.get('entropy_statistics', {})
                report.append(f"Mean Entropy: {entropy_stats.get('mean', 0):.3f}")
                report.append(f"Std Entropy: {entropy_stats.get('std', 0):.3f}")

                uncertainty_dist = uncertainty.get('uncertainty_distribution', {})
                report.append(f"High Uncertainty Samples: {uncertainty_dist.get('high', 0)}")
                report.append(f"Low Uncertainty Samples: {uncertainty_dist.get('low', 0)}")
                report.append("")

            # Regime Characteristics
            regime_chars = analysis_results.get('regime_characteristics', {})
            if 'error' not in regime_chars:
                report.append("🎯 REGIME CHARACTERISTICS")
                report.append("-" * 40)
                for regime_key, regime_data in regime_chars.items():
                    if isinstance(regime_data, dict) and 'error' not in regime_data:
                        report.append(f"{regime_key.upper()}:")
                        report.append(f"  Sample Count: {regime_data.get('sample_count', 0)}")
                        report.append(f"  Percentage: {regime_data.get('percentage', 0):.1f}%")
                        report.append(f"  Avg Confidence: {regime_data.get('avg_confidence', 0):.3f}")
                        report.append(f"  Confidence Consistency: {regime_data.get('confidence_consistency', 0):.3f}")
                        report.append("")

            # Ensemble Analysis
            ensemble = analysis_results.get('ensemble_analysis', {})
            if 'error' not in ensemble and ensemble:
                report.append("🤝 ENSEMBLE ANALYSIS")
                report.append("-" * 40)
                report.append(f"Number of Models: {ensemble.get('n_models', 0)}")
                consensus = ensemble.get('ensemble_consensus', {})
                report.append(f"Consensus Strength: {consensus.get('consensus_strength', 0):.3f}")
                report.append(f"Disagreement Level: {consensus.get('disagreement_level', 0):.3f}")
                report.append("")

            report.append("=" * 80)
            report.append("END OF REPORT")
            report.append("=" * 80)

            report_text = "\n".join(report)

            # Save report if output path is provided
            if output_path:
                try:
                    with open(output_path, 'w') as f:
                        f.write(report_text)
                    tprint(f"✅ [REGIME_ANALYZER] Report saved to {output_path}", color="green")
                except Exception as e:
                    tprint(f"⚠️ [REGIME_ANALYZER] Failed to save report: {e}", color="yellow")

            return report_text

        except Exception as e:
            tprint(f"❌ [REGIME_ANALYZER] Report generation failed: {e}", color="red")
            return f"Error generating report: {e}"

    def export_analysis_to_json(
        self,
        analysis_results: Dict[str, Any],
        output_path: str
    ) -> bool:
        """
        Export analysis results to JSON file.

        Args:
            analysis_results: Dictionary containing analysis results
            output_path: Path to save the JSON file

        Returns:
            Boolean indicating success
        """
        try:
            tprint(f"💾 [REGIME_ANALYZER] Exporting analysis to {output_path}", color="cyan")

            # Convert numpy arrays to lists for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {key: convert_numpy(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                else:
                    return obj

            json_data = convert_numpy(analysis_results)

            with open(output_path, 'w') as f:
                json.dump(json_data, f, indent=2, default=str)

            tprint(f"✅ [REGIME_ANALYZER] Analysis exported successfully", color="green")
            return True

        except Exception as e:
            tprint(f"❌ [REGIME_ANALYZER] Export failed: {e}", color="red")
            return False
