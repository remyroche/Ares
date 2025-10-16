"""
Comparison Utilities for NAS/TAS Systems

This module provides comprehensive comparison utilities for analyzing and
comparing results between different NAS and TAS implementations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu, wilcoxon
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error,
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)

from .result_manager import UnifiedArchitectureResult, ArchitectureResult, ComparisonResult

@dataclass
class ArchitectureComparison:
    """Detailed architecture comparison results."""

    # Architecture identifiers
    arch_1_id: str = ""
    arch_2_id: str = ""

    # Architecture configurations
    arch_1_config: Dict[str, Any] = field(default_factory=dict)
    arch_2_config: Dict[str, Any] = field(default_factory=dict)

    # Performance comparison
    performance_differences: Dict[str, float] = field(default_factory=dict)
    performance_significance: Dict[str, float] = field(default_factory=dict)

    # Complexity comparison
    complexity_comparison: Dict[str, float] = field(default_factory=dict)

    # Efficiency metrics
    efficiency_metrics: Dict[str, float] = field(default_factory=dict)

    # Overall comparison
    better_architecture: str = ""
    improvement_magnitude: float = 0.0
    confidence_level: float = 0.95

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'arch_1_id': self.arch_1_id,
            'arch_2_id': self.arch_2_id,
            'arch_1_config': self.arch_1_config,
            'arch_2_config': self.arch_2_config,
            'performance_differences': self.performance_differences,
            'performance_significance': self.performance_significance,
            'complexity_comparison': self.complexity_comparison,
            'efficiency_metrics': self.efficiency_metrics,
            'better_architecture': self.better_architecture,
            'improvement_magnitude': self.improvement_magnitude,
            'confidence_level': self.confidence_level
        }

@dataclass
class PerformanceComparison:
    """Performance-focused comparison results."""

    # Performance metrics comparison
    accuracy_comparison: Dict[str, float] = field(default_factory=dict)
    precision_comparison: Dict[str, float] = field(default_factory=dict)
    recall_comparison: Dict[str, float] = field(default_factory=dict)
    f1_comparison: Dict[str, float] = field(default_factory=dict)

    # Statistical significance
    statistical_tests: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Performance distribution
    performance_distributions: Dict[str, List[float]] = field(default_factory=dict)

    # Confidence intervals
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    # Effect sizes
    effect_sizes: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            'accuracy_comparison': self.accuracy_comparison,
            'precision_comparison': self.precision_comparison,
            'recall_comparison': self.recall_comparison,
            'f1_comparison': self.f1_comparison,
            'statistical_tests': self.statistical_tests,
            'performance_distributions': self.performance_distributions,
            'effect_sizes': self.effect_sizes
        }

        # Convert confidence intervals
        result['confidence_intervals'] = {
            k: list(v) for k, v in self.confidence_intervals.items()
        }

        return result

@dataclass
class FinancialComparison:
    """Financial performance comparison results."""

    # Financial metrics comparison
    return_comparison: Dict[str, float] = field(default_factory=dict)
    risk_comparison: Dict[str, float] = field(default_factory=dict)
    risk_adjusted_return_comparison: Dict[str, float] = field(default_factory=dict)

    # Trading performance
    trading_metrics_comparison: Dict[str, float] = field(default_factory=dict)

    # Risk-adjusted metrics
    sharpe_comparison: Dict[str, float] = field(default_factory=dict)
    sortino_comparison: Dict[str, float] = field(default_factory=dict)
    calmar_comparison: Dict[str, float] = field(default_factory=dict)

    # Drawdown analysis
    drawdown_comparison: Dict[str, float] = field(default_factory=dict)

    # Portfolio metrics
    portfolio_metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'return_comparison': self.return_comparison,
            'risk_comparison': self.risk_comparison,
            'risk_adjusted_return_comparison': self.risk_adjusted_return_comparison,
            'trading_metrics_comparison': self.trading_metrics_comparison,
            'sharpe_comparison': self.sharpe_comparison,
            'sortino_comparison': self.sortino_comparison,
            'calmar_comparison': self.calmar_comparison,
            'drawdown_comparison': self.drawdown_comparison,
            'portfolio_metrics': self.portfolio_metrics
        }

@dataclass
class RegimeComparison:
    """Regime-specific comparison results."""

    # Regime performance
    regime_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Regime stability
    regime_stability_comparison: Dict[str, float] = field(default_factory=dict)

    # Adaptation speed
    adaptation_comparison: Dict[str, float] = field(default_factory=dict)

    # Regime transition analysis
    transition_analysis: Dict[str, Any] = field(default_factory=dict)

    # Regime-specific recommendations
    regime_recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'regime_performance': self.regime_performance,
            'regime_stability_comparison': self.regime_stability_comparison,
            'adaptation_comparison': self.adaptation_comparison,
            'transition_analysis': self.transition_analysis,
            'regime_recommendations': self.regime_recommendations
        }

class ResultComparison:
    """
    Comprehensive result comparison utility.

    This class consolidates comparison logic that was previously scattered
    across NAS and TAS implementations, providing unified comparison
    capabilities with statistical analysis.
    """

    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize result comparison utility.

        Args:
            confidence_level: Statistical confidence level for tests
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        self.logger = logging.getLogger(self.__class__.__name__)

        tprint_info(f"Result comparison utility initialized (confidence: {confidence_level:.1%})")

    def compare_architectures(
        self,
        arch_1: ArchitectureResult,
        arch_2: ArchitectureResult
    ) -> ArchitectureComparison:
        """
        Compare two individual architectures.

        Args:
            arch_1: First architecture result
            arch_2: Second architecture result

        Returns:
            ArchitectureComparison with detailed comparison results
        """
        tprint_info(f"Comparing architectures: {arch_1.architecture_id} vs {arch_2.architecture_id}")

        try:
            comparison = ArchitectureComparison(
                arch_1_id=arch_1.architecture_id,
                arch_2_id=arch_2.architecture_id,
                arch_1_config=arch_1.architecture_config,
                arch_2_config=arch_2.architecture_config,
                confidence_level=self.confidence_level
            )

            # Performance comparison
            comparison.performance_differences = self._compare_performance_metrics(
                arch_1.performance_metrics,
                arch_2.performance_metrics
            )

            # Complexity comparison
            comparison.complexity_comparison = self._compare_complexity(arch_1, arch_2)

            # Efficiency comparison
            comparison.efficiency_metrics = self._compare_efficiency(arch_1, arch_2)

            # Determine better architecture
            comparison.better_architecture, comparison.improvement_magnitude = self._determine_better_architecture(
                arch_1, arch_2, comparison
            )

            tprint_success(f"Architecture comparison completed: {comparison.better_architecture} is better "
                          f"({comparison.improvement_magnitude:.1%} improvement)")

            return comparison

        except Exception as e:
            tprint_error(f"Error comparing architectures: {e}")
            return ArchitectureComparison()

    def compare_results(
        self,
        result_1: UnifiedArchitectureResult,
        result_2: UnifiedArchitectureResult
    ) -> Dict[str, Any]:
        """
        Compare two complete search results.

        Args:
            result_1: First search result
            result_2: Second search result

        Returns:
            Comprehensive comparison results
        """
        tprint_info(f"Comparing results: {result_1.result_id} vs {result_2.result_id}")

        try:
            comparison_results = {}

            # Architecture comparison
            if result_1.best_architecture and result_2.best_architecture:
                arch_comparison = self.compare_architectures(
                    result_1.best_architecture,
                    result_2.best_architecture
                )
                comparison_results['architecture_comparison'] = arch_comparison.to_dict()

            # Performance comparison
            performance_comparison = self._compare_result_performance(result_1, result_2)
            comparison_results['performance_comparison'] = performance_comparison.to_dict()

            # Financial comparison
            financial_comparison = self._compare_financial_performance(result_1, result_2)
            comparison_results['financial_comparison'] = financial_comparison.to_dict()

            # Regime comparison
            regime_comparison = self._compare_regime_performance(result_1, result_2)
            comparison_results['regime_comparison'] = regime_comparison.to_dict()

            # Overall comparison
            comparison_results['overall_comparison'] = self._generate_overall_comparison(
                result_1, result_2, comparison_results
            )

            tprint_success("Results comparison completed")
            return comparison_results

        except Exception as e:
            tprint_error(f"Error comparing results: {e}")
            return {}

    def _compare_performance_metrics(
        self,
        metrics_1: Dict[str, float],
        metrics_2: Dict[str, float]
    ) -> Dict[str, float]:
        """Compare performance metrics between two architectures."""
        differences = {}

        # Common metrics to compare
        common_metrics = set(metrics_1.keys()) & set(metrics_2.keys())

        for metric in common_metrics:
            diff = metrics_1[metric] - metrics_2[metric]
            differences[f"{metric}_difference"] = diff
            differences[f"{metric}_relative_improvement"] = (
                (metrics_1[metric] - metrics_2[metric]) / max(metrics_2[metric], 1e-8)
            )

        return differences

    def _compare_complexity(self, arch_1: ArchitectureResult, arch_2: ArchitectureResult) -> Dict[str, float]:
        """Compare architecture complexity."""
        comparison = {}

        # Model size comparison
        comparison['model_size_difference_mb'] = arch_1.model_size_mb - arch_2.model_size_mb
        comparison['model_size_ratio'] = arch_1.model_size_mb / max(arch_2.model_size_mb, 1e-8)

        # Model complexity comparison
        comparison['complexity_difference'] = arch_1.model_complexity - arch_2.model_complexity
        comparison['complexity_ratio'] = arch_1.model_complexity / max(arch_2.model_complexity, 1e-8)

        # Training complexity (if available)
        if 'training_time' in arch_1.training_info and 'training_time' in arch_2.training_info:
            comparison['training_time_difference'] = (
                arch_1.training_info['training_time'] - arch_2.training_info['training_time']
            )

        return comparison

    def _compare_efficiency(self, arch_1: ArchitectureResult, arch_2: ArchitectureResult) -> Dict[str, float]:
        """Compare architecture efficiency."""
        efficiency = {}

        # Performance per complexity
        if arch_1.model_complexity > 0 and arch_2.model_complexity > 0:
            if 'accuracy' in arch_1.performance_metrics and 'accuracy' in arch_2.performance_metrics:
                eff_1 = arch_1.performance_metrics['accuracy'] / arch_1.model_complexity
                eff_2 = arch_2.performance_metrics['accuracy'] / arch_2.model_complexity
                efficiency['accuracy_per_complexity_ratio'] = eff_1 / max(eff_2, 1e-8)

        # Performance per size
        if arch_1.model_size_mb > 0 and arch_2.model_size_mb > 0:
            if 'accuracy' in arch_1.performance_metrics and 'accuracy' in arch_2.performance_metrics:
                eff_1 = arch_1.performance_metrics['accuracy'] / arch_1.model_size_mb
                eff_2 = arch_2.performance_metrics['accuracy'] / arch_2.model_size_mb
                efficiency['accuracy_per_size_ratio'] = eff_1 / max(eff_2, 1e-8)

        return efficiency

    def _determine_better_architecture(
        self,
        arch_1: ArchitectureResult,
        arch_2: ArchitectureResult,
        comparison: ArchitectureComparison
    ) -> Tuple[str, float]:
        """Determine which architecture is better and by how much."""
        score_1 = self._calculate_architecture_score(arch_1)
        score_2 = self._calculate_architecture_score(arch_2)

        if score_1 > score_2:
            improvement = (score_1 - score_2) / max(score_2, 1e-8)
            return arch_1.architecture_id, improvement
        else:
            improvement = (score_2 - score_1) / max(score_1, 1e-8)
            return arch_2.architecture_id, improvement

    def _calculate_architecture_score(self, arch: ArchitectureResult) -> float:
        """Calculate overall architecture score."""
        score = 0.0

        # Performance score (40%)
        if 'accuracy' in arch.performance_metrics:
            score += arch.performance_metrics['accuracy'] * 0.4

        if 'f1_score' in arch.performance_metrics:
            score += arch.performance_metrics['f1_score'] * 0.2

        # Financial score (30%)
        if arch.financial_metrics:
            score += arch.financial_metrics.sharpe_ratio * 0.2
            score += (1 - abs(arch.financial_metrics.max_drawdown)) * 0.1

        # Efficiency score (20%)
        if arch.model_complexity > 0 and 'accuracy' in arch.performance_metrics:
            efficiency = arch.performance_metrics['accuracy'] / arch.model_complexity
            score += efficiency * 0.2

        # Size efficiency (10%)
        if arch.model_size_mb > 0 and 'accuracy' in arch.performance_metrics:
            size_efficiency = arch.performance_metrics['accuracy'] / arch.model_size_mb
            score += size_efficiency * 0.1

        return score

    def _compare_result_performance(
        self,
        result_1: UnifiedArchitectureResult,
        result_2: UnifiedArchitectureResult
    ) -> PerformanceComparison:
        """Compare performance between two results."""
        comparison = PerformanceComparison()

        # Compare best architectures
        if result_1.best_architecture and result_2.best_architecture:
            arch_1_perf = result_1.best_architecture.performance_metrics
            arch_2_perf = result_2.best_architecture.performance_metrics

            # Individual metric comparisons
            comparison.accuracy_comparison = self._compare_single_metric(
                arch_1_perf.get('accuracy', 0),
                arch_2_perf.get('accuracy', 0),
                'accuracy'
            )

            comparison.precision_comparison = self._compare_single_metric(
                arch_1_perf.get('precision', 0),
                arch_2_perf.get('precision', 0),
                'precision'
            )

            comparison.recall_comparison = self._compare_single_metric(
                arch_1_perf.get('recall', 0),
                arch_2_perf.get('recall', 0),
                'recall'
            )

            comparison.f1_comparison = self._compare_single_metric(
                arch_1_perf.get('f1_score', 0),
                arch_2_perf.get('f1_score', 0),
                'f1_score'
            )

        # Compare all architectures if available
        if len(result_1.all_architectures) > 1 and len(result_2.all_architectures) > 1:
            comparison.performance_distributions = self._compare_performance_distributions(
                result_1.all_architectures,
                result_2.all_architectures
            )

        return comparison

    def _compare_single_metric(self, value_1: float, value_2: float, metric_name: str) -> Dict[str, float]:
        """Compare a single metric between two values."""
        return {
            f'{metric_name}_1': value_1,
            f'{metric_name}_2': value_2,
            f'{metric_name}_difference': value_1 - value_2,
            f'{metric_name}_relative_improvement': (value_1 - value_2) / max(value_2, 1e-8)
        }

    def _compare_performance_distributions(
        self,
        architectures_1: List[ArchitectureResult],
        architectures_2: List[ArchitectureResult]
    ) -> Dict[str, List[float]]:
        """Compare performance distributions between result sets."""
        distributions = {}

        # Extract performance metrics
        metrics = ['accuracy', 'f1_score', 'precision', 'recall']

        for metric in metrics:
            values_1 = []
            values_2 = []

            for arch in architectures_1:
                if metric in arch.performance_metrics:
                    values_1.append(arch.performance_metrics[metric])

            for arch in architectures_2:
                if metric in arch.performance_metrics:
                    values_2.append(arch.performance_metrics[metric])

            if values_1 and values_2:
                distributions[f'{metric}_distribution_1'] = values_1
                distributions[f'{metric}_distribution_2'] = values_2

        return distributions

    def _compare_financial_performance(
        self,
        result_1: UnifiedArchitectureResult,
        result_2: UnifiedArchitectureResult
    ) -> FinancialComparison:
        """Compare financial performance between two results."""
        comparison = FinancialComparison()

        # Compare best architectures' financial metrics
        if (result_1.best_architecture and result_1.best_architecture.financial_metrics and
            result_2.best_architecture and result_2.best_architecture.financial_metrics):

            fin_1 = result_1.best_architecture.financial_metrics
            fin_2 = result_2.best_architecture.financial_metrics

            # Return comparison
            comparison.return_comparison = {
                'total_return_1': fin_1.total_return,
                'total_return_2': fin_2.total_return,
                'return_difference': fin_1.total_return - fin_2.total_return,
                'annualized_return_1': fin_1.annualized_return,
                'annualized_return_2': fin_2.annualized_return,
                'annualized_return_difference': fin_1.annualized_return - fin_2.annualized_return
            }

            # Risk comparison
            comparison.risk_comparison = {
                'volatility_1': fin_1.volatility,
                'volatility_2': fin_2.volatility,
                'volatility_difference': fin_1.volatility - fin_2.volatility,
                'max_drawdown_1': fin_1.max_drawdown,
                'max_drawdown_2': fin_2.max_drawdown,
                'max_drawdown_difference': fin_1.max_drawdown - fin_2.max_drawdown
            }

            # Risk-adjusted return comparison
            comparison.sharpe_comparison = {
                'sharpe_1': fin_1.sharpe_ratio,
                'sharpe_2': fin_2.sharpe_ratio,
                'sharpe_difference': fin_1.sharpe_ratio - fin_2.sharpe_ratio
            }

            comparison.sortino_comparison = {
                'sortino_1': fin_1.sortino_ratio,
                'sortino_2': fin_2.sortino_ratio,
                'sortino_difference': fin_1.sortino_ratio - fin_2.sortino_ratio
            }

            comparison.calmar_comparison = {
                'calmar_1': fin_1.calmar_ratio,
                'calmar_2': fin_2.calmar_ratio,
                'calmar_difference': fin_1.calmar_ratio - fin_2.calmar_ratio
            }

            # Trading metrics
            comparison.trading_metrics_comparison = {
                'win_rate_1': fin_1.win_rate,
                'win_rate_2': fin_2.win_rate,
                'win_rate_difference': fin_1.win_rate - fin_2.win_rate,
                'profit_factor_1': fin_1.profit_factor,
                'profit_factor_2': fin_2.profit_factor,
                'profit_factor_difference': fin_1.profit_factor - fin_2.profit_factor
            }

        return comparison

    def _compare_regime_performance(
        self,
        result_1: UnifiedArchitectureResult,
        result_2: UnifiedArchitectureResult
    ) -> RegimeComparison:
        """Compare regime-specific performance between two results."""
        comparison = RegimeComparison()

        # Compare regime analysis if available
        if 'regime_performance' in result_1.regime_analysis and 'regime_performance' in result_2.regime_analysis:
            regime_perf_1 = result_1.regime_analysis['regime_performance']
            regime_perf_2 = result_2.regime_analysis['regime_performance']

            comparison.regime_performance = {
                'result_1': regime_perf_1,
                'result_2': regime_perf_2
            }

        # Compare regime stability
        if 'regime_stability' in result_1.regime_analysis and 'regime_stability' in result_2.regime_analysis:
            stability_1 = result_1.regime_analysis['regime_stability']
            stability_2 = result_2.regime_analysis['regime_stability']

            comparison.regime_stability_comparison = {
                'stability_1': stability_1,
                'stability_2': stability_2,
                'stability_difference': stability_1 - stability_2
            }

        # Compare adaptation speed
        if 'adaptation_speed' in result_1.regime_analysis and 'adaptation_speed' in result_2.regime_analysis:
            adaptation_1 = result_1.regime_analysis['adaptation_speed']
            adaptation_2 = result_2.regime_analysis['adaptation_speed']

            comparison.adaptation_comparison = {
                'adaptation_1': adaptation_1,
                'adaptation_2': adaptation_2,
                'adaptation_difference': adaptation_1 - adaptation_2
            }

        return comparison

    def _generate_overall_comparison(
        self,
        result_1: UnifiedArchitectureResult,
        result_2: UnifiedArchitectureResult,
        comparison_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate overall comparison summary."""
        overall = {
            'result_1_id': result_1.result_id,
            'result_2_id': result_2.result_id,
            'comparison_timestamp': datetime.now().isoformat(),
            'search_type_1': result_1.search_type,
            'search_type_2': result_2.search_type,
            'architecture_count_1': result_1.architecture_count,
            'architecture_count_2': result_2.architecture_count
        }

        # Determine overall winner
        if 'architecture_comparison' in comparison_results:
            arch_comp = comparison_results['architecture_comparison']
            overall['better_architecture'] = arch_comp.get('better_architecture', '')
            overall['improvement_magnitude'] = arch_comp.get('improvement_magnitude', 0.0)

        # Summary recommendations
        recommendations = []

        if 'financial_comparison' in comparison_results:
            fin_comp = comparison_results['financial_comparison']
            if fin_comp.get('sharpe_comparison', {}).get('sharpe_difference', 0) > 0:
                recommendations.append("Result 1 shows better risk-adjusted returns")
            elif fin_comp.get('sharpe_comparison', {}).get('sharpe_difference', 0) < 0:
                recommendations.append("Result 2 shows better risk-adjusted returns")

        overall['recommendations'] = recommendations

        return overall

    def generate_comparison_report(
        self,
        comparison_results: Dict[str, Any],
        output_file: Optional[str] = None
    ) -> str:
        """Generate a detailed comparison report."""
        report_lines = []

        # Header
        report_lines.append("=" * 80)
        report_lines.append("NAS/TAS RESULTS COMPARISON REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")

        # Overall comparison
        if 'overall_comparison' in comparison_results:
            overall = comparison_results['overall_comparison']
            report_lines.append("OVERALL COMPARISON")
            report_lines.append("-" * 40)
            report_lines.append(f"Result 1 ID: {overall.get('result_1_id', 'N/A')}")
            report_lines.append(f"Result 2 ID: {overall.get('result_2_id', 'N/A')}")
            report_lines.append(f"Better Architecture: {overall.get('better_architecture', 'N/A')}")
            report_lines.append(f"Improvement: {overall.get('improvement_magnitude', 0):.1%}")
            report_lines.append("")

        # Performance comparison
        if 'performance_comparison' in comparison_results:
            perf_comp = comparison_results['performance_comparison']
            report_lines.append("PERFORMANCE COMPARISON")
            report_lines.append("-" * 40)

            for metric in ['accuracy_comparison', 'f1_comparison', 'precision_comparison', 'recall_comparison']:
                if metric in perf_comp:
                    comp = perf_comp[metric]
                    report_lines.append(f"{metric.replace('_comparison', '').title()}:")
                    metric_name = metric.replace('_comparison', '')
                    report_lines.append(f"  Result 1: {comp.get(f'{metric_name}_1', 0):.3f}")
                    report_lines.append(f"  Result 2: {comp.get(f'{metric_name}_2', 0):.3f}")
                    report_lines.append(f"  Difference: {comp.get(f'{metric_name}_difference', 0):.3f}")
                    report_lines.append("")

        # Financial comparison
        if 'financial_comparison' in comparison_results:
            fin_comp = comparison_results['financial_comparison']
            report_lines.append("FINANCIAL COMPARISON")
            report_lines.append("-" * 40)

            if 'sharpe_comparison' in fin_comp:
                sharpe = fin_comp['sharpe_comparison']
                report_lines.append(f"Sharpe Ratio:")
                report_lines.append(f"  Result 1: {sharpe.get('sharpe_1', 0):.3f}")
                report_lines.append(f"  Result 2: {sharpe.get('sharpe_2', 0):.3f}")
                report_lines.append(f"  Difference: {sharpe.get('sharpe_difference', 0):.3f}")
                report_lines.append("")

            if 'risk_comparison' in fin_comp:
                risk = fin_comp['risk_comparison']
                report_lines.append(f"Max Drawdown:")
                report_lines.append(f"  Result 1: {risk.get('max_drawdown_1', 0):.3f}")
                report_lines.append(f"  Result 2: {risk.get('max_drawdown_2', 0):.3f}")
                report_lines.append(f"  Difference: {risk.get('max_drawdown_difference', 0):.3f}")
                report_lines.append("")

        # Recommendations
        if 'overall_comparison' in comparison_results:
            recommendations = comparison_results['overall_comparison'].get('recommendations', [])
            if recommendations:
                report_lines.append("RECOMMENDATIONS")
                report_lines.append("-" * 40)
                for i, rec in enumerate(recommendations, 1):
                    report_lines.append(f"{i}. {rec}")
                report_lines.append("")

        report_text = "\n".join(report_lines)

        # Save to file if requested
        if output_file:
            try:
                with open(output_file, 'w') as f:
                    f.write(report_text)
                tprint_success(f"Comparison report saved to {output_file}")
            except Exception as e:
                tprint_error(f"Failed to save comparison report: {e}")

        return report_text
