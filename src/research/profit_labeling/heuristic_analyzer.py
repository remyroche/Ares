"""
Heuristic Analyzer for Multi-Horizon Profit Labeling

This module provides data-driven analysis of the heuristics used in multi-horizon profit
labeling, similar to how we analyze HMM clustering. It examines the effectiveness of
different labeling strategies, parameter choices, and quality scoring methods.

Key Analysis Areas:
1. Target/Horizon Combination Effectiveness
2. Quality Scoring Heuristics Validation
3. Fee-Awareness Impact Analysis
4. Leverage-Adjusted Scoring Analysis
5. Time-to-Hit vs Quality Trade-offs
6. Composite Score Component Analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import json
from datetime import datetime
import warnings

from src.utils.logger import get_logger
from src.training.steps.pre_training.profit_labeling.consolidated_profit_labeler import (
    MultiHorizonProfitLabeler,
    MultiHorizonConfig
)

class AnalysisMetric(Enum):
    """Enumeration of analysis metrics for heuristic evaluation."""
    PREDICTIVE_POWER = "predictive_power"
    LABEL_STABILITY = "label_stability"
    TARGET_HIT_RATE = "target_hit_rate"
    QUALITY_CONSISTENCY = "quality_consistency"
    COMPOSITE_COHERENCE = "composite_coherence"
    PARAMETER_SENSITIVITY = "parameter_sensitivity"
    FEE_IMPACT_ANALYSIS = "fee_impact_analysis"
    LEVERAGE_EFFECTIVENESS = "leverage_effectiveness"
    TIME_DECAY_PATTERNS = "time_decay_patterns"
    REVERSAL_CAPTURE_QUALITY = "reversal_capture_quality"

@dataclass
class HeuristicAnalysisConfig:
    """Configuration for heuristic analysis."""
    # Analysis scope
    analyze_target_combinations: bool = True
    analyze_quality_scoring: bool = True
    analyze_composite_scores: bool = True
    analyze_parameter_sensitivity: bool = True

    # Validation parameters
    min_samples_per_analysis: int = 1000
    bootstrap_samples: int = 500
    confidence_level: float = 0.95

    # Sensitivity analysis
    parameter_variation_range: float = 0.5  # ±50% variation
    sensitivity_steps: int = 10

    # Quality thresholds
    min_predictive_power: float = 0.55  # Minimum AUC for useful labels
    min_stability_score: float = 0.7    # Minimum stability for reliable labels
    min_hit_rate: float = 0.1           # Minimum hit rate for valid targets

    # Comparison baselines
    compare_to_random: bool = True
    compare_to_simple_threshold: bool = True
    random_seed: int = 42

@dataclass
class HeuristicAnalysisResult:
    """Result container for heuristic analysis."""
    analysis_type: AnalysisMetric
    metric_value: float
    confidence_interval: Optional[Tuple[float, float]]
    statistical_significance: Optional[float]
    interpretation: str
    recommendations: List[str]
    metadata: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)

class HeuristicAnalyzer:
    """
    Comprehensive analyzer for multi-horizon profit labeling heuristics.

    This class provides data-driven analysis of the labeling system's heuristics,
    similar to how we analyze HMM clustering effectiveness. It examines:

    1. **Target/Horizon Effectiveness**: Which combinations work best?
    2. **Quality Scoring Validation**: Are quality scores predictive?
    3. **Parameter Sensitivity**: How sensitive are results to parameter changes?
    4. **Fee Impact**: How much do transaction costs affect labeling?
    5. **Composite Score Analysis**: Are composite scores adding value?
    """

    def __init__(self, config: Optional[HeuristicAnalysisConfig] = None):
        """Initialize the heuristic analyzer."""
        self.config = config or HeuristicAnalysisConfig()
        self.logger = get_logger('HeuristicAnalyzer')

        # Analysis results storage
        self.analysis_results: Dict[str, HeuristicAnalysisResult] = {}
        self.comparison_baselines: Dict[str, Any] = {}

        self.logger.info('🔬 Heuristic Analyzer initialized')
        self.logger.info(f'   → Analysis scope: {self._get_analysis_scope()}')

    def _get_analysis_scope(self) -> str:
        """Get human-readable analysis scope."""
        scope_items = []
        if self.config.analyze_target_combinations:
            scope_items.append("Target Combinations")
        if self.config.analyze_quality_scoring:
            scope_items.append("Quality Scoring")
        if self.config.analyze_composite_scores:
            scope_items.append("Composite Scores")
        if self.config.analyze_parameter_sensitivity:
            scope_items.append("Parameter Sensitivity")
        return ", ".join(scope_items)

    def analyze_labeling_heuristics(self,
                                  market_data: pd.DataFrame,
                                  labeling_config: Optional[MultiHorizonConfig] = None) -> Dict[str, HeuristicAnalysisResult]:
        """
        Comprehensive analysis of labeling heuristics.

        Args:
            market_data: OHLCV market data for analysis
            labeling_config: Configuration for the labeler (optional)

        Returns:
            Dictionary of analysis results by metric type
        """
        self.logger.info('🔍 Starting comprehensive heuristic analysis')

        if len(market_data) < self.config.min_samples_per_analysis:
            raise ValueError(f"Insufficient data: need {self.config.min_samples_per_analysis}, got {len(market_data)}")

        # Generate labels for analysis
        labeler = MultiHorizonProfitLabeler(labeling_config)
        labeled_data = labeler.generate_labels(market_data.copy())

        # Run all enabled analyses
        analyses = []

        if self.config.analyze_target_combinations:
            analyses.append(self._analyze_target_combinations)
        if self.config.analyze_quality_scoring:
            analyses.append(self._analyze_quality_scoring)
        if self.config.analyze_composite_scores:
            analyses.append(self._analyze_composite_scores)
        if self.config.analyze_parameter_sensitivity:
            analyses.append(self._analyze_parameter_sensitivity)

        # Execute analyses
        for analysis_func in analyses:
            try:
                result = analysis_func(labeled_data, labeler.config)
                if isinstance(result, dict):
                    self.analysis_results.update(result)
                else:
                    self.analysis_results[result.analysis_type.value] = result
            except Exception as e:
                self.logger.error(f"Analysis failed: {analysis_func.__name__}: {e}")

        # Generate baselines for comparison
        if self.config.compare_to_random or self.config.compare_to_simple_threshold:
            self._generate_comparison_baselines(market_data, labeled_data)

        self.logger.info(f'✅ Heuristic analysis completed: {len(self.analysis_results)} analyses')
        return self.analysis_results

    def _analyze_target_combinations(self,
                                   labeled_data: pd.DataFrame,
                                   config: MultiHorizonConfig) -> Dict[str, HeuristicAnalysisResult]:
        """Analyze effectiveness of different target/horizon combinations."""
        self.logger.info('🎯 Analyzing target/horizon combinations')

        results = {}

        # Extract target/horizon probability columns
        prob_columns = [col for col in labeled_data.columns if col.endswith('_prob')]

        for col in prob_columns:
            if '_prob' not in col:
                continue

            # Extract target and horizon from column name
            base_name = col.replace('_prob', '')
            parts = base_name.split('_')
            if len(parts) >= 2:
                target_name = parts[0]
                horizon_name = parts[1]

                # Analyze this combination
                prob_values = labeled_data[col].dropna()
                if len(prob_values) < 100:  # Skip if too few samples
                    continue

                # Calculate effectiveness metrics
                hit_rate = (prob_values > 0.5).mean()
                predictive_power = self._calculate_predictive_power(prob_values, labeled_data, col)
                stability = self._calculate_label_stability(prob_values)

                # Generate analysis result
                result = HeuristicAnalysisResult(
                    analysis_type=AnalysisMetric.TARGET_HIT_RATE,
                    metric_value=hit_rate,
                    confidence_interval=self._bootstrap_confidence_interval(prob_values, np.mean),
                    statistical_significance=None,
                    interpretation=f"Target {target_name} at {horizon_name} horizon has {hit_rate:.2%} hit rate",
                    recommendations=self._generate_target_recommendations(hit_rate, predictive_power, stability),
                    metadata={
                        'target': target_name,
                        'horizon': horizon_name,
                        'predictive_power': predictive_power,
                        'stability': stability,
                        'sample_size': len(prob_values)
                    }
                )

                results[f"{target_name}_{horizon_name}_effectiveness"] = result

        return results

    def _analyze_quality_scoring(self,
                               labeled_data: pd.DataFrame,
                               config: MultiHorizonConfig) -> HeuristicAnalysisResult:
        """Analyze the effectiveness of quality scoring heuristics."""
        self.logger.info('⭐ Analyzing quality scoring effectiveness')

        # Extract quality score columns
        quality_columns = [col for col in labeled_data.columns if col.endswith('_quality_score')]

        if not quality_columns:
            return HeuristicAnalysisResult(
                analysis_type=AnalysisMetric.QUALITY_CONSISTENCY,
                metric_value=0.0,
                confidence_interval=None,
                statistical_significance=None,
                interpretation="No quality scores found in labeled data",
                recommendations=["Enable quality scoring in labeling configuration"],
                metadata={'error': 'no_quality_scores'}
            )

        # Analyze quality score consistency and predictiveness
        quality_consistency_scores = []

        for col in quality_columns:
            quality_scores = labeled_data[col].dropna()
            if len(quality_scores) > 50:
                # Check if quality scores correlate with actual outcomes
                prob_col = col.replace('_quality_score', '_prob')
                if prob_col in labeled_data.columns:
                    correlation = np.corrcoef(quality_scores, labeled_data[prob_col].dropna())[0, 1]
                    if not np.isnan(correlation):
                        quality_consistency_scores.append(abs(correlation))

        if not quality_consistency_scores:
            consistency_score = 0.0
        else:
            consistency_score = np.mean(quality_consistency_scores)

        return HeuristicAnalysisResult(
            analysis_type=AnalysisMetric.QUALITY_CONSISTENCY,
            metric_value=consistency_score,
            confidence_interval=self._bootstrap_confidence_interval(
                np.array(quality_consistency_scores), np.mean
            ) if quality_consistency_scores else None,
            statistical_significance=None,
            interpretation=f"Quality scoring shows {consistency_score:.2%} consistency with probabilities",
            recommendations=self._generate_quality_recommendations(consistency_score),
            metadata={
                'analyzed_columns': len(quality_columns),
                'valid_correlations': len(quality_consistency_scores),
                'individual_scores': quality_consistency_scores
            }
        )

    def _analyze_composite_scores(self,
                                labeled_data: pd.DataFrame,
                                config: MultiHorizonConfig) -> Dict[str, HeuristicAnalysisResult]:
        """Analyze composite score effectiveness."""
        self.logger.info('🎯 Analyzing composite scores')

        results = {}

        # Key composite scores to analyze
        composite_columns = [
            'overall_opportunity',
            'leverage_adjusted_score',
            'immediate_opportunity',
            'short_term_opportunity',
            'reversal_capture_score'
        ]

        for col in composite_columns:
            if col not in labeled_data.columns:
                continue

            composite_values = labeled_data[col].dropna()
            if len(composite_values) < 100:
                continue

            # Analyze composite score properties
            coherence = self._calculate_composite_coherence(composite_values, labeled_data, col)
            predictive_power = self._calculate_predictive_power(composite_values, labeled_data, col)

            result = HeuristicAnalysisResult(
                analysis_type=AnalysisMetric.COMPOSITE_COHERENCE,
                metric_value=coherence,
                confidence_interval=None,
                statistical_significance=None,
                interpretation=f"Composite score {col} shows {coherence:.2%} coherence",
                recommendations=self._generate_composite_recommendations(col, coherence, predictive_power),
                metadata={
                    'score_type': col,
                    'predictive_power': predictive_power,
                    'value_range': (float(composite_values.min()), float(composite_values.max())),
                    'mean_value': float(composite_values.mean()),
                    'std_value': float(composite_values.std())
                }
            )

            results[f"{col}_analysis"] = result

        return results

    def _analyze_parameter_sensitivity(self,
                                     labeled_data: pd.DataFrame,
                                     config: MultiHorizonConfig) -> HeuristicAnalysisResult:
        """Analyze sensitivity to parameter changes."""
        self.logger.info('🔧 Analyzing parameter sensitivity')

        # This would require running the labeler with different parameters
        # For now, return a placeholder analysis

        return HeuristicAnalysisResult(
            analysis_type=AnalysisMetric.PARAMETER_SENSITIVITY,
            metric_value=0.5,  # Placeholder
            confidence_interval=None,
            statistical_significance=None,
            interpretation="Parameter sensitivity analysis requires multiple labeling runs",
            recommendations=[
                "Implement systematic parameter variation testing",
                "Test sensitivity to profit targets, time horizons, and quality weights",
                "Use grid search or Bayesian optimization for parameter tuning"
            ],
            metadata={'status': 'placeholder_implementation'}
        )

    def _calculate_predictive_power(self,
                                  values: pd.Series,
                                  labeled_data: pd.DataFrame,
                                  column: str) -> float:
        """Calculate predictive power of labels using future returns."""
        try:
            # Simple proxy: correlation with next-period returns
            if 'close' in labeled_data.columns and len(values) > 10:
                returns = labeled_data['close'].pct_change().shift(-1).dropna()
                common_idx = values.index.intersection(returns.index)

                if len(common_idx) > 10:
                    correlation = np.corrcoef(
                        values.loc[common_idx],
                        returns.loc[common_idx]
                    )[0, 1]
                    return abs(correlation) if not np.isnan(correlation) else 0.0

            return 0.0
        except Exception:
            return 0.0

    def _calculate_label_stability(self, values: pd.Series) -> float:
        """Calculate stability of labels over time."""
        try:
            if len(values) < 20:
                return 0.0

            # Calculate rolling correlation with itself
            window = min(50, len(values) // 4)
            rolling_std = values.rolling(window=window).std()
            stability = 1.0 - (rolling_std.mean() / values.std()) if values.std() > 0 else 0.0

            return max(0.0, min(1.0, stability))
        except Exception:
            return 0.0

    def _calculate_composite_coherence(self,
                                     composite_values: pd.Series,
                                     labeled_data: pd.DataFrame,
                                     column: str) -> float:
        """Calculate coherence of composite scores with individual components."""
        try:
            # Find related individual probability columns
            if column == 'overall_opportunity':
                related_cols = [col for col in labeled_data.columns if col.endswith('_prob')]
            elif column == 'immediate_opportunity':
                related_cols = [col for col in labeled_data.columns if 'immediate' in col and col.endswith('_prob')]
            elif column == 'short_term_opportunity':
                related_cols = [col for col in labeled_data.columns if 'short' in col and col.endswith('_prob')]
            else:
                related_cols = []

            if not related_cols:
                return 0.5  # Neutral coherence if no related columns

            # Calculate average correlation with related columns
            correlations = []
            for related_col in related_cols:
                if related_col in labeled_data.columns:
                    common_idx = composite_values.index.intersection(labeled_data[related_col].dropna().index)
                    if len(common_idx) > 10:
                        corr = np.corrcoef(
                            composite_values.loc[common_idx],
                            labeled_data[related_col].loc[common_idx]
                        )[0, 1]
                        if not np.isnan(corr):
                            correlations.append(abs(corr))

            return np.mean(correlations) if correlations else 0.5

        except Exception:
            return 0.5

    def _bootstrap_confidence_interval(self,
                                     data: Union[pd.Series, np.ndarray],
                                     statistic_func) -> Optional[Tuple[float, float]]:
        """Calculate bootstrap confidence interval."""
        try:
            if len(data) < 10:
                return None

            np.random.seed(self.config.random_seed)
            bootstrap_stats = []

            for _ in range(self.config.bootstrap_samples):
                sample = np.random.choice(data, size=len(data), replace=True)
                stat = statistic_func(sample)
                if not np.isnan(stat):
                    bootstrap_stats.append(stat)

            if not bootstrap_stats:
                return None

            alpha = 1 - self.config.confidence_level
            lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
            upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

            return (float(lower), float(upper))

        except Exception:
            return None

    def _generate_target_recommendations(self,
                                       hit_rate: float,
                                       predictive_power: float,
                                       stability: float) -> List[str]:
        """Generate recommendations for target/horizon combinations."""
        recommendations = []

        if hit_rate < self.config.min_hit_rate:
            recommendations.append(f"⚠️ Low hit rate ({hit_rate:.2%}) - consider adjusting target size or horizon")

        if predictive_power < self.config.min_predictive_power:
            recommendations.append(f"📉 Low predictive power ({predictive_power:.3f}) - labels may not be useful for ML")

        if stability < self.config.min_stability_score:
            recommendations.append(f"🔄 Low stability ({stability:.2%}) - labels are inconsistent over time")

        if hit_rate > 0.8:
            recommendations.append("✅ High hit rate - target may be too easy, consider tightening")

        if predictive_power > 0.7 and stability > 0.8:
            recommendations.append("🎯 Excellent combination - maintain current parameters")

        if not recommendations:
            recommendations.append("📊 Performance within acceptable ranges")

        return recommendations

    def _generate_quality_recommendations(self, consistency_score: float) -> List[str]:
        """Generate recommendations for quality scoring."""
        recommendations = []

        if consistency_score < 0.3:
            recommendations.extend([
                "⚠️ Quality scores poorly correlated with probabilities",
                "Consider revising quality scoring formula",
                "Review speed, risk, and profitability weightings"
            ])
        elif consistency_score < 0.6:
            recommendations.extend([
                "📊 Moderate quality score consistency",
                "Fine-tune quality scoring parameters",
                "Consider domain-specific adjustments"
            ])
        else:
            recommendations.append("✅ Quality scoring shows good consistency")

        return recommendations

    def _generate_composite_recommendations(self,
                                          score_type: str,
                                          coherence: float,
                                          predictive_power: float) -> List[str]:
        """Generate recommendations for composite scores."""
        recommendations = []

        if coherence < 0.4:
            recommendations.append(f"⚠️ {score_type} shows low coherence with components")

        if predictive_power < 0.3:
            recommendations.append(f"📉 {score_type} has low predictive power")

        if coherence > 0.7 and predictive_power > 0.5:
            recommendations.append(f"✅ {score_type} is well-calibrated and predictive")

        # Specific recommendations by score type
        if score_type == 'leverage_adjusted_score':
            recommendations.append("💡 Consider adjusting leverage weights based on market conditions")
        elif score_type == 'reversal_capture_score':
            recommendations.append("🔄 Validate reversal capture logic with backtesting")

        return recommendations

    def _generate_comparison_baselines(self,
                                     market_data: pd.DataFrame,
                                     labeled_data: pd.DataFrame):
        """Generate comparison baselines for analysis."""
        self.logger.info('📊 Generating comparison baselines')

        # Random baseline
        if self.config.compare_to_random:
            np.random.seed(self.config.random_seed)
            random_labels = np.random.random(len(labeled_data))
            self.comparison_baselines['random'] = {
                'hit_rate': (random_labels > 0.5).mean(),
                'predictive_power': 0.0,  # Random should have no predictive power
                'stability': 0.5  # Random is moderately stable
            }

        # Simple threshold baseline
        if self.config.compare_to_simple_threshold:
            if 'close' in market_data.columns:
                returns = market_data['close'].pct_change()
                simple_threshold = (returns > 0.005).astype(float)  # 0.5% threshold
                self.comparison_baselines['simple_threshold'] = {
                    'hit_rate': simple_threshold.mean(),
                    'predictive_power': self._calculate_predictive_power(
                        simple_threshold, labeled_data, 'simple'
                    ),
                    'stability': self._calculate_label_stability(simple_threshold)
                }

    def generate_analysis_report(self) -> str:
        """Generate comprehensive analysis report."""
        if not self.analysis_results:
            return "No analysis results available. Run analyze_labeling_heuristics() first."

        report_lines = [
            "# Multi-Horizon Profit Labeling Heuristic Analysis Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Executive Summary",
            f"Analyzed {len(self.analysis_results)} heuristic components",
            ""
        ]

        # Group results by analysis type
        by_type = {}
        for key, result in self.analysis_results.items():
            analysis_type = result.analysis_type.value
            if analysis_type not in by_type:
                by_type[analysis_type] = []
            by_type[analysis_type].append((key, result))

        # Generate sections for each analysis type
        for analysis_type, results in by_type.items():
            report_lines.extend([
                f"## {analysis_type.replace('_', ' ').title()} Analysis",
                ""
            ])

            for key, result in results:
                report_lines.extend([
                    f"### {key}",
                    f"**Metric Value**: {result.metric_value:.4f}",
                    f"**Interpretation**: {result.interpretation}",
                    ""
                ])

                if result.recommendations:
                    report_lines.append("**Recommendations**:")
                    for rec in result.recommendations:
                        report_lines.append(f"- {rec}")
                    report_lines.append("")

        # Add baseline comparisons if available
        if self.comparison_baselines:
            report_lines.extend([
                "## Baseline Comparisons",
                ""
            ])
            for baseline_name, metrics in self.comparison_baselines.items():
                report_lines.extend([
                    f"### {baseline_name.replace('_', ' ').title()} Baseline",
                    f"- Hit Rate: {metrics.get('hit_rate', 0):.2%}",
                    f"- Predictive Power: {metrics.get('predictive_power', 0):.3f}",
                    f"- Stability: {metrics.get('stability', 0):.2%}",
                    ""
                ])

        return "\n".join(report_lines)

    def save_results(self, output_path: Union[str, Path]):
        """Save analysis results to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert results to serializable format
        serializable_results = {}
        for key, result in self.analysis_results.items():
            serializable_results[key] = {
                'analysis_type': result.analysis_type.value,
                'metric_value': result.metric_value,
                'confidence_interval': result.confidence_interval,
                'statistical_significance': result.statistical_significance,
                'interpretation': result.interpretation,
                'recommendations': result.recommendations,
                'metadata': result.metadata,
                'timestamp': result.timestamp.isoformat()
            }

        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump({
                'analysis_results': serializable_results,
                'comparison_baselines': self.comparison_baselines,
                'config': {
                    'min_samples_per_analysis': self.config.min_samples_per_analysis,
                    'bootstrap_samples': self.config.bootstrap_samples,
                    'confidence_level': self.config.confidence_level
                }
            }, f, indent=2)

        self.logger.info(f'💾 Analysis results saved to {output_path}')

# Convenience functions
def analyze_profit_labeling_heuristics(market_data: pd.DataFrame,
                                     labeling_config: Optional[MultiHorizonConfig] = None,
                                     analysis_config: Optional[HeuristicAnalysisConfig] = None) -> Dict[str, HeuristicAnalysisResult]:
    """Convenience function to analyze profit labeling heuristics."""
    analyzer = HeuristicAnalyzer(analysis_config)
    return analyzer.analyze_labeling_heuristics(market_data, labeling_config)

def generate_heuristic_analysis_report(market_data: pd.DataFrame,
                                     labeling_config: Optional[MultiHorizonConfig] = None,
                                     analysis_config: Optional[HeuristicAnalysisConfig] = None) -> str:
    """Convenience function to generate analysis report."""
    analyzer = HeuristicAnalyzer(analysis_config)
    analyzer.analyze_labeling_heuristics(market_data, labeling_config)
    return analyzer.generate_analysis_report()
