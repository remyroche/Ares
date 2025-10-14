"""
Metric Orthogonalization Module.

This module reduces redundancy between economic metrics by:
1. Merging conceptually similar metrics
2. Orthogonalizing overlapping measures
3. Creating composite metrics that capture unique aspects
4. Preventing double-counting of similar effects

Metric Consolidation:
- Momentum Intensity + Trend Acceleration → "Momentum Dynamics"
- Instability + Transition Trigger → "Risk Regime Pressure"
- Duration + Persistence → "Regime Stability"
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging

from src.utils.logger import system_logger


class OrthogonalMetric(Enum):
    """Orthogonalized economic metrics (reduced redundancy)."""
    
    # Core Price Action Metrics (orthogonalized)
    MOMENTUM_DYNAMICS = "momentum_dynamics"  # Combines intensity + acceleration
    REVERSAL_CHARACTERISTICS = "reversal_characteristics"  # Violence + asymmetric response
    RISK_REGIME_PRESSURE = "risk_regime_pressure"  # Instability + transition triggers + tail dependence
    REGIME_STABILITY = "regime_stability"  # Duration + persistence
    
    # Fundamental Economic Metrics
    RETURN_SEPARABILITY = "return_separability"
    VOLATILITY_SEPARABILITY = "volatility_separability"
    VOLUME_PROFILE_DIFFERENCE = "volume_profile_difference"
    SHARPE_RATIO_DIFFERENCE = "sharpe_ratio_difference"


@dataclass
class OrthogonalMetricResult:
    """Result for orthogonalized metric."""
    metric: OrthogonalMetric
    composite_score: float
    component_scores: Dict[str, float]
    economic_significance: bool
    trading_implications: str
    regime_specific_values: Dict[int, float]
    statistical_tests: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'metric': self.metric.value,
            'composite_score': self.composite_score,
            'component_scores': self.component_scores,
            'economic_significance': self.economic_significance,
            'trading_implications': self.trading_implications,
            'regime_specific_values': self.regime_specific_values,
            'statistical_tests': self.statistical_tests
        }


class MetricOrthogonalizer:
    """
    Orthogonalizes economic metrics to reduce redundancy and double-counting.
    
    Creates composite metrics that capture unique aspects of regime behavior
    while eliminating overlap between similar measures.
    """
    
    def __init__(self):
        self.logger = system_logger.getChild('MetricOrthogonalizer')
    
    def orthogonalize_metrics(self, 
                            raw_economic_results: Dict[str, Any]) -> Dict[OrthogonalMetric, OrthogonalMetricResult]:
        """
        Orthogonalize raw economic metrics into non-redundant composite measures.
        
        Args:
            raw_economic_results: Raw economic validation results
            
        Returns:
            Dictionary of orthogonalized metrics
        """
        
        self.logger.info("🔧 Orthogonalizing economic metrics to reduce redundancy")
        
        orthogonal_results = {}
        
        # 1. Momentum Dynamics (combines intensity + acceleration)
        orthogonal_results[OrthogonalMetric.MOMENTUM_DYNAMICS] = self._create_momentum_dynamics_metric(raw_economic_results)
        
        # 2. Reversal Characteristics (combines violence + asymmetric response)
        orthogonal_results[OrthogonalMetric.REVERSAL_CHARACTERISTICS] = self._create_reversal_characteristics_metric(raw_economic_results)
        
        # 3. Risk Regime Pressure (combines instability + transitions + tail dependence)
        orthogonal_results[OrthogonalMetric.RISK_REGIME_PRESSURE] = self._create_risk_regime_pressure_metric(raw_economic_results)
        
        # 4. Regime Stability (combines duration + persistence)
        orthogonal_results[OrthogonalMetric.REGIME_STABILITY] = self._create_regime_stability_metric(raw_economic_results)
        
        # 5. Keep fundamental metrics as-is (already orthogonal)
        orthogonal_results.update(self._preserve_fundamental_metrics(raw_economic_results))
        
        self.logger.info(f"✅ Reduced {len(raw_economic_results)} raw metrics to {len(orthogonal_results)} orthogonal metrics")
        
        return orthogonal_results
    
    def _create_momentum_dynamics_metric(self, raw_results: Dict[str, Any]) -> OrthogonalMetricResult:
        """Create orthogonalized momentum dynamics metric."""
        
        # Extract component metrics
        momentum_intensity = self._extract_metric_value(raw_results, 'momentum_intensity_effect')
        trend_acceleration = self._extract_metric_value(raw_results, 'trend_acceleration_impact')
        
        # Orthogonalize using PCA-like approach
        if momentum_intensity is not None and trend_acceleration is not None:
            # Check correlation between components
            component_correlation = self._estimate_component_correlation(momentum_intensity, trend_acceleration)
            
            # If highly correlated (>0.7), create weighted combination
            if abs(component_correlation) > 0.7:
                # Weight by unique variance contribution
                momentum_weight = 0.7  # Momentum typically more important
                acceleration_weight = 0.3
            else:
                # If orthogonal, equal weights
                momentum_weight = 0.5
                acceleration_weight = 0.5
            
            # Create composite score
            composite_score = (
                momentum_intensity['value'] * momentum_weight +
                trend_acceleration['value'] * acceleration_weight
            )
            
            # Combine regime-specific values
            regime_values = {}
            if momentum_intensity['regime_values'] and trend_acceleration['regime_values']:
                for regime in momentum_intensity['regime_values'].keys():
                    if regime in trend_acceleration['regime_values']:
                        regime_values[regime] = (
                            momentum_intensity['regime_values'][regime] * momentum_weight +
                            trend_acceleration['regime_values'][regime] * acceleration_weight
                        )
            
            # Economic significance
            economically_significant = composite_score > 0.01  # Combined threshold
            
            trading_implications = self._generate_momentum_dynamics_implications(
                momentum_intensity, trend_acceleration, composite_score
            )
            
        else:
            composite_score = 0.0
            regime_values = {}
            economically_significant = False
            trading_implications = "Insufficient data for momentum dynamics analysis"
        
        return OrthogonalMetricResult(
            metric=OrthogonalMetric.MOMENTUM_DYNAMICS,
            composite_score=composite_score,
            component_scores={
                'momentum_intensity': momentum_intensity['value'] if momentum_intensity else 0.0,
                'trend_acceleration': trend_acceleration['value'] if trend_acceleration else 0.0
            },
            economic_significance=economically_significant,
            trading_implications=trading_implications,
            regime_specific_values=regime_values,
            statistical_tests={'component_correlation': component_correlation if 'component_correlation' in locals() else 0.0}
        )
    
    def _create_reversal_characteristics_metric(self, raw_results: Dict[str, Any]) -> OrthogonalMetricResult:
        """Create orthogonalized reversal characteristics metric."""
        
        # Extract component metrics
        reversal_violence = self._extract_metric_value(raw_results, 'reversal_violence_modulation')
        asymmetric_response = self._extract_metric_value(raw_results, 'asymmetric_volatility_response')
        
        if reversal_violence is not None and asymmetric_response is not None:
            # These metrics are conceptually orthogonal (speed vs asymmetry)
            # Use equal weights
            composite_score = (reversal_violence['value'] + asymmetric_response['value']) / 2
            
            # Combine regime-specific values
            regime_values = {}
            if reversal_violence['regime_values'] and asymmetric_response['regime_values']:
                for regime in reversal_violence['regime_values'].keys():
                    if regime in asymmetric_response['regime_values']:
                        regime_values[regime] = (
                            reversal_violence['regime_values'][regime] +
                            asymmetric_response['regime_values'][regime]
                        ) / 2
            
            economically_significant = composite_score > 0.05  # Combined threshold
            
            trading_implications = f"Reversal characteristics show {'significant' if economically_significant else 'limited'} regime differences for tail risk management"
            
        else:
            composite_score = 0.0
            regime_values = {}
            economically_significant = False
            trading_implications = "Insufficient data for reversal characteristics analysis"
        
        return OrthogonalMetricResult(
            metric=OrthogonalMetric.REVERSAL_CHARACTERISTICS,
            composite_score=composite_score,
            component_scores={
                'reversal_violence': reversal_violence['value'] if reversal_violence else 0.0,
                'asymmetric_response': asymmetric_response['value'] if asymmetric_response else 0.0
            },
            economic_significance=economically_significant,
            trading_implications=trading_implications,
            regime_specific_values=regime_values,
            statistical_tests={}
        )
    
    def _create_risk_regime_pressure_metric(self, raw_results: Dict[str, Any]) -> OrthogonalMetricResult:
        """Create orthogonalized risk regime pressure metric."""
        
        # Extract component metrics
        instability = self._extract_metric_value(raw_results, 'price_instability_influence')
        transition_trigger = self._extract_metric_value(raw_results, 'price_regime_transition_trigger')
        tail_dependence = self._extract_metric_value(raw_results, 'tail_dependence_intensity')
        
        components = [instability, transition_trigger, tail_dependence]
        valid_components = [comp for comp in components if comp is not None]
        
        if valid_components:
            # Weight components by their theoretical importance for risk
            weights = {
                'instability': 0.4,      # Most important for risk
                'transition_trigger': 0.3,  # Important for regime changes
                'tail_dependence': 0.3   # Important for tail risk
            }
            
            composite_score = 0.0
            total_weight = 0.0
            
            component_scores = {}
            
            if instability:
                composite_score += instability['value'] * weights['instability']
                total_weight += weights['instability']
                component_scores['instability'] = instability['value']
            
            if transition_trigger:
                composite_score += transition_trigger['value'] * weights['transition_trigger']
                total_weight += weights['transition_trigger']
                component_scores['transition_trigger'] = transition_trigger['value']
            
            if tail_dependence:
                composite_score += tail_dependence['value'] * weights['tail_dependence']
                total_weight += weights['tail_dependence']
                component_scores['tail_dependence'] = tail_dependence['value']
            
            # Normalize by total weight
            if total_weight > 0:
                composite_score /= total_weight
            
            economically_significant = composite_score > 0.1  # Risk pressure threshold
            
            trading_implications = f"Risk regime pressure shows {'high' if economically_significant else 'low'} regime-specific risk characteristics"
            
        else:
            composite_score = 0.0
            component_scores = {}
            economically_significant = False
            trading_implications = "Insufficient data for risk regime pressure analysis"
        
        return OrthogonalMetricResult(
            metric=OrthogonalMetric.RISK_REGIME_PRESSURE,
            composite_score=composite_score,
            component_scores=component_scores,
            economic_significance=economically_significant,
            trading_implications=trading_implications,
            regime_specific_values={},  # Would need to combine regime values
            statistical_tests={}
        )
    
    def _create_regime_stability_metric(self, raw_results: Dict[str, Any]) -> OrthogonalMetricResult:
        """Create orthogonalized regime stability metric."""
        
        # Extract component metrics
        duration_impact = self._extract_metric_value(raw_results, 'trend_duration_impact')
        persistence_score = self._extract_metric_value(raw_results, 'regime_persistence_score')
        
        if duration_impact is not None and persistence_score is not None:
            # These are related but measure different aspects
            # Duration = how long trends last, Persistence = how long regimes last
            
            # Normalize scores to similar scales
            duration_normalized = duration_impact['value'] / 20.0  # Normalize by typical max duration
            persistence_normalized = persistence_score['value'] / 50.0  # Normalize by typical max persistence
            
            # Create composite stability score
            composite_score = (duration_normalized * 0.6 + persistence_normalized * 0.4)
            
            economically_significant = composite_score > 0.2  # Stability threshold
            
            trading_implications = f"Regime stability shows {'high' if economically_significant else 'low'} predictability for strategy commitment"
            
        else:
            composite_score = 0.0
            economically_significant = False
            trading_implications = "Insufficient data for regime stability analysis"
        
        return OrthogonalMetricResult(
            metric=OrthogonalMetric.REGIME_STABILITY,
            composite_score=composite_score,
            component_scores={
                'duration_impact': duration_impact['value'] if duration_impact else 0.0,
                'persistence_score': persistence_score['value'] if persistence_score else 0.0
            },
            economic_significance=economically_significant,
            trading_implications=trading_implications,
            regime_specific_values={},
            statistical_tests={}
        )
    
    def _extract_metric_value(self, raw_results: Dict[str, Any], metric_name: str) -> Optional[Dict[str, Any]]:
        """Extract metric value and regime-specific data from raw results."""
        
        if metric_name in raw_results:
            metric_data = raw_results[metric_name]
            return {
                'value': metric_data.get('value', 0.0),
                'regime_values': metric_data.get('regime_specific_values', {}),
                'economic_significance': metric_data.get('economic_significance', False)
            }
        return None
    
    def _estimate_component_correlation(self, metric1: Dict[str, Any], metric2: Dict[str, Any]) -> float:
        """Estimate correlation between two metric components."""
        
        # If we have regime-specific values, calculate correlation
        if metric1['regime_values'] and metric2['regime_values']:
            common_regimes = set(metric1['regime_values'].keys()).intersection(
                set(metric2['regime_values'].keys())
            )
            
            if len(common_regimes) > 2:
                values1 = [metric1['regime_values'][regime] for regime in common_regimes]
                values2 = [metric2['regime_values'][regime] for regime in common_regimes]
                
                correlation = np.corrcoef(values1, values2)[0, 1]
                return correlation if not np.isnan(correlation) else 0.0
        
        return 0.0  # Assume orthogonal if can't calculate
    
    def _generate_momentum_dynamics_implications(self,
                                               momentum_intensity: Dict[str, Any],
                                               trend_acceleration: Dict[str, Any],
                                               composite_score: float) -> str:
        """Generate trading implications for momentum dynamics."""
        
        if composite_score > 0.02:
            implications = "Strong momentum dynamics across regimes enable:"
            implications += "\n- Regime-specific momentum strategy calibration"
            implications += "\n- Dynamic trend following parameters"
            implications += "\n- Momentum-based position sizing"
            
            # Specific regime recommendations
            if momentum_intensity['regime_values']:
                best_momentum_regime = max(momentum_intensity['regime_values'], key=momentum_intensity['regime_values'].get)
                implications += f"\n- Focus momentum strategies on Regime {best_momentum_regime}"
        else:
            implications = "Limited momentum dynamics differences - consider single momentum approach"
        
        return implications
    
    def calculate_metric_independence(self, 
                                    orthogonal_results: Dict[OrthogonalMetric, OrthogonalMetricResult]) -> Dict[str, float]:
        """Calculate independence between orthogonalized metrics."""
        
        independence_matrix = {}
        
        metrics = list(orthogonal_results.keys())
        
        for i, metric1 in enumerate(metrics):
            for j, metric2 in enumerate(metrics):
                if i < j:  # Only upper triangle
                    # Calculate correlation between composite scores
                    result1 = orthogonal_results[metric1]
                    result2 = orthogonal_results[metric2]
                    
                    if result1.regime_specific_values and result2.regime_specific_values:
                        common_regimes = set(result1.regime_specific_values.keys()).intersection(
                            set(result2.regime_specific_values.keys())
                        )
                        
                        if len(common_regimes) > 2:
                            values1 = [result1.regime_specific_values[regime] for regime in common_regimes]
                            values2 = [result2.regime_specific_values[regime] for regime in common_regimes]
                            
                            correlation = np.corrcoef(values1, values2)[0, 1]
                            independence = 1 - abs(correlation) if not np.isnan(correlation) else 1.0
                            
                            independence_matrix[f"{metric1.value}_vs_{metric2.value}"] = independence
        
        # Calculate average independence
        if independence_matrix:
            avg_independence = np.mean(list(independence_matrix.values()))
            independence_matrix['average_independence'] = avg_independence
            
            self.logger.info(f"📊 Metric independence: {avg_independence:.3f} (1.0 = perfectly independent)")
        
        return independence_matrix
    
    def generate_orthogonalization_report(self, 
                                        orthogonal_results: Dict[OrthogonalMetric, OrthogonalMetricResult],
                                        independence_matrix: Dict[str, float]) -> str:
        """Generate report on metric orthogonalization."""
        
        report = []
        report.append("# Metric Orthogonalization Report")
        report.append("## Reducing Redundancy in Economic Validation")
        report.append("")
        
        # Orthogonalization summary
        report.append("## Metric Consolidation")
        report.append("")
        report.append("**Original → Orthogonalized:**")
        report.append("- Momentum Intensity + Trend Acceleration → **Momentum Dynamics**")
        report.append("- Reversal Violence + Asymmetric Response → **Reversal Characteristics**")
        report.append("- Instability + Transitions + Tail Dependence → **Risk Regime Pressure**")
        report.append("- Duration + Persistence → **Regime Stability**")
        report.append("")
        
        # Independence analysis
        avg_independence = independence_matrix.get('average_independence', 0.0)
        report.append("## Metric Independence Analysis")
        report.append("")
        report.append(f"**Average Independence Score**: {avg_independence:.3f}")
        
        if avg_independence > 0.8:
            report.append("✅ **High Independence** - Metrics capture unique aspects")
        elif avg_independence > 0.6:
            report.append("⚠️ **Moderate Independence** - Some overlap remains")
        else:
            report.append("❌ **Low Independence** - Significant redundancy detected")
        
        report.append("")
        
        # Detailed metric analysis
        report.append("## Orthogonalized Metric Results")
        report.append("")
        
        for metric, result in orthogonal_results.items():
            status = "✅" if result.economic_significance else "❌"
            report.append(f"{status} **{metric.value.upper()}**")
            report.append(f"- **Composite Score**: {result.composite_score:.4f}")
            report.append(f"- **Economic Significance**: {'Yes' if result.economic_significance else 'No'}")
            report.append(f"- **Trading Implications**: {result.trading_implications}")
            
            # Component breakdown
            if result.component_scores:
                report.append("- **Component Breakdown**:")
                for component, score in result.component_scores.items():
                    report.append(f"  - {component.replace('_', ' ').title()}: {score:.4f}")
            
            report.append("")
        
        return "\n".join(report)
    
    def _preserve_fundamental_metrics(self, raw_results: Dict[str, Any]) -> Dict[OrthogonalMetric, OrthogonalMetricResult]:
        """Preserve fundamental economic metrics that are already orthogonal."""
        
        preserved_metrics = {}
        
        # Map fundamental metrics to orthogonal metrics
        fundamental_mapping = {
            'return_separability': OrthogonalMetric.RETURN_SEPARABILITY,
            'volatility_separability': OrthogonalMetric.VOLATILITY_SEPARABILITY,
            'volume_profile_difference': OrthogonalMetric.VOLUME_PROFILE_DIFFERENCE,
            'sharpe_ratio_difference': OrthogonalMetric.SHARPE_RATIO_DIFFERENCE
        }
        
        for raw_metric_name, orthogonal_metric in fundamental_mapping.items():
            metric_data = self._extract_metric_value(raw_results, raw_metric_name)
            
            if metric_data:
                preserved_metrics[orthogonal_metric] = OrthogonalMetricResult(
                    metric=orthogonal_metric,
                    composite_score=metric_data['value'],
                    component_scores={raw_metric_name: metric_data['value']},
                    economic_significance=metric_data['economic_significance'],
                    trading_implications=f"{raw_metric_name.replace('_', ' ').title()} shows regime differences",
                    regime_specific_values=metric_data['regime_values'],
                    statistical_tests={}
                )
        
        return preserved_metrics
    
    def calculate_orthogonalization_quality(self, 
                                          raw_results: Dict[str, Any],
                                          orthogonal_results: Dict[OrthogonalMetric, OrthogonalMetricResult]) -> Dict[str, float]:
        """Calculate quality of orthogonalization process."""
        
        quality_metrics = {}
        
        # 1. Compression ratio
        n_raw_metrics = len(raw_results)
        n_orthogonal_metrics = len(orthogonal_results)
        compression_ratio = n_orthogonal_metrics / n_raw_metrics if n_raw_metrics > 0 else 1.0
        
        quality_metrics['compression_ratio'] = compression_ratio
        quality_metrics['metrics_reduced'] = n_raw_metrics - n_orthogonal_metrics
        
        # 2. Information preservation
        # Calculate how much information is preserved in orthogonalization
        raw_economic_significance_rate = sum(
            1 for metric_data in raw_results.values() 
            if isinstance(metric_data, dict) and metric_data.get('economic_significance', False)
        ) / n_raw_metrics if n_raw_metrics > 0 else 0
        
        orthogonal_significance_rate = sum(
            1 for result in orthogonal_results.values() 
            if result.economic_significance
        ) / n_orthogonal_metrics if n_orthogonal_metrics > 0 else 0
        
        information_preservation = orthogonal_significance_rate / raw_economic_significance_rate if raw_economic_significance_rate > 0 else 1.0
        quality_metrics['information_preservation'] = information_preservation
        
        # 3. Independence calculation
        independence_matrix = self.calculate_metric_independence(orthogonal_results)
        quality_metrics['average_independence'] = independence_matrix.get('average_independence', 1.0)
        
        # 4. Overall orthogonalization quality
        overall_quality = (
            (1 - compression_ratio) * 0.3 +  # Reward compression
            information_preservation * 0.4 +  # Reward information preservation
            quality_metrics['average_independence'] * 0.3  # Reward independence
        )
        
        quality_metrics['overall_orthogonalization_quality'] = overall_quality
        
        return quality_metrics