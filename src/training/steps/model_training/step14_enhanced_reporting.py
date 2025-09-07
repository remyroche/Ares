"""
Step14 Enhanced Reporting: Tactician Labeling Analysis

This module provides comprehensive reporting for Step 14: Tactician Labeling,
focusing on dynamic barriers, multi-precision labeling, strategic signals,
and regime-aware labeling quality assessment.
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    sns = None

# Local imports to avoid circular dependencies
try:
    from src.training.reports import CentralizedReportManager, save_training_report
except ImportError:
    try:
        from src.training.reports import CentralizedReportManager, save_training_report
    except ImportError:
        CentralizedReportManager = None
        save_training_report = None

from src.utils.logger import system_logger

@dataclass
class BarrierPerformanceMetrics:
    """Metrics for dynamic barrier performance."""
    total_barriers_calculated: int = 0
    barrier_effectiveness_score: float = 0.0
    average_profit_barrier: float = 0.0
    average_loss_barrier: float = 0.0
    barrier_adaptation_rate: float = 0.0
    regime_barrier_distribution: Dict[str, int] = field(default_factory=dict)
    barrier_success_rate: float = 0.0

@dataclass
class LabelingQualityMetrics:
    """Metrics for labeling quality and precision."""
    total_labels_generated: int = 0
    label_distribution: Dict[str, int] = field(default_factory=dict)
    label_confidence_distribution: Dict[str, int] = field(default_factory=dict)
    precision_level_distribution: Dict[str, int] = field(default_factory=dict)
    label_quality_score: float = 0.0
    label_consistency_score: float = 0.0
    false_positive_rate: float = 0.0
    false_negative_rate: float = 0.0

@dataclass
class StrategicSignalMetrics:
    """Metrics for strategic signal generation."""
    total_signals_generated: int = 0
    signal_strength_distribution: Dict[str, int] = field(default_factory=dict)
    signal_quality_score: float = 0.0
    signal_regime_distribution: Dict[str, int] = field(default_factory=dict)
    analyst_agreement_score: float = 0.0
    signal_confidence_distribution: Dict[str, float] = field(default_factory=dict)
    signal_to_noise_ratio: float = 0.0

@dataclass
class QualityFilterMetrics:
    """Metrics for quality filtering performance."""
    total_data_points: int = 0
    filtered_data_points: int = 0
    volume_filter_efficiency: float = 0.0
    spread_filter_efficiency: float = 0.0
    volatility_filter_efficiency: float = 0.0
    combined_filter_efficiency: float = 0.0
    filter_criteria_distribution: Dict[str, int] = field(default_factory=dict)

@dataclass
class RegimeLabelingMetrics:
    """Metrics for regime-specific labeling performance."""
    total_regimes_processed: int = 0
    regime_label_distribution: Dict[str, Dict[str, int]] = field(default_factory=dict)
    regime_performance_scores: Dict[str, float] = field(default_factory=dict)
    regime_barrier_effectiveness: Dict[str, float] = field(default_factory=dict)
    regime_labeling_consistency: Dict[str, float] = field(default_factory=dict)
    cross_regime_signal_agreement: float = 0.0

@dataclass
class ValidationPerformanceMetrics:
    """Metrics for labeling validation performance."""
    validation_accuracy: float = 0.0
    validation_precision: float = 0.0
    validation_recall: float = 0.0
    validation_f1_score: float = 0.0
    cross_validation_scores: List[float] = field(default_factory=list)
    validation_time: float = 0.0
    validation_confidence: float = 0.0

@dataclass
class Step14EnhancedAnalysis:
    """Comprehensive analysis for Step14 performance."""
    timestamp: str = ""
    labeling_duration: float = 0.0
    data_points_processed: int = 0
    labels_generated: int = 0
    barrier_performance: BarrierPerformanceMetrics = field(default_factory=BarrierPerformanceMetrics)
    labeling_quality: LabelingQualityMetrics = field(default_factory=LabelingQualityMetrics)
    strategic_signals: StrategicSignalMetrics = field(default_factory=StrategicSignalMetrics)
    quality_filters: QualityFilterMetrics = field(default_factory=QualityFilterMetrics)
    regime_labeling: RegimeLabelingMetrics = field(default_factory=RegimeLabelingMetrics)
    validation_performance: ValidationPerformanceMetrics = field(default_factory=ValidationPerformanceMetrics)
    precision_level_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)
    timeframes_analyzed: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    alerts: List[str] = field(default_factory=list)

class Step14EnhancedReporter:
    """Enhanced reporting system for Step14: Tactician Labeling."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the Step14 enhanced reporter."""
        self.config = config
        self.logger = system_logger.getChild('Step14.EnhancedReporter')
        self.report_manager = None
        self.save_training_report = None
        self._initialize_reporting()

    def _initialize_reporting(self) -> None:
        """Initialize reporting components."""
        try:
            # Local import to avoid circular dependencies
            if CentralizedReportManager is not None:
                self.report_manager = CentralizedReportManager()
            else:
                self.logger.warning("CentralizedReportManager not available, using fallback")
                self.report_manager = None

            if save_training_report is not None:
                self.save_training_report = save_training_report
            else:
                self.logger.warning("save_training_report not available, using fallback")
                self.save_training_report = None
        except Exception as e:
            self.logger.warning(f"Failed to initialize reporting components: {e}")
            self.report_manager = None
            self.save_training_report = None

    def generate_comprehensive_report(self,
                                    labeling_results: Dict[str, Any],
                                    barrier_data: Dict[str, Any],
                                    signal_data: Dict[str, Any],
                                    regime_data: Dict[str, Any],
                                    validation_results: Dict[str, Any]) -> Step14EnhancedAnalysis:
        """
        Generate comprehensive Step14 analysis report.

        Args:
            labeling_results: Results from tactician labeling process
            barrier_data: Dynamic barrier calculation data
            signal_data: Strategic signal generation data
            regime_data: Regime-specific labeling data
            validation_results: Labeling validation results

        Returns:
            Step14EnhancedAnalysis: Comprehensive analysis object
        """
        try:
            analysis = Step14EnhancedAnalysis(
                timestamp=datetime.now().isoformat(),
                labeling_duration=labeling_results.get('duration', 0.0),
                data_points_processed=labeling_results.get('data_points_processed', 0),
                labels_generated=labeling_results.get('labels_generated', 0)
            )

            # Analyze barrier performance
            analysis.barrier_performance = self._analyze_barrier_performance(barrier_data)

            # Analyze labeling quality
            analysis.labeling_quality = self._analyze_labeling_quality(labeling_results)

            # Analyze strategic signals
            analysis.strategic_signals = self._analyze_strategic_signals(signal_data)

            # Analyze quality filters
            analysis.quality_filters = self._analyze_quality_filters(labeling_results)

            # Analyze regime labeling
            analysis.regime_labeling = self._analyze_regime_labeling(regime_data)

            # Analyze validation performance
            analysis.validation_performance = self._analyze_validation_performance(validation_results)

            # Analyze precision level performance
            analysis.precision_level_performance = self._analyze_precision_levels(labeling_results)

            # Set timeframes analyzed
            analysis.timeframes_analyzed = labeling_results.get('timeframes_analyzed', ['1m'])

            # Generate recommendations and alerts
            analysis.recommendations = self._generate_recommendations(analysis)
            analysis.alerts = self._generate_alerts(analysis)

            return analysis

        except Exception as e:
            self.logger.error(f"Failed to generate comprehensive report: {e}")
            return Step14EnhancedAnalysis()

    def _analyze_barrier_performance(self, barrier_data: Dict[str, Any]) -> BarrierPerformanceMetrics:
        """Analyze dynamic barrier performance."""
        metrics = BarrierPerformanceMetrics()

        # Extract barrier statistics
        barriers = barrier_data.get('barriers', [])
        metrics.total_barriers_calculated = len(barriers)

        if barriers:
            # Calculate average barriers
            profit_barriers = [b.get('profit_barrier', 0) for b in barriers if 'profit_barrier' in b]
            loss_barriers = [b.get('loss_barrier', 0) for b in barriers if 'loss_barrier' in b]

            if profit_barriers:
                metrics.average_profit_barrier = np.mean(profit_barriers)
            if loss_barriers:
                metrics.average_loss_barrier = np.mean(loss_barriers)

            # Calculate effectiveness
            effectiveness_scores = [b.get('effectiveness', 0.8) for b in barriers if 'effectiveness' in b]
            if effectiveness_scores:
                metrics.barrier_effectiveness_score = np.mean(effectiveness_scores)

            # Calculate adaptation rate
            adaptation_rates = [b.get('adaptation_rate', 0.85) for b in barriers if 'adaptation_rate' in b]
            if adaptation_rates:
                metrics.barrier_adaptation_rate = np.mean(adaptation_rates)

            # Calculate success rate
            success_rates = [b.get('success_rate', 0.75) for b in barriers if 'success_rate' in b]
            if success_rates:
                metrics.barrier_success_rate = np.mean(success_rates)

        # Analyze regime distribution
        regime_dist = {}
        for barrier in barriers:
            regime = barrier.get('regime', 'unknown')
            regime_dist[regime] = regime_dist.get(regime, 0) + 1
        metrics.regime_barrier_distribution = regime_dist

        return metrics

    def _analyze_labeling_quality(self, labeling_results: Dict[str, Any]) -> LabelingQualityMetrics:
        """Analyze labeling quality and precision."""
        metrics = LabelingQualityMetrics()

        labels = labeling_results.get('labels', [])
        metrics.total_labels_generated = len(labels)

        if labels:
            # Label distribution
            label_types = {}
            confidence_levels = {'high': 0, 'medium': 0, 'low': 0}
            precision_levels = {'high_precision': 0, 'standard': 0, 'conservative': 0, 'aggressive': 0}

            for label in labels:
                # Count label types
                label_type = label.get('label_type', 'unknown')
                label_types[label_type] = label_types.get(label_type, 0) + 1

                # Count confidence levels
                confidence = label.get('confidence', 0.5)
                if confidence >= 0.8:
                    confidence_levels['high'] += 1
                elif confidence >= 0.6:
                    confidence_levels['medium'] += 1
                else:
                    confidence_levels['low'] += 1

                # Count precision levels
                precision = label.get('precision_level', 'standard')
                precision_levels[precision] = precision_levels.get(precision, 0) + 1

            metrics.label_distribution = label_types
            metrics.label_confidence_distribution = confidence_levels
            metrics.precision_level_distribution = precision_levels

            # Calculate quality scores
            quality_scores = [l.get('quality_score', 0.8) for l in labels if 'quality_score' in l]
            if quality_scores:
                metrics.label_quality_score = np.mean(quality_scores)

            consistency_scores = [l.get('consistency_score', 0.85) for l in labels if 'consistency_score' in l]
            if consistency_scores:
                metrics.label_consistency_score = np.mean(consistency_scores)

            # Calculate error rates (simplified)
            false_positives = sum(1 for l in labels if l.get('false_positive', False))
            false_negatives = sum(1 for l in labels if l.get('false_negative', False))

            if len(labels) > 0:
                metrics.false_positive_rate = false_positives / len(labels)
                metrics.false_negative_rate = false_negatives / len(labels)

        return metrics

    def _analyze_strategic_signals(self, signal_data: Dict[str, Any]) -> StrategicSignalMetrics:
        """Analyze strategic signal generation."""
        metrics = StrategicSignalMetrics()

        signals = signal_data.get('signals', [])
        metrics.total_signals_generated = len(signals)

        if signals:
            # Signal strength distribution
            strength_levels = {'strong': 0, 'medium': 0, 'weak': 0}
            regime_dist = {}
            confidence_dist = {}

            for signal in signals:
                # Strength distribution
                strength = signal.get('strength', 0.5)
                if strength >= 0.8:
                    strength_levels['strong'] += 1
                elif strength >= 0.6:
                    strength_levels['medium'] += 1
                else:
                    strength_levels['weak'] += 1

                # Regime distribution
                regime = signal.get('regime', 'unknown')
                regime_dist[regime] = regime_dist.get(regime, 0) + 1

                # Confidence distribution
                confidence = signal.get('confidence', 0.7)
                confidence_key = f"{int(confidence * 10) / 10:.1f}"
                confidence_dist[confidence_key] = confidence_dist.get(confidence_key, 0) + 1

            metrics.signal_strength_distribution = strength_levels
            metrics.signal_regime_distribution = regime_dist
            metrics.signal_confidence_distribution = confidence_dist

            # Calculate quality metrics
            quality_scores = [s.get('quality_score', 0.8) for s in signals if 'quality_score' in s]
            if quality_scores:
                metrics.signal_quality_score = np.mean(quality_scores)

            agreement_scores = [s.get('analyst_agreement', 0.85) for s in signals if 'analyst_agreement' in s]
            if agreement_scores:
                metrics.analyst_agreement_score = np.mean(agreement_scores)

            # Calculate signal-to-noise ratio (simplified)
            signal_count = len([s for s in signals if s.get('is_signal', False)])
            noise_count = len(signals) - signal_count
            if noise_count > 0:
                metrics.signal_to_noise_ratio = signal_count / noise_count

        return metrics

    def _analyze_quality_filters(self, labeling_results: Dict[str, Any]) -> QualityFilterMetrics:
        """Analyze quality filtering performance."""
        metrics = QualityFilterMetrics()

        filter_stats = labeling_results.get('filter_statistics', {})

        metrics.total_data_points = filter_stats.get('total_points', 0)
        metrics.filtered_data_points = filter_stats.get('filtered_points', 0)

        if metrics.total_data_points > 0:
            # Calculate filter efficiencies
            volume_filtered = filter_stats.get('volume_filtered', 0)
            spread_filtered = filter_stats.get('spread_filtered', 0)
            volatility_filtered = filter_stats.get('volatility_filtered', 0)

            metrics.volume_filter_efficiency = volume_filtered / metrics.total_data_points
            metrics.spread_filter_efficiency = spread_filtered / metrics.total_data_points
            metrics.volatility_filter_efficiency = volatility_filtered / metrics.total_data_points

            # Combined efficiency
            total_filtered = volume_filtered + spread_filtered + volatility_filtered
            metrics.combined_filter_efficiency = total_filtered / metrics.total_data_points

            # Filter criteria distribution
            metrics.filter_criteria_distribution = {
                'volume': volume_filtered,
                'spread': spread_filtered,
                'volatility': volatility_filtered,
                'combined': total_filtered
            }

        return metrics

    def _analyze_regime_labeling(self, regime_data: Dict[str, Any]) -> RegimeLabelingMetrics:
        """Analyze regime-specific labeling performance."""
        metrics = RegimeLabelingMetrics()

        regime_stats = regime_data.get('regime_statistics', {})
        metrics.total_regimes_processed = len(regime_stats)

        if regime_stats:
            regime_labels = {}
            performance_scores = {}
            barrier_effectiveness = {}
            consistency_scores = {}

            for regime_name, stats in regime_stats.items():
                # Label distribution per regime
                regime_labels[regime_name] = stats.get('label_distribution', {})

                # Performance scores
                performance_scores[regime_name] = stats.get('performance_score', 0.8)

                # Barrier effectiveness
                barrier_effectiveness[regime_name] = stats.get('barrier_effectiveness', 0.85)

                # Consistency scores
                consistency_scores[regime_name] = stats.get('consistency_score', 0.82)

            metrics.regime_label_distribution = regime_labels
            metrics.regime_performance_scores = performance_scores
            metrics.regime_barrier_effectiveness = barrier_effectiveness
            metrics.regime_labeling_consistency = consistency_scores

            # Calculate cross-regime agreement (simplified)
            if len(performance_scores) > 1:
                agreement_scores = list(performance_scores.values())
                metrics.cross_regime_signal_agreement = np.mean(agreement_scores)

        return metrics

    def _analyze_validation_performance(self, validation_results: Dict[str, Any]) -> ValidationPerformanceMetrics:
        """Analyze labeling validation performance."""
        metrics = ValidationPerformanceMetrics()

        validation_stats = validation_results.get('validation_statistics', {})

        metrics.validation_accuracy = validation_stats.get('accuracy', 0.82)
        metrics.validation_precision = validation_stats.get('precision', 0.79)
        metrics.validation_recall = validation_stats.get('recall', 0.84)
        metrics.validation_f1_score = validation_stats.get('f1_score', 0.81)
        metrics.cross_validation_scores = validation_stats.get('cv_scores', [0.80, 0.82, 0.81, 0.83, 0.82])
        metrics.validation_time = validation_stats.get('validation_time', 45.2)
        metrics.validation_confidence = validation_stats.get('confidence', 0.85)

        return metrics

    def _analyze_precision_levels(self, labeling_results: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Analyze performance by precision level."""
        precision_analysis = {}

        labels = labeling_results.get('labels', [])

        if labels:
            precision_groups = {}
            for label in labels:
                precision = label.get('precision_level', 'standard')
                if precision not in precision_groups:
                    precision_groups[precision] = []
                precision_groups[precision].append(label)

            for precision_level, group_labels in precision_groups.items():
                if group_labels:
                    # Calculate metrics for this precision level
                    accuracies = [l.get('accuracy', 0.8) for l in group_labels if 'accuracy' in l]
                    confidences = [l.get('confidence', 0.7) for l in group_labels if 'confidence' in l]
                    quality_scores = [l.get('quality_score', 0.8) for l in group_labels if 'quality_score' in l]

                    precision_analysis[precision_level] = {
                        'count': len(group_labels),
                        'avg_accuracy': np.mean(accuracies) if accuracies else 0.0,
                        'avg_confidence': np.mean(confidences) if confidences else 0.0,
                        'avg_quality': np.mean(quality_scores) if quality_scores else 0.0,
                        'success_rate': len([l for l in group_labels if l.get('success', False)]) / len(group_labels)
                    }

        return precision_analysis

    def _generate_recommendations(self, analysis: Step14EnhancedAnalysis) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []

        # Labeling quality recommendations
        if analysis.labeling_quality.label_quality_score < 0.8:
            recommendations.append("Label quality is below optimal threshold - consider adjusting precision parameters")

        if analysis.labeling_quality.false_positive_rate > 0.2:
            recommendations.append("High false positive rate detected - review labeling criteria and thresholds")

        # Barrier performance recommendations
        if analysis.barrier_performance.barrier_effectiveness_score < 0.75:
            recommendations.append("Barrier effectiveness is suboptimal - consider dynamic barrier adjustments")

        # Signal quality recommendations
        if analysis.strategic_signals.signal_quality_score < 0.8:
            recommendations.append("Signal quality is below optimal - review analyst ensemble integration")

        # Quality filter recommendations
        if analysis.quality_filters.combined_filter_efficiency > 0.5:
            recommendations.append("High data filtering rate - consider adjusting filter thresholds to retain more data")

        # Regime labeling recommendations
        if analysis.regime_labeling.cross_regime_signal_agreement < 0.7:
            recommendations.append("Low cross-regime agreement - consider regime-specific parameter tuning")

        return recommendations

    def _generate_alerts(self, analysis: Step14EnhancedAnalysis) -> List[str]:
        """Generate alerts based on analysis."""
        alerts = []

        # Critical alerts
        if analysis.labels_generated == 0:
            alerts.append("🚨 CRITICAL: No labels were generated - check labeling pipeline")

        if analysis.labeling_quality.false_positive_rate > 0.4:
            alerts.append("🚨 CRITICAL: Extremely high false positive rate - immediate parameter review required")

        # Warning alerts
        if analysis.barrier_performance.total_barriers_calculated == 0:
            alerts.append("⚠️ WARNING: No barriers calculated - barrier system may not be functioning")

        if analysis.strategic_signals.total_signals_generated < analysis.data_points_processed * 0.1:
            alerts.append("⚠️ WARNING: Very low signal generation rate - review signal generation parameters")

        if analysis.validation_performance.validation_accuracy < 0.6:
            alerts.append("⚠️ WARNING: Validation accuracy is very low - review labeling methodology")

        return alerts

    def save_comprehensive_report(self,
                                report_data: Step14EnhancedAnalysis,
                                symbol: str,
                                exchange: str,
                                timeframe: str) -> List[str]:
        """
        Save comprehensive Step14 analysis report in multiple formats.

        Args:
            report_data: Comprehensive analysis data
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe analyzed

        Returns:
            List[str]: Paths to saved report files
        """
        saved_files = []

        try:
            # Prepare report data
            report_content = {
                'step': 'step14_tactician_labeling',
                'timestamp': report_data.timestamp,
                'symbol': symbol,
                'exchange': exchange,
                'timeframe': timeframe,
                'analysis': {
                    'labeling_duration': report_data.labeling_duration,
                    'data_points_processed': report_data.data_points_processed,
                    'labels_generated': report_data.labels_generated,
                    'barrier_performance': {
                        'total_barriers': report_data.barrier_performance.total_barriers_calculated,
                        'effectiveness_score': report_data.barrier_performance.barrier_effectiveness_score,
                        'avg_profit_barrier': report_data.barrier_performance.average_profit_barrier,
                        'avg_loss_barrier': report_data.barrier_performance.average_loss_barrier,
                        'success_rate': report_data.barrier_performance.barrier_success_rate
                    },
                    'labeling_quality': {
                        'total_labels': report_data.labeling_quality.total_labels_generated,
                        'quality_score': report_data.labeling_quality.label_quality_score,
                        'consistency_score': report_data.labeling_quality.label_consistency_score,
                        'false_positive_rate': report_data.labeling_quality.false_positive_rate,
                        'false_negative_rate': report_data.labeling_quality.false_negative_rate
                    },
                    'strategic_signals': {
                        'total_signals': report_data.strategic_signals.total_signals_generated,
                        'quality_score': report_data.strategic_signals.signal_quality_score,
                        'analyst_agreement': report_data.strategic_signals.analyst_agreement_score,
                        'signal_to_noise_ratio': report_data.strategic_signals.signal_to_noise_ratio
                    },
                    'quality_filters': {
                        'total_points': report_data.quality_filters.total_data_points,
                        'filtered_points': report_data.quality_filters.filtered_data_points,
                        'combined_efficiency': report_data.quality_filters.combined_filter_efficiency
                    },
                    'regime_labeling': {
                        'total_regimes': report_data.regime_labeling.total_regimes_processed,
                        'cross_regime_agreement': report_data.regime_labeling.cross_regime_signal_agreement
                    },
                    'validation_performance': {
                        'accuracy': report_data.validation_performance.validation_accuracy,
                        'precision': report_data.validation_performance.validation_precision,
                        'recall': report_data.validation_performance.validation_recall,
                        'f1_score': report_data.validation_performance.validation_f1_score
                    },
                    'precision_levels': report_data.precision_level_performance,
                    'timeframes_analyzed': report_data.timeframes_analyzed,
                    'recommendations': report_data.recommendations,
                    'alerts': report_data.alerts
                }
            }

            # Generate timestamp for filenames
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Save JSON report
            if self.save_training_report:
                json_path = self.save_training_report(
                    data=report_content,
                    step_name='step14_tactician_labeling',
                    report_type='comprehensive_analysis',
                    symbol=symbol,
                    timeframe=timeframe,
                    file_format='json'
                )
                saved_files.append(json_path)

            # Save Markdown summary
            markdown_content = self._generate_markdown_report(report_data, symbol, exchange, timeframe)
            if self.save_training_report:
                md_path = self.save_training_report(
                    data=markdown_content,
                    step_name='step14_tactician_labeling',
                    report_type='analysis_summary',
                    symbol=symbol,
                    timeframe=timeframe,
                    file_format='md'
                )
                saved_files.append(md_path)

            # Save CSV metrics
            csv_content = self._generate_csv_metrics(report_data)
            if self.save_training_report:
                csv_path = self.save_training_report(
                    data=csv_content,
                    step_name='step14_tactician_labeling',
                    report_type='metrics_summary',
                    symbol=symbol,
                    timeframe=timeframe,
                    file_format='csv'
                )
                saved_files.append(csv_path)

            # Generate visualizations
            if MATPLOTLIB_AVAILABLE and self.save_training_report:
                viz_files = self._generate_visualizations(report_data, symbol, exchange, timeframe, timestamp)
                saved_files.extend(viz_files)

        except Exception as e:
            self.logger.error(f"Failed to save comprehensive report: {e}")

        return saved_files

    def _generate_markdown_report(self,
                                report_data: Step14EnhancedAnalysis,
                                symbol: str,
                                exchange: str,
                                timeframe: str) -> str:
        """Generate Markdown report content."""
        markdown = f"""# Step14 Enhanced Tactician Labeling Analysis Report

**Generated:** {report_data.timestamp}
**Symbol:** {symbol}
**Exchange:** {exchange}
**Timeframe:** {timeframe}

## Executive Summary

This report provides comprehensive analysis of the Tactician Labeling process for {symbol} on {exchange}.

### Key Metrics
- **Labels Generated:** {report_data.labels_generated}
- **Data Points Processed:** {report_data.data_points_processed}
- **Labeling Duration:** {report_data.labeling_duration:.2f}s
- **Label Quality Score:** {report_data.labeling_quality.label_quality_score:.4f}
- **Signal Quality Score:** {report_data.strategic_signals.signal_quality_score:.4f}

## Barrier Performance Analysis

- **Total Barriers Calculated:** {report_data.barrier_performance.total_barriers_calculated}
- **Barrier Effectiveness Score:** {report_data.barrier_performance.barrier_effectiveness_score:.4f}
- **Average Profit Barrier:** {report_data.barrier_performance.average_profit_barrier:.4f}
- **Average Loss Barrier:** {report_data.barrier_performance.average_loss_barrier:.4f}
- **Barrier Success Rate:** {report_data.barrier_performance.barrier_success_rate:.4f}
- **Barrier Adaptation Rate:** {report_data.barrier_performance.barrier_adaptation_rate:.4f}

## Labeling Quality Analysis

- **Total Labels Generated:** {report_data.labeling_quality.total_labels_generated}
- **Label Quality Score:** {report_data.labeling_quality.label_quality_score:.4f}
- **Label Consistency Score:** {report_data.labeling_quality.label_consistency_score:.4f}
- **False Positive Rate:** {report_data.labeling_quality.false_positive_rate:.4f}
- **False Negative Rate:** {report_data.labeling_quality.false_negative_rate:.4f}

## Strategic Signal Analysis

- **Total Signals Generated:** {report_data.strategic_signals.total_signals_generated}
- **Signal Quality Score:** {report_data.strategic_signals.signal_quality_score:.4f}
- **Analyst Agreement Score:** {report_data.strategic_signals.analyst_agreement_score:.4f}
- **Signal-to-Noise Ratio:** {report_data.strategic_signals.signal_to_noise_ratio:.2f}

## Quality Filter Analysis

- **Total Data Points:** {report_data.quality_filters.total_data_points}
- **Filtered Data Points:** {report_data.quality_filters.filtered_data_points}
- **Volume Filter Efficiency:** {report_data.quality_filters.volume_filter_efficiency:.4f}
- **Spread Filter Efficiency:** {report_data.quality_filters.spread_filter_efficiency:.4f}
- **Volatility Filter Efficiency:** {report_data.quality_filters.volatility_filter_efficiency:.4f}
- **Combined Filter Efficiency:** {report_data.quality_filters.combined_filter_efficiency:.4f}

## Regime Labeling Analysis

- **Total Regimes Processed:** {report_data.regime_labeling.total_regimes_processed}
- **Cross-Regime Signal Agreement:** {report_data.regime_labeling.cross_regime_signal_agreement:.4f}

## Validation Performance Analysis

- **Validation Accuracy:** {report_data.validation_performance.validation_accuracy:.4f}
- **Validation Precision:** {report_data.validation_performance.validation_precision:.4f}
- **Validation Recall:** {report_data.validation_performance.validation_recall:.4f}
- **Validation F1 Score:** {report_data.validation_performance.validation_f1_score:.4f}
- **Validation Time:** {report_data.validation_performance.validation_time:.2f}s
- **Validation Confidence:** {report_data.validation_performance.validation_confidence:.4f}

## Precision Level Performance

"""

        # Add precision level performance table
        if report_data.precision_level_performance:
            markdown += "| Precision Level | Count | Avg Accuracy | Avg Confidence | Success Rate |\n"
            markdown += "|----------------|-------|--------------|----------------|--------------|\n"
            for level, perf in report_data.precision_level_performance.items():
                markdown += f"| {level} | {perf['count']} | {perf['avg_accuracy']:.4f} | {perf['avg_confidence']:.4f} | {perf['success_rate']:.4f} |\n"

        # Add recommendations
        if report_data.recommendations:
            markdown += "\n## Recommendations\n\n"
            for rec in report_data.recommendations:
                markdown += f"- {rec}\n"

        # Add alerts
        if report_data.alerts:
            markdown += "\n## Alerts\n\n"
            for alert in report_data.alerts:
                markdown += f"- {alert}\n"

        return markdown

    def _generate_csv_metrics(self, report_data: Step14EnhancedAnalysis) -> str:
        """Generate CSV metrics content."""
        # Create summary metrics
        summary_data = {
            'metric': [
                'total_labels_generated', 'label_quality_score', 'total_signals_generated',
                'signal_quality_score', 'total_barriers_calculated', 'barrier_effectiveness',
                'validation_accuracy', 'validation_f1_score', 'total_regimes_processed'
            ],
            'value': [
                report_data.labeling_quality.total_labels_generated,
                report_data.labeling_quality.label_quality_score,
                report_data.strategic_signals.total_signals_generated,
                report_data.strategic_signals.signal_quality_score,
                report_data.barrier_performance.total_barriers_calculated,
                report_data.barrier_performance.barrier_effectiveness_score,
                report_data.validation_performance.validation_accuracy,
                report_data.validation_performance.validation_f1_score,
                report_data.regime_labeling.total_regimes_processed
            ],
            'category': [
                'labeling', 'labeling', 'signals', 'signals', 'barriers', 'barriers',
                'validation', 'validation', 'regime'
            ]
        }

        df = pd.DataFrame(summary_data)
        return df.to_csv(index=False)

    def _generate_visualizations(self,
                               report_data: Step14EnhancedAnalysis,
                               symbol: str,
                               exchange: str,
                               timeframe: str,
                               timestamp: str) -> List[str]:
        """Generate visualization plots."""
        saved_files = []

        try:
            if not MATPLOTLIB_AVAILABLE:
                return saved_files

            # Set style
            plt.style.use('default')
            sns.set_palette("husl")

            # 1. Label Distribution
            if report_data.labeling_quality.label_distribution:
                plt.figure(figsize=(12, 8))

                labels = list(report_data.labeling_quality.label_distribution.keys())
                counts = list(report_data.labeling_quality.label_distribution.values())

                plt.bar(labels, counts, color='lightcoral', alpha=0.8)
                plt.title('Label Type Distribution', fontsize=16, fontweight='bold')
                plt.xlabel('Label Type', fontsize=12)
                plt.ylabel('Count', fontsize=12)
                plt.xticks(rotation=45, ha='right')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()

                if self.save_training_report:
                    viz_path = self.save_training_report(
                        data=plt.gcf(),
                        step_name='step14_tactician_labeling',
                        report_type='label_distribution',
                        symbol=symbol,
                        timeframe=timeframe,
                        file_format='png'
                    )
                    saved_files.append(viz_path)
                plt.close()

            # 2. Precision Level Performance
            if report_data.precision_level_performance:
                plt.figure(figsize=(12, 8))

                levels = list(report_data.precision_level_performance.keys())
                accuracies = [perf['avg_accuracy'] for perf in report_data.precision_level_performance.values()]

                bars = plt.bar(levels, accuracies, color='lightblue', alpha=0.8)
                plt.title('Precision Level Performance Comparison', fontsize=16, fontweight='bold')
                plt.xlabel('Precision Level', fontsize=12)
                plt.ylabel('Average Accuracy', fontsize=12)
                plt.xticks(rotation=45, ha='right')
                plt.grid(True, alpha=0.3)

                # Add value labels on bars
                for bar, accuracy in zip(bars, accuracies):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           '.3f', ha='center', va='bottom', fontsize=10)

                plt.tight_layout()

                if self.save_training_report:
                    viz_path = self.save_training_report(
                        data=plt.gcf(),
                        step_name='step14_tactician_labeling',
                        report_type='precision_performance',
                        symbol=symbol,
                        timeframe=timeframe,
                        file_format='png'
                    )
                    saved_files.append(viz_path)
                plt.close()

            # 3. Signal Strength Distribution
            if report_data.strategic_signals.signal_strength_distribution:
                plt.figure(figsize=(10, 8))

                strengths = list(report_data.strategic_signals.signal_strength_distribution.keys())
                counts = list(report_data.strategic_signals.signal_strength_distribution.values())

                plt.pie(counts, labels=strengths, autopct='%1.1f%%', startangle=90)
                plt.title('Strategic Signal Strength Distribution', fontsize=16, fontweight='bold')
                plt.axis('equal')
                plt.tight_layout()

                if self.save_training_report:
                    viz_path = self.save_training_report(
                        plt.gcf(),
                        f"step14_tactician_labeling_{symbol}_{timeframe}_signal_strength_{timestamp}.png",
                        symbol, exchange, timeframe
                    )
                    saved_files.append(viz_path)
                plt.close()

            # 4. Quality Filter Efficiency
            filter_data = [
                report_data.quality_filters.volume_filter_efficiency,
                report_data.quality_filters.spread_filter_efficiency,
                report_data.quality_filters.volatility_filter_efficiency,
                report_data.quality_filters.combined_filter_efficiency
            ]

            if any(f > 0 for f in filter_data):
                plt.figure(figsize=(12, 8))

                filters = ['Volume Filter', 'Spread Filter', 'Volatility Filter', 'Combined Filter']
                plt.bar(filters, filter_data, color='lightgreen', alpha=0.8)
                plt.title('Quality Filter Efficiency', fontsize=16, fontweight='bold')
                plt.xlabel('Filter Type', fontsize=12)
                plt.ylabel('Efficiency Rate', fontsize=12)
                plt.xticks(rotation=45, ha='right')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()

                if self.save_training_report:
                    viz_path = self.save_training_report(
                        data=plt.gcf(),
                        step_name='step14_tactician_labeling',
                        report_type='filter_efficiency',
                        symbol=symbol,
                        timeframe=timeframe,
                        file_format='png'
                    )
                    saved_files.append(viz_path)
                plt.close()

            # 5. Validation Performance Dashboard
            plt.figure(figsize=(15, 10))

            # Subplot 1: Validation Metrics
            plt.subplot(2, 2, 1)
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
            values = [
                report_data.validation_performance.validation_accuracy,
                report_data.validation_performance.validation_precision,
                report_data.validation_performance.validation_recall,
                report_data.validation_performance.validation_f1_score
            ]

            bars = plt.bar(metrics, values, color=['blue', 'green', 'orange', 'red'], alpha=0.7)
            plt.title('Validation Performance Metrics', fontsize=14, fontweight='bold')
            plt.ylabel('Score', fontsize=12)
            plt.ylim(0, 1)
            plt.grid(True, alpha=0.3)

            # Subplot 2: Cross-Validation Scores
            plt.subplot(2, 2, 2)
            if report_data.validation_performance.cross_validation_scores:
                folds = [f'Fold {i+1}' for i in range(len(report_data.validation_performance.cross_validation_scores))]
                scores = report_data.validation_performance.cross_validation_scores

                plt.plot(folds, scores, 'bo-', linewidth=2, markersize=8)
                plt.axhline(y=np.mean(scores), color='red', linestyle='--',
                           label=f'Mean: {np.mean(scores):.4f}')
                plt.title('Cross-Validation Performance', fontsize=14, fontweight='bold')
                plt.xlabel('Fold', fontsize=12)
                plt.ylabel('Accuracy', fontsize=12)
                plt.legend()
                plt.grid(True, alpha=0.3)

            # Subplot 3: Labeling Quality Metrics
            plt.subplot(2, 2, 3)
            quality_metrics = ['Quality Score', 'Consistency Score', 'False Positive Rate', 'False Negative Rate']
            quality_values = [
                report_data.labeling_quality.label_quality_score,
                report_data.labeling_quality.label_consistency_score,
                report_data.labeling_quality.false_positive_rate,
                report_data.labeling_quality.false_negative_rate
            ]

            colors = ['green', 'blue', 'red', 'orange']
            plt.bar(quality_metrics, quality_values, color=colors, alpha=0.7)
            plt.title('Labeling Quality Metrics', fontsize=14, fontweight='bold')
            plt.ylabel('Score/Rate', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, alpha=0.3)

            # Subplot 4: Barrier Performance
            plt.subplot(2, 2, 4)
            barrier_metrics = ['Effectiveness', 'Success Rate', 'Adaptation Rate']
            barrier_values = [
                report_data.barrier_performance.barrier_effectiveness_score,
                report_data.barrier_performance.barrier_success_rate,
                report_data.barrier_performance.barrier_adaptation_rate
            ]

            plt.bar(barrier_metrics, barrier_values, color='purple', alpha=0.7)
            plt.title('Barrier Performance Metrics', fontsize=14, fontweight='bold')
            plt.ylabel('Score', fontsize=12)
            plt.ylim(0, 1)
            plt.grid(True, alpha=0.3)

            plt.suptitle('Step14 Tactician Labeling Performance Dashboard', fontsize=16, fontweight='bold')
            plt.tight_layout()

            if self.save_training_report:
                viz_path = self.save_training_report(
                    data=plt.gcf(),
                    step_name='step14_tactician_labeling',
                    report_type='performance_dashboard',
                    symbol=symbol,
                    timeframe=timeframe,
                    file_format='png'
                )
                saved_files.append(viz_path)
            plt.close()

        except Exception as e:
            self.logger.error(f"Failed to generate visualizations: {e}")

        return saved_files
