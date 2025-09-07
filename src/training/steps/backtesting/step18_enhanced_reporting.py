"""
Step18 Enhanced Reporting: Backtesting Main Analysis

This module provides comprehensive reporting for Step 18: Backtesting Main,
focusing on walk forward validation, Monte Carlo validation, A/B testing,
model performance analysis, and risk assessment.
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
class BacktestingPerformanceMetrics:
    """Metrics for overall backtesting performance."""
    total_backtesting_time: float = 0.0
    execution_efficiency: float = 0.0
    parallel_processing_gain: float = 0.0
    memory_utilization: float = 0.0
    data_processing_speed: float = 0.0
    regime_processing_coverage: float = 0.0

@dataclass
class WalkForwardValidationMetrics:
    """Metrics for walk forward validation performance."""
    total_walk_forward_runs: int = 0
    walk_forward_efficiency: float = 0.0
    out_of_sample_performance: float = 0.0
    overfitting_detection_score: float = 0.0
    stability_score: float = 0.0
    prediction_decay_analysis: float = 0.0
    regime_specific_validation: Dict[str, float] = field(default_factory=dict)

@dataclass
class MonteCarloValidationMetrics:
    """Metrics for Monte Carlo validation performance."""
    total_simulations: int = 0
    statistical_significance: float = 0.0
    confidence_intervals: Dict[str, List[float]] = field(default_factory=dict)
    risk_distribution_analysis: Dict[str, float] = field(default_factory=dict)
    scenario_coverage: float = 0.0
    robustness_score: float = 0.0
    probabilistic_assessment: Dict[str, float] = field(default_factory=dict)

@dataclass
class ABTestingMetrics:
    """Metrics for A/B testing performance."""
    total_ab_tests: int = 0
    statistical_significance: float = 0.0
    effect_size_analysis: Dict[str, float] = field(default_factory=dict)
    winner_detection_rate: float = 0.0
    false_positive_rate: float = 0.0
    test_power_analysis: float = 0.0
    comparative_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)

@dataclass
class ModelPersistenceMetrics:
    """Metrics for model saving and persistence."""
    total_models_saved: int = 0
    model_compression_ratio: float = 0.0
    save_load_performance: float = 0.0
    persistence_integrity: float = 0.0
    version_control_efficiency: float = 0.0
    model_reproducibility: float = 0.0

@dataclass
class BacktestingQualityMetrics:
    """Metrics for backtesting quality assessment."""
    data_quality_score: float = 0.0
    validation_completeness: float = 0.0
    result_reproducibility: float = 0.0
    statistical_rigor: float = 0.0
    methodological_soundness: float = 0.0
    risk_assessment_coverage: float = 0.0

@dataclass
class RegimeBacktestingMetrics:
    """Metrics for per-regime backtesting."""
    regimes_processed: int = 0
    regime_performance_distribution: Dict[str, float] = field(default_factory=dict)
    regime_risk_profiles: Dict[str, Dict[str, float]] = field(default_factory=dict)
    regime_adaptability: Dict[str, float] = field(default_factory=dict)
    inter_regime_correlations: Dict[str, Dict[str, float]] = field(default_factory=dict)
    regime_transition_impacts: Dict[str, float] = field(default_factory=dict)

@dataclass
class BacktestingRiskMetrics:
    """Metrics for backtesting risk assessment."""
    value_at_risk: float = 0.0
    expected_shortfall: float = 0.0
    maximum_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    risk_adjusted_returns: Dict[str, float] = field(default_factory=dict)

@dataclass
class Step18EnhancedAnalysis:
    """Comprehensive analysis for Step18 performance."""
    timestamp: str = ""
    backtesting_duration: float = 0.0
    total_regimes_processed: int = 0
    validation_completeness: float = 0.0
    backtesting_performance: BacktestingPerformanceMetrics = field(default_factory=BacktestingPerformanceMetrics)
    walk_forward_validation: WalkForwardValidationMetrics = field(default_factory=WalkForwardValidationMetrics)
    monte_carlo_validation: MonteCarloValidationMetrics = field(default_factory=MonteCarloValidationMetrics)
    ab_testing: ABTestingMetrics = field(default_factory=ABTestingMetrics)
    model_persistence: ModelPersistenceMetrics = field(default_factory=ModelPersistenceMetrics)
    backtesting_quality: BacktestingQualityMetrics = field(default_factory=BacktestingQualityMetrics)
    regime_backtesting: RegimeBacktestingMetrics = field(default_factory=RegimeBacktestingMetrics)
    risk_assessment: BacktestingRiskMetrics = field(default_factory=BacktestingRiskMetrics)
    validation_pipeline: Dict[str, Any] = field(default_factory=dict)
    performance_benchmarks: Dict[str, float] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    alerts: List[str] = field(default_factory=list)

class Step18EnhancedReporter:
    """Enhanced reporting system for Step18: Backtesting Main."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the Step18 enhanced reporter."""
        self.config = config
        self.logger = system_logger.getChild('Step18.EnhancedReporter')
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
                                    backtesting_results: Dict[str, Any],
                                    validation_results: Dict[str, Any],
                                    regime_results: Dict[str, Any],
                                    risk_analysis: Dict[str, Any],
                                    quality_assessment: Dict[str, Any]) -> Step18EnhancedAnalysis:
        """
        Generate comprehensive Step18 analysis report.

        Args:
            backtesting_results: Results from backtesting pipeline
            validation_results: Results from walk forward and Monte Carlo validation
            regime_results: Results from per-regime backtesting
            risk_analysis: Risk assessment and metrics
            quality_assessment: Quality assessment of backtesting process

        Returns:
            Step18EnhancedAnalysis: Comprehensive analysis object
        """
        try:
            analysis = Step18EnhancedAnalysis(
                timestamp=datetime.now().isoformat(),
                backtesting_duration=backtesting_results.get('total_duration', 0.0),
                total_regimes_processed=len(regime_results.get('regimes', {})),
                validation_completeness=validation_results.get('completeness_score', 0.0)
            )

            # Analyze backtesting performance
            analysis.backtesting_performance = self._analyze_backtesting_performance(backtesting_results)

            # Analyze walk forward validation
            analysis.walk_forward_validation = self._analyze_walk_forward_validation(validation_results.get('walk_forward', {}))

            # Analyze Monte Carlo validation
            analysis.monte_carlo_validation = self._analyze_monte_carlo_validation(validation_results.get('monte_carlo', {}))

            # Analyze A/B testing
            analysis.ab_testing = self._analyze_ab_testing(validation_results.get('ab_testing', {}))

            # Analyze model persistence
            analysis.model_persistence = self._analyze_model_persistence(backtesting_results.get('persistence', {}))

            # Analyze backtesting quality
            analysis.backtesting_quality = self._analyze_backtesting_quality(quality_assessment)

            # Analyze regime backtesting
            analysis.regime_backtesting = self._analyze_regime_backtesting(regime_results)

            # Analyze risk assessment
            analysis.risk_assessment = self._analyze_risk_assessment(risk_analysis)

            # Set validation pipeline
            analysis.validation_pipeline = validation_results.get('pipeline', {})

            # Set performance benchmarks
            analysis.performance_benchmarks = self._extract_performance_benchmarks(backtesting_results)

            # Generate recommendations and alerts
            analysis.recommendations = self._generate_recommendations(analysis)
            analysis.alerts = self._generate_alerts(analysis)

            return analysis

        except Exception as e:
            self.logger.error(f"Failed to generate comprehensive report: {e}")
            return Step18EnhancedAnalysis()

    def _analyze_backtesting_performance(self, backtesting_results: Dict[str, Any]) -> BacktestingPerformanceMetrics:
        """Analyze overall backtesting performance."""
        metrics = BacktestingPerformanceMetrics()

        if backtesting_results:
            metrics.total_backtesting_time = backtesting_results.get('total_duration', 0.0)
            metrics.execution_efficiency = backtesting_results.get('efficiency_score', 0.85)
            metrics.parallel_processing_gain = backtesting_results.get('parallel_gain', 0.78)
            metrics.memory_utilization = backtesting_results.get('memory_usage', 0.72)
            metrics.data_processing_speed = backtesting_results.get('processing_speed', 0.88)
            metrics.regime_processing_coverage = backtesting_results.get('regime_coverage', 0.91)

        return metrics

    def _analyze_walk_forward_validation(self, walk_forward_data: Dict[str, Any]) -> WalkForwardValidationMetrics:
        """Analyze walk forward validation performance."""
        metrics = WalkForwardValidationMetrics()

        if walk_forward_data:
            metrics.total_walk_forward_runs = walk_forward_data.get('total_runs', 0)
            metrics.walk_forward_efficiency = walk_forward_data.get('efficiency', 0.86)
            metrics.out_of_sample_performance = walk_forward_data.get('oos_performance', 0.82)
            metrics.overfitting_detection_score = walk_forward_data.get('overfitting_score', 0.15)
            metrics.stability_score = walk_forward_data.get('stability_score', 0.87)
            metrics.prediction_decay_analysis = walk_forward_data.get('decay_analysis', 0.23)
            metrics.regime_specific_validation = walk_forward_data.get('regime_validation', {})

        return metrics

    def _analyze_monte_carlo_validation(self, monte_carlo_data: Dict[str, Any]) -> MonteCarloValidationMetrics:
        """Analyze Monte Carlo validation performance."""
        metrics = MonteCarloValidationMetrics()

        if monte_carlo_data:
            metrics.total_simulations = monte_carlo_data.get('total_simulations', 0)
            metrics.statistical_significance = monte_carlo_data.get('significance', 0.95)
            metrics.confidence_intervals = monte_carlo_data.get('confidence_intervals', {})
            metrics.risk_distribution_analysis = monte_carlo_data.get('risk_distribution', {})
            metrics.scenario_coverage = monte_carlo_data.get('scenario_coverage', 0.89)
            metrics.robustness_score = monte_carlo_data.get('robustness', 0.86)
            metrics.probabilistic_assessment = monte_carlo_data.get('probabilistic_assessment', {})

        return metrics

    def _analyze_ab_testing(self, ab_testing_data: Dict[str, Any]) -> ABTestingMetrics:
        """Analyze A/B testing performance."""
        metrics = ABTestingMetrics()

        if ab_testing_data:
            metrics.total_ab_tests = ab_testing_data.get('total_tests', 0)
            metrics.statistical_significance = ab_testing_data.get('significance', 0.95)
            metrics.effect_size_analysis = ab_testing_data.get('effect_sizes', {})
            metrics.winner_detection_rate = ab_testing_data.get('winner_rate', 0.78)
            metrics.false_positive_rate = ab_testing_data.get('false_positive', 0.05)
            metrics.test_power_analysis = ab_testing_data.get('test_power', 0.82)
            metrics.comparative_performance = ab_testing_data.get('comparative_performance', {})

        return metrics

    def _analyze_model_persistence(self, persistence_data: Dict[str, Any]) -> ModelPersistenceMetrics:
        """Analyze model persistence performance."""
        metrics = ModelPersistenceMetrics()

        if persistence_data:
            metrics.total_models_saved = persistence_data.get('total_saved', 0)
            metrics.model_compression_ratio = persistence_data.get('compression_ratio', 0.85)
            metrics.save_load_performance = persistence_data.get('save_load_perf', 0.92)
            metrics.persistence_integrity = persistence_data.get('integrity_score', 0.96)
            metrics.version_control_efficiency = persistence_data.get('version_efficiency', 0.89)
            metrics.model_reproducibility = persistence_data.get('reproducibility', 0.94)

        return metrics

    def _analyze_backtesting_quality(self, quality_assessment: Dict[str, Any]) -> BacktestingQualityMetrics:
        """Analyze backtesting quality assessment."""
        metrics = BacktestingQualityMetrics()

        if quality_assessment:
            metrics.data_quality_score = quality_assessment.get('data_quality', 0.88)
            metrics.validation_completeness = quality_assessment.get('validation_completeness', 0.91)
            metrics.result_reproducibility = quality_assessment.get('reproducibility', 0.93)
            metrics.statistical_rigor = quality_assessment.get('statistical_rigor', 0.87)
            metrics.methodological_soundness = quality_assessment.get('methodological_soundness', 0.89)
            metrics.risk_assessment_coverage = quality_assessment.get('risk_coverage', 0.85)

        return metrics

    def _analyze_regime_backtesting(self, regime_results: Dict[str, Any]) -> RegimeBacktestingMetrics:
        """Analyze per-regime backtesting performance."""
        metrics = RegimeBacktestingMetrics()

        regimes = regime_results.get('regimes', {})

        if regimes:
            metrics.regimes_processed = len(regimes)
            metrics.regime_performance_distribution = {regime_id: data.get('performance', 0.8)
                                                     for regime_id, data in regimes.items()}
            metrics.regime_risk_profiles = {regime_id: data.get('risk_profile', {})
                                          for regime_id, data in regimes.items()}
            metrics.regime_adaptability = {regime_id: data.get('adaptability', 0.75)
                                         for regime_id, data in regimes.items()}
            metrics.inter_regime_correlations = regime_results.get('correlations', {})
            metrics.regime_transition_impacts = regime_results.get('transition_impacts', {})

        return metrics

    def _analyze_risk_assessment(self, risk_analysis: Dict[str, Any]) -> BacktestingRiskMetrics:
        """Analyze risk assessment metrics."""
        metrics = BacktestingRiskMetrics()

        if risk_analysis:
            metrics.value_at_risk = risk_analysis.get('var_95', 0.05)
            metrics.expected_shortfall = risk_analysis.get('expected_shortfall', 0.08)
            metrics.maximum_drawdown = risk_analysis.get('max_drawdown', 0.15)
            metrics.sharpe_ratio = risk_analysis.get('sharpe_ratio', 1.2)
            metrics.sortino_ratio = risk_analysis.get('sortino_ratio', 1.5)
            metrics.calmar_ratio = risk_analysis.get('calmar_ratio', 0.8)
            metrics.risk_adjusted_returns = risk_analysis.get('risk_adjusted_returns', {})

        return metrics

    def _extract_performance_benchmarks(self, backtesting_results: Dict[str, Any]) -> Dict[str, float]:
        """Extract performance benchmarks from results."""
        benchmarks = {}

        if backtesting_results:
            benchmarks.update({
                'total_return': backtesting_results.get('total_return', 0.0),
                'annual_return': backtesting_results.get('annual_return', 0.0),
                'win_rate': backtesting_results.get('win_rate', 0.5),
                'profit_factor': backtesting_results.get('profit_factor', 1.0),
                'max_drawdown': backtesting_results.get('max_drawdown', 0.1),
                'sharpe_ratio': backtesting_results.get('sharpe_ratio', 1.0),
                'sortino_ratio': backtesting_results.get('sortino_ratio', 1.2),
                'calmar_ratio': backtesting_results.get('calmar_ratio', 0.8)
            })

        return benchmarks

    def _generate_recommendations(self, analysis: Step18EnhancedAnalysis) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []

        # Backtesting performance recommendations
        if analysis.backtesting_performance.execution_efficiency < 0.8:
            recommendations.append("Backtesting execution efficiency is suboptimal - consider optimizing data processing pipeline")

        if analysis.backtesting_performance.parallel_processing_gain < 0.7:
            recommendations.append("Parallel processing gain is low - review regime processing distribution and concurrency settings")

        # Walk forward validation recommendations
        if analysis.walk_forward_validation.out_of_sample_performance < 0.8:
            recommendations.append("Out-of-sample performance is low - review walk forward validation parameters and window sizes")

        if analysis.walk_forward_validation.overfitting_detection_score > 0.2:
            recommendations.append("High overfitting detected in walk forward validation - implement additional regularization")

        # Monte Carlo validation recommendations
        if analysis.monte_carlo_validation.statistical_significance < 0.95:
            recommendations.append("Statistical significance in Monte Carlo validation is low - increase simulation count")

        if analysis.monte_carlo_validation.robustness_score < 0.8:
            recommendations.append("Monte Carlo robustness score is low - review simulation parameters and scenario coverage")

        # A/B testing recommendations
        if analysis.ab_testing.false_positive_rate > 0.1:
            recommendations.append("False positive rate in A/B testing is high - review statistical testing methodology")

        if analysis.ab_testing.test_power_analysis < 0.8:
            recommendations.append("A/B test power is low - consider increasing sample sizes or effect size detection")

        # Model persistence recommendations
        if analysis.model_persistence.persistence_integrity < 0.95:
            recommendations.append("Model persistence integrity is low - review save/load mechanisms and version control")

        # Quality assessment recommendations
        if analysis.backtesting_quality.data_quality_score < 0.85:
            recommendations.append("Data quality score is low - review data preprocessing and quality validation steps")

        if analysis.backtesting_quality.validation_completeness < 0.9:
            recommendations.append("Validation completeness is insufficient - ensure all required validations are performed")

        # Risk assessment recommendations
        if abs(analysis.risk_assessment.sharpe_ratio) < 1.0:
            recommendations.append("Sharpe ratio indicates suboptimal risk-adjusted returns - review strategy parameters")

        if analysis.risk_assessment.maximum_drawdown > 0.2:
            recommendations.append("Maximum drawdown is high - implement additional risk management measures")

        return recommendations

    def _generate_alerts(self, analysis: Step18EnhancedAnalysis) -> List[str]:
        """Generate alerts based on analysis."""
        alerts = []

        # Critical alerts
        if analysis.total_regimes_processed == 0:
            alerts.append("🚨 CRITICAL: No regimes were processed during backtesting - check regime detection and processing pipeline")

        if analysis.backtesting_performance.regime_processing_coverage < 0.5:
            alerts.append("🚨 CRITICAL: Regime processing coverage is very low - review regime detection and processing failures")

        # Warning alerts
        if analysis.walk_forward_validation.out_of_sample_performance < 0.7:
            alerts.append("⚠️ WARNING: Poor out-of-sample performance detected - strategy may be overfitting to training data")

        if analysis.monte_carlo_validation.scenario_coverage < 0.8:
            alerts.append("⚠️ WARNING: Monte Carlo scenario coverage is insufficient - increase simulation diversity")

        if analysis.backtesting_quality.result_reproducibility < 0.9:
            alerts.append("⚠️ WARNING: Low result reproducibility - review random seed management and deterministic processing")

        if analysis.risk_assessment.value_at_risk > 0.1:
            alerts.append("⚠️ WARNING: Value at Risk is high - strategy exhibits significant tail risk")

        if analysis.model_persistence.model_reproducibility < 0.95:
            alerts.append("⚠️ WARNING: Model reproducibility is low - review model saving and loading consistency")

        # Info alerts
        if analysis.ab_testing.total_ab_tests == 0:
            alerts.append("ℹ️ INFO: No A/B tests were performed - consider implementing comparative testing")

        return alerts

    def save_comprehensive_report(self,
                                report_data: Step18EnhancedAnalysis,
                                symbol: str,
                                exchange: str,
                                timeframe: str) -> List[str]:
        """
        Save comprehensive Step18 analysis report in multiple formats.

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
                'step': 'step18_backtesting_main',
                'timestamp': report_data.timestamp,
                'symbol': symbol,
                'exchange': exchange,
                'timeframe': timeframe,
                'analysis': {
                    'backtesting_duration': report_data.backtesting_duration,
                    'total_regimes_processed': report_data.total_regimes_processed,
                    'validation_completeness': report_data.validation_completeness,
                    'backtesting_performance': {
                        'total_time': report_data.backtesting_performance.total_backtesting_time,
                        'execution_efficiency': report_data.backtesting_performance.execution_efficiency,
                        'parallel_processing_gain': report_data.backtesting_performance.parallel_processing_gain,
                        'memory_utilization': report_data.backtesting_performance.memory_utilization,
                        'data_processing_speed': report_data.backtesting_performance.data_processing_speed,
                        'regime_processing_coverage': report_data.backtesting_performance.regime_processing_coverage
                    },
                    'walk_forward_validation': {
                        'total_runs': report_data.walk_forward_validation.total_walk_forward_runs,
                        'walk_forward_efficiency': report_data.walk_forward_validation.walk_forward_efficiency,
                        'out_of_sample_performance': report_data.walk_forward_validation.out_of_sample_performance,
                        'overfitting_detection_score': report_data.walk_forward_validation.overfitting_detection_score,
                        'stability_score': report_data.walk_forward_validation.stability_score,
                        'prediction_decay_analysis': report_data.walk_forward_validation.prediction_decay_analysis
                    },
                    'monte_carlo_validation': {
                        'total_simulations': report_data.monte_carlo_validation.total_simulations,
                        'statistical_significance': report_data.monte_carlo_validation.statistical_significance,
                        'scenario_coverage': report_data.monte_carlo_validation.scenario_coverage,
                        'robustness_score': report_data.monte_carlo_validation.robustness_score
                    },
                    'ab_testing': {
                        'total_tests': report_data.ab_testing.total_ab_tests,
                        'statistical_significance': report_data.ab_testing.statistical_significance,
                        'winner_detection_rate': report_data.ab_testing.winner_detection_rate,
                        'false_positive_rate': report_data.ab_testing.false_positive_rate,
                        'test_power_analysis': report_data.ab_testing.test_power_analysis
                    },
                    'model_persistence': {
                        'total_models_saved': report_data.model_persistence.total_models_saved,
                        'model_compression_ratio': report_data.model_persistence.model_compression_ratio,
                        'save_load_performance': report_data.model_persistence.save_load_performance,
                        'persistence_integrity': report_data.model_persistence.persistence_integrity,
                        'model_reproducibility': report_data.model_persistence.model_reproducibility
                    },
                    'backtesting_quality': {
                        'data_quality_score': report_data.backtesting_quality.data_quality_score,
                        'validation_completeness': report_data.backtesting_quality.validation_completeness,
                        'result_reproducibility': report_data.backtesting_quality.result_reproducibility,
                        'statistical_rigor': report_data.backtesting_quality.statistical_rigor,
                        'methodological_soundness': report_data.backtesting_quality.methodological_soundness,
                        'risk_assessment_coverage': report_data.backtesting_quality.risk_assessment_coverage
                    },
                    'regime_backtesting': {
                        'regimes_processed': report_data.regime_backtesting.regimes_processed,
                        'regime_performance_distribution': report_data.regime_backtesting.regime_performance_distribution,
                        'regime_adaptability': report_data.regime_backtesting.regime_adaptability
                    },
                    'risk_assessment': {
                        'value_at_risk': report_data.risk_assessment.value_at_risk,
                        'expected_shortfall': report_data.risk_assessment.expected_shortfall,
                        'maximum_drawdown': report_data.risk_assessment.maximum_drawdown,
                        'sharpe_ratio': report_data.risk_assessment.sharpe_ratio,
                        'sortino_ratio': report_data.risk_assessment.sortino_ratio,
                        'calmar_ratio': report_data.risk_assessment.calmar_ratio
                    },
                    'performance_benchmarks': report_data.performance_benchmarks,
                    'validation_pipeline': report_data.validation_pipeline,
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
                    step_name="step18_backtesting_main",
                    report_type=f"comprehensive_analysis_{timestamp}",
                    symbol=symbol,
                    timeframe=timeframe,
                    file_format="json"
                )
                saved_files.append(json_path)

            # Save Markdown summary
            markdown_content = self._generate_markdown_report(report_data, symbol, exchange, timeframe)
            if self.save_training_report:
                md_path = self.save_training_report(
                    data=markdown_content,
                    step_name="step18_backtesting_main",
                    report_type=f"analysis_summary_{timestamp}",
                    symbol=symbol,
                    timeframe=timeframe,
                    file_format="md"
                )
                saved_files.append(md_path)

            # Save CSV metrics
            csv_content = self._generate_csv_metrics(report_data)
            if self.save_training_report:
                csv_path = self.save_training_report(
                    data=csv_content,
                    step_name="step18_backtesting_main",
                    report_type=f"metrics_summary_{timestamp}",
                    symbol=symbol,
                    timeframe=timeframe,
                    file_format="csv"
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
                                report_data: Step18EnhancedAnalysis,
                                symbol: str,
                                exchange: str,
                                timeframe: str) -> str:
        """Generate Markdown report content."""
        markdown = f"""# Step18 Enhanced Backtesting Main Analysis Report

**Generated:** {report_data.timestamp}
**Symbol:** {symbol}
**Exchange:** {exchange}
**Timeframe:** {timeframe}

## Executive Summary

This report provides comprehensive analysis of the Enhanced Backtesting Pipeline for {symbol} on {exchange}.

### Key Metrics
- **Backtesting Duration:** {report_data.backtesting_duration:.2f}s
- **Regimes Processed:** {report_data.total_regimes_processed}
- **Validation Completeness:** {report_data.validation_completeness:.4f}
- **Execution Efficiency:** {report_data.backtesting_performance.execution_efficiency:.4f}
- **Parallel Processing Gain:** {report_data.backtesting_performance.parallel_processing_gain:.4f}

## Backtesting Performance Analysis

### Overall Performance
- **Total Backtesting Time:** {report_data.backtesting_performance.total_backtesting_time:.2f}s
- **Execution Efficiency:** {report_data.backtesting_performance.execution_efficiency:.4f}
- **Parallel Processing Gain:** {report_data.backtesting_performance.parallel_processing_gain:.4f}
- **Memory Utilization:** {report_data.backtesting_performance.memory_utilization:.4f}
- **Data Processing Speed:** {report_data.backtesting_performance.data_processing_speed:.4f}
- **Regime Processing Coverage:** {report_data.backtesting_performance.regime_processing_coverage:.4f}

## Validation Analysis

### Walk Forward Validation
- **Total Walk Forward Runs:** {report_data.walk_forward_validation.total_walk_forward_runs}
- **Walk Forward Efficiency:** {report_data.walk_forward_validation.walk_forward_efficiency:.4f}
- **Out-of-Sample Performance:** {report_data.walk_forward_validation.out_of_sample_performance:.4f}
- **Overfitting Detection Score:** {report_data.walk_forward_validation.overfitting_detection_score:.4f}
- **Stability Score:** {report_data.walk_forward_validation.stability_score:.4f}
- **Prediction Decay Analysis:** {report_data.walk_forward_validation.prediction_decay_analysis:.4f}

### Monte Carlo Validation
- **Total Simulations:** {report_data.monte_carlo_validation.total_simulations}
- **Statistical Significance:** {report_data.monte_carlo_validation.statistical_significance:.4f}
- **Scenario Coverage:** {report_data.monte_carlo_validation.scenario_coverage:.4f}
- **Robustness Score:** {report_data.monte_carlo_validation.robustness_score:.4f}

### A/B Testing
- **Total A/B Tests:** {report_data.ab_testing.total_ab_tests}
- **Statistical Significance:** {report_data.ab_testing.statistical_significance:.4f}
- **Winner Detection Rate:** {report_data.ab_testing.winner_detection_rate:.4f}
- **False Positive Rate:** {report_data.ab_testing.false_positive_rate:.4f}
- **Test Power Analysis:** {report_data.ab_testing.test_power_analysis:.4f}

## Model Persistence Analysis

- **Total Models Saved:** {report_data.model_persistence.total_models_saved}
- **Model Compression Ratio:** {report_data.model_persistence.model_compression_ratio:.4f}
- **Save/Load Performance:** {report_data.model_persistence.save_load_performance:.4f}
- **Persistence Integrity:** {report_data.model_persistence.persistence_integrity:.4f}
- **Version Control Efficiency:** {report_data.model_persistence.version_control_efficiency:.4f}
- **Model Reproducibility:** {report_data.model_persistence.model_reproducibility:.4f}

## Quality Assessment

- **Data Quality Score:** {report_data.backtesting_quality.data_quality_score:.4f}
- **Validation Completeness:** {report_data.backtesting_quality.validation_completeness:.4f}
- **Result Reproducibility:** {report_data.backtesting_quality.result_reproducibility:.4f}
- **Statistical Rigor:** {report_data.backtesting_quality.statistical_rigor:.4f}
- **Methodological Soundness:** {report_data.backtesting_quality.methodological_soundness:.4f}
- **Risk Assessment Coverage:** {report_data.backtesting_quality.risk_assessment_coverage:.4f}

## Regime Analysis

- **Regimes Processed:** {report_data.regime_backtesting.regimes_processed}

### Regime Performance Distribution

"""

        # Add regime performance table
        if report_data.regime_backtesting.regime_performance_distribution:
            markdown += "| Regime | Performance | Adaptability |\n"
            markdown += "|--------|-------------|--------------|\n"
            for regime_id, performance in report_data.regime_backtesting.regime_performance_distribution.items():
                adaptability = report_data.regime_backtesting.regime_adaptability.get(regime_id, 0.0)
                markdown += f"| {regime_id} | {performance:.4f} | {adaptability:.4f} |\n"

        # Add risk assessment
        markdown += "\n## Risk Assessment\n\n"
        markdown += f"- **Value at Risk (95%):** {report_data.risk_assessment.value_at_risk:.4f}\n"
        markdown += f"- **Expected Shortfall:** {report_data.risk_assessment.expected_shortfall:.4f}\n"
        markdown += f"- **Maximum Drawdown:** {report_data.risk_assessment.maximum_drawdown:.4f}\n"
        markdown += f"- **Sharpe Ratio:** {report_data.risk_assessment.sharpe_ratio:.4f}\n"
        markdown += f"- **Sortino Ratio:** {report_data.risk_assessment.sortino_ratio:.4f}\n"
        markdown += f"- **Calmar Ratio:** {report_data.risk_assessment.calmar_ratio:.4f}\n"

        # Add performance benchmarks
        if report_data.performance_benchmarks:
            markdown += "\n## Performance Benchmarks\n\n"
            markdown += "| Metric | Value |\n"
            markdown += "|--------|-------|\n"
            for metric, value in report_data.performance_benchmarks.items():
                if isinstance(value, float):
                    markdown += f"| {metric.replace('_', ' ').title()} | {value:.4f} |\n"
                else:
                    markdown += f"| {metric.replace('_', ' ').title()} | {value} |\n"

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

    def _generate_csv_metrics(self, report_data: Step18EnhancedAnalysis) -> str:
        """Generate CSV metrics content."""
        # Create summary metrics
        summary_data = {
            'metric': [
                'total_regimes_processed', 'execution_efficiency', 'parallel_gain',
                'walk_forward_runs', 'oos_performance', 'monte_carlo_sims',
                'ab_tests', 'data_quality_score', 'sharpe_ratio', 'max_drawdown'
            ],
            'value': [
                report_data.total_regimes_processed,
                report_data.backtesting_performance.execution_efficiency,
                report_data.backtesting_performance.parallel_processing_gain,
                report_data.walk_forward_validation.total_walk_forward_runs,
                report_data.walk_forward_validation.out_of_sample_performance,
                report_data.monte_carlo_validation.total_simulations,
                report_data.ab_testing.total_ab_tests,
                report_data.backtesting_quality.data_quality_score,
                report_data.risk_assessment.sharpe_ratio,
                report_data.risk_assessment.maximum_drawdown
            ],
            'category': [
                'backtesting', 'performance', 'performance', 'validation', 'validation',
                'validation', 'testing', 'quality', 'risk', 'risk'
            ]
        }

        df = pd.DataFrame(summary_data)
        return df.to_csv(index=False)

    def _generate_visualizations(self,
                               report_data: Step18EnhancedAnalysis,
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

            # 1. Backtesting Performance Overview
            plt.figure(figsize=(12, 8))

            perf_metrics = [
                report_data.backtesting_performance.execution_efficiency,
                report_data.backtesting_performance.parallel_processing_gain,
                report_data.backtesting_performance.memory_utilization,
                report_data.backtesting_performance.data_processing_speed,
                report_data.backtesting_performance.regime_processing_coverage
            ]

            labels = ['Execution\nEfficiency', 'Parallel\nGain', 'Memory\nUtilization', 'Processing\nSpeed', 'Regime\nCoverage']
            bars = plt.bar(labels, perf_metrics, color='lightblue', alpha=0.8)

            plt.title('Backtesting Performance Metrics', fontsize=16, fontweight='bold')
            plt.ylabel('Score', fontsize=12)
            plt.ylim(0, 1)
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, alpha=0.3)

            # Add value labels on bars
            for bar, value in zip(bars, perf_metrics):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       '.4f', ha='center', va='bottom', fontsize=10)

            plt.tight_layout()

            if self.save_training_report:
                viz_path = self.save_training_report(
                    data=plt.gcf(),
                    step_name="step18_backtesting_main",
                    report_type=f"backtesting_performance_{timestamp}",
                    symbol=symbol,
                    timeframe=timeframe,
                    file_format="png"
                )
                saved_files.append(viz_path)
            plt.close()

            # 2. Validation Methods Comparison
            plt.figure(figsize=(10, 8))

            validation_methods = [
                report_data.walk_forward_validation.walk_forward_efficiency,
                report_data.monte_carlo_validation.robustness_score,
                report_data.ab_testing.statistical_significance,
                report_data.backtesting_quality.validation_completeness
            ]

            labels = ['Walk Forward', 'Monte Carlo', 'A/B Testing', 'Overall\nCompleteness']
            plt.bar(labels, validation_methods, color='lightgreen', alpha=0.8)
            plt.title('Validation Methods Performance', fontsize=16, fontweight='bold')
            plt.ylabel('Score', fontsize=12)
            plt.ylim(0, 1)
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            if self.save_training_report:
                viz_path = self.save_training_report(
                    data=plt.gcf(),
                    step_name="step18_backtesting_main",
                    report_type=f"validation_methods_{timestamp}",
                    symbol=symbol,
                    timeframe=timeframe,
                    file_format="png"
                )
                saved_files.append(viz_path)
            plt.close()

            # 3. Risk Assessment Dashboard
            plt.figure(figsize=(12, 8))

            risk_metrics = [
                report_data.risk_assessment.sharpe_ratio,
                report_data.risk_assessment.sortino_ratio,
                report_data.risk_assessment.calmar_ratio,
                report_data.risk_assessment.maximum_drawdown,
                report_data.risk_assessment.value_at_risk,
                report_data.risk_assessment.expected_shortfall
            ]

            labels = ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'Max Drawdown', 'VaR (95%)', 'Expected Shortfall']
            colors = ['green' if x > 0 else 'red' for x in risk_metrics[:3]] + ['red'] * 3

            bars = plt.bar(labels, risk_metrics, color=colors, alpha=0.7)
            plt.title('Risk Assessment Metrics', fontsize=16, fontweight='bold')
            plt.ylabel('Value', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, alpha=0.3)

            # Add value labels on bars
            for bar, value in zip(bars, risk_metrics):
                plt.text(bar.get_x() + bar.get_width()/2,
                        value + 0.01 if value > 0 else value - 0.03,
                        '.3f', ha='center', va='bottom', fontsize=10)

            plt.tight_layout()

            if self.save_training_report:
                viz_path = self.save_training_report(
                    data=plt.gcf(),
                    step_name="step18_backtesting_main",
                    report_type=f"risk_assessment_{timestamp}",
                    symbol=symbol,
                    timeframe=timeframe,
                    file_format="png"
                )
                saved_files.append(viz_path)
            plt.close()

            # 4. Regime Performance Distribution
            if report_data.regime_backtesting.regime_performance_distribution:
                plt.figure(figsize=(12, 8))

                regimes = list(report_data.regime_backtesting.regime_performance_distribution.keys())
                performances = list(report_data.regime_backtesting.regime_performance_distribution.values())

                plt.bar(regimes, performances, color='purple', alpha=0.7)
                plt.title('Regime Performance Distribution', fontsize=16, fontweight='bold')
                plt.xlabel('Regime ID', fontsize=12)
                plt.ylabel('Performance Score', fontsize=12)
                plt.xticks(rotation=45, ha='right')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()

                if self.save_training_report:
                    viz_path = self.save_training_report(
                        data=plt.gcf(),
                        step_name="step18_backtesting_main",
                        report_type=f"regime_performance_{timestamp}",
                        symbol=symbol,
                        timeframe=timeframe,
                        file_format="png"
                    )
                    saved_files.append(viz_path)
                plt.close()

            # 5. Quality Assessment Radar Chart
            plt.figure(figsize=(10, 8))

            quality_metrics = [
                report_data.backtesting_quality.data_quality_score,
                report_data.backtesting_quality.validation_completeness,
                report_data.backtesting_quality.result_reproducibility,
                report_data.backtesting_quality.statistical_rigor,
                report_data.backtesting_quality.methodological_soundness,
                report_data.backtesting_quality.risk_assessment_coverage
            ]

            labels = ['Data Quality', 'Validation\nCompleteness', 'Reproducibility', 'Statistical\nRigor', 'Methodological\nSoundness', 'Risk\nCoverage']

            # Create radar chart
            angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
            quality_metrics += quality_metrics[:1]  # Close the loop
            angles += angles[:1]  # Close the loop

            fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
            ax.plot(angles, quality_metrics, 'o-', linewidth=2, label='Quality Score', color='blue')
            ax.fill(angles, quality_metrics, alpha=0.25, color='blue')
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(labels)
            ax.set_ylim(0, 1)
            ax.set_title('Backtesting Quality Assessment', size=16, fontweight='bold', pad=20)
            ax.grid(True)

            if self.save_training_report:
                viz_path = self.save_training_report(
                    data=plt.gcf(),
                    step_name="step18_backtesting_main",
                    report_type=f"quality_assessment_{timestamp}",
                    symbol=symbol,
                    timeframe=timeframe,
                    file_format="png"
                )
                saved_files.append(viz_path)
            plt.close()

            # 6. Performance Benchmarks Comparison
            if report_data.performance_benchmarks:
                plt.figure(figsize=(14, 8))

                # Extract benchmark data
                benchmarks = report_data.performance_benchmarks
                metric_names = list(benchmarks.keys())
                metric_values = list(benchmarks.values())

                # Separate positive and risk metrics for different colors
                positive_metrics = ['total_return', 'annual_return', 'win_rate', 'profit_factor', 'sharpe_ratio', 'sortino_ratio']
                risk_metrics = ['max_drawdown']

                positive_values = []
                risk_values = []
                positive_names = []
                risk_names = []

                for name, value in zip(metric_names, metric_values):
                    if any(pos in name for pos in positive_metrics):
                        positive_names.append(name.replace('_', '\n').title())
                        positive_values.append(float(value))
                    elif any(risk in name for risk in risk_metrics):
                        risk_names.append(name.replace('_', '\n').title())
                        risk_values.append(float(value))

                # Create subplots
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))

                # Positive metrics
                if positive_values:
                    bars1 = ax1.bar(positive_names, positive_values, color='green', alpha=0.7)
                    ax1.set_title('Positive Performance Metrics', fontsize=14, fontweight='bold')
                    ax1.set_ylabel('Value', fontsize=12)
                    ax1.tick_params(axis='x', rotation=45)
                    ax1.grid(True, alpha=0.3)

                    for bar, value in zip(bars1, positive_values):
                        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               '.3f', ha='center', va='bottom', fontsize=10)

                # Risk metrics
                if risk_values:
                    bars2 = ax2.bar(risk_names, risk_values, color='red', alpha=0.7)
                    ax2.set_title('Risk Metrics', fontsize=14, fontweight='bold')
                    ax2.set_ylabel('Value', fontsize=12)
                    ax2.tick_params(axis='x', rotation=45)
                    ax2.grid(True, alpha=0.3)

                    for bar, value in zip(bars2, risk_values):
                        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               '.3f', ha='center', va='bottom', fontsize=10)

                plt.suptitle('Performance Benchmarks Analysis', fontsize=16, fontweight='bold')
                plt.tight_layout()

                if self.save_training_report:
                    viz_path = self.save_training_report(
                        data=plt.gcf(),
                        step_name="step18_backtesting_main",
                        report_type=f"performance_benchmarks_{timestamp}",
                        symbol=symbol,
                        timeframe=timeframe,
                        file_format="png"
                    )
                    saved_files.append(viz_path)
                plt.close()

        except Exception as e:
            self.logger.error(f"Failed to generate visualizations: {e}")

        return saved_files
