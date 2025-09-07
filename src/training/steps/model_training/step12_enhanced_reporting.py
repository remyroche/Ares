"""
Step12 Enhanced Reporting: Final Parameters Optimization Analysis

This module provides comprehensive reporting for Step 12: Final Parameters Optimization,
focusing on regime-aware analyst model enhancement, hyperparameter optimization,
feature selection, and advanced model optimizations.
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

from src.utils.logger import system_logger

# Local imports to avoid circular dependencies
try:
    from src.training.reports import CentralizedReportManager, save_training_report
except ImportError:
    try:
        from src.training.reports import CentralizedReportManager, save_training_report
    except ImportError:
        CentralizedReportManager = None
        save_training_report = None

@dataclass
class HyperparameterOptimizationMetrics:
    """Metrics for hyperparameter optimization performance."""
    model_type: str = ""
    total_trials: int = 0
    completed_trials: int = 0
    best_score: float = 0.0
    best_params: Dict[str, Any] = field(default_factory=dict)
    optimization_time: float = 0.0
    convergence_score: float = 0.0
    early_stopping_trials: int = 0
    pruning_efficiency: float = 0.0
    parameter_ranges: Dict[str, Dict[str, Union[int, float]]] = field(default_factory=dict)

@dataclass
class FeatureSelectionMetrics:
    """Metrics for feature selection performance."""
    method: str = ""
    original_feature_count: int = 0
    selected_feature_count: int = 0
    selection_score: float = 0.0
    feature_importance_scores: Dict[str, float] = field(default_factory=dict)
    correlation_reduction: float = 0.0
    vif_improvement: float = 0.0
    selection_time: float = 0.0
    stability_score: float = 0.0

@dataclass
class ModelEnhancementMetrics:
    """Metrics for model enhancement performance."""
    model_type: str = ""
    original_accuracy: float = 0.0
    enhanced_accuracy: float = 0.0
    improvement_percentage: float = 0.0
    enhancement_time: float = 0.0
    optimization_applied: List[str] = field(default_factory=list)
    memory_usage_mb: float = 0.0
    training_speedup: float = 0.0

@dataclass
class RegimeOptimizationMetrics:
    """Metrics for regime-specific optimization."""
    regime_name: str = ""
    models_enhanced: int = 0
    total_regime_samples: int = 0
    optimization_efficiency: float = 0.0
    regime_specific_improvements: Dict[str, float] = field(default_factory=dict)
    feature_selection_efficiency: float = 0.0
    hyperparameter_optimization_score: float = 0.0

@dataclass
class HardwareOptimizationMetrics:
    """Metrics for hardware acceleration performance."""
    gpu_utilization: float = 0.0
    m1_gpu_available: bool = False
    memory_efficiency: float = 0.0
    parallel_processing_efficiency: float = 0.0
    vectorized_operations_count: int = 0
    matrix_operations_speedup: float = 0.0
    batch_processing_time: float = 0.0

@dataclass
class AdvancedOptimizationMetrics:
    """Metrics for advanced model optimizations."""
    quantization_applied: bool = False
    pruning_applied: bool = False
    distillation_applied: bool = False
    model_size_reduction: float = 0.0
    inference_speed_improvement: float = 0.0
    accuracy_retention: float = 0.0
    compression_ratio: float = 0.0

@dataclass
class ParallelProcessingMetrics:
    """Metrics for parallel processing performance."""
    total_regimes: int = 0
    concurrent_regimes: int = 0
    total_processing_time: float = 0.0
    average_regime_time: float = 0.0
    processing_efficiency: float = 0.0
    memory_usage_pattern: str = ""
    bottleneck_analysis: str = ""

@dataclass
class Step12EnhancedAnalysis:
    """Comprehensive analysis for Step12 performance."""
    timestamp: str = ""
    total_models_enhanced: int = 0
    total_regimes_processed: int = 0
    overall_accuracy_improvement: float = 0.0
    total_optimization_time: float = 0.0
    hpo_metrics: HyperparameterOptimizationMetrics = field(default_factory=HyperparameterOptimizationMetrics)
    feature_selection_metrics: FeatureSelectionMetrics = field(default_factory=FeatureSelectionMetrics)
    model_enhancement_metrics: ModelEnhancementMetrics = field(default_factory=ModelEnhancementMetrics)
    regime_metrics: List[RegimeOptimizationMetrics] = field(default_factory=list)
    hardware_metrics: HardwareOptimizationMetrics = field(default_factory=HardwareOptimizationMetrics)
    advanced_optimization_metrics: AdvancedOptimizationMetrics = field(default_factory=AdvancedOptimizationMetrics)
    parallel_processing_metrics: ParallelProcessingMetrics = field(default_factory=ParallelProcessingMetrics)
    model_type_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    alerts: List[str] = field(default_factory=list)

class Step12EnhancedReporter:
    """Enhanced reporting system for Step12: Final Parameters Optimization."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the Step12 enhanced reporter."""
        self.config = config
        self.logger = system_logger.getChild('Step12.EnhancedReporter')
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
                                    optimization_results: Dict[str, Any],
                                    enhanced_models_summary: Dict[str, Any],
                                    hpo_metrics: Dict[str, Any],
                                    hardware_metrics: Dict[str, Any],
                                    parallel_metrics: Dict[str, Any]) -> Step12EnhancedAnalysis:
        """
        Generate comprehensive Step12 analysis report.

        Args:
            optimization_results: Results from optimization process
            enhanced_models_summary: Summary of enhanced models
            hpo_metrics: Hyperparameter optimization metrics
            hardware_metrics: Hardware acceleration metrics
            parallel_metrics: Parallel processing metrics

        Returns:
            Step12EnhancedAnalysis: Comprehensive analysis object
        """
        try:
            analysis = Step12EnhancedAnalysis(
                timestamp=datetime.now().isoformat(),
                total_models_enhanced=self._calculate_total_models_enhanced(enhanced_models_summary),
                total_regimes_processed=len(enhanced_models_summary),
                overall_accuracy_improvement=self._calculate_overall_improvement(enhanced_models_summary),
                total_optimization_time=optimization_results.get('duration', 0.0)
            )

            # Analyze hyperparameter optimization
            analysis.hpo_metrics = self._analyze_hyperparameter_optimization(hpo_metrics)

            # Analyze feature selection
            analysis.feature_selection_metrics = self._analyze_feature_selection(enhanced_models_summary)

            # Analyze model enhancement
            analysis.model_enhancement_metrics = self._analyze_model_enhancement(enhanced_models_summary)

            # Analyze regime-specific metrics
            analysis.regime_metrics = self._analyze_regime_metrics(enhanced_models_summary)

            # Analyze hardware optimization
            analysis.hardware_metrics = self._analyze_hardware_optimization(hardware_metrics)

            # Analyze advanced optimizations
            analysis.advanced_optimization_metrics = self._analyze_advanced_optimizations(enhanced_models_summary)

            # Analyze parallel processing
            analysis.parallel_processing_metrics = self._analyze_parallel_processing(parallel_metrics)

            # Analyze model type performance
            analysis.model_type_performance = self._analyze_model_type_performance(enhanced_models_summary)

            # Generate recommendations and alerts
            analysis.recommendations = self._generate_recommendations(analysis)
            analysis.alerts = self._generate_alerts(analysis)

            return analysis

        except Exception as e:
            self.logger.error(f"Failed to generate comprehensive report: {e}")
            return Step12EnhancedAnalysis()

    def _calculate_total_models_enhanced(self, enhanced_models_summary: Dict[str, Any]) -> int:
        """Calculate total number of models enhanced across all regimes."""
        total = 0
        for regime_data in enhanced_models_summary.values():
            if isinstance(regime_data, dict):
                models = regime_data.get('models', {})
                if isinstance(models, dict):
                    total += len(models)
                else:
                    total += 1
        return total

    def _calculate_overall_improvement(self, enhanced_models_summary: Dict[str, Any]) -> float:
        """Calculate overall accuracy improvement across all enhanced models."""
        improvements = []
        for regime_data in enhanced_models_summary.values():
            if isinstance(regime_data, dict):
                models = regime_data.get('models', {})
                if isinstance(models, dict):
                    for model_data in models.values():
                        if isinstance(model_data, dict):
                            original_acc = model_data.get('enhancement_metadata', {}).get('original_accuracy')
                            final_acc = model_data.get('enhancement_metadata', {}).get('final_accuracy')
                            if original_acc and final_acc and original_acc != 'N/A':
                                try:
                                    improvement = (float(final_acc) - float(original_acc)) / float(original_acc) * 100
                                    improvements.append(improvement)
                                except (ValueError, TypeError):
                                    continue
        return np.mean(improvements) if improvements else 0.0

    def _analyze_hyperparameter_optimization(self, hpo_metrics: Dict[str, Any]) -> HyperparameterOptimizationMetrics:
        """Analyze hyperparameter optimization performance."""
        metrics = HyperparameterOptimizationMetrics()

        # Extract HPO performance data
        metrics.total_trials = hpo_metrics.get('total_trials', 0)
        metrics.completed_trials = hpo_metrics.get('completed_trials', 0)
        metrics.best_score = hpo_metrics.get('best_score', 0.0)
        metrics.best_params = hpo_metrics.get('best_params', {})
        metrics.optimization_time = hpo_metrics.get('optimization_time', 0.0)
        metrics.convergence_score = hpo_metrics.get('convergence_score', 0.0)
        metrics.early_stopping_trials = hpo_metrics.get('early_stopping_trials', 0)
        metrics.pruning_efficiency = hpo_metrics.get('pruning_efficiency', 0.0)

        return metrics

    def _analyze_feature_selection(self, enhanced_models_summary: Dict[str, Any]) -> FeatureSelectionMetrics:
        """Analyze feature selection performance."""
        metrics = FeatureSelectionMetrics()

        feature_counts = []
        selection_scores = []
        feature_importances = {}

        for regime_data in enhanced_models_summary.values():
            if isinstance(regime_data, dict):
                models = regime_data.get('models', {})
                if isinstance(models, dict):
                    for model_data in models.values():
                        if isinstance(model_data, dict):
                            metadata = model_data.get('enhancement_metadata', {})

                            # Original vs selected feature counts
                            original_count = metadata.get('original_feature_count', 0)
                            selected_count = metadata.get('selected_feature_count', 0)

                            if original_count > 0:
                                feature_counts.append((original_count, selected_count))

                            # Selection scores
                            if 'feature_selection_score' in metadata:
                                selection_scores.append(metadata['feature_selection_score'])

                            # Feature importance scores
                            if 'feature_importance' in metadata:
                                for feature, importance in metadata['feature_importance'].items():
                                    if feature in feature_importances:
                                        feature_importances[feature] = max(feature_importances[feature], importance)
                                    else:
                                        feature_importances[feature] = importance

        if feature_counts:
            original_avg = np.mean([x[0] for x in feature_counts])
            selected_avg = np.mean([x[1] for x in feature_counts])
            metrics.original_feature_count = int(original_avg)
            metrics.selected_feature_count = int(selected_avg)
            metrics.selection_score = np.mean(selection_scores) if selection_scores else 0.0

        metrics.feature_importance_scores = dict(sorted(feature_importances.items(),
                                                       key=lambda x: x[1], reverse=True)[:20])

        return metrics

    def _analyze_model_enhancement(self, enhanced_models_summary: Dict[str, Any]) -> ModelEnhancementMetrics:
        """Analyze model enhancement performance."""
        metrics = ModelEnhancementMetrics()

        accuracy_improvements = []
        enhancement_times = []
        model_types = []

        for regime_data in enhanced_models_summary.values():
            if isinstance(regime_data, dict):
                models = regime_data.get('models', {})
                if isinstance(models, dict):
                    for model_name, model_data in models.items():
                        if isinstance(model_data, dict):
                            metadata = model_data.get('enhancement_metadata', {})

                            original_acc = metadata.get('original_accuracy')
                            final_acc = metadata.get('final_accuracy')

                            if original_acc and final_acc and original_acc != 'N/A':
                                try:
                                    original_acc = float(original_acc)
                                    final_acc = float(final_acc)
                                    improvement = (final_acc - original_acc) / original_acc * 100
                                    accuracy_improvements.append(improvement)
                                    model_types.append(model_name)
                                except (ValueError, TypeError):
                                    continue

                            # Enhancement time
                            if 'enhancement_time' in metadata:
                                enhancement_times.append(metadata['enhancement_time'])

        if accuracy_improvements:
            metrics.improvement_percentage = np.mean(accuracy_improvements)
            # Find most common model type
            if model_types:
                from collections import Counter
                metrics.model_type = Counter(model_types).most_common(1)[0][0]

        if enhancement_times:
            metrics.enhancement_time = np.mean(enhancement_times)

        return metrics

    def _analyze_regime_metrics(self, enhanced_models_summary: Dict[str, Any]) -> List[RegimeOptimizationMetrics]:
        """Analyze regime-specific optimization metrics."""
        regime_metrics = []

        for regime_name, regime_data in enhanced_models_summary.items():
            if isinstance(regime_data, dict):
                metrics = RegimeOptimizationMetrics(regime_name=regime_name)

                # Count models enhanced in this regime
                models = regime_data.get('models', {})
                if isinstance(models, dict):
                    metrics.models_enhanced = len(models)

                # Get regime-specific improvements
                regime_improvements = {}
                for model_name, model_data in models.items():
                    if isinstance(model_data, dict):
                        metadata = model_data.get('enhancement_metadata', {})
                        original_acc = metadata.get('original_accuracy')
                        final_acc = metadata.get('final_accuracy')

                        if original_acc and final_acc and original_acc != 'N/A':
                            try:
                                improvement = (float(final_acc) - float(original_acc)) / float(original_acc) * 100
                                regime_improvements[model_name] = improvement
                            except (ValueError, TypeError):
                                continue

                metrics.regime_specific_improvements = regime_improvements
                metrics.optimization_efficiency = np.mean(list(regime_improvements.values())) if regime_improvements else 0.0

                regime_metrics.append(metrics)

        return regime_metrics

    def _analyze_hardware_optimization(self, hardware_metrics: Dict[str, Any]) -> HardwareOptimizationMetrics:
        """Analyze hardware optimization performance."""
        metrics = HardwareOptimizationMetrics()

        metrics.gpu_utilization = hardware_metrics.get('gpu_utilization', 0.0)
        metrics.m1_gpu_available = hardware_metrics.get('m1_gpu_available', False)
        metrics.memory_efficiency = hardware_metrics.get('memory_efficiency', 0.0)
        metrics.parallel_processing_efficiency = hardware_metrics.get('parallel_processing_efficiency', 0.0)
        metrics.vectorized_operations_count = hardware_metrics.get('vectorized_operations_count', 0)
        metrics.matrix_operations_speedup = hardware_metrics.get('matrix_operations_speedup', 0.0)
        metrics.batch_processing_time = hardware_metrics.get('batch_processing_time', 0.0)

        return metrics

    def _analyze_advanced_optimizations(self, enhanced_models_summary: Dict[str, Any]) -> AdvancedOptimizationMetrics:
        """Analyze advanced optimization techniques applied."""
        metrics = AdvancedOptimizationMetrics()

        # Check which optimizations were applied across models
        quantization_count = 0
        pruning_count = 0
        distillation_count = 0
        total_models = 0

        for regime_data in enhanced_models_summary.values():
            if isinstance(regime_data, dict):
                models = regime_data.get('models', {})
                if isinstance(models, dict):
                    for model_data in models.values():
                        if isinstance(model_data, dict):
                            metadata = model_data.get('enhancement_metadata', {})
                            optimizations = metadata.get('applied_optimizations', [])

                            total_models += 1
                            if 'quantization' in optimizations:
                                quantization_count += 1
                            if 'pruning' in optimizations:
                                pruning_count += 1
                            if 'distillation' in optimizations:
                                distillation_count += 1

        if total_models > 0:
            metrics.quantization_applied = quantization_count > 0
            metrics.pruning_applied = pruning_count > 0
            metrics.distillation_applied = distillation_count > 0

        # Calculate optimization ratios
        if total_models > 0:
            metrics.model_size_reduction = np.mean([
                metadata.get('model_size_reduction', 0.0)
                for regime_data in enhanced_models_summary.values()
                if isinstance(regime_data, dict)
                for model_data in regime_data.get('models', {}).values()
                if isinstance(model_data, dict)
                for metadata in [model_data.get('enhancement_metadata', {})]
                if 'model_size_reduction' in metadata
            ] or [0.0])

        return metrics

    def _analyze_parallel_processing(self, parallel_metrics: Dict[str, Any]) -> ParallelProcessingMetrics:
        """Analyze parallel processing performance."""
        metrics = ParallelProcessingMetrics()

        metrics.total_regimes = parallel_metrics.get('total_regimes', 0)
        metrics.concurrent_regimes = parallel_metrics.get('concurrent_regimes', 0)
        metrics.total_processing_time = parallel_metrics.get('total_processing_time', 0.0)
        metrics.average_regime_time = parallel_metrics.get('average_regime_time', 0.0)
        metrics.processing_efficiency = parallel_metrics.get('processing_efficiency', 0.0)
        metrics.memory_usage_pattern = parallel_metrics.get('memory_usage_pattern', '')
        metrics.bottleneck_analysis = parallel_metrics.get('bottleneck_analysis', '')

        return metrics

    def _analyze_model_type_performance(self, enhanced_models_summary: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Analyze performance by model type."""
        model_performance = {}

        for regime_data in enhanced_models_summary.values():
            if isinstance(regime_data, dict):
                models = regime_data.get('models', {})
                if isinstance(models, dict):
                    for model_name, model_data in models.items():
                        if isinstance(model_data, dict):
                            metadata = model_data.get('enhancement_metadata', {})

                            if model_name not in model_performance:
                                model_performance[model_name] = {
                                    'count': 0,
                                    'avg_improvement': 0.0,
                                    'avg_accuracy': 0.0,
                                    'total_time': 0.0
                                }

                            original_acc = metadata.get('original_accuracy')
                            final_acc = metadata.get('final_accuracy')

                            if original_acc and final_acc and original_acc != 'N/A':
                                try:
                                    original_acc = float(original_acc)
                                    final_acc = float(final_acc)
                                    improvement = (final_acc - original_acc) / original_acc * 100

                                    perf = model_performance[model_name]
                                    perf['count'] += 1
                                    perf['avg_improvement'] = (perf['avg_improvement'] * (perf['count'] - 1) + improvement) / perf['count']
                                    perf['avg_accuracy'] = (perf['avg_accuracy'] * (perf['count'] - 1) + final_acc) / perf['count']
                                except (ValueError, TypeError):
                                    continue

                            if 'enhancement_time' in metadata:
                                perf['total_time'] += metadata['enhancement_time']

        return model_performance

    def _generate_recommendations(self, analysis: Step12EnhancedAnalysis) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []

        # Hardware recommendations
        if analysis.hardware_metrics.gpu_utilization < 70.0:
            recommendations.append("Consider optimizing GPU utilization - current usage is below 70%")

        if not analysis.hardware_metrics.m1_gpu_available:
            recommendations.append("M1 GPU not detected - consider enabling hardware acceleration for better performance")

        # Model enhancement recommendations
        if analysis.overall_accuracy_improvement < 5.0:
            recommendations.append("Overall accuracy improvement is low (<5%) - consider reviewing optimization parameters")

        # Feature selection recommendations
        if analysis.feature_selection_metrics.selection_score < 0.7:
            recommendations.append("Feature selection effectiveness is low - consider alternative feature selection methods")

        # Parallel processing recommendations
        if analysis.parallel_processing_metrics.processing_efficiency < 80.0:
            recommendations.append("Parallel processing efficiency is suboptimal - consider adjusting concurrent regime processing")

        return recommendations

    def _generate_alerts(self, analysis: Step12EnhancedAnalysis) -> List[str]:
        """Generate alerts based on analysis."""
        alerts = []

        # Critical alerts
        if analysis.total_models_enhanced == 0:
            alerts.append("🚨 CRITICAL: No models were successfully enhanced")

        if analysis.hardware_metrics.memory_efficiency < 50.0:
            alerts.append("🚨 CRITICAL: Memory efficiency is very low - potential memory issues")

        # Warning alerts
        if analysis.hpo_metrics.convergence_score < 0.5:
            alerts.append("⚠️ WARNING: Hyperparameter optimization convergence is poor")

        if len(analysis.feature_selection_metrics.feature_importance_scores) < 5:
            alerts.append("⚠️ WARNING: Very few features selected - potential overfitting risk")

        return alerts

    def save_comprehensive_report(self,
                                report_data: Step12EnhancedAnalysis,
                                symbol: str,
                                exchange: str,
                                timeframe: str) -> List[str]:
        """
        Save comprehensive Step12 analysis report in multiple formats.

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
                'step': 'step12_final_parameters_optimization',
                'timestamp': report_data.timestamp,
                'symbol': symbol,
                'exchange': exchange,
                'timeframe': timeframe,
                'analysis': {
                    'total_models_enhanced': report_data.total_models_enhanced,
                    'total_regimes_processed': report_data.total_regimes_processed,
                    'overall_accuracy_improvement': report_data.overall_accuracy_improvement,
                    'total_optimization_time': report_data.total_optimization_time,
                    'hyperparameter_optimization': {
                        'total_trials': report_data.hpo_metrics.total_trials,
                        'best_score': report_data.hpo_metrics.best_score,
                        'optimization_time': report_data.hpo_metrics.optimization_time,
                        'convergence_score': report_data.hpo_metrics.convergence_score
                    },
                    'feature_selection': {
                        'original_features': report_data.feature_selection_metrics.original_feature_count,
                        'selected_features': report_data.feature_selection_metrics.selected_feature_count,
                        'selection_score': report_data.feature_selection_metrics.selection_score
                    },
                    'model_enhancement': {
                        'improvement_percentage': report_data.model_enhancement_metrics.improvement_percentage,
                        'average_enhancement_time': report_data.model_enhancement_metrics.enhancement_time
                    },
                    'hardware_optimization': {
                        'gpu_utilization': report_data.hardware_metrics.gpu_utilization,
                        'memory_efficiency': report_data.hardware_metrics.memory_efficiency,
                        'parallel_efficiency': report_data.hardware_metrics.parallel_processing_efficiency
                    },
                    'regime_performance': [
                        {
                            'regime': regime.regime_name,
                            'models_enhanced': regime.models_enhanced,
                            'optimization_efficiency': regime.optimization_efficiency
                        } for regime in report_data.regime_metrics
                    ],
                    'model_type_performance': report_data.model_type_performance,
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
                    step_name="step12_final_parameters_optimization",
                    report_type="comprehensive_analysis",
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
                    step_name="step12_final_parameters_optimization",
                    report_type="analysis_summary",
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
                    step_name="step12_final_parameters_optimization",
                    report_type="metrics_summary",
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
                                report_data: Step12EnhancedAnalysis,
                                symbol: str,
                                exchange: str,
                                timeframe: str) -> str:
        """Generate Markdown report content."""
        markdown = f"""# Step12 Enhanced Final Parameters Optimization Analysis Report

**Generated:** {report_data.timestamp}
**Symbol:** {symbol}
**Exchange:** {exchange}
**Timeframe:** {timeframe}

## Executive Summary

This report provides comprehensive analysis of the Final Parameters Optimization process for {symbol} on {exchange}.

### Key Metrics
- **Models Enhanced:** {report_data.total_models_enhanced}
- **Regimes Processed:** {report_data.total_regimes_processed}
- **Overall Accuracy Improvement:** {report_data.overall_accuracy_improvement:.2f}%
- **Total Optimization Time:** {report_data.total_optimization_time:.2f}s

## Hyperparameter Optimization Analysis

- **Total Trials:** {report_data.hpo_metrics.total_trials}
- **Completed Trials:** {report_data.hpo_metrics.completed_trials}
- **Best Score:** {report_data.hpo_metrics.best_score:.4f}
- **Optimization Time:** {report_data.hpo_metrics.optimization_time:.2f}s
- **Convergence Score:** {report_data.hpo_metrics.convergence_score:.4f}
- **Early Stopping Trials:** {report_data.hpo_metrics.early_stopping_trials}
- **Pruning Efficiency:** {report_data.hpo_metrics.pruning_efficiency:.2f}%

## Feature Selection Analysis

- **Original Features:** {report_data.feature_selection_metrics.original_feature_count}
- **Selected Features:** {report_data.feature_selection_metrics.selected_feature_count}
- **Selection Score:** {report_data.feature_selection_metrics.selection_score:.4f}
- **Correlation Reduction:** {report_data.feature_selection_metrics.correlation_reduction:.2f}%
- **VIF Improvement:** {report_data.feature_selection_metrics.vif_improvement:.2f}%

## Model Enhancement Analysis

- **Average Improvement:** {report_data.model_enhancement_metrics.improvement_percentage:.2f}%
- **Average Enhancement Time:** {report_data.model_enhancement_metrics.enhancement_time:.2f}s
- **Training Speedup:** {report_data.model_enhancement_metrics.training_speedup:.2f}x
- **Memory Usage:** {report_data.model_enhancement_metrics.memory_usage_mb:.1f} MB

## Hardware Optimization Analysis

- **GPU Utilization:** {report_data.hardware_metrics.gpu_utilization:.1f}%
- **M1 GPU Available:** {report_data.hardware_metrics.m1_gpu_available}
- **Memory Efficiency:** {report_data.hardware_metrics.memory_efficiency:.2f}%
- **Parallel Processing Efficiency:** {report_data.hardware_metrics.parallel_processing_efficiency:.2f}%
- **Matrix Operations Speedup:** {report_data.hardware_metrics.matrix_operations_speedup:.2f}x

## Regime-Specific Performance

"""

        # Add regime performance table
        if report_data.regime_metrics:
            markdown += "| Regime | Models Enhanced | Optimization Efficiency |\n"
            markdown += "|--------|----------------|------------------------|\n"
            for regime in report_data.regime_metrics:
                markdown += f"| {regime.regime_name} | {regime.models_enhanced} | {regime.optimization_efficiency:.2f}% |\n"

        # Add model type performance
        markdown += "\n## Model Type Performance\n\n"
        if report_data.model_type_performance:
            markdown += "| Model Type | Count | Avg Improvement | Avg Accuracy |\n"
            markdown += "|------------|-------|----------------|--------------|\n"
            for model_type, perf in report_data.model_type_performance.items():
                markdown += f"| {model_type} | {perf['count']} | {perf['avg_improvement']:.2f}% | {perf['avg_accuracy']:.4f} |\n"

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

    def _generate_csv_metrics(self, report_data: Step12EnhancedAnalysis) -> str:
        """Generate CSV metrics content."""
        # Create summary metrics
        summary_data = {
            'metric': ['total_models_enhanced', 'total_regimes_processed', 'overall_accuracy_improvement',
                      'total_optimization_time', 'hpo_total_trials', 'hpo_best_score', 'hpo_optimization_time',
                      'feature_original_count', 'feature_selected_count', 'feature_selection_score',
                      'model_improvement_percentage', 'model_enhancement_time', 'gpu_utilization',
                      'memory_efficiency', 'parallel_efficiency'],
            'value': [report_data.total_models_enhanced, report_data.total_regimes_processed,
                     report_data.overall_accuracy_improvement, report_data.total_optimization_time,
                     report_data.hpo_metrics.total_trials, report_data.hpo_metrics.best_score,
                     report_data.hpo_metrics.optimization_time,
                     report_data.feature_selection_metrics.original_feature_count,
                     report_data.feature_selection_metrics.selected_feature_count,
                     report_data.feature_selection_metrics.selection_score,
                     report_data.model_enhancement_metrics.improvement_percentage,
                     report_data.model_enhancement_metrics.enhancement_time,
                     report_data.hardware_metrics.gpu_utilization,
                     report_data.hardware_metrics.memory_efficiency,
                     report_data.hardware_metrics.parallel_processing_efficiency]
        }

        df = pd.DataFrame(summary_data)
        return df.to_csv(index=False)

    def _generate_visualizations(self,
                               report_data: Step12EnhancedAnalysis,
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

            # 1. Model Performance Comparison
            if report_data.model_type_performance:
                plt.figure(figsize=(12, 8))
                model_types = list(report_data.model_type_performance.keys())
                improvements = [perf['avg_improvement'] for perf in report_data.model_type_performance.values()]

                bars = plt.bar(model_types, improvements, color='skyblue', alpha=0.8)
                plt.title('Model Type Performance Improvement', fontsize=16, fontweight='bold')
                plt.xlabel('Model Type', fontsize=12)
                plt.ylabel('Average Improvement (%)', fontsize=12)
                plt.xticks(rotation=45, ha='right')
                plt.grid(True, alpha=0.3)

                # Add value labels on bars
                for bar, improvement in zip(bars, improvements):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                           '.1f', ha='center', va='bottom', fontsize=10)

                plt.tight_layout()

                viz_path = self.save_training_report(
                    data=plt.gcf(),
                    step_name="step12_final_parameters_optimization",
                    report_type="model_performance",
                    symbol=symbol,
                    timeframe=timeframe,
                    file_format="png"
                )
                saved_files.append(viz_path)
                plt.close()

            # 2. Regime Performance Distribution
            if report_data.regime_metrics:
                plt.figure(figsize=(12, 8))
                regime_names = [r.regime_name for r in report_data.regime_metrics]
                efficiencies = [r.optimization_efficiency for r in report_data.regime_metrics]

                plt.bar(regime_names, efficiencies, color='lightgreen', alpha=0.8)
                plt.title('Regime-Specific Optimization Efficiency', fontsize=16, fontweight='bold')
                plt.xlabel('Regime', fontsize=12)
                plt.ylabel('Optimization Efficiency (%)', fontsize=12)
                plt.xticks(rotation=45, ha='right')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()

                viz_path = self.save_training_report(
                    data=plt.gcf(),
                    step_name="step12_final_parameters_optimization",
                    report_type="regime_performance",
                    symbol=symbol,
                    timeframe=timeframe,
                    file_format="png"
                )
                saved_files.append(viz_path)
                plt.close()

            # 3. Hardware Utilization Dashboard
            plt.figure(figsize=(15, 10))

            # Subplot 1: GPU and Memory
            plt.subplot(2, 2, 1)
            metrics = ['GPU Utilization', 'Memory Efficiency', 'Parallel Efficiency']
            values = [report_data.hardware_metrics.gpu_utilization,
                     report_data.hardware_metrics.memory_efficiency,
                     report_data.hardware_metrics.parallel_processing_efficiency]

            bars = plt.bar(metrics, values, color=['red', 'blue', 'green'], alpha=0.7)
            plt.title('Hardware Optimization Metrics', fontsize=14, fontweight='bold')
            plt.ylabel('Percentage (%)', fontsize=12)
            plt.ylim(0, 100)
            plt.grid(True, alpha=0.3)

            # Subplot 2: Processing Times
            plt.subplot(2, 2, 2)
            times = ['HPO Time', 'Feature Selection', 'Model Enhancement']
            time_values = [report_data.hpo_metrics.optimization_time,
                          report_data.feature_selection_metrics.selection_time,
                          report_data.model_enhancement_metrics.enhancement_time]

            plt.bar(times, time_values, color='orange', alpha=0.7)
            plt.title('Processing Times', fontsize=14, fontweight='bold')
            plt.ylabel('Time (seconds)', fontsize=12)
            plt.grid(True, alpha=0.3)

            # Subplot 3: Model Enhancement Results
            plt.subplot(2, 2, 3)
            enhancement_metrics = ['Accuracy Improvement', 'Feature Reduction', 'Training Speedup']
            enhancement_values = [report_data.model_enhancement_metrics.improvement_percentage,
                                (1 - report_data.feature_selection_metrics.selected_feature_count /
                                 max(report_data.feature_selection_metrics.original_feature_count, 1)) * 100,
                                report_data.model_enhancement_metrics.training_speedup]

            plt.bar(enhancement_metrics, enhancement_values, color='purple', alpha=0.7)
            plt.title('Model Enhancement Results', fontsize=14, fontweight='bold')
            plt.ylabel('Percentage (%)', fontsize=12)
            plt.grid(True, alpha=0.3)

            # Subplot 4: Advanced Optimizations
            plt.subplot(2, 2, 4)
            advanced_metrics = ['Quantization', 'Pruning', 'Distillation', 'Model Compression']
            advanced_values = [1 if report_data.advanced_optimization_metrics.quantization_applied else 0,
                             1 if report_data.advanced_optimization_metrics.pruning_applied else 0,
                             1 if report_data.advanced_optimization_metrics.distillation_applied else 0,
                             report_data.advanced_optimization_metrics.compression_ratio]

            colors = ['green' if v > 0 else 'red' for v in advanced_values[:-1]] + ['blue']
            plt.bar(advanced_metrics, advanced_values, color=colors, alpha=0.7)
            plt.title('Advanced Optimization Status', fontsize=14, fontweight='bold')
            plt.ylabel('Applied / Ratio', fontsize=12)
            plt.grid(True, alpha=0.3)

            plt.suptitle('Step12 Final Parameters Optimization Dashboard', fontsize=16, fontweight='bold')
            plt.tight_layout()

            viz_path = self.save_training_report(
                data=plt.gcf(),
                step_name="step12_final_parameters_optimization",
                report_type="optimization_dashboard",
                symbol=symbol,
                timeframe=timeframe,
                file_format="png"
            )
            saved_files.append(viz_path)
            plt.close()

        except Exception as e:
            self.logger.error(f"Failed to generate visualizations: {e}")

        return saved_files
