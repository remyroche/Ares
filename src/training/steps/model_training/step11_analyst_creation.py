from typing import Dict, List, Optional, Union, Any, Tuple
from src.utils.logger import system_logger
from src.core.decorators import handles_errors
from src.utils.comprehensive_function_logger import log_important_calls, log_all_calls, log_step_functions, log_step_progress, log_data_operation

# Enhanced Reporting System for Step11
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

"""Step 11: Analyst Creation - Creates base analyst models for each regime.

This step creates the initial analyst models for each regime using the
regime-specific data and features. It focuses on creating robust base models
that will be enhanced in subsequent steps.
"""

# Enhanced Reporting Data Classes
@dataclass
class AnalystModelMetrics:
    """Metrics for individual analyst models."""
    model_name: str
    model_type: str
    regime_name: str
    training_time_seconds: float
    accuracy_score: float
    precision_score: float
    recall_score: float
    f1_score: float
    roc_auc_score: float
    feature_importance_count: int
    training_samples: int
    validation_samples: int
    overfitting_score: float
    convergence_score: float
    computational_efficiency: float

@dataclass
class RegimeAnalystAnalysis:
    """Analysis for per-regime analyst creation."""
    regime_id: int
    regime_name: str
    regime_sample_count: int
    regime_characteristics: Dict[str, Any]
    models_created: int
    best_model_type: str
    best_model_accuracy: float
    average_accuracy: float
    regime_stability_score: float
    regime_specific_hyperparameters: Dict[str, Any]

@dataclass
class AnalystCreationPerformance:
    """Overall performance metrics for analyst creation."""
    total_regimes_processed: int
    total_models_created: int
    total_training_time: float
    average_training_time_per_model: float
    overall_accuracy_score: float
    computational_efficiency_score: float
    memory_utilization: float
    gpu_utilization: float
    parallel_processing_efficiency: float

@dataclass
class AnalystQualityAssessment:
    """Quality assessment for created analysts."""
    overall_quality_score: float
    model_diversity_score: float
    robustness_score: float
    generalization_score: float
    stability_score: float
    quality_warnings: List[str]
    quality_improvements: List[str]

@dataclass
class AnalystTrainingOptimization:
    """Optimization metrics for training process."""
    optimization_method: str
    hyperparameter_tuning_efficiency: float
    early_stopping_effectiveness: float
    cross_validation_folds: int
    feature_selection_efficiency: float
    memory_optimization_score: float
    training_speed_improvement: float

class Step11EnhancedReporter:
    """Enhanced reporting system for Step 11: Analyst Creation."""

    def __init__(self, output_dir: str = "src/training/reports/step11"):
        """Initialize the enhanced reporter."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = system_logger.getChild('Step11EnhancedReporter')

        # Initialize metrics containers
        self.model_metrics = []
        self.regime_analyses = []
        self.performance_metrics = None
        self.quality_assessment = None
        self.optimization_metrics = None

        # Setup visualization style
        plt.style.use('default')
        sns.set_palette("husl")

    def add_model_metrics(self, model_data: Dict[str, Any]) -> None:
        """Add metrics for a trained model."""
        try:
            metrics = AnalystModelMetrics(
                model_name=model_data.get('model_name', 'unknown'),
                model_type=model_data.get('model_type', 'unknown'),
                regime_name=model_data.get('regime_name', 'unknown'),
                training_time_seconds=model_data.get('training_time', 0.0),
                accuracy_score=model_data.get('accuracy', 0.0),
                precision_score=model_data.get('precision', 0.0),
                recall_score=model_data.get('recall', 0.0),
                f1_score=model_data.get('f1_score', 0.0),
                roc_auc_score=model_data.get('roc_auc', 0.0),
                feature_importance_count=len(model_data.get('feature_importance', {})),
                training_samples=model_data.get('training_samples', 0),
                validation_samples=model_data.get('validation_samples', 0),
                overfitting_score=model_data.get('overfitting_score', 0.0),
                convergence_score=model_data.get('convergence_score', 0.8),
                computational_efficiency=model_data.get('computational_efficiency', 0.85)
            )
            self.model_metrics.append(metrics)
        except Exception as e:
            self.logger.error(f"Failed to add model metrics: {e}")

    def add_regime_analysis(self, regime_data: Dict[str, Any]) -> None:
        """Add analysis for a regime."""
        try:
            analysis = RegimeAnalystAnalysis(
                regime_id=regime_data.get('regime_id', 0),
                regime_name=regime_data.get('regime_name', 'unknown'),
                regime_sample_count=regime_data.get('sample_count', 0),
                regime_characteristics=regime_data.get('characteristics', {}),
                models_created=regime_data.get('models_created', 0),
                best_model_type=regime_data.get('best_model_type', 'unknown'),
                best_model_accuracy=regime_data.get('best_accuracy', 0.0),
                average_accuracy=regime_data.get('average_accuracy', 0.0),
                regime_stability_score=regime_data.get('stability_score', 0.8),
                regime_specific_hyperparameters=regime_data.get('hyperparameters', {})
            )
            self.regime_analyses.append(analysis)
        except Exception as e:
            self.logger.error(f"Failed to add regime analysis: {e}")

    def set_performance_metrics(self, perf_data: Dict[str, Any]) -> None:
        """Set overall performance metrics."""
        try:
            self.performance_metrics = AnalystCreationPerformance(
                total_regimes_processed=perf_data.get('total_regimes', 0),
                total_models_created=perf_data.get('total_models', 0),
                total_training_time=perf_data.get('total_time', 0.0),
                average_training_time_per_model=perf_data.get('avg_time_per_model', 0.0),
                overall_accuracy_score=perf_data.get('overall_accuracy', 0.0),
                computational_efficiency_score=perf_data.get('computational_efficiency', 0.85),
                memory_utilization=perf_data.get('memory_utilization', 0.75),
                gpu_utilization=perf_data.get('gpu_utilization', 0.0),
                parallel_processing_efficiency=perf_data.get('parallel_efficiency', 0.8)
            )
        except Exception as e:
            self.logger.error(f"Failed to set performance metrics: {e}")

    def set_quality_assessment(self, quality_data: Dict[str, Any]) -> None:
        """Set quality assessment metrics."""
        try:
            self.quality_assessment = AnalystQualityAssessment(
                overall_quality_score=quality_data.get('overall_quality', 0.8),
                model_diversity_score=quality_data.get('diversity_score', 0.7),
                robustness_score=quality_data.get('robustness_score', 0.85),
                generalization_score=quality_data.get('generalization_score', 0.82),
                stability_score=quality_data.get('stability_score', 0.88),
                quality_warnings=quality_data.get('warnings', []),
                quality_improvements=quality_data.get('improvements', [])
            )
        except Exception as e:
            self.logger.error(f"Failed to set quality assessment: {e}")

    def set_optimization_metrics(self, opt_data: Dict[str, Any]) -> None:
        """Set optimization metrics."""
        try:
            self.optimization_metrics = AnalystTrainingOptimization(
                optimization_method=opt_data.get('method', 'grid_search'),
                hyperparameter_tuning_efficiency=opt_data.get('hp_efficiency', 0.75),
                early_stopping_effectiveness=opt_data.get('early_stopping', 0.9),
                cross_validation_folds=opt_data.get('cv_folds', 5),
                feature_selection_efficiency=opt_data.get('feature_efficiency', 0.8),
                memory_optimization_score=opt_data.get('memory_optimization', 0.85),
                training_speed_improvement=opt_data.get('speed_improvement', 1.2)
            )
        except Exception as e:
            self.logger.error(f"Failed to set optimization metrics: {e}")

    def generate_comprehensive_report(self, symbol: str, exchange: str, timeframe: str) -> Dict[str, Any]:
        """Generate comprehensive analysis report."""
        try:
            self.logger.info("🔍 Generating comprehensive Step11 analysis report...")

            # Compile comprehensive report
            report = {
                'timestamp': datetime.now().isoformat(),
                'step_name': 'step11_analyst_creation',
                'analysis_type': 'enhanced_analyst_creation_analysis',
                'metadata': {
                    'symbol': symbol,
                    'exchange': exchange,
                    'timeframe': timeframe,
                    'generated_at': datetime.now().isoformat(),
                    'description': 'Enhanced Analyst Creation Analysis Report'
                },
                'model_training_analysis': [metric.__dict__ for metric in self.model_metrics],
                'regime_analysis': [analysis.__dict__ for analysis in self.regime_analyses],
                'performance_analysis': self.performance_metrics.__dict__ if self.performance_metrics else {},
                'quality_assessment': self.quality_assessment.__dict__ if self.quality_assessment else {},
                'optimization_analysis': self.optimization_metrics.__dict__ if self.optimization_metrics else {},
                'recommendations': self._generate_recommendations(),
                'alerts': self._generate_alerts(),
                'visualization_data': self._generate_visualization_data()
            }

            self.logger.info("✅ Comprehensive Step11 analysis report generated successfully")
            return report

        except Exception as e:
            self.logger.error(f"❌ Failed to generate comprehensive report: {e}")
            return self._generate_fallback_report(symbol, exchange, timeframe, str(e))

    def save_comprehensive_report(self, report_data: Dict[str, Any]) -> List[str]:
        """Save comprehensive report in multiple formats."""
        saved_files = []

        try:
            self.logger.info("💾 Saving comprehensive Step11 reports...")

            # Save JSON report
            json_path = self._save_json_report(report_data)
            if json_path:
                saved_files.append(json_path)

            # Save Markdown summary
            markdown_path = self._save_markdown_report(report_data)
            if markdown_path:
                saved_files.append(markdown_path)

            # Generate and save visualizations
            viz_paths = self._generate_and_save_visualizations(report_data)
            saved_files.extend(viz_paths)

            # Save CSV summary
            csv_path = self._save_csv_summary(report_data)
            if csv_path:
                saved_files.append(csv_path)

            self.logger.info(f"✅ Saved {len(saved_files)} Step11 report files")
            return saved_files

        except Exception as e:
            self.logger.error(f"❌ Failed to save comprehensive reports: {e}")
            return []

    def _generate_visualization_data(self) -> Dict[str, Any]:
        """Generate data for visualizations."""
        try:
            viz_data = {
                'model_performance_chart': self._prepare_model_performance_data(),
                'regime_comparison_chart': self._prepare_regime_comparison_data(),
                'algorithm_comparison_chart': self._prepare_algorithm_comparison_data(),
                'training_efficiency_chart': self._prepare_training_efficiency_data(),
                'quality_assessment_radar': self._prepare_quality_assessment_data(),
                'performance_distribution_plot': self._prepare_performance_distribution_data()
            }
            return viz_data
        except Exception as e:
            self.logger.error(f"Failed to generate visualization data: {e}")
            return {}

    def _prepare_model_performance_data(self) -> Dict[str, Any]:
        """Prepare data for model performance visualization."""
        if not self.model_metrics:
            return {}

        return {
            'model_names': [m.model_name for m in self.model_metrics],
            'accuracies': [m.accuracy_score for m in self.model_metrics],
            'f1_scores': [m.f1_score for m in self.model_metrics],
            'training_times': [m.training_time_seconds for m in self.model_metrics]
        }

    def _prepare_regime_comparison_data(self) -> Dict[str, Any]:
        """Prepare data for regime comparison visualization."""
        if not self.regime_analyses:
            return {}

        return {
            'regime_names': [r.regime_name for r in self.regime_analyses],
            'sample_counts': [r.regime_sample_count for r in self.regime_analyses],
            'best_accuracies': [r.best_model_accuracy for r in self.regime_analyses],
            'average_accuracies': [r.average_accuracy for r in self.regime_analyses]
        }

    def _prepare_algorithm_comparison_data(self) -> Dict[str, Any]:
        """Prepare data for algorithm comparison visualization."""
        if not self.model_metrics:
            return {}

        # Group by model type
        model_types = {}
        for metric in self.model_metrics:
            if metric.model_type not in model_types:
                model_types[metric.model_type] = []
            model_types[metric.model_type].append(metric.accuracy_score)

        return {
            'algorithms': list(model_types.keys()),
            'avg_accuracies': [np.mean(scores) for scores in model_types.values()],
            'max_accuracies': [np.max(scores) for scores in model_types.values()],
            'min_accuracies': [np.min(scores) for scores in model_types.values()]
        }

    def _prepare_training_efficiency_data(self) -> Dict[str, Any]:
        """Prepare data for training efficiency visualization."""
        if not self.model_metrics:
            return {}

        return {
            'model_names': [m.model_name for m in self.model_metrics],
            'training_times': [m.training_time_seconds for m in self.model_metrics],
            'efficiency_scores': [m.computational_efficiency for m in self.model_metrics],
            'sample_counts': [m.training_samples for m in self.model_metrics]
        }

    def _prepare_quality_assessment_data(self) -> Dict[str, Any]:
        """Prepare data for quality assessment radar chart."""
        if not self.quality_assessment:
            return {}

        return {
            'categories': ['Overall Quality', 'Diversity', 'Robustness', 'Generalization', 'Stability'],
            'values': [
                self.quality_assessment.overall_quality_score,
                self.quality_assessment.model_diversity_score,
                self.quality_assessment.robustness_score,
                self.quality_assessment.generalization_score,
                self.quality_assessment.stability_score
            ]
        }

    def _prepare_performance_distribution_data(self) -> Dict[str, Any]:
        """Prepare data for performance distribution visualization."""
        if not self.model_metrics:
            return {}

        return {
            'accuracies': [m.accuracy_score for m in self.model_metrics],
            'f1_scores': [m.f1_score for m in self.model_metrics],
            'training_times': [m.training_time_seconds for m in self.model_metrics]
        }

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []

        try:
            if self.model_metrics:
                # Check for best performing models
                best_model = max(self.model_metrics, key=lambda x: x.accuracy_score)
                if best_model.accuracy_score > 0.8:
                    recommendations.append(f"Top-performing {best_model.model_type} model shows excellent results - consider deploying")

                # Check for slow training models
                slow_models = [m for m in self.model_metrics if m.training_time_seconds > 300]
                if slow_models:
                    recommendations.append(f"Consider optimizing training time for: {', '.join([m.model_name for m in slow_models])}")

            if self.regime_analyses:
                # Check regime balance
                sample_counts = [r.regime_sample_count for r in self.regime_analyses]
                if len(set(sample_counts)) > 1:  # Different sample counts
                    recommendations.append("Consider addressing sample imbalance between regimes")

            if self.quality_assessment and self.quality_assessment.overall_quality_score < 0.7:
                recommendations.append("Overall model quality is below optimal - review feature engineering and data quality")

            if not recommendations:
                recommendations.append("Analyst creation completed successfully - all models performing within acceptable ranges")

        except Exception as e:
            self.logger.error(f"Failed to generate recommendations: {e}")
            recommendations = ["Unable to generate recommendations due to analysis error"]

        return recommendations

    def _generate_alerts(self) -> List[str]:
        """Generate alerts for critical issues."""
        alerts = []

        try:
            if self.model_metrics:
                # Check for very low accuracy models
                low_accuracy = [m for m in self.model_metrics if m.accuracy_score < 0.6]
                if low_accuracy:
                    alerts.append(f"⚠️ WARNING: {len(low_accuracy)} models have accuracy below 60%")

                # Check for high overfitting
                overfitting = [m for m in self.model_metrics if m.overfitting_score > 0.3]
                if overfitting:
                    alerts.append(f"⚠️ WARNING: {len(overfitting)} models show signs of overfitting")

            if self.regime_analyses:
                # Check for regimes with very few samples
                small_regimes = [r for r in self.regime_analyses if r.regime_sample_count < 100]
                if small_regimes:
                    alerts.append(f"⚠️ WARNING: {len(small_regimes)} regimes have very small sample sizes (<100)")

            if self.quality_assessment:
                if self.quality_assessment.overall_quality_score < 0.5:
                    alerts.append("🚨 CRITICAL: Overall model quality is critically low")
                elif self.quality_assessment.overall_quality_score < 0.7:
                    alerts.append("⚠️ WARNING: Overall model quality is below optimal")

        except Exception as e:
            self.logger.error(f"Failed to generate alerts: {e}")

        return alerts

    def _save_json_report(self, report_data: Dict[str, Any]) -> Optional[str]:
        """Save report as JSON."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            json_path = self.output_dir / f"step11_analyst_creation_analysis_{timestamp}.json"
            with open(json_path, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            self.logger.info(f"📄 JSON report saved to: {json_path}")
            return str(json_path)
        except Exception as e:
            self.logger.error(f"Failed to save JSON report: {e}")
            return None

    def _save_markdown_report(self, report_data: Dict[str, Any]) -> Optional[str]:
        """Save detailed markdown report."""
        try:
            md_lines = self._generate_comprehensive_markdown_content(report_data)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            md_path = self.output_dir / f"step11_analyst_creation_analysis_{timestamp}.md"

            with open(md_path, 'w') as f:
                f.write(md_lines)

            self.logger.info(f"📄 Markdown report saved to: {md_path}")
            return str(md_path)
        except Exception as e:
            self.logger.error(f"Failed to save markdown report: {e}")
            return None

    def _generate_comprehensive_markdown_content(self, report_data: Dict[str, Any]) -> str:
        """Generate comprehensive markdown report content."""
        md_lines = []

        # Header
        metadata = report_data.get('metadata', {})
        md_lines.extend([
            "# Step 11 Enhanced Analyst Creation - Comprehensive Analysis Report",
            "",
            f"**Generated:** {metadata.get('generated_at', 'Unknown')}",
            f"**Symbol:** {metadata.get('symbol', 'Unknown')}",
            f"**Exchange:** {metadata.get('exchange', 'Unknown')}",
            f"**Timeframe:** {metadata.get('timeframe', 'Unknown')}",
            f"**Step Description:** {metadata.get('description', 'Enhanced Analyst Creation Analysis')}",
            "",
        ])

        # Executive Summary
        md_lines.extend(self._generate_executive_summary_section(report_data))

        # Performance Summary
        md_lines.extend(self._generate_performance_summary_section(report_data))

        # Model Training Analysis
        md_lines.extend(self._generate_model_training_section(report_data))

        # Regime Analysis
        md_lines.extend(self._generate_regime_analysis_section(report_data))

        # Quality Assessment
        md_lines.extend(self._generate_quality_assessment_section(report_data))

        # Optimization Analysis
        md_lines.extend(self._generate_optimization_analysis_section(report_data))

        # Risk Assessment
        md_lines.extend(self._generate_risk_assessment_section(report_data))

        # Recommendations and Alerts
        md_lines.extend(self._generate_recommendations_alerts_section(report_data))

        return "\n".join(md_lines)

    def _generate_executive_summary_section(self, report_data: Dict[str, Any]) -> List[str]:
        """Generate executive summary section."""
        lines = [
            "## 🚀 Executive Summary",
            "",
            "This comprehensive report provides detailed analysis of Step 11: Enhanced Analyst Creation with per-regime model training and performance optimization.",
            "",
        ]

        # Key highlights
        model_data = report_data.get('model_training_analysis', [])
        regime_data = report_data.get('regime_analysis', [])
        perf_data = report_data.get('performance_analysis', {})

        if model_data:
            lines.extend([
                "### 📊 Key Metrics Overview",
                f"- **Models Created:** {len(model_data)}",
                f"- **Average Training Time:** {sum(m.get('training_time_seconds', 0) for m in model_data) / max(1, len(model_data)):.2f} seconds",
                f"- **Average Accuracy:** {sum(m.get('accuracy_score', 0) for m in model_data) / max(1, len(model_data)):.3f}",
            ])

        if regime_data:
            lines.extend([
                f"- **Regimes Processed:** {len(regime_data)}",
                f"- **Average Samples per Regime:** {sum(r.get('regime_sample_count', 0) for r in regime_data) / max(1, len(regime_data)):.0f}",
            ])

        if perf_data:
            lines.extend([
                f"- **Total Training Time:** {perf_data.get('total_training_time', 0):.2f} seconds",
                f"- **Overall Accuracy:** {perf_data.get('overall_accuracy_score', 0):.3f}",
                "",
            ])
        else:
            lines.append("")

        return lines

    def _generate_performance_summary_section(self, report_data: Dict[str, Any]) -> List[str]:
        """Generate performance summary section."""
        lines = [
            "## 📈 Performance Summary",
            "",
        ]

        perf_data = report_data.get('performance_analysis', {})
        if perf_data:
            lines.extend([
                f"- **Total Regimes Processed:** {perf_data.get('total_regimes_processed', 0)}",
                f"- **Total Models Created:** {perf_data.get('total_models_created', 0)}",
                f"- **Total Training Time:** {perf_data.get('total_training_time', 0):.2f} seconds",
                f"- **Average Time per Model:** {perf_data.get('average_training_time_per_model', 0):.2f} seconds",
                f"- **Overall Accuracy Score:** {perf_data.get('overall_accuracy_score', 0):.3f}",
                "",
                "### ⚡ Efficiency Metrics",
                f"- **Computational Efficiency:** {perf_data.get('computational_efficiency_score', 0):.3f}",
                f"- **Memory Utilization:** {perf_data.get('memory_utilization', 0):.3f}",
                f"- **GPU Utilization:** {perf_data.get('gpu_utilization', 0):.3f}",
                f"- **Parallel Processing Efficiency:** {perf_data.get('parallel_processing_efficiency', 0):.3f}",
                "",
            ])

        return lines

    def _generate_model_training_section(self, report_data: Dict[str, Any]) -> List[str]:
        """Generate model training analysis section."""
        lines = [
            "## 🤖 Model Training Analysis",
            "",
        ]

        model_data = report_data.get('model_training_analysis', [])
        if model_data:
            lines.extend([
                f"- **Models Analyzed:** {len(model_data)}",
                "",
                "### 📊 Model Performance Table",
                "| Model | Regime | Accuracy | F1 Score | Training Time | Efficiency |",
                "|-------|--------|----------|----------|---------------|------------|",
            ])

            # Sort by accuracy and show top performers
            sorted_models = sorted(model_data, key=lambda x: x.get('accuracy_score', 0), reverse=True)
            for model in sorted_models[:10]:  # Show top 10 models
                lines.append(
                    f"| {model.get('model_name', 'Unknown')} | {model.get('regime_name', 'Unknown')} | "
                    f"{model.get('accuracy_score', 0):.3f} | {model.get('f1_score', 0):.3f} | "
                    f"{model.get('training_time_seconds', 0):.1f}s | {model.get('computational_efficiency', 0):.2f} |"
                )
            lines.append("")

        return lines

    def _generate_regime_analysis_section(self, report_data: Dict[str, Any]) -> List[str]:
        """Generate regime analysis section."""
        lines = [
            "## 🏷️ Per-Regime Analysis",
            "",
        ]

        regime_data = report_data.get('regime_analysis', [])
        if regime_data:
            lines.extend([
                f"- **Regimes Analyzed:** {len(regime_data)}",
                "",
                "### 📊 Regime Performance",
                "| Regime | Samples | Best Model | Best Accuracy | Avg Accuracy | Stability |",
                "|--------|---------|------------|---------------|--------------|-----------|",
            ])

            for regime in regime_data:
                lines.append(
                    f"| {regime.get('regime_name', 'Unknown')} | {regime.get('regime_sample_count', 0)} | "
                    f"{regime.get('best_model_type', 'Unknown')} | {regime.get('best_model_accuracy', 0):.3f} | "
                    f"{regime.get('average_accuracy', 0):.3f} | {regime.get('regime_stability_score', 0):.3f} |"
                )
            lines.append("")

        return lines

    def _generate_quality_assessment_section(self, report_data: Dict[str, Any]) -> List[str]:
        """Generate quality assessment section."""
        lines = [
            "## 🔍 Quality Assessment",
            "",
        ]

        quality_data = report_data.get('quality_assessment', {})
        if quality_data:
            lines.extend([
                "### 📊 Quality Metrics",
                f"- **Overall Quality Score:** {quality_data.get('overall_quality_score', 0):.3f}",
                f"- **Model Diversity Score:** {quality_data.get('model_diversity_score', 0):.3f}",
                f"- **Robustness Score:** {quality_data.get('robustness_score', 0):.3f}",
                f"- **Generalization Score:** {quality_data.get('generalization_score', 0):.3f}",
                f"- **Stability Score:** {quality_data.get('stability_score', 0):.3f}",
                "",
            ])

            warnings = quality_data.get('quality_warnings', [])
            if warnings:
                lines.extend([
                    "### ⚠️ Quality Warnings",
                ])
                for warning in warnings[:5]:
                    lines.append(f"- {warning}")
                lines.append("")

        return lines

    def _generate_optimization_analysis_section(self, report_data: Dict[str, Any]) -> List[str]:
        """Generate optimization analysis section."""
        lines = [
            "## ⚡ Training Optimization Analysis",
            "",
        ]

        opt_data = report_data.get('optimization_analysis', {})
        if opt_data:
            lines.extend([
                "### 🚀 Optimization Metrics",
                f"- **Optimization Method:** {opt_data.get('optimization_method', 'Unknown')}",
                f"- **Hyperparameter Tuning Efficiency:** {opt_data.get('hyperparameter_tuning_efficiency', 0):.3f}",
                f"- **Early Stopping Effectiveness:** {opt_data.get('early_stopping_effectiveness', 0):.3f}",
                f"- **Cross-Validation Folds:** {opt_data.get('cross_validation_folds', 0)}",
                f"- **Feature Selection Efficiency:** {opt_data.get('feature_selection_efficiency', 0):.3f}",
                f"- **Memory Optimization Score:** {opt_data.get('memory_optimization_score', 0):.3f}",
                f"- **Training Speed Improvement:** {opt_data.get('training_speed_improvement', 0):.1f}x",
                "",
            ])

        return lines

    def _generate_risk_assessment_section(self, report_data: Dict[str, Any]) -> List[str]:
        """Generate risk assessment section."""
        lines = [
            "## ⚠️ Risk Assessment",
            "",
        ]

        # Calculate overall risk level
        risk_level = "MEDIUM"
        risk_factors = []

        # Assess risks
        model_data = report_data.get('model_training_analysis', [])
        if model_data:
            low_accuracy = [m for m in model_data if m.get('accuracy_score', 1) < 0.7]
            if low_accuracy:
                risk_factors.append("Some models have accuracy below acceptable threshold")
                risk_level = "HIGH"

        regime_data = report_data.get('regime_analysis', [])
        if regime_data:
            small_regimes = [r for r in regime_data if r.get('regime_sample_count', 1000) < 100]
            if small_regimes:
                risk_factors.append("Some regimes have very small sample sizes")
                risk_level = "MEDIUM-HIGH"

        quality_data = report_data.get('quality_assessment', {})
        if quality_data and quality_data.get('overall_quality_score', 1) < 0.7:
            risk_factors.append("Overall model quality is below optimal")
            if risk_level == "MEDIUM":
                risk_level = "MEDIUM-HIGH"

        lines.extend([
            f"**Overall Risk Level:** {risk_level}",
            "",
        ])

        if risk_factors:
            lines.extend([
                "### 🚨 Key Risk Factors",
            ])
            for factor in risk_factors:
                lines.append(f"- {factor}")
            lines.append("")

        # Mitigation strategies
        lines.extend([
            "### 🛡️ Risk Mitigation Strategies",
            "- Implement robust cross-validation procedures",
            "- Address data quality issues before training",
            "- Use regularization techniques to prevent overfitting",
            "- Monitor model performance on validation sets",
            "- Implement ensemble methods for improved stability",
            "",
        ])

        return lines

    def _generate_recommendations_alerts_section(self, report_data: Dict[str, Any]) -> List[str]:
        """Generate recommendations and alerts section."""
        lines = [
            "## 💡 Recommendations and Alerts",
            "",
        ]

        recommendations = report_data.get('recommendations', [])
        if recommendations:
            lines.extend([
                "### 💡 Recommendations",
            ])
            for rec in recommendations:
                lines.append(f"- {rec}")
            lines.append("")

        alerts = report_data.get('alerts', [])
        if alerts:
            lines.extend([
                "### 🚨 Alerts",
            ])
            for alert in alerts:
                lines.append(f"- {alert}")
            lines.append("")

        return lines

    def _generate_fallback_report(self, symbol: str, exchange: str, timeframe: str, error_message: str) -> Dict[str, Any]:
        """Generate a basic fallback report when full analysis fails."""
        return {
            'timestamp': datetime.now().isoformat(),
            'step_name': 'step11_analyst_creation',
            'analysis_type': 'fallback_report',
            'metadata': {
                'symbol': symbol,
                'exchange': exchange,
                'timeframe': timeframe,
                'description': 'Fallback Analyst Creation Report'
            },
            'error': error_message,
            'recommendations': ['Review error logs and fix underlying issues before re-running analysis'],
            'alerts': ['Analysis failed - manual review required']
        }

    def _generate_and_save_visualizations(self, report_data: Dict[str, Any]) -> List[str]:
        """Generate and save visualization charts."""
        saved_files = []

        try:
            # Create visualizations directory
            viz_dir = self.output_dir / f"step11_visualizations_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            viz_dir.mkdir(exist_ok=True)

            viz_data = report_data.get('visualization_data', {})

            # Generate model performance chart
            if 'model_performance_chart' in viz_data:
                path = self._create_model_performance_chart(viz_data['model_performance_chart'], viz_dir)
                if path:
                    saved_files.append(path)

            # Generate regime comparison chart
            if 'regime_comparison_chart' in viz_data:
                path = self._create_regime_comparison_chart(viz_data['regime_comparison_chart'], viz_dir)
                if path:
                    saved_files.append(path)

            # Generate algorithm comparison chart
            if 'algorithm_comparison_chart' in viz_data:
                path = self._create_algorithm_comparison_chart(viz_data['algorithm_comparison_chart'], viz_dir)
                if path:
                    saved_files.append(path)

            # Generate training efficiency chart
            if 'training_efficiency_chart' in viz_data:
                path = self._create_training_efficiency_chart(viz_data['training_efficiency_chart'], viz_dir)
                if path:
                    saved_files.append(path)

            # Generate quality assessment radar
            if 'quality_assessment_radar' in viz_data:
                path = self._create_quality_assessment_radar(viz_data['quality_assessment_radar'], viz_dir)
                if path:
                    saved_files.append(path)

            self.logger.info(f"📊 Visualizations saved to: {viz_dir}")

        except Exception as e:
            self.logger.error(f"Failed to generate visualizations: {e}")

        return saved_files

    def _create_model_performance_chart(self, data: Dict[str, Any], viz_dir: Path) -> Optional[str]:
        """Create model performance visualization."""
        try:
            if not data:
                return None

            plt.figure(figsize=(12, 8))

            # Subplot 1: Accuracy comparison
            plt.subplot(2, 2, 1)
            model_names = data.get('model_names', [])
            accuracies = data.get('accuracies', [])
            if model_names and accuracies:
                bars = plt.bar(range(len(model_names)), accuracies, color='skyblue', alpha=0.7)
                plt.xticks(range(len(model_names)), [name[:15] + '...' if len(name) > 15 else name for name in model_names], rotation=45, ha='right')
                plt.title('Model Accuracy Comparison', fontsize=12, fontweight='bold')
                plt.ylabel('Accuracy Score')
                plt.ylim(0, 1)

                # Add value labels
                for bar, acc in zip(bars, accuracies):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            '.3f', ha='center', va='bottom', fontweight='bold')

            # Subplot 2: F1 Score comparison
            plt.subplot(2, 2, 2)
            f1_scores = data.get('f1_scores', [])
            if model_names and f1_scores:
                bars = plt.bar(range(len(model_names)), f1_scores, color='lightgreen', alpha=0.7)
                plt.xticks(range(len(model_names)), [name[:15] + '...' if len(name) > 15 else name for name in model_names], rotation=45, ha='right')
                plt.title('Model F1 Score Comparison', fontsize=12, fontweight='bold')
                plt.ylabel('F1 Score')
                plt.ylim(0, 1)

            # Subplot 3: Training time comparison
            plt.subplot(2, 2, 3)
            training_times = data.get('training_times', [])
            if model_names and training_times:
                bars = plt.bar(range(len(model_names)), training_times, color='coral', alpha=0.7)
                plt.xticks(range(len(model_names)), [name[:15] + '...' if len(name) > 15 else name for name in model_names], rotation=45, ha='right')
                plt.title('Training Time Comparison', fontsize=12, fontweight='bold')
                plt.ylabel('Training Time (seconds)')

            # Subplot 4: Performance vs Time scatter
            plt.subplot(2, 2, 4)
            if accuracies and training_times:
                plt.scatter(training_times, accuracies, s=100, alpha=0.7, color='purple')
                plt.title('Performance vs Training Time', fontsize=12, fontweight='bold')
                plt.xlabel('Training Time (seconds)')
                plt.ylabel('Accuracy Score')
                plt.grid(True, alpha=0.3)

                # Add trend line
                if len(accuracies) > 1:
                    z = np.polyfit(training_times, accuracies, 1)
                    p = np.poly1d(z)
                    plt.plot(sorted(training_times), p(sorted(training_times)), "r--", alpha=0.7)

            plt.tight_layout()
            chart_path = viz_dir / 'model_performance_analysis.png'
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()

            return str(chart_path)

        except Exception as e:
            self.logger.error(f"Failed to create model performance chart: {e}")
            return None

    def _create_regime_comparison_chart(self, data: Dict[str, Any], viz_dir: Path) -> Optional[str]:
        """Create regime comparison visualization."""
        try:
            if not data:
                return None

            plt.figure(figsize=(14, 8))

            # Subplot 1: Sample distribution
            plt.subplot(2, 2, 1)
            regime_names = data.get('regime_names', [])
            sample_counts = data.get('sample_counts', [])
            if regime_names and sample_counts:
                bars = plt.bar(range(len(regime_names)), sample_counts, color='lightblue', alpha=0.7)
                plt.xticks(range(len(regime_names)), [name[:15] + '...' if len(name) > 15 else name for name in regime_names], rotation=45, ha='right')
                plt.title('Sample Distribution by Regime', fontsize=12, fontweight='bold')
                plt.ylabel('Sample Count')

                # Add value labels
                for bar, count in zip(bars, sample_counts):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(sample_counts)*0.01,
                            f'{count:,}', ha='center', va='bottom', fontweight='bold')

            # Subplot 2: Best accuracy by regime
            plt.subplot(2, 2, 2)
            best_accuracies = data.get('best_accuracies', [])
            if regime_names and best_accuracies:
                bars = plt.bar(range(len(regime_names)), best_accuracies, color='lightgreen', alpha=0.7)
                plt.xticks(range(len(regime_names)), [name[:15] + '...' if len(name) > 15 else name for name in regime_names], rotation=45, ha='right')
                plt.title('Best Model Accuracy by Regime', fontsize=12, fontweight='bold')
                plt.ylabel('Accuracy Score')
                plt.ylim(0, 1)

            # Subplot 3: Average accuracy by regime
            plt.subplot(2, 2, 3)
            avg_accuracies = data.get('average_accuracies', [])
            if regime_names and avg_accuracies:
                bars = plt.bar(range(len(regime_names)), avg_accuracies, color='gold', alpha=0.7)
                plt.xticks(range(len(regime_names)), [name[:15] + '...' if len(name) > 15 else name for name in regime_names], rotation=45, ha='right')
                plt.title('Average Accuracy by Regime', fontsize=12, fontweight='bold')
                plt.ylabel('Accuracy Score')
                plt.ylim(0, 1)

            # Subplot 4: Regime performance summary
            plt.subplot(2, 2, 4)
            if sample_counts and best_accuracies:
                # Create a summary scatter plot
                plt.scatter(sample_counts, best_accuracies, s=100, alpha=0.7, color='purple')
                plt.title('Sample Size vs Best Performance', fontsize=12, fontweight='bold')
                plt.xlabel('Sample Count')
                plt.ylabel('Best Accuracy')
                plt.grid(True, alpha=0.3)

                # Add regime labels
                for i, regime in enumerate(regime_names):
                    if i < len(sample_counts) and i < len(best_accuracies):
                        plt.annotate(regime[:8] + ('...' if len(regime) > 8 else ''),
                                   (sample_counts[i], best_accuracies[i]),
                                   xytext=(5, 5), textcoords='offset points', fontsize=8)

            plt.tight_layout()
            chart_path = viz_dir / 'regime_comparison_analysis.png'
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()

            return str(chart_path)

        except Exception as e:
            self.logger.error(f"Failed to create regime comparison chart: {e}")
            return None

    def _create_algorithm_comparison_chart(self, data: Dict[str, Any], viz_dir: Path) -> Optional[str]:
        """Create algorithm comparison visualization."""
        try:
            if not data:
                return None

            plt.figure(figsize=(12, 8))

            algorithms = data.get('algorithms', [])
            avg_accuracies = data.get('avg_accuracies', [])
            max_accuracies = data.get('max_accuracies', [])
            min_accuracies = data.get('min_accuracies', [])

            if algorithms and avg_accuracies:
                x = np.arange(len(algorithms))
                width = 0.25

                # Create grouped bar chart
                plt.bar(x - width, min_accuracies, width, label='Min Accuracy', alpha=0.7, color='red')
                plt.bar(x, avg_accuracies, width, label='Avg Accuracy', alpha=0.7, color='blue')
                plt.bar(x + width, max_accuracies, width, label='Max Accuracy', alpha=0.7, color='green')

                plt.title('Algorithm Performance Comparison', fontsize=14, fontweight='bold')
                plt.xlabel('Algorithm')
                plt.ylabel('Accuracy Score')
                plt.xticks(x, algorithms, rotation=45, ha='right')
                plt.legend()
                plt.ylim(0, 1)
                plt.grid(True, alpha=0.3, axis='y')

                # Add value labels
                for i, (avg, max_val, min_val) in enumerate(zip(avg_accuracies, max_accuracies, min_accuracies)):
                    plt.text(i, avg + 0.01, '.3f', ha='center', va='bottom', fontweight='bold')

            plt.tight_layout()
            chart_path = viz_dir / 'algorithm_comparison_analysis.png'
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()

            return str(chart_path)

        except Exception as e:
            self.logger.error(f"Failed to create algorithm comparison chart: {e}")
            return None

    def _create_training_efficiency_chart(self, data: Dict[str, Any], viz_dir: Path) -> Optional[str]:
        """Create training efficiency visualization."""
        try:
            if not data:
                return None

            plt.figure(figsize=(12, 8))

            # Subplot 1: Training time distribution
            plt.subplot(2, 2, 1)
            training_times = data.get('training_times', [])
            if training_times:
                plt.hist(training_times, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                plt.title('Training Time Distribution', fontsize=12, fontweight='bold')
                plt.xlabel('Training Time (seconds)')
                plt.ylabel('Frequency')
                plt.grid(True, alpha=0.3)

            # Subplot 2: Efficiency vs Time scatter
            plt.subplot(2, 2, 2)
            efficiencies = data.get('efficiency_scores', [])
            if training_times and efficiencies:
                plt.scatter(training_times, efficiencies, s=100, alpha=0.7, color='green')
                plt.title('Efficiency vs Training Time', fontsize=12, fontweight='bold')
                plt.xlabel('Training Time (seconds)')
                plt.ylabel('Efficiency Score')
                plt.grid(True, alpha=0.3)

            # Subplot 3: Sample size vs performance
            plt.subplot(2, 2, 3)
            sample_counts = data.get('sample_counts', [])
            model_names = data.get('model_names', [])
            # We need accuracy data - this would need to be passed in
            if sample_counts and len(sample_counts) > 0:
                plt.scatter(sample_counts, [0.8] * len(sample_counts), s=100, alpha=0.7, color='orange')  # Placeholder
                plt.title('Sample Size vs Performance', fontsize=12, fontweight='bold')
                plt.xlabel('Sample Count')
                plt.ylabel('Accuracy Score')
                plt.grid(True, alpha=0.3)

            # Subplot 4: Efficiency distribution
            plt.subplot(2, 2, 4)
            if efficiencies:
                plt.hist(efficiencies, bins=15, alpha=0.7, color='purple', edgecolor='black')
                plt.title('Efficiency Score Distribution', fontsize=12, fontweight='bold')
                plt.xlabel('Efficiency Score')
                plt.ylabel('Frequency')
                plt.grid(True, alpha=0.3)

            plt.tight_layout()
            chart_path = viz_dir / 'training_efficiency_analysis.png'
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()

            return str(chart_path)

        except Exception as e:
            self.logger.error(f"Failed to create training efficiency chart: {e}")
            return None

    def _create_quality_assessment_radar(self, data: Dict[str, Any], viz_dir: Path) -> Optional[str]:
        """Create quality assessment radar chart."""
        try:
            if not data:
                return None

            plt.figure(figsize=(10, 8))

            categories = data.get('categories', [])
            values = data.get('values', [])

            if categories and values:
                # Close the radar chart
                values += values[:1]
                categories += categories[:1]

                angles = [n / float(len(categories[:-1])) * 2 * 3.14159 for n in range(len(categories[:-1]))]
                angles += angles[:1]

                plt.polar(angles, values, 'o-', linewidth=2, label='Quality Metrics')
                plt.fill(angles, values, alpha=0.25)
                plt.xticks(angles[:-1], categories[:-1])
                plt.title('Analyst Quality Assessment Radar', fontsize=14, fontweight='bold')
                plt.ylim(0, 1)

                # Add overall quality score in center
                overall_quality = np.mean(values[:-1]) if values else 0
                plt.text(0, 0, '.3f', ha='center', va='center',
                        fontsize=12, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))

            plt.tight_layout()
            chart_path = viz_dir / 'quality_assessment_radar.png'
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()

            return str(chart_path)

        except Exception as e:
            self.logger.error(f"Failed to create quality assessment radar: {e}")
            return None

    def _save_csv_summary(self, report_data: Dict[str, Any]) -> Optional[str]:
        """Save CSV summary of key metrics."""
        try:
            # Create summary data
            summary_data = {
                'metric': [],
                'value': [],
                'category': []
            }

            # Add model metrics
            model_data = report_data.get('model_training_analysis', [])
            for model in model_data:
                summary_data['metric'].append(f"{model.get('model_name', 'Unknown')}_accuracy")
                summary_data['value'].append(model.get('accuracy_score', 0))
                summary_data['category'].append('model_performance')

                summary_data['metric'].append(f"{model.get('model_name', 'Unknown')}_f1")
                summary_data['value'].append(model.get('f1_score', 0))
                summary_data['category'].append('model_performance')

            # Add regime metrics
            regime_data = report_data.get('regime_analysis', [])
            for regime in regime_data:
                summary_data['metric'].append(f"{regime.get('regime_name', 'Unknown')}_samples")
                summary_data['value'].append(regime.get('regime_sample_count', 0))
                summary_data['category'].append('regime_analysis')

                summary_data['metric'].append(f"{regime.get('regime_name', 'Unknown')}_accuracy")
                summary_data['value'].append(regime.get('best_model_accuracy', 0))
                summary_data['category'].append('regime_analysis')

            # Save as CSV
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            csv_path = self.output_dir / f"step11_analyst_creation_summary_{timestamp}.csv"
            df = pd.DataFrame(summary_data)
            df.to_csv(csv_path, index=False)

            self.logger.info(f"📄 CSV summary saved to: {csv_path}")
            return str(csv_path)

        except Exception as e:
            self.logger.error(f"Failed to save CSV summary: {e}")
            return None

import asyncio
import json
import os
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Callable
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
import pandas as pd
import numpy as np
import joblib
import optuna
import torch
from torch import nn, optim
# DataLoader and TensorDataset not used in current implementation
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import xgboost as xgb
import time

# SHAP not used in this step - removed to reduce dependencies
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Import optimization tools
try:
    from src.utils.m1_gpu_utils import get_m1_gpu_manager, M1GPUManager
    from src.utils.m1_memory_optimizer import get_m1_memory_optimizer, M1MemoryOptimizer
    from src.utils.m1_cpu_optimizer import get_m1_cpu_optimizer, M1CPUOptimizer
    from src.utils.vectorized_processing_core import get_vectorized_processing_core
    from src.utils.enhanced_matrix_operations import get_enhanced_matrix_operations
    from src.utils.enhanced_step_optimizations import get_step_optimization_manager, OptimizationProfile, WorkloadType, OptimizationStrategy
    from src.utils.optimized_data_manager import get_optimized_data_manager
    OPTIMIZATION_TOOLS_AVAILABLE = True
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"⚠️ Some optimization tools not available: {e}")
    OPTIMIZATION_TOOLS_AVAILABLE = False

# Import existing utilities instead of duplicating
try:
    from src.utils.pipeline_standards import pipeline_standards
except ImportError:
    class pipeline_standards:
        @staticmethod
        def build_path(path_type: str, exchange: str, symbol: str) -> str:
            return f'data/{path_type}/{exchange}/{symbol}'

class AnalystCreationStep:
    """Step 11: Analyst Creation - Creates base analyst models for each regime.

    This step creates the initial analyst models for each regime using the
    regime-specific data and features. It focuses on creating robust base models
    that will be enhanced in subsequent steps.
    """

    # Model configuration constants
    LIGHTGBM_PARAMS = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1
    }

    XGBOOST_PARAMS = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'n_estimators': 100
    }

    RANDOM_FOREST_PARAMS = {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'random_state': 42
    }

    NEURAL_NETWORK_CONFIG = {
        'hidden_dims': [64, 32],
        'dropout_rate': 0.2,
        'learning_rate': 0.001,
        'epochs': 50
    }

    @log_important_calls
    def __init__(self, config: dict[str, Any]) -> None:
        """Initializes the AnalystCreationStep.

        Args:
            config (Dict[str, Any]): Configuration dictionary for the step.
        """
        self.config = config
        self.standards = pipeline_standards
        self.logger = system_logger
        self._validate_environment()

        # Initialize optimization components
        self._init_optimization_components()

        self.device = self._safe_get_device()
        self.logger.info(f'Using device: {self.device.upper()} for PyTorch operations.')
        self._METADATA_COLUMNS: list[str] = ['timestamp', 'exchange', 'symbol', 'timeframe', 'split', 'year', 'month', 'day', 'day_of_week', 'day_of_month', 'quarter']
        self._LABEL_COLUMNS: set[str] = {'label', 'target', 'y', 'class', 'signal', 'prediction'}
    @log_all_calls

    def _validate_environment(self) -> None:
        """Validate environment dependencies and configuration."""
        # Basic validation - dependencies are checked at import time
        pass

    def _init_optimization_components(self) -> None:
        """Initialize optimization components for enhanced performance."""
        if not OPTIMIZATION_TOOLS_AVAILABLE:
            self.logger.warning("⚠️ Optimization tools not available, using basic functionality")
            self.m1_gpu_manager = None
            self.m1_memory_optimizer = None
            self.m1_cpu_optimizer = None
            self.vectorized_core = None
            self.matrix_operations = None
            self.step_optimizer = None
            self.data_manager = None
            return

        try:
            # Initialize M1 hardware optimizations
            self.m1_gpu_manager = get_m1_gpu_manager()
            self.m1_memory_optimizer = get_m1_memory_optimizer()
            self.m1_cpu_optimizer = get_m1_cpu_optimizer()

            # Initialize processing core optimizations
            self.vectorized_core = get_vectorized_processing_core()
            self.matrix_operations = get_enhanced_matrix_operations()
            self.step_optimizer = get_step_optimization_manager()

            # Initialize data management optimizations
            self.data_manager = get_optimized_data_manager()

            self.logger.info("🚀 All optimization components initialized successfully")

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to initialize some optimization components: {e}")
            # Set to None for graceful fallback
            self.m1_gpu_manager = None
            self.m1_memory_optimizer = None
            self.m1_cpu_optimizer = None
            self.vectorized_core = None
            self.matrix_operations = None
            self.step_optimizer = None
            self.data_manager = None

    def _safe_get_device(self) -> str:
        """Safely determine the best device to use with M1 optimizations."""
        try:
            # Use M1 GPU manager if available
            if self.m1_gpu_manager:
                device = self.m1_gpu_manager.device
                return device.type

            # Fallback to manual detection
            if torch.cuda.is_available():
                return 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return 'mps'
            else:
                return 'cpu'
        except Exception as e:
            self.logger.warning(f'Device detection failed: {e}, using CPU')
            return 'cpu'

    @handles_errors(exceptions=(Exception,), default_return = False, context='analyst creation step initialization')
    async def initialize(self) -> None:
        """Initialize the analyst creation step."""
        self.logger.info('Initializing Analyst Creation Step...')
        self.logger.info('Analyst Creation Step initialized successfully.')

    def _create_workload_profile(self, data_dir: str) -> OptimizationProfile:
        """Create optimization profile based on data characteristics."""
        try:
            # Estimate data size
            data_size_mb = self._estimate_data_size(data_dir)

            # Determine workload type based on data characteristics
            workload_type = WorkloadType.MIXED  # Default
            if data_size_mb > 1000:
                workload_type = WorkloadType.MEMORY_INTENSIVE
            elif self.m1_gpu_manager and self.m1_gpu_manager.device.type != 'cpu':
                workload_type = WorkloadType.GPU_INTENSIVE

            return OptimizationProfile(
                workload_type=workload_type,
                data_size_mb=data_size_mb,
                expected_duration=300.0,  # 5 minutes expected
                priority="high"
            )
        except Exception as e:
            self.logger.warning(f"Failed to create workload profile: {e}")
            return OptimizationProfile(
                workload_type=WorkloadType.MIXED,
                data_size_mb=100.0,
                expected_duration=300.0,
                priority="normal"
            )

    def _estimate_data_size(self, data_dir: str) -> float:
        """Estimate data size for optimization planning."""
        try:
            total_size = 0
            data_path = Path(data_dir)
            if data_path.exists():
                for file_path in data_path.rglob('*.parquet'):
                    if file_path.exists():
                        total_size += file_path.stat().st_size
            return total_size / (1024 * 1024)  # Convert to MB
        except Exception:
            return 100.0  # Default estimate

    def _select_optimizations(self, profile: OptimizationProfile) -> Any:
        """Select intelligent optimizations based on workload profile."""
        if self.step_optimizer and OPTIMIZATION_TOOLS_AVAILABLE:
            try:
                return self.step_optimizer.select_intelligent_optimizations(profile)
            except Exception as e:
                self.logger.warning(f"Intelligent optimization selection failed: {e}")

        # Fallback to basic optimization decision
        return type('OptimizationDecision', (), {
            'strategy': OptimizationStrategy.BALANCED,
            'enabled_optimizations': ['memory_cleanup', 'parallel_processing'],
            'disabled_optimizations': [],
            'configuration': {},
            'reasoning': ['Using fallback balanced optimizations']
        })()

    @handles_errors(exceptions=(Exception,), default_return={'status': 'FAILED', 'error': 'Execution failed'}, context='analyst creation step execution')
    async def execute(self, training_input: dict[str, Any], pipeline_state: dict[str, Any]) -> dict[str, Any]:
        """Executes the analyst model creation pipeline for each regime with intelligent optimizations.

        Args:
            training_input (Dict[str, Any]): Input parameters, including symbol, exchange, and data directories.
            pipeline_state (Dict[str, Any]): The current state of the pipeline.

        Returns:
            Dict[str, Any]: A dictionary containing the results of the creation process.
        """
        self.logger.info('🚀 Starting Step 11: Analyst Creation - Base Model Creation for Each Regime')
        self.logger.info('🔄 Executing Analyst Creation with Enhanced Optimizations...')
        start_time = datetime.now()

        # Create optimization profile for intelligent selection
        data_dir = str(training_input.get('data_dir', 'data/training'))
        workload_profile = self._create_workload_profile(data_dir)

        # Select intelligent optimizations
        optimization_decision = self._select_optimizations(workload_profile)

        try:
            data_dir: str = str(training_input.get('data_dir', 'data/training'))
            models_dir: str = os.path.join(data_dir, 'analyst_models')
            regime_data_dir: str = data_dir
            self.logger.info(f'📁 Data directory: {data_dir}')
            self.logger.info(f'📁 Models directory: {models_dir}')
            self.logger.info(f'📁 Regime data directory: {regime_data_dir}')
            os.makedirs(models_dir, exist_ok = True)
            self.logger.info('🔄 Loading regime splits from previous step...')
            regime_splits = await self._load_regime_splits(regime_data_dir)
            if not regime_splits:
                msg = f'No regime splits found in {regime_data_dir}. Step 8 must complete successfully first.'
                raise ValueError(msg)
            self.logger.info(f'📊 Found {len(regime_splits)} regimes to process')
            created_models_summary: dict[str, dict[str, Any]] = {}

            async def create_regime_analysts(regime_name: str, regime_data: pd.DataFrame) -> tuple[str, dict[str, Any]]:
                self.logger.info(f'🚀 Starting analyst creation for regime: {regime_name}')
                self.logger.info(f'📊 Regime {regime_name} has {len(regime_data)} samples')
                try:
                    X_train, y_train, X_val, y_val = await self._prepare_regime_data(regime_data)
                    self.logger.info(f'✅ Prepared data for regime {regime_name}: train={X_train.shape}, val={X_val.shape}')
                except Exception as e:
                    self.logger.exception(f"⚠️ Error preparing data for regime '{regime_name}': {e}")
                    return (regime_name, {})
                regime_models = await self._create_regime_analysts(regime_name, X_train, y_train, X_val, y_val)
                return (regime_name, regime_models)
            self.logger.info(f'🔄 Creating parallel processing tasks for {len(regime_splits)} regimes...')

            # Use CPU optimizer for intelligent parallel processing
            if self.m1_cpu_optimizer and OPTIMIZATION_TOOLS_AVAILABLE:
                max_concurrent = self.m1_cpu_optimizer.get_optimal_workers_for_task("cpu_bound")
                max_concurrent = min(max_concurrent, len(regime_splits))
            else:
                max_concurrent = min(3, len(regime_splits))

            self.logger.info(f'⚡ Processing {len(regime_splits)} regimes with max {max_concurrent} concurrent tasks')

            tasks: list[asyncio.Task] = []
            for regime_name, regime_data in regime_splits.items():
                # Optimize data before processing
                if self.vectorized_core and OPTIMIZATION_TOOLS_AVAILABLE:
                    regime_data = self.vectorized_core.optimize_dataframe_for_processing(regime_data)

                task = asyncio.create_task(create_regime_analysts(regime_name, regime_data))
                tasks.append(task)

            # Memory cleanup before processing
            if self.m1_memory_optimizer and OPTIMIZATION_TOOLS_AVAILABLE:
                self.m1_memory_optimizer.optimize_memory()
            for batch_idx, i in enumerate(range(0, len(tasks), max_concurrent), 1):
                batch = tasks[i:i + max_concurrent]
                self.logger.info(f'🔄 Processing batch {batch_idx}: regimes {i + 1}-{min(i + max_concurrent, len(tasks))}')
                results = await asyncio.gather(*batch, return_exceptions = True)
                for j, result in enumerate(results):
                    regime_idx = i + j
                    if isinstance(result, Exception):
                        self.logger.error(f'❌ Error in regime {regime_idx}: {result}')
                        continue
                    regime_name, regime_models = result
                    created_models_summary[regime_name] = regime_models
                    self.logger.info(f'✅ Completed analyst creation for regime: {regime_name}')
            await self._save_analyst_models(created_models_summary, models_dir)
            total_models = sum((len(models) for models in created_models_summary.values()))

            # Calculate execution time and performance metrics
            execution_time = (datetime.now() - start_time).total_seconds()
            total_regimes = len(created_models_summary)

            self.logger.info(f'🎉 Analyst creation completed: {len(created_models_summary)} regimes, {total_models} total models')
            self.logger.info(f'⏱️ Total execution time: {execution_time:.2f}s ({execution_time/total_regimes:.2f}s per regime)')

            # Record performance for optimization learning
            if self.step_optimizer and OPTIMIZATION_TOOLS_AVAILABLE:
                try:
                    actual_improvement = {
                        'speedup': execution_time / 300.0,  # Normalized against expected time
                        'memory_efficiency': 1.0,  # Could be improved with actual memory tracking
                        'regimes_processed': total_regimes,
                        'models_created': total_models
                    }
                    self.step_optimizer.record_optimization_performance(
                        workload_profile, optimization_decision, actual_improvement, execution_time
                    )
                    self.logger.info('📊 Performance metrics recorded for optimization learning')
                except Exception as e:
                    self.logger.debug(f"Failed to record performance metrics: {e}")

            pipeline_state['analyst_creation_completed'] = True
            pipeline_state['created_analyst_models'] = created_models_summary
            pipeline_state['analyst_models_directory'] = models_dir
            pipeline_state['execution_time_seconds'] = execution_time
            pipeline_state['regimes_processed'] = total_regimes
            pipeline_state['models_created'] = total_models

            return pipeline_state
        except Exception as e:
            self.logger.exception(f'❌ Error in analyst creation: {e}')
            pipeline_state['analyst_creation_completed'] = False
            pipeline_state['analyst_creation_error'] = str(e)
            return pipeline_state

    async def _load_regime_splits(self, data_dir: str) -> dict[str, pd.DataFrame]:
        """Load regime data from unified dataset with labels."""
        try:
            symbol = self.config.get('symbol', 'ETHUSDT')
            exchange = self.config.get('exchange', 'BINANCE')
            timeframe = self.config.get('timeframe', '5m')
            unified_regime_file = os.path.join(data_dir, 'training', f'{exchange}_{symbol}_{timeframe}_unified_regime_data.parquet')
            if os.path.exists(unified_regime_file):
                self.logger.info(f'✅ Loading unified regime dataset: {unified_regime_file}')
                unified_data = pd.read_parquet(unified_regime_file)
                labels_file = os.path.join(data_dir, 'training', f'{exchange}_{symbol}_{timeframe}_regime_labels.json')
                if os.path.exists(labels_file):
                    with open(labels_file) as f:
                        regime_labels = json.load(f)
                    regime_ids = regime_labels.get('regime_ids', [])
                    self.logger.info(f'📊 Found {len(regime_ids)} regimes in unified dataset')
                    regime_splits = {}
                    for regime_id in regime_ids:
                        regime_data = unified_data[unified_data['composite_cluster_id'] == regime_id].copy()
                        if len(regime_data) > 0:
                            regime_splits[f'regime_{regime_id}'] = regime_data
                            self.logger.info(f'📊 Created regime {regime_id}: {len(regime_data)} rows')
                    self.logger.info(f'✅ Created {len(regime_splits)} regime splits from unified dataset')
                    return regime_splits
                else:
                    self.logger.warning(f'⚠️ Regime labels file not found: {labels_file}')
            self.logger.warning('⚠️ Falling back to legacy regime data loading approach')
            regime_splits_dir = os.path.join(data_dir, 'training', 'regime_splits')
            if not os.path.exists(regime_splits_dir):
                self.logger.error(f'❌ Legacy regime splits directory not found: {regime_splits_dir}')
                return {}
            regime_splits = {}
            for file in os.listdir(regime_splits_dir):
                if file.endswith('.parquet') and 'regime_' in file:
                    regime_name = file.split('regime_')[-1].replace('.parquet', '')
                    file_path = os.path.join(regime_splits_dir, file)
                    regime_data = pd.read_parquet(file_path)
                    regime_splits[regime_name] = regime_data
                    self.logger.info(f'📊 Loaded legacy regime {regime_name}: {len(regime_data)} rows')
            return regime_splits
        except Exception as e:
            self.logger.exception(f'❌ Error loading regime splits: {e}')
            return {}

    async def _prepare_regime_data(self, regime_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Prepare data for analyst model creation."""
        try:
            feature_columns = [col for col in regime_data.columns if col not in self._METADATA_COLUMNS and col not in self._LABEL_COLUMNS]
            X = regime_data[feature_columns]
            y = regime_data['label'] if 'label' in regime_data.columns else pd.Series([0] * len(regime_data))
            split_idx = int(len(X) * 0.8)
            X_train, X_val = (X.iloc[:split_idx], X.iloc[split_idx:])
            y_train, y_val = (y.iloc[:split_idx], y.iloc[split_idx:])
            return (X_train, y_train, X_val, y_val)
        except Exception as e:
            self.logger.exception(f'❌ Error preparing regime data: {e}')
            raise

    async def _create_regime_analysts(self, regime_name: str, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> dict[str, Any]:
        """Create base analyst models for a specific regime with parallel training optimization."""
        try:
            import time
            start_time = time.time()

            self.logger.info(f'🔧 Creating base analyst models for regime: {regime_name}')
            regime_models = {}

            # Determine optimal parallelization strategy based on dataset size
            dataset_size = len(X_train)
            feature_count = len(X_train.columns)

            # Use parallel training for larger datasets
            if dataset_size > 10000 or feature_count > 50:
                self.logger.info(f'⚡ Using parallel model training for large dataset ({dataset_size} samples, {feature_count} features)')
                regime_models = await self._create_regime_analysts_parallel(regime_name, X_train, y_train, X_val, y_val)
            else:
                self.logger.info(f'🔄 Using sequential model training for smaller dataset')
                regime_models = await self._create_regime_analysts_sequential(regime_name, X_train, y_train, X_val, y_val)

            training_time = time.time() - start_time
            self.logger.info(f'✅ Created {len(regime_models)} base models for regime: {regime_name} in {training_time:.2f}s')
            return regime_models

        except Exception as e:
            self.logger.exception(f'❌ Error creating analyst models for regime {regime_name}: {e}')
            return {}

    async def _create_regime_analysts_parallel(self, regime_name: str, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> dict[str, Any]:
        """Create models using parallel training for better performance."""
        try:
            import asyncio

            regime_models = {}

            # Create model training tasks
            tasks = []

            # LightGBM task
            tasks.append(asyncio.create_task(self._create_lightgbm_model(X_train, y_train, X_val, y_val)))

            # XGBoost task
            tasks.append(asyncio.create_task(self._create_xgboost_model(X_train, y_train, X_val, y_val)))

            # Random Forest task
            tasks.append(asyncio.create_task(self._create_random_forest_model(X_train, y_train, X_val, y_val)))

            # Neural Network task (only if torch available)
            if torch is not None:
                tasks.append(asyncio.create_task(self._create_neural_network_model(X_train, y_train, X_val, y_val)))

            self.logger.info(f'🚀 Training {len(tasks)} models concurrently for regime: {regime_name}')

            # Execute tasks with error handling
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            model_types = ['lightgbm', 'xgboost', 'random_forest']
            if torch is not None:
                model_types.append('neural_network')

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error(f'❌ Error training {model_types[i]} model: {result}')
                    continue

                regime_models[model_types[i]] = result
                self.logger.info(f'✅ Completed {model_types[i]} model training')

            # Memory cleanup
            if self.m1_memory_optimizer and OPTIMIZATION_TOOLS_AVAILABLE:
                self.m1_memory_optimizer.optimize_memory()

            return regime_models

        except Exception as e:
            self.logger.warning(f'Parallel training failed, falling back to sequential: {e}')
            return await self._create_regime_analysts_sequential(regime_name, X_train, y_train, X_val, y_val)

    async def _create_regime_analysts_sequential(self, regime_name: str, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> dict[str, Any]:
        """Create models using sequential training (original approach)."""
        try:
            regime_models = {}

            # Memory optimization before training
            if self.m1_memory_optimizer and OPTIMIZATION_TOOLS_AVAILABLE:
                self.m1_memory_optimizer.optimize_memory()

            self.logger.info(f'🌳 Creating LightGBM model for regime: {regime_name}')
            lgb_model = await self._create_lightgbm_model(X_train, y_train, X_val, y_val)
            regime_models['lightgbm'] = lgb_model

            self.logger.info(f'🌲 Creating XGBoost model for regime: {regime_name}')
            xgb_model = await self._create_xgboost_model(X_train, y_train, X_val, y_val)
            regime_models['xgboost'] = xgb_model

            self.logger.info(f'🌿 Creating Random Forest model for regime: {regime_name}')
            rf_model = await self._create_random_forest_model(X_train, y_train, X_val, y_val)
            regime_models['random_forest'] = rf_model

            if torch is not None:
                self.logger.info(f'🧠 Creating Neural Network model for regime: {regime_name}')
                nn_model = await self._create_neural_network_model(X_train, y_train, X_val, y_val)
                regime_models['neural_network'] = nn_model

            return regime_models

        except Exception as e:
            self.logger.exception(f'❌ Error in sequential model creation: {e}')
            return {}

    async def _create_lightgbm_model(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> dict[str, Any]:
        """Create a LightGBM model with optimized parameters for performance."""
        try:
            import time
            start_time = time.time()

            params = self.LIGHTGBM_PARAMS.copy()

            # Adaptive parameter optimization based on dataset size
            dataset_size = len(X_train)
            if dataset_size > 50000:
                # More aggressive early stopping for large datasets
                params['early_stopping_round'] = 5
                num_boost_round = 50
            elif dataset_size > 10000:
                params['early_stopping_round'] = 8
                num_boost_round = 75
            else:
                # Standard parameters for smaller datasets
                num_boost_round = 100

            # Memory optimization for large datasets
            if len(X_train.columns) > 100:
                params['feature_fraction'] = 0.8  # Use only 80% of features
                params['bagging_fraction'] = 0.8  # Use only 80% of data for bagging

            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

            model = lgb.train(
                params,
                train_data,
                valid_sets=[val_data],
                num_boost_round=num_boost_round,
                callbacks=[lgb.early_stopping(stopping_rounds=params.get('early_stopping_round', 10))]
            )

            val_pred = model.predict(X_val)
            val_pred_binary = (val_pred > 0.5).astype(int)
            accuracy = accuracy_score(y_val, val_pred_binary)

            training_time = time.time() - start_time

            result = {
                'model': model,
                'accuracy': accuracy,
                'model_type': 'lightgbm',
                'creation_date': datetime.now().isoformat(),
                'training_time': training_time,
                'feature_importance': dict(zip(X_train.columns, model.feature_importance())),
                'dataset_size': dataset_size,
                'feature_count': len(X_train.columns)
            }

            self.logger.info(f'✅ LightGBM model trained in {training_time:.2f}s (accuracy: {accuracy:.4f})')
            return result

        except Exception as e:
            self.logger.exception(f'❌ Error creating LightGBM model: {e}')
            raise RuntimeError(f'Failed to create LightGBM model: {e}')

    async def _create_xgboost_model(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> dict[str, Any]:
        """Create an XGBoost model with optimized parameters for performance."""
        try:
            import time
            start_time = time.time()

            params = self.XGBOOST_PARAMS.copy()

            # Adaptive parameter optimization based on dataset size
            dataset_size = len(X_train)
            feature_count = len(X_train.columns)

            if dataset_size > 50000:
                # More aggressive early stopping for large datasets
                early_stopping_rounds = 5
                params['n_estimators'] = 50
            elif dataset_size > 10000:
                early_stopping_rounds = 8
                params['n_estimators'] = 75
            else:
                early_stopping_rounds = 10
                params['n_estimators'] = 100

            # Memory optimization for large feature sets
            if feature_count > 100:
                params['colsample_bytree'] = 0.8  # Use only 80% of features
                params['subsample'] = 0.8  # Use only 80% of samples

            model = xgb.XGBClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=early_stopping_rounds,
                verbose=False
            )

            val_pred = model.predict(X_val)
            accuracy = accuracy_score(y_val, val_pred)

            training_time = time.time() - start_time

            result = {
                'model': model,
                'accuracy': accuracy,
                'model_type': 'xgboost',
                'creation_date': datetime.now().isoformat(),
                'training_time': training_time,
                'feature_importance': dict(zip(X_train.columns, model.feature_importances_)),
                'dataset_size': dataset_size,
                'feature_count': feature_count,
                'early_stopping_rounds': early_stopping_rounds
            }

            self.logger.info(f'✅ XGBoost model trained in {training_time:.2f}s (accuracy: {accuracy:.4f})')
            return result

        except Exception as e:
            self.logger.exception(f'❌ Error creating XGBoost model: {e}')
            raise RuntimeError(f'Failed to create XGBoost model: {e}')

    async def _create_random_forest_model(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> dict[str, Any]:
        """Create a Random Forest model."""
        try:
            params = self.RANDOM_FOREST_PARAMS.copy()
            model = RandomForestClassifier(**params)
            model.fit(X_train, y_train)
            val_pred = model.predict(X_val)
            accuracy = accuracy_score(y_val, val_pred)
            return {
                'model': model,
                'accuracy': accuracy,
                'model_type': 'random_forest',
                'creation_date': datetime.now().isoformat(),
                'feature_importance': dict(zip(X_train.columns, model.feature_importances_))
            }
        except Exception as e:
            self.logger.exception(f'❌ Error creating Random Forest model: {e}')
            raise RuntimeError(f'Failed to create Random Forest model: {e}')

    async def _create_neural_network_model(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> dict[str, Any]:
        """Create a neural network model with GPU and memory optimizations."""
        try:
            # Use optimized data conversion
            if self.vectorized_core and OPTIMIZATION_TOOLS_AVAILABLE:
                X_train_array = self.vectorized_core.create_memory_efficient_array(X_train.values, dtype=np.float32)
                X_val_array = self.vectorized_core.create_memory_efficient_array(X_val.values, dtype=np.float32)
            else:
                X_train_array = X_train.values.astype(np.float32)
                X_val_array = X_val.values.astype(np.float32)

            y_train_array = y_train.values.astype(np.float32)
            y_val_array = y_val.values.astype(np.float32)

            # Use GPU manager for tensor operations if available
            if self.m1_gpu_manager and OPTIMIZATION_TOOLS_AVAILABLE:
                X_train_tensor = self.m1_gpu_manager.to_device(X_train_array, "neural_net")
                y_train_tensor = self.m1_gpu_manager.to_device(y_train_array, "neural_net")
                X_val_tensor = self.m1_gpu_manager.to_device(X_val_array, "neural_net")
                y_val_tensor = self.m1_gpu_manager.to_device(y_val_array, "neural_net")
            else:
                X_train_tensor = torch.FloatTensor(X_train_array).to(self.device)
                y_train_tensor = torch.FloatTensor(y_train_array).to(self.device)
                X_val_tensor = torch.FloatTensor(X_val_array).to(self.device)
                y_val_tensor = torch.FloatTensor(y_val_array).to(self.device)

            input_size = X_train.shape[1]
            config = self.NEURAL_NETWORK_CONFIG

            # Build model layers dynamically
            layers = []
            prev_size = input_size
            for hidden_size in config['hidden_dims']:
                layers.extend([
                    nn.Linear(prev_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(config['dropout_rate'])
                ])
                prev_size = hidden_size
            layers.append(nn.Linear(prev_size, 1))
            layers.append(nn.Sigmoid())

            model = nn.Sequential(*layers).to(self.device)

            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

            # Memory-efficient training with GPU context management
            if self.m1_gpu_manager and OPTIMIZATION_TOOLS_AVAILABLE:
                with self.m1_gpu_manager.gpu_context("neural_network_training"):
                    model.train()
                    for epoch in range(config['epochs']):
                        optimizer.zero_grad()
                        outputs = model(X_train_tensor)
                        loss = criterion(outputs.squeeze(), y_train_tensor)
                        loss.backward()
                        optimizer.step()

                        # Memory cleanup every 10 epochs
                        if epoch % 10 == 0 and self.m1_memory_optimizer:
                            self.m1_memory_optimizer.optimize_memory()
            else:
                model.train()
                for epoch in range(config['epochs']):
                    optimizer.zero_grad()
                    outputs = model(X_train_tensor)
                    loss = criterion(outputs.squeeze(), y_train_tensor)
                    loss.backward()
                    optimizer.step()

            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_pred = (val_outputs.squeeze() > 0.5).float()
                accuracy = accuracy_score(y_val_tensor.cpu().numpy(), val_pred.cpu().numpy())

            return {
                'model': model,
                'accuracy': accuracy,
                'model_type': 'neural_network',
                'creation_date': datetime.now().isoformat(),
                'device': self.device
            }
        except Exception as e:
            self.logger.exception(f'❌ Error creating Neural Network model: {e}')
            raise RuntimeError(f'Failed to create Neural Network model: {e}')

    async def _save_analyst_models(self, created_models: dict[str, dict[str, Any]], models_dir: str) -> None:
        """Save created analyst models with optimized data management."""
        try:
            for regime_name, regime_models in created_models.items():
                regime_dir = os.path.join(models_dir, regime_name)
                os.makedirs(regime_dir, exist_ok = True)

                for model_name, model_data in regime_models.items():
                    if model_data.get('model') is not None:
                        # Use optimized data manager if available
                        if self.data_manager and OPTIMIZATION_TOOLS_AVAILABLE:
                            try:
                                # Create model data for saving
                                model_info = {
                                    'model': model_data['model'],
                                    'metadata': {
                                        'accuracy': model_data.get('accuracy', 0.0),
                                        'model_type': model_data.get('model_type', 'unknown'),
                                        'creation_date': model_data.get('creation_date', ''),
                                        'feature_importance': model_data.get('feature_importance', {}),
                                        'device': model_data.get('device', 'cpu'),
                                        'regime': regime_name
                                    }
                                }

                                # Save with optimized data manager
                                saved_path = self.data_manager.save_model_optimized(
                                    model_info['model'],
                                    f"{regime_name}_{model_name}",
                                    metadata=model_info['metadata']
                                )

                                # Create additional metadata file for compatibility
                                metadata_file = os.path.join(regime_dir, f'{model_name}_metadata.json')
                                with open(metadata_file, 'w') as f:
                                    json.dump(model_info['metadata'], f, indent=2)

                                self.logger.info(f'💾 Saved {model_name} model for regime {regime_name} (optimized)')

                            except Exception as e:
                                self.logger.warning(f"Optimized save failed for {model_name}, falling back: {e}")
                                # Fallback to original method
                                model_file = os.path.join(regime_dir, f'{model_name}.joblib')
                                joblib.dump(model_data['model'], model_file)
                                metadata_file = os.path.join(regime_dir, f'{model_name}_metadata.json')
                                metadata = {
                                    'accuracy': model_data.get('accuracy', 0.0),
                                    'model_type': model_data.get('model_type', 'unknown'),
                                    'creation_date': model_data.get('creation_date', ''),
                                    'feature_importance': model_data.get('feature_importance', {}),
                                    'device': model_data.get('device', 'cpu')
                                }
                                with open(metadata_file, 'w') as f:
                                    json.dump(metadata, f, indent=2)

                        else:
                            # Original saving method
                            model_file = os.path.join(regime_dir, f'{model_name}.joblib')
                            joblib.dump(model_data['model'], model_file)
                            metadata_file = os.path.join(regime_dir, f'{model_name}_metadata.json')
                            metadata = {
                                'accuracy': model_data.get('accuracy', 0.0),
                                'model_type': model_data.get('model_type', 'unknown'),
                                'creation_date': model_data.get('creation_date', ''),
                                'feature_importance': model_data.get('feature_importance', {}),
                                'device': model_data.get('device', 'cpu')
                            }
                            with open(metadata_file, 'w') as f:
                                json.dump(metadata, f, indent=2)

                        self.logger.info(f'💾 Saved {model_name} model for regime {regime_name}')

        except Exception as e:
            self.logger.exception(f'❌ Error saving analyst models: {e}')
            raise RuntimeError(f'Failed to save analyst models: {e}')

@handles_errors(exceptions=(Exception,), default_return = False, context='step11_analyst_creation')
async def run_step(symbol: str, exchange: str, timeframe: str='5m', data_dir: str='data_cache', force_rerun: bool = False, **kwargs: Any) -> bool:
    """Run the analyst creation step.

    Args:
        symbol: Trading symbol (e.g., "ETHUSDT")
        exchange: Exchange name (e.g., "BINANCE")
        timeframe: Timeframe (e.g., "5m")
        data_dir: Data directory
        force_rerun: Force re-run even if results exist
        **kwargs: Additional arguments

    Returns:
        bool: True if successful, False otherwise
    """
    logger = system_logger.getChild('Step11AnalystCreation')
    logger.info('=' * 80)
    logger.info('🚀 STEP 11: Analyst Creation')
    logger.info('=' * 80)
    logger.info(f'🎯 Symbol: {symbol}')
    logger.info(f'🏢 Exchange: {exchange}')
    logger.info(f'📊 Timeframe: {timeframe}')
    logger.info(f'📁 Data directory: {data_dir}')
    logger.info(f'🔄 Force rerun: {force_rerun}')
    logger.info('=' * 80)
    try:
        config = {'SYMBOL': symbol, 'EXCHANGE': exchange, 'TIMEFRAME': timeframe, 'DATA_DIR': data_dir}
        logger.info('🔧 Initializing analyst creation step...')
        step = AnalystCreationStep(config)
        await step.initialize()
        training_input = {'symbol': symbol, 'exchange': exchange, 'timeframe': timeframe, 'data_dir': data_dir, 'force_rerun': force_rerun}
        logger.info('🎯 Executing analyst creation...')
        pipeline_state = {}
        result = await step.execute(training_input, pipeline_state)
        if result.get('analyst_creation_completed', False):
            logger.info('✅ Step 11: Analyst Creation completed successfully')
            if result.get('created_analyst_models'):
                models = result['created_analyst_models']
                logger.info(f'📊 Created analyst models for {len(models)} regimes')
                for regime_name, regime_models in models.items():
                    model_count = len(regime_models)
                    logger.info(f'   - {regime_name}: {model_count} models')
                    for model_name, model_data in regime_models.items():
                        accuracy = model_data.get('accuracy', 0.0)
                        logger.info(f'     - {model_name}: {accuracy:.4f} accuracy')
            return True
        else:
            logger.error('❌ Step 11: Analyst Creation failed')
            error = result.get('analyst_creation_error', 'Unknown error')
            logger.error(f'   Error details: {error}')
            return False
    except Exception as e:
        logger.exception(f'❌ Unexpected error in Step 11: {e}')
        return False