"""
Detailed Pipeline Reporter for UnifiedDataDrivenPipeline

This module provides comprehensive reporting capabilities for the unified data-driven pipeline,
including detailed metrics about feature selection, feature creation, transforms, interactions,
and global pipeline performance.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from pathlib import Path

from ..utils.tprint import tprint_info, tprint_success, tprint_warning, tprint_error


@dataclass
class FeatureMetrics:
    """Metrics for individual features."""
    feature_name: str
    feature_type: str
    parent_features: List[str]
    transform_type: Optional[str] = None
    interaction_type: Optional[str] = None
    lookback_period: Optional[int] = None
    mutual_information: Optional[float] = None
    shap_score: Optional[float] = None
    lgbm_score: Optional[float] = None
    correlation_with_target: Optional[float] = None
    variance_score: Optional[float] = None
    stability_score: Optional[float] = None
    created_at_step: str = "unknown"
    is_selected: bool = False
    selection_rank: Optional[int] = None


@dataclass
class StepMetrics:
    """Metrics for each pipeline step."""
    step_name: str
    step_order: int
    input_features_count: int
    output_features_count: int
    features_selected: List[str]
    features_filtered: List[str]
    features_created: List[FeatureMetrics]
    top_features: List[Tuple[str, float]]  # (feature_name, score)
    top_transform_types: List[Tuple[str, int]]  # (transform_type, count)
    top_interaction_types: List[Tuple[str, int]]  # (interaction_type, count)
    top_lookback_periods: List[Tuple[int, int]]  # (lookback_period, count)
    execution_time: float
    memory_usage_mb: float
    success: bool
    error_message: Optional[str] = None
    additional_metrics: Dict[str, Any] = None


@dataclass
class GlobalMetrics:
    """Global pipeline metrics."""
    total_execution_time: float
    total_features_created: int
    total_features_selected: int
    total_interactions_generated: int
    total_htf_interactions: int
    total_lookback_optimizations: int
    peak_memory_usage_mb: float
    average_memory_usage_mb: float
    vectorbt_operations: int
    pandas_fallbacks: int
    cache_hit_rate: float
    optimization_iterations: int
    convergence_achieved: bool
    feature_diversity_score: float
    pipeline_success_rate: float
    cross_validation_splits: int
    candidates_evaluated: int


@dataclass
class DetailedPipelineReport:
    """Comprehensive pipeline report."""
    report_id: str
    generated_at: datetime
    pipeline_config: Dict[str, Any]
    data_info: Dict[str, Any]
    step_metrics: List[StepMetrics]
    global_metrics: GlobalMetrics
    feature_analysis: Dict[str, Any]
    performance_analysis: Dict[str, Any]
    recommendations: List[str]
    warnings: List[str]
    errors: List[str]


class DetailedPipelineReporter:
    """Comprehensive reporter for the unified data-driven pipeline."""
    
    def __init__(self, outcomes_dir: str = "outcomes"):
        """Initialize the reporter.
        
        Args:
            outcomes_dir: Directory to store outcome files
        """
        self.outcomes_dir = Path(outcomes_dir)
        self.outcomes_dir.mkdir(exist_ok=True)
        
        # Initialize metrics collection
        self.step_metrics: List[StepMetrics] = []
        self.feature_metrics: Dict[str, FeatureMetrics] = {}
        self.current_step = 0
        
    def start_step(self, step_name: str, input_features_count: int) -> None:
        """Start tracking a new pipeline step.
        
        Args:
            step_name: Name of the pipeline step
            input_features_count: Number of input features
        """
        self.current_step += 1
        tprint_info(f"📊 Starting metrics collection for step: {step_name}")
        
        # Create step metrics entry
        step_metric = StepMetrics(
            step_name=step_name,
            step_order=self.current_step,
            input_features_count=input_features_count,
            output_features_count=0,
            features_selected=[],
            features_filtered=[],
            features_created=[],
            top_features=[],
            top_transform_types=[],
            top_interaction_types=[],
            top_lookback_periods=[],
            execution_time=0.0,
            memory_usage_mb=0.0,
            success=False,
            additional_metrics={}
        )
        
        self.step_metrics.append(step_metric)
        
    def end_step(self, step_name: str, output_features_count: int, 
                 execution_time: float, memory_usage_mb: float, 
                 success: bool = True, error_message: Optional[str] = None) -> None:
        """End tracking for the current pipeline step.
        
        Args:
            step_name: Name of the pipeline step
            output_features_count: Number of output features
            execution_time: Execution time in seconds
            memory_usage_mb: Memory usage in MB
            success: Whether the step was successful
            error_message: Error message if step failed
        """
        if not self.step_metrics:
            tprint_warning("⚠️ No active step to end")
            return
            
        current_step = self.step_metrics[-1]
        if current_step.step_name != step_name:
            tprint_warning(f"⚠️ Step name mismatch: expected {current_step.step_name}, got {step_name}")
            
        # Update step metrics
        current_step.output_features_count = output_features_count
        current_step.execution_time = execution_time
        current_step.memory_usage_mb = memory_usage_mb
        current_step.success = success
        current_step.error_message = error_message
        
        # Calculate top features, transforms, etc.
        self._calculate_step_summaries(current_step)
        
        tprint_success(f"✅ Step {step_name} completed: {output_features_count} features, {execution_time:.3f}s")
        
    def track_feature_selection(self, selected_features: List[str], 
                               feature_importance: Dict[str, float],
                               selection_metrics: Dict[str, Any]) -> None:
        """Track feature selection results.
        
        Args:
            selected_features: List of selected feature names
            feature_importance: Feature importance scores
            selection_metrics: Additional selection metrics
        """
        if not self.step_metrics:
            return
            
        current_step = self.step_metrics[-1]
        current_step.features_selected = selected_features.copy()
        
        # Update feature metrics
        for i, feature_name in enumerate(selected_features):
            if feature_name in self.feature_metrics:
                self.feature_metrics[feature_name].is_selected = True
                self.feature_metrics[feature_name].selection_rank = i + 1
                self.feature_metrics[feature_name].lgbm_score = feature_importance.get(feature_name, 0.0)
        
        # Store additional metrics
        current_step.additional_metrics.update(selection_metrics)
        
    def track_feature_filtering(self, filtered_features: List[str], 
                               filter_reason: str) -> None:
        """Track feature filtering results.
        
        Args:
            filtered_features: List of filtered out feature names
            filter_reason: Reason for filtering
        """
        if not self.step_metrics:
            return
            
        current_step = self.step_metrics[-1]
        current_step.features_filtered = filtered_features.copy()
        
        # Update additional metrics
        if 'filtering_reasons' not in current_step.additional_metrics:
            current_step.additional_metrics['filtering_reasons'] = {}
        current_step.additional_metrics['filtering_reasons'][filter_reason] = len(filtered_features)
        
    def track_feature_creation(self, feature_name: str, feature_type: str,
                              parent_features: List[str], transform_type: Optional[str] = None,
                              interaction_type: Optional[str] = None, 
                              lookback_period: Optional[int] = None,
                              mutual_information: Optional[float] = None,
                              shap_score: Optional[float] = None,
                              correlation_with_target: Optional[float] = None,
                              variance_score: Optional[float] = None,
                              stability_score: Optional[float] = None) -> None:
        """Track feature creation with detailed metrics.
        
        Args:
            feature_name: Name of the created feature
            feature_type: Type of the feature
            parent_features: Parent features used to create this feature
            transform_type: Type of transform applied
            interaction_type: Type of interaction if applicable
            lookback_period: Lookback period if applicable
            mutual_information: Mutual information score
            shap_score: SHAP score
            correlation_with_target: Correlation with target
            variance_score: Variance score
            stability_score: Stability score
        """
        if not self.step_metrics:
            return
            
        # Create feature metrics
        feature_metric = FeatureMetrics(
            feature_name=feature_name,
            feature_type=feature_type,
            parent_features=parent_features,
            transform_type=transform_type,
            interaction_type=interaction_type,
            lookback_period=lookback_period,
            mutual_information=mutual_information,
            shap_score=shap_score,
            correlation_with_target=correlation_with_target,
            variance_score=variance_score,
            stability_score=stability_score,
            created_at_step=self.step_metrics[-1].step_name
        )
        
        # Store feature metrics
        self.feature_metrics[feature_name] = feature_metric
        self.step_metrics[-1].features_created.append(feature_metric)
        
    def track_interaction_generation(self, interactions: List[Any], 
                                   interaction_metrics: Dict[str, Any]) -> None:
        """Track interaction generation results.
        
        Args:
            interactions: List of generated interactions
            interaction_metrics: Interaction-specific metrics
        """
        if not self.step_metrics:
            return
            
        current_step = self.step_metrics[-1]
        current_step.additional_metrics.update(interaction_metrics)
        
        # Track interaction types
        interaction_types = {}
        for interaction in interactions:
            if hasattr(interaction, 'interaction_type'):
                interaction_type = interaction.interaction_type
                interaction_types[interaction_type] = interaction_types.get(interaction_type, 0) + 1
        
        current_step.top_interaction_types = sorted(interaction_types.items(), 
                                                   key=lambda x: x[1], reverse=True)
        
    def track_lookback_optimization(self, optimized_lookbacks: Dict[str, int],
                                  lookback_metrics: Dict[str, Any]) -> None:
        """Track lookback optimization results.
        
        Args:
            optimized_lookbacks: Optimized lookback periods
            lookback_metrics: Lookback-specific metrics
        """
        if not self.step_metrics:
            return
            
        current_step = self.step_metrics[-1]
        current_step.additional_metrics.update(lookback_metrics)
        
        # Track lookback periods
        lookback_counts = {}
        for feature, lookback in optimized_lookbacks.items():
            lookback_counts[lookback] = lookback_counts.get(lookback, 0) + 1
            
        current_step.top_lookback_periods = sorted(lookback_counts.items(),
                                                  key=lambda x: x[1], reverse=True)
        
    def _calculate_step_summaries(self, step: StepMetrics) -> None:
        """Calculate summary statistics for a step.
        
        Args:
            step: Step metrics to update
        """
        # Calculate top features by score
        feature_scores = []
        for feature in step.features_created:
            if feature.lgbm_score is not None:
                feature_scores.append((feature.feature_name, feature.lgbm_score))
            elif feature.shap_score is not None:
                feature_scores.append((feature.feature_name, feature.shap_score))
            elif feature.mutual_information is not None:
                feature_scores.append((feature.feature_name, feature.mutual_information))
        
        step.top_features = sorted(feature_scores, key=lambda x: x[1], reverse=True)[:10]
        
        # Calculate top transform types
        transform_types = {}
        for feature in step.features_created:
            if feature.transform_type:
                transform_types[feature.transform_type] = transform_types.get(feature.transform_type, 0) + 1
        
        step.top_transform_types = sorted(transform_types.items(), 
                                         key=lambda x: x[1], reverse=True)
        
    def generate_detailed_report(self, pipeline_result: Any, 
                               pipeline_config: Dict[str, Any],
                               data_info: Dict[str, Any]) -> DetailedPipelineReport:
        """Generate a comprehensive pipeline report.
        
        Args:
            pipeline_result: Result from the unified pipeline
            pipeline_config: Pipeline configuration
            data_info: Information about the input data
            
        Returns:
            DetailedPipelineReport: Comprehensive report
        """
        tprint_info("📊 Generating detailed pipeline report...")
        
        # Calculate global metrics
        global_metrics = self._calculate_global_metrics(pipeline_result)
        
        # Analyze features
        feature_analysis = self._analyze_features()
        
        # Analyze performance
        performance_analysis = self._analyze_performance()
        
        # Generate recommendations
        recommendations = self._generate_recommendations()
        
        # Collect warnings and errors
        warnings, errors = self._collect_warnings_and_errors()
        
        # Create report
        report = DetailedPipelineReport(
            report_id=f"pipeline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            generated_at=datetime.now(),
            pipeline_config=pipeline_config,
            data_info=data_info,
            step_metrics=self.step_metrics.copy(),
            global_metrics=global_metrics,
            feature_analysis=feature_analysis,
            performance_analysis=performance_analysis,
            recommendations=recommendations,
            warnings=warnings,
            errors=errors
        )
        
        return report
        
    def _calculate_global_metrics(self, pipeline_result: Any) -> GlobalMetrics:
        """Calculate global pipeline metrics.
        
        Args:
            pipeline_result: Result from the unified pipeline
            
        Returns:
            GlobalMetrics: Global metrics
        """
        total_features_created = sum(len(step.features_created) for step in self.step_metrics)
        total_features_selected = sum(len(step.features_selected) for step in self.step_metrics)
        total_execution_time = sum(step.execution_time for step in self.step_metrics)
        total_memory_usage = sum(step.memory_usage_mb for step in self.step_metrics)
        successful_steps = sum(1 for step in self.step_metrics if step.success)
        
        return GlobalMetrics(
            total_execution_time=total_execution_time,
            total_features_created=total_features_created,
            total_features_selected=total_features_selected,
            total_interactions_generated=len(pipeline_result.generated_interactions) if hasattr(pipeline_result, 'generated_interactions') else 0,
            total_htf_interactions=len(pipeline_result.htf_interactions) if hasattr(pipeline_result, 'htf_interactions') else 0,
            total_lookback_optimizations=len(pipeline_result.optimized_lookbacks) if hasattr(pipeline_result, 'optimized_lookbacks') else 0,
            peak_memory_usage_mb=max(step.memory_usage_mb for step in self.step_metrics) if self.step_metrics else 0.0,
            average_memory_usage_mb=total_memory_usage / len(self.step_metrics) if self.step_metrics else 0.0,
            vectorbt_operations=getattr(pipeline_result, 'vectorbt_operations', 0),
            pandas_fallbacks=getattr(pipeline_result, 'pandas_fallbacks', 0),
            cache_hit_rate=getattr(pipeline_result, 'cache_hit_rate', 0.0),
            optimization_iterations=getattr(pipeline_result, 'optimization_iterations', 0),
            convergence_achieved=getattr(pipeline_result, 'convergence_achieved', False),
            feature_diversity_score=getattr(pipeline_result, 'feature_diversity_score', 0.0),
            pipeline_success_rate=successful_steps / len(self.step_metrics) if self.step_metrics else 0.0,
            cross_validation_splits=getattr(pipeline_result, 'n_cv_splits', 0),
            candidates_evaluated=getattr(pipeline_result, 'n_candidates_evaluated', 0)
        )
        
    def _analyze_features(self) -> Dict[str, Any]:
        """Analyze feature characteristics.
        
        Returns:
            Dict containing feature analysis
        """
        if not self.feature_metrics:
            return {}
            
        # Analyze by feature type
        feature_types = {}
        transform_types = {}
        interaction_types = {}
        lookback_periods = {}
        
        for feature in self.feature_metrics.values():
            # Feature types
            feature_types[feature.feature_type] = feature_types.get(feature.feature_type, 0) + 1
            
            # Transform types
            if feature.transform_type:
                transform_types[feature.transform_type] = transform_types.get(feature.transform_type, 0) + 1
                
            # Interaction types
            if feature.interaction_type:
                interaction_types[feature.interaction_type] = interaction_types.get(feature.interaction_type, 0) + 1
                
            # Lookback periods
            if feature.lookback_period:
                lookback_periods[feature.lookback_period] = lookback_periods.get(feature.lookback_period, 0) + 1
        
        # Calculate score distributions
        lgbm_scores = [f.lgbm_score for f in self.feature_metrics.values() if f.lgbm_score is not None]
        shap_scores = [f.shap_score for f in self.feature_metrics.values() if f.shap_score is not None]
        mi_scores = [f.mutual_information for f in self.feature_metrics.values() if f.mutual_information is not None]
        
        return {
            'feature_types': feature_types,
            'transform_types': transform_types,
            'interaction_types': interaction_types,
            'lookback_periods': lookback_periods,
            'score_distributions': {
                'lgbm_scores': {
                    'mean': np.mean(lgbm_scores) if lgbm_scores else 0.0,
                    'std': np.std(lgbm_scores) if lgbm_scores else 0.0,
                    'min': np.min(lgbm_scores) if lgbm_scores else 0.0,
                    'max': np.max(lgbm_scores) if lgbm_scores else 0.0
                },
                'shap_scores': {
                    'mean': np.mean(shap_scores) if shap_scores else 0.0,
                    'std': np.std(shap_scores) if shap_scores else 0.0,
                    'min': np.min(shap_scores) if shap_scores else 0.0,
                    'max': np.max(shap_scores) if shap_scores else 0.0
                },
                'mutual_information': {
                    'mean': np.mean(mi_scores) if mi_scores else 0.0,
                    'std': np.std(mi_scores) if mi_scores else 0.0,
                    'min': np.min(mi_scores) if mi_scores else 0.0,
                    'max': np.max(mi_scores) if mi_scores else 0.0
                }
            },
            'total_features': len(self.feature_metrics),
            'selected_features': len([f for f in self.feature_metrics.values() if f.is_selected])
        }
        
    def _analyze_performance(self) -> Dict[str, Any]:
        """Analyze pipeline performance.
        
        Returns:
            Dict containing performance analysis
        """
        if not self.step_metrics:
            return {}
            
        execution_times = [step.execution_time for step in self.step_metrics]
        memory_usage = [step.memory_usage_mb for step in self.step_metrics]
        
        return {
            'execution_time_analysis': {
                'total': sum(execution_times),
                'mean': np.mean(execution_times),
                'std': np.std(execution_times),
                'min': np.min(execution_times),
                'max': np.max(execution_times),
                'slowest_step': self.step_metrics[np.argmax(execution_times)].step_name if execution_times else None
            },
            'memory_usage_analysis': {
                'peak': max(memory_usage) if memory_usage else 0.0,
                'mean': np.mean(memory_usage) if memory_usage else 0.0,
                'std': np.std(memory_usage) if memory_usage else 0.0,
                'total': sum(memory_usage)
            },
            'step_success_rate': len([s for s in self.step_metrics if s.success]) / len(self.step_metrics),
            'bottleneck_analysis': self._identify_bottlenecks()
        }
        
    def _identify_bottlenecks(self) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks.
        
        Returns:
            List of bottleneck information
        """
        bottlenecks = []
        
        if not self.step_metrics:
            return bottlenecks
            
        execution_times = [step.execution_time for step in self.step_metrics]
        memory_usage = [step.memory_usage_mb for step in self.step_metrics]
        
        if execution_times:
            mean_time = np.mean(execution_times)
            std_time = np.std(execution_times)
            
            for i, step in enumerate(self.step_metrics):
                if step.execution_time > mean_time + 2 * std_time:
                    bottlenecks.append({
                        'type': 'execution_time',
                        'step': step.step_name,
                        'value': step.execution_time,
                        'threshold': mean_time + 2 * std_time,
                        'severity': 'high' if step.execution_time > mean_time + 3 * std_time else 'medium'
                    })
        
        if memory_usage:
            mean_memory = np.mean(memory_usage)
            std_memory = np.std(memory_usage)
            
            for i, step in enumerate(self.step_metrics):
                if step.memory_usage_mb > mean_memory + 2 * std_memory:
                    bottlenecks.append({
                        'type': 'memory_usage',
                        'step': step.step_name,
                        'value': step.memory_usage_mb,
                        'threshold': mean_memory + 2 * std_memory,
                        'severity': 'high' if step.memory_usage_mb > mean_memory + 3 * std_memory else 'medium'
                    })
        
        return bottlenecks
        
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on analysis.
        
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if not self.step_metrics:
            return recommendations
            
        # Check for failed steps
        failed_steps = [step for step in self.step_metrics if not step.success]
        if failed_steps:
            recommendations.append(f"Address {len(failed_steps)} failed steps to improve pipeline reliability")
        
        # Check for performance bottlenecks
        bottlenecks = self._identify_bottlenecks()
        if bottlenecks:
            recommendations.append(f"Optimize {len(bottlenecks)} identified performance bottlenecks")
        
        # Check feature diversity
        feature_analysis = self._analyze_features()
        if feature_analysis.get('feature_diversity_score', 0) < 0.5:
            recommendations.append("Improve feature diversity to enhance model robustness")
        
        # Check memory usage
        if self.step_metrics:
            avg_memory = np.mean([step.memory_usage_mb for step in self.step_metrics])
            if avg_memory > 1000:  # 1GB threshold
                recommendations.append("Consider memory optimization strategies for large datasets")
        
        return recommendations
        
    def _collect_warnings_and_errors(self) -> Tuple[List[str], List[str]]:
        """Collect warnings and errors from the pipeline.
        
        Returns:
            Tuple of (warnings, errors)
        """
        warnings = []
        errors = []
        
        for step in self.step_metrics:
            if not step.success and step.error_message:
                errors.append(f"{step.step_name}: {step.error_message}")
            
            # Check for potential issues
            if step.output_features_count == 0:
                warnings.append(f"{step.step_name}: No features generated")
            
            if step.execution_time > 300:  # 5 minutes
                warnings.append(f"{step.step_name}: Long execution time ({step.execution_time:.1f}s)")
        
        return warnings, errors
        
    def save_report(self, report: DetailedPipelineReport, 
                   format: str = "json") -> str:
        """Save the report to file.
        
        Args:
            report: Report to save
            format: Output format ("json", "txt", "both")
            
        Returns:
            str: Path to saved file
        """
        timestamp = report.generated_at.strftime("%Y%m%d_%H%M%S")
        
        if format in ["json", "both"]:
            json_path = self.outcomes_dir / f"unified_pipeline_detailed_report_{timestamp}.json"
            
            # Convert dataclass to dict for JSON serialization
            report_dict = asdict(report)
            report_dict['generated_at'] = report.generated_at.isoformat()
            
            with open(json_path, 'w') as f:
                json.dump(report_dict, f, indent=2, default=str)
            
            tprint_success(f"✅ Detailed report saved to: {json_path}")
        
        if format in ["txt", "both"]:
            txt_path = self.outcomes_dir / f"unified_pipeline_detailed_report_{timestamp}.txt"
            self._save_human_readable_report(report, txt_path)
            
            tprint_success(f"✅ Human-readable report saved to: {txt_path}")
        
        return str(self.outcomes_dir / f"unified_pipeline_detailed_report_{timestamp}")
        
    def _save_human_readable_report(self, report: DetailedPipelineReport, 
                                   file_path: Path) -> None:
        """Save a human-readable version of the report.
        
        Args:
            report: Report to save
            file_path: Path to save the file
        """
        with open(file_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("UNIFIED DATA-DRIVEN PIPELINE DETAILED REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Report ID: {report.report_id}\n")
            f.write(f"Generated at: {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Pipeline Success Rate: {report.global_metrics.pipeline_success_rate:.2%}\n")
            f.write(f"Total Execution Time: {report.global_metrics.total_execution_time:.2f} seconds\n")
            f.write(f"Total Features Created: {report.global_metrics.total_features_created}\n")
            f.write(f"Total Features Selected: {report.global_metrics.total_features_selected}\n\n")
            
            # Data Information
            f.write("DATA INFORMATION\n")
            f.write("-" * 40 + "\n")
            for key, value in report.data_info.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
            
            # Step-by-step analysis
            f.write("STEP-BY-STEP ANALYSIS\n")
            f.write("-" * 40 + "\n")
            for step in report.step_metrics:
                f.write(f"\nStep {step.step_order}: {step.step_name}\n")
                f.write(f"  Input Features: {step.input_features_count}\n")
                f.write(f"  Output Features: {step.output_features_count}\n")
                f.write(f"  Features Selected: {len(step.features_selected)}\n")
                f.write(f"  Features Created: {len(step.features_created)}\n")
                f.write(f"  Execution Time: {step.execution_time:.2f}s\n")
                f.write(f"  Memory Usage: {step.memory_usage_mb:.2f} MB\n")
                f.write(f"  Success: {step.success}\n")
                
                if step.top_features:
                    f.write(f"  Top Features: {', '.join([f'{name}({score:.3f})' for name, score in step.top_features[:5]])}\n")
                
                if step.top_transform_types:
                    f.write(f"  Top Transform Types: {', '.join([f'{ttype}({count})' for ttype, count in step.top_transform_types[:3]])}\n")
                
                if step.top_interaction_types:
                    f.write(f"  Top Interaction Types: {', '.join([f'{itype}({count})' for itype, count in step.top_interaction_types[:3]])}\n")
                
                if step.top_lookback_periods:
                    f.write(f"  Top Lookback Periods: {', '.join([f'{period}({count})' for period, count in step.top_lookback_periods[:3]])}\n")
            
            # Feature Analysis
            f.write("\n\nFEATURE ANALYSIS\n")
            f.write("-" * 40 + "\n")
            if report.feature_analysis:
                f.write(f"Total Features: {report.feature_analysis.get('total_features', 0)}\n")
                f.write(f"Selected Features: {report.feature_analysis.get('selected_features', 0)}\n")
                
                if 'feature_types' in report.feature_analysis:
                    f.write("\nFeature Types:\n")
                    for ftype, count in sorted(report.feature_analysis['feature_types'].items(), key=lambda x: x[1], reverse=True):
                        f.write(f"  {ftype}: {count}\n")
                
                if 'transform_types' in report.feature_analysis:
                    f.write("\nTransform Types:\n")
                    for ttype, count in sorted(report.feature_analysis['transform_types'].items(), key=lambda x: x[1], reverse=True):
                        f.write(f"  {ttype}: {count}\n")
                
                if 'interaction_types' in report.feature_analysis:
                    f.write("\nInteraction Types:\n")
                    for itype, count in sorted(report.feature_analysis['interaction_types'].items(), key=lambda x: x[1], reverse=True):
                        f.write(f"  {itype}: {count}\n")
            
            # Performance Analysis
            f.write("\n\nPERFORMANCE ANALYSIS\n")
            f.write("-" * 40 + "\n")
            if report.performance_analysis:
                exec_analysis = report.performance_analysis.get('execution_time_analysis', {})
                f.write(f"Total Execution Time: {exec_analysis.get('total', 0):.2f}s\n")
                f.write(f"Average Step Time: {exec_analysis.get('mean', 0):.2f}s\n")
                f.write(f"Slowest Step: {exec_analysis.get('slowest_step', 'N/A')}\n")
                
                mem_analysis = report.performance_analysis.get('memory_usage_analysis', {})
                f.write(f"Peak Memory Usage: {mem_analysis.get('peak', 0):.2f} MB\n")
                f.write(f"Average Memory Usage: {mem_analysis.get('mean', 0):.2f} MB\n")
            
            # Recommendations
            if report.recommendations:
                f.write("\n\nRECOMMENDATIONS\n")
                f.write("-" * 40 + "\n")
                for i, rec in enumerate(report.recommendations, 1):
                    f.write(f"{i}. {rec}\n")
            
            # Warnings and Errors
            if report.warnings:
                f.write("\n\nWARNINGS\n")
                f.write("-" * 40 + "\n")
                for warning in report.warnings:
                    f.write(f"⚠️ {warning}\n")
            
            if report.errors:
                f.write("\n\nERRORS\n")
                f.write("-" * 40 + "\n")
                for error in report.errors:
                    f.write(f"❌ {error}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")