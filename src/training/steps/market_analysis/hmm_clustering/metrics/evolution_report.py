"""
Metrics evolution reporting for HMM clustering pipeline.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json
import logging

try:
    from src.utils.matrix_operations import (
        get_unified_matrix_operations,
        get_vectorized_processing_core,
        get_enhanced_matrix_operations,
        get_batch_matrix_processor
    )
    MATRIX_OPERATIONS_AVAILABLE = True
except ImportError:
    MATRIX_OPERATIONS_AVAILABLE = False

try:
    from src.utils.hardware import (
        get_hardware_accelerator,
        get_memory_manager,
        get_performance_monitor
    )
    HARDWARE_ACCELERATION_AVAILABLE = True
except ImportError:
    HARDWARE_ACCELERATION_AVAILABLE = False

from .basic_metrics import BasicClusteringMetrics, BasicMetricsResult
from .detailed_metrics import DetailedClusteringMetrics, DetailedMetricsResult

# Import MSM-specific metrics if available
try:
    from .msm_metrics import MSMSpecificMetrics, MSMReport
    MSM_METRICS_AVAILABLE = True
except ImportError:
    MSM_METRICS_AVAILABLE = False
    MSMSpecificMetrics = None
    MSMReport = None

logger = logging.getLogger(__name__)


@dataclass
class MetricsEvolutionReport:
    """Comprehensive report of clustering metrics evolution across pipeline steps."""
    
    # Basic metrics evolution
    silhouette_evolution: List[Dict[str, Any]]
    cluster_cv_evolution: List[Dict[str, Any]]
    cluster_count_evolution: List[Dict[str, Any]]
    noise_percentage_evolution: List[Dict[str, Any]]
    
    # Detailed metrics for final enhanced clustering
    enhanced_clustering_metrics: Optional[Dict[str, Any]] = None
    
    # Performance summary
    performance_summary: Dict[str, Any] = None
    
    # Step-by-step analysis
    step_analysis: List[Dict[str, Any]] = None
    
    # Quality improvement tracking
    quality_improvements: Dict[str, Any] = None
    
    # Report metadata
    report_timestamp: str = None
    pipeline_config: Dict[str, Any] = None


class MetricsEvolutionReporter:
    """Analyzes and generates comprehensive metrics evolution reports."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the metrics evolution reporter.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize metrics calculators
        self.basic_metrics_calc = BasicClusteringMetrics(config)
        self.detailed_metrics_calc = DetailedClusteringMetrics(config)
        
        # Initialize hardware acceleration if available
        self.hardware_accelerator = None
        self.memory_manager = None
        self.performance_monitor = None
        
        if HARDWARE_ACCELERATION_AVAILABLE:
            try:
                self.hardware_accelerator = get_hardware_accelerator()
                self.memory_manager = get_memory_manager()
                self.performance_monitor = get_performance_monitor()
                self.logger.info("✅ Hardware acceleration initialized for metrics reporting")
            except Exception as e:
                self.logger.warning(f"⚠️ Hardware acceleration not available for metrics reporting: {e}")
        
        # Initialize matrix operations if available
        self.matrix_ops = None
        self.vectorized_core = None
        self.enhanced_ops = None
        self.batch_processor = None
        
        if MATRIX_OPERATIONS_AVAILABLE:
            try:
                self.matrix_ops = get_unified_matrix_operations()
                self.vectorized_core = get_vectorized_processing_core()
                self.enhanced_ops = get_enhanced_matrix_operations()
                self.batch_processor = get_batch_matrix_processor()
                self.logger.info("✅ Matrix operations initialized for metrics reporting")
            except Exception as e:
                self.logger.warning(f"⚠️ Matrix operations not available for metrics reporting: {e}")
        
        # Metrics thresholds for quality assessment
        self.silhouette_thresholds = {
            'excellent': 0.7,
            'good': 0.5,
            'fair': 0.3,
            'poor': 0.0
        }
        
        self.cv_thresholds = {
            'excellent': 0.2,
            'good': 0.4,
            'fair': 0.6,
            'poor': 1.0
        }

    def generate_comprehensive_report(
        self,
        standard_metrics_evolution: Dict[str, Any],
        enhanced_metrics: Optional[Dict[str, Any]] = None,
        pipeline_config: Optional[Dict[str, Any]] = None
    ) -> MetricsEvolutionReport:
        """Generate a comprehensive metrics evolution report.

        Args:
            standard_metrics_evolution: Metrics evolution from standard clustering
            enhanced_metrics: Detailed metrics from enhanced clustering
            pipeline_config: Pipeline configuration for context

        Returns:
            Comprehensive metrics evolution report
        """
        self.logger.info("📊 Generating comprehensive metrics evolution report...")
        
        # Monitor performance
        if self.performance_monitor:
            self.performance_monitor.start_monitoring("comprehensive_report_generation")
        
        try:
            # Extract basic metrics evolution
            silhouette_evolution = self._extract_silhouette_evolution(standard_metrics_evolution)
            cluster_cv_evolution = self._extract_cluster_cv_evolution(standard_metrics_evolution)
            cluster_count_evolution = self._extract_cluster_count_evolution(standard_metrics_evolution)
            noise_percentage_evolution = self._extract_noise_percentage_evolution(standard_metrics_evolution)
            
            # Generate performance summary
            performance_summary = self._generate_performance_summary(
                standard_metrics_evolution, enhanced_metrics
            )
            
            # Generate step-by-step analysis
            step_analysis = self._generate_step_analysis(standard_metrics_evolution)
            
            # Generate quality improvement tracking
            quality_improvements = self._generate_quality_improvements(
                standard_metrics_evolution, enhanced_metrics
            )
            
            # Create comprehensive report
            report = MetricsEvolutionReport(
                silhouette_evolution=silhouette_evolution,
                cluster_cv_evolution=cluster_cv_evolution,
                cluster_count_evolution=cluster_count_evolution,
                noise_percentage_evolution=noise_percentage_evolution,
                enhanced_clustering_metrics=enhanced_metrics,
                performance_summary=performance_summary,
                step_analysis=step_analysis,
                quality_improvements=quality_improvements,
                report_timestamp=datetime.now().isoformat(),
                pipeline_config=pipeline_config or self.config
            )
            
            # Stop performance monitoring
            if self.performance_monitor:
                perf_metrics = self.performance_monitor.stop_monitoring("comprehensive_report_generation")
                self.logger.info(f"📊 Report generation performance: {perf_metrics}")
            
            self.logger.info("✅ Comprehensive metrics evolution report generated")
            return report
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate comprehensive report: {e}")
            # Return empty report
            return MetricsEvolutionReport(
                silhouette_evolution=[],
                cluster_cv_evolution=[],
                cluster_count_evolution=[],
                noise_percentage_evolution=[],
                performance_summary={'error': str(e)},
                step_analysis=[],
                quality_improvements={'error': str(e)},
                report_timestamp=datetime.now().isoformat(),
                pipeline_config=pipeline_config or self.config
            )

    def _extract_silhouette_evolution(self, metrics_evolution: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract silhouette score evolution across steps."""
        silhouette_evolution = []
        
        for step_name, step_metrics in metrics_evolution.items():
            if 'basic_metrics' in step_metrics:
                basic_metrics = step_metrics['basic_metrics']
                if 'silhouette' in basic_metrics:
                    silhouette_evolution.append({
                        'step': step_name,
                        'silhouette_score': basic_metrics['silhouette'],
                        'quality_rating': self._rate_silhouette_quality(basic_metrics['silhouette']),
                        'timestamp': step_metrics.get('timestamp', 'unknown')
                    })
        
        return silhouette_evolution

    def _extract_cluster_cv_evolution(self, metrics_evolution: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract cluster CV evolution across steps."""
        cluster_cv_evolution = []
        
        for step_name, step_metrics in metrics_evolution.items():
            if 'basic_metrics' in step_metrics:
                basic_metrics = step_metrics['basic_metrics']
                if 'average_cluster_cv' in basic_metrics:
                    cluster_cv_evolution.append({
                        'step': step_name,
                        'average_cluster_cv': basic_metrics['average_cluster_cv'],
                        'quality_rating': self._rate_cv_quality(basic_metrics['average_cluster_cv']),
                        'timestamp': step_metrics.get('timestamp', 'unknown')
                    })
        
        return cluster_cv_evolution

    def _extract_cluster_count_evolution(self, metrics_evolution: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract cluster count evolution across steps."""
        cluster_count_evolution = []
        
        for step_name, step_metrics in metrics_evolution.items():
            if 'basic_metrics' in step_metrics:
                basic_metrics = step_metrics['basic_metrics']
                if 'n_clusters' in basic_metrics:
                    cluster_count_evolution.append({
                        'step': step_name,
                        'n_clusters': basic_metrics['n_clusters'],
                        'n_valid_points': basic_metrics.get('n_valid_points', 0),
                        'n_noise_points': basic_metrics.get('n_noise_points', 0),
                        'timestamp': step_metrics.get('timestamp', 'unknown')
                    })
        
        return cluster_count_evolution

    def _extract_noise_percentage_evolution(self, metrics_evolution: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract noise percentage evolution across steps."""
        noise_percentage_evolution = []
        
        for step_name, step_metrics in metrics_evolution.items():
            if 'basic_metrics' in step_metrics:
                basic_metrics = step_metrics['basic_metrics']
                n_valid = basic_metrics.get('n_valid_points', 0)
                n_noise = basic_metrics.get('n_noise_points', 0)
                total = n_valid + n_noise
                
                if total > 0:
                    noise_percentage = (n_noise / total) * 100
                    noise_percentage_evolution.append({
                        'step': step_name,
                        'noise_percentage': noise_percentage,
                        'n_noise_points': n_noise,
                        'n_valid_points': n_valid,
                        'timestamp': step_metrics.get('timestamp', 'unknown')
                    })
        
        return noise_percentage_evolution

    def _generate_performance_summary(
        self,
        standard_metrics_evolution: Dict[str, Any],
        enhanced_metrics: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate performance summary across all steps."""
        
        # Calculate overall performance metrics
        silhouette_scores = []
        cluster_cvs = []
        cluster_counts = []
        
        for step_metrics in standard_metrics_evolution.values():
            if 'basic_metrics' in step_metrics:
                basic_metrics = step_metrics['basic_metrics']
                if 'silhouette' in basic_metrics:
                    silhouette_scores.append(basic_metrics['silhouette'])
                if 'average_cluster_cv' in basic_metrics:
                    cluster_cvs.append(basic_metrics['average_cluster_cv'])
                if 'n_clusters' in basic_metrics:
                    cluster_counts.append(basic_metrics['n_clusters'])
        
        # Calculate statistics
        performance_summary = {
            'silhouette_statistics': {
                'mean': float(np.mean(silhouette_scores)) if silhouette_scores else 0.0,
                'std': float(np.std(silhouette_scores)) if silhouette_scores else 0.0,
                'min': float(np.min(silhouette_scores)) if silhouette_scores else 0.0,
                'max': float(np.max(silhouette_scores)) if silhouette_scores else 0.0,
                'final': float(silhouette_scores[-1]) if silhouette_scores else 0.0
            },
            'cluster_cv_statistics': {
                'mean': float(np.mean(cluster_cvs)) if cluster_cvs else 0.0,
                'std': float(np.std(cluster_cvs)) if cluster_cvs else 0.0,
                'min': float(np.min(cluster_cvs)) if cluster_cvs else 0.0,
                'max': float(np.max(cluster_cvs)) if cluster_cvs else 0.0,
                'final': float(cluster_cvs[-1]) if cluster_cvs else 0.0
            },
            'cluster_count_statistics': {
                'mean': float(np.mean(cluster_counts)) if cluster_counts else 0.0,
                'std': float(np.std(cluster_counts)) if cluster_counts else 0.0,
                'min': int(np.min(cluster_counts)) if cluster_counts else 0,
                'max': int(np.max(cluster_counts)) if cluster_counts else 0,
                'final': int(cluster_counts[-1]) if cluster_counts else 0
            },
            'enhanced_clustering_available': enhanced_metrics is not None,
            'total_steps_analyzed': len(standard_metrics_evolution),
            'matrix_ops_used': self.matrix_ops is not None,
            'hardware_acceleration_used': self.hardware_accelerator is not None
        }
        
        # Add enhanced clustering comparison if available
        if enhanced_metrics:
            performance_summary['enhanced_vs_standard'] = self._compare_enhanced_vs_standard(
                standard_metrics_evolution, enhanced_metrics
            )
        
        return performance_summary

    def _generate_step_analysis(self, metrics_evolution: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate detailed analysis for each step."""
        step_analysis = []
        
        for step_name, step_metrics in metrics_evolution.items():
            analysis = {
                'step_name': step_name,
                'timestamp': step_metrics.get('timestamp', 'unknown'),
                'basic_metrics': step_metrics.get('basic_metrics', {}),
                'step_specific_metrics': {k: v for k, v in step_metrics.items() if k != 'basic_metrics' and k != 'timestamp'},
                'quality_assessment': self._assess_step_quality(step_metrics),
                'improvements_suggestions': self._suggest_step_improvements(step_name, step_metrics)
            }
            step_analysis.append(analysis)
        
        return step_analysis

    def _generate_quality_improvements(
        self,
        standard_metrics_evolution: Dict[str, Any],
        enhanced_metrics: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate quality improvement tracking and recommendations."""
        
        improvements = {
            'silhouette_improvement': self._calculate_silhouette_improvement(standard_metrics_evolution),
            'cluster_cv_improvement': self._calculate_cluster_cv_improvement(standard_metrics_evolution),
            'cluster_count_stability': self._assess_cluster_count_stability(standard_metrics_evolution),
            'noise_handling_effectiveness': self._assess_noise_handling_effectiveness(standard_metrics_evolution),
            'overall_quality_trend': self._assess_overall_quality_trend(standard_metrics_evolution),
            'recommendations': self._generate_quality_recommendations(standard_metrics_evolution, enhanced_metrics)
        }
        
        return improvements

    def _rate_silhouette_quality(self, silhouette_score: float) -> str:
        """Rate silhouette score quality."""
        if silhouette_score >= self.silhouette_thresholds['excellent']:
            return 'excellent'
        elif silhouette_score >= self.silhouette_thresholds['good']:
            return 'good'
        elif silhouette_score >= self.silhouette_thresholds['fair']:
            return 'fair'
        else:
            return 'poor'

    def _rate_cv_quality(self, cv_score: float) -> str:
        """Rate cluster CV quality (lower is better)."""
        if cv_score <= self.cv_thresholds['excellent']:
            return 'excellent'
        elif cv_score <= self.cv_thresholds['good']:
            return 'good'
        elif cv_score <= self.cv_thresholds['fair']:
            return 'fair'
        else:
            return 'poor'

    def _assess_step_quality(self, step_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Assess quality of a specific step."""
        basic_metrics = step_metrics.get('basic_metrics', {})
        
        quality_assessment = {
            'silhouette_quality': self._rate_silhouette_quality(basic_metrics.get('silhouette', 0.0)),
            'cluster_cv_quality': self._rate_cv_quality(basic_metrics.get('average_cluster_cv', 1.0)),
            'cluster_count_appropriate': self._assess_cluster_count_appropriateness(basic_metrics.get('n_clusters', 0)),
            'noise_handling_quality': self._assess_noise_handling_quality(step_metrics),
            'overall_quality': 'unknown'
        }
        
        # Calculate overall quality
        quality_scores = []
        if basic_metrics.get('silhouette', 0.0) >= self.silhouette_thresholds['good']:
            quality_scores.append(1)
        if basic_metrics.get('average_cluster_cv', 1.0) <= self.cv_thresholds['good']:
            quality_scores.append(1)
        if quality_assessment['cluster_count_appropriate']:
            quality_scores.append(1)
        if quality_assessment['noise_handling_quality'] == 'good':
            quality_scores.append(1)
        
        overall_quality = len(quality_scores) / 4.0
        if overall_quality >= 0.75:
            quality_assessment['overall_quality'] = 'excellent'
        elif overall_quality >= 0.5:
            quality_assessment['overall_quality'] = 'good'
        elif overall_quality >= 0.25:
            quality_assessment['overall_quality'] = 'fair'
        else:
            quality_assessment['overall_quality'] = 'poor'
        
        return quality_assessment

    def _assess_cluster_count_appropriateness(self, n_clusters: int) -> bool:
        """Assess if cluster count is appropriate."""
        return 2 <= n_clusters <= 20

    def _assess_noise_handling_quality(self, step_metrics: Dict[str, Any]) -> str:
        """Assess noise handling quality."""
        basic_metrics = step_metrics.get('basic_metrics', {})
        n_noise = basic_metrics.get('n_noise_points', 0)
        n_valid = basic_metrics.get('n_valid_points', 0)
        total = n_noise + n_valid
        
        if total == 0:
            return 'unknown'
        
        noise_percentage = (n_noise / total) * 100
        
        # Consider noise handling good if noise percentage is reasonable (5-30%)
        if 5 <= noise_percentage <= 30:
            return 'good'
        elif noise_percentage < 5:
            return 'excellent'
        elif noise_percentage > 50:
            return 'poor'
        else:
            return 'fair'

    def _suggest_step_improvements(self, step_name: str, step_metrics: Dict[str, Any]) -> List[str]:
        """Suggest improvements for a specific step."""
        suggestions = []
        basic_metrics = step_metrics.get('basic_metrics', {})
        
        # Silhouette score suggestions
        silhouette = basic_metrics.get('silhouette', 0.0)
        if silhouette < self.silhouette_thresholds['good']:
            suggestions.append(f"Improve silhouette score (current: {silhouette:.3f}) - consider feature engineering or parameter tuning")
        
        # Cluster CV suggestions
        cluster_cv = basic_metrics.get('average_cluster_cv', 1.0)
        if cluster_cv > self.cv_thresholds['good']:
            suggestions.append(f"Reduce cluster size variation (current CV: {cluster_cv:.3f}) - consider constraint enforcement")
        
        # Cluster count suggestions
        n_clusters = basic_metrics.get('n_clusters', 0)
        if n_clusters < 2:
            suggestions.append("Increase cluster count - current clustering may be too aggressive")
        elif n_clusters > 20:
            suggestions.append("Reduce cluster count - current clustering may be too granular")
        
        # Noise handling suggestions
        n_noise = basic_metrics.get('n_noise_points', 0)
        n_valid = basic_metrics.get('n_valid_points', 0)
        total = n_noise + n_valid
        if total > 0:
            noise_percentage = (n_noise / total) * 100
            if noise_percentage > 50:
                suggestions.append(f"High noise percentage ({noise_percentage:.1f}%) - consider noise reduction parameters")
            elif noise_percentage < 5:
                suggestions.append(f"Low noise percentage ({noise_percentage:.1f}%) - consider if noise detection is working properly")
        
        return suggestions

    def _calculate_silhouette_improvement(self, metrics_evolution: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate silhouette score improvement across steps."""
        silhouette_scores = []
        step_names = []
        
        for step_name, step_metrics in metrics_evolution.items():
            if 'basic_metrics' in step_metrics and 'silhouette' in step_metrics['basic_metrics']:
                silhouette_scores.append(step_metrics['basic_metrics']['silhouette'])
                step_names.append(step_name)
        
        if len(silhouette_scores) < 2:
            return {'improvement': 0.0, 'trend': 'insufficient_data'}
        
        # Calculate improvement from first to last
        first_score = silhouette_scores[0]
        last_score = silhouette_scores[-1]
        improvement = last_score - first_score
        
        # Calculate trend
        if improvement > 0.1:
            trend = 'improving'
        elif improvement < -0.1:
            trend = 'declining'
        else:
            trend = 'stable'
        
        return {
            'improvement': float(improvement),
            'trend': trend,
            'first_score': float(first_score),
            'last_score': float(last_score),
            'improvement_percentage': float((improvement / first_score) * 100) if first_score != 0 else 0.0
        }

    def _calculate_cluster_cv_improvement(self, metrics_evolution: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate cluster CV improvement across steps (lower is better)."""
        cluster_cvs = []
        step_names = []
        
        for step_name, step_metrics in metrics_evolution.items():
            if 'basic_metrics' in step_metrics and 'average_cluster_cv' in step_metrics['basic_metrics']:
                cluster_cvs.append(step_metrics['basic_metrics']['average_cluster_cv'])
                step_names.append(step_name)
        
        if len(cluster_cvs) < 2:
            return {'improvement': 0.0, 'trend': 'insufficient_data'}
        
        # Calculate improvement from first to last (negative improvement is good for CV)
        first_cv = cluster_cvs[0]
        last_cv = cluster_cvs[-1]
        improvement = first_cv - last_cv  # Negative improvement means CV got worse
        
        # Calculate trend
        if improvement > 0.1:
            trend = 'improving'  # CV decreased (good)
        elif improvement < -0.1:
            trend = 'declining'  # CV increased (bad)
        else:
            trend = 'stable'
        
        return {
            'improvement': float(improvement),
            'trend': trend,
            'first_cv': float(first_cv),
            'last_cv': float(last_cv),
            'improvement_percentage': float((improvement / first_cv) * 100) if first_cv != 0 else 0.0
        }

    def _assess_cluster_count_stability(self, metrics_evolution: Dict[str, Any]) -> Dict[str, Any]:
        """Assess cluster count stability across steps."""
        cluster_counts = []
        
        for step_metrics in metrics_evolution.values():
            if 'basic_metrics' in step_metrics and 'n_clusters' in step_metrics['basic_metrics']:
                cluster_counts.append(step_metrics['basic_metrics']['n_clusters'])
        
        if len(cluster_counts) < 2:
            return {'stability': 'insufficient_data', 'variance': 0.0}
        
        variance = float(np.var(cluster_counts))
        std_dev = float(np.std(cluster_counts))
        mean_count = float(np.mean(cluster_counts))
        
        # Assess stability
        if std_dev <= 1.0:
            stability = 'highly_stable'
        elif std_dev <= 2.0:
            stability = 'stable'
        elif std_dev <= 3.0:
            stability = 'moderately_stable'
        else:
            stability = 'unstable'
        
        return {
            'stability': stability,
            'variance': variance,
            'std_dev': std_dev,
            'mean_count': mean_count,
            'count_range': [int(np.min(cluster_counts)), int(np.max(cluster_counts))]
        }

    def _assess_noise_handling_effectiveness(self, metrics_evolution: Dict[str, Any]) -> Dict[str, Any]:
        """Assess noise handling effectiveness across steps."""
        noise_percentages = []
        
        for step_metrics in metrics_evolution.values():
            if 'basic_metrics' in step_metrics:
                basic_metrics = step_metrics['basic_metrics']
                n_noise = basic_metrics.get('n_noise_points', 0)
                n_valid = basic_metrics.get('n_valid_points', 0)
                total = n_noise + n_valid
                
                if total > 0:
                    noise_percentages.append((n_noise / total) * 100)
        
        if len(noise_percentages) < 2:
            return {'effectiveness': 'insufficient_data', 'trend': 'unknown'}
        
        # Calculate trend
        first_noise = noise_percentages[0]
        last_noise = noise_percentages[-1]
        change = last_noise - first_noise
        
        if change > 10:
            trend = 'increasing_noise'
        elif change < -10:
            trend = 'decreasing_noise'
        else:
            trend = 'stable_noise'
        
        # Assess effectiveness
        mean_noise = float(np.mean(noise_percentages))
        if 5 <= mean_noise <= 30:
            effectiveness = 'effective'
        elif mean_noise < 5:
            effectiveness = 'over_aggressive'
        elif mean_noise > 50:
            effectiveness = 'under_aggressive'
        else:
            effectiveness = 'moderate'
        
        return {
            'effectiveness': effectiveness,
            'trend': trend,
            'mean_noise_percentage': mean_noise,
            'noise_change': float(change),
            'noise_range': [float(np.min(noise_percentages)), float(np.max(noise_percentages))]
        }

    def _assess_overall_quality_trend(self, metrics_evolution: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall quality trend across steps."""
        quality_scores = []
        
        for step_metrics in metrics_evolution.values():
            if 'basic_metrics' in step_metrics:
                basic_metrics = step_metrics['basic_metrics']
                
                # Calculate composite quality score
                silhouette = basic_metrics.get('silhouette', 0.0)
                cluster_cv = basic_metrics.get('average_cluster_cv', 1.0)
                
                # Normalize scores (higher is better for both)
                silhouette_score = min(silhouette / self.silhouette_thresholds['excellent'], 1.0)
                cv_score = max(0, 1.0 - (cluster_cv / self.cv_thresholds['poor']))
                
                composite_score = (silhouette_score + cv_score) / 2.0
                quality_scores.append(composite_score)
        
        if len(quality_scores) < 2:
            return {'trend': 'insufficient_data', 'quality_change': 0.0}
        
        # Calculate trend
        first_quality = quality_scores[0]
        last_quality = quality_scores[-1]
        quality_change = last_quality - first_quality
        
        if quality_change > 0.1:
            trend = 'improving'
        elif quality_change < -0.1:
            trend = 'declining'
        else:
            trend = 'stable'
        
        return {
            'trend': trend,
            'quality_change': float(quality_change),
            'first_quality': float(first_quality),
            'last_quality': float(last_quality),
            'mean_quality': float(np.mean(quality_scores)),
            'quality_variance': float(np.var(quality_scores))
        }

    def _generate_quality_recommendations(
        self,
        standard_metrics_evolution: Dict[str, Any],
        enhanced_metrics: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Generate quality improvement recommendations."""
        recommendations = []
        
        # Analyze silhouette trends
        silhouette_improvement = self._calculate_silhouette_improvement(standard_metrics_evolution)
        if silhouette_improvement['trend'] == 'declining':
            recommendations.append("Silhouette scores are declining - consider feature engineering or parameter tuning")
        
        # Analyze cluster CV trends
        cluster_cv_improvement = self._calculate_cluster_cv_improvement(standard_metrics_evolution)
        if cluster_cv_improvement['trend'] == 'declining':
            recommendations.append("Cluster size variation is increasing - strengthen constraint enforcement")
        
        # Analyze cluster count stability
        cluster_stability = self._assess_cluster_count_stability(standard_metrics_evolution)
        if cluster_stability['stability'] == 'unstable':
            recommendations.append("Cluster counts are unstable - consider parameter stabilization")
        
        # Analyze noise handling
        noise_effectiveness = self._assess_noise_handling_effectiveness(standard_metrics_evolution)
        if noise_effectiveness['effectiveness'] == 'over_aggressive':
            recommendations.append("Noise detection may be too aggressive - consider relaxing noise parameters")
        elif noise_effectiveness['effectiveness'] == 'under_aggressive':
            recommendations.append("Noise detection may be too lenient - consider tightening noise parameters")
        
        # Enhanced clustering recommendations
        if enhanced_metrics:
            recommendations.append("Enhanced clustering is available - consider using it for better performance")
        else:
            recommendations.append("Consider implementing enhanced clustering for improved results")
        
        return recommendations

    def _compare_enhanced_vs_standard(
        self,
        standard_metrics_evolution: Dict[str, Any],
        enhanced_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare enhanced clustering results with standard clustering."""
        
        # Get final standard metrics
        final_standard_metrics = None
        for step_metrics in standard_metrics_evolution.values():
            if 'basic_metrics' in step_metrics:
                final_standard_metrics = step_metrics['basic_metrics']
        
        if not final_standard_metrics:
            return {'comparison': 'insufficient_data'}
        
        # Compare key metrics
        comparison = {
            'silhouette_comparison': {
                'standard': final_standard_metrics.get('silhouette', 0.0),
                'enhanced': enhanced_metrics.get('silhouette', 0.0),
                'improvement': enhanced_metrics.get('silhouette', 0.0) - final_standard_metrics.get('silhouette', 0.0)
            },
            'cluster_cv_comparison': {
                'standard': final_standard_metrics.get('average_cluster_cv', 1.0),
                'enhanced': enhanced_metrics.get('average_cluster_cv', 1.0),
                'improvement': final_standard_metrics.get('average_cluster_cv', 1.0) - enhanced_metrics.get('average_cluster_cv', 1.0)
            },
            'cluster_count_comparison': {
                'standard': final_standard_metrics.get('n_clusters', 0),
                'enhanced': enhanced_metrics.get('n_clusters', 0),
                'difference': enhanced_metrics.get('n_clusters', 0) - final_standard_metrics.get('n_clusters', 0)
            },
            'overall_improvement': 'unknown'
        }
        
        # Calculate overall improvement
        silhouette_improvement = comparison['silhouette_comparison']['improvement']
        cv_improvement = comparison['cluster_cv_comparison']['improvement']
        
        if silhouette_improvement > 0.05 and cv_improvement > 0.1:
            comparison['overall_improvement'] = 'significant'
        elif silhouette_improvement > 0.02 or cv_improvement > 0.05:
            comparison['overall_improvement'] = 'moderate'
        elif silhouette_improvement > 0 or cv_improvement > 0:
            comparison['overall_improvement'] = 'minor'
        else:
            comparison['overall_improvement'] = 'none'
        
        return comparison

    def export_report_to_json(self, report: MetricsEvolutionReport, filepath: str) -> None:
        """Export report to JSON file."""
        try:
            report_dict = {
                'silhouette_evolution': report.silhouette_evolution,
                'cluster_cv_evolution': report.cluster_cv_evolution,
                'cluster_count_evolution': report.cluster_count_evolution,
                'noise_percentage_evolution': report.noise_percentage_evolution,
                'enhanced_clustering_metrics': report.enhanced_clustering_metrics,
                'performance_summary': report.performance_summary,
                'step_analysis': report.step_analysis,
                'quality_improvements': report.quality_improvements,
                'report_timestamp': report.report_timestamp,
                'pipeline_config': report.pipeline_config
            }
            
            with open(filepath, 'w') as f:
                json.dump(report_dict, f, indent=2, default=str)
            
            self.logger.info(f"📄 Metrics evolution report exported to {filepath}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to export report to JSON: {e}")

    def export_report_to_csv(self, report: MetricsEvolutionReport, base_filepath: str) -> None:
        """Export report to CSV files."""
        try:
            # Export silhouette evolution
            if report.silhouette_evolution:
                silhouette_df = pd.DataFrame(report.silhouette_evolution)
                silhouette_df.to_csv(f"{base_filepath}_silhouette_evolution.csv", index=False)
            
            # Export cluster CV evolution
            if report.cluster_cv_evolution:
                cv_df = pd.DataFrame(report.cluster_cv_evolution)
                cv_df.to_csv(f"{base_filepath}_cluster_cv_evolution.csv", index=False)
            
            # Export cluster count evolution
            if report.cluster_count_evolution:
                count_df = pd.DataFrame(report.cluster_count_evolution)
                count_df.to_csv(f"{base_filepath}_cluster_count_evolution.csv", index=False)
            
            # Export noise percentage evolution
            if report.noise_percentage_evolution:
                noise_df = pd.DataFrame(report.noise_percentage_evolution)
                noise_df.to_csv(f"{base_filepath}_noise_percentage_evolution.csv", index=False)
            
            # Export step analysis
            if report.step_analysis:
                step_df = pd.DataFrame(report.step_analysis)
                step_df.to_csv(f"{base_filepath}_step_analysis.csv", index=False)
            
            self.logger.info(f"📊 Metrics evolution report exported to CSV files with base path: {base_filepath}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to export report to CSV: {e}")