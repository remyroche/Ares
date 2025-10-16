"""
Comprehensive Metrics Evolution Report Generator

This module generates detailed reports showing the evolution of clustering metrics
across all stages of the clustering pipeline, helping identify what works and what doesn't.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
import json
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class MetricsEvolutionReporter:
    """Generator for comprehensive metrics evolution reports."""

    def __init__(self):
        """Initialize the metrics evolution reporter."""
        self.logger = logging.getLogger(__name__)

    def generate_comprehensive_report(self, comprehensive_metrics: Dict[str, Any],
                                    output_dir: str = "clustering_reports") -> Dict[str, Any]:
        """Generate comprehensive metrics evolution report.

        Args:
            comprehensive_metrics: Comprehensive metrics from clustering pipeline
            output_dir: Output directory for reports

        Returns:
            Dictionary containing all report components
        """
        try:
            self.logger.info("📊 Generating comprehensive metrics evolution report...")

            # Create output directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Extract metrics evolution
            metrics_evolution = comprehensive_metrics.get('metrics_evolution', {})

            # Generate individual report components
            reports = {}

            # 1. Metrics Evolution Summary
            reports['metrics_evolution_summary'] = self._generate_metrics_evolution_summary(metrics_evolution)

            # 2. Step-by-Step Analysis
            reports['step_by_step_analysis'] = self._generate_step_by_step_analysis(metrics_evolution)

            # 3. Performance Analysis
            reports['performance_analysis'] = self._generate_performance_analysis(comprehensive_metrics)

            # 4. Quality Metrics Trends
            reports['quality_metrics_trends'] = self._generate_quality_metrics_trends(metrics_evolution)

            # 5. What Works vs What Doesn't Analysis
            reports['effectiveness_analysis'] = self._generate_effectiveness_analysis(metrics_evolution)

            # 6. Hardware Optimization Impact
            reports['hardware_optimization_impact'] = self._generate_hardware_optimization_impact(comprehensive_metrics)

            # 7. Enhanced vs Standard Comparison
            reports['enhanced_vs_standard_comparison'] = self._generate_enhanced_vs_standard_comparison(comprehensive_metrics)

            # Generate visualizations
            self._generate_visualizations(metrics_evolution, output_path)

            # Save reports
            self._save_reports(reports, output_path)

            # Generate final summary
            final_summary = self._generate_final_summary(reports, comprehensive_metrics)

            self.logger.info("✅ Comprehensive metrics evolution report generated successfully")
            return final_summary

        except Exception as e:
            self.logger.error(f"❌ Failed to generate comprehensive report: {e}")
            return {'error': str(e)}

    def _generate_metrics_evolution_summary(self, metrics_evolution: Dict[str, Any]) -> Dict[str, Any]:
        """Generate metrics evolution summary.

        Args:
            metrics_evolution: Metrics evolution data

        Returns:
            Summary of metrics evolution
        """
        summary = {
            'total_steps': len(metrics_evolution),
            'steps_completed': 0,
            'steps_with_errors': 0,
            'silhouette_progression': [],
            'cluster_cv_progression': [],
            'cluster_count_progression': [],
            'noise_points_progression': []
        }

        for step_name, step_metrics in metrics_evolution.items():
            if 'error' in step_metrics:
                summary['steps_with_errors'] += 1
                continue

            summary['steps_completed'] += 1

            # Extract basic metrics if available
            basic_metrics = step_metrics.get('basic_metrics', {})
            if basic_metrics:
                summary['silhouette_progression'].append({
                    'step': step_name,
                    'silhouette': basic_metrics.get('silhouette', 0.0),
                    'n_clusters': basic_metrics.get('n_clusters', 0)
                })

                summary['cluster_cv_progression'].append({
                    'step': step_name,
                    'average_cluster_cv': basic_metrics.get('average_cluster_cv', 0.0),
                    'n_clusters': basic_metrics.get('n_clusters', 0)
                })

                summary['cluster_count_progression'].append({
                    'step': step_name,
                    'n_clusters': basic_metrics.get('n_clusters', 0),
                    'n_valid_points': basic_metrics.get('n_valid_points', 0)
                })

                summary['noise_points_progression'].append({
                    'step': step_name,
                    'n_noise_points': basic_metrics.get('n_noise_points', 0),
                    'noise_percentage': (basic_metrics.get('n_noise_points', 0) /
                                       (basic_metrics.get('n_valid_points', 0) + basic_metrics.get('n_noise_points', 0)) * 100)
                })

        return summary

    def _generate_step_by_step_analysis(self, metrics_evolution: Dict[str, Any]) -> Dict[str, Any]:
        """Generate step-by-step analysis.

        Args:
            metrics_evolution: Metrics evolution data

        Returns:
            Step-by-step analysis
        """
        analysis = {}

        for step_name, step_metrics in metrics_evolution.items():
            step_analysis = {
                'step_name': step_name,
                'success': 'error' not in step_metrics,
                'error': step_metrics.get('error'),
                'metrics': {},
                'insights': []
            }

            if 'error' not in step_metrics:
                # Analyze basic metrics
                basic_metrics = step_metrics.get('basic_metrics', {})
                if basic_metrics:
                    silhouette = basic_metrics.get('silhouette', 0.0)
                    avg_cv = basic_metrics.get('average_cluster_cv', 0.0)
                    n_clusters = basic_metrics.get('n_clusters', 0)

                    step_analysis['metrics'] = {
                        'silhouette': silhouette,
                        'average_cluster_cv': avg_cv,
                        'n_clusters': n_clusters,
                        'n_valid_points': basic_metrics.get('n_valid_points', 0),
                        'n_noise_points': basic_metrics.get('n_noise_points', 0)
                    }

                    # Generate insights
                    if silhouette > 0.3:
                        step_analysis['insights'].append("✅ Good cluster separation achieved")
                    elif silhouette > 0.1:
                        step_analysis['insights'].append("⚠️ Moderate cluster separation")
                    else:
                        step_analysis['insights'].append("❌ Poor cluster separation")

                    if avg_cv < 0.5:
                        step_analysis['insights'].append("✅ Low cluster variability (good)")
                    elif avg_cv < 1.0:
                        step_analysis['insights'].append("⚠️ Moderate cluster variability")
                    else:
                        step_analysis['insights'].append("❌ High cluster variability")

                    if 15 <= n_clusters <= 25:
                        step_analysis['insights'].append("✅ Optimal cluster count range")
                    else:
                        step_analysis['insights'].append(f"⚠️ Cluster count ({n_clusters}) outside optimal range (15-25)")

            analysis[step_name] = step_analysis

        return analysis

    def _generate_performance_analysis(self, comprehensive_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate performance analysis.

        Args:
            comprehensive_metrics: Comprehensive metrics

        Returns:
            Performance analysis
        """
        performance_metrics = comprehensive_metrics.get('performance_metrics', {})
        hardware_status = comprehensive_metrics.get('hardware_optimization_status', {})

        analysis = {
            'execution_times': {
                'total_pipeline_time': performance_metrics.get('total_pipeline_time', 0.0),
                'standard_clustering_time': performance_metrics.get('standard_clustering_time', 0.0),
                'enhanced_clustering_time': performance_metrics.get('enhanced_clustering_time', 0.0)
            },
            'hardware_optimizations': {
                'matrix_operations_used': hardware_status.get('matrix_operations', False),
                'gpu_acceleration_used': hardware_status.get('gpu_acceleration', False),
                'hardware_optimizations_used': hardware_status.get('hardware_optimizations', False)
            },
            'memory_usage': {
                'peak_memory_percent': performance_metrics.get('memory_usage_percent', 0.0)
            },
            'efficiency_analysis': {
                'enhanced_clustering_overhead': (
                    performance_metrics.get('enhanced_clustering_time', 0.0) /
                    performance_metrics.get('total_pipeline_time', 1.0) * 100
                ),
                'standard_clustering_efficiency': (
                    performance_metrics.get('standard_clustering_time', 0.0) /
                    performance_metrics.get('total_pipeline_time', 1.0) * 100
                )
            }
        }

        return analysis

    def _generate_quality_metrics_trends(self, metrics_evolution: Dict[str, Any]) -> Dict[str, Any]:
        """Generate quality metrics trends analysis.

        Args:
            metrics_evolution: Metrics evolution data

        Returns:
            Quality metrics trends
        """
        trends = {
            'silhouette_trend': [],
            'cluster_cv_trend': [],
            'cluster_count_trend': [],
            'quality_improvements': [],
            'quality_degradations': []
        }

        previous_silhouette = None
        previous_cv = None
        previous_clusters = None

        for step_name, step_metrics in metrics_evolution.items():
            if 'error' in step_metrics:
                continue

            basic_metrics = step_metrics.get('basic_metrics', {})
            if not basic_metrics:
                continue

            silhouette = basic_metrics.get('silhouette', 0.0)
            avg_cv = basic_metrics.get('average_cluster_cv', 0.0)
            n_clusters = basic_metrics.get('n_clusters', 0)

            trends['silhouette_trend'].append({
                'step': step_name,
                'silhouette': silhouette,
                'change': silhouette - previous_silhouette if previous_silhouette is not None else 0.0
            })

            trends['cluster_cv_trend'].append({
                'step': step_name,
                'average_cluster_cv': avg_cv,
                'change': avg_cv - previous_cv if previous_cv is not None else 0.0
            })

            trends['cluster_count_trend'].append({
                'step': step_name,
                'n_clusters': n_clusters,
                'change': n_clusters - previous_clusters if previous_clusters is not None else 0
            })

            # Track improvements and degradations
            if previous_silhouette is not None:
                silhouette_change = silhouette - previous_silhouette
                if silhouette_change > 0.05:
                    trends['quality_improvements'].append({
                        'step': step_name,
                        'metric': 'silhouette',
                        'improvement': silhouette_change,
                        'from': previous_silhouette,
                        'to': silhouette
                    })
                elif silhouette_change < -0.05:
                    trends['quality_degradations'].append({
                        'step': step_name,
                        'metric': 'silhouette',
                        'degradation': abs(silhouette_change),
                        'from': previous_silhouette,
                        'to': silhouette
                    })

            if previous_cv is not None:
                cv_change = avg_cv - previous_cv
                if cv_change < -0.1:  # Lower CV is better
                    trends['quality_improvements'].append({
                        'step': step_name,
                        'metric': 'cluster_cv',
                        'improvement': abs(cv_change),
                        'from': previous_cv,
                        'to': avg_cv
                    })
                elif cv_change > 0.1:
                    trends['quality_degradations'].append({
                        'step': step_name,
                        'metric': 'cluster_cv',
                        'degradation': cv_change,
                        'from': previous_cv,
                        'to': avg_cv
                    })

            previous_silhouette = silhouette
            previous_cv = avg_cv
            previous_clusters = n_clusters

        return trends

    def _generate_effectiveness_analysis(self, metrics_evolution: Dict[str, Any]) -> Dict[str, Any]:
        """Generate what works vs what doesn't analysis.

        Args:
            metrics_evolution: Metrics evolution data

        Returns:
            Effectiveness analysis
        """
        analysis = {
            'most_effective_steps': [],
            'least_effective_steps': [],
            'critical_improvement_steps': [],
            'noise_handling_effectiveness': {},
            'constraint_enforcement_effectiveness': {},
            'recommendations': []
        }

        # Analyze each step's effectiveness
        for step_name, step_metrics in metrics_evolution.items():
            if 'error' in step_metrics:
                analysis['least_effective_steps'].append({
                    'step': step_name,
                    'reason': 'Failed with error',
                    'error': step_metrics.get('error')
                })
                continue

            basic_metrics = step_metrics.get('basic_metrics', {})
            if not basic_metrics:
                continue

            silhouette = basic_metrics.get('silhouette', 0.0)
            avg_cv = basic_metrics.get('average_cluster_cv', 0.0)
            n_clusters = basic_metrics.get('n_clusters', 0)

            # Calculate effectiveness score
            effectiveness_score = (
                silhouette * 0.4 +  # 40% weight on silhouette
                (1.0 / (1.0 + avg_cv)) * 0.3 +  # 30% weight on low CV (inverted)
                (1.0 - abs(n_clusters - 20) / 20.0) * 0.3  # 30% weight on target cluster count
            )

            step_effectiveness = {
                'step': step_name,
                'effectiveness_score': effectiveness_score,
                'silhouette': silhouette,
                'average_cluster_cv': avg_cv,
                'n_clusters': n_clusters
            }

            if effectiveness_score > 0.7:
                analysis['most_effective_steps'].append(step_effectiveness)
            elif effectiveness_score < 0.3:
                analysis['least_effective_steps'].append(step_effectiveness)

            # Analyze specific step types
            if 'noise_reduction' in step_name.lower():
                analysis['noise_handling_effectiveness'][step_name] = {
                    'n_noise_points': basic_metrics.get('n_noise_points', 0),
                    'noise_percentage': (basic_metrics.get('n_noise_points', 0) /
                                       (basic_metrics.get('n_valid_points', 0) + basic_metrics.get('n_noise_points', 0)) * 100),
                    'effectiveness': 'Good' if basic_metrics.get('n_noise_points', 0) < 100 else 'Needs improvement'
                }

            if 'constraint' in step_name.lower():
                analysis['constraint_enforcement_effectiveness'][step_name] = {
                    'n_clusters': n_clusters,
                    'target_met': abs(n_clusters - 20) <= 2,
                    'effectiveness': 'Good' if abs(n_clusters - 20) <= 2 else 'Needs improvement'
                }

        # Generate recommendations
        if analysis['most_effective_steps']:
            best_step = max(analysis['most_effective_steps'], key=lambda x: x['effectiveness_score'])
            analysis['recommendations'].append(f"✅ {best_step['step']} is most effective (score: {best_step['effectiveness_score']:.3f})")

        if analysis['least_effective_steps']:
            worst_step = min(analysis['least_effective_steps'], key=lambda x: x['effectiveness_score'])
            analysis['recommendations'].append(f"⚠️ {worst_step['step']} needs improvement (score: {worst_step['effectiveness_score']:.3f})")

        return analysis

    def _generate_hardware_optimization_impact(self, comprehensive_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate hardware optimization impact analysis.

        Args:
            comprehensive_metrics: Comprehensive metrics

        Returns:
            Hardware optimization impact analysis
        """
        hardware_status = comprehensive_metrics.get('hardware_optimization_status', {})
        performance_metrics = comprehensive_metrics.get('performance_metrics', {})

        analysis = {
            'optimization_status': hardware_status,
            'performance_impact': {
                'matrix_operations_enabled': hardware_status.get('matrix_operations', False),
                'gpu_acceleration_enabled': hardware_status.get('gpu_acceleration', False),
                'total_execution_time': performance_metrics.get('total_pipeline_time', 0.0)
            },
            'efficiency_gains': {},
            'recommendations': []
        }

        # Analyze efficiency gains
        if hardware_status.get('matrix_operations', False):
            analysis['efficiency_gains']['matrix_operations'] = {
                'status': 'Enabled',
                'impact': 'High - Accelerates matrix computations',
                'estimated_speedup': '2-5x for large datasets'
            }
        else:
            analysis['efficiency_gains']['matrix_operations'] = {
                'status': 'Disabled',
                'impact': 'Performance degradation expected',
                'recommendation': 'Enable matrix operations for better performance'
            }
            analysis['recommendations'].append("🔧 Enable matrix operations for better performance")

        if hardware_status.get('gpu_acceleration', False):
            analysis['efficiency_gains']['gpu_acceleration'] = {
                'status': 'Enabled',
                'impact': 'High - GPU-accelerated computations',
                'estimated_speedup': '3-10x for parallel operations'
            }
        else:
            analysis['efficiency_gains']['gpu_acceleration'] = {
                'status': 'Disabled',
                'impact': 'CPU-only computations',
                'recommendation': 'Enable GPU acceleration if available'
            }
            analysis['recommendations'].append("🚀 Enable GPU acceleration for maximum performance")

        return analysis

    def _generate_enhanced_vs_standard_comparison(self, comprehensive_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate enhanced vs standard clustering comparison.

        Args:
            comprehensive_metrics: Comprehensive metrics

        Returns:
            Enhanced vs standard comparison
        """
        standard_metrics = comprehensive_metrics.get('standard_clustering_metrics', {})
        enhanced_metrics = comprehensive_metrics.get('enhanced_clustering_metrics')

        comparison = {
            'enhanced_clustering_available': enhanced_metrics is not None,
            'enhanced_clustering_success': enhanced_metrics.get('success', False) if enhanced_metrics else False,
            'performance_comparison': {},
            'quality_comparison': {},
            'recommendations': []
        }

        if enhanced_metrics and enhanced_metrics.get('success', False):
            # Performance comparison
            standard_time = standard_metrics.get('performance_metrics', {}).get('total_time', 0.0)
            enhanced_time = enhanced_metrics.get('execution_time', 0.0)

            comparison['performance_comparison'] = {
                'standard_clustering_time': standard_time,
                'enhanced_clustering_time': enhanced_time,
                'total_time': standard_time + enhanced_time,
                'enhanced_overhead': enhanced_time / (standard_time + enhanced_time) * 100 if (standard_time + enhanced_time) > 0 else 0
            }

            # Quality comparison
            standard_quality = standard_metrics.get('quality_metrics', {})
            enhanced_quality = enhanced_metrics.get('quality_metrics', {})

            comparison['quality_comparison'] = {
                'silhouette_improvement': enhanced_quality.get('silhouette', 0.0) - standard_quality.get('silhouette', 0.0),
                'davies_bouldin_improvement': standard_quality.get('davies_bouldin', float('inf')) - enhanced_quality.get('davies_bouldin', float('inf')),
                'cluster_count_standard': len(np.unique(standard_metrics.get('labels', []))),
                'cluster_count_enhanced': len(np.unique(enhanced_metrics.get('labels', []))),
                'enhanced_features': {
                    'frontiers_established': len(enhanced_metrics.get('frontiers', {})),
                    'transfers_applied': len(enhanced_metrics.get('transfer_history', []))
                }
            }

            # Generate recommendations
            silhouette_improvement = comparison['quality_comparison']['silhouette_improvement']
            if silhouette_improvement > 0.05:
                comparison['recommendations'].append("✅ Enhanced clustering provides significant quality improvement")
            elif silhouette_improvement > 0.01:
                comparison['recommendations'].append("⚠️ Enhanced clustering provides modest quality improvement")
            else:
                comparison['recommendations'].append("❌ Enhanced clustering provides minimal quality improvement")

            enhanced_overhead = comparison['performance_comparison']['enhanced_overhead']
            if enhanced_overhead < 30:
                comparison['recommendations'].append("✅ Enhanced clustering overhead is acceptable")
            elif enhanced_overhead < 50:
                comparison['recommendations'].append("⚠️ Enhanced clustering overhead is moderate")
            else:
                comparison['recommendations'].append("❌ Enhanced clustering overhead is high")

        else:
            comparison['recommendations'].append("⚠️ Enhanced clustering not available or failed")

        return comparison

    def _generate_visualizations(self, metrics_evolution: Dict[str, Any], output_path: Path):
        """Generate visualization plots.

        Args:
            metrics_evolution: Metrics evolution data
            output_path: Output directory path
        """
        try:
            # Extract data for plotting
            steps = []
            silhouettes = []
            cluster_cvs = []
            cluster_counts = []

            for step_name, step_metrics in metrics_evolution.items():
                if 'error' in step_metrics:
                    continue

                basic_metrics = step_metrics.get('basic_metrics', {})
                if basic_metrics:
                    steps.append(step_name.replace('step_', '').replace('_', ' ').title())
                    silhouettes.append(basic_metrics.get('silhouette', 0.0))
                    cluster_cvs.append(basic_metrics.get('average_cluster_cv', 0.0))
                    cluster_counts.append(basic_metrics.get('n_clusters', 0))

            if not steps:
                return

            # Create figure with subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Clustering Metrics Evolution', fontsize=16)

            # Silhouette score evolution
            ax1.plot(range(len(steps)), silhouettes, 'bo-', linewidth=2, markersize=8)
            ax1.set_title('Silhouette Score Evolution')
            ax1.set_xlabel('Clustering Steps')
            ax1.set_ylabel('Silhouette Score')
            ax1.set_xticks(range(len(steps)))
            ax1.set_xticklabels(steps, rotation=45, ha='right')
            ax1.grid(True, alpha=0.3)
            ax1.axhline(y=0.3, color='green', linestyle='--', alpha=0.7, label='Good (>0.3)')
            ax1.axhline(y=0.1, color='orange', linestyle='--', alpha=0.7, label='Moderate (>0.1)')
            ax1.legend()

            # Cluster CV evolution
            ax2.plot(range(len(steps)), cluster_cvs, 'ro-', linewidth=2, markersize=8)
            ax2.set_title('Average Cluster CV Evolution')
            ax2.set_xlabel('Clustering Steps')
            ax2.set_ylabel('Average Cluster CV')
            ax2.set_xticks(range(len(steps)))
            ax2.set_xticklabels(steps, rotation=45, ha='right')
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=0.5, color='green', linestyle='--', alpha=0.7, label='Good (<0.5)')
            ax2.axhline(y=1.0, color='orange', linestyle='--', alpha=0.7, label='Moderate (<1.0)')
            ax2.legend()

            # Cluster count evolution
            ax3.plot(range(len(steps)), cluster_counts, 'go-', linewidth=2, markersize=8)
            ax3.set_title('Cluster Count Evolution')
            ax3.set_xlabel('Clustering Steps')
            ax3.set_ylabel('Number of Clusters')
            ax3.set_xticks(range(len(steps)))
            ax3.set_xticklabels(steps, rotation=45, ha='right')
            ax3.grid(True, alpha=0.3)
            ax3.axhline(y=20, color='red', linestyle='--', alpha=0.7, label='Target (20)')
            ax3.axhspan(15, 25, alpha=0.2, color='green', label='Optimal Range')
            ax3.legend()

            # Quality improvement heatmap
            improvement_data = []
            for i in range(len(steps)):
                if i > 0:
                    silhouette_improvement = silhouettes[i] - silhouettes[i-1]
                    cv_improvement = cluster_cvs[i-1] - cluster_cvs[i]  # Lower CV is better
                    improvement_data.append([silhouette_improvement, cv_improvement])
                else:
                    improvement_data.append([0, 0])

            improvement_df = pd.DataFrame(improvement_data,
                                        index=steps,
                                        columns=['Silhouette\nImprovement', 'CV\nImprovement'])

            sns.heatmap(improvement_df.T, annot=True, cmap='RdYlGn', center=0,
                       ax=ax4, cbar_kws={'label': 'Improvement'})
            ax4.set_title('Quality Improvements Between Steps')
            ax4.set_xlabel('Clustering Steps')
            ax4.set_ylabel('Metrics')

            plt.tight_layout()
            plt.savefig(output_path / 'metrics_evolution.png', dpi=300, bbox_inches='tight')
            plt.close()

            self.logger.info("✅ Visualization plots generated successfully")

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to generate visualizations: {e}")

    def _save_reports(self, reports: Dict[str, Any], output_path: Path):
        """Save reports to files.

        Args:
            reports: Report data
            output_path: Output directory path
        """
        try:
            # Save JSON report
            with open(output_path / 'comprehensive_metrics_report.json', 'w') as f:
                json.dump(reports, f, indent=2, default=str)

            # Save markdown report
            markdown_content = self._generate_markdown_report(reports)
            with open(output_path / 'comprehensive_metrics_report.md', 'w') as f:
                f.write(markdown_content)

            self.logger.info("✅ Reports saved successfully")

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to save reports: {e}")

    def _generate_markdown_report(self, reports: Dict[str, Any]) -> str:
        """Generate markdown report.

        Args:
            reports: Report data

        Returns:
            Markdown content
        """
        md_content = f"""# Comprehensive Clustering Metrics Evolution Report

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This report analyzes the evolution of clustering metrics across all stages of the clustering pipeline, identifying what works and what doesn't.

## Metrics Evolution Summary

"""

        # Add metrics evolution summary
        evolution_summary = reports.get('metrics_evolution_summary', {})
        md_content += f"""
- **Total Steps**: {evolution_summary.get('total_steps', 0)}
- **Steps Completed**: {evolution_summary.get('steps_completed', 0)}
- **Steps with Errors**: {evolution_summary.get('steps_with_errors', 0)}

### Silhouette Score Progression
"""

        for step_data in evolution_summary.get('silhouette_progression', []):
            md_content += f"- **{step_data['step']}**: {step_data['silhouette']:.3f} ({step_data['n_clusters']} clusters)\n"

        # Add effectiveness analysis
        effectiveness = reports.get('effectiveness_analysis', {})
        md_content += f"""

## What Works vs What Doesn't

### Most Effective Steps
"""

        for step in effectiveness.get('most_effective_steps', []):
            md_content += f"- **{step['step']}**: Effectiveness Score {step['effectiveness_score']:.3f} (Silhouette: {step['silhouette']:.3f})\n"

        md_content += f"""

### Least Effective Steps
"""

        for step in effectiveness.get('least_effective_steps', []):
            md_content += f"- **{step['step']}**: Effectiveness Score {step['effectiveness_score']:.3f} (Silhouette: {step['silhouette']:.3f})\n"

        # Add recommendations
        md_content += f"""

## Recommendations

"""

        for recommendation in effectiveness.get('recommendations', []):
            md_content += f"- {recommendation}\n"

        return md_content

    def _generate_final_summary(self, reports: Dict[str, Any], comprehensive_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final summary.

        Args:
            reports: All reports
            comprehensive_metrics: Comprehensive metrics

        Returns:
            Final summary
        """
        return {
            'timestamp': datetime.now().isoformat(),
            'reports_generated': list(reports.keys()),
            'comprehensive_metrics': comprehensive_metrics,
            'detailed_reports': reports,
            'summary': {
                'total_steps_analyzed': len(comprehensive_metrics.get('metrics_evolution', {})),
                'enhanced_clustering_available': comprehensive_metrics.get('enhanced_clustering_success', False),
                'hardware_optimizations_enabled': comprehensive_metrics.get('hardware_optimization_status', {}),
                'overall_success': comprehensive_metrics.get('standard_clustering_success', False)
            }
        }

def generate_metrics_evolution_report(comprehensive_metrics: Dict[str, Any],
                                    output_dir: str = "clustering_reports") -> Dict[str, Any]:
    """Generate comprehensive metrics evolution report.

    Args:
        comprehensive_metrics: Comprehensive metrics from clustering pipeline
        output_dir: Output directory for reports

    Returns:
        Complete report
    """
    reporter = MetricsEvolutionReporter()
    return reporter.generate_comprehensive_report(comprehensive_metrics, output_dir)
