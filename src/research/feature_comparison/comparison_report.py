"""
Comparison Report Generator

This module generates comprehensive reports comparing different feature versions
using various relevance metrics and visualization tools.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class ComparisonReport:
    """
    Generates comprehensive comparison reports for different feature versions.
    """
    
    def __init__(self, output_dir: str = "/workspace/src/research/feature_comparison/reports"):
        """
        Initialize the comparison report generator.
        
        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def generate_comprehensive_report(self, analysis_results: Dict[str, Any], 
                                   feature_versions: 'FeatureVersions',
                                   save_plots: bool = True) -> Dict[str, Any]:
        """
        Generate a comprehensive comparison report.
        
        Args:
            analysis_results: Results from relevance analysis for each version
            feature_versions: FeatureVersions object
            save_plots: Whether to save plots to files
            
        Returns:
            Dictionary with report data
        """
        logger.info("Generating comprehensive comparison report...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {},
            'detailed_results': {},
            'plots': {}
        }
        
        # Generate summary statistics
        report['summary'] = self._generate_summary(analysis_results, feature_versions)
        
        # Generate detailed results for each version
        for version_name, results in analysis_results.items():
            report['detailed_results'][version_name] = self._generate_version_details(
                version_name, results, feature_versions
            )
        
        # Generate comparison plots
        if save_plots:
            report['plots'] = self._generate_comparison_plots(
                analysis_results, feature_versions
            )
        
        # Save report to file
        self._save_report(report)
        
        logger.info("Comprehensive report generated successfully.")
        return report
    
    def _generate_summary(self, analysis_results: Dict[str, Any], 
                         feature_versions: 'FeatureVersions') -> Dict[str, Any]:
        """Generate summary statistics."""
        summary = {
            'total_versions': len(analysis_results),
            'version_names': list(analysis_results.keys()),
            'feature_counts': {},
            'performance_metrics': {},
            'top_features': {},
            'robust_evaluation': {}
        }
        
        # Feature counts
        version_info = feature_versions.get_version_info()
        for version_name in analysis_results.keys():
            summary['feature_counts'][version_name] = version_info[version_name]['n_features']
        
        # Performance metrics
        for version_name, results in analysis_results.items():
            if 'lgbm_shap' in results and 'performance' in results['lgbm_shap']:
                summary['performance_metrics'][version_name] = results['lgbm_shap']['performance']
        
        # Top features
        for version_name, results in analysis_results.items():
            if 'combined_ranking' in results:
                top_features = results['combined_ranking'].head(10)['feature'].tolist()
                summary['top_features'][version_name] = top_features
        
        # Robust evaluation metrics
        for version_name, results in analysis_results.items():
            robust_metrics = {}
            
            # Rank correlations
            if 'rank_correlations' in results:
                rank_corr = results['rank_correlations']
                robust_metrics['mean_rank_correlation'] = rank_corr.get('mean_correlation', 0)
                robust_metrics['significant_correlations'] = sum(1 for v in rank_corr.values() 
                                                               if isinstance(v, dict) and v.get('is_significant', False))
            
            # Bootstrap analysis
            if 'bootstrap_analysis' in results:
                bootstrap = results['bootstrap_analysis']
                if 'method_results' in bootstrap:
                    for method, method_results in bootstrap['method_results'].items():
                        robust_metrics[f'{method}_mean_cv'] = method_results.get('cv_importance', pd.Series()).mean()
                        robust_metrics[f'{method}_performance_std'] = method_results.get('std_performance', 0)
            
            # Temporal stability
            if 'temporal_stability' in results:
                temporal = results['temporal_stability']
                if 'stability_metrics' in temporal:
                    robust_metrics['mean_temporal_stability'] = temporal['stability_metrics'].get('mean_stability', 0)
                    robust_metrics['stable_features_count'] = len(temporal['stability_metrics'].get('stable_features', []))
            
            # Scaling validation
            if 'scaling_validation' in results:
                scaling = results['scaling_validation']
                robust_metrics['scaling_method'] = scaling.get('method', 'unknown')
            
            summary['robust_evaluation'][version_name] = robust_metrics
        
        return summary
    
    def _generate_version_details(self, version_name: str, results: Dict[str, Any], 
                                feature_versions: 'FeatureVersions') -> Dict[str, Any]:
        """Generate detailed results for a specific version."""
        details = {
            'version_name': version_name,
            'feature_count': feature_versions.get_version_info()[version_name]['n_features'],
            'analysis_methods': list(results.keys()),
            'method_results': {}
        }
        
        # LGBM-SHAP results
        if 'lgbm_shap' in results and results['lgbm_shap']:
            lgbm_data = results['lgbm_shap']
            details['method_results']['lgbm_shap'] = {
                'performance': lgbm_data.get('performance', {}),
                'top_features': lgbm_data.get('feature_importance', pd.DataFrame()).head(10).to_dict('records') if 'feature_importance' in lgbm_data else [],
                'shap_top_features': lgbm_data.get('shap_importance', pd.DataFrame()).head(10).to_dict('records') if 'shap_importance' in lgbm_data else []
            }
        
        # LASSO results
        if 'lasso' in results and results['lasso']:
            lasso_data = results['lasso']
            details['method_results']['lasso'] = {
                'performance': lasso_data.get('performance', {}),
                'alpha': lasso_data.get('alpha', None),
                'selected_features': lasso_data.get('selected_features', []),
                'top_features': lasso_data.get('feature_coefficients', pd.DataFrame()).head(10).to_dict('records') if 'feature_coefficients' in lasso_data else []
            }
        
        # Mutual Information results
        if 'mutual_info' in results and results['mutual_info']:
            mi_data = results['mutual_info']
            details['method_results']['mutual_info'] = {
                'mean_mi': mi_data.get('mean_mi', 0),
                'std_mi': mi_data.get('std_mi', 0),
                'top_features': mi_data.get('mutual_info_scores', pd.DataFrame()).head(10).to_dict('records') if 'mutual_info_scores' in mi_data else []
            }
        
        # Correlation results
        if 'correlation' in results and results['correlation']:
            corr_data = results['correlation']
            details['method_results']['correlation'] = {
                'mean_correlation': corr_data.get('mean_correlation', 0),
                'max_correlation': corr_data.get('max_correlation', 0),
                'top_features': corr_data.get('correlations', pd.DataFrame()).head(10).to_dict('records') if 'correlations' in corr_data else []
            }
        
        # Combined ranking
        if 'combined_ranking' in results and not results['combined_ranking'].empty:
            details['combined_ranking'] = results['combined_ranking'].head(20).to_dict('records')
        
        return details
    
    def _generate_comparison_plots(self, analysis_results: Dict[str, Any], 
                                 feature_versions: 'FeatureVersions') -> Dict[str, str]:
        """Generate comparison plots."""
        plots = {}
        
        try:
            # Plot 1: Feature count comparison
            plots['feature_count_comparison'] = self._plot_feature_counts(feature_versions)
            
            # Plot 2: Performance comparison
            plots['performance_comparison'] = self._plot_performance_comparison(analysis_results)
            
            # Plot 3: Top features comparison
            plots['top_features_comparison'] = self._plot_top_features_comparison(analysis_results)
            
            # Plot 4: Method agreement heatmap
            plots['method_agreement'] = self._plot_method_agreement(analysis_results)
            
            # Plot 5: Robust evaluation metrics
            plots['robust_evaluation'] = self._plot_robust_evaluation(analysis_results)
            
            # Plot 6: Bootstrap stability
            plots['bootstrap_stability'] = self._plot_bootstrap_stability(analysis_results)
            
            # Plot 7: Temporal stability
            plots['temporal_stability'] = self._plot_temporal_stability(analysis_results)
            
        except Exception as e:
            logger.error(f"Error generating plots: {e}")
            plots['error'] = str(e)
        
        return plots
    
    def _plot_feature_counts(self, feature_versions: 'FeatureVersions') -> str:
        """Plot feature count comparison."""
        comparison_df = feature_versions.compare_feature_counts()
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=comparison_df, x='version', y='n_features')
        plt.title('Feature Count Comparison Across Versions')
        plt.xlabel('Feature Version')
        plt.ylabel('Number of Features')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plot_path = self.output_dir / 'feature_count_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    def _plot_performance_comparison(self, analysis_results: Dict[str, Any]) -> str:
        """Plot performance comparison across versions."""
        performance_data = []
        
        for version_name, results in analysis_results.items():
            if 'lgbm_shap' in results and 'performance' in results['lgbm_shap']:
                perf = results['lgbm_shap']['performance']
                for metric, value in perf.items():
                    performance_data.append({
                        'version': version_name,
                        'metric': metric,
                        'value': value
                    })
        
        if not performance_data:
            return "No performance data available"
        
        perf_df = pd.DataFrame(performance_data)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(data=perf_df, x='version', y='value', hue='metric')
        plt.title('Performance Comparison Across Versions')
        plt.xlabel('Feature Version')
        plt.ylabel('Performance Value')
        plt.xticks(rotation=45)
        plt.legend(title='Metric')
        plt.tight_layout()
        
        plot_path = self.output_dir / 'performance_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    def _plot_top_features_comparison(self, analysis_results: Dict[str, Any]) -> str:
        """Plot top features comparison."""
        # Get top 10 features from each version
        top_features_data = []
        
        for version_name, results in analysis_results.items():
            if 'combined_ranking' in results and not results['combined_ranking'].empty:
                top_10 = results['combined_ranking'].head(10)
                for idx, row in top_10.iterrows():
                    top_features_data.append({
                        'version': version_name,
                        'feature': row['feature'],
                        'avg_rank': row['avg_rank'],
                        'rank': idx + 1
                    })
        
        if not top_features_data:
            return "No ranking data available"
        
        top_df = pd.DataFrame(top_features_data)
        
        # Create a pivot table for heatmap
        pivot_df = top_df.pivot(index='feature', columns='version', values='avg_rank')
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_df, annot=True, fmt='.2f', cmap='YlOrRd')
        plt.title('Top Features Ranking Comparison (Lower is Better)')
        plt.xlabel('Feature Version')
        plt.ylabel('Feature Name')
        plt.tight_layout()
        
        plot_path = self.output_dir / 'top_features_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    def _plot_method_agreement(self, analysis_results: Dict[str, Any]) -> str:
        """Plot method agreement heatmap."""
        # This would require more complex analysis of method agreement
        # For now, return a placeholder
        return "Method agreement plot not implemented yet"
    
    def _plot_robust_evaluation(self, analysis_results: Dict[str, Any]) -> str:
        """Plot robust evaluation metrics comparison."""
        try:
            # Extract robust evaluation metrics
            robust_data = []
            for version_name, results in analysis_results.items():
                if 'robust_evaluation' in results:
                    robust_metrics = results['robust_evaluation']
                    for metric, value in robust_metrics.items():
                        if isinstance(value, (int, float)):
                            robust_data.append({
                                'version': version_name,
                                'metric': metric,
                                'value': value
                            })
            
            if not robust_data:
                return "No robust evaluation data available"
            
            robust_df = pd.DataFrame(robust_data)
            
            # Create subplots for different metric categories
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Robust Evaluation Metrics Comparison', fontsize=16)
            
            # Rank correlation metrics
            rank_corr_data = robust_df[robust_df['metric'].str.contains('rank_correlation', na=False)]
            if not rank_corr_data.empty:
                sns.barplot(data=rank_corr_data, x='version', y='value', hue='metric', ax=axes[0,0])
                axes[0,0].set_title('Rank Correlation Metrics')
                axes[0,0].tick_params(axis='x', rotation=45)
            
            # Bootstrap stability metrics
            bootstrap_data = robust_df[robust_df['metric'].str.contains('cv|std', na=False)]
            if not bootstrap_data.empty:
                sns.barplot(data=bootstrap_data, x='version', y='value', hue='metric', ax=axes[0,1])
                axes[0,1].set_title('Bootstrap Stability Metrics')
                axes[0,1].tick_params(axis='x', rotation=45)
            
            # Temporal stability metrics
            temporal_data = robust_df[robust_df['metric'].str.contains('temporal|stability', na=False)]
            if not temporal_data.empty:
                sns.barplot(data=temporal_data, x='version', y='value', hue='metric', ax=axes[1,0])
                axes[1,0].set_title('Temporal Stability Metrics')
                axes[1,0].tick_params(axis='x', rotation=45)
            
            # Scaling method info
            scaling_data = robust_df[robust_df['metric'] == 'scaling_method']
            if not scaling_data.empty:
                axes[1,1].text(0.5, 0.5, f"Scaling Methods:\n{scaling_data.to_string()}", 
                              ha='center', va='center', transform=axes[1,1].transAxes)
                axes[1,1].set_title('Scaling Methods')
                axes[1,1].axis('off')
            
            plt.tight_layout()
            
            plot_path = self.output_dir / 'robust_evaluation_metrics.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(plot_path)
            
        except Exception as e:
            logger.error(f"Error plotting robust evaluation: {e}")
            return f"Error: {e}"
    
    def _plot_bootstrap_stability(self, analysis_results: Dict[str, Any]) -> str:
        """Plot bootstrap stability analysis."""
        try:
            # Extract bootstrap results
            bootstrap_data = []
            for version_name, results in analysis_results.items():
                if 'bootstrap_analysis' in results and 'method_results' in results['bootstrap_analysis']:
                    for method, method_results in results['bootstrap_analysis']['method_results'].items():
                        if 'cv_importance' in method_results:
                            cv_series = method_results['cv_importance']
                            for feature, cv_value in cv_series.items():
                                bootstrap_data.append({
                                    'version': version_name,
                                    'method': method,
                                    'feature': feature,
                                    'cv_importance': cv_value
                                })
            
            if not bootstrap_data:
                return "No bootstrap data available"
            
            bootstrap_df = pd.DataFrame(bootstrap_data)
            
            # Plot coefficient of variation for feature importance
            plt.figure(figsize=(15, 8))
            
            # Group by version and method
            for i, (version, group) in enumerate(bootstrap_df.groupby('version')):
                plt.subplot(2, 2, i+1)
                
                for method, method_group in group.groupby('method'):
                    # Get top 10 features by mean CV
                    top_features = method_group.nlargest(10, 'cv_importance')
                    sns.barplot(data=top_features, x='cv_importance', y='feature', hue='method')
                
                plt.title(f'{version.replace("_", " ").title()} - Feature Importance Stability')
                plt.xlabel('Coefficient of Variation')
                plt.ylabel('Feature')
            
            plt.tight_layout()
            
            plot_path = self.output_dir / 'bootstrap_stability.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(plot_path)
            
        except Exception as e:
            logger.error(f"Error plotting bootstrap stability: {e}")
            return f"Error: {e}"
    
    def _plot_temporal_stability(self, analysis_results: Dict[str, Any]) -> str:
        """Plot temporal stability analysis."""
        try:
            # Extract temporal stability results
            stability_data = []
            for version_name, results in analysis_results.items():
                if 'temporal_stability' in results and 'stability_metrics' in results['temporal_stability']:
                    feature_stability = results['temporal_stability']['stability_metrics'].get('feature_stability', {})
                    for feature, stability_info in feature_stability.items():
                        stability_data.append({
                            'version': version_name,
                            'feature': feature,
                            'stability_score': stability_info.get('stability_score', 0),
                            'mean_rank': stability_info.get('mean_rank', 0),
                            'std_rank': stability_info.get('std_rank', 0)
                        })
            
            if not stability_data:
                return "No temporal stability data available"
            
            stability_df = pd.DataFrame(stability_data)
            
            # Create temporal stability plots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Temporal Stability Analysis', fontsize=16)
            
            # Stability scores by version
            sns.boxplot(data=stability_df, x='version', y='stability_score', ax=axes[0,0])
            axes[0,0].set_title('Feature Stability Scores by Version')
            axes[0,0].tick_params(axis='x', rotation=45)
            
            # Top stable features
            top_stable = stability_df.nlargest(20, 'stability_score')
            sns.barplot(data=top_stable, x='stability_score', y='feature', hue='version', ax=axes[0,1])
            axes[0,1].set_title('Top 20 Most Stable Features')
            
            # Rank variance vs mean rank
            sns.scatterplot(data=stability_df, x='mean_rank', y='std_rank', hue='version', ax=axes[1,0])
            axes[1,0].set_title('Rank Variance vs Mean Rank')
            axes[1,0].set_xlabel('Mean Rank')
            axes[1,0].set_ylabel('Rank Standard Deviation')
            
            # Stability distribution
            sns.histplot(data=stability_df, x='stability_score', hue='version', kde=True, ax=axes[1,1])
            axes[1,1].set_title('Stability Score Distribution')
            axes[1,1].set_xlabel('Stability Score')
            
            plt.tight_layout()
            
            plot_path = self.output_dir / 'temporal_stability.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(plot_path)
            
        except Exception as e:
            logger.error(f"Error plotting temporal stability: {e}")
            return f"Error: {e}"
    
    def _save_report(self, report: Dict[str, Any]) -> None:
        """Save report to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f'feature_comparison_report_{timestamp}.json'
        
        # Convert DataFrames to dict for JSON serialization
        def convert_dataframes(obj):
            if isinstance(obj, pd.DataFrame):
                return obj.to_dict('records')
            elif isinstance(obj, dict):
                return {k: convert_dataframes(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_dataframes(item) for item in obj]
            else:
                return obj
        
        report_serializable = convert_dataframes(report)
        
        with open(report_path, 'w') as f:
            json.dump(report_serializable, f, indent=2, default=str)
        
        logger.info(f"Report saved to: {report_path}")
    
    def generate_markdown_report(self, analysis_results: Dict[str, Any], 
                               feature_versions: 'FeatureVersions') -> str:
        """
        Generate a markdown report.
        
        Args:
            analysis_results: Results from relevance analysis
            feature_versions: FeatureVersions object
            
        Returns:
            Markdown report string
        """
        report_lines = []
        
        # Header
        report_lines.append("# Feature Engineering Comparison Report")
        report_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Summary
        report_lines.append("## Summary")
        version_info = feature_versions.get_version_info()
        for version_name, info in version_info.items():
            report_lines.append(f"### {version_name.replace('_', ' ').title()}")
            report_lines.append(f"- Features: {info['n_features']}")
            report_lines.append(f"- Samples: {info['n_samples']}")
            report_lines.append(f"- Has NaN: {info['has_nan']}")
            report_lines.append("")
        
        # Performance comparison
        report_lines.append("## Performance Comparison")
        for version_name, results in analysis_results.items():
            report_lines.append(f"### {version_name.replace('_', ' ').title()}")
            
            if 'lgbm_shap' in results and 'performance' in results['lgbm_shap']:
                perf = results['lgbm_shap']['performance']
                report_lines.append("**LGBM Performance:**")
                for metric, value in perf.items():
                    report_lines.append(f"- {metric}: {value:.4f}")
                report_lines.append("")
            
            if 'lasso' in results and 'performance' in results['lasso']:
                perf = results['lasso']['performance']
                report_lines.append("**LASSO Performance:**")
                for metric, value in perf.items():
                    report_lines.append(f"- {metric}: {value:.4f}")
                report_lines.append("")
        
        # Top features
        report_lines.append("## Top Features by Version")
        for version_name, results in analysis_results.items():
            report_lines.append(f"### {version_name.replace('_', ' ').title()}")
            
            if 'combined_ranking' in results and not results['combined_ranking'].empty:
                top_10 = results['combined_ranking'].head(10)
                report_lines.append("| Rank | Feature | Average Rank |")
                report_lines.append("|------|---------|--------------|")
                for idx, row in top_10.iterrows():
                    report_lines.append(f"| {idx + 1} | {row['feature']} | {row['avg_rank']:.2f} |")
                report_lines.append("")
        
        return "\n".join(report_lines)