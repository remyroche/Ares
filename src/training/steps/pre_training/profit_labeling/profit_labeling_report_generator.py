"""
Profit Labeling Report Generator.

This module generates comprehensive reports based on profit labeling outcomes,
including regime-aware analysis and integration with subsequent pipeline steps.
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from abc import ABC

# Import BaseStep
from src.training.steps.base_step import BaseStep

from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success
from src.utils.serialization_utils import UniversalSerializer


@dataclass
class ProfitLabelingReport:
    """Structure for profit labeling reports."""

    # Basic information
    symbol: str
    exchange: str
    timeframe: str
    timestamp: datetime

    # Processing statistics
    processing_time: float
    n_samples: int
    n_targets: int
    n_horizons: int

    # Quality metrics
    quality_scores: Dict[str, Any]
    regime_statistics: Dict[str, Any]

    # Label distribution
    label_distribution: Dict[str, Any]

    # Feature compatibility
    feature_lookback_compatibility: Dict[str, Any]

    # Recommendations
    recommendations: List[str]


class ProfitLabelingReportGenerator(BaseStep):
    """
    Generator for comprehensive profit labeling reports.

    This class creates detailed reports that analyze labeling quality,
    regime-specific performance, and compatibility with downstream steps.
    Inherits from BaseStep for standardized pipeline integration.
    """

    def __init__(self):
        """Initialize the report generator."""
        super().__init__()
        self.serializer = UniversalSerializer()
        tprint_info("📊 Profit Labeling Report Generator initialized")
    
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the profit labeling report generation step.
        
        Args:
            config: Configuration dictionary containing:
                - labeling_result: Results from multi-horizon profit labeling
                - regime_data: Optional regime data for regime-aware analysis
                - feature_lookback_data: Optional feature lookback optimization data
                - output_directory: Optional directory to save reports
                - symbol: Optional symbol for context
                - exchange: Optional exchange for context
                - information: Optional information for context
                - direction: Optional direction for context
                - model: Optional model type for context
        
        Returns:
            Dictionary containing:
                - success: Boolean indicating success
                - report: ProfitLabelingReport object
                - report_path: Path to generated report file
                - artifacts: List of generated artifacts
        """
        try:
            # Set context for enhanced file naming and operations
            self._set_context(
                symbol=config.get('symbol'),
                exchange=config.get('exchange'),
                information=config.get('information'),
                direction=config.get('direction', 'long'),
                model=config.get('model', 'Analyst')
            )
            
            # Extract parameters from config
            labeling_result = config.get('labeling_result')
            regime_data = config.get('regime_data')
            feature_lookback_data = config.get('feature_lookback_data')
            output_directory = config.get('output_directory', 'profit_labeling_reports')
            
            if labeling_result is None:
                return {
                    'success': False,
                    'error': 'Missing required parameter: labeling_result'
                }
            
            # Generate report
            report = self.generate_report(
                labeling_result=labeling_result,
                regime_data=regime_data,
                feature_lookback_data=feature_lookback_data,
                output_directory=output_directory
            )
            
            # Save artifacts
            artifacts = []
            
            # Save report as metadata
            report_path = self._save_metadata(
                {
                    'symbol': report.symbol,
                    'exchange': report.exchange,
                    'timeframe': report.timeframe,
                    'timestamp': report.timestamp.isoformat(),
                    'processing_time': report.processing_time,
                    'n_samples': report.n_samples,
                    'n_targets': report.n_targets,
                    'n_horizons': report.n_horizons,
                    'quality_scores': report.quality_scores,
                    'regime_statistics': report.regime_statistics,
                    'label_distribution': report.label_distribution,
                    'feature_lookback_compatibility': report.feature_lookback_compatibility,
                    'recommendations': report.recommendations
                },
                'profit_labeling_report'
            )
            if report_path:
                artifacts.append(report_path)
            
            # Generate outcome file
            outcome_content = self._generate_outcome_content(report, artifacts)
            self._save_outcome_file(outcome_content, 'profit_labeling_report_outcome')
            
            return {
                'success': True,
                'report': report,
                'report_path': report_path,
                'artifacts': artifacts
            }
            
        except Exception as e:
            error_msg = f"Profit labeling report generation failed: {str(e)}"
            if TPRINT_AVAILABLE:
                tprint_error(f"❌ {error_msg}")
            return {
                'success': False,
                'error': error_msg
            }
    
    def _generate_outcome_content(self, report: ProfitLabelingReport, artifacts: List[str]) -> str:
        """Generate outcome file content."""
        content = f"""# Profit Labeling Report Generation Outcome

## Summary
- **Status**: Success
- **Symbol**: {report.symbol}
- **Exchange**: {report.exchange}
- **Timeframe**: {report.timeframe}
- **Processing Time**: {report.processing_time:.2f} seconds
- **Samples**: {report.n_samples}
- **Targets**: {report.n_targets}
- **Horizons**: {report.n_horizons}
- **Artifacts Generated**: {len(artifacts)}

## Quality Scores
"""
        
        for metric, score in report.quality_scores.items():
            content += f"- **{metric}**: {score:.3f}\n"
        
        content += f"""
## Regime Statistics
"""
        
        for regime, stats in report.regime_statistics.items():
            content += f"- **{regime}**: {stats}\n"
        
        content += f"""
## Label Distribution
"""
        
        for target, distribution in report.label_distribution.items():
            content += f"- **{target}**: {distribution}\n"
        
        content += f"""
## Recommendations
{chr(10).join(f"- {rec}" for rec in report.recommendations)}

## Generated Artifacts
{chr(10).join(f"- {artifact}" for artifact in artifacts)}
"""
        
        return content

    def generate_report(
        self,
        labeling_result: Dict[str, Any],
        regime_data: Optional[Dict[str, Any]] = None,
        feature_lookback_data: Optional[Dict[str, Any]] = None,
        output_directory: str = "profit_labeling_reports"
    ) -> ProfitLabelingReport:
        """
        Generate comprehensive profit labeling report.

        Args:
            labeling_result: Results from multi-horizon profit labeling
            regime_data: Optional regime data for regime-aware analysis
            feature_lookback_data: Optional feature lookback optimization data
            output_directory: Directory to save reports

        Returns:
            ProfitLabelingReport with comprehensive analysis
        """
        try:
            tprint_info("📋 Generating comprehensive profit labeling report")

            # Extract basic information
            metadata = labeling_result.get('multi_horizon_labeling_result', {}).get('metadata', {})
            quality_scores = labeling_result.get('multi_horizon_labeling_result', {}).get('quality_scores', {})

            # Create report structure
            report = ProfitLabelingReport(
                symbol=metadata.get('symbol', 'Unknown'),
                exchange=metadata.get('exchange', 'Unknown'),
                timeframe=metadata.get('timeframe', 'Unknown'),
                timestamp=datetime.now(),
                processing_time=metadata.get('processing_time', 0.0),
                n_samples=metadata.get('n_samples', 0),
                n_targets=metadata.get('n_targets', 0),
                n_horizons=metadata.get('n_horizons', 0),
                quality_scores=self._analyze_quality_scores(quality_scores),
                regime_statistics=self._analyze_regime_statistics(regime_data) if regime_data else {},
                label_distribution=self._analyze_label_distribution(labeling_result),
                feature_lookback_compatibility=self._analyze_feature_compatibility(feature_lookback_data) if feature_lookback_data else {},
                recommendations=self._generate_recommendations(labeling_result, regime_data, feature_lookback_data)
            )

            # Save report to files
            self._save_report(report, output_directory)

            tprint_success("✅ Comprehensive profit labeling report generated")
            return report

        except Exception as e:
            tprint_error(f"❌ Error generating report: {e}")
            # Return basic error report
            return ProfitLabelingReport(
                symbol='Error',
                exchange='Error',
                timeframe='Error',
                timestamp=datetime.now(),
                processing_time=0.0,
                n_samples=0,
                n_targets=0,
                n_horizons=0,
                quality_scores={},
                regime_statistics={},
                label_distribution={},
                feature_lookback_compatibility={},
                recommendations=[f"Report generation failed: {str(e)}"]
            )

    def _analyze_quality_scores(self, quality_scores: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze quality scores from labeling results."""
        try:
            if not quality_scores:
                return {'error': 'No quality scores available'}

            analysis = {
                'overall_summary': {
                    'total_targets': len(quality_scores),
                    'avg_overall_quality': np.mean([qs.get('overall_quality', 0) for qs in quality_scores.values()]),
                    'avg_predictability': np.mean([qs.get('predictability', 0) for qs in quality_scores.values()]),
                    'avg_stability': np.mean([qs.get('stability', 0) for qs in quality_scores.values()]),
                    'avg_balance': np.mean([qs.get('balance', 0) for qs in quality_scores.values()])
                },
                'target_details': {}
            }

            # Analyze individual targets
            for target_name, quality_score in quality_scores.items():
                target_analysis = {
                    'overall_quality': quality_score.get('overall_quality', 0),
                    'predictability': quality_score.get('predictability', 0),
                    'stability': quality_score.get('stability', 0),
                    'balance': quality_score.get('balance', 0),
                    'auc_mean': quality_score.get('auc_mean', 0),
                    'class_balance': quality_score.get('class_balance', 0),
                    'quality_rating': self._rate_quality(quality_score)
                }
                analysis['target_details'][target_name] = target_analysis

            return analysis

        except Exception as e:
            tprint_warning(f"⚠️ Error analyzing quality scores: {e}")
            return {'error': str(e)}

    def _analyze_regime_statistics(self, regime_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze regime-specific statistics."""
        try:
            if not regime_data:
                return {'error': 'No regime data provided'}

            analysis = {
                'regime_count': regime_data.get('regime_count', 0),
                'regime_distribution': regime_data.get('regime_distribution', {}),
                'regime_continuity_score': regime_data.get('regime_continuity_score', 0),
                'regime_specific_insights': {}
            }

            # Analyze regime-specific insights
            if 'regime_stats' in regime_data:
                for regime_id, stats in regime_data['regime_stats'].items():
                    analysis['regime_specific_insights'][regime_id] = {
                        'data_points': stats.get('data_points', 0),
                        'percentage': stats.get('percentage', 0),
                        'volatility_std': stats.get('volatility_std', 0),
                        'mean_volume': stats.get('mean_volume', 0),
                        'labeling_suitability': self._assess_regime_labeling_suitability(stats)
                    }

            return analysis

        except Exception as e:
            tprint_warning(f"⚠️ Error analyzing regime statistics: {e}")
            return {'error': str(e)}

    def _analyze_label_distribution(self, labeling_result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze label distribution across targets."""
        try:
            labels = labeling_result.get('multi_horizon_labeling_result', {}).get('labels')
            if labels is None or labels.empty:
                return {'error': 'No labels available for distribution analysis'}

            analysis = {
                'total_samples': len(labels),
                'target_distributions': {},
                'class_balance_summary': {}
            }

            # Analyze each target column
            for column in labels.columns:
                if 'target' in column.lower():
                    target_dist = labels[column].value_counts().to_dict()
                    analysis['target_distributions'][column] = {
                        'distribution': target_dist,
                        'class_balance': self._calculate_class_balance(target_dist),
                        'unique_classes': len(target_dist)
                    }

            # Calculate overall class balance
            if analysis['target_distributions']:
                avg_balance = np.mean([
                    dist['class_balance']
                    for dist in analysis['target_distributions'].values()
                ])
                analysis['class_balance_summary'] = {
                    'average_balance': avg_balance,
                    'balance_rating': 'Good' if 0.35 <= avg_balance <= 0.65 else 'Poor'
                }

            return analysis

        except Exception as e:
            tprint_warning(f"⚠️ Error analyzing label distribution: {e}")
            return {'error': str(e)}

    def _analyze_feature_compatibility(self, feature_lookback_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze compatibility with feature lookback optimization."""
        try:
            if not feature_lookback_data:
                return {'error': 'No feature lookback data provided'}

            analysis = {
                'lookback_periods_optimized': feature_lookback_data.get('lookback_periods_optimized', False),
                'optimal_periods': feature_lookback_data.get('optimal_periods', {}),
                'compatibility_score': 0.0,
                'recommendations': []
            }

            # Assess compatibility
            if analysis['lookback_periods_optimized']:
                analysis['compatibility_score'] = 0.9  # High compatibility
                analysis['recommendations'].append("✅ Feature lookback optimization is compatible with labeling results")
            else:
                analysis['compatibility_score'] = 0.5  # Medium compatibility
                analysis['recommendations'].append("⚠️ Feature lookback optimization may need adjustment for optimal results")

            return analysis

        except Exception as e:
            tprint_warning(f"⚠️ Error analyzing feature compatibility: {e}")
            return {'error': str(e)}

    def _generate_recommendations(
        self,
        labeling_result: Dict[str, Any],
        regime_data: Optional[Dict[str, Any]],
        feature_lookback_data: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []

        try:
            # Quality-based recommendations
            quality_scores = labeling_result.get('multi_horizon_labeling_result', {}).get('quality_scores', {})
            if quality_scores:
                avg_quality = np.mean([qs.get('overall_quality', 0) for qs in quality_scores.values()])

                if avg_quality < 0.5:
                    recommendations.append("⚠️ Low overall label quality detected - consider adjusting labeling parameters")
                elif avg_quality > 0.8:
                    recommendations.append("✅ High label quality - proceed with confidence")

            # Sample size recommendations
            n_samples = labeling_result.get('multi_horizon_labeling_result', {}).get('metadata', {}).get('n_samples', 0)
            if n_samples < 1000:
                recommendations.append("⚠️ Small sample size may limit model performance - consider extending data range")
            elif n_samples > 10000:
                recommendations.append("✅ Sufficient sample size for robust model training")

            # Regime-based recommendations
            if regime_data:
                regime_count = regime_data.get('regime_count', 0)
                if regime_count < 3:
                    recommendations.append("⚠️ Limited regime diversity - consider more sophisticated regime detection")
                elif regime_count > 10:
                    recommendations.append("✅ Good regime diversity for differentiated labeling")

            # Feature compatibility recommendations
            if feature_lookback_data:
                compatibility_score = self._analyze_feature_compatibility(feature_lookback_data).get('compatibility_score', 0.5)
                if compatibility_score < 0.7:
                    recommendations.append("⚠️ Potential compatibility issues with feature lookback optimization")

            # Default recommendation
            if not recommendations:
                recommendations.append("✅ Labeling results appear suitable for downstream processing")

            return recommendations

        except Exception as e:
            tprint_warning(f"⚠️ Error generating recommendations: {e}")
            return [f"Recommendation generation failed: {str(e)}"]

    def _rate_quality(self, quality_score: Dict[str, Any]) -> str:
        """Rate overall quality of a target."""
        try:
            overall_quality = quality_score.get('overall_quality', 0)
            predictability = quality_score.get('predictability', 0)
            balance = quality_score.get('balance', 0)

            if overall_quality > 0.8 and predictability > 0.7 and 0.35 <= balance <= 0.65:
                return 'Excellent'
            elif overall_quality > 0.6 and predictability > 0.5 and 0.25 <= balance <= 0.75:
                return 'Good'
            elif overall_quality > 0.4 and predictability > 0.3:
                return 'Fair'
            else:
                return 'Poor'

        except Exception:
            return 'Unknown'

    def _calculate_class_balance(self, distribution: Dict[str, Any]) -> float:
        """Calculate class balance metric (0-1, where 0.5 is perfectly balanced)."""
        try:
            if not distribution:
                return 0.0

            total_samples = sum(distribution.values())
            if total_samples == 0:
                return 0.0

            # For binary classification, calculate balance between positive and negative classes
            positive_samples = sum(count for label, count in distribution.items() if label in [1, '1', 'positive'])
            negative_samples = sum(count for label, count in distribution.items() if label in [-1, '0', 'negative'])

            if positive_samples + negative_samples == 0:
                return 0.5  # Neutral balance if no clear classes

            balance = min(positive_samples, negative_samples) / (positive_samples + negative_samples)
            return balance

        except Exception:
            return 0.5  # Default to balanced

    def _assess_regime_labeling_suitability(self, regime_stats: Dict[str, Any]) -> str:
        """Assess how suitable a regime is for labeling."""
        try:
            data_points = regime_stats.get('data_points', 0)
            volatility = regime_stats.get('volatility_std', 0)
            volume = regime_stats.get('mean_volume', 0)

            if data_points < 100:
                return 'Insufficient Data'
            elif volatility < 0.001:
                return 'Too Stable'
            elif volatility > 0.1:
                return 'Too Volatile'
            elif volume < 1000:
                return 'Low Liquidity'
            else:
                return 'Suitable'

        except Exception:
            return 'Unknown'

    def _save_report(self, report: ProfitLabelingReport, output_directory: str):
        """Save report to files."""
        try:
            # Create output directory
            output_path = Path(output_directory)
            output_path.mkdir(parents=True, exist_ok=True)

            # Save JSON report
            json_path = output_path / f"profit_labeling_report_{report.symbol}_{report.timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            report_dict = {
                'symbol': report.symbol,
                'exchange': report.exchange,
                'timeframe': report.timeframe,
                'timestamp': report.timestamp.isoformat(),
                'processing_time': report.processing_time,
                'n_samples': report.n_samples,
                'n_targets': report.n_targets,
                'n_horizons': report.n_horizons,
                'quality_scores': report.quality_scores,
                'regime_statistics': report.regime_statistics,
                'label_distribution': report.label_distribution,
                'feature_lookback_compatibility': report.feature_lookback_compatibility,
                'recommendations': report.recommendations
            }

            with open(json_path, 'w') as f:
                json.dump(report_dict, f, indent=2, default=str)

            tprint_info(f"💾 Report saved to {json_path}")

            # Generate and save visualizations if matplotlib is available
            try:
                self._generate_visualizations(report, output_path)
            except ImportError:
                tprint_warning("⚠️ Matplotlib not available for visualization generation")
            except Exception as e:
                tprint_warning(f"⚠️ Error generating visualizations: {e}")

        except Exception as e:
            tprint_error(f"❌ Error saving report: {e}")

    def _generate_visualizations(self, report: ProfitLabelingReport, output_path: Path):
        """Generate visualization plots for the report."""
        try:
            # Set up plotting style
            plt.style.use('default')
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f'Profit Labeling Report - {report.symbol} {report.timeframe}', fontsize=16)

            # Quality scores bar chart
            if report.quality_scores and 'target_details' in report.quality_scores:
                targets = list(report.quality_scores['target_details'].keys())[:5]  # Limit to 5 targets
                qualities = [report.quality_scores['target_details'][t]['overall_quality'] for t in targets]

                axes[0, 0].bar(range(len(targets)), qualities)
                axes[0, 0].set_xticks(range(len(targets)))
                axes[0, 0].set_xticklabels(targets, rotation=45, ha='right')
                axes[0, 0].set_title('Target Quality Scores')
                axes[0, 0].set_ylabel('Quality Score')

            # Label distribution pie chart
            if report.label_distribution and 'target_distributions' in report.label_distribution:
                target_names = list(report.label_distribution['target_distributions'].keys())[:3]  # Limit to 3 targets
                for i, target in enumerate(target_names):
                    if i < 4:  # Only use first 4 subplots
                        ax = axes.flatten()[i+1] if i > 0 else axes[0, 1]

                        dist = report.label_distribution['target_distributions'][target]['distribution']
                        labels = list(dist.keys())
                        sizes = list(dist.values())

                        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
                        ax.set_title(f'{target} Distribution')

            plt.tight_layout()
            plt.savefig(output_path / 'profit_labeling_visualization.png', dpi=300, bbox_inches='tight')
            plt.close()

            tprint_info("📊 Visualizations saved")

        except Exception as e:
            tprint_warning(f"⚠️ Error generating visualizations: {e}")


def generate_profit_labeling_report(
    labeling_result: Dict[str, Any],
    regime_data: Optional[Dict[str, Any]] = None,
    feature_lookback_data: Optional[Dict[str, Any]] = None,
    output_directory: str = "profit_labeling_reports"
) -> ProfitLabelingReport:
    """
    Convenience function to generate a profit labeling report.

    Args:
        labeling_result: Results from multi-horizon profit labeling
        regime_data: Optional regime data for regime-aware analysis
        feature_lookback_data: Optional feature lookback optimization data
        output_directory: Directory to save reports

    Returns:
        ProfitLabelingReport with comprehensive analysis
    """
    generator = ProfitLabelingReportGenerator()
    return generator.generate_report(
        labeling_result,
        regime_data,
        feature_lookback_data,
        output_directory
    )