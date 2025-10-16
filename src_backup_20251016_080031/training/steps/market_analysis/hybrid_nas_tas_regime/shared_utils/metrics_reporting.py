"""
Metrics Reporting Utilities for Hybrid NAS-TAS Regime Detection.

Provides common metrics reporting utilities for consolidated output.
Delivers similar outputs to hmm_clustering but with enhanced hybrid metrics
including NAS, TAS, and consolidated regime analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from dataclasses import dataclass
import time
from datetime import datetime
import json
from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, 
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)

logger = logging.getLogger(__name__)


@dataclass
class MetricsReportingConfig:
    """Configuration for metrics reporting operations."""
    include_detailed_metrics: bool = True
    include_visualization_data: bool = True
    include_performance_metrics: bool = True
    include_economic_metrics: bool = True
    include_trading_metrics: bool = True
    report_format: str = "json"  # "json", "csv", "html"
    save_to_file: bool = True
    output_directory: str = "reports"


@dataclass
class ConsolidatedMetricsReport:
    """Consolidated metrics report for hybrid NAS-TAS regime detection."""
    nas_metrics: Dict[str, Any]
    tas_metrics: Dict[str, Any]
    hybrid_metrics: Dict[str, Any]
    comparison_metrics: Dict[str, Any]
    performance_summary: Dict[str, Any]
    economic_summary: Dict[str, Any]
    trading_summary: Dict[str, Any]
    consolidated_clusters: Dict[str, Any]
    report_metadata: Dict[str, Any]
    execution_time: float
    success: bool
    error_message: Optional[str] = None


class MetricsReporter:
    """Reporter for consolidated metrics and analysis results."""
    
    def __init__(self, config: MetricsReportingConfig):
        """Initialize the metrics reporter.
        
        Args:
            config: Metrics reporting configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Create output directory if needed
        if config.save_to_file:
            import os
            os.makedirs(config.output_directory, exist_ok=True)
        
        self.logger.info("✅ Metrics Reporter initialized")
    
    def generate_consolidated_report(self, nas_results: Dict[str, Any], tas_results: Dict[str, Any], 
                                   hybrid_results: Dict[str, Any]) -> ConsolidatedMetricsReport:
        """Generate consolidated metrics report.
        
        Args:
            nas_results: NAS regime detection results
            tas_results: TAS regime detection results
            hybrid_results: Hybrid consolidation results
            
        Returns:
            ConsolidatedMetricsReport with comprehensive metrics
        """
        try:
            self.logger.info("📊 Generating consolidated metrics report...")
            start_time = time.time()
            
            # Extract metrics from each system
            nas_metrics = self._extract_nas_metrics(nas_results)
            tas_metrics = self._extract_tas_metrics(tas_results)
            hybrid_metrics = self._extract_hybrid_metrics(hybrid_results)
            
            # Generate comparison metrics
            comparison_metrics = self._generate_comparison_metrics(nas_metrics, tas_metrics, hybrid_metrics)
            
            # Generate performance summary
            performance_summary = self._generate_performance_summary(nas_results, tas_results, hybrid_results)
            
            # Generate economic summary
            economic_summary = self._generate_economic_summary(nas_results, tas_results, hybrid_results)
            
            # Generate trading summary
            trading_summary = self._generate_trading_summary(nas_results, tas_results, hybrid_results)
            
            # Generate consolidated clusters
            consolidated_clusters = self._generate_consolidated_clusters(nas_results, tas_results, hybrid_results)
            
            # Create report metadata
            report_metadata = {
                'report_timestamp': datetime.now().isoformat(),
                'nas_system': 'Neural Architecture Search',
                'tas_system': 'Tree Architecture Search',
                'hybrid_system': 'Hybrid NAS-TAS',
                'report_version': '1.0',
                'config': {
                    'include_detailed_metrics': self.config.include_detailed_metrics,
                    'include_visualization_data': self.config.include_visualization_data,
                    'include_performance_metrics': self.config.include_performance_metrics
                }
            }
            
            execution_time = time.time() - start_time
            
            # Create consolidated report
            consolidated_report = ConsolidatedMetricsReport(
                nas_metrics=nas_metrics,
                tas_metrics=tas_metrics,
                hybrid_metrics=hybrid_metrics,
                comparison_metrics=comparison_metrics,
                performance_summary=performance_summary,
                economic_summary=economic_summary,
                trading_summary=trading_summary,
                consolidated_clusters=consolidated_clusters,
                report_metadata=report_metadata,
                execution_time=execution_time,
                success=True
            )
            
            # Save report if configured
            if self.config.save_to_file:
                self._save_report(consolidated_report)
            
            self.logger.info(f"✅ Consolidated metrics report generated in {execution_time:.2f}s")
            
            return consolidated_report
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ Consolidated report generation failed: {e}")
            return ConsolidatedMetricsReport(
                nas_metrics={},
                tas_metrics={},
                hybrid_metrics={},
                comparison_metrics={},
                performance_summary={},
                economic_summary={},
                trading_summary={},
                consolidated_clusters={},
                report_metadata={'error': str(e)},
                execution_time=execution_time,
                success=False,
                error_message=str(e)
            )
    
    def _extract_nas_metrics(self, nas_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metrics from NAS results."""
        try:
            nas_metrics = {
                'regime_count': nas_results.get('regime_count', 0),
                'regime_distribution': nas_results.get('regime_distribution', {}),
                'clustering_quality': nas_results.get('clustering_quality', {}),
                'economic_significance': nas_results.get('economic_significance', {}),
                'trading_viability': nas_results.get('trading_viability', {}),
                'performance_metrics': nas_results.get('performance_metrics', {}),
                'execution_time': nas_results.get('execution_time', 0.0),
                'success': nas_results.get('success', False)
            }
            
            return nas_metrics
            
        except Exception as e:
            self.logger.warning(f"⚠️ NAS metrics extraction failed: {e}")
            return {}
    
    def _extract_tas_metrics(self, tas_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metrics from TAS results."""
        try:
            tas_metrics = {
                'regime_count': tas_results.get('regime_count', 0),
                'regime_distribution': tas_results.get('regime_distribution', {}),
                'clustering_quality': tas_results.get('clustering_quality', {}),
                'economic_significance': tas_results.get('economic_significance', {}),
                'trading_viability': tas_results.get('trading_viability', {}),
                'performance_metrics': tas_results.get('performance_metrics', {}),
                'execution_time': tas_results.get('execution_time', 0.0),
                'success': tas_results.get('success', False)
            }
            
            return tas_metrics
            
        except Exception as e:
            self.logger.warning(f"⚠️ TAS metrics extraction failed: {e}")
            return {}
    
    def _extract_hybrid_metrics(self, hybrid_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metrics from hybrid results."""
        try:
            hybrid_metrics = {
                'consolidated_regime_count': hybrid_results.get('consolidated_regime_count', 0),
                'consolidated_regime_distribution': hybrid_results.get('consolidated_regime_distribution', {}),
                'consolidation_quality': hybrid_results.get('consolidation_quality', {}),
                'consensus_metrics': hybrid_results.get('consensus_metrics', {}),
                'disagreement_metrics': hybrid_results.get('disagreement_metrics', {}),
                'performance_metrics': hybrid_results.get('performance_metrics', {}),
                'execution_time': hybrid_results.get('execution_time', 0.0),
                'success': hybrid_results.get('success', False)
            }
            
            return hybrid_metrics
            
        except Exception as e:
            self.logger.warning(f"⚠️ Hybrid metrics extraction failed: {e}")
            return {}
    
    def _generate_comparison_metrics(self, nas_metrics: Dict[str, Any], tas_metrics: Dict[str, Any], 
                                   hybrid_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comparison metrics between systems."""
        try:
            comparison_metrics = {
                'regime_count_comparison': {
                    'nas': nas_metrics.get('regime_count', 0),
                    'tas': tas_metrics.get('regime_count', 0),
                    'hybrid': hybrid_metrics.get('consolidated_regime_count', 0),
                    'nas_tas_difference': abs(nas_metrics.get('regime_count', 0) - tas_metrics.get('regime_count', 0))
                },
                'execution_time_comparison': {
                    'nas': nas_metrics.get('execution_time', 0.0),
                    'tas': tas_metrics.get('execution_time', 0.0),
                    'hybrid': hybrid_metrics.get('execution_time', 0.0),
                    'total_time': nas_metrics.get('execution_time', 0.0) + tas_metrics.get('execution_time', 0.0) + hybrid_metrics.get('execution_time', 0.0)
                },
                'success_rate_comparison': {
                    'nas_success': nas_metrics.get('success', False),
                    'tas_success': tas_metrics.get('success', False),
                    'hybrid_success': hybrid_metrics.get('success', False),
                    'overall_success': nas_metrics.get('success', False) and tas_metrics.get('success', False) and hybrid_metrics.get('success', False)
                },
                'clustering_quality_comparison': self._compare_clustering_quality(nas_metrics, tas_metrics, hybrid_metrics),
                'economic_significance_comparison': self._compare_economic_significance(nas_metrics, tas_metrics, hybrid_metrics),
                'trading_viability_comparison': self._compare_trading_viability(nas_metrics, tas_metrics, hybrid_metrics)
            }
            
            return comparison_metrics
            
        except Exception as e:
            self.logger.warning(f"⚠️ Comparison metrics generation failed: {e}")
            return {}
    
    def _compare_clustering_quality(self, nas_metrics: Dict[str, Any], tas_metrics: Dict[str, Any], 
                                  hybrid_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Compare clustering quality between systems."""
        try:
            nas_quality = nas_metrics.get('clustering_quality', {})
            tas_quality = tas_metrics.get('clustering_quality', {})
            hybrid_quality = hybrid_metrics.get('consolidation_quality', {})
            
            comparison = {
                'silhouette_scores': {
                    'nas': nas_quality.get('silhouette_score', 0.0),
                    'tas': tas_quality.get('silhouette_score', 0.0),
                    'hybrid': hybrid_quality.get('silhouette_score', 0.0)
                },
                'calinski_harabasz_scores': {
                    'nas': nas_quality.get('calinski_harabasz_score', 0.0),
                    'tas': tas_quality.get('calinski_harabasz_score', 0.0),
                    'hybrid': hybrid_quality.get('calinski_harabasz_score', 0.0)
                },
                'best_system': self._determine_best_system(nas_quality, tas_quality, hybrid_quality)
            }
            
            return comparison
            
        except Exception as e:
            self.logger.warning(f"⚠️ Clustering quality comparison failed: {e}")
            return {}
    
    def _compare_economic_significance(self, nas_metrics: Dict[str, Any], tas_metrics: Dict[str, Any], 
                                     hybrid_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Compare economic significance between systems."""
        try:
            nas_economic = nas_metrics.get('economic_significance', {})
            tas_economic = tas_metrics.get('economic_significance', {})
            hybrid_economic = hybrid_metrics.get('consensus_metrics', {})
            
            comparison = {
                'overall_scores': {
                    'nas': nas_economic.get('overall_score', 0.0),
                    'tas': tas_economic.get('overall_score', 0.0),
                    'hybrid': hybrid_economic.get('economic_consensus_score', 0.0)
                },
                'significant_regimes': {
                    'nas': nas_economic.get('significant_regimes_count', 0),
                    'tas': tas_economic.get('significant_regimes_count', 0),
                    'hybrid': hybrid_economic.get('consolidated_significant_regimes', 0)
                }
            }
            
            return comparison
            
        except Exception as e:
            self.logger.warning(f"⚠️ Economic significance comparison failed: {e}")
            return {}
    
    def _compare_trading_viability(self, nas_metrics: Dict[str, Any], tas_metrics: Dict[str, Any], 
                                 hybrid_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Compare trading viability between systems."""
        try:
            nas_trading = nas_metrics.get('trading_viability', {})
            tas_trading = tas_metrics.get('trading_viability', {})
            hybrid_trading = hybrid_metrics.get('consensus_metrics', {})
            
            comparison = {
                'overall_scores': {
                    'nas': nas_trading.get('overall_score', 0.0),
                    'tas': tas_trading.get('overall_score', 0.0),
                    'hybrid': hybrid_trading.get('trading_consensus_score', 0.0)
                },
                'viable_regimes': {
                    'nas': nas_trading.get('viable_regimes_count', 0),
                    'tas': tas_trading.get('viable_regimes_count', 0),
                    'hybrid': hybrid_trading.get('consolidated_viable_regimes', 0)
                }
            }
            
            return comparison
            
        except Exception as e:
            self.logger.warning(f"⚠️ Trading viability comparison failed: {e}")
            return {}
    
    def _determine_best_system(self, nas_quality: Dict[str, Any], tas_quality: Dict[str, Any], 
                              hybrid_quality: Dict[str, Any]) -> str:
        """Determine the best performing system."""
        try:
            nas_score = nas_quality.get('silhouette_score', 0.0)
            tas_score = tas_quality.get('silhouette_score', 0.0)
            hybrid_score = hybrid_quality.get('silhouette_score', 0.0)
            
            if hybrid_score >= max(nas_score, tas_score):
                return 'hybrid'
            elif nas_score >= tas_score:
                return 'nas'
            else:
                return 'tas'
                
        except Exception:
            return 'unknown'
    
    def _generate_performance_summary(self, nas_results: Dict[str, Any], tas_results: Dict[str, Any], 
                                    hybrid_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate performance summary."""
        try:
            performance_summary = {
                'total_execution_time': (
                    nas_results.get('execution_time', 0.0) + 
                    tas_results.get('execution_time', 0.0) + 
                    hybrid_results.get('execution_time', 0.0)
                ),
                'system_breakdown': {
                    'nas_time': nas_results.get('execution_time', 0.0),
                    'tas_time': tas_results.get('execution_time', 0.0),
                    'hybrid_time': hybrid_results.get('execution_time', 0.0)
                },
                'memory_usage': {
                    'nas_memory': nas_results.get('memory_usage', {}),
                    'tas_memory': tas_results.get('memory_usage', {}),
                    'hybrid_memory': hybrid_results.get('memory_usage', {})
                },
                'hardware_optimization': {
                    'nas_optimization': nas_results.get('hardware_optimization', {}),
                    'tas_optimization': tas_results.get('hardware_optimization', {}),
                    'hybrid_optimization': hybrid_results.get('hardware_optimization', {})
                }
            }
            
            return performance_summary
            
        except Exception as e:
            self.logger.warning(f"⚠️ Performance summary generation failed: {e}")
            return {}
    
    def _generate_economic_summary(self, nas_results: Dict[str, Any], tas_results: Dict[str, Any], 
                                 hybrid_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate economic summary."""
        try:
            economic_summary = {
                'consolidated_economic_metrics': {
                    'nas_economic': nas_results.get('economic_significance', {}),
                    'tas_economic': tas_results.get('economic_significance', {}),
                    'hybrid_economic': hybrid_results.get('consensus_metrics', {})
                },
                'economic_consensus': hybrid_results.get('consensus_metrics', {}).get('economic_consensus_score', 0.0),
                'economic_disagreement': hybrid_results.get('disagreement_metrics', {}).get('economic_disagreement_score', 0.0),
                'recommended_system': self._recommend_economic_system(nas_results, tas_results, hybrid_results)
            }
            
            return economic_summary
            
        except Exception as e:
            self.logger.warning(f"⚠️ Economic summary generation failed: {e}")
            return {}
    
    def _generate_trading_summary(self, nas_results: Dict[str, Any], tas_results: Dict[str, Any], 
                                hybrid_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading summary."""
        try:
            trading_summary = {
                'consolidated_trading_metrics': {
                    'nas_trading': nas_results.get('trading_viability', {}),
                    'tas_trading': tas_results.get('trading_viability', {}),
                    'hybrid_trading': hybrid_results.get('consensus_metrics', {})
                },
                'trading_consensus': hybrid_results.get('consensus_metrics', {}).get('trading_consensus_score', 0.0),
                'trading_disagreement': hybrid_results.get('disagreement_metrics', {}).get('trading_disagreement_score', 0.0),
                'recommended_system': self._recommend_trading_system(nas_results, tas_results, hybrid_results)
            }
            
            return trading_summary
            
        except Exception as e:
            self.logger.warning(f"⚠️ Trading summary generation failed: {e}")
            return {}
    
    def _generate_consolidated_clusters(self, nas_results: Dict[str, Any], tas_results: Dict[str, Any], 
                                      hybrid_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate consolidated cluster information."""
        try:
            consolidated_clusters = {
                'nas_clusters': {
                    'regime_assignments': nas_results.get('regime_assignments', []),
                    'regime_characteristics': nas_results.get('regime_characteristics', {}),
                    'cluster_centers': nas_results.get('cluster_centers', [])
                },
                'tas_clusters': {
                    'regime_assignments': tas_results.get('regime_assignments', []),
                    'regime_characteristics': tas_results.get('regime_characteristics', {}),
                    'cluster_centers': tas_results.get('cluster_centers', [])
                },
                'hybrid_clusters': {
                    'consolidated_assignments': hybrid_results.get('consolidated_assignments', []),
                    'consolidated_characteristics': hybrid_results.get('consolidated_characteristics', {}),
                    'consolidated_centers': hybrid_results.get('consolidated_centers', []),
                    'consensus_mapping': hybrid_results.get('consensus_mapping', {}),
                    'disagreement_analysis': hybrid_results.get('disagreement_analysis', {})
                }
            }
            
            return consolidated_clusters
            
        except Exception as e:
            self.logger.warning(f"⚠️ Consolidated clusters generation failed: {e}")
            return {}
    
    def _recommend_economic_system(self, nas_results: Dict[str, Any], tas_results: Dict[str, Any], 
                                  hybrid_results: Dict[str, Any]) -> str:
        """Recommend the best system for economic analysis."""
        try:
            nas_score = nas_results.get('economic_significance', {}).get('overall_score', 0.0)
            tas_score = tas_results.get('economic_significance', {}).get('overall_score', 0.0)
            hybrid_score = hybrid_results.get('consensus_metrics', {}).get('economic_consensus_score', 0.0)
            
            if hybrid_score >= max(nas_score, tas_score):
                return 'hybrid'
            elif nas_score >= tas_score:
                return 'nas'
            else:
                return 'tas'
                
        except Exception:
            return 'hybrid'
    
    def _recommend_trading_system(self, nas_results: Dict[str, Any], tas_results: Dict[str, Any], 
                                hybrid_results: Dict[str, Any]) -> str:
        """Recommend the best system for trading analysis."""
        try:
            nas_score = nas_results.get('trading_viability', {}).get('overall_score', 0.0)
            tas_score = tas_results.get('trading_viability', {}).get('overall_score', 0.0)
            hybrid_score = hybrid_results.get('consensus_metrics', {}).get('trading_consensus_score', 0.0)
            
            if hybrid_score >= max(nas_score, tas_score):
                return 'hybrid'
            elif nas_score >= tas_score:
                return 'nas'
            else:
                return 'tas'
                
        except Exception:
            return 'hybrid'
    
    def _save_report(self, report: ConsolidatedMetricsReport):
        """Save the consolidated report to file."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if self.config.report_format == "json":
                filename = f"consolidated_metrics_report_{timestamp}.json"
                filepath = f"{self.config.output_directory}/{filename}"
                
                # Convert report to JSON-serializable format
                report_dict = {
                    'nas_metrics': report.nas_metrics,
                    'tas_metrics': report.tas_metrics,
                    'hybrid_metrics': report.hybrid_metrics,
                    'comparison_metrics': report.comparison_metrics,
                    'performance_summary': report.performance_summary,
                    'economic_summary': report.economic_summary,
                    'trading_summary': report.trading_summary,
                    'consolidated_clusters': report.consolidated_clusters,
                    'report_metadata': report.report_metadata,
                    'execution_time': report.execution_time,
                    'success': report.success
                }
                
                with open(filepath, 'w') as f:
                    json.dump(report_dict, f, indent=2, default=str)
                
                self.logger.info(f"📄 Report saved to: {filepath}")
            
            elif self.config.report_format == "csv":
                # Save key metrics as CSV
                filename = f"consolidated_metrics_summary_{timestamp}.csv"
                filepath = f"{self.config.output_directory}/{filename}"
                
                # Create summary DataFrame
                summary_data = {
                    'Metric': ['NAS Regime Count', 'TAS Regime Count', 'Hybrid Regime Count', 
                              'NAS Execution Time', 'TAS Execution Time', 'Hybrid Execution Time',
                              'Overall Success'],
                    'Value': [
                        report.nas_metrics.get('regime_count', 0),
                        report.tas_metrics.get('regime_count', 0),
                        report.hybrid_metrics.get('consolidated_regime_count', 0),
                        report.nas_metrics.get('execution_time', 0.0),
                        report.tas_metrics.get('execution_time', 0.0),
                        report.hybrid_metrics.get('execution_time', 0.0),
                        report.success
                    ]
                }
                
                df = pd.DataFrame(summary_data)
                df.to_csv(filepath, index=False)
                
                self.logger.info(f"📄 Summary saved to: {filepath}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Report saving failed: {e}")
    
    def generate_summary_report(self, consolidated_report: ConsolidatedMetricsReport) -> str:
        """Generate a human-readable summary report."""
        try:
            summary = f"""
# Hybrid NAS-TAS Regime Detection Report

## Executive Summary
- **Total Execution Time**: {consolidated_report.execution_time:.2f} seconds
- **Overall Success**: {'✅' if consolidated_report.success else '❌'}
- **Report Generated**: {consolidated_report.report_metadata.get('report_timestamp', 'Unknown')}

## System Performance Comparison
- **NAS Regime Count**: {consolidated_report.nas_metrics.get('regime_count', 0)}
- **TAS Regime Count**: {consolidated_report.tas_metrics.get('regime_count', 0)}
- **Hybrid Regime Count**: {consolidated_report.hybrid_metrics.get('consolidated_regime_count', 0)}

## Clustering Quality
- **Best System**: {consolidated_report.comparison_metrics.get('clustering_quality_comparison', {}).get('best_system', 'Unknown')}
- **NAS Silhouette Score**: {consolidated_report.nas_metrics.get('clustering_quality', {}).get('silhouette_score', 0.0):.3f}
- **TAS Silhouette Score**: {consolidated_report.tas_metrics.get('clustering_quality', {}).get('silhouette_score', 0.0):.3f}
- **Hybrid Silhouette Score**: {consolidated_report.hybrid_metrics.get('consolidation_quality', {}).get('silhouette_score', 0.0):.3f}

## Economic Significance
- **Recommended System**: {consolidated_report.economic_summary.get('recommended_system', 'Unknown')}
- **Economic Consensus Score**: {consolidated_report.economic_summary.get('economic_consensus', 0.0):.3f}

## Trading Viability
- **Recommended System**: {consolidated_report.trading_summary.get('recommended_system', 'Unknown')}
- **Trading Consensus Score**: {consolidated_report.trading_summary.get('trading_consensus', 0.0):.3f}

## Recommendations
1. **Primary System**: Use {consolidated_report.economic_summary.get('recommended_system', 'hybrid')} for economic analysis
2. **Trading System**: Use {consolidated_report.trading_summary.get('recommended_system', 'hybrid')} for trading decisions
3. **Overall**: The hybrid system provides the most comprehensive analysis combining both NAS and TAS strengths
"""
            
            return summary
            
        except Exception as e:
            self.logger.warning(f"⚠️ Summary report generation failed: {e}")
            return f"Error generating summary: {e}"


def create_metrics_reporter(config: MetricsReportingConfig) -> MetricsReporter:
    """Create a metrics reporter instance.
    
    Args:
        config: Metrics reporting configuration
        
    Returns:
        MetricsReporter instance
    """
    return MetricsReporter(config)