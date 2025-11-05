"""
Quality Assessment Integration with Cluster Quality Assessor

This module integrates comprehensive quality assessment with existing
cluster_quality_assessor.py and provides CSV export with datetime.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import csv
import json
from pathlib import Path

# Import existing cluster quality assessor
try:
    from ...clusters.cluster_quality_assessor import ClusterQualityAssessor, ClusterQualityMetrics
    CLUSTER_ASSESSOR_AVAILABLE = True
except ImportError:
    CLUSTER_ASSESSOR_AVAILABLE = False

# Import utilities
try:
    from src.utils.tprint import (
        tprint_info, tprint_success, tprint_warning, tprint_error
    )
except ImportError:
    def tprint_info(msg): print(f'ℹ️  {msg}')
    def tprint_success(msg): print(f'✅ {msg}')
    def tprint_warning(msg): print(f'⚠️  {msg}')
    def tprint_error(msg): print(f'❌ {msg}')


@dataclass
class QualityAssessmentConfig:
    """Configuration for quality assessment integration."""
    # Output settings
    output_dir: str = "outcomes"
    include_datetime: bool = True
    csv_format: str = "detailed"  # "detailed", "summary", "both"
    
    # Assessment components
    enable_stability_metrics: bool = True
    enable_calibration_tests: bool = True
    enable_residual_analysis: bool = True
    enable_sensitivity_analysis: bool = True
    enable_change_point_detection: bool = True
    enable_economic_validation: bool = True
    
    # Integration settings
    integrate_with_cluster_assessor: bool = True
    merge_with_existing_metrics: bool = True


class QualityAssessmentIntegrator:
    """
    Integrates comprehensive quality assessment with cluster_quality_assessor.
    
    This class combines all quality assessment components and provides
    standardized CSV export with datetime in filename.
    """
    
    def __init__(self, config: Optional[QualityAssessmentConfig] = None):
        """
        Initialize quality assessment integrator.
        
        Args:
            config: Configuration for quality assessment
        """
        self.config = config or QualityAssessmentConfig()
        
        tprint_info("🔧 Initialized Quality Assessment Integrator")
        
        # Initialize cluster quality assessor
        if CLUSTER_ASSESSOR_AVAILABLE and self.config.integrate_with_cluster_assessor:
            self.cluster_assessor = ClusterQualityAssessor()
        else:
            self.cluster_assessor = None
            tprint_warning("⚠️ Cluster quality assessor not available")
        
        # Initialize assessment components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize assessment components."""
        tprint_info("🔧 Initializing assessment components")
        
        if self.config.enable_stability_metrics:
            from .stability_metrics import StabilityMetricsCalculator
            self.stability_calculator = StabilityMetricsCalculator()
            tprint_info("✅ Stability metrics calculator initialized")
        
        if self.config.enable_calibration_tests:
            from .calibration_tests import CalibrationTester
            self.calibration_tester = CalibrationTester()
            tprint_info("✅ Calibration tester initialized")
        
        if self.config.enable_residual_analysis:
            from .residual_tests import ResidualAnalyzer
            self.residual_analyzer = ResidualAnalyzer()
            tprint_info("✅ Residual analyzer initialized")
        
        if self.config.enable_sensitivity_analysis:
            from .sensitivity_analysis import SensitivityAnalyzer
            self.sensitivity_analyzer = SensitivityAnalyzer()
            tprint_info("✅ Sensitivity analyzer initialized")
        
        if self.config.enable_change_point_detection:
            from .change_point_detection import ChangePointDetector
            self.change_point_detector = ChangePointDetector()
            tprint_info("✅ Change point detector initialized")
        
        if self.config.enable_economic_validation:
            from .economic_validation import EconomicValidator
            self.economic_validator = EconomicValidator()
            tprint_info("✅ Economic validator initialized")
        
        tprint_success("✅ All assessment components initialized")
    
    def assess_quality(self, 
                     model: Any,
                     data: pd.DataFrame,
                     regime_labels: np.ndarray,
                     forward_returns: Optional[pd.Series] = None,
                     timestamps: Optional[pd.DatetimeIndex] = None,
                     symbol: str = "UNKNOWN") -> Dict[str, Any]:
        """
        Perform comprehensive quality assessment.
        
        Args:
            model: Fitted clustering model
            data: Feature data
            regime_labels: Predicted regime labels
            forward_returns: Optional forward returns
            timestamps: Optional timestamps
            symbol: Symbol identifier
            
        Returns:
            Comprehensive quality assessment results
        """
        tprint_info("🔍 Starting comprehensive quality assessment")
        
        try:
            # Initialize results dictionary
            results = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'assessment_config': self.config.__dict__
            }
            
            # 1. Standard cluster quality assessment
            if self.cluster_assessor is not None:
                tprint_info("📊 Running standard cluster quality assessment")
                standard_metrics = self.cluster_assessor.assess_quality(
                    regime_labels=regime_labels,
                    feature_data=data,
                    forward_returns=forward_returns,
                    timestamps=timestamps
                )
                results['standard_quality_metrics'] = standard_metrics.to_dict()
            
            # 2. Enhanced stability metrics
            if self.config.enable_stability_metrics:
                tprint_info("📈 Calculating stability metrics")
                stability_metrics = self.stability_calculator.calculate_stability(
                    regime_labels, data, model
                )
                results['stability_metrics'] = stability_metrics
            
            # 3. Calibration tests
            if self.config.enable_calibration_tests:
                tprint_info("🎯 Running calibration tests")
                calibration_metrics = self.calibration_tester.test_calibration(
                    model, regime_labels, data
                )
                results['calibration_metrics'] = calibration_metrics
            
            # 4. Residual analysis
            if self.config.enable_residual_analysis:
                tprint_info("🔍 Analyzing residuals")
                residual_metrics = self.residual_analyzer.analyze_residuals(
                    model, data, regime_labels
                )
                results['residual_metrics'] = residual_metrics
            
            # 5. Sensitivity analysis
            if self.config.enable_sensitivity_analysis:
                tprint_info("📊 Performing sensitivity analysis")
                sensitivity_metrics = self.sensitivity_analyzer.analyze_sensitivity(
                    model, data, regime_labels
                )
                results['sensitivity_metrics'] = sensitivity_metrics
            
            # 6. Change point detection
            if self.config.enable_change_point_detection:
                tprint_info("📍 Detecting change points")
                change_point_metrics = self.change_point_detector.detect_change_points(
                    data, regime_labels
                )
                results['change_point_metrics'] = change_point_metrics
            
            # 7. Economic validation
            if self.config.enable_economic_validation and forward_returns is not None:
                tprint_info("💰 Performing economic validation")
                economic_metrics = self.economic_validator.validate_economics(
                    regime_labels, forward_returns, data
                )
                results['economic_metrics'] = economic_metrics
            
            # 8. Generate comprehensive CSV reports
            csv_paths = self._generate_csv_reports(results, symbol)
            results['csv_reports'] = csv_paths
            
            tprint_success("✅ Comprehensive quality assessment complete")
            return results
            
        except Exception as e:
            tprint_error(f"❌ Quality assessment failed: {e}")
            raise
    
    def _generate_csv_reports(self, 
                           results: Dict[str, Any],
                           symbol: str) -> Dict[str, str]:
        """Generate comprehensive CSV reports with datetime in filename."""
        tprint_info("📄 Generating CSV reports")
        
        # Create output directory
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        csv_paths = {}
        
        # 1. Detailed quality metrics CSV
        if self.config.csv_format in ['detailed', 'both']:
            detailed_path = output_dir / f"quality_assessment_detailed_{symbol}_{timestamp}.csv"
            self._write_detailed_csv(results, detailed_path)
            csv_paths['detailed'] = str(detailed_path)
        
        # 2. Summary quality metrics CSV
        if self.config.csv_format in ['summary', 'both']:
            summary_path = output_dir / f"quality_assessment_summary_{symbol}_{timestamp}.csv"
            self._write_summary_csv(results, summary_path)
            csv_paths['summary'] = str(summary_path)
        
        # 3. Integration with cluster_quality_assessor CSV
        if self.cluster_assessor is not None and 'standard_quality_metrics' in results:
            cluster_path = output_dir / f"cluster_quality_metrics_{symbol}_{timestamp}.csv"
            self._write_cluster_quality_csv(
                results['standard_quality_metrics'], cluster_path
            )
            csv_paths['cluster_quality'] = str(cluster_path)
        
        return csv_paths
    
    def _write_detailed_csv(self, results: Dict[str, Any], output_path: Path):
        """Write detailed quality metrics to CSV."""
        tprint_info(f"📝 Writing detailed CSV to: {output_path}")
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow(['Category', 'Metric', 'Value', 'Description'])
            
            # Write standard quality metrics
            if 'standard_quality_metrics' in results:
                tprint_info("📊 Writing standard quality metrics")
                self._write_standard_metrics_to_csv(
                    writer, results['standard_quality_metrics'], 'Standard Quality'
                )
            
            # Write stability metrics
            if 'stability_metrics' in results:
                tprint_info("📈 Writing stability metrics")
                self._write_stability_metrics_to_csv(
                    writer, results['stability_metrics'], 'Stability Metrics'
                )
            
            # Write calibration metrics
            if 'calibration_metrics' in results:
                tprint_info("🎯 Writing calibration metrics")
                self._write_calibration_metrics_to_csv(
                    writer, results['calibration_metrics'], 'Calibration Tests'
                )
            
            # Write residual metrics
            if 'residual_metrics' in results:
                tprint_info("🔍 Writing residual metrics")
                self._write_residual_metrics_to_csv(
                    writer, results['residual_metrics'], 'Residual Analysis'
                )
            
            # Write sensitivity metrics
            if 'sensitivity_metrics' in results:
                tprint_info("📊 Writing sensitivity metrics")
                self._write_sensitivity_metrics_to_csv(
                    writer, results['sensitivity_metrics'], 'Sensitivity Analysis'
                )
            
            # Write change point metrics
            if 'change_point_metrics' in results:
                tprint_info("📍 Writing change point metrics")
                self._write_change_point_metrics_to_csv(
                    writer, results['change_point_metrics'], 'Change Point Detection'
                )
            
            # Write economic metrics
            if 'economic_metrics' in results:
                tprint_info("💰 Writing economic metrics")
                self._write_economic_metrics_to_csv(
                    writer, results['economic_metrics'], 'Economic Validation'
                )
        
        tprint_success(f"✅ Detailed CSV written: {output_path}")
    
    def _write_summary_csv(self, results: Dict[str, Any], output_path: Path):
        """Write summary quality metrics to CSV."""
        tprint_info(f"📝 Writing summary CSV to: {output_path}")
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow(['Metric', 'Value', 'Category', 'Importance'])
            
            # Extract key summary metrics
            tprint_info("📊 Extracting summary metrics")
            summary_metrics = self._extract_summary_metrics(results)
            
            for metric, info in summary_metrics.items():
                writer.writerow([
                    metric,
                    info['value'],
                    info['category'],
                    info['importance']
                ])
        
        tprint_success(f"✅ Summary CSV written: {output_path}")
    
    def _write_cluster_quality_csv(self, metrics: Dict[str, Any], output_path: Path):
        """Write cluster quality metrics in format compatible with cluster_quality_assessor."""
        tprint_info(f"📝 Writing cluster quality CSV to: {output_path}")
        
        if not CLUSTER_ASSESSOR_AVAILABLE:
            tprint_warning("⚠️ Cluster quality assessor not available, skipping CSV generation")
            return
        
        # Use existing cluster quality assessor CSV generation
        if self.cluster_assessor is not None:
            tprint_info("🔄 Converting metrics to ClusterQualityMetrics format")
            
            # Convert to ClusterQualityMetrics if needed
            if not isinstance(metrics, ClusterQualityMetrics):
                # This is simplified - in practice you'd properly reconstruct
                cluster_metrics = ClusterQualityMetrics()
                for key, value in metrics.items():
                    if hasattr(cluster_metrics, key):
                        setattr(cluster_metrics, key, value)
            else:
                cluster_metrics = metrics
            
            # Generate CSV using cluster quality assessor
            tprint_info("📊 Generating CSV using cluster quality assessor")
            csv_path, _ = self.cluster_assessor.generate_comprehensive_csv_report(
                cluster_metrics,
                symbol=metrics.get('symbol', 'UNKNOWN'),
                output_dir=str(output_path.parent)
            )
            
            # Move to our desired location
            if csv_path and csv_path != output_path:
                tprint_info(f"📁 Moving CSV from {csv_path} to {output_path}")
                import shutil
                shutil.move(csv_path, output_path)
                tprint_success("✅ CSV file moved successfully")
    
    def _extract_summary_metrics(self, results: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Extract key summary metrics from comprehensive results."""
        tprint_info("📊 Extracting summary metrics from comprehensive results")
        summary = {}
        
        # Extract from standard quality metrics
        if 'standard_quality_metrics' in results:
            tprint_info("📈 Extracting from standard quality metrics")
            std_metrics = results['standard_quality_metrics']
            
            if 'quality_score' in std_metrics:
                summary['Overall Quality Score'] = {
                    'value': std_metrics['quality_score'],
                    'category': 'Overall',
                    'importance': 'High'
                }
            
            if 'silhouette_score' in std_metrics:
                summary['Silhouette Score'] = {
                    'value': std_metrics['silhouette_score'],
                    'category': 'Clustering',
                    'importance': 'High'
                }
            
            if 'temporal_smoothness' in std_metrics:
                summary['Temporal Smoothness'] = {
                    'value': std_metrics['temporal_smoothness'],
                    'category': 'Temporal',
                    'importance': 'Medium'
                }
        
        # Extract from stability metrics
        if 'stability_metrics' in results:
            tprint_info("📈 Extracting from stability metrics")
            stability = results['stability_metrics']
            
            if 'mean_ari' in stability:
                summary['Stability ARI'] = {
                    'value': stability['mean_ari'],
                    'category': 'Stability',
                    'importance': 'High'
                }
        
        # Extract from economic metrics
        if 'economic_metrics' in results:
            tprint_info("💰 Extracting from economic metrics")
            economic = results['economic_metrics']
            
            if 'regime_sharpe_ratio' in economic:
                summary['Regime Sharpe Ratio'] = {
                    'value': economic['regime_sharpe_ratio'],
                    'category': 'Economic',
                    'importance': 'High'
                }
        
        tprint_success(f"✅ Extracted {len(summary)} summary metrics")
        return summary
    
    def _write_standard_metrics_to_csv(self, writer, metrics: Dict[str, Any], category: str):
        """Write standard quality metrics to CSV."""
        tprint_info(f"📝 Writing {category} metrics to CSV")
        
        descriptions = {
            'quality_score': 'Overall clustering quality (0-1, higher is better)',
            'silhouette_score': 'Cluster separation and cohesion (-1 to 1)',
            'davies_bouldin_score': 'Cluster similarity (lower is better)',
            'calinski_harabasz_score': 'Between-cluster dispersion (higher is better)',
            'temporal_smoothness': 'Regime persistence over time (0-1)',
            'regime_persistence': 'Average regime duration',
            'n_regimes': 'Number of regimes discovered',
            'noise_ratio': 'Ratio of noise points'
        }
        
        metrics_written = 0
        for metric, value in metrics.items():
            if metric in descriptions:
                writer.writerow([
                    category,
                    metric,
                    str(value),
                    descriptions[metric]
                ])
                metrics_written += 1
        
        tprint_success(f"✅ Wrote {metrics_written} {category} metrics to CSV")
    
    def _write_stability_metrics_to_csv(self, writer, metrics: Dict[str, Any], category: str):
        """Write stability metrics to CSV."""
        tprint_info(f"📝 Writing {category} metrics to CSV")
        
        descriptions = {
            'mean_ari': 'Average Adjusted Rand Index across bootstrap samples',
            'std_ari': 'Standard deviation of ARI across samples',
            'mean_nmi': 'Average Normalized Mutual Information across samples',
            'bootstrap_stability': 'Overall stability score from bootstrap analysis'
        }
        
        metrics_written = 0
        for metric, value in metrics.items():
            if metric in descriptions:
                writer.writerow([
                    category,
                    metric,
                    str(value),
                    descriptions[metric]
                ])
                metrics_written += 1
        
        tprint_success(f"✅ Wrote {metrics_written} {category} metrics to CSV")
    
    def _write_calibration_metrics_to_csv(self, writer, metrics: Dict[str, Any], category: str):
        """Write calibration metrics to CSV."""
        tprint_info(f"📝 Writing {category} metrics to CSV")
        
        descriptions = {
            'reliability_score': 'Reliability diagram score (0-1, higher is better)',
            'calibration_error': 'Mean calibration error (lower is better)',
            'probability_calibration': 'Overall probability calibration assessment'
        }
        
        metrics_written = 0
        for metric, value in metrics.items():
            if metric in descriptions:
                writer.writerow([
                    category,
                    metric,
                    str(value),
                    descriptions[metric]
                ])
                metrics_written += 1
        
        tprint_success(f"✅ Wrote {metrics_written} {category} metrics to CSV")
    
    def _write_residual_metrics_to_csv(self, writer, metrics: Dict[str, Any], category: str):
        """Write residual metrics to CSV."""
        tprint_info(f"📝 Writing {category} metrics to CSV")
        
        descriptions = {
            'ljung_box_pvalue': 'Ljung-Box test p-value for autocorrelation',
            'breusch_pagan_pvalue': 'Breusch-Pagan test p-value for heteroscedasticity',
            'jarque_bera_pvalue': 'Jarque-Bera test p-value for normality',
            'residual_skewness': 'Skewness of residuals',
            'residual_kurtosis': 'Kurtosis of residuals'
        }
        
        metrics_written = 0
        for metric, value in metrics.items():
            if metric in descriptions:
                writer.writerow([
                    category,
                    metric,
                    str(value),
                    descriptions[metric]
                ])
                metrics_written += 1
        
        tprint_success(f"✅ Wrote {metrics_written} {category} metrics to CSV")
    
    def _write_sensitivity_metrics_to_csv(self, writer, metrics: Dict[str, Any], category: str):
        """Write sensitivity metrics to CSV."""
        tprint_info(f"📝 Writing {category} metrics to CSV")
        
        descriptions = {
            'parameter_sensitivity': 'Parameter sensitivity scores',
            'lookback_sensitivity': 'Sensitivity to lookback window changes',
            'robustness_score': 'Overall robustness to perturbations'
        }
        
        metrics_written = 0
        for metric, value in metrics.items():
            if metric in descriptions:
                writer.writerow([
                    category,
                    metric,
                    str(value),
                    descriptions[metric]
                ])
                metrics_written += 1
        
        tprint_success(f"✅ Wrote {metrics_written} {category} metrics to CSV")
    
    def _write_change_point_metrics_to_csv(self, writer, metrics: Dict[str, Any], category: str):
        """Write change point metrics to CSV."""
        tprint_info(f"📝 Writing {category} metrics to CSV")
        
        descriptions = {
            'n_change_points': 'Number of detected change points',
            'change_point_alignment': 'Alignment with regime boundaries',
            'ruptures_score': 'Change point detection score'
        }
        
        metrics_written = 0
        for metric, value in metrics.items():
            if metric in descriptions:
                writer.writerow([
                    category,
                    metric,
                    str(value),
                    descriptions[metric]
                ])
                metrics_written += 1
        
        tprint_success(f"✅ Wrote {metrics_written} {category} metrics to CSV")
    
    def _write_economic_metrics_to_csv(self, writer, metrics: Dict[str, Any], category: str):
        """Write economic metrics to CSV."""
        tprint_info(f"📝 Writing {category} metrics to CSV")
        
        descriptions = {
            'regime_sharpe_ratio': 'Sharpe ratio by regime',
            'regime_hit_rate': 'Hit rate by regime',
            'regime_max_drawdown': 'Maximum drawdown by regime',
            'economic_utility': 'Overall economic utility score',
            'turnover_penalty': 'Turnover penalty for regime switches'
        }
        
        metrics_written = 0
        for metric, value in metrics.items():
            if metric in descriptions:
                writer.writerow([
                    category,
                    metric,
                    str(value),
                    descriptions[metric]
                ])
                metrics_written += 1
        
        tprint_success(f"✅ Wrote {metrics_written} {category} metrics to CSV")


def create_quality_assessment_integrator(
    output_dir: str = "outcomes",
    include_datetime: bool = True,
    integrate_with_cluster_assessor: bool = True,
    enable_all_assessments: bool = True
) -> QualityAssessmentIntegrator:
    """
    Factory function to create quality assessment integrator.
    
    Args:
        output_dir: Output directory for reports
        include_datetime: Include datetime in filenames
        integrate_with_cluster_assessor: Integrate with cluster_quality_assessor
        enable_all_assessments: Enable all assessment components
        
    Returns:
        QualityAssessmentIntegrator instance
    """
    tprint_info("🏭 Creating Quality Assessment Integrator with factory function")
    
    config = QualityAssessmentConfig(
        output_dir=output_dir,
        include_datetime=include_datetime,
        integrate_with_cluster_assessor=integrate_with_cluster_assessor,
        enable_stability_metrics=enable_all_assessments,
        enable_calibration_tests=enable_all_assessments,
        enable_residual_analysis=enable_all_assessments,
        enable_sensitivity_analysis=enable_all_assessments,
        enable_change_point_detection=enable_all_assessments,
        enable_economic_validation=enable_all_assessments
    )
    
    tprint_success("✅ Quality Assessment Integrator created successfully")
    return QualityAssessmentIntegrator(config)