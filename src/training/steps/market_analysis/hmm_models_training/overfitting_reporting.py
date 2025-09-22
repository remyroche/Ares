"""
Enhanced Overfitting Detection Reporting

Comprehensive reporting system for overfitting detection with detailed analysis,
actionable insights, and visual reporting capabilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

@dataclass
class OverfittingReport:
    """Comprehensive overfitting detection report."""
    
    # Basic metrics
    train_accuracy: float
    val_accuracy: float
    accuracy_gap: float
    train_f1: float
    val_f1: float
    f1_gap: float
    
    # Overfitting status
    is_overfitting: bool
    severity: str  # 'none', 'moderate', 'high', 'severe'
    confidence_level: float  # 0.0 to 1.0
    
    # Detailed analysis
    indicators: List[str]
    warnings: List[str]
    recommendations: List[str]
    
    # Advanced metrics
    train_confidence: Optional[float] = None
    val_confidence: Optional[float] = None
    confidence_gap: Optional[float] = None
    overconfident_ratio: Optional[float] = None
    feature_concentration: Optional[float] = None
    cv_variance: Optional[float] = None
    cv_test_gap: Optional[float] = None
    
    # Performance tracking
    detection_timestamp: str = None
    model_name: str = "unknown"
    fold_number: Optional[int] = None
    
    def __post_init__(self):
        """Initialize timestamp if not provided."""
        if self.detection_timestamp is None:
            self.detection_timestamp = datetime.now().isoformat()

@dataclass
class OverfittingTrend:
    """Track overfitting trends across training epochs/folds."""
    
    epoch_fold: int
    train_accuracy: float
    val_accuracy: float
    accuracy_gap: float
    is_overfitting: bool
    severity: str
    timestamp: str

class OverfittingReporter:
    """Enhanced overfitting detection reporter with comprehensive analysis."""
    
    def __init__(self, 
                 save_reports: bool = True,
                 report_directory: str = "reports/overfitting",
                 enable_visualization: bool = True,
                 detailed_logging: bool = True):
        """
        Initialize overfitting reporter.
        
        Args:
            save_reports: Whether to save reports to disk
            report_directory: Directory to save reports
            enable_visualization: Whether to generate visualizations
            detailed_logging: Whether to enable detailed logging
        """
        self.save_reports = save_reports
        self.report_directory = Path(report_directory)
        self.enable_visualization = enable_visualization
        self.detailed_logging = detailed_logging
        
        # Create report directory
        if self.save_reports:
            self.report_directory.mkdir(parents=True, exist_ok=True)
        
        # Track trends
        self.overfitting_trends = []
        self.report_history = []
        
        # Severity thresholds
        self.severity_thresholds = {
            'none': 0.0,
            'moderate': 0.05,
            'high': 0.10,
            'severe': 0.15
        }
    
    def generate_comprehensive_report(self, 
                                    overfitting_analysis: Dict[str, Any],
                                    model_name: str = "unknown",
                                    fold_number: Optional[int] = None) -> OverfittingReport:
        """
        Generate comprehensive overfitting report.
        
        Args:
            overfitting_analysis: Overfitting analysis results
            model_name: Name of the model
            fold_number: Fold number (for cross-validation)
            
        Returns:
            OverfittingReport: Comprehensive report
        """
        try:
            # Extract basic metrics
            metrics = overfitting_analysis.get('metrics', {})
            train_acc = metrics.get('train_accuracy', 0.0)
            val_acc = metrics.get('val_accuracy', 0.0)
            accuracy_gap = train_acc - val_acc
            
            train_f1 = metrics.get('train_f1', 0.0)
            val_f1 = metrics.get('val_f1', 0.0)
            f1_gap = train_f1 - val_f1
            
            # Determine severity
            severity = self._determine_severity(accuracy_gap, f1_gap, overfitting_analysis)
            confidence_level = self._calculate_confidence_level(overfitting_analysis)
            
            # Generate indicators and warnings
            indicators = self._generate_indicators(overfitting_analysis)
            warnings = self._generate_warnings(overfitting_analysis, severity)
            recommendations = self._generate_recommendations(overfitting_analysis, severity)
            
            # Create comprehensive report
            report = OverfittingReport(
                train_accuracy=train_acc,
                val_accuracy=val_acc,
                accuracy_gap=accuracy_gap,
                train_f1=train_f1,
                val_f1=val_f1,
                f1_gap=f1_gap,
                is_overfitting=overfitting_analysis.get('is_overfitting', False),
                severity=severity,
                confidence_level=confidence_level,
                indicators=indicators,
                warnings=warnings,
                recommendations=recommendations,
                train_confidence=metrics.get('train_confidence'),
                val_confidence=metrics.get('val_confidence'),
                confidence_gap=metrics.get('confidence_gap'),
                overconfident_ratio=metrics.get('overconfident_ratio'),
                feature_concentration=metrics.get('feature_concentration'),
                cv_variance=metrics.get('cv_variance'),
                cv_test_gap=metrics.get('cv_test_gap'),
                model_name=model_name,
                fold_number=fold_number
            )
            
            # Track trends
            self._track_trend(report)
            
            # Save report
            if self.save_reports:
                self._save_report(report)
            
            # Generate visualizations
            if self.enable_visualization:
                self._generate_visualizations(report)
            
            # Log detailed information
            if self.detailed_logging:
                self._log_detailed_report(report)
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate overfitting report: {e}")
            return self._create_error_report(str(e), model_name, fold_number)
    
    def _determine_severity(self, 
                           accuracy_gap: float, 
                           f1_gap: float, 
                           analysis: Dict[str, Any]) -> str:
        """Determine overfitting severity level."""
        # Check accuracy gap
        if accuracy_gap >= self.severity_thresholds['severe']:
            return 'severe'
        elif accuracy_gap >= self.severity_thresholds['high']:
            return 'high'
        elif accuracy_gap >= self.severity_thresholds['moderate']:
            return 'moderate'
        
        # Check F1 gap
        if f1_gap >= 0.10:  # 10% F1 gap
            return 'severe'
        elif f1_gap >= 0.05:  # 5% F1 gap
            return 'high'
        elif f1_gap >= 0.03:  # 3% F1 gap
            return 'moderate'
        
        # Check other indicators
        if analysis.get('is_overfitting', False):
            indicators = analysis.get('indicators', [])
            if len(indicators) >= 3:
                return 'high'
            elif len(indicators) >= 1:
                return 'moderate'
        
        return 'none'
    
    def _calculate_confidence_level(self, analysis: Dict[str, Any]) -> float:
        """Calculate confidence level for overfitting detection."""
        confidence = 0.5  # Base confidence
        
        # Increase confidence based on multiple indicators
        if analysis.get('is_overfitting', False):
            indicators = analysis.get('indicators', [])
            confidence += len(indicators) * 0.1
            
            # High confidence for severe cases
            if 'severe_accuracy_gap' in indicators or 'severe_f1_gap' in indicators:
                confidence += 0.2
            
            # Additional confidence for specific patterns
            if 'confidence_gap' in indicators:
                confidence += 0.1
            if 'overconfident' in indicators:
                confidence += 0.1
            if 'feature_concentration' in indicators:
                confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _generate_indicators(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate overfitting indicators."""
        indicators = []
        
        if not analysis.get('is_overfitting', False):
            return indicators
        
        # Extract indicators from analysis
        warnings = analysis.get('warnings', [])
        
        for warning in warnings:
            if 'SEVERE' in warning.upper():
                if 'accuracy' in warning.lower():
                    indicators.append('severe_accuracy_gap')
                elif 'f1' in warning.lower():
                    indicators.append('severe_f1_gap')
            elif 'Overfitting' in warning:
                if 'accuracy' in warning.lower():
                    indicators.append('accuracy_gap')
                elif 'f1' in warning.lower():
                    indicators.append('f1_gap')
            elif 'Confidence' in warning:
                indicators.append('confidence_gap')
            elif 'Overconfident' in warning:
                indicators.append('overconfident')
            elif 'Feature concentration' in warning:
                indicators.append('feature_concentration')
            elif 'High CV variance' in warning:
                indicators.append('cv_variance')
            elif 'CV/test discrepancy' in warning:
                indicators.append('cv_test_gap')
        
        return indicators
    
    def _generate_warnings(self, analysis: Dict[str, Any], severity: str) -> List[str]:
        """Generate actionable warnings."""
        warnings = []
        
        if not analysis.get('is_overfitting', False):
            return warnings
        
        # Severity-based warnings
        if severity == 'severe':
            warnings.extend([
                "🚨 CRITICAL: Severe overfitting detected - immediate action required",
                "🚨 Model is likely to fail in production",
                "🚨 Consider stopping training and redesigning approach"
            ])
        elif severity == 'high':
            warnings.extend([
                "⚠️ HIGH RISK: Significant overfitting detected",
                "⚠️ Model performance will likely degrade in production",
                "⚠️ Immediate intervention recommended"
            ])
        elif severity == 'moderate':
            warnings.extend([
                "📊 MODERATE: Overfitting detected - monitor closely",
                "📊 Consider regularization or early stopping",
                "📊 Performance may be unstable"
            ])
        
        # Add specific warnings from analysis
        analysis_warnings = analysis.get('warnings', [])
        warnings.extend(analysis_warnings)
        
        return warnings
    
    def _generate_recommendations(self, analysis: Dict[str, Any], severity: str) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        if not analysis.get('is_overfitting', False):
            return recommendations
        
        # Severity-based recommendations
        if severity == 'severe':
            recommendations.extend([
                "🛑 STOP TRAINING: Implement aggressive regularization",
                "🛑 REDUCE COMPLEXITY: Use simpler model architecture",
                "🛑 INCREASE DATA: Collect more training data",
                "🛑 CROSS-VALIDATION: Use stricter validation strategy"
            ])
        elif severity == 'high':
            recommendations.extend([
                "🔧 INCREASE REGULARIZATION: Add L1/L2 penalties",
                "🔧 EARLY STOPPING: Implement early stopping",
                "🔧 DROPOUT: Add dropout layers if applicable",
                "🔧 ENSEMBLE: Use ensemble methods for stability"
            ])
        elif severity == 'moderate':
            recommendations.extend([
                "📈 MONITOR: Track performance closely",
                "📈 REGULARIZE: Add light regularization",
                "📈 VALIDATE: Use cross-validation",
                "📈 FEATURES: Review feature selection"
            ])
        
        # Add specific recommendations from analysis
        analysis_recommendations = analysis.get('recommendations', [])
        recommendations.extend(analysis_recommendations)
        
        return recommendations
    
    def _track_trend(self, report: OverfittingReport):
        """Track overfitting trends over time."""
        trend = OverfittingTrend(
            epoch_fold=report.fold_number or len(self.overfitting_trends),
            train_accuracy=report.train_accuracy,
            val_accuracy=report.val_accuracy,
            accuracy_gap=report.accuracy_gap,
            is_overfitting=report.is_overfitting,
            severity=report.severity,
            timestamp=report.detection_timestamp
        )
        
        self.overfitting_trends.append(trend)
        self.report_history.append(report)
    
    def _save_report(self, report: OverfittingReport):
        """Save report to disk."""
        try:
            # Convert to dictionary
            report_dict = asdict(report)
            
            # Create filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"overfitting_report_{report.model_name}_{timestamp}.json"
            filepath = self.report_directory / filename
            
            # Save JSON report
            with open(filepath, 'w') as f:
                json.dump(report_dict, f, indent=2)
            
            logger.info(f"Overfitting report saved: {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save overfitting report: {e}")
    
    def _generate_visualizations(self, report: OverfittingReport):
        """Generate visualization plots."""
        if not self.enable_visualization:
            return
        
        try:
            # Create visualization directory
            viz_dir = self.report_directory / "visualizations"
            viz_dir.mkdir(exist_ok=True)
            
            # Generate plots
            self._plot_accuracy_comparison(report, viz_dir)
            self._plot_overfitting_indicators(report, viz_dir)
            self._plot_trend_analysis(viz_dir)
            
        except Exception as e:
            logger.error(f"Failed to generate visualizations: {e}")
    
    def _plot_accuracy_comparison(self, report: OverfittingReport, viz_dir: Path):
        """Plot accuracy comparison."""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Accuracy comparison
            categories = ['Train', 'Validation']
            accuracies = [report.train_accuracy, report.val_accuracy]
            colors = ['#2E8B57', '#DC143C'] if report.is_overfitting else ['#2E8B57', '#4169E1']
            
            bars = ax1.bar(categories, accuracies, color=colors, alpha=0.7)
            ax1.set_title(f'Accuracy Comparison - {report.severity.upper()} Overfitting')
            ax1.set_ylabel('Accuracy')
            ax1.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, acc in zip(bars, accuracies):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{acc:.3f}', ha='center', va='bottom')
            
            # Accuracy gap
            ax2.bar(['Accuracy Gap'], [report.accuracy_gap], 
                   color='red' if report.is_overfitting else 'green', alpha=0.7)
            ax2.set_title('Accuracy Gap')
            ax2.set_ylabel('Gap')
            ax2.axhline(y=0.05, color='orange', linestyle='--', label='Warning (5%)')
            ax2.axhline(y=0.15, color='red', linestyle='--', label='Severe (15%)')
            ax2.legend()
            
            plt.tight_layout()
            
            # Save plot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"accuracy_comparison_{report.model_name}_{timestamp}.png"
            plt.savefig(viz_dir / filename, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Failed to create accuracy comparison plot: {e}")
    
    def _plot_overfitting_indicators(self, report: OverfittingReport, viz_dir: Path):
        """Plot overfitting indicators."""
        try:
            if not report.indicators:
                return
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Count indicators
            indicator_counts = {}
            for indicator in report.indicators:
                indicator_counts[indicator] = indicator_counts.get(indicator, 0) + 1
            
            # Create bar plot
            indicators = list(indicator_counts.keys())
            counts = list(indicator_counts.values())
            colors = ['red' if 'severe' in ind else 'orange' if 'high' in ind else 'yellow' 
                     for ind in indicators]
            
            bars = ax.bar(indicators, counts, color=colors, alpha=0.7)
            ax.set_title(f'Overfitting Indicators - {report.model_name}')
            ax.set_ylabel('Count')
            ax.set_xlabel('Indicator Type')
            plt.xticks(rotation=45, ha='right')
            
            # Add value labels
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       str(count), ha='center', va='bottom')
            
            plt.tight_layout()
            
            # Save plot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"overfitting_indicators_{report.model_name}_{timestamp}.png"
            plt.savefig(viz_dir / filename, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Failed to create overfitting indicators plot: {e}")
    
    def _plot_trend_analysis(self, viz_dir: Path):
        """Plot overfitting trends over time."""
        try:
            if len(self.overfitting_trends) < 2:
                return
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Extract data
            epochs = [t.epoch_fold for t in self.overfitting_trends]
            train_accs = [t.train_accuracy for t in self.overfitting_trends]
            val_accs = [t.val_accuracy for t in self.overfitting_trends]
            gaps = [t.accuracy_gap for t in self.overfitting_trends]
            
            # Plot accuracies
            ax1.plot(epochs, train_accs, 'b-', label='Train Accuracy', linewidth=2)
            ax1.plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2)
            ax1.set_title('Training Progress - Accuracy Trends')
            ax1.set_xlabel('Epoch/Fold')
            ax1.set_ylabel('Accuracy')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot accuracy gap
            colors = ['red' if gap > 0.15 else 'orange' if gap > 0.05 else 'green' 
                     for gap in gaps]
            ax2.bar(epochs, gaps, color=colors, alpha=0.7)
            ax2.set_title('Overfitting Trend - Accuracy Gap')
            ax2.set_xlabel('Epoch/Fold')
            ax2.set_ylabel('Accuracy Gap')
            ax2.axhline(y=0.05, color='orange', linestyle='--', label='Warning (5%)')
            ax2.axhline(y=0.15, color='red', linestyle='--', label='Severe (15%)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"overfitting_trends_{timestamp}.png"
            plt.savefig(viz_dir / filename, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Failed to create trend analysis plot: {e}")
    
    def _log_detailed_report(self, report: OverfittingReport):
        """Log detailed overfitting report."""
        logger.info("=" * 60)
        logger.info("OVERFITTING DETECTION REPORT")
        logger.info("=" * 60)
        logger.info(f"Model: {report.model_name}")
        logger.info(f"Fold: {report.fold_number}")
        logger.info(f"Timestamp: {report.detection_timestamp}")
        logger.info("")
        
        # Basic metrics
        logger.info("PERFORMANCE METRICS:")
        logger.info(f"  Train Accuracy: {report.train_accuracy:.4f}")
        logger.info(f"  Val Accuracy:   {report.val_accuracy:.4f}")
        logger.info(f"  Accuracy Gap:   {report.accuracy_gap:.4f}")
        logger.info(f"  Train F1:       {report.train_f1:.4f}")
        logger.info(f"  Val F1:         {report.val_f1:.4f}")
        logger.info(f"  F1 Gap:         {report.f1_gap:.4f}")
        logger.info("")
        
        # Overfitting status
        logger.info("OVERFITTING STATUS:")
        logger.info(f"  Detected:       {report.is_overfitting}")
        logger.info(f"  Severity:       {report.severity.upper()}")
        logger.info(f"  Confidence:     {report.confidence_level:.2f}")
        logger.info("")
        
        # Indicators
        if report.indicators:
            logger.info("OVERFITTING INDICATORS:")
            for indicator in report.indicators:
                logger.info(f"  - {indicator}")
            logger.info("")
        
        # Warnings
        if report.warnings:
            logger.info("WARNINGS:")
            for warning in report.warnings:
                logger.info(f"  {warning}")
            logger.info("")
        
        # Recommendations
        if report.recommendations:
            logger.info("RECOMMENDATIONS:")
            for rec in report.recommendations:
                logger.info(f"  {rec}")
            logger.info("")
        
        logger.info("=" * 60)
    
    def _create_error_report(self, error_message: str, model_name: str, fold_number: Optional[int]) -> OverfittingReport:
        """Create error report when analysis fails."""
        return OverfittingReport(
            train_accuracy=0.0,
            val_accuracy=0.0,
            accuracy_gap=0.0,
            train_f1=0.0,
            val_f1=0.0,
            f1_gap=0.0,
            is_overfitting=False,
            severity='none',
            confidence_level=0.0,
            indicators=[],
            warnings=[f"❌ Analysis failed: {error_message}"],
            recommendations=["Fix analysis error and retry"],
            model_name=model_name,
            fold_number=fold_number
        )
    
    def get_summary_report(self) -> Dict[str, Any]:
        """Get summary of all overfitting reports."""
        if not self.report_history:
            return {'message': 'No reports available'}
        
        # Calculate summary statistics
        total_reports = len(self.report_history)
        overfitting_detected = sum(1 for r in self.report_history if r.is_overfitting)
        severity_counts = {}
        
        for report in self.report_history:
            severity = report.severity
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Calculate average metrics
        avg_train_acc = np.mean([r.train_accuracy for r in self.report_history])
        avg_val_acc = np.mean([r.val_accuracy for r in self.report_history])
        avg_gap = np.mean([r.accuracy_gap for r in self.report_history])
        
        return {
            'total_reports': total_reports,
            'overfitting_detected': overfitting_detected,
            'overfitting_rate': overfitting_detected / total_reports,
            'severity_distribution': severity_counts,
            'average_metrics': {
                'train_accuracy': avg_train_acc,
                'val_accuracy': avg_val_acc,
                'accuracy_gap': avg_gap
            },
            'trend_analysis': {
                'total_trends': len(self.overfitting_trends),
                'latest_severity': self.overfitting_trends[-1].severity if self.overfitting_trends else 'none'
            }
        }

# Global reporter instance
DEFAULT_OVERFITTING_REPORTER = OverfittingReporter()

def get_overfitting_reporter() -> OverfittingReporter:
    """Get the default overfitting reporter."""
    return DEFAULT_OVERFITTING_REPORTER

def create_overfitting_reporter(**kwargs) -> OverfittingReporter:
    """Create a custom overfitting reporter."""
    return OverfittingReporter(**kwargs)