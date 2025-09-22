"""
Enhanced Overfitting Detection for ML Common

Universal overfitting detection and reporting system that can be used across
all ML models in the ml_common framework.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, log_loss, roc_auc_score

logger = logging.getLogger(__name__)

@dataclass
class OverfittingConfig:
    """Configuration for overfitting detection across all ML models."""
    
    # Detection thresholds
    accuracy_gap_threshold: float = 0.05  # 5% gap triggers warning
    severe_accuracy_gap_threshold: float = 0.15  # 15% gap triggers early stopping
    f1_gap_threshold: float = 0.03  # 3% F1 gap triggers warning
    severe_f1_gap_threshold: float = 0.10  # 10% F1 gap triggers early stopping
    
    # Confidence-based detection
    confidence_gap_threshold: float = 0.1  # 10% confidence gap
    overconfident_ratio_threshold: float = 0.3  # 30% overconfident predictions
    
    # Feature-based detection
    feature_concentration_threshold: float = 0.8  # 80% of importance in top features
    correlation_threshold: float = 0.95  # High correlation indicates overfitting
    
    # Cross-validation detection
    cv_variance_threshold: float = 0.05  # 5% CV variance threshold
    cv_test_gap_threshold: float = 0.08  # 8% gap between CV and test
    
    # Early stopping
    enable_early_stopping: bool = True
    patience: int = 5
    min_delta: float = 0.001
    monitor_metric: str = 'validation_loss'
    mode: str = 'min'  # 'min' for loss, 'max' for accuracy
    
    # Reporting
    save_reports: bool = True
    report_directory: str = "reports/overfitting"
    enable_visualization: bool = True
    detailed_logging: bool = True

@dataclass
class OverfittingReport:
    """Universal overfitting detection report for any ML model."""
    
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
    
    # Model metadata
    model_name: str = "unknown"
    model_type: str = "unknown"
    fold_number: Optional[int] = None
    detection_timestamp: str = None
    
    def __post_init__(self):
        """Initialize timestamp if not provided."""
        if self.detection_timestamp is None:
            self.detection_timestamp = datetime.now().isoformat()

class UniversalOverfittingDetector:
    """Universal overfitting detector for all ML models."""
    
    def __init__(self, config: Optional[OverfittingConfig] = None):
        """
        Initialize universal overfitting detector.
        
        Args:
            config: Overfitting detection configuration
        """
        self.config = config or OverfittingConfig()
        self.detection_history = []
        self.report_history = []
        
        # Create report directory
        if self.config.save_reports:
            Path(self.config.report_directory).mkdir(parents=True, exist_ok=True)
    
    def detect_overfitting(self, 
                          train_predictions: np.ndarray,
                          val_predictions: np.ndarray,
                          train_labels: np.ndarray,
                          val_labels: np.ndarray,
                          train_probabilities: Optional[np.ndarray] = None,
                          val_probabilities: Optional[np.ndarray] = None,
                          feature_importance: Optional[np.ndarray] = None,
                          model_name: str = "unknown",
                          model_type: str = "unknown",
                          fold_number: Optional[int] = None) -> OverfittingReport:
        """
        Detect overfitting for any ML model.
        
        Args:
            train_predictions: Training predictions
            val_predictions: Validation predictions
            train_labels: Training labels
            val_labels: Validation labels
            train_probabilities: Training probabilities (optional)
            val_probabilities: Validation probabilities (optional)
            feature_importance: Feature importance scores (optional)
            model_name: Name of the model
            model_type: Type of model (e.g., 'random_forest', 'neural_network')
            fold_number: Fold number for cross-validation
            
        Returns:
            OverfittingReport: Comprehensive overfitting analysis
        """
        try:
            # Calculate basic metrics
            train_acc = accuracy_score(train_labels, train_predictions)
            val_acc = accuracy_score(val_labels, val_predictions)
            accuracy_gap = train_acc - val_acc
            
            train_f1 = f1_score(train_labels, train_predictions, average='weighted')
            val_f1 = f1_score(val_labels, val_predictions, average='weighted')
            f1_gap = train_f1 - val_f1
            
            # Determine severity
            severity = self._determine_severity(accuracy_gap, f1_gap)
            confidence_level = self._calculate_confidence_level(accuracy_gap, f1_gap)
            
            # Generate indicators and analysis
            indicators = self._generate_indicators(accuracy_gap, f1_gap, train_probabilities, val_probabilities, feature_importance)
            warnings = self._generate_warnings(severity, indicators)
            recommendations = self._generate_recommendations(severity, indicators)
            
            # Advanced metrics
            train_conf, val_conf, conf_gap = self._calculate_confidence_metrics(train_probabilities, val_probabilities)
            overconfident_ratio = self._calculate_overconfident_ratio(val_probabilities)
            feature_concentration = self._calculate_feature_concentration(feature_importance)
            
            # Create comprehensive report
            report = OverfittingReport(
                train_accuracy=train_acc,
                val_accuracy=val_acc,
                accuracy_gap=accuracy_gap,
                train_f1=train_f1,
                val_f1=val_f1,
                f1_gap=f1_gap,
                is_overfitting=severity != 'none',
                severity=severity,
                confidence_level=confidence_level,
                indicators=indicators,
                warnings=warnings,
                recommendations=recommendations,
                train_confidence=train_conf,
                val_confidence=val_conf,
                confidence_gap=conf_gap,
                overconfident_ratio=overconfident_ratio,
                feature_concentration=feature_concentration,
                model_name=model_name,
                model_type=model_type,
                fold_number=fold_number
            )
            
            # Track history
            self._track_detection(report)
            
            # Save report
            if self.config.save_reports:
                self._save_report(report)
            
            # Generate visualizations
            if self.config.enable_visualization:
                self._generate_visualizations(report)
            
            # Log detailed information
            if self.config.detailed_logging:
                self._log_detailed_report(report)
            
            return report
            
        except Exception as e:
            logger.error(f"Overfitting detection failed: {e}")
            return self._create_error_report(str(e), model_name, model_type, fold_number)
    
    def _determine_severity(self, accuracy_gap: float, f1_gap: float) -> str:
        """Determine overfitting severity level."""
        # Check accuracy gap
        if accuracy_gap >= self.config.severe_accuracy_gap_threshold:
            return 'severe'
        elif accuracy_gap >= self.config.accuracy_gap_threshold:
            return 'moderate'
        
        # Check F1 gap
        if f1_gap >= self.config.severe_f1_gap_threshold:
            return 'severe'
        elif f1_gap >= self.config.f1_gap_threshold:
            return 'moderate'
        
        return 'none'
    
    def _calculate_confidence_level(self, accuracy_gap: float, f1_gap: float) -> float:
        """Calculate confidence level for overfitting detection."""
        confidence = 0.5  # Base confidence
        
        # Increase confidence based on gap size
        if accuracy_gap > 0:
            confidence += min(accuracy_gap * 2, 0.3)  # Up to 0.3 for accuracy gap
        if f1_gap > 0:
            confidence += min(f1_gap * 3, 0.2)  # Up to 0.2 for F1 gap
        
        return min(confidence, 1.0)
    
    def _generate_indicators(self, 
                           accuracy_gap: float, 
                           f1_gap: float,
                           train_probabilities: Optional[np.ndarray],
                           val_probabilities: Optional[np.ndarray],
                           feature_importance: Optional[np.ndarray]) -> List[str]:
        """Generate overfitting indicators."""
        indicators = []
        
        # Accuracy gap indicators
        if accuracy_gap >= self.config.severe_accuracy_gap_threshold:
            indicators.append('severe_accuracy_gap')
        elif accuracy_gap >= self.config.accuracy_gap_threshold:
            indicators.append('accuracy_gap')
        
        # F1 gap indicators
        if f1_gap >= self.config.severe_f1_gap_threshold:
            indicators.append('severe_f1_gap')
        elif f1_gap >= self.config.f1_gap_threshold:
            indicators.append('f1_gap')
        
        # Confidence-based indicators
        if train_probabilities is not None and val_probabilities is not None:
            train_conf = np.mean(np.max(train_probabilities, axis=1))
            val_conf = np.mean(np.max(val_probabilities, axis=1))
            conf_gap = train_conf - val_conf
            
            if conf_gap > self.config.confidence_gap_threshold:
                indicators.append('confidence_gap')
            
            # Overconfident predictions
            overconfident_threshold = 0.9
            val_overconfident = np.mean(np.max(val_probabilities, axis=1) > overconfident_threshold)
            if val_overconfident > self.config.overconfident_ratio_threshold:
                indicators.append('overconfident')
        
        # Feature concentration
        if feature_importance is not None:
            concentration = self._calculate_feature_concentration(feature_importance)
            if concentration > self.config.feature_concentration_threshold:
                indicators.append('feature_concentration')
        
        return indicators
    
    def _generate_warnings(self, severity: str, indicators: List[str]) -> List[str]:
        """Generate actionable warnings."""
        warnings = []
        
        if severity == 'severe':
            warnings.extend([
                "🚨 CRITICAL: Severe overfitting detected - immediate action required",
                "🚨 Model is likely to fail in production",
                "🚨 Consider stopping training and redesigning approach"
            ])
        elif severity == 'moderate':
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
        
        return warnings
    
    def _generate_recommendations(self, severity: str, indicators: List[str]) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
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
        
        return recommendations
    
    def _calculate_confidence_metrics(self, 
                                    train_probabilities: Optional[np.ndarray],
                                    val_probabilities: Optional[np.ndarray]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Calculate confidence-based metrics."""
        if train_probabilities is None or val_probabilities is None:
            return None, None, None
        
        train_conf = np.mean(np.max(train_probabilities, axis=1))
        val_conf = np.mean(np.max(val_probabilities, axis=1))
        conf_gap = train_conf - val_conf
        
        return float(train_conf), float(val_conf), float(conf_gap)
    
    def _calculate_overconfident_ratio(self, val_probabilities: Optional[np.ndarray]) -> Optional[float]:
        """Calculate overconfident prediction ratio."""
        if val_probabilities is None:
            return None
        
        overconfident_threshold = 0.9
        overconfident_ratio = np.mean(np.max(val_probabilities, axis=1) > overconfident_threshold)
        return float(overconfident_ratio)
    
    def _calculate_feature_concentration(self, feature_importance: Optional[np.ndarray]) -> Optional[float]:
        """Calculate feature importance concentration."""
        if feature_importance is None:
            return None
        
        sorted_importance = np.sort(feature_importance)[::-1]
        top_features_ratio = 0.1  # Top 10% of features
        n_top = max(1, int(len(sorted_importance) * top_features_ratio))
        concentration = np.sum(sorted_importance[:n_top]) / np.sum(sorted_importance)
        
        return float(concentration)
    
    def _track_detection(self, report: OverfittingReport):
        """Track detection history."""
        self.detection_history.append(report)
        self.report_history.append(report)
    
    def _save_report(self, report: OverfittingReport):
        """Save report to disk."""
        try:
            report_dict = asdict(report)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"overfitting_report_{report.model_name}_{timestamp}.json"
            filepath = Path(self.config.report_directory) / filename
            
            with open(filepath, 'w') as f:
                json.dump(report_dict, f, indent=2)
            
            logger.info(f"Overfitting report saved: {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save overfitting report: {e}")
    
    def _generate_visualizations(self, report: OverfittingReport):
        """Generate visualization plots."""
        try:
            viz_dir = Path(self.config.report_directory) / "visualizations"
            viz_dir.mkdir(exist_ok=True)
            
            # Generate plots
            self._plot_accuracy_comparison(report, viz_dir)
            self._plot_overfitting_indicators(report, viz_dir)
            
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
            
            # Add value labels
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
    
    def _log_detailed_report(self, report: OverfittingReport):
        """Log detailed overfitting report."""
        logger.info("=" * 60)
        logger.info("OVERFITTING DETECTION REPORT")
        logger.info("=" * 60)
        logger.info(f"Model: {report.model_name} ({report.model_type})")
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
    
    def _create_error_report(self, error_message: str, model_name: str, model_type: str, fold_number: Optional[int]) -> OverfittingReport:
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
            model_type=model_type,
            fold_number=fold_number
        )
    
    def get_summary_report(self) -> Dict[str, Any]:
        """Get summary of all overfitting detections."""
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
            }
        }

# Global detector instance
DEFAULT_OVERFITTING_DETECTOR = UniversalOverfittingDetector()

def get_overfitting_detector(config: Optional[OverfittingConfig] = None) -> UniversalOverfittingDetector:
    """Get overfitting detector instance."""
    if config is None:
        return DEFAULT_OVERFITTING_DETECTOR
    return UniversalOverfittingDetector(config)

def detect_overfitting_for_model(model, 
                                X_train: np.ndarray, 
                                X_val: np.ndarray,
                                y_train: np.ndarray, 
                                y_val: np.ndarray,
                                model_name: str = "unknown",
                                model_type: str = "unknown",
                                fold_number: Optional[int] = None,
                                config: Optional[OverfittingConfig] = None) -> OverfittingReport:
    """
    Convenience function to detect overfitting for any ML model.
    
    Args:
        model: Trained ML model
        X_train: Training features
        X_val: Validation features
        y_train: Training labels
        y_val: Validation labels
        model_name: Name of the model
        model_type: Type of model
        fold_number: Fold number for cross-validation
        config: Overfitting detection configuration
        
    Returns:
        OverfittingReport: Comprehensive overfitting analysis
    """
    detector = get_overfitting_detector(config)
    
    # Get predictions
    train_predictions = model.predict(X_train)
    val_predictions = model.predict(X_val)
    
    # Get probabilities if available
    train_probabilities = None
    val_probabilities = None
    if hasattr(model, 'predict_proba'):
        try:
            train_probabilities = model.predict_proba(X_train)
            val_probabilities = model.predict_proba(X_val)
        except:
            pass
    
    # Get feature importance if available
    feature_importance = None
    if hasattr(model, 'feature_importances_'):
        feature_importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        feature_importance = np.abs(model.coef_).flatten()
    
    # Detect overfitting
    return detector.detect_overfitting(
        train_predictions=train_predictions,
        val_predictions=val_predictions,
        train_labels=y_train,
        val_labels=y_val,
        train_probabilities=train_probabilities,
        val_probabilities=val_probabilities,
        feature_importance=feature_importance,
        model_name=model_name,
        model_type=model_type,
        fold_number=fold_number
    )