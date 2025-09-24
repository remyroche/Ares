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

        # Time series specific indicators
        indicators.extend(self._detect_time_series_overfitting(
            train_predictions, val_predictions, train_labels, val_labels
        ))

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
        else:
            warnings.append("✅ No significant overfitting detected")

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

    def _detect_time_series_overfitting(self,
                                      train_predictions: np.ndarray,
                                      val_predictions: np.ndarray,
                                      train_labels: np.ndarray,
                                      val_labels: np.ndarray) -> List[str]:
        """Detect overfitting specific to time series models."""
        indicators = []

        try:
            # 1. Check for regime overfitting (model memorizes specific market conditions)
            # Calculate prediction variance within similar label ranges
            if len(train_labels) > 10 and len(val_labels) > 10:
                # Group by label ranges and check if predictions are too consistent
                train_label_ranges = pd.cut(train_labels, bins=5, duplicates='drop')
                val_label_ranges = pd.cut(val_labels, bins=5, duplicates='drop')

                for range_name in train_label_ranges.unique():
                    if pd.isna(range_name):
                        continue

                    train_mask = train_label_ranges == range_name
                    val_mask = val_label_ranges == range_name

                    if np.sum(train_mask) > 5 and np.sum(val_mask) > 5:
                        train_preds_range = train_predictions[train_mask]
                        val_preds_range = val_predictions[val_mask]

                        # If model is overfitting, it might show very low variance in training predictions
                        train_variance = np.var(train_preds_range)
                        val_variance = np.var(val_preds_range)

                        if train_variance < 0.01 and val_variance > 0.1:  # Very low train variance, higher val variance
                            indicators.append('regime_memorization')

            # 2. Check for temporal overfitting (model performs well only on recent data)
            # This would be detected by the temporal validation system, but we can add additional checks
            if len(val_predictions) > 20:
                # Check if validation predictions are becoming more erratic over time
                # This might indicate the model is overfitting to recent patterns
                recent_val_preds = val_predictions[-10:]
                early_val_preds = val_predictions[:10]

                if len(recent_val_preds) >= 5 and len(early_val_preds) >= 5:
                    recent_variance = np.var(recent_val_preds)
                    early_variance = np.var(early_val_preds)

                    # If recent predictions have much higher variance, model may be unstable
                    variance_ratio = recent_variance / early_variance if early_variance > 0 else float('inf')
                    if variance_ratio > 3.0:  # 3x higher variance in recent predictions
                        indicators.append('temporal_instability')

            # 3. Check for label distribution shift overfitting
            # If train and validation have different label distributions, model may overfit to train distribution
            if len(train_labels) > 10 and len(val_labels) > 10:
                train_unique, train_counts = np.unique(train_labels, return_counts=True)
                val_unique, val_counts = np.unique(val_labels, return_counts=True)

                # Calculate distribution similarity (simple overlap coefficient)
                train_dist = dict(zip(train_unique, train_counts / len(train_labels)))
                val_dist = dict(zip(val_unique, val_counts / len(val_labels)))

                all_labels = set(train_unique) | set(val_unique)
                overlap = sum(min(train_dist.get(label, 0), val_dist.get(label, 0)) for label in all_labels)

                if overlap < 0.7:  # Less than 70% distribution overlap
                    indicators.append('distribution_shift')

        except Exception as e:
            logger.warning(f"Time series overfitting detection failed: {e}")

        return indicators

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

def detect_overfitting_with_learning_curves(model: Any,
                                          X_train: np.ndarray,
                                          X_val: np.ndarray,
                                          y_train: np.ndarray,
                                          y_val: np.ndarray,
                                          X_test: Optional[np.ndarray] = None,
                                          y_test: Optional[np.ndarray] = None,
                                          model_name: str = "unknown",
                                          model_type: str = "unknown",
                                          fold_number: Optional[int] = None,
                                          config: Optional[OverfittingConfig] = None) -> OverfittingReport:
    """
    Enhanced overfitting detection with learning curve analysis.
    
    This function integrates learning curve analysis into the existing overfitting detection
    to provide more comprehensive overfitting assessment.
    """
    try:
        # Import learning curve analyzer
        from ..evaluation.enhanced_learning_curve_analysis import EnhancedLearningCurveAnalyzer
        
        # Initialize detector
        detector = get_overfitting_detector(config)
        
        # Get basic predictions
        train_predictions = model.predict(X_train)
        val_predictions = model.predict(X_val)
        
        # Get probabilities if available
        train_probabilities = None
        val_probabilities = None
        if hasattr(model, 'predict_proba'):
            try:
                train_probabilities = model.predict_proba(X_train)
                val_probabilities = model.predict_proba(X_val)
            except Exception as e:
                logger.warning(f"Could not get probabilities: {e}")
        
        # Get feature importance if available
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            feature_importance = np.abs(model.coef_).flatten()
        
        # Perform basic overfitting detection
        basic_report = detector.detect_overfitting(
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
        
        # Perform learning curve analysis
        try:
            learning_curve_analyzer = EnhancedLearningCurveAnalyzer()
            
            # Combine train and val for learning curve analysis
            X_combined = np.vstack([X_train, X_val])
            y_combined = np.concatenate([y_train, y_val])
            
            # Determine scoring metric
            is_classification = len(np.unique(y_combined)) <= 10
            scoring = 'accuracy' if is_classification else 'r2'
            
            # Perform learning curve analysis
            learning_curve_result = learning_curve_analyzer.analyze_learning_curve(
                model=model,
                X_train=X_combined,
                y_train=y_combined,
                X_test=X_test if X_test is not None else X_val,
                y_test=y_test if y_test is not None else y_val,
                scoring=scoring
            )
            
            # Add learning curve indicators to the report
            if learning_curve_result.overfitting_risk in ["high", "severe"]:
                basic_report.indicators.append("learning_curve_overfitting")
                basic_report.warnings.append("Learning curve analysis indicates overfitting risk")
            
            if learning_curve_result.convergence_stability == "poor":
                basic_report.recommendations.append("Poor convergence stability - consider learning rate adjustment")
            
            if learning_curve_result.training_efficiency == "low":
                basic_report.recommendations.append("Low training efficiency - consider model simplification")
            
            # Update severity based on learning curve analysis
            if learning_curve_result.overfitting_risk == "severe":
                if basic_report.severity == "moderate":
                    basic_report.severity = "high"
                elif basic_report.severity == "none":
                    basic_report.severity = "moderate"
            
            logger.info(f"✅ Learning curve analysis integrated for {model_name}")
            
        except Exception as e:
            logger.warning(f"Learning curve analysis failed: {e}")
            basic_report.warnings.append("Learning curve analysis unavailable")
        
        return basic_report
        
    except Exception as e:
        logger.error(f"Enhanced overfitting detection with learning curves failed: {e}")
        # Fallback to basic detection
        detector = get_overfitting_detector(config)
        return detector.detect_overfitting(
            train_predictions=model.predict(X_train),
            val_predictions=model.predict(X_val),
            train_labels=y_train,
            val_labels=y_val,
            model_name=model_name,
            model_type=model_type,
            fold_number=fold_number
        )


class ModelEnhancementDetector:
    """Detect models that could benefit from parameter tuning and optimization."""

    def __init__(self):
        """Initialize model enhancement detector."""
        self.logger = logging.getLogger('ModelEnhancementDetector')

    def detect_enhancement_opportunities(self,
                                       model,
                                       X_train: np.ndarray,
                                       X_val: np.ndarray,
                                       y_train: np.ndarray,
                                       y_val: np.ndarray,
                                       model_name: str = "unknown",
                                       model_type: str = "unknown") -> Dict[str, Any]:
        """
        Detect opportunities for model enhancement and parameter tuning.

        Args:
            model: Trained ML model
            X_train: Training features
            X_val: Validation features
            y_train: Training labels
            y_val: Validation labels
            model_name: Name of the model
            model_type: Type of model

        Returns:
            Dict: Enhancement opportunities and recommendations
        """
        opportunities = {
            'model_name': model_name,
            'model_type': model_type,
            'enhancement_opportunities': [],
            'parameter_tuning_suggestions': [],
            'performance_issues': [],
            'data_issues': [],
            'confidence_level': 0.0,
            'priority': 'low',  # low, medium, high, critical
            'estimated_improvement_potential': 0.0
        }

        try:
            # 1. Check if model is underfitting (too simple)
            underfitting_score = self._check_underfitting(model, X_train, X_val, y_train, y_val)
            if underfitting_score > 0.7:
                opportunities['enhancement_opportunities'].append('model_complexity_increase')
                opportunities['parameter_tuning_suggestions'].append({
                    'warning': 'Model appears to be underfitting - may be too simple for the data',
                    'reason': 'High training and validation errors suggest insufficient model capacity',
                    'improvement_potential': 'Investigate increasing model complexity'
                })

            # 2. Check for parameter sensitivity
            sensitivity_analysis = self._analyze_parameter_sensitivity(model, X_train, y_train)
            if sensitivity_analysis['high_sensitivity']:
                opportunities['enhancement_opportunities'].append('parameter_tuning_needed')
                opportunities['parameter_tuning_suggestions'].append({
                    'warning': 'Model shows parameter sensitivity - room for improvement through tuning',
                    'reason': f'{model_type} models typically benefit from parameter optimization',
                    'improvement_potential': 'Consider hyperparameter optimization for better performance'
                })

            # 3. Check for feature importance imbalance
            importance_analysis = self._analyze_feature_importance(model, X_train)
            if importance_analysis['imbalanced']:
                opportunities['enhancement_opportunities'].append('feature_engineering')
                opportunities['parameter_tuning_suggestions'].append({
                    'warning': 'Feature importance is heavily imbalanced',
                    'reason': f'{importance_analysis["concentration_ratio"]:.2%} of importance in top 10% of features',
                    'improvement_potential': 'Review feature selection and consider feature engineering'
                })

            # 4. Check for overfitting potential
            overfitting_potential = self._check_overfitting_potential(model, X_train, X_val, y_train, y_val)
            if overfitting_potential > 0.6:
                opportunities['enhancement_opportunities'].append('regularization_increase')
                opportunities['parameter_tuning_suggestions'].append({
                    'warning': 'Model shows signs of potential overfitting',
                    'reason': f'Overfitting potential score: {overfitting_potential:.2f}',
                    'improvement_potential': 'Consider increasing regularization to prevent overfitting'
                })

            # 5. Check for optimization opportunities
            optimization_opportunities = self._check_optimization_opportunities(model, model_type)
            opportunities['enhancement_opportunities'].extend(optimization_opportunities)

            # Add warnings for optimization opportunities instead of specific recommendations
            for opportunity in optimization_opportunities:
                opportunities['parameter_tuning_suggestions'].append({
                    'warning': f'Model-specific optimization opportunity detected: {opportunity.replace("_", " ")}',
                    'reason': f'{model_type} models can benefit from {opportunity.replace("_", " ")}',
                    'improvement_potential': f'Consider model-specific optimizations for {opportunity.replace("_", " ")}'
                })

            # Calculate overall enhancement potential
            opportunities['confidence_level'] = self._calculate_enhancement_confidence(opportunities)
            opportunities['priority'] = self._determine_priority(opportunities)
            opportunities['estimated_improvement_potential'] = self._estimate_improvement_potential(opportunities)

            # Generate detailed recommendations
            opportunities['detailed_recommendations'] = self._generate_detailed_recommendations(opportunities)

        except Exception as e:
            self.logger.error(f"Model enhancement detection failed: {e}")
            opportunities['error'] = str(e)

        return opportunities

    def _check_underfitting(self, model, X_train, X_val, y_train, y_val) -> float:
        """Check if model is underfitting (score from 0.0 to 1.0)."""
        try:
            # Get predictions
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)

            # Calculate metrics
            train_mse = np.mean((y_train - train_pred) ** 2)
            val_mse = np.mean((y_val - val_pred) ** 2)

            # Normalize by target variance
            target_var = np.var(y_train)
            if target_var == 0:
                return 0.0

            train_normalized_error = train_mse / target_var
            val_normalized_error = val_mse / target_var

            # Underfitting score: higher when both train and val errors are high
            underfitting_score = min(1.0, (train_normalized_error + val_normalized_error) / 2.0)

            return underfitting_score

        except Exception as e:
            self.logger.warning(f"Underfitting check failed: {e}")
            return 0.0

    def _analyze_parameter_sensitivity(self, model, X_train, y_train) -> Dict[str, Any]:
        """Analyze parameter sensitivity to determine tuning needs."""
        analysis = {
            'high_sensitivity': False,
            'model_type': model.__class__.__name__.lower()
        }

        try:
            # Simple parameter sensitivity check based on model type
            model_type = model.__class__.__name__.lower()

            # All these model types typically benefit from parameter tuning
            if ('xgb' in model_type or 'xgboost' in model_type or
                'lgbm' in model_type or 'lightgbm' in model_type or
                'catboost' in model_type or
                'randomforest' in model_type or 'neural' in model_type or
                'torch' in model_type or 'keras' in model_type or
                'deepscaler' in model_type or 'mamba' in model_type or
                'linear' in model_type or 'ridge' in model_type or
                'lasso' in model_type or 'elasticnet' in model_type):

                analysis['high_sensitivity'] = True

        except Exception as e:
            self.logger.warning(f"Parameter sensitivity analysis failed: {e}")

        return analysis

    def _analyze_feature_importance(self, model, X_train) -> Dict[str, Any]:
        """Analyze feature importance distribution."""
        analysis = {
            'imbalanced': False,
            'concentration_ratio': 0.0,
            'top_features': []
        }

        try:
            # Get feature importance
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance = np.abs(model.coef_).flatten()
            else:
                return analysis  # No feature importance available

            # Calculate concentration
            sorted_importance = np.sort(importance)[::-1]
            top_10_percent = sorted_importance[:max(1, len(sorted_importance) // 10)]
            analysis['concentration_ratio'] = np.sum(top_10_percent) / np.sum(sorted_importance)

            if analysis['concentration_ratio'] > 0.8:  # 80% of importance in top 10% features
                analysis['imbalanced'] = True

            # Get top feature indices
            top_indices = np.argsort(importance)[::-1][:10]
            analysis['top_features'] = top_indices.tolist()

        except Exception as e:
            self.logger.warning(f"Feature importance analysis failed: {e}")

        return analysis

    def _check_overfitting_potential(self, model, X_train, X_val, y_train, y_val) -> float:
        """Check potential for overfitting (score from 0.0 to 1.0)."""
        try:
            # Get predictions
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)

            # Calculate train vs validation performance gap
            train_mse = np.mean((y_train - train_pred) ** 2)
            val_mse = np.mean((y_val - val_pred) ** 2)

            # Calculate overfitting potential
            if train_mse == 0:
                return 1.0  # Perfect training fit = high overfitting risk

            performance_ratio = val_mse / train_mse
            overfitting_potential = min(1.0, max(0.0, 1.0 - 1.0 / (1.0 + performance_ratio)))

            return overfitting_potential

        except Exception as e:
            self.logger.warning(f"Overfitting potential check failed: {e}")
            return 0.0

    def _check_optimization_opportunities(self, model, model_type: str) -> List[str]:
        """Check for model-specific optimization opportunities."""
        opportunities = []

        try:
            # Model-specific optimization opportunities
            if ('neural' in model_type.lower() or 'torch' in model_type.lower() or
                'keras' in model_type.lower() or 'deepscaler' in model_type.lower() or
                'mamba' in model_type.lower()):
                opportunities.extend([
                    'learning_rate_scheduling',
                    'batch_normalization',
                    'gradient_clipping',
                    'early_stopping_optimization',
                    'architecture_optimization',
                    'attention_mechanism_tuning'
                ])

                # Add specific optimizations for advanced architectures
                if 'deepscaler' in model_type.lower():
                    opportunities.extend([
                        'scaling_factor_optimization',
                        'time_series_preprocessing_tuning',
                        'multi_scale_feature_integration'
                    ])
                elif 'mamba' in model_type.lower():
                    opportunities.extend([
                        'state_space_optimization',
                        'selective_scan_tuning',
                        'hardware_aware_optimization'
                    ])

            elif 'xgb' in model_type.lower() or 'xgboost' in model_type.lower() or 'lgbm' in model_type.lower() or 'lightgbm' in model_type.lower() or 'catboost' in model_type.lower():
                opportunities.extend([
                    'tree_structure_optimization',
                    'feature_interaction_constraints',
                    'monotone_constraints',
                    'categorical_feature_handling',
                    'boosting_round_optimization'
                ])

            elif 'linear' in model_type.lower() or 'ridge' in model_type.lower() or 'lasso' in model_type.lower() or 'elasticnet' in model_type.lower():
                opportunities.extend([
                    'regularization_optimization',
                    'feature_scaling_check',
                    'multicollinearity_analysis'
                ])

            elif 'randomforest' in model_type.lower() or 'extratrees' in model_type.lower():
                opportunities.extend([
                    'ensemble_diversity_optimization',
                    'feature_sampling_optimization',
                    'bootstrap_optimization'
                ])

            elif 'svm' in model_type.lower() or 'svc' in model_type.lower():
                opportunities.extend([
                    'kernel_optimization',
                    'gamma_parameter_tuning',
                    'class_weight_optimization'
                ])

            elif 'knn' in model_type.lower():
                opportunities.extend([
                    'distance_metric_optimization',
                    'neighbor_count_optimization',
                    'weight_function_optimization'
                ])

            elif 'bayesian' in model_type.lower() or 'naive' in model_type.lower():
                opportunities.extend([
                    'prior_optimization',
                    'smoothing_parameter_tuning',
                    'feature_independence_assumptions'
                ])

            # Default opportunities for unknown model types
            else:
                opportunities.extend([
                    'general_hyperparameter_tuning',
                    'ensemble_methods',
                    'cross_validation_optimization'
                ])

        except Exception as e:
            self.logger.warning(f"Optimization opportunities check failed: {e}")

        return opportunities


    def _calculate_enhancement_confidence(self, opportunities: Dict[str, Any]) -> float:
        """Calculate confidence level for enhancement recommendations."""
        confidence_factors = []

        # Base confidence
        base_confidence = 0.5

        # Factor based on number of opportunities found
        n_opportunities = len(opportunities['enhancement_opportunities'])
        opportunity_factor = min(0.3, n_opportunities * 0.1)

        # Factor based on parameter tuning suggestions
        n_suggestions = len(opportunities['parameter_tuning_suggestions'])
        suggestion_factor = min(0.2, n_suggestions * 0.05)

        total_confidence = base_confidence + opportunity_factor + suggestion_factor

        return min(1.0, total_confidence)

    def _determine_priority(self, opportunities: Dict[str, Any]) -> str:
        """Determine priority level for enhancement."""
        n_opportunities = len(opportunities['enhancement_opportunities'])
        confidence = opportunities['confidence_level']

        if n_opportunities >= 3 and confidence > 0.8:
            return 'critical'
        elif n_opportunities >= 2 and confidence > 0.6:
            return 'high'
        elif n_opportunities >= 1 and confidence > 0.4:
            return 'medium'
        else:
            return 'low'

    def _estimate_improvement_potential(self, opportunities: Dict[str, Any]) -> float:
        """Estimate potential improvement from enhancements."""
        improvement_factors = {
            'model_complexity_increase': 0.15,
            'parameter_tuning_needed': 0.20,
            'feature_engineering': 0.10,
            'regularization_increase': 0.05,
            'learning_rate_scheduling': 0.08,
            'tree_structure_optimization': 0.12,
            'categorical_feature_handling': 0.10,
            'boosting_round_optimization': 0.09,
            'architecture_optimization': 0.14,
            'attention_mechanism_tuning': 0.11,
            'scaling_factor_optimization': 0.13,
            'time_series_preprocessing_tuning': 0.12,
            'multi_scale_feature_integration': 0.11,
            'state_space_optimization': 0.15,
            'selective_scan_tuning': 0.13,
            'hardware_aware_optimization': 0.10
        }

        total_potential = 0.0
        for opportunity in opportunities['enhancement_opportunities']:
            if opportunity in improvement_factors:
                total_potential += improvement_factors[opportunity]

        return min(0.5, total_potential)  # Cap at 50% potential improvement

    def _generate_detailed_recommendations(self, opportunities: Dict[str, Any]) -> List[str]:
        """Generate detailed recommendations based on analysis."""
        recommendations = []

        if opportunities['priority'] == 'critical':
            recommendations.append("🚨 CRITICAL: Immediate model enhancement required")
        elif opportunities['priority'] == 'high':
            recommendations.append("⚠️ HIGH: Strong enhancement opportunities identified")
        elif opportunities['priority'] == 'medium':
            recommendations.append("📊 MEDIUM: Moderate enhancement opportunities available")
        else:
            recommendations.append("✅ LOW: Minimal enhancement opportunities found")

        # Add specific recommendations based on opportunities
        for opportunity in opportunities['enhancement_opportunities']:
            if opportunity == 'model_complexity_increase':
                recommendations.append("🔧 Consider increasing model complexity (deeper trees, more estimators, additional layers)")
            elif opportunity == 'parameter_tuning_needed':
                recommendations.append("🔧 Perform comprehensive hyperparameter optimization")
            elif opportunity == 'feature_engineering':
                recommendations.append("🔧 Review feature selection and consider feature engineering")
            elif opportunity == 'regularization_increase':
                recommendations.append("🔧 Increase regularization to prevent overfitting")

        return recommendations


class UniversalMLValidationOrchestrator:
    """Unified validation system that orchestrates all ML validation components."""

    def __init__(self):
        """Initialize the validation orchestrator."""
        self.logger = logging.getLogger('UniversalMLValidationOrchestrator')
        self.overfitting_detector = UniversalOverfittingDetector()
        self.enhancement_detector = ModelEnhancementDetector()

        # Import temporal validation components
        try:
            from ..validation.universal_temporal_validation import UniversalTemporalValidator, UniversalTemporalCrossValidator
            self.temporal_validator = UniversalTemporalValidator()
            self.temporal_cv = UniversalTemporalCrossValidator()
        except ImportError:
            self.temporal_validator = None
            self.temporal_cv = None

    def comprehensive_model_validation(self,
                                     model,
                                     X_train: np.ndarray,
                                     X_val: np.ndarray,
                                     y_train: np.ndarray,
                                     y_val: np.ndarray,
                                     X_test: Optional[np.ndarray] = None,
                                     y_test: Optional[np.ndarray] = None,
                                     model_name: str = "unknown",
                                     model_type: str = "unknown",
                                     timestamps: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Perform comprehensive validation of any ML model.

        Args:
            model: Trained ML model
            X_train: Training features
            X_val: Validation features
            y_train: Training labels
            y_val: Validation labels
            X_test: Optional test features
            y_test: Optional test labels
            model_name: Name of the model
            model_type: Type of model
            timestamps: Optional timestamps for temporal validation

        Returns:
            Dict: Comprehensive validation report
        """
        self.logger.info(f"🚀 Starting comprehensive validation for {model_name} ({model_type})")

        validation_report = {
            'model_name': model_name,
            'model_type': model_type,
            'validation_timestamp': datetime.now().isoformat(),
            'validation_status': 'completed',
            'overall_score': 0.0,
            'components': {}
        }

        try:
            # 1. Overfitting Detection
            self.logger.info("🔍 Running overfitting detection...")
            overfitting_report = self.overfitting_detector.detect_overfitting(
                train_predictions=model.predict(X_train),
                val_predictions=model.predict(X_val),
                train_labels=y_train,
                val_labels=y_val,
                model_name=model_name,
                model_type=model_type
            )
            validation_report['components']['overfitting'] = overfitting_report
            self.logger.info(f"✅ Overfitting detection completed: {'DETECTED' if overfitting_report.is_overfitting else 'NOT DETECTED'}")

            # 2. Model Enhancement Detection
            self.logger.info("🔍 Running model enhancement detection...")
            enhancement_report = self.enhancement_detector.detect_enhancement_opportunities(
                model, X_train, X_val, y_train, y_val, model_name, model_type
            )
            validation_report['components']['enhancement'] = enhancement_report
            self.logger.info(f"✅ Enhancement detection completed: {enhancement_report['priority']} priority")

            # 3. Temporal Validation (if timestamps available)
            if timestamps is not None and self.temporal_validator:
                self.logger.info("🔍 Running temporal validation...")
                temporal_report = self.temporal_validator.validate_temporal_split(
                    X_train, X_val, y_train, y_val, timestamps, model_name, model_type
                )
                validation_report['components']['temporal'] = temporal_report
                self.logger.info(f"✅ Temporal validation completed: {temporal_report.temporal_order_valid}")

            # 4. Cross-Validation Analysis (if cross-validator available)
            if self.temporal_cv:
                self.logger.info("🔍 Running temporal cross-validation...")
                try:
                    cv_results = self.temporal_cv.cross_validate(
                        model, X_train, y_train, timestamps, model_name=model_name, model_type=model_type
                    )
                    validation_report['components']['cross_validation'] = cv_results
                    self.logger.info(f"✅ Cross-validation completed: {cv_results['mean_score']:.4f} mean score")
                except Exception as e:
                    self.logger.warning(f"Temporal cross-validation failed: {e}")

            # 5. Performance Analysis
            self.logger.info("🔍 Running performance analysis...")
            performance_analysis = self._analyze_model_performance(
                model, X_train, X_val, X_test, y_train, y_val, y_test
            )
            validation_report['components']['performance'] = performance_analysis
            self.logger.info("✅ Performance analysis completed")

            # 6. Data Quality Assessment
            self.logger.info("🔍 Running data quality assessment...")
            data_quality = self._assess_data_quality(X_train, X_val, y_train, y_val)
            validation_report['components']['data_quality'] = data_quality
            self.logger.info("✅ Data quality assessment completed")

            # Calculate overall validation score
            validation_report['overall_score'] = self._calculate_overall_score(validation_report)

            # Generate comprehensive per-model report
            model_specific_report = self.generate_model_specific_report(
                model_name, model_type, validation_report
            )
            validation_report['model_specific_report'] = model_specific_report

            # Generate summary and recommendations
            validation_report['summary'] = self._generate_validation_summary(validation_report)
            validation_report['recommendations'] = self._generate_recommendations(validation_report)

            # Add per-model insights to recommendations
            model_insights = model_specific_report.get('model_specific_insights', {})
            if model_insights.get('priority_actions'):
                validation_report['recommendations'].extend(model_insights['priority_actions'])

            self.logger.info(f"✅ Comprehensive validation completed with overall score: {validation_report['overall_score']:.3f}")
            self.logger.info(f"📊 Generated comprehensive per-model report for {model_name}")

        except Exception as e:
            self.logger.error(f"Comprehensive validation failed: {e}")
            validation_report['validation_status'] = 'failed'
            validation_report['error'] = str(e)

        return validation_report

    def _analyze_model_performance(self, model, X_train, X_val, X_test, y_train, y_val, y_test) -> Dict[str, Any]:
        """Analyze model performance across different datasets."""
        analysis = {
            'train_metrics': {},
            'validation_metrics': {},
            'test_metrics': {},
            'performance_stability': 0.0,
            'generalization_score': 0.0
        }

        try:
            # Calculate predictions for all datasets
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)
            test_pred = model.predict(X_test) if X_test is not None else None

            # Calculate metrics
            analysis['train_metrics'] = self._calculate_metrics(y_train, train_pred)
            analysis['validation_metrics'] = self._calculate_metrics(y_val, val_pred)

            if test_pred is not None and y_test is not None:
                analysis['test_metrics'] = self._calculate_metrics(y_test, test_pred)

                # Calculate performance stability
                train_score = analysis['train_metrics'].get('accuracy', 0)
                val_score = analysis['validation_metrics'].get('accuracy', 0)
                test_score = analysis['test_metrics'].get('accuracy', 0)

                if train_score > 0:
                    # Performance stability: lower when train >> val > test (overfitting pattern)
                    stability_score = 1.0 - abs(train_score - val_score) / max(train_score, 0.01)
                    analysis['performance_stability'] = max(0.0, stability_score)

                    # Generalization score: higher when validation ≈ test
                    if val_score > 0 and test_score > 0:
                        generalization_score = 1.0 - abs(val_score - test_score) / max(val_score, test_score)
                        analysis['generalization_score'] = max(0.0, generalization_score)

        except Exception as e:
            self.logger.warning(f"Performance analysis failed: {e}")

        return analysis

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive metrics."""
        try:

            metrics = {}

            # Classification metrics
            try:
                metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
                metrics['f1'] = float(f1_score(y_true, y_pred, average='weighted'))
                metrics['precision'] = float(precision_score(y_true, y_pred, average='weighted'))
                metrics['recall'] = float(recall_score(y_true, y_pred, average='weighted'))
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to calculate classification metrics: {e}")
                metrics['precision'] = None
                metrics['recall'] = None

            # Regression metrics
            try:
                metrics['mse'] = float(mean_squared_error(y_true, y_pred))
                metrics['rmse'] = float(np.sqrt(metrics['mse']))
                metrics['r2'] = float(r2_score(y_true, y_pred))
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to calculate regression metrics: {e}")
                metrics['mse'] = None
                metrics['rmse'] = None
                metrics['r2'] = None

            return metrics

        except Exception as e:
            return {'error': str(e)}

    def _assess_data_quality(self, X_train, X_val, y_train, y_val) -> Dict[str, Any]:
        """Assess data quality issues that might affect model performance."""
        assessment = {
            'train_data_issues': [],
            'validation_data_issues': [],
            'recommendations': []
        }

        try:
            # Check for missing values
            if np.isnan(X_train).any():
                assessment['train_data_issues'].append('missing_values')
            if np.isnan(X_val).any():
                assessment['validation_data_issues'].append('missing_values')

            # Check for infinite values
            if np.isinf(X_train).any():
                assessment['train_data_issues'].append('infinite_values')
            if np.isinf(X_val).any():
                assessment['validation_data_issues'].append('infinite_values')

            # Check for data leakage potential (features with very high correlation between train/val)
            if X_train.shape[1] > 1:
                try:
                    train_corr = np.abs(np.corrcoef(X_train.T))
                    val_corr = np.abs(np.corrcoef(X_val.T))

                    # Check for suspiciously high correlations
                    high_corr_threshold = 0.95
                    if np.any(train_corr > high_corr_threshold) or np.any(val_corr > high_corr_threshold):
                        assessment['train_data_issues'].append('high_feature_correlation')
                        assessment['validation_data_issues'].append('high_feature_correlation')
                except Exception as e:
                    self.logger.debug(f"⚠️ Failed to check feature correlation: {e}")

            # Check target distribution differences
            if len(y_train) > 10 and len(y_val) > 10:
                train_mean, train_std = np.mean(y_train), np.std(y_train)
                val_mean, val_std = np.mean(y_val), np.std(y_val)

                # Check for significant distribution differences
                if abs(train_mean - val_mean) > train_std:
                    assessment['validation_data_issues'].append('distribution_shift')

            # Generate recommendations
            if assessment['train_data_issues'] or assessment['validation_data_issues']:
                assessment['recommendations'].extend([
                    'Review data preprocessing pipeline',
                    'Consider data imputation or outlier removal',
                    'Verify train/validation split methodology'
                ])

        except Exception as e:
            self.logger.warning(f"Data quality assessment failed: {e}")

        return assessment

    def _calculate_overall_score(self, validation_report: Dict[str, Any]) -> float:
        """Calculate overall validation score from all components."""
        try:
            scores = []

            # Overfitting score (higher is worse, so invert)
            overfitting_report = validation_report['components'].get('overfitting', {})
            if overfitting_report and 'confidence_level' in overfitting_report:
                overfitting_score = 1.0 - overfitting_report['confidence_level']  # Invert for overall score
                scores.append(overfitting_score)

            # Enhancement score (higher priority = lower score)
            enhancement_report = validation_report['components'].get('enhancement', {})
            if enhancement_report and 'confidence_level' in enhancement_report:
                priority_scores = {'low': 1.0, 'medium': 0.8, 'high': 0.5, 'critical': 0.2}
                enhancement_score = priority_scores.get(enhancement_report['priority'], 0.5)
                scores.append(enhancement_score)

            # Temporal validation score
            temporal_report = validation_report['components'].get('temporal', {})
            if temporal_report and hasattr(temporal_report, 'validation_score'):
                scores.append(temporal_report.validation_score)

            # Performance stability score
            performance_analysis = validation_report['components'].get('performance', {})
            if performance_analysis and 'performance_stability' in performance_analysis:
                scores.append(performance_analysis['performance_stability'])

            # Calculate weighted average
            if scores:
                weights = [0.3, 0.25, 0.25, 0.2]  # Weights for different components
                weights = weights[:len(scores)]  # Adjust weights to match available scores
                overall_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
            else:
                overall_score = 0.5  # Default moderate score

            return min(1.0, max(0.0, overall_score))

        except Exception as e:
            self.logger.warning(f"Overall score calculation failed: {e}")
            return 0.5

    def _generate_validation_summary(self, validation_report: Dict[str, Any]) -> str:
        """Generate a human-readable validation summary."""
        try:
            overall_score = validation_report['overall_score']

            # Determine overall assessment
            if overall_score >= 0.8:
                assessment = "EXCELLENT"
            elif overall_score >= 0.6:
                assessment = "GOOD"
            elif overall_score >= 0.4:
                assessment = "MODERATE"
            elif overall_score >= 0.2:
                assessment = "CONCERNING"
            else:
                assessment = "CRITICAL"

            summary = f"Model validation completed with {assessment} overall score ({overall_score:.3f}). "

            # Add key findings
            overfitting_report = validation_report['components'].get('overfitting', {})
            if overfitting_report.get('is_overfitting'):
                summary += f"Overfitting detected with {overfitting_report.get('severity', 'unknown')} severity. "

            enhancement_report = validation_report['components'].get('enhancement', {})
            priority = enhancement_report.get('priority', 'low')
            if priority in ['high', 'critical']:
                summary += f"High priority enhancement opportunities identified. "

            return summary

        except Exception as e:
            return f"Validation summary generation failed: {e}"

    def _generate_recommendations(self, validation_report: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on validation results."""
        recommendations = []

        try:
            # Add recommendations from individual components
            overfitting_report = validation_report['components'].get('overfitting', {})
            if overfitting_report.get('recommendations'):
                recommendations.extend(overfitting_report['recommendations'])

            enhancement_report = validation_report['components'].get('enhancement', {})
            if enhancement_report.get('detailed_recommendations'):
                recommendations.extend(enhancement_report['detailed_recommendations'])

            # Add general recommendations based on overall score
            overall_score = validation_report['overall_score']
            if overall_score < 0.4:
                recommendations.append("🚨 Consider comprehensive model redesign and retraining")
            elif overall_score < 0.6:
                recommendations.append("⚠️ Review model architecture and training procedure")
            elif overall_score < 0.8:
                recommendations.append("📊 Consider hyperparameter optimization and validation improvements")

            # Remove duplicates while preserving order
            seen = set()
            unique_recommendations = []
            for rec in recommendations:
                if rec not in seen:
                    seen.add(rec)
                    unique_recommendations.append(rec)

            return unique_recommendations

        except Exception as e:
            return [f"Recommendation generation failed: {e}"]

    def generate_model_specific_report(self,
                                     model_name: str,
                                     model_type: str,
                                     validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive per-model validation report including all validation components.

        Args:
            model_name: Name of the model
            model_type: Type of the model
            validation_results: Results from comprehensive validation

        Returns:
            Dict: Comprehensive per-model report
        """
        try:
            # Extract all validation components
            overfitting_report = validation_results.get('components', {}).get('overfitting', {})
            enhancement_report = validation_results.get('components', {}).get('enhancement', {})
            temporal_report = validation_results.get('components', {}).get('temporal', {})
            performance_analysis = validation_results.get('components', {}).get('performance', {})
            data_quality = validation_results.get('components', {}).get('data_quality', {})

            # Create comprehensive per-model report
            model_report = {
                'model_name': model_name,
                'model_type': model_type,
                'validation_timestamp': validation_results.get('validation_timestamp'),
                'overall_validation_score': validation_results.get('overall_score', 0.0),
                'validation_status': validation_results.get('validation_status', 'unknown'),

                # Overfitting Analysis
                'overfitting_analysis': {
                    'is_overfitting': overfitting_report.get('is_overfitting', False),
                    'severity': overfitting_report.get('severity', 'none'),
                    'confidence_level': overfitting_report.get('confidence_level', 0.0),
                    'accuracy_gap': overfitting_report.get('accuracy_gap', 0.0),
                    'f1_gap': overfitting_report.get('f1_gap', 0.0),
                    'train_accuracy': overfitting_report.get('train_accuracy', 0.0),
                    'val_accuracy': overfitting_report.get('val_accuracy', 0.0),
                    'indicators': overfitting_report.get('indicators', []),
                    'warnings': overfitting_report.get('warnings', []),
                    'recommendations': overfitting_report.get('recommendations', [])
                },

                # Enhancement Opportunities
                'enhancement_analysis': {
                    'enhancement_opportunities': enhancement_report.get('enhancement_opportunities', []),
                    'priority': enhancement_report.get('priority', 'low'),
                    'confidence_level': enhancement_report.get('confidence_level', 0.0),
                    'estimated_improvement_potential': enhancement_report.get('estimated_improvement_potential', 0.0),
                    'warnings': [suggestion.get('warning', '') for suggestion in enhancement_report.get('parameter_tuning_suggestions', []) if suggestion.get('warning')],
                    'improvement_suggestions': [suggestion.get('improvement_potential', '') for suggestion in enhancement_report.get('parameter_tuning_suggestions', []) if suggestion.get('improvement_potential')],
                    'detailed_recommendations': enhancement_report.get('detailed_recommendations', [])
                },

                # Temporal Validation (if available)
                'temporal_validation': {
                    'temporal_order_valid': temporal_report.get('temporal_order_valid', True) if hasattr(temporal_report, 'get') else False,
                    'leakage_detected': temporal_report.get('leakage_detected', False) if hasattr(temporal_report, 'get') else False,
                    'validation_score': temporal_report.get('validation_score', 0.0) if hasattr(temporal_report, 'get') else 0.0,
                    'warnings': temporal_report.get('warnings', []) if hasattr(temporal_report, 'get') else [],
                    'recommendations': temporal_report.get('recommendations', []) if hasattr(temporal_report, 'get') else []
                } if temporal_report else None,

                # Performance Analysis
                'performance_analysis': {
                    'train_metrics': performance_analysis.get('train_metrics', {}),
                    'validation_metrics': performance_analysis.get('validation_metrics', {}),
                    'test_metrics': performance_analysis.get('test_metrics', {}),
                    'performance_stability': performance_analysis.get('performance_stability', 0.0),
                    'generalization_score': performance_analysis.get('generalization_score', 0.0)
                },

                # Data Quality Assessment
                'data_quality': {
                    'train_data_issues': data_quality.get('train_data_issues', []),
                    'validation_data_issues': data_quality.get('validation_data_issues', []),
                    'recommendations': data_quality.get('recommendations', [])
                },

                # Walk-forward validation results (from cross-validation)
                'walk_forward_validation': validation_results.get('components', {}).get('cross_validation', {}),

                # Stacking-specific results (if applicable)
                'stacking_analysis': self._extract_stacking_analysis(validation_results),

                # Summary and final recommendations
                'summary': validation_results.get('summary', ''),
                'final_recommendations': validation_results.get('recommendations', []),
                'validation_passed': validation_results.get('overall_score', 0.0) >= 0.6  # Threshold for passing
            }

            # Add model-specific insights
            model_report['model_specific_insights'] = self._generate_model_specific_insights(
                model_name, model_type, model_report
            )

            return model_report

        except Exception as e:
            self.logger.error(f"Failed to generate model-specific report for {model_name}: {e}")
            return {
                'model_name': model_name,
                'model_type': model_type,
                'error': f"Report generation failed: {str(e)}",
                'validation_status': 'error'
            }

    def _extract_stacking_analysis(self, validation_results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract stacking-specific analysis from validation results."""
        try:
            # Check if this is a stacking ensemble by looking for ensemble-specific data
            components = validation_results.get('components', {})

            # Look for stacking results in any of the components
            for component_name, component_data in components.items():
                if component_data and isinstance(component_data, dict):
                    # Check for stacking-specific fields
                    if any(key in component_data for key in ['base_model_count', 'meta_model_count', 'oof_predictions', 'ensemble_performance']):
                        return {
                            'component': component_name,
                            'base_model_count': component_data.get('base_model_count', 0),
                            'meta_model_count': component_data.get('meta_model_count', 0),
                            'ensemble_performance': component_data.get('ensemble_performance', {}),
                            'oof_scores': component_data.get('oof_scores', {}),
                            'model_weights': component_data.get('model_weights', {}),
                            'stacking_method': component_data.get('stacking_method', 'unknown'),
                            'optimization_used': component_data.get('optimization_used', [])
                        }

            return None

        except Exception as e:
            self.logger.warning(f"Failed to extract stacking analysis: {e}")
            return None

    def _generate_model_specific_insights(self, model_name: str, model_type: str, model_report: Dict[str, Any]) -> Dict[str, Any]:
        """Generate model-specific insights and recommendations."""
        insights = {
            'key_strengths': [],
            'key_concerns': [],
            'priority_actions': [],
            'monitoring_recommendations': []
        }

        try:
            # Analyze overfitting status
            overfitting = model_report['overfitting_analysis']
            if not overfitting['is_overfitting']:
                insights['key_strengths'].append("No significant overfitting detected")
            else:
                severity = overfitting['severity']
                if severity == 'severe':
                    insights['key_concerns'].append("🚨 Severe overfitting detected - immediate action required")
                    insights['priority_actions'].append("🔧 Implement aggressive regularization and consider model redesign")
                elif severity == 'high':
                    insights['key_concerns'].append("⚠️ Significant overfitting detected")
                    insights['priority_actions'].append("🔧 Increase regularization and monitor closely")
                else:
                    insights['key_concerns'].append("📊 Moderate overfitting detected")
                    insights['priority_actions'].append("📊 Monitor performance and consider regularization")

            # Analyze enhancement opportunities
            enhancement = model_report['enhancement_analysis']
            priority = enhancement['priority']

            if priority == 'critical':
                insights['key_concerns'].append("🚨 Critical enhancement opportunities identified")
                insights['priority_actions'].append("🔧 Immediate model enhancement required")
            elif priority == 'high':
                insights['key_concerns'].append("⚠️ High priority enhancement opportunities")
                insights['priority_actions'].append("🔧 Strong enhancement opportunities identified")
            elif priority == 'medium':
                insights['key_concerns'].append("📊 Moderate enhancement opportunities")
                insights['priority_actions'].append("📊 Enhancement opportunities available")
            else:
                insights['key_strengths'].append("✅ Minimal enhancement opportunities needed")

            # Analyze performance stability
            performance = model_report['performance_analysis']
            stability = performance.get('performance_stability', 0.0)
            generalization = performance.get('generalization_score', 0.0)

            if stability > 0.8:
                insights['key_strengths'].append("✅ High performance stability")
            elif stability < 0.6:
                insights['key_concerns'].append(f"📊 Low performance stability ({stability:.2f})")
                insights['priority_actions'].append("🔧 Investigate performance instability")

            if generalization > 0.8:
                insights['key_strengths'].append("✅ Good generalization to unseen data")
            elif generalization < 0.6:
                insights['key_concerns'].append(f"📊 Poor generalization score ({generalization:.2f})")
                insights['priority_actions'].append("🔧 Improve model generalization")

            # Analyze temporal validation
            temporal = model_report['temporal_validation']
            if temporal:
                if not temporal['temporal_order_valid']:
                    insights['key_concerns'].append("🚨 Temporal order violation detected")
                    insights['priority_actions'].append("🔧 Fix temporal data ordering")
                if temporal['leakage_detected']:
                    insights['key_concerns'].append("🚨 Data leakage detected")
                    insights['priority_actions'].append("🔧 Eliminate data leakage")

            # Generate monitoring recommendations
            if overfitting['is_overfitting']:
                insights['monitoring_recommendations'].append("📊 Monitor training curves for overfitting signs")
            if enhancement['estimated_improvement_potential'] > 0.1:
                insights['monitoring_recommendations'].append("📊 Track model performance improvements after enhancements")
            if performance.get('performance_stability', 0.0) < 0.8:
                insights['monitoring_recommendations'].append("📊 Monitor performance stability across different data splits")

            # Add model-specific recommendations
            model_type_lower = model_type.lower()
            if 'xgboost' in model_type_lower or 'lightgbm' in model_type_lower or 'catboost' in model_type_lower:
                insights['monitoring_recommendations'].append("📊 Monitor feature importance stability")
                insights['monitoring_recommendations'].append("📊 Track boosting round convergence")
            elif ('neural' in model_type_lower or 'torch' in model_type_lower or
                  'keras' in model_type_lower or 'deepscaler' in model_type_lower or
                  'mamba' in model_type_lower):
                insights['monitoring_recommendations'].append("📊 Monitor gradient norms and learning curves")
                if 'deepscaler' in model_type_lower:
                    insights['monitoring_recommendations'].append("📊 Monitor scaling factor convergence")
                    insights['monitoring_recommendations'].append("📊 Track multi-scale feature integration quality")
                elif 'mamba' in model_type_lower:
                    insights['monitoring_recommendations'].append("📊 Monitor state space dynamics")
                    insights['monitoring_recommendations'].append("📊 Track selective scan efficiency")
            elif 'linear' in model_type_lower:
                insights['monitoring_recommendations'].append("📊 Monitor coefficient stability and multicollinearity")

        except Exception as e:
            self.logger.warning(f"Failed to generate model-specific insights: {e}")
            insights['error'] = f"Insight generation failed: {str(e)}"

        return insights


# Global validation orchestrator instance
DEFAULT_VALIDATION_ORCHESTRATOR = UniversalMLValidationOrchestrator()

def get_validation_orchestrator() -> UniversalMLValidationOrchestrator:
    """Get global validation orchestrator instance."""
    return DEFAULT_VALIDATION_ORCHESTRATOR

def validate_model_comprehensively(model,
                                 X_train: np.ndarray,
                                 X_val: np.ndarray,
                                 y_train: np.ndarray,
                                 y_val: np.ndarray,
                                 model_name: str = "unknown",
                                 model_type: str = "unknown",
                                 X_test: Optional[np.ndarray] = None,
                                 y_test: Optional[np.ndarray] = None,
                                 timestamps: Optional[np.ndarray] = None,
                                 include_per_model_report: bool = True) -> Dict[str, Any]:
    """
    Convenience function for comprehensive model validation with all components.

    This provides a unified interface that integrates:
    1. Overfitting detection with financial time series indicators
    2. Model enhancement detection with warnings (not specific parameters) for:
       - XGBoost, LightGBM, CatBoost, Random Forest
       - Neural Networks, DeepScaler, Advanced Mamba Hybrid
       - Linear models, SVM, KNN, Bayesian models
    3. Temporal validation and walk-forward analysis
    4. Performance analysis and data quality assessment
    5. Comprehensive per-model reporting with insights

    Args:
        model: Trained ML model
        X_train: Training features
        X_val: Validation features
        y_train: Training labels
        y_val: Validation labels
        model_name: Name of the model
        model_type: Type of model
        X_test: Optional test features
        y_test: Optional test labels
        timestamps: Optional timestamps for temporal validation
        include_per_model_report: Whether to include detailed per-model report

    Returns:
        Dict: Comprehensive validation results with per-model report
    """
    orchestrator = get_validation_orchestrator()

    # Run comprehensive validation
    validation_results = orchestrator.comprehensive_model_validation(
        model=model,
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        model_name=model_name,
        model_type=model_type,
        timestamps=timestamps
    )

    # Add per-model report if requested
    if include_per_model_report:
        model_specific_report = orchestrator.generate_model_specific_report(
            model_name, model_type, validation_results
        )
        validation_results['per_model_report'] = model_specific_report

    return validation_results

def get_model_validation_summary(model_name: str, model_type: str, validation_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get a concise summary of model validation results.

    Args:
        model_name: Name of the model
        model_type: Type of model
        validation_results: Results from comprehensive validation

    Returns:
        Dict: Concise validation summary
    """
    try:
        overall_score = validation_results.get('overall_score', 0.0)

        # Determine status
        if overall_score >= 0.8:
            status = "EXCELLENT"
            status_emoji = "✅"
        elif overall_score >= 0.6:
            status = "GOOD"
            status_emoji = "✅"
        elif overall_score >= 0.4:
            status = "MODERATE"
            status_emoji = "📊"
        elif overall_score >= 0.2:
            status = "CONCERNING"
            status_emoji = "⚠️"
        else:
            status = "CRITICAL"
            status_emoji = "🚨"

        summary = {
            'model_name': model_name,
            'model_type': model_type,
            'overall_score': overall_score,
            'status': status,
            'status_emoji': status_emoji,
            'validation_passed': overall_score >= 0.6,
            'summary': validation_results.get('summary', ''),
            'key_issues': [],
            'priority_actions': [],
            'warnings_count': 0
        }

        # Extract key issues
        overfitting_analysis = validation_results.get('components', {}).get('overfitting', {})
        if overfitting_analysis.get('is_overfitting'):
            severity = overfitting_analysis.get('severity', 'unknown')
            summary['key_issues'].append(f"Overfitting detected ({severity} severity)")

        temporal_validation = validation_results.get('components', {}).get('temporal', {})
        if temporal_validation and not temporal_validation.get('temporal_order_valid', True):
            summary['key_issues'].append("Temporal order violation detected")

        if temporal_validation and temporal_validation.get('leakage_detected', False):
            summary['key_issues'].append("Data leakage detected")

        # Extract enhancement warnings
        enhancement_analysis = validation_results.get('components', {}).get('enhancement', {})
        warnings = enhancement_analysis.get('warnings', [])
        summary['warnings_count'] = len(warnings)

        if warnings:
            summary['key_issues'].extend(warnings[:3])  # Include top 3 warnings

        # Extract priority actions
        model_insights = validation_results.get('model_specific_report', {}).get('model_specific_insights', {})
        summary['priority_actions'] = model_insights.get('priority_actions', [])

        return summary

    except Exception as e:
        logger.error(f"Failed to generate validation summary: {e}")
        return {
            'model_name': model_name,
            'model_type': model_type,
            'error': f"Summary generation failed: {str(e)}",
            'status': 'ERROR',
            'status_emoji': '❌'
        }


# Integration with existing training pipeline
class ValidationIntegrationManager:
    """
    Manages integration of comprehensive validation with existing training pipeline.
    Ensures no redundancy and proper wiring of all validation components.
    """

    def __init__(self):
        """Initialize validation integration manager."""
        self.logger = logging.getLogger('ValidationIntegrationManager')
        self.orchestrator = get_validation_orchestrator()

    def validate_model_with_reporting(self,
                                    model,
                                    X_train: np.ndarray,
                                    X_val: np.ndarray,
                                    y_train: np.ndarray,
                                    y_val: np.ndarray,
                                    model_name: str = "unknown",
                                    model_type: str = "unknown",
                                    X_test: Optional[np.ndarray] = None,
                                    y_test: Optional[np.ndarray] = None,
                                    timestamps: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Validate model with comprehensive reporting and ensure all components are included.

        This is the main entry point for model validation that ensures:
        1. All validation results are included in per-model reporting
        2. Enhancement warnings are provided instead of specific parameter tuning
        3. No redundancy with existing validation systems
        4. Proper integration with existing training pipeline

        Args:
            model: Trained ML model
            X_train: Training features
            X_val: Validation features
            y_train: Training labels
            y_val: Validation labels
            model_name: Name of the model
            model_type: Type of model
            X_test: Optional test features
            y_test: Optional test labels
            timestamps: Optional timestamps for temporal validation

        Returns:
            Dict: Complete validation report with per-model analysis
        """
        try:
            self.logger.info(f"🚀 Starting comprehensive validation for {model_name} ({model_type})")

            # Run comprehensive validation
            validation_results = self.orchestrator.comprehensive_model_validation(
                model=model,
                X_train=X_train,
                X_val=X_val,
                y_train=y_train,
                y_val=y_val,
                X_test=X_test,
                y_test=y_test,
                model_name=model_name,
                model_type=model_type,
                timestamps=timestamps
            )

            # Generate per-model report
            model_specific_report = self.orchestrator.generate_model_specific_report(
                model_name, model_type, validation_results
            )
            validation_results['per_model_report'] = model_specific_report

            # Generate concise summary
            validation_summary = get_model_validation_summary(
                model_name, model_type, validation_results
            )
            validation_results['validation_summary'] = validation_summary

            # Log key findings
            self._log_validation_findings(model_name, validation_summary, model_specific_report)

            return validation_results

        except Exception as e:
            self.logger.error(f"Validation integration failed for {model_name}: {e}")
            return {
                'model_name': model_name,
                'model_type': model_type,
                'validation_status': 'error',
                'error': str(e),
                'validation_summary': {
                    'model_name': model_name,
                    'model_type': model_type,
                    'status': 'ERROR',
                    'status_emoji': '❌',
                    'error': str(e)
                }
            }

    def _log_validation_findings(self, model_name: str, validation_summary: Dict[str, Any], model_report: Dict[str, Any]):
        """Log key validation findings."""
        try:
            self.logger.info(f"📊 Validation Summary for {model_name}:")
            self.logger.info(f"   Status: {validation_summary.get('status', 'UNKNOWN')} {validation_summary.get('status_emoji', '')}")
            self.logger.info(f"   Overall Score: {validation_summary.get('overall_score', 0.0):.3f}")
            self.logger.info(f"   Validation Passed: {validation_summary.get('validation_passed', False)}")

            key_issues = validation_summary.get('key_issues', [])
            if key_issues:
                self.logger.info(f"   Key Issues ({len(key_issues)}):")
                for issue in key_issues[:3]:  # Show top 3 issues
                    self.logger.info(f"     - {issue}")

            priority_actions = validation_summary.get('priority_actions', [])
            if priority_actions:
                self.logger.info(f"   Priority Actions ({len(priority_actions)}):")
                for action in priority_actions[:3]:  # Show top 3 actions
                    self.logger.info(f"     - {action}")

            # Log model-specific insights
            model_insights = model_report.get('model_specific_insights', {})
            key_strengths = model_insights.get('key_strengths', [])
            key_concerns = model_insights.get('key_concerns', [])

            if key_strengths:
                self.logger.info(f"   Strengths ({len(key_strengths)}):")
                for strength in key_strengths[:2]:  # Show top 2 strengths
                    self.logger.info(f"     ✅ {strength}")

            if key_concerns:
                self.logger.info(f"   Concerns ({len(key_concerns)}):")
                for concern in key_concerns[:2]:  # Show top 2 concerns
                    self.logger.info(f"     ⚠️ {concern}")

        except Exception as e:
            self.logger.warning(f"Failed to log validation findings: {e}")

    def get_validation_status(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Get concise validation status."""
        summary = validation_results.get('validation_summary', {})

        return {
            'model_name': summary.get('model_name', 'unknown'),
            'model_type': summary.get('model_type', 'unknown'),
            'overall_score': summary.get('overall_score', 0.0),
            'status': summary.get('status', 'UNKNOWN'),
            'status_emoji': summary.get('status_emoji', '❓'),
            'validation_passed': summary.get('validation_passed', False),
            'warnings_count': summary.get('warnings_count', 0),
            'key_issues': summary.get('key_issues', []),
            'priority_actions': summary.get('priority_actions', [])
        }

# Global integration manager instance
DEFAULT_VALIDATION_INTEGRATION_MANAGER = ValidationIntegrationManager()

def get_validation_integration_manager() -> ValidationIntegrationManager:
    """Get global validation integration manager instance."""
    return DEFAULT_VALIDATION_INTEGRATION_MANAGER

# Convenience function for easy model validation
def validate_and_report_model(model,
                             X_train: np.ndarray,
                             X_val: np.ndarray,
                             y_train: np.ndarray,
                             y_val: np.ndarray,
                             model_name: str = "unknown",
                             model_type: str = "unknown",
                             X_test: Optional[np.ndarray] = None,
                             y_test: Optional[np.ndarray] = None,
                             timestamps: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Convenience function to validate model and get comprehensive report.

    This provides a simple interface that ensures all validation components
    are properly integrated and all results are included in per-model reporting.

    Args:
        model: Trained ML model
        X_train: Training features
        X_val: Validation features
        y_train: Training labels
        y_val: Validation labels
        model_name: Name of the model
        model_type: Type of model
        X_test: Optional test features
        y_test: Optional test labels
        timestamps: Optional timestamps for temporal validation

    Returns:
        Dict: Complete validation report with per-model analysis and summary
    """
    manager = get_validation_integration_manager()
    return manager.validate_model_with_reporting(
        model, X_train, X_val, y_train, y_val, model_name, model_type, X_test, y_test, timestamps
    )


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
        except Exception as e:
            logger.warning(f"Could not get probabilities from model: {e}")
            # Continue without probabilities - they're optional
    
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


class ModelEnhancementDetector:
    """Detect models that could benefit from parameter tuning and optimization."""

    def __init__(self):
        """Initialize model enhancement detector."""
        self.logger = logging.getLogger('ModelEnhancementDetector')

    def detect_enhancement_opportunities(self,
                                       model,
                                       X_train: np.ndarray,
                                       X_val: np.ndarray,
                                       y_train: np.ndarray,
                                       y_val: np.ndarray,
                                       model_name: str = "unknown",
                                       model_type: str = "unknown") -> Dict[str, Any]:
        """
        Detect opportunities for model enhancement and parameter tuning.

        Args:
            model: Trained ML model
            X_train: Training features
            X_val: Validation features
            y_train: Training labels
            y_val: Validation labels
            model_name: Name of the model
            model_type: Type of model

        Returns:
            Dict: Enhancement opportunities and recommendations
        """
        opportunities = {
            'model_name': model_name,
            'model_type': model_type,
            'enhancement_opportunities': [],
            'parameter_tuning_suggestions': [],
            'performance_issues': [],
            'data_issues': [],
            'confidence_level': 0.0,
            'priority': 'low',  # low, medium, high, critical
            'estimated_improvement_potential': 0.0
        }

        try:
            # 1. Check if model is underfitting (too simple)
            underfitting_score = self._check_underfitting(model, X_train, X_val, y_train, y_val)
            if underfitting_score > 0.7:
                opportunities['enhancement_opportunities'].append('model_complexity_increase')
                opportunities['parameter_tuning_suggestions'].append({
                    'warning': 'Model appears to be underfitting - may be too simple for the data',
                    'reason': 'High training and validation errors suggest insufficient model capacity',
                    'improvement_potential': 'Investigate increasing model complexity'
                })

            # 2. Check for parameter sensitivity
            sensitivity_analysis = self._analyze_parameter_sensitivity(model, X_train, y_train)
            if sensitivity_analysis['high_sensitivity']:
                opportunities['enhancement_opportunities'].append('parameter_tuning_needed')
                opportunities['parameter_tuning_suggestions'].append({
                    'warning': 'Model shows parameter sensitivity - room for improvement through tuning',
                    'reason': f'{model_type} models typically benefit from parameter optimization',
                    'improvement_potential': 'Consider hyperparameter optimization for better performance'
                })

            # 3. Check for feature importance imbalance
            importance_analysis = self._analyze_feature_importance(model, X_train)
            if importance_analysis['imbalanced']:
                opportunities['enhancement_opportunities'].append('feature_engineering')
                opportunities['parameter_tuning_suggestions'].append({
                    'warning': 'Feature importance is heavily imbalanced',
                    'reason': f'{importance_analysis["concentration_ratio"]:.2%} of importance in top 10% of features',
                    'improvement_potential': 'Review feature selection and consider feature engineering'
                })

            # 4. Check for overfitting potential
            overfitting_potential = self._check_overfitting_potential(model, X_train, X_val, y_train, y_val)
            if overfitting_potential > 0.6:
                opportunities['enhancement_opportunities'].append('regularization_increase')
                opportunities['parameter_tuning_suggestions'].append({
                    'warning': 'Model shows signs of potential overfitting',
                    'reason': f'Overfitting potential score: {overfitting_potential:.2f}',
                    'improvement_potential': 'Consider increasing regularization to prevent overfitting'
                })

            # 5. Check for optimization opportunities
            optimization_opportunities = self._check_optimization_opportunities(model, model_type)
            opportunities['enhancement_opportunities'].extend(optimization_opportunities)

            # Add warnings for optimization opportunities instead of specific recommendations
            for opportunity in optimization_opportunities:
                opportunities['parameter_tuning_suggestions'].append({
                    'warning': f'Model-specific optimization opportunity detected: {opportunity.replace("_", " ")}',
                    'reason': f'{model_type} models can benefit from {opportunity.replace("_", " ")}',
                    'improvement_potential': f'Consider model-specific optimizations for {opportunity.replace("_", " ")}'
                })

            # Calculate overall enhancement potential
            opportunities['confidence_level'] = self._calculate_enhancement_confidence(opportunities)
            opportunities['priority'] = self._determine_priority(opportunities)
            opportunities['estimated_improvement_potential'] = self._estimate_improvement_potential(opportunities)

            # Generate detailed recommendations
            opportunities['detailed_recommendations'] = self._generate_detailed_recommendations(opportunities)

        except Exception as e:
            self.logger.error(f"Model enhancement detection failed: {e}")
            opportunities['error'] = str(e)

        return opportunities

    def _check_underfitting(self, model, X_train, X_val, y_train, y_val) -> float:
        """Check if model is underfitting (score from 0.0 to 1.0)."""
        try:
            # Get predictions
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)

            # Calculate metrics
            train_mse = np.mean((y_train - train_pred) ** 2)
            val_mse = np.mean((y_val - val_pred) ** 2)

            # Normalize by target variance
            target_var = np.var(y_train)
            if target_var == 0:
                return 0.0

            train_normalized_error = train_mse / target_var
            val_normalized_error = val_mse / target_var

            # Underfitting score: higher when both train and val errors are high
            underfitting_score = min(1.0, (train_normalized_error + val_normalized_error) / 2.0)

            return underfitting_score

        except Exception as e:
            self.logger.warning(f"Underfitting check failed: {e}")
            return 0.0

    def _analyze_parameter_sensitivity(self, model, X_train, y_train) -> Dict[str, Any]:
        """Analyze parameter sensitivity to determine tuning needs."""
        analysis = {
            'high_sensitivity': False,
            'model_type': model.__class__.__name__.lower()
        }

        try:
            # Simple parameter sensitivity check based on model type
            model_type = model.__class__.__name__.lower()

            # All these model types typically benefit from parameter tuning
            if ('xgb' in model_type or 'xgboost' in model_type or
                'lgbm' in model_type or 'lightgbm' in model_type or
                'catboost' in model_type or
                'randomforest' in model_type or 'neural' in model_type or
                'torch' in model_type or 'keras' in model_type or
                'deepscaler' in model_type or 'mamba' in model_type or
                'linear' in model_type or 'ridge' in model_type or
                'lasso' in model_type or 'elasticnet' in model_type):

                analysis['high_sensitivity'] = True

        except Exception as e:
            self.logger.warning(f"Parameter sensitivity analysis failed: {e}")

        return analysis

    def _analyze_feature_importance(self, model, X_train) -> Dict[str, Any]:
        """Analyze feature importance distribution."""
        analysis = {
            'imbalanced': False,
            'concentration_ratio': 0.0,
            'top_features': []
        }

        try:
            # Get feature importance
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance = np.abs(model.coef_).flatten()
            else:
                return analysis  # No feature importance available

            # Calculate concentration
            sorted_importance = np.sort(importance)[::-1]
            top_10_percent = sorted_importance[:max(1, len(sorted_importance) // 10)]
            analysis['concentration_ratio'] = np.sum(top_10_percent) / np.sum(sorted_importance)

            if analysis['concentration_ratio'] > 0.8:  # 80% of importance in top 10% features
                analysis['imbalanced'] = True

            # Get top feature indices
            top_indices = np.argsort(importance)[::-1][:10]
            analysis['top_features'] = top_indices.tolist()

        except Exception as e:
            self.logger.warning(f"Feature importance analysis failed: {e}")

        return analysis

    def _check_overfitting_potential(self, model, X_train, X_val, y_train, y_val) -> float:
        """Check potential for overfitting (score from 0.0 to 1.0)."""
        try:
            # Get predictions
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)

            # Calculate train vs validation performance gap
            train_mse = np.mean((y_train - train_pred) ** 2)
            val_mse = np.mean((y_val - val_pred) ** 2)

            # Calculate overfitting potential
            if train_mse == 0:
                return 1.0  # Perfect training fit = high overfitting risk

            performance_ratio = val_mse / train_mse
            overfitting_potential = min(1.0, max(0.0, 1.0 - 1.0 / (1.0 + performance_ratio)))

            return overfitting_potential

        except Exception as e:
            self.logger.warning(f"Overfitting potential check failed: {e}")
            return 0.0

    def _check_optimization_opportunities(self, model, model_type: str) -> List[str]:
        """Check for model-specific optimization opportunities."""
        opportunities = []

        try:
            # Model-specific optimization opportunities
            if ('neural' in model_type.lower() or 'torch' in model_type.lower() or
                'keras' in model_type.lower() or 'deepscaler' in model_type.lower() or
                'mamba' in model_type.lower()):
                opportunities.extend([
                    'learning_rate_scheduling',
                    'batch_normalization',
                    'gradient_clipping',
                    'early_stopping_optimization',
                    'architecture_optimization',
                    'attention_mechanism_tuning'
                ])

                # Add specific optimizations for advanced architectures
                if 'deepscaler' in model_type.lower():
                    opportunities.extend([
                        'scaling_factor_optimization',
                        'time_series_preprocessing_tuning',
                        'multi_scale_feature_integration'
                    ])
                elif 'mamba' in model_type.lower():
                    opportunities.extend([
                        'state_space_optimization',
                        'selective_scan_tuning',
                        'hardware_aware_optimization'
                    ])

            elif 'xgb' in model_type.lower() or 'xgboost' in model_type.lower() or 'lgbm' in model_type.lower() or 'lightgbm' in model_type.lower() or 'catboost' in model_type.lower():
                opportunities.extend([
                    'tree_structure_optimization',
                    'feature_interaction_constraints',
                    'monotone_constraints',
                    'categorical_feature_handling',
                    'boosting_round_optimization'
                ])

            elif 'linear' in model_type.lower() or 'ridge' in model_type.lower() or 'lasso' in model_type.lower() or 'elasticnet' in model_type.lower():
                opportunities.extend([
                    'regularization_optimization',
                    'feature_scaling_check',
                    'multicollinearity_analysis'
                ])

            elif 'randomforest' in model_type.lower() or 'extratrees' in model_type.lower():
                opportunities.extend([
                    'ensemble_diversity_optimization',
                    'feature_sampling_optimization',
                    'bootstrap_optimization'
                ])

            elif 'svm' in model_type.lower() or 'svc' in model_type.lower():
                opportunities.extend([
                    'kernel_optimization',
                    'gamma_parameter_tuning',
                    'class_weight_optimization'
                ])

            elif 'knn' in model_type.lower():
                opportunities.extend([
                    'distance_metric_optimization',
                    'neighbor_count_optimization',
                    'weight_function_optimization'
                ])

            elif 'bayesian' in model_type.lower() or 'naive' in model_type.lower():
                opportunities.extend([
                    'prior_optimization',
                    'smoothing_parameter_tuning',
                    'feature_independence_assumptions'
                ])

            # Default opportunities for unknown model types
            else:
                opportunities.extend([
                    'general_hyperparameter_tuning',
                    'ensemble_methods',
                    'cross_validation_optimization'
                ])

        except Exception as e:
            self.logger.warning(f"Optimization opportunities check failed: {e}")

        return opportunities

    def _calculate_enhancement_confidence(self, opportunities: Dict[str, Any]) -> float:
        """Calculate confidence level for enhancement recommendations."""
        confidence_factors = []

        # Base confidence
        base_confidence = 0.5

        # Factor based on number of opportunities found
        n_opportunities = len(opportunities['enhancement_opportunities'])
        opportunity_factor = min(0.3, n_opportunities * 0.1)

        # Factor based on parameter tuning suggestions
        n_suggestions = len(opportunities['parameter_tuning_suggestions'])
        suggestion_factor = min(0.2, n_suggestions * 0.05)

        total_confidence = base_confidence + opportunity_factor + suggestion_factor

        return min(1.0, total_confidence)

    def _determine_priority(self, opportunities: Dict[str, Any]) -> str:
        """Determine priority level for enhancement."""
        n_opportunities = len(opportunities['enhancement_opportunities'])
        confidence = opportunities['confidence_level']

        if n_opportunities >= 3 and confidence > 0.8:
            return 'critical'
        elif n_opportunities >= 2 and confidence > 0.6:
            return 'high'
        elif n_opportunities >= 1 and confidence > 0.4:
            return 'medium'
        else:
            return 'low'

    def _estimate_improvement_potential(self, opportunities: Dict[str, Any]) -> float:
        """Estimate potential improvement from enhancements."""
        improvement_factors = {
            'model_complexity_increase': 0.15,
            'parameter_tuning_needed': 0.20,
            'feature_engineering': 0.10,
            'regularization_increase': 0.05,
            'learning_rate_scheduling': 0.08,
            'tree_structure_optimization': 0.12,
            'categorical_feature_handling': 0.10,
            'boosting_round_optimization': 0.09,
            'architecture_optimization': 0.14,
            'attention_mechanism_tuning': 0.11,
            'scaling_factor_optimization': 0.13,
            'time_series_preprocessing_tuning': 0.12,
            'multi_scale_feature_integration': 0.11,
            'state_space_optimization': 0.15,
            'selective_scan_tuning': 0.13,
            'hardware_aware_optimization': 0.10
        }

        total_potential = 0.0
        for opportunity in opportunities['enhancement_opportunities']:
            if opportunity in improvement_factors:
                total_potential += improvement_factors[opportunity]

        return min(0.5, total_potential)  # Cap at 50% potential improvement

    def _generate_detailed_recommendations(self, opportunities: Dict[str, Any]) -> List[str]:
        """Generate detailed recommendations based on analysis."""
        recommendations = []

        if opportunities['priority'] == 'critical':
            recommendations.append("🚨 CRITICAL: Immediate model enhancement required")
        elif opportunities['priority'] == 'high':
            recommendations.append("⚠️ HIGH: Strong enhancement opportunities identified")
        elif opportunities['priority'] == 'medium':
            recommendations.append("📊 MEDIUM: Moderate enhancement opportunities available")
        else:
            recommendations.append("✅ LOW: Minimal enhancement opportunities found")

        # Add specific recommendations based on opportunities
        for opportunity in opportunities['enhancement_opportunities']:
            if opportunity == 'model_complexity_increase':
                recommendations.append("🔧 Consider increasing model complexity (deeper trees, more estimators, additional layers)")
            elif opportunity == 'parameter_tuning_needed':
                recommendations.append("🔧 Perform comprehensive hyperparameter optimization")
            elif opportunity == 'feature_engineering':
                recommendations.append("🔧 Review feature selection and consider feature engineering")
            elif opportunity == 'regularization_increase':
                recommendations.append("🔧 Increase regularization to prevent overfitting")

        return recommendations