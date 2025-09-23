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
                    'action': 'increase_model_complexity',
                    'parameters': ['max_depth', 'n_estimators', 'hidden_layers'],
                    'reason': 'Model appears to be underfitting - may be too simple'
                })

            # 2. Check for parameter sensitivity
            sensitivity_analysis = self._analyze_parameter_sensitivity(model, X_train, y_train)
            if sensitivity_analysis['high_sensitivity']:
                opportunities['enhancement_opportunities'].append('parameter_tuning_needed')
                opportunities['parameter_tuning_suggestions'].extend(sensitivity_analysis['suggestions'])

            # 3. Check for feature importance imbalance
            importance_analysis = self._analyze_feature_importance(model, X_train)
            if importance_analysis['imbalanced']:
                opportunities['enhancement_opportunities'].append('feature_engineering')
                opportunities['parameter_tuning_suggestions'].append({
                    'action': 'feature_selection_regularization',
                    'parameters': ['feature_fraction', 'colsample_bytree', 'max_features'],
                    'reason': 'Feature importance is heavily imbalanced'
                })

            # 4. Check for overfitting potential
            overfitting_potential = self._check_overfitting_potential(model, X_train, X_val, y_train, y_val)
            if overfitting_potential > 0.6:
                opportunities['enhancement_opportunities'].append('regularization_increase')
                opportunities['parameter_tuning_suggestions'].append({
                    'action': 'increase_regularization',
                    'parameters': ['reg_lambda', 'reg_alpha', 'dropout', 'l2_penalty'],
                    'reason': 'Model shows signs of potential overfitting'
                })

            # 5. Check for optimization opportunities
            optimization_opportunities = self._check_optimization_opportunities(model, model_type)
            opportunities['enhancement_opportunities'].extend(optimization_opportunities)

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
            'suggestions': []
        }

        try:
            # Simple parameter sensitivity check based on model type
            model_type = model.__class__.__name__.lower()

            if 'xgb' in model_type or 'xgboost' in model_type:
                analysis['suggestions'].extend([
                    {'parameter': 'learning_rate', 'range': [0.001, 0.3], 'method': 'log_scale'},
                    {'parameter': 'max_depth', 'range': [3, 12], 'method': 'linear'},
                    {'parameter': 'n_estimators', 'range': [50, 1000], 'method': 'linear'},
                    {'parameter': 'reg_lambda', 'range': [0.1, 10.0], 'method': 'log_scale'}
                ])
                analysis['high_sensitivity'] = True

            elif 'lgbm' in model_type or 'lightgbm' in model_type:
                analysis['suggestions'].extend([
                    {'parameter': 'learning_rate', 'range': [0.001, 0.3], 'method': 'log_scale'},
                    {'parameter': 'num_leaves', 'range': [10, 200], 'method': 'linear'},
                    {'parameter': 'feature_fraction', 'range': [0.4, 1.0], 'method': 'linear'},
                    {'parameter': 'bagging_fraction', 'range': [0.4, 1.0], 'method': 'linear'}
                ])
                analysis['high_sensitivity'] = True

            elif 'randomforest' in model_type:
                analysis['suggestions'].extend([
                    {'parameter': 'n_estimators', 'range': [50, 500], 'method': 'linear'},
                    {'parameter': 'max_depth', 'range': [5, 30], 'method': 'linear'},
                    {'parameter': 'min_samples_split', 'range': [2, 20], 'method': 'linear'},
                    {'parameter': 'min_samples_leaf', 'range': [1, 10], 'method': 'linear'}
                ])
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
        """Check for optimization opportunities based on model type."""
        opportunities = []

        try:
            # Model-specific optimization opportunities
            if 'neural' in model_type.lower() or 'torch' in model_type.lower():
                opportunities.extend([
                    'learning_rate_scheduling',
                    'batch_normalization',
                    'gradient_clipping',
                    'early_stopping_optimization'
                ])

            elif 'xgb' in model_type.lower() or 'lgbm' in model_type.lower():
                opportunities.extend([
                    'tree_structure_optimization',
                    'feature_interaction_constraints',
                    'monotone_constraints'
                ])

            elif 'linear' in model_type.lower():
                opportunities.extend([
                    'regularization_optimization',
                    'feature_scaling_check',
                    'multicollinearity_analysis'
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
            'tree_structure_optimization': 0.12
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

            # Generate summary and recommendations
            validation_report['summary'] = self._generate_validation_summary(validation_report)
            validation_report['recommendations'] = self._generate_recommendations(validation_report)

            self.logger.info(f"✅ Comprehensive validation completed with overall score: {validation_report['overall_score']:.3f}")

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
            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_squared_error, r2_score

            metrics = {}

            # Classification metrics
            try:
                metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
                metrics['f1'] = float(f1_score(y_true, y_pred, average='weighted'))
                metrics['precision'] = float(precision_score(y_true, y_pred, average='weighted'))
                metrics['recall'] = float(recall_score(y_true, y_pred, average='weighted'))
            except:
                pass

            # Regression metrics
            try:
                metrics['mse'] = float(mean_squared_error(y_true, y_pred))
                metrics['rmse'] = float(np.sqrt(metrics['mse']))
                metrics['r2'] = float(r2_score(y_true, y_pred))
            except:
                pass

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
                except:
                    pass

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