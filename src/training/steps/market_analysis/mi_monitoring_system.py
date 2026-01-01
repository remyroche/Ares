"""
MI Monitoring System - Real-time Mutual Information Tracking

This system provides:
- Real-time MI monitoring during training
- Automated compliance checking
- Performance dashboard
- Alert system for MI optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path

from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import roc_auc_score, accuracy_score

logger = logging.getLogger(__name__)


class MIStatus(Enum):
    """MI status enumeration."""
    CRITICAL = "CRITICAL"  # MI < 0.005
    LOW = "LOW"           # 0.005 <= MI < 0.01
    MODERATE = "MODERATE" # 0.01 <= MI < 0.02
    GOOD = "GOOD"         # 0.02 <= MI < 0.05
    EXCELLENT = "EXCELLENT" # MI >= 0.05


@dataclass
class MIMetrics:
    """MI metrics data structure."""
    timestamp: datetime
    specialist_name: str
    prediction_mi: float
    feature_mi_avg: float
    feature_mi_max: float
    high_mi_features: int
    total_features: int
    auc_score: Optional[float] = None
    accuracy_score: Optional[float] = None
    training_epoch: Optional[int] = None
    status: MIStatus = MIStatus.CRITICAL
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'specialist_name': self.specialist_name,
            'prediction_mi': self.prediction_mi,
            'feature_mi_avg': self.feature_mi_avg,
            'feature_mi_max': self.feature_mi_max,
            'high_mi_features': self.high_mi_features,
            'total_features': self.total_features,
            'auc_score': self.auc_score,
            'accuracy_score': self.accuracy_score,
            'training_epoch': self.training_epoch,
            'status': self.status.value
        }


@dataclass
class MIAlert:
    """MI alert data structure."""
    timestamp: datetime
    specialist_name: str
    alert_type: str
    message: str
    mi_score: float
    target_mi: float
    severity: str  # INFO, WARNING, CRITICAL
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'specialist_name': self.specialist_name,
            'alert_type': self.alert_type,
            'message': self.message,
            'mi_score': self.mi_score,
            'target_mi': self.target_mi,
            'severity': self.severity
        }


class MIMonitor:
    """Real-time MI monitoring system."""
    
    def __init__(self, target_mi: float = 0.02, alert_threshold: float = 0.01):
        self.target_mi = target_mi
        self.alert_threshold = alert_threshold
        self.mi_history: List[MIMetrics] = []
        self.alerts: List[MIAlert] = []
        self.active_specialists: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def compute_mi_metrics(self, specialist_name: str, features: pd.DataFrame, 
                          labels: pd.Series, predictions: np.ndarray, 
                          probabilities: np.ndarray, epoch: Optional[int] = None,
                          auc_score: Optional[float] = None,
                          accuracy_score: Optional[float] = None) -> MIMetrics:
        """Compute comprehensive MI metrics."""
        
        try:
            # Prediction MI to target
            pred_mi = mutual_info_regression(
                predictions.reshape(-1, 1), labels.values
            )[0]
            
            # Feature MI analysis
            feature_mi_scores = []
            for col in features.select_dtypes(include=[np.number]).columns:
                mi_score = mutual_info_regression(
                    features[col].values.reshape(-1, 1), labels.values
                )[0]
                feature_mi_scores.append(mi_score)
            
            if feature_mi_scores:
                feature_mi_avg = np.mean(feature_mi_scores)
                feature_mi_max = np.max(feature_mi_scores)
                high_mi_features = sum(1 for mi in feature_mi_scores if mi > self.target_mi)
            else:
                feature_mi_avg = 0.0
                feature_mi_max = 0.0
                high_mi_features = 0
            
            # Determine status
            if pred_mi < 0.005:
                status = MIStatus.CRITICAL
            elif pred_mi < 0.01:
                status = MIStatus.LOW
            elif pred_mi < 0.02:
                status = MIStatus.MODERATE
            elif pred_mi < 0.05:
                status = MIStatus.GOOD
            else:
                status = MIStatus.EXCELLENT
            
            metrics = MIMetrics(
                timestamp=datetime.utcnow(),
                specialist_name=specialist_name,
                prediction_mi=pred_mi,
                feature_mi_avg=feature_mi_avg,
                feature_mi_max=feature_mi_max,
                high_mi_features=high_mi_features,
                total_features=len(features.columns),
                auc_score=auc_score,
                accuracy_score=accuracy_score,
                training_epoch=epoch,
                status=status
            )
            
            # Store metrics
            self.mi_history.append(metrics)
            
            # Check for alerts
            self._check_alerts(metrics)
            
            # Update active specialist
            self.active_specialists[specialist_name] = {
                'latest_mi': pred_mi,
                'latest_status': status.value,
                'last_update': datetime.utcnow(),
                'total_features': len(features.columns),
                'high_mi_features': high_mi_features
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"MI computation failed for {specialist_name}: {e}")
            # Return default metrics
            return MIMetrics(
                timestamp=datetime.utcnow(),
                specialist_name=specialist_name,
                prediction_mi=0.0,
                feature_mi_avg=0.0,
                feature_mi_max=0.0,
                high_mi_features=0,
                total_features=len(features.columns),
                status=MIStatus.CRITICAL
            )
    
    def _check_alerts(self, metrics: MIMetrics):
        """Check for MI alerts and generate notifications."""
        
        # Critical alert for very low MI
        if metrics.prediction_mi < 0.005:
            alert = MIAlert(
                timestamp=datetime.utcnow(),
                specialist_name=metrics.specialist_name,
                alert_type="CRITICAL_MI",
                message=f"Critical MI detected: {metrics.prediction_mi:.4f} << target {self.target_mi}",
                mi_score=metrics.prediction_mi,
                target_mi=self.target_mi,
                severity="CRITICAL"
            )
            self.alerts.append(alert)
            self.logger.error(f"🚨 CRITICAL MI ALERT: {alert.message}")
        
        # Warning alert for low MI
        elif metrics.prediction_mi < self.alert_threshold:
            alert = MIAlert(
                timestamp=datetime.utcnow(),
                specialist_name=metrics.specialist_name,
                alert_type="LOW_MI",
                message=f"Low MI detected: {metrics.prediction_mi:.4f} < target {self.target_mi}",
                mi_score=metrics.prediction_mi,
                target_mi=self.target_mi,
                severity="WARNING"
            )
            self.alerts.append(alert)
            self.logger.warning(f"⚠️ LOW MI ALERT: {alert.message}")
        
        # Success alert for achieving target
        elif metrics.prediction_mi >= self.target_mi:
            alert = MIAlert(
                timestamp=datetime.utcnow(),
                specialist_name=metrics.specialist_name,
                alert_type="TARGET_ACHIEVED",
                message=f"Target MI achieved: {metrics.prediction_mi:.4f} >= target {self.target_mi}",
                mi_score=metrics.prediction_mi,
                target_mi=self.target_mi,
                severity="INFO"
            )
            self.alerts.append(alert)
            self.logger.info(f"✅ MI SUCCESS: {alert.message}")
    
    def get_specialist_summary(self, specialist_name: str) -> Dict[str, Any]:
        """Get summary for a specific specialist."""
        specialist_metrics = [m for m in self.mi_history if m.specialist_name == specialist_name]
        
        if not specialist_metrics:
            return {'error': f'No metrics found for {specialist_name}'}
        
        latest = specialist_metrics[-1]
        
        # Calculate trends
        if len(specialist_metrics) >= 2:
            recent_mi = [m.prediction_mi for m in specialist_metrics[-5:]]
            mi_trend = np.mean(np.diff(recent_mi)) if len(recent_mi) > 1 else 0.0
        else:
            mi_trend = 0.0
        
        return {
            'specialist_name': specialist_name,
            'latest_mi': latest.prediction_mi,
            'latest_status': latest.status.value,
            'target_met': latest.prediction_mi >= self.target_mi,
            'total_features': latest.total_features,
            'high_mi_features': latest.high_mi_features,
            'mi_trend': mi_trend,
            'total_measurements': len(specialist_metrics),
            'latest_timestamp': latest.timestamp.isoformat(),
            'alerts_count': len([a for a in self.alerts if a.specialist_name == specialist_name])
        }
    
    def get_overall_summary(self) -> Dict[str, Any]:
        """Get overall MI monitoring summary."""
        
        if not self.mi_history:
            return {'error': 'No MI metrics available'}
        
        # Latest metrics per specialist
        latest_metrics = {}
        for metrics in self.mi_history:
            if metrics.specialist_name not in latest_metrics or metrics.timestamp > latest_metrics[metrics.specialist_name].timestamp:
                latest_metrics[metrics.specialist_name] = metrics
        
        # Overall statistics
        all_mi_scores = [m.prediction_mi for m in self.mi_history]
        latest_mi_scores = [m.prediction_mi for m in latest_metrics.values()]
        
        # Status distribution
        status_counts = {}
        for status in MIStatus:
            status_counts[status.value] = sum(1 for m in latest_metrics.values() if m.status == status)
        
        # Compliance
        compliant_specialists = sum(1 for m in latest_metrics.values() if m.prediction_mi >= self.target_mi)
        
        return {
            'total_specialists': len(latest_metrics),
            'compliant_specialists': compliant_specialists,
            'compliance_rate': compliant_specialists / len(latest_metrics) if latest_metrics else 0.0,
            'overall_avg_mi': np.mean(latest_mi_scores) if latest_mi_scores else 0.0,
            'overall_max_mi': np.max(latest_mi_scores) if latest_mi_scores else 0.0,
            'overall_min_mi': np.min(latest_mi_scores) if latest_mi_scores else 0.0,
            'status_distribution': status_counts,
            'total_measurements': len(self.mi_history),
            'total_alerts': len(self.alerts),
            'critical_alerts': len([a for a in self.alerts if a.severity == 'CRITICAL']),
            'warning_alerts': len([a for a in self.alerts if a.severity == 'WARNING']),
            'target_mi': self.target_mi,
            'latest_timestamp': max(m.timestamp for m in self.mi_history).isoformat()
        }
    
    def get_mi_trends(self, specialist_name: Optional[str] = None, hours: int = 24) -> Dict[str, Any]:
        """Get MI trends over time."""
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        if specialist_name:
            relevant_metrics = [m for m in self.mi_history 
                             if m.specialist_name == specialist_name and m.timestamp > cutoff_time]
        else:
            relevant_metrics = [m for m in self.mi_history if m.timestamp > cutoff_time]
        
        if not relevant_metrics:
            return {'error': 'No recent metrics found'}
        
        # Group by specialist
        trends = {}
        for metrics in relevant_metrics:
            if metrics.specialist_name not in trends:
                trends[metrics.specialist_name] = []
            trends[metrics.specialist_name].append({
                'timestamp': metrics.timestamp.isoformat(),
                'mi_score': metrics.prediction_mi,
                'status': metrics.status.value
            })
        
        # Sort by timestamp
        for specialist in trends:
            trends[specialist].sort(key=lambda x: x['timestamp'])
        
        return {
            'specialist_name': specialist_name,
            'timeframe_hours': hours,
            'trends': trends,
            'total_metrics': len(relevant_metrics)
        }
    
    def export_metrics(self, filepath: str, format: str = 'json'):
        """Export MI metrics to file."""
        
        export_data = {
            'summary': self.get_overall_summary(),
            'metrics': [m.to_dict() for m in self.mi_history],
            'alerts': [a.to_dict() for a in self.alerts],
            'active_specialists': self.active_specialists,
            'export_timestamp': datetime.utcnow().isoformat()
        }
        
        if format.lower() == 'json':
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
        elif format.lower() == 'csv':
            # Export metrics to CSV
            df_metrics = pd.DataFrame([m.to_dict() for m in self.mi_history])
            df_metrics.to_csv(filepath, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        self.logger.info(f"MI metrics exported to {filepath}")
    
    def generate_report(self) -> str:
        """Generate comprehensive MI monitoring report."""
        
        summary = self.get_overall_summary()
        
        if 'error' in summary:
            return "No MI data available for reporting"
        
        report = f"""
# MI Monitoring Report

**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}
**Target MI:** {self.target_mi}

## Overall Summary

- **Total Specialists:** {summary['total_specialists']}
- **Compliant Specialists:** {summary['compliant_specialists']} ({summary['compliance_rate']:.1%})
- **Overall Average MI:** {summary['overall_avg_mi']:.4f}
- **Overall Max MI:** {summary['overall_max_mi']:.4f}
- **Overall Min MI:** {summary['overall_min_mi']:.4f}

## Status Distribution

"""
        
        for status, count in summary['status_distribution'].items():
            report += f"- **{status}:** {count}\n"
        
        report += f"""
## Alerts Summary

- **Total Alerts:** {summary['total_alerts']}
- **Critical Alerts:** {summary['critical_alerts']}
- **Warning Alerts:** {summary['warning_alerts']}

## Specialist Details

"""
        
        for specialist_name in self.active_specialists:
            specialist_summary = self.get_specialist_summary(specialist_name)
            if 'error' not in specialist_summary:
                report += f"""
### {specialist_name}

- **Latest MI:** {specialist_summary['latest_mi']:.4f}
- **Status:** {specialist_summary['latest_status']}
- **Target Met:** {'✅' if specialist_summary['target_met'] else '❌'}
- **Features:** {specialist_summary['total_features']} total, {specialist_summary['high_mi_features']} high-MI
- **MI Trend:** {specialist_summary['mi_trend']:+.4f}
- **Alerts:** {specialist_summary['alerts_count']}

"""
        
        report += f"""
## Recommendations

"""
        
        if summary['compliance_rate'] < 0.5:
            report += "- **URGENT:** Less than 50% of specialists meet MI target. Consider feature engineering improvements.\n"
        elif summary['compliance_rate'] < 0.8:
            report += "- **MODERATE:** Some specialists need MI improvement. Focus on non-linear features.\n"
        else:
            report += "- **GOOD:** Most specialists meet MI target. Continue monitoring.\n"
        
        if summary['critical_alerts'] > 0:
            report += "- **CRITICAL:** Address critical MI alerts immediately.\n"
        
        if summary['warning_alerts'] > summary['total_specialists']:
            report += "- **ATTENTION:** High number of warnings. Review feature selection.\n"
        
        return report


# Global MI monitor instance
mi_monitor = MIMonitor()


def get_mi_monitor() -> MIMonitor:
    """Get the global MI monitor instance."""
    return mi_monitor


def monitor_mi_during_training(specialist_name: str, features: pd.DataFrame, 
                             labels: pd.Series, predictions: np.ndarray,
                             probabilities: np.ndarray, epoch: Optional[int] = None) -> MIMetrics:
    """Convenience function to monitor MI during training."""
    monitor = get_mi_monitor()
    return monitor.compute_mi_metrics(
        specialist_name, features, labels, predictions, probabilities, epoch
    )
