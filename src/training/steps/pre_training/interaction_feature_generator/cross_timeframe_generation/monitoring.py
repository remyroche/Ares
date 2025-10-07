"""
Monitoring and Automation System

Implements comprehensive monitoring with:
- Adaptive penalties meta-learning
- BOCPD triggers for regime changes
- Performance dashboards
- Automated retraining triggers
- Alert systems
- State persistence and recovery
"""

from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from .statistical_selection import SelectionResult

# Try to import dashboard libraries
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logging.warning("Plotly not available, using simplified dashboards")


@dataclass
class MonitoringAlert:
    """Monitoring alert."""
    alert_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    message: str
    timestamp: datetime
    metadata: Dict[str, Any]


@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring."""
    timestamp: datetime
    ic: float
    ic_std: float
    sharpe: float
    max_drawdown: float
    feature_count: int
    regime: str
    metadata: Dict[str, Any]


@dataclass
class SystemState:
    """System state for persistence."""
    timestamp: datetime
    pipeline_state: Dict[str, Any]
    performance_metrics: List[PerformanceMetrics]
    alerts: List[MonitoringAlert]
    penalty_parameters: Dict[str, float]
    bocpd_state: Dict[str, Any]
    metadata: Dict[str, Any]


class AdaptivePenaltyLearner:
    """Meta-learns penalty parameters based on recent performance."""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize penalty parameters
        self.penalty_parameters = {
            'lambda_unc': 0.10,
            'lambda_cost': 0.05,
            'lambda_stale': 0.05
        }
        
        # Learning parameters
        self.learning_rate = 0.01
        self.adaptation_range = config.meta_learning_range
        self.performance_history = []
        self.penalty_history = []
    
    def update_penalties(self, 
                        recent_performance: List[PerformanceMetrics],
                        market_conditions: Dict[str, Any]) -> Dict[str, float]:
        """
        Update penalty parameters based on recent performance.
        
        Args:
            recent_performance: Recent performance metrics
            market_conditions: Current market conditions
            
        Returns:
            Updated penalty parameters
        """
        if not recent_performance:
            return self.penalty_parameters
        
        # Analyze recent performance
        recent_ics = [p.ic for p in recent_performance if not pd.isna(p.ic)]
        recent_sharpes = [p.sharpe for p in recent_performance if not pd.isna(p.sharpe)]
        
        if not recent_ics:
            return self.penalty_parameters
        
        avg_ic = np.mean(recent_ics)
        ic_std = np.std(recent_ics)
        avg_sharpe = np.mean(recent_sharpes) if recent_sharpes else 0.0
        
        # Market conditions
        volatility_level = market_conditions.get('volatility_level', 0.5)
        news_proximity = market_conditions.get('news_proximity', 0.0)
        
        # Update penalties based on performance
        if avg_ic < 0.05:  # Low IC, increase uncertainty penalty
            self.penalty_parameters['lambda_unc'] = min(
                0.20, 
                self.penalty_parameters['lambda_unc'] + self.learning_rate
            )
        elif avg_ic > 0.15:  # High IC, decrease uncertainty penalty
            self.penalty_parameters['lambda_unc'] = max(
                0.05, 
                self.penalty_parameters['lambda_unc'] - self.learning_rate
            )
        
        # Adjust based on volatility
        if volatility_level > 0.7:  # High volatility
            self.penalty_parameters['lambda_unc'] = min(
                0.20, 
                self.penalty_parameters['lambda_unc'] + self.learning_rate * 0.5
            )
        
        # Adjust based on news proximity
        if news_proximity > 0.5:  # Near news events
            self.penalty_parameters['lambda_stale'] = min(
                0.15, 
                self.penalty_parameters['lambda_stale'] + self.learning_rate
            )
        
        # Adjust cost penalty based on feature count
        recent_feature_counts = [p.feature_count for p in recent_performance]
        if recent_feature_counts:
            avg_feature_count = np.mean(recent_feature_counts)
            if avg_feature_count > 100:  # Too many features
                self.penalty_parameters['lambda_cost'] = min(
                    0.10, 
                    self.penalty_parameters['lambda_cost'] + self.learning_rate
                )
            elif avg_feature_count < 50:  # Too few features
                self.penalty_parameters['lambda_cost'] = max(
                    0.02, 
                    self.penalty_parameters['lambda_cost'] - self.learning_rate
                )
        
        # Record history
        self.performance_history.append({
            'timestamp': datetime.now(),
            'avg_ic': avg_ic,
            'ic_std': ic_std,
            'avg_sharpe': avg_sharpe,
            'volatility_level': volatility_level,
            'news_proximity': news_proximity
        })
        
        self.penalty_history.append(self.penalty_parameters.copy())
        
        return self.penalty_parameters
    
    def get_penalty_parameters(self) -> Dict[str, float]:
        """Get current penalty parameters."""
        return self.penalty_parameters.copy()
    
    def save_state(self, filepath: str):
        """Save penalty learner state."""
        state = {
            'penalty_parameters': self.penalty_parameters,
            'performance_history': self.performance_history,
            'penalty_history': self.penalty_history
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
    
    def load_state(self, filepath: str):
        """Load penalty learner state."""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.penalty_parameters = state.get('penalty_parameters', self.penalty_parameters)
        self.performance_history = state.get('performance_history', [])
        self.penalty_history = state.get('penalty_history', [])


class BOCPDTrigger:
    """BOCPD-based triggers for regime changes."""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        self.hazard = config.bocpd_hazard
        self.change_threshold = 0.5
        self.alert_threshold = 0.8
        
        # BOCPD state
        self.run_length = 0
        self.last_change_time = None
        self.change_history = []
    
    def check_for_triggers(self, 
                          new_observation: float,
                          timestamp: datetime) -> List[MonitoringAlert]:
        """
        Check for BOCPD triggers.
        
        Args:
            new_observation: New observation value
            timestamp: Current timestamp
            
        Returns:
            List of triggered alerts
        """
        alerts = []
        
        # Calculate change point probability
        cp_prob = self.hazard / (self.hazard + self.run_length)
        
        # Update run length
        self.run_length += 1
        
        # Check for change point
        if cp_prob > self.change_threshold:
            # Change point detected
            self.run_length = 0
            self.last_change_time = timestamp
            
            # Record change
            self.change_history.append({
                'timestamp': timestamp,
                'probability': cp_prob,
                'run_length': self.run_length
            })
            
            # Create alert
            alert = MonitoringAlert(
                alert_type='regime_change',
                severity='high' if cp_prob > self.alert_threshold else 'medium',
                message=f'Regime change detected with probability {cp_prob:.3f}',
                timestamp=timestamp,
                metadata={
                    'probability': cp_prob,
                    'run_length': self.run_length,
                    'observation': new_observation
                }
            )
            alerts.append(alert)
        
        return alerts
    
    def get_change_history(self) -> List[Dict[str, Any]]:
        """Get change point history."""
        return self.change_history.copy()
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current BOCPD state."""
        return {
            'run_length': self.run_length,
            'last_change_time': self.last_change_time,
            'hazard': self.hazard,
            'change_threshold': self.change_threshold
        }


class PerformanceDashboard:
    """Performance monitoring dashboard."""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        self.metrics_history = []
        self.dashboard_data = {}
    
    def update_metrics(self, metrics: PerformanceMetrics):
        """Update performance metrics."""
        self.metrics_history.append(metrics)
        
        # Keep only recent history (last 1000 points)
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]
    
    def generate_dashboard(self) -> Dict[str, Any]:
        """Generate dashboard data."""
        if not self.metrics_history:
            return {'error': 'No metrics available'}
        
        # Extract time series data
        timestamps = [m.timestamp for m in self.metrics_history]
        ics = [m.ic for m in self.metrics_history if not pd.isna(m.ic)]
        sharpes = [m.sharpe for m in self.metrics_history if not pd.isna(m.sharpe)]
        drawdowns = [m.max_drawdown for m in self.metrics_history if not pd.isna(m.max_drawdown)]
        feature_counts = [m.feature_count for m in self.metrics_history]
        
        # Calculate summary statistics
        summary_stats = {
            'current_ic': ics[-1] if ics else 0.0,
            'avg_ic': np.mean(ics) if ics else 0.0,
            'ic_std': np.std(ics) if ics else 0.0,
            'current_sharpe': sharpes[-1] if sharpes else 0.0,
            'avg_sharpe': np.mean(sharpes) if sharpes else 0.0,
            'current_drawdown': drawdowns[-1] if drawdowns else 0.0,
            'max_drawdown': min(drawdowns) if drawdowns else 0.0,
            'current_feature_count': feature_counts[-1] if feature_counts else 0,
            'avg_feature_count': np.mean(feature_counts) if feature_counts else 0
        }
        
        # Create time series data
        time_series_data = {
            'timestamps': timestamps,
            'ics': ics,
            'sharpes': sharpes,
            'drawdowns': drawdowns,
            'feature_counts': feature_counts
        }
        
        # Generate plots if Plotly is available
        plots = {}
        if PLOTLY_AVAILABLE:
            plots = self._generate_plots(time_series_data)
        
        dashboard_data = {
            'summary_stats': summary_stats,
            'time_series_data': time_series_data,
            'plots': plots,
            'last_updated': datetime.now().isoformat()
        }
        
        return dashboard_data
    
    def _generate_plots(self, time_series_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate Plotly plots."""
        plots = {}
        
        try:
            # IC plot
            ic_fig = go.Figure()
            ic_fig.add_trace(go.Scatter(
                x=time_series_data['timestamps'],
                y=time_series_data['ics'],
                mode='lines',
                name='IC',
                line=dict(color='blue')
            ))
            ic_fig.update_layout(
                title='Information Coefficient Over Time',
                xaxis_title='Time',
                yaxis_title='IC'
            )
            plots['ic_plot'] = ic_fig.to_json()
            
            # Sharpe ratio plot
            sharpe_fig = go.Figure()
            sharpe_fig.add_trace(go.Scatter(
                x=time_series_data['timestamps'],
                y=time_series_data['sharpes'],
                mode='lines',
                name='Sharpe Ratio',
                line=dict(color='green')
            ))
            sharpe_fig.update_layout(
                title='Sharpe Ratio Over Time',
                xaxis_title='Time',
                yaxis_title='Sharpe Ratio'
            )
            plots['sharpe_plot'] = sharpe_fig.to_json()
            
            # Feature count plot
            feature_fig = go.Figure()
            feature_fig.add_trace(go.Scatter(
                x=time_series_data['timestamps'],
                y=time_series_data['feature_counts'],
                mode='lines',
                name='Feature Count',
                line=dict(color='orange')
            ))
            feature_fig.update_layout(
                title='Feature Count Over Time',
                xaxis_title='Time',
                yaxis_title='Feature Count'
            )
            plots['feature_count_plot'] = feature_fig.to_json()
            
        except Exception as e:
            self.logger.warning(f"Failed to generate plots: {e}")
            plots = {'error': str(e)}
        
        return plots


class AlertSystem:
    """Alert system for monitoring."""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        self.alerts = []
        self.alert_rules = self._create_alert_rules()
    
    def _create_alert_rules(self) -> List[Dict[str, Any]]:
        """Create alert rules."""
        return [
            {
                'name': 'low_ic_alert',
                'condition': lambda m: m.ic < 0.05,
                'severity': 'medium',
                'message': 'IC below 0.05'
            },
            {
                'name': 'high_drawdown_alert',
                'condition': lambda m: m.max_drawdown < -0.1,
                'severity': 'high',
                'message': 'Maximum drawdown exceeds 10%'
            },
            {
                'name': 'low_sharpe_alert',
                'condition': lambda m: m.sharpe < 0.5,
                'severity': 'medium',
                'message': 'Sharpe ratio below 0.5'
            },
            {
                'name': 'high_feature_count_alert',
                'condition': lambda m: m.feature_count > 150,
                'severity': 'low',
                'message': 'Feature count exceeds 150'
            }
        ]
    
    def check_alerts(self, metrics: PerformanceMetrics) -> List[MonitoringAlert]:
        """Check for alerts based on metrics."""
        alerts = []
        
        for rule in self.alert_rules:
            try:
                if rule['condition'](metrics):
                    alert = MonitoringAlert(
                        alert_type=rule['name'],
                        severity=rule['severity'],
                        message=rule['message'],
                        timestamp=metrics.timestamp,
                        metadata={
                            'ic': metrics.ic,
                            'sharpe': metrics.sharpe,
                            'max_drawdown': metrics.max_drawdown,
                            'feature_count': metrics.feature_count
                        }
                    )
                    alerts.append(alert)
            except Exception as e:
                self.logger.warning(f"Alert rule {rule['name']} failed: {e}")
                continue
        
        return alerts
    
    def add_alert(self, alert: MonitoringAlert):
        """Add an alert to the system."""
        self.alerts.append(alert)
        
        # Keep only recent alerts (last 1000)
        if len(self.alerts) > 1000:
            self.alerts = self.alerts[-1000:]
    
    def get_recent_alerts(self, hours: int = 24) -> List[MonitoringAlert]:
        """Get recent alerts."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [a for a in self.alerts if a.timestamp > cutoff_time]
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary."""
        severity_counts = {}
        for alert in self.alerts:
            severity = alert.severity
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            'total_alerts': len(self.alerts),
            'severity_counts': severity_counts,
            'recent_alerts': len(self.get_recent_alerts(24))
        }


class RetrainingTrigger:
    """Automated retraining triggers."""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        self.retraining_rules = self._create_retraining_rules()
        self.last_retraining = None
    
    def _create_retraining_rules(self) -> List[Dict[str, Any]]:
        """Create retraining rules."""
        return [
            {
                'name': 'performance_degradation',
                'condition': lambda m, h: self._check_performance_degradation(m, h),
                'message': 'Performance degradation detected'
            },
            {
                'name': 'regime_change',
                'condition': lambda m, h: self._check_regime_change(m, h),
                'message': 'Regime change detected'
            },
            {
                'name': 'time_based',
                'condition': lambda m, h: self._check_time_based(m, h),
                'message': 'Time-based retraining trigger'
            }
        ]
    
    def _check_performance_degradation(self, 
                                     current_metrics: PerformanceMetrics,
                                     history: List[PerformanceMetrics]) -> bool:
        """Check for performance degradation."""
        if len(history) < 10:
            return False
        
        # Compare current IC with recent average
        recent_ics = [m.ic for m in history[-10:] if not pd.isna(m.ic)]
        if not recent_ics:
            return False
        
        recent_avg_ic = np.mean(recent_ics)
        current_ic = current_metrics.ic
        
        # Trigger if current IC is significantly below recent average
        return current_ic < recent_avg_ic - 0.05
    
    def _check_regime_change(self, 
                           current_metrics: PerformanceMetrics,
                           history: List[PerformanceMetrics]) -> bool:
        """Check for regime change."""
        # This would integrate with BOCPD results
        # For now, return False
        return False
    
    def _check_time_based(self, 
                        current_metrics: PerformanceMetrics,
                        history: List[PerformanceMetrics]) -> bool:
        """Check for time-based retraining."""
        if self.last_retraining is None:
            return True
        
        # Retrain every 7 days
        time_since_retraining = current_metrics.timestamp - self.last_retraining
        return time_since_retraining.days >= 7
    
    def check_retraining_triggers(self, 
                                current_metrics: PerformanceMetrics,
                                history: List[PerformanceMetrics]) -> List[Dict[str, Any]]:
        """Check for retraining triggers."""
        triggers = []
        
        for rule in self.retraining_rules:
            try:
                if rule['condition'](current_metrics, history):
                    trigger = {
                        'rule_name': rule['name'],
                        'message': rule['message'],
                        'timestamp': current_metrics.timestamp,
                        'metrics': {
                            'ic': current_metrics.ic,
                            'sharpe': current_metrics.sharpe,
                            'max_drawdown': current_metrics.max_drawdown
                        }
                    }
                    triggers.append(trigger)
            except Exception as e:
                self.logger.warning(f"Retraining rule {rule['name']} failed: {e}")
                continue
        
        return triggers
    
    def mark_retraining_completed(self, timestamp: datetime):
        """Mark retraining as completed."""
        self.last_retraining = timestamp


class MonitoringSystem:
    """Main monitoring and automation system."""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        self.penalty_learner = AdaptivePenaltyLearner(config)
        self.bocpd_trigger = BOCPDTrigger(config)
        self.dashboard = PerformanceDashboard(config)
        self.alert_system = AlertSystem(config)
        self.retraining_trigger = RetrainingTrigger(config)
        
        self.system_state = None
        self.monitoring_enabled = True
    
    def setup_monitoring(self,
                        selection_result: Optional[SelectionResult],
                        final_features: List[str],
                        evaluation_results: Optional[Any],
                        regime_segments: Optional[Dict[str, Any]] = None):
        """Setup monitoring system.

        Args:
            selection_result: Metadata returned from statistical selection.
            final_features: Plain list of feature names for downstream consumers.
            evaluation_results: Results from the evaluation stage (if available).
            regime_segments: Regime segmentation details for contextual monitoring.
        """
        self.logger.info("Setting up monitoring system")

        # Initialize system state
        self.system_state = SystemState(
            timestamp=datetime.now(),
            pipeline_state={
                'selection_result': selection_result,
                'final_features': final_features,
                'evaluation_results': evaluation_results,
                'regime_segments': regime_segments,
            },
            performance_metrics=[],
            alerts=[],
            penalty_parameters=self.penalty_learner.get_penalty_parameters(),
            bocpd_state=self.bocpd_trigger.get_current_state(),
            metadata={'monitoring_enabled': True}
        )
        
        self.logger.info("Monitoring system setup completed")
    
    def update_monitoring(self, 
                         new_metrics: PerformanceMetrics,
                         market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Update monitoring with new metrics."""
        if not self.monitoring_enabled:
            return {'status': 'monitoring_disabled'}
        
        # Update dashboard
        self.dashboard.update_metrics(new_metrics)
        
        # Check alerts
        alerts = self.alert_system.check_alerts(new_metrics)
        for alert in alerts:
            self.alert_system.add_alert(alert)
        
        # Check BOCPD triggers
        bocpd_alerts = self.bocpd_trigger.check_for_triggers(
            new_metrics.ic, new_metrics.timestamp
        )
        for alert in bocpd_alerts:
            self.alert_system.add_alert(alert)
        
        # Update penalty parameters
        if self.config.adaptive_penalties:
            self.penalty_learner.update_penalties(
                [new_metrics], market_conditions
            )
        
        # Check retraining triggers
        retraining_triggers = self.retraining_trigger.check_retraining_triggers(
            new_metrics, self.dashboard.metrics_history
        )
        
        # Update system state
        if self.system_state:
            self.system_state.performance_metrics.append(new_metrics)
            self.system_state.alerts.extend(alerts)
            self.system_state.alerts.extend(bocpd_alerts)
            self.system_state.penalty_parameters = self.penalty_learner.get_penalty_parameters()
            self.system_state.bocpd_state = self.bocpd_trigger.get_current_state()
            self.system_state.timestamp = datetime.now()
        
        return {
            'status': 'updated',
            'alerts_generated': len(alerts) + len(bocpd_alerts),
            'retraining_triggers': retraining_triggers,
            'penalty_parameters': self.penalty_learner.get_penalty_parameters()
        }
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get dashboard data."""
        return self.dashboard.generate_dashboard()
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary."""
        return self.alert_system.get_alert_summary()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status."""
        if not self.system_state:
            return {'status': 'not_initialized'}
        
        return {
            'status': 'active' if self.monitoring_enabled else 'disabled',
            'last_update': self.system_state.timestamp.isoformat(),
            'total_metrics': len(self.system_state.performance_metrics),
            'total_alerts': len(self.system_state.alerts),
            'penalty_parameters': self.system_state.penalty_parameters,
            'bocpd_state': self.system_state.bocpd_state
        }
    
    def save_system_state(self, filepath: str):
        """Save system state to disk."""
        if not self.system_state:
            return
        
        state_data = {
            'timestamp': self.system_state.timestamp.isoformat(),
            'pipeline_state': self.system_state.pipeline_state,
            'performance_metrics': [
                {
                    'timestamp': m.timestamp.isoformat(),
                    'ic': m.ic,
                    'ic_std': m.ic_std,
                    'sharpe': m.sharpe,
                    'max_drawdown': m.max_drawdown,
                    'feature_count': m.feature_count,
                    'regime': m.regime,
                    'metadata': m.metadata
                }
                for m in self.system_state.performance_metrics
            ],
            'alerts': [
                {
                    'alert_type': a.alert_type,
                    'severity': a.severity,
                    'message': a.message,
                    'timestamp': a.timestamp.isoformat(),
                    'metadata': a.metadata
                }
                for a in self.system_state.alerts
            ],
            'penalty_parameters': self.system_state.penalty_parameters,
            'bocpd_state': self.system_state.bocpd_state,
            'metadata': self.system_state.metadata
        }
        
        with open(filepath, 'w') as f:
            json.dump(state_data, f, indent=2)
    
    def load_system_state(self, filepath: str):
        """Load system state from disk."""
        with open(filepath, 'r') as f:
            state_data = json.load(f)
        
        # Reconstruct performance metrics
        performance_metrics = []
        for m_data in state_data.get('performance_metrics', []):
            metrics = PerformanceMetrics(
                timestamp=datetime.fromisoformat(m_data['timestamp']),
                ic=m_data['ic'],
                ic_std=m_data['ic_std'],
                sharpe=m_data['sharpe'],
                max_drawdown=m_data['max_drawdown'],
                feature_count=m_data['feature_count'],
                regime=m_data['regime'],
                metadata=m_data['metadata']
            )
            performance_metrics.append(metrics)
        
        # Reconstruct alerts
        alerts = []
        for a_data in state_data.get('alerts', []):
            alert = MonitoringAlert(
                alert_type=a_data['alert_type'],
                severity=a_data['severity'],
                message=a_data['message'],
                timestamp=datetime.fromisoformat(a_data['timestamp']),
                metadata=a_data['metadata']
            )
            alerts.append(alert)
        
        # Reconstruct system state
        self.system_state = SystemState(
            timestamp=datetime.fromisoformat(state_data['timestamp']),
            pipeline_state=state_data['pipeline_state'],
            performance_metrics=performance_metrics,
            alerts=alerts,
            penalty_parameters=state_data['penalty_parameters'],
            bocpd_state=state_data['bocpd_state'],
            metadata=state_data['metadata']
        )
        
        # Update components
        self.dashboard.metrics_history = performance_metrics
        self.alert_system.alerts = alerts
        self.penalty_learner.penalty_parameters = state_data['penalty_parameters']
    
    def enable_monitoring(self):
        """Enable monitoring."""
        self.monitoring_enabled = True
        self.logger.info("Monitoring enabled")
    
    def disable_monitoring(self):
        """Disable monitoring."""
        self.monitoring_enabled = False
        self.logger.info("Monitoring disabled")