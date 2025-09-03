from __future__ import annotations
'Performance tracking and monitoring for fractional implementations.'
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import pandas as pd
from src.config.fractional_implementations_config import FractionalImplementationsConfig
from src.utils.logger import get_logger
from copy import copy

class FractionalPerformanceTracker:
    """Comprehensive performance tracking for fractional implementations."""

    def __init__(self, config: FractionalImplementationsConfig, output_dir: str='data/fractional_performance') -> None:
        """Initialize performance tracker.

        Args:
            config: Fractional implementations configuration
            output_dir: Directory to store performance data
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger('FractionalPerformanceTracker')
        self.baseline_metrics = {}
        self.current_metrics = {}
        self.historical_metrics = []
        self.performance_alerts = []
        self.start_time = datetime.now()
        self.last_check = None
        self.check_count = 0
        self._initialize_tracking()

    def _initialize_tracking(self) -> None:
        """Initialize performance tracking."""
        self.logger.info('Initializing fractional performance tracking')
        self.metrics_file = self.output_dir / 'performance_metrics.json'
        self.alerts_file = self.output_dir / 'performance_alerts.json'
        self.dashboard_file = self.output_dir / 'performance_dashboard.html'
        self._load_existing_data()
        self._create_dashboard()

    def _load_existing_data(self) -> None:
        """Load existing performance data."""
        try:
            if self.metrics_file.exists():
                with open(self.metrics_file) as f:
                    data = json.load(f)
                    self.baseline_metrics = data.get('baseline', {})
                    self.current_metrics = data.get('current', {})
                    self.historical_metrics = data.get('historical', [])
            if self.alerts_file.exists():
                with open(self.alerts_file) as f:
                    self.performance_alerts = json.load(f)
        except Exception as e:
            self.logger.warning(f'Could not load existing performance data: {e}')

    def set_baseline_metrics(self, metrics: dict[str, Any]) -> None:
        """Set baseline performance metrics.

        Args:
            metrics: Dictionary of baseline metrics
        """
        self.baseline_metrics = metrics.copy()
        self.logger.info(f'Set baseline metrics: {list(metrics.keys())}')
        self._save_metrics()

    def update_current_metrics(self, metrics: dict[str, Any]) -> None:
        """Update current performance metrics.

        Args:
            metrics: Dictionary of current metrics
        """
        self.current_metrics = metrics.copy()
        self.check_count += 1
        self.current_metrics['timestamp'] = datetime.now().isoformat()
        self.current_metrics['check_count'] = self.check_count
        self.historical_metrics.append(self.current_metrics.copy())
        max_history = 1000
        if len(self.historical_metrics) > max_history:
            self.historical_metrics = self.historical_metrics[-max_history:]
        self._check_performance_alerts()
        self._save_metrics()
        self._create_dashboard()
        self.last_check = datetime.now()

    def _check_performance_alerts(self) -> None:
        """Check for performance alerts."""
        if not self.baseline_metrics or not self.current_metrics:
            return
        alerts = []
        key_metrics = ['sharpe_ratio', 'max_drawdown', 'win_rate', 'profit_factor']
        for metric in key_metrics:
            if metric in self.baseline_metrics and metric in self.current_metrics:
                baseline = self.baseline_metrics[metric]
                current = self.current_metrics[metric]
                if baseline != 0:
                    degradation = (baseline - current) / abs(baseline)
                    if degradation > self.config.alert_threshold:
                        alert = {'timestamp': datetime.now().isoformat(), 'metric': metric, 'baseline': baseline, 'current': current, 'degradation': degradation, 'severity': 'high' if degradation > 0.1 else 'medium'}
                        alerts.append(alert)
        self.performance_alerts.extend(alerts)
        for alert in alerts:
            self.logger.warning(f"Performance alert: {alert['metric']} degraded by {alert['degradation']:.2%} (baseline: {alert['baseline']:.4f}, current: {alert['current']:.4f})")

    def _save_metrics(self) -> None:
        """Save performance metrics to file."""
        try:
            data = {'baseline': self.baseline_metrics, 'current': self.current_metrics, 'historical': self.historical_metrics, 'last_updated': datetime.now().isoformat()}
            with open(self.metrics_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.exception(f'Failed to save metrics: {e}')

    def _create_dashboard(self) -> None:
        """Create performance dashboard."""
        try:
            if not self.historical_metrics:
                return
            df = pd.DataFrame(self.historical_metrics)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Fractional Implementations Performance Dashboard', fontsize=16)
            if 'sharpe_ratio' in df.columns:
                axes[0, 0].plot(df['timestamp'], df['sharpe_ratio'], label='Current')
                if self.baseline_metrics.get('sharpe_ratio'):
                    axes[0, 0].axhline(y=self.baseline_metrics['sharpe_ratio'], color='r', linestyle='--', label='Baseline')
                axes[0, 0].set_title('Sharpe Ratio')
                axes[0, 0].legend()
                axes[0, 0].tick_params(axis='x', rotation=45)
            if 'max_drawdown' in df.columns:
                axes[0, 1].plot(df['timestamp'], df['max_drawdown'], label='Current')
                if self.baseline_metrics.get('max_drawdown'):
                    axes[0, 1].axhline(y=self.baseline_metrics['max_drawdown'], color='r', linestyle='--', label='Baseline')
                axes[0, 1].set_title('Maximum Drawdown')
                axes[0, 1].legend()
                axes[0, 1].tick_params(axis='x', rotation=45)
            if 'win_rate' in df.columns:
                axes[1, 0].plot(df['timestamp'], df['win_rate'], label='Current')
                if self.baseline_metrics.get('win_rate'):
                    axes[1, 0].axhline(y=self.baseline_metrics['win_rate'], color='r', linestyle='--', label='Baseline')
                axes[1, 0].set_title('Win Rate')
                axes[1, 0].legend()
                axes[1, 0].tick_params(axis='x', rotation=45)
            if 'profit_factor' in df.columns:
                axes[1, 1].plot(df['timestamp'], df['profit_factor'], label='Current')
                if self.baseline_metrics.get('profit_factor'):
                    axes[1, 1].axhline(y=self.baseline_metrics['profit_factor'], color='r', linestyle='--', label='Baseline')
                axes[1, 1].set_title('Profit Factor')
                axes[1, 1].legend()
                axes[1, 1].tick_params(axis='x', rotation=45)
            plt.tight_layout()
            plot_file = self.output_dir / 'performance_dashboard.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            self._create_html_dashboard(df)
        except Exception as e:
            self.logger.exception(f'Failed to create dashboard: {e}')

    def _create_html_dashboard(self, df: pd.DataFrame) -> None:
        """Create HTML dashboard."""
        try:
            html_content = f"""\n            <!DOCTYPE html>\n            <html>\n            <head>\n                <title>Fractional Implementations Performance Dashboard</title>\n                <style>\n                    body {{ font-family: Arial, sans-serif; margin: 20px; }}\n                    .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}\n                    .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}\n                    .metric-card {{ background-color: #fff; border: 1px solid #ddd; padding: 15px; border-radius: 5px; }}\n                    .metric-value {{ font-size: 24px; font-weight: bold; }}\n                    .metric-label {{ color: #666; margin-bottom: 5px; }}\n                    .improvement {{ color: green; }}\n                    .degradation {{ color: red; }}\n                    .chart {{ margin: 20px 0; }}\n                </style>\n            </head>\n            <body>\n                <div class="header">\n                    <h1>Fractional Implementations Performance Dashboard</h1>\n                    <p>Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>\n                    <p>Total checks: {self.check_count}</p>\n                </div>\n\n                <div class="metrics">\n            """
            if self.current_metrics:
                for metric, value in self.current_metrics.items():
                    if metric not in ['timestamp', 'check_count']:
                        baseline = self.baseline_metrics.get(metric, 0)
                        if baseline != 0:
                            change = (value - baseline) / abs(baseline)
                            change_class = 'improvement' if change > 0 else 'degradation'
                            change_text = f'({change:+.2%})'
                        else:
                            change_text = ''
                            change_class = ''
                        html_content += f'''\n                        <div class="metric-card">\n                            <div class="metric-label">{metric.replace('_', ' ').title()}</div>\n                            <div class="metric-value">{value:.4f} <span class="{change_class}">{change_text}</span></div>\n                        </div>\n                        '''
            html_content += '\n                </div>\n\n                <div class="chart">\n                    <img src="performance_dashboard.png" alt="Performance Charts" style="width: 100%; max-width: 1200px;">\n                </div>\n            </body>\n            </html>\n            '
            with open(self.dashboard_file, 'w') as f:
                f.write(html_content)
        except Exception as e:
            self.logger.exception(f'Failed to create HTML dashboard: {e}')

    def get_performance_summary(self) -> dict[str, Any]:
        """Get performance summary.

        Returns:
            Dictionary with performance summary
        """
        if not self.baseline_metrics or not self.current_metrics:
            return {}
        summary = {'baseline_metrics': self.baseline_metrics.copy(), 'current_metrics': self.current_metrics.copy(), 'improvements': {}, 'degradations': {}, 'overall_status': 'stable'}
        for metric in self.baseline_metrics:
            if metric in self.current_metrics:
                baseline = self.baseline_metrics[metric]
                current = self.current_metrics[metric]
                if baseline != 0:
                    change = (current - baseline) / abs(baseline)
                    if change > 0:
                        summary['improvements'][metric] = change
                    elif change < 0:
                        summary['degradations'][metric] = change
        if summary['degradations']:
            max_degradation = max(summary['degradations'].values())
            if max_degradation < -0.1:
                summary['overall_status'] = 'critical'
            elif max_degradation < -0.05:
                summary['overall_status'] = 'warning'
        return summary

    def get_alert_summary(self) -> list[dict[str, Any]]:
        """Get recent performance alerts.

        Returns:
            List of recent alerts
        """
        cutoff_time = datetime.now() - timedelta(hours=24)
        recent_alerts = []
        for alert in self.performance_alerts:
            alert_time = datetime.fromisoformat(alert['timestamp'])
            if alert_time > cutoff_time:
                recent_alerts.append(alert)
        return recent_alerts

    def export_performance_report(self, output_file: str=None) -> str:
        """Export comprehensive performance report.

        Args:
            output_file: Output file path (optional)

        Returns:
            Path to exported report
        """
        if output_file is None:
            output_file = self.output_dir / f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report = {'report_generated': datetime.now().isoformat(), 'tracking_started': self.start_time.isoformat(), 'total_checks': self.check_count, 'baseline_metrics': self.baseline_metrics, 'current_metrics': self.current_metrics, 'performance_summary': self.get_performance_summary(), 'recent_alerts': self.get_alert_summary(), 'historical_metrics_count': len(self.historical_metrics), 'configuration': {'current_phase': self.config.current_phase, 'target_improvements': {'sharpe_ratio': self.config.target_sharpe_improvement, 'drawdown_reduction': self.config.target_drawdown_reduction, 'accuracy_improvement': self.config.target_accuracy_improvement}}}
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        self.logger.info(f'Performance report exported to: {output_file}')
        return str(output_file)