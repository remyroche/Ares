# src/monitoring/fractional_performance_tracker.py

"""Performance tracking and monitoring for fractional implementations."""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import matplotlib.pyplot as plt

from src.utils.logger import get_logger
from src.config.fractional_implementations_config import FractionalImplementationsConfig


class FractionalPerformanceTracker:
    """Comprehensive performance tracking for fractional implementations."""

    def __init__(self, config: FractionalImplementationsConfig, output_dir: str = "data/fractional_performance"):
        """Initialize performance tracker.

        Args:
            config: Fractional implementations configuration
            output_dir: Directory to store performance data
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger = get_logger("FractionalPerformanceTracker")

        # Performance storage
        self.baseline_metrics = {}
        self.current_metrics = {}
        self.historical_metrics = []
        self.performance_alerts = []

        # Tracking state
        self.start_time = datetime.now()
        self.last_check = None
        self.check_count = 0

        # Initialize tracking
        self._initialize_tracking()

    def _load_existing_data(self):
        """Load existing performance data."""
        try:
            if self.metrics_file.exists():
                with open(self.metrics_file, 'r') as f:
                    data = json.load(f)
                    self.baseline_metrics = data.get('baseline', {})
                    self.current_metrics = data.get('current', {})
                    self.historical_metrics = data.get('historical', [])

            if self.alerts_file.exists():
                with open(self.alerts_file, 'r') as f:
                    self.performance_alerts = json.load(f)
        except Exception as e:
            self.logger.warning(f"Could not load existing performance data: {e}")

    def _check_performance_alerts(self):
        """Check for performance alerts."""
        if not self.baseline_metrics or not self.current_metrics:
            return

        alerts = []

        # Check key metrics
        key_metrics = ['sharpe_ratio', 'max_drawdown', 'win_rate', 'profit_factor']

        for metric in key_metrics:
            if metric in self.baseline_metrics and metric in self.current_metrics:
                baseline = self.baseline_metrics[metric]
                current = self.current_metrics[metric]

                # Calculate degradation
                if baseline != 0:
                    degradation = (baseline - current) / abs(baseline)

                    if degradation > self.config.alert_threshold:
                        alert = {
                            'timestamp': datetime.now().isoformat(),
                            'metric': metric,
                            'baseline': baseline,
                            'current': current,
                            'degradation': degradation,
                            'severity': 'high' if degradation > 0.1 else 'medium'
                        }
                        alerts.append(alert)

        # Add new alerts
        self.performance_alerts.extend(alerts)

        # Log alerts
        for alert in alerts:
            self.logger.warning(
                f"Performance alert: {alert['metric']} degraded by "
                f"{alert['degradation']:.2%} (baseline: {alert['baseline']:.4f}, "
                f"current: {alert['current']:.4f})"
            )

    def _save_metrics(self):
        """Save performance metrics to file."""
        try:
            data = {
                'baseline': self.baseline_metrics,
                'current': self.current_metrics,
                'historical': self.historical_metrics,
                'last_updated': datetime.now().isoformat()
            }

            with open(self.metrics_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save metrics: {e}")

    def _create_dashboard(self):
        """Create performance dashboard."""
        try:
            if not self.historical_metrics:
                return

            # Convert to DataFrame
            df = pd.DataFrame(self.historical_metrics)
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Create plots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Fractional Implementations Performance Dashboard', fontsize=16)

            # Sharpe Ratio
            if 'sharpe_ratio' in df.columns:
                axes[0, 0].plot(df['timestamp'], df['sharpe_ratio'], label='Current')
                if self.baseline_metrics.get('sharpe_ratio'):
                    axes[0, 0].axhline(y=self.baseline_metrics['sharpe_ratio'],
                                      color='r', linestyle='--', label='Baseline')
                axes[0, 0].set_title('Sharpe Ratio')
                axes[0, 0].legend()
                axes[0, 0].tick_params(axis='x', rotation=45)

            # Maximum Drawdown
            if 'max_drawdown' in df.columns:
                axes[0, 1].plot(df['timestamp'], df['max_drawdown'], label='Current')
                if self.baseline_metrics.get('max_drawdown'):
                    axes[0, 1].axhline(y=self.baseline_metrics['max_drawdown'],
                                      color='r', linestyle='--', label='Baseline')
                axes[0, 1].set_title('Maximum Drawdown')
                axes[0, 1].legend()
                axes[0, 1].tick_params(axis='x', rotation=45)

            # Win Rate
            if 'win_rate' in df.columns:
                axes[1, 0].plot(df['timestamp'], df['win_rate'], label='Current')
                if self.baseline_metrics.get('win_rate'):
                    axes[1, 0].axhline(y=self.baseline_metrics['win_rate'],
                                      color='r', linestyle='--', label='Baseline')
                axes[1, 0].set_title('Win Rate')
                axes[1, 0].legend()
                axes[1, 0].tick_params(axis='x', rotation=45)

            # Profit Factor
            if 'profit_factor' in df.columns:
                axes[1, 1].plot(df['timestamp'], df['profit_factor'], label='Current')
                if self.baseline_metrics.get('profit_factor'):
                    axes[1, 1].axhline(y=self.baseline_metrics['profit_factor'],
                                      color='r', linestyle='--', label='Baseline')
                axes[1, 1].set_title('Profit Factor')
                axes[1, 1].legend()
                axes[1, 1].tick_params(axis='x', rotation=45)

            plt.tight_layout()

            # Save plot
            plot_file = self.output_dir / "performance_dashboard.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()

            # Create HTML dashboard
            self._create_html_dashboard(df)

        except Exception as e:
            self.logger.error(f"Failed to create dashboard: {e}")

    def _create_html_dashboard(self, df: pd.DataFrame):
        """Create HTML dashboard."""
        try:
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Fractional Implementations Performance Dashboard</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                    .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
                    .metric-card {{ background-color: #fff; border: 1px solid #ddd; padding: 15px; border-radius: 5px; }}
                    .metric-value {{ font-size: 24px; font-weight: bold; }}
                    .metric-label {{ color: #666; margin-bottom: 5px; }}
                    .improvement {{ color: green; }}
                    .degradation {{ color: red; }}
                    .chart {{ margin: 20px 0; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>Fractional Implementations Performance Dashboard</h1>
                    <p>Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p>Total checks: {self.check_count}</p>
                </div>

                <div class="metrics">
            """

            # Add current metrics
            if self.current_metrics:
                for metric, value in self.current_metrics.items():
                    if metric not in ['timestamp', 'check_count']:
                        baseline = self.baseline_metrics.get(metric, 0)
                        if baseline != 0:
                            change = (value - baseline) / abs(baseline)
                            change_class = 'improvement' if change > 0 else 'degradation'
                            change_text = f"({change:+.2%})"
                        else:
                            change_text = ""
                            change_class = ""

                        html_content += f"""
                        <div class="metric-card">
                            <div class="metric-label">{metric.replace('_', ' ').title()}</div>
                            <div class="metric-value">{value:.4f} <span class="{change_class}">{change_text}</span></div>
                        </div>
                        """

            html_content += """
                </div>

                <div class="chart">
                    <img src="performance_dashboard.png" alt="Performance Charts" style="width: 100%; max-width: 1200px;">
                </div>
            </body>
            </html>
            """

            with open(self.dashboard_file, 'w') as f:
                f.write(html_content)

        except Exception as e:
            self.logger.error(f"Failed to create HTML dashboard: {e}")
