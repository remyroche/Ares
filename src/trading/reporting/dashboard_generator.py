"""
Dashboard Generator

Real-time dashboard generation for trading operations with
live metrics, charts, and detailed trade analysis.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
import json
from pathlib import Path
import pandas as pd
import numpy as np

from src.utils.logger import system_logger
from src.utils.tprint import tprint_info, tprint_warning, tprint_error, tprint_success
from ..monitoring.comprehensive_trade_monitor import DetailedTradeMetrics, TradingSessionMetrics
from ..utils.error_handling import TradingError, TradingErrorSeverity, trading_error_handler

logger = system_logger.getChild('DashboardGenerator')

class DashboardGenerator:
    """
    Real-time dashboard generator for trading operations.

    Creates interactive dashboards with:
    - Live performance metrics
    - Model performance tracking
    - Risk monitoring
    - Trade execution quality
    - SHAP/LIME explanations
    - Regime analysis
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logger.getChild('DashboardGenerator')

        # Dashboard configuration
        self.dashboard_directory = Path(self.config.get('dashboard_directory', 'trading_dashboards'))
        self.update_interval = self.config.get('update_interval', 30)  # seconds
        self.enable_live_updates = self.config.get('enable_live_updates', True)

        # Dashboard state
        self.last_update = datetime.now()
        self.dashboard_data: Dict[str, Any] = {}

        # Ensure dashboard directory exists
        self.dashboard_directory.mkdir(parents=True, exist_ok=True)

    @trading_error_handler(
        error_types=(Exception,),
        severity=TradingErrorSeverity.LOW,
        raise_on_error=False
    )
    async def generate_live_dashboard(
        self,
        trades: List[DetailedTradeMetrics],
        session_metrics: Optional[TradingSessionMetrics] = None,
        active_trades: Optional[Dict[str, DetailedTradeMetrics]] = None
    ) -> Dict[str, Any]:
        """
        Generate live trading dashboard.

        Args:
            trades: Completed trades
            session_metrics: Current session metrics
            active_trades: Currently active trades

        Returns:
            Dashboard data dictionary
        """
        try:
            tprint_info("📊 Generating live trading dashboard...")

            dashboard = {
                'dashboard_metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'update_interval': self.update_interval,
                    'dashboard_type': 'live_trading'
                },
                'live_metrics': await self._generate_live_metrics(trades, session_metrics),
                'performance_charts': await self._generate_performance_charts(trades),
                'model_dashboard': await self._generate_model_dashboard(trades),
                'risk_dashboard': await self._generate_risk_dashboard(trades),
                'regime_dashboard': await self._generate_regime_dashboard(trades),
                'active_trades_panel': await self._generate_active_trades_panel(active_trades or {}),
                'recent_trades_panel': await self._generate_recent_trades_panel(trades[-10:] if trades else [])
            }

            # Cache dashboard data
            self.dashboard_data = dashboard
            self.last_update = datetime.now()

            # Export dashboard
            await self._export_dashboard(dashboard, 'live_dashboard')

            tprint_success("✅ Generated live trading dashboard")

            return dashboard

        except Exception as e:
            tprint_error(f"❌ Failed to generate live dashboard: {e}")
            return {}

    async def _generate_live_metrics(
        self,
        trades: List[DetailedTradeMetrics],
        session_metrics: Optional[TradingSessionMetrics]
    ) -> Dict[str, Any]:
        """Generate real-time performance metrics."""
        try:
            current_time = datetime.now()

            # Basic metrics
            total_trades = len(trades)
            recent_trades = [t for t in trades if (current_time - t.timestamp).total_seconds() < 3600]  # Last hour

            # Performance metrics
            if trades:
                total_pnl = sum(t.pnl_absolute for t in trades if t.pnl_absolute is not None)
                recent_pnl = sum(t.pnl_absolute for t in recent_trades if t.pnl_absolute is not None)

                winning_trades = len([t for t in trades if t.pnl_absolute and t.pnl_absolute > 0])
                win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

                # Calculate current drawdown
                pnl_series = [t.pnl_absolute for t in trades if t.pnl_absolute is not None]
                if pnl_series:
                    cumulative_pnl = np.cumsum(pnl_series)
                    peak = np.maximum.accumulate(cumulative_pnl)
                    current_drawdown = (peak[-1] - cumulative_pnl[-1]) / peak[-1] if peak[-1] > 0 else 0.0
                else:
                    current_drawdown = 0.0
            else:
                total_pnl = recent_pnl = win_rate = current_drawdown = 0.0

            # Trading velocity
            if trades:
                # Sort trades by timestamp to ensure correct order
                sorted_trades = sorted(trades, key=lambda t: t.timestamp)
                trading_duration = (sorted_trades[-1].timestamp - sorted_trades[0].timestamp).total_seconds() / 3600  # hours
                trades_per_hour = total_trades / trading_duration if trading_duration > 0 else 0.0
            else:
                trades_per_hour = 0.0

            # Model activity
            active_models = set()
            for trade in recent_trades:
                active_models.update(trade.models_used.keys())

            return {
                'current_performance': {
                    'total_trades': total_trades,
                    'total_pnl': total_pnl,
                    'recent_pnl_1h': recent_pnl,
                    'win_rate': win_rate,
                    'current_drawdown': current_drawdown,
                    'trades_per_hour': trades_per_hour
                },
                'session_info': {
                    'session_id': session_metrics.session_id if session_metrics else 'unknown',
                    'session_duration_hours': (current_time - session_metrics.start_time).total_seconds() / 3600 if session_metrics else 0.0,
                    'session_pnl': session_metrics.total_pnl if session_metrics else 0.0
                },
                'model_activity': {
                    'active_models_count': len(active_models),
                    'active_models': list(active_models),
                    'recent_model_usage': await self._calculate_recent_model_usage(recent_trades)
                },
                'market_status': await self._get_current_market_status(trades)
            }

        except Exception as e:
            tprint_error(f"❌ Failed to generate live metrics: {e}")
            return {}

    async def _generate_performance_charts(self, trades: List[DetailedTradeMetrics]) -> Dict[str, Any]:
        """Generate data for performance charts."""
        try:
            if not trades:
                return {}

            # PnL curve data
            pnl_data = []
            cumulative_pnl = 0.0

            for trade in trades:
                if trade.pnl_absolute is not None:
                    cumulative_pnl += trade.pnl_absolute

                pnl_data.append({
                    'timestamp': trade.timestamp.isoformat(),
                    'trade_pnl': trade.pnl_absolute or 0.0,
                    'cumulative_pnl': cumulative_pnl
                })

            # Confidence vs Performance scatter
            confidence_performance = []
            for trade in trades:
                if trade.pnl_percentage is not None:
                    confidence_performance.append({
                        'confidence': trade.signal_confidence,
                        'pnl_percentage': trade.pnl_percentage,
                        'trade_id': trade.trade_id
                    })

            # Model performance over time
            model_performance_timeline = await self._generate_model_performance_timeline(trades)

            # Regime performance distribution
            regime_performance = {}
            for trade in trades:
                regime = trade.regime_type
                if regime not in regime_performance:
                    regime_performance[regime] = {'trades': 0, 'pnl': 0.0}

                regime_performance[regime]['trades'] += 1
                if trade.pnl_absolute:
                    regime_performance[regime]['pnl'] += trade.pnl_absolute

            return {
                'pnl_curve': pnl_data,
                'confidence_vs_performance': confidence_performance,
                'model_performance_timeline': model_performance_timeline,
                'regime_performance_distribution': regime_performance,
                'trade_frequency': await self._generate_trade_frequency_data(trades)
            }

        except Exception as e:
            tprint_error(f"❌ Failed to generate performance charts: {e}")
            return {}

    async def _generate_model_dashboard(self, trades: List[DetailedTradeMetrics]) -> Dict[str, Any]:
        """Generate model-specific dashboard data."""
        try:
            model_dashboard = {}

            # Collect all models
            all_models = set()
            for trade in trades:
                all_models.update(trade.models_used.keys())

            # Generate dashboard for each model
            for model_id in all_models:
                model_trades = [t for t in trades if model_id in t.models_used]

                if model_trades:
                    # Recent performance (last 24 hours)
                    recent_cutoff = datetime.now() - timedelta(hours=24)
                    recent_model_trades = [t for t in model_trades if t.timestamp >= recent_cutoff]

                    # Model metrics
                    model_pnl = [t.pnl_absolute for t in model_trades if t.pnl_absolute is not None]
                    recent_pnl = [t.pnl_absolute for t in recent_model_trades if t.pnl_absolute is not None]

                    # Confidence tracking
                    confidences = [t.model_confidences.get(model_id, 0.0) for t in model_trades]
                    recent_confidences = [t.model_confidences.get(model_id, 0.0) for t in recent_model_trades]
                    
                    # Confidence trend: compare recent average to historical average
                    confidence_trend = 'stable'
                    if len(recent_confidences) > 0 and len(confidences) > len(recent_confidences):
                        recent_avg = np.mean(recent_confidences)
                        historical_avg = np.mean(confidences[:-len(recent_confidences)])
                        if recent_avg > historical_avg + 0.05:  # 5% threshold
                            confidence_trend = 'improving'
                        elif recent_avg < historical_avg - 0.05:
                            confidence_trend = 'declining'

                    model_dashboard[model_id] = {
                        'usage_stats': {
                            'total_usage': len(model_trades),
                            'recent_usage_24h': len(recent_model_trades),
                            'usage_frequency': len(model_trades) / len(trades) if trades else 0.0
                        },
                        'performance_stats': {
                            'total_pnl': sum(model_pnl) if model_pnl else 0.0,
                            'recent_pnl_24h': sum(recent_pnl) if recent_pnl else 0.0,
                            'avg_pnl_per_trade': np.mean(model_pnl) if model_pnl else 0.0,
                            'win_rate': len([p for p in model_pnl if p > 0]) / len(model_pnl) if model_pnl else 0.0
                        },
                        'confidence_stats': {
                            'avg_confidence': np.mean(confidences) if confidences else 0.0,
                            'recent_avg_confidence': np.mean(recent_confidences) if recent_confidences else 0.0,
                            'confidence_trend': confidence_trend
                        },
                        'feature_importance': await self._get_model_feature_importance(model_id, model_trades)
                    }

            return model_dashboard

        except Exception as e:
            tprint_error(f"❌ Failed to generate model dashboard: {e}")
            return {}

    async def _generate_risk_dashboard(self, trades: List[DetailedTradeMetrics]) -> Dict[str, Any]:
        """Generate risk monitoring dashboard."""
        try:
            # Current risk levels
            recent_trades = [t for t in trades if (datetime.now() - t.timestamp).total_seconds() < 3600]

            current_risk = {
                'portfolio_risk': np.mean([t.portfolio_risk for t in recent_trades if t.portfolio_risk > 0]) if recent_trades else 0.0,
                'avg_leverage': np.mean([t.leverage for t in recent_trades if t.leverage > 0]) if recent_trades else 1.0,
                'avg_position_size': np.mean([t.position_size for t in recent_trades if t.position_size > 0]) if recent_trades else 0.0
            }

            # Risk alerts
            risk_alerts = []
            if current_risk['portfolio_risk'] > 0.05:  # 5% portfolio risk
                risk_alerts.append({
                    'type': 'high_portfolio_risk',
                    'message': f"High portfolio risk: {current_risk['portfolio_risk']:.2%}",
                    'severity': 'warning'
                })

            if current_risk['avg_leverage'] > 5.0:
                risk_alerts.append({
                    'type': 'high_leverage',
                    'message': f"High average leverage: {current_risk['avg_leverage']:.1f}x",
                    'severity': 'warning'
                })

            # Risk trend analysis
            risk_timeline = []
            for trade in trades[-50:]:  # Last 50 trades
                risk_timeline.append({
                    'timestamp': trade.timestamp.isoformat(),
                    'portfolio_risk': trade.portfolio_risk,
                    'leverage': trade.leverage,
                    'var_95': trade.var_95
                })

            return {
                'current_risk_levels': current_risk,
                'risk_alerts': risk_alerts,
                'risk_timeline': risk_timeline,
                'risk_limits': {
                    'max_portfolio_risk': 0.05,
                    'max_leverage': 10.0,
                    'max_position_size': 0.25,
                    'max_drawdown': 0.15
                }
            }

        except Exception as e:
            tprint_error(f"❌ Failed to generate risk dashboard: {e}")
            return {}

    async def _generate_regime_dashboard(self, trades: List[DetailedTradeMetrics]) -> Dict[str, Any]:
        """Generate regime analysis dashboard."""
        try:
            # Current regime distribution
            recent_trades = [t for t in trades if (datetime.now() - t.timestamp).total_seconds() < 3600]

            current_regimes = {}
            for trade in recent_trades:
                regime = trade.regime_type
                current_regimes[regime] = current_regimes.get(regime, 0) + 1

            # Regime performance summary
            regime_performance = {}
            for trade in trades:
                regime = trade.regime_type
                if regime not in regime_performance:
                    regime_performance[regime] = {
                        'total_trades': 0,
                        'total_pnl': 0.0,
                        'avg_confidence': 0.0,
                        'confidences': []
                    }

                regime_performance[regime]['total_trades'] += 1
                if trade.pnl_absolute:
                    regime_performance[regime]['total_pnl'] += trade.pnl_absolute
                regime_performance[regime]['confidences'].append(trade.regime_confidence)

            # Calculate averages
            for regime, data in regime_performance.items():
                if data['confidences']:
                    data['avg_confidence'] = np.mean(data['confidences'])
                    data['avg_pnl_per_trade'] = data['total_pnl'] / data['total_trades']
                del data['confidences']  # Remove for JSON serialization

            # Regime transition analysis
            regime_transitions = await self._analyze_regime_transitions(trades)

            return {
                'current_regime_distribution': current_regimes,
                'regime_performance_summary': regime_performance,
                'regime_transitions': regime_transitions,
                'best_performing_regime': max(regime_performance.items(), key=lambda x: x[1]['total_pnl']) if regime_performance else None,
                'most_stable_regime': max(regime_performance.items(), key=lambda x: x[1]['avg_confidence']) if regime_performance else None
            }

        except Exception as e:
            tprint_error(f"❌ Failed to generate regime dashboard: {e}")
            return {}

    async def _generate_active_trades_panel(self, active_trades: Dict[str, DetailedTradeMetrics]) -> Dict[str, Any]:
        """Generate active trades monitoring panel."""
        try:
            active_panel = {
                'active_trade_count': len(active_trades),
                'active_trades': []
            }

            for trade_id, trade in active_trades.items():
                # Calculate unrealized PnL (would need current market price)
                current_time = datetime.now()
                duration = (current_time - trade.timestamp).total_seconds() / 60  # minutes

                trade_info = {
                    'trade_id': trade_id,
                    'symbol': trade.symbol,
                    'action': trade.action,
                    'entry_price': trade.price,
                    'quantity': trade.quantity,
                    'duration_minutes': duration,
                    'confidence': trade.signal_confidence,
                    'regime': trade.regime_type,
                    'position_size': trade.position_size,
                    'leverage': trade.leverage,
                    'unrealized_pnl': 0.0  # Would calculate with current market price
                }

                active_panel['active_trades'].append(trade_info)

            # Sort by duration (oldest first)
            active_panel['active_trades'].sort(key=lambda x: x['duration_minutes'], reverse=True)

            return active_panel

        except Exception as e:
            tprint_error(f"❌ Failed to generate active trades panel: {e}")
            return {}

    async def _generate_recent_trades_panel(self, recent_trades: List[DetailedTradeMetrics]) -> Dict[str, Any]:
        """Generate recent trades panel."""
        try:
            recent_panel = {
                'recent_trade_count': len(recent_trades),
                'recent_trades': []
            }

            for trade in recent_trades:
                trade_info = {
                    'trade_id': trade.trade_id,
                    'timestamp': trade.timestamp.strftime('%H:%M:%S'),
                    'symbol': trade.symbol,
                    'action': trade.action,
                    'price': trade.price,
                    'quantity': trade.quantity,
                    'pnl': trade.pnl_absolute or 0.0,
                    'confidence': trade.signal_confidence,
                    'regime': trade.regime_type,
                    'models_used': len(trade.models_used),
                    'execution_quality': trade.execution_quality,
                    'outcome': 'profit' if trade.pnl_absolute and trade.pnl_absolute > 0 else 'loss' if trade.pnl_absolute and trade.pnl_absolute < 0 else 'pending'
                }

                recent_panel['recent_trades'].append(trade_info)

            return recent_panel

        except Exception as e:
            tprint_error(f"❌ Failed to generate recent trades panel: {e}")
            return {}

    async def _export_dashboard(self, dashboard: Dict[str, Any], dashboard_name: str):
        """Export dashboard to files."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            # Export JSON dashboard data
            json_file = self.dashboard_directory / f"{dashboard_name}_{timestamp}.json"
            with open(json_file, 'w') as f:
                json.dump(dashboard, f, indent=2, default=str)

            # Export current dashboard (overwrite for live updates)
            current_file = self.dashboard_directory / f"{dashboard_name}_current.json"
            with open(current_file, 'w') as f:
                json.dump(dashboard, f, indent=2, default=str)

            # Generate HTML dashboard
            await self._generate_html_dashboard(dashboard, dashboard_name, timestamp)

            tprint_info(f"📊 Dashboard exported to {self.dashboard_directory}")

        except Exception as e:
            tprint_error(f"❌ Failed to export dashboard: {e}")

    async def _generate_html_dashboard(self, dashboard: Dict[str, Any], dashboard_name: str, timestamp: str):
        """Generate interactive HTML dashboard."""
        try:
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Live Trading Dashboard</title>
                <meta http-equiv="refresh" content="{self.update_interval}">
                <style>
                    body {{
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        margin: 0;
                        padding: 20px;
                        background-color: #f5f5f5;
                    }}
                    .dashboard-header {{
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        padding: 20px;
                        border-radius: 10px;
                        margin-bottom: 20px;
                        text-align: center;
                    }}
                    .metrics-grid {{
                        display: grid;
                        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                        gap: 20px;
                        margin-bottom: 20px;
                    }}
                    .metric-card {{
                        background: white;
                        padding: 20px;
                        border-radius: 10px;
                        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                        text-align: center;
                    }}
                    .metric-value {{
                        font-size: 2em;
                        font-weight: bold;
                        margin: 10px 0;
                    }}
                    .positive {{ color: #28a745; }}
                    .negative {{ color: #dc3545; }}
                    .neutral {{ color: #6c757d; }}
                    .panel {{
                        background: white;
                        margin: 20px 0;
                        padding: 20px;
                        border-radius: 10px;
                        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    }}
                    .panel h3 {{
                        margin-top: 0;
                        color: #333;
                        border-bottom: 2px solid #eee;
                        padding-bottom: 10px;
                    }}
                    table {{
                        width: 100%;
                        border-collapse: collapse;
                        margin: 10px 0;
                    }}
                    th, td {{
                        padding: 12px;
                        text-align: left;
                        border-bottom: 1px solid #ddd;
                    }}
                    th {{
                        background-color: #f8f9fa;
                        font-weight: bold;
                    }}
                    .status-indicator {{
                        width: 12px;
                        height: 12px;
                        border-radius: 50%;
                        display: inline-block;
                        margin-right: 8px;
                    }}
                    .status-active {{ background-color: #28a745; }}
                    .status-completed {{ background-color: #17a2b8; }}
                    .status-failed {{ background-color: #dc3545; }}
                </style>
            </head>
            <body>
                <div class="dashboard-header">
                    <h1>🚀 Live Trading Dashboard</h1>
                    <p>Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p>Auto-refresh: {self.update_interval} seconds</p>
                </div>
            """

            # Add live metrics
            if 'live_metrics' in dashboard:
                html_content += self._generate_live_metrics_html(dashboard['live_metrics'])

            # Add active trades panel
            if 'active_trades_panel' in dashboard:
                html_content += self._generate_active_trades_html(dashboard['active_trades_panel'])

            # Add recent trades panel
            if 'recent_trades_panel' in dashboard:
                html_content += self._generate_recent_trades_html(dashboard['recent_trades_panel'])

            # Add model performance panel
            if 'model_dashboard' in dashboard:
                html_content += self._generate_model_dashboard_html(dashboard['model_dashboard'])

            html_content += """
                <script>
                    // Auto-refresh functionality
                    setInterval(function() {
                        location.reload();
                    }, """ + str(self.update_interval * 1000) + """);

                    // Add timestamp to show last update
                    document.addEventListener('DOMContentLoaded', function() {
                        const now = new Date();
                        console.log('Dashboard updated at:', now.toLocaleString());
                    });
                </script>
            </body>
            </html>
            """

            # Save HTML dashboard
            html_file = self.dashboard_directory / f"{dashboard_name}_live.html"
            with open(html_file, 'w') as f:
                f.write(html_content)

            tprint_success(f"✅ Generated HTML dashboard: {html_file}")

        except Exception as e:
            tprint_error(f"❌ Failed to generate HTML dashboard: {e}")

    def _generate_live_metrics_html(self, live_metrics: Dict[str, Any]) -> str:
        """Generate HTML for live metrics."""
        html = '<div class="metrics-grid">'

        if 'current_performance' in live_metrics:
            perf = live_metrics['current_performance']

            # Total PnL
            pnl_class = 'positive' if perf.get('total_pnl', 0) > 0 else 'negative' if perf.get('total_pnl', 0) < 0 else 'neutral'
            html += f'''
            <div class="metric-card">
                <h4>Total PnL</h4>
                <div class="metric-value {pnl_class}">${perf.get('total_pnl', 0):.2f}</div>
            </div>
            '''

            # Win Rate
            win_rate = perf.get('win_rate', 0)
            win_rate_class = 'positive' if win_rate > 0.6 else 'negative' if win_rate < 0.4 else 'neutral'
            html += f'''
            <div class="metric-card">
                <h4>Win Rate</h4>
                <div class="metric-value {win_rate_class}">{win_rate:.1%}</div>
            </div>
            '''

            # Total Trades
            html += f'''
            <div class="metric-card">
                <h4>Total Trades</h4>
                <div class="metric-value neutral">{perf.get('total_trades', 0)}</div>
            </div>
            '''

            # Current Drawdown
            drawdown = perf.get('current_drawdown', 0)
            drawdown_class = 'negative' if drawdown > 0.1 else 'neutral'
            html += f'''
            <div class="metric-card">
                <h4>Current Drawdown</h4>
                <div class="metric-value {drawdown_class}">{drawdown:.1%}</div>
            </div>
            '''

        html += '</div>'
        return html

    def _generate_active_trades_html(self, active_trades: Dict[str, Any]) -> str:
        """Generate HTML for active trades panel."""
        html = '''
        <div class="panel">
            <h3><span class="status-indicator status-active"></span>Active Trades</h3>
        '''

        if active_trades.get('active_trades'):
            html += '''
            <table>
                <tr>
                    <th>Trade ID</th>
                    <th>Symbol</th>
                    <th>Action</th>
                    <th>Price</th>
                    <th>Duration</th>
                    <th>Confidence</th>
                    <th>Regime</th>
                </tr>
            '''

            for trade in active_trades['active_trades']:
                duration_str = f"{trade['duration_minutes']:.0f}m"
                confidence_class = 'positive' if trade['confidence'] > 0.7 else 'neutral'

                html += f'''
                <tr>
                    <td>{trade['trade_id'][:8]}...</td>
                    <td>{trade['symbol']}</td>
                    <td>{trade['action'].upper()}</td>
                    <td>${trade['entry_price']:.4f}</td>
                    <td>{duration_str}</td>
                    <td class="{confidence_class}">{trade['confidence']:.1%}</td>
                    <td>{trade['regime'].replace('_', ' ').title()}</td>
                </tr>
                '''

            html += '</table>'
        else:
            html += '<p>No active trades</p>'

        html += '</div>'
        return html

    def _generate_recent_trades_html(self, recent_trades: Dict[str, Any]) -> str:
        """Generate HTML for recent trades panel."""
        html = '''
        <div class="panel">
            <h3><span class="status-indicator status-completed"></span>Recent Trades</h3>
        '''

        if recent_trades.get('recent_trades'):
            html += '''
            <table>
                <tr>
                    <th>Time</th>
                    <th>Symbol</th>
                    <th>Action</th>
                    <th>Price</th>
                    <th>PnL</th>
                    <th>Confidence</th>
                    <th>Models</th>
                    <th>Outcome</th>
                </tr>
            '''

            for trade in recent_trades['recent_trades']:
                pnl_class = 'positive' if trade['pnl'] > 0 else 'negative' if trade['pnl'] < 0 else 'neutral'
                outcome_indicator = 'status-active' if trade['outcome'] == 'pending' else 'status-completed' if trade['pnl'] > 0 else 'status-failed'

                html += f'''
                <tr>
                    <td>{trade['timestamp']}</td>
                    <td>{trade['symbol']}</td>
                    <td>{trade['action'].upper()}</td>
                    <td>${trade['price']:.4f}</td>
                    <td class="{pnl_class}">${trade['pnl']:.2f}</td>
                    <td>{trade['confidence']:.1%}</td>
                    <td>{trade['models_used']}</td>
                    <td><span class="status-indicator {outcome_indicator}"></span>{trade['outcome'].title()}</td>
                </tr>
                '''

            html += '</table>'
        else:
            html += '<p>No recent trades</p>'

        html += '</div>'
        return html

    def _generate_model_dashboard_html(self, model_dashboard: Dict[str, Any]) -> str:
        """Generate HTML for model performance dashboard."""
        html = '''
        <div class="panel">
            <h3>🤖 Model Performance Dashboard</h3>
        '''

        if model_dashboard:
            html += '''
            <table>
                <tr>
                    <th>Model ID</th>
                    <th>Usage (24h)</th>
                    <th>Total PnL</th>
                    <th>Recent PnL</th>
                    <th>Win Rate</th>
                    <th>Avg Confidence</th>
                </tr>
            '''

            for model_id, metrics in model_dashboard.items():
                usage_stats = metrics.get('usage_stats', {})
                perf_stats = metrics.get('performance_stats', {})
                conf_stats = metrics.get('confidence_stats', {})

                pnl_class = 'positive' if perf_stats.get('total_pnl', 0) > 0 else 'negative'

                html += f'''
                <tr>
                    <td>{model_id}</td>
                    <td>{usage_stats.get('recent_usage_24h', 0)}</td>
                    <td class="{pnl_class}">${perf_stats.get('total_pnl', 0):.2f}</td>
                    <td class="{pnl_class}">${perf_stats.get('recent_pnl_24h', 0):.2f}</td>
                    <td>{perf_stats.get('win_rate', 0):.1%}</td>
                    <td>{conf_stats.get('avg_confidence', 0):.1%}</td>
                </tr>
                '''

            html += '</table>'
        else:
            html += '<p>No model performance data available</p>'

        html += '</div>'
        return html

    async def _calculate_recent_model_usage(self, recent_trades: List[DetailedTradeMetrics]) -> Dict[str, int]:
        """Calculate recent model usage statistics."""
        try:
            model_usage = {}
            for trade in recent_trades:
                for model_id in trade.models_used.keys():
                    model_usage[model_id] = model_usage.get(model_id, 0) + 1
            return model_usage
        except Exception as e:
            tprint_error(f"❌ Failed to calculate recent model usage: {e}")
            return {}

    async def _get_current_market_status(self, trades: List[DetailedTradeMetrics]) -> Dict[str, Any]:
        """Get current market status from recent trades."""
        try:
            if not trades:
                return {'status': 'unknown'}
            
            recent_trades = trades[-10:] if len(trades) > 10 else trades
            
            # Analyze recent market conditions
            avg_volatility = np.mean([t.volatility_estimate for t in recent_trades if t.volatility_estimate > 0]) if recent_trades else 0.0
            avg_confidence = np.mean([t.signal_confidence for t in recent_trades if t.signal_confidence > 0]) if recent_trades else 0.0
            
            # Most recent regime
            current_regime = recent_trades[-1].regime_type if recent_trades else "unknown"
            
            return {
                'status': 'active' if recent_trades else 'inactive',
                'avg_volatility': avg_volatility,
                'avg_confidence': avg_confidence,
                'current_regime': current_regime,
                'recent_trade_count': len(recent_trades)
            }
        except Exception as e:
            tprint_error(f"❌ Failed to get current market status: {e}")
            return {'status': 'unknown'}

    async def _generate_model_performance_timeline(self, trades: List[DetailedTradeMetrics]) -> List[Dict[str, Any]]:
        """Generate timeline of model performance over time."""
        try:
            if not trades:
                return []
            
            # Sort trades by timestamp
            sorted_trades = sorted(trades, key=lambda t: t.timestamp)
            
            timeline = []
            model_performance_by_time = {}
            
            for trade in sorted_trades:
                time_key = trade.timestamp.strftime('%Y-%m-%d %H:%M')
                
                for model_id in trade.models_used.keys():
                    if model_id not in model_performance_by_time:
                        model_performance_by_time[model_id] = {'pnl': [], 'timestamps': []}
                    
                    if trade.pnl_absolute is not None:
                        model_performance_by_time[model_id]['pnl'].append(trade.pnl_absolute)
                        model_performance_by_time[model_id]['timestamps'].append(trade.timestamp.isoformat())
            
            # Aggregate by time windows
            for model_id, data in model_performance_by_time.items():
                if data['timestamps']:
                    timeline.append({
                        'model_id': model_id,
                        'timestamps': data['timestamps'],
                        'cumulative_pnl': np.cumsum(data['pnl']).tolist() if data['pnl'] else []
                    })
            
            return timeline
        except Exception as e:
            tprint_error(f"❌ Failed to generate model performance timeline: {e}")
            return []

    async def _generate_trade_frequency_data(self, trades: List[DetailedTradeMetrics]) -> Dict[str, Any]:
        """Generate trade frequency analysis data."""
        try:
            if not trades:
                return {}
            
            # Sort by timestamp
            sorted_trades = sorted(trades, key=lambda t: t.timestamp)
            
            # Group by hour
            hourly_counts = {}
            for trade in sorted_trades:
                hour = trade.timestamp.hour
                hourly_counts[hour] = hourly_counts.get(hour, 0) + 1
            
            # Calculate trades per hour average
            if len(sorted_trades) > 1:
                duration_hours = (sorted_trades[-1].timestamp - sorted_trades[0].timestamp).total_seconds() / 3600
                avg_trades_per_hour = len(trades) / duration_hours if duration_hours > 0 else 0.0
            else:
                avg_trades_per_hour = 0.0
            
            return {
                'hourly_distribution': hourly_counts,
                'total_trades': len(trades),
                'avg_trades_per_hour': avg_trades_per_hour,
                'time_range': {
                    'start': sorted_trades[0].timestamp.isoformat() if sorted_trades else None,
                    'end': sorted_trades[-1].timestamp.isoformat() if sorted_trades else None
                }
            }
        except Exception as e:
            tprint_error(f"❌ Failed to generate trade frequency data: {e}")
            return {}

    async def _get_model_feature_importance(self, model_id: str, model_trades: List[DetailedTradeMetrics]) -> Dict[str, float]:
        """Get aggregated feature importance for a specific model."""
        try:
            if not model_trades:
                return {}
            
            feature_importance = {}
            feature_counts = {}
            
            for trade in model_trades:
                # Get SHAP values for this model
                if model_id in trade.shap_explanations:
                    shap_values = trade.shap_explanations[model_id]
                    for feature, importance in shap_values.items():
                        if feature not in feature_importance:
                            feature_importance[feature] = 0.0
                            feature_counts[feature] = 0
                        feature_importance[feature] += abs(importance)
                        feature_counts[feature] += 1
            
            # Average feature importance
            for feature in feature_importance:
                if feature_counts[feature] > 0:
                    feature_importance[feature] /= feature_counts[feature]
            
            # Sort by importance and return top 10
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            
            return dict(sorted_features)
        except Exception as e:
            tprint_error(f"❌ Failed to get model feature importance: {e}")
            return {}

    async def _analyze_regime_transitions(self, trades: List[DetailedTradeMetrics]) -> Dict[str, Any]:
        """Analyze regime transitions in trading history."""
        try:
            if not trades or len(trades) < 2:
                return {}
            
            # Sort by timestamp
            sorted_trades = sorted(trades, key=lambda t: t.timestamp)
            
            transitions = []
            transition_matrix = {}
            
            for i in range(1, len(sorted_trades)):
                prev_regime = sorted_trades[i-1].regime_type
                curr_regime = sorted_trades[i].regime_type
                
                if prev_regime != curr_regime:
                    transitions.append({
                        'timestamp': sorted_trades[i].timestamp.isoformat(),
                        'from_regime': prev_regime,
                        'to_regime': curr_regime
                    })
                    
                    # Update transition matrix
                    key = f"{prev_regime} -> {curr_regime}"
                    transition_matrix[key] = transition_matrix.get(key, 0) + 1
            
            return {
                'total_transitions': len(transitions),
                'transition_matrix': transition_matrix,
                'recent_transitions': transitions[-10:] if transitions else [],
                'avg_transitions_per_day': len(transitions) / ((sorted_trades[-1].timestamp - sorted_trades[0].timestamp).days + 1) if len(sorted_trades) > 1 and (sorted_trades[-1].timestamp - sorted_trades[0].timestamp).days >= 0 else 0.0
            }
        except Exception as e:
            tprint_error(f"❌ Failed to analyze regime transitions: {e}")
            return {}

# Global instance
dashboard_generator = DashboardGenerator()

# Convenience functions
async def create_trading_dashboard(
    trades: List[DetailedTradeMetrics],
    session_metrics: Optional[TradingSessionMetrics] = None,
    active_trades: Optional[Dict[str, DetailedTradeMetrics]] = None
) -> Dict[str, Any]:
    """Create comprehensive trading dashboard."""
    return await dashboard_generator.generate_live_dashboard(trades, session_metrics, active_trades)
