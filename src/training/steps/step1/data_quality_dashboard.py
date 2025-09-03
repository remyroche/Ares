from typing import Dict, List, Optional, Union, Any, Tuple
"""Data Quality Dashboard Web Interface.

This module provides a web-based dashboard for monitoring and managing data quality.
It includes real-time metrics, alert management, and quality control features.
"""
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
from src.core.decorators import traced
from src.utils.logger import system_logger
logger = system_logger.getChild('DataQualityDashboard')
try:
    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logger.warning('⚠️ FastAPI not available - dashboard will use basic HTTP server')
try:
    import uvicorn
    UVICORN_AVAILABLE = True
except ImportError:
    UVICORN_AVAILABLE = False
    logger.warning('⚠️ Uvicorn not available - dashboard server not available')

class DashboardConfig(BaseModel):
    """Dashboard configuration."""
    host: str = '0.0.0.0'
    port: int = 8080
    refresh_interval: int = 30
    max_alerts: int = 100
    enable_websocket: bool = True

class DataQualityDashboard:
    """Web-based data quality dashboard."""

    def __init__(self, data_cache_path: str='data_cache', config: DashboardConfig | None=None) -> None:
        self.data_cache_path = Path(data_cache_path)
        self.data_cache_path.mkdir(exist_ok=True)
        self.config = config or DashboardConfig()
        self.quality_manager = None
        self.monitor = None
        self.app = None
        self.websocket_connections: list[WebSocket] = []
        self._initialize_components()

    def _initialize_components(self) -> None:
        """Initialize dashboard components."""
        try:
            from .enhanced_data_quality_manager import EnhancedDataQualityManager
            self.quality_manager = EnhancedDataQualityManager(str(self.data_cache_path))
            logger.info('✅ Enhanced data quality manager initialized for dashboard')
        except ImportError as e:
            logger.warning(f'⚠️ Could not import EnhancedDataQualityManager: {e}')
        try:
            from .data_quality_monitor import DataQualityMonitor
            self.monitor = DataQualityMonitor(str(self.data_cache_path))
            logger.info('✅ Data quality monitor initialized for dashboard')
        except ImportError as e:
            logger.warning(f'⚠️ Could not import DataQualityMonitor: {e}')
        if FASTAPI_AVAILABLE:
            self._create_fastapi_app()
        else:
            logger.warning('⚠️ FastAPI not available - dashboard will be limited')

    def _create_fastapi_app(self) -> None:
        """Create FastAPI application with routes."""
        self.app = FastAPI(title='Data Quality Dashboard', description='Real-time data quality monitoring and management', version='1.0.0')
        self._add_routes()
        static_dir = self.data_cache_path / 'dashboard_static'
        static_dir.mkdir(exist_ok=True)
        self.app.mount('/static', StaticFiles(directory=str(static_dir)), name='static')

    def _add_routes(self) -> None:
        """Add API routes to the FastAPI app."""

        @self.app.get('/', response_class=HTMLResponse)
        async def dashboard_home() -> None:
            """Main dashboard page."""
            return self._generate_dashboard_html()

        @self.app.get('/api/status')
        async def get_status() -> Any:
            """Get overall system status."""
            return await self._get_system_status()

        @self.app.get('/api/metrics')
        async def get_metrics() -> Any:
            """Get current quality metrics."""
            return await self._get_quality_metrics()

        @self.app.get('/api/alerts')
        async def get_alerts(symbol: str | None=None, exchange: str | None=None, severity: str | None=None, limit: int=50) -> Any:
            """Get filtered alerts."""
            return await self._get_alerts(symbol, exchange, severity, limit)

        @self.app.post('/api/alerts/{alert_id}/acknowledge')
        async def acknowledge_alert(alert_id: int) -> None:
            """Acknowledge an alert."""
            return await self._acknowledge_alert(alert_id)

        @self.app.post('/api/alerts/{alert_id}/resolve')
        async def resolve_alert(alert_id: int) -> None:
            """Resolve an alert."""
            return await self._resolve_alert(alert_id)

        @self.app.post('/api/quality-check')
        async def run_quality_check(symbol: str, exchange: str, timeframe: str='1m') -> Any:
            """Run a quality check for specific data."""
            return await self._run_quality_check(symbol, exchange, timeframe)

        @self.app.get('/api/monitoring/status')
        async def get_monitoring_status() -> Any:
            """Get monitoring status."""
            return await self._get_monitoring_status()

        @self.app.post('/api/monitoring/start')
        async def start_monitoring(symbols: list[str], exchanges: list[str], timeframes: list[str]) -> None:
            """Start monitoring."""
            return await self._start_monitoring(symbols, exchanges, timeframes)

        @self.app.post('/api/monitoring/stop')
        async def stop_monitoring() -> None:
            """Stop monitoring."""
            return await self._stop_monitoring()
        if self.config.enable_websocket:

            @self.app.websocket('/ws')
            async def websocket_endpoint(websocket: WebSocket) -> None:
                """WebSocket endpoint for real-time updates."""
                await self._handle_websocket(websocket)

    def _generate_dashboard_html(self) -> str:
        """Generate the main dashboard HTML."""
        return f"""\n<!DOCTYPE html>\n<html lang="en">\n<head>\n    <meta charset="UTF-8">\n    <meta name="viewport" content="width=device-width, initial-scale=1.0">\n    <title>Data Quality Dashboard</title>\n    <style>\n        body {{\n            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;\n            margin: 0;\n            padding: 20px;\n            background-color: #f5f5f5;\n        }}\n        .container {{\n            max-width: 1200px;\n            margin: 0 auto;\n        }}\n        .header {{\n            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);\n            color: white;\n            padding: 20px;\n            border-radius: 10px;\n            margin-bottom: 20px;\n        }}\n        .header h1 {{\n            margin: 0;\n            font-size: 2.5em;\n        }}\n        .header p {{\n            margin: 10px 0 0 0;\n            opacity: 0.9;\n        }}\n        .dashboard-grid {{\n            display: grid;\n            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));\n            gap: 20px;\n            margin-bottom: 20px;\n        }}\n        .card {{\n            background: white;\n            border-radius: 10px;\n            padding: 20px;\n            box-shadow: 0 2px 10px rgba(0,0,0,0.1);\n        }}\n        .card h3 {{\n            margin-top: 0;\n            color: #333;\n            border-bottom: 2px solid #667eea;\n            padding-bottom: 10px;\n        }}\n        .metric {{\n            display: flex;\n            justify-content: space-between;\n            align-items: center;\n            padding: 10px 0;\n            border-bottom: 1px solid #eee;\n        }}\n        .metric:last-child {{\n            border-bottom: none;\n        }}\n        .metric-value {{\n            font-weight: bold;\n            font-size: 1.2em;\n        }}\n        .status-good {{ color: #28a745; }}\n        .status-warning {{ color: #ffc107; }}\n        .status-error {{ color: #dc3545; }}\n        .alerts-list {{\n            max-height: 400px;\n            overflow-y: auto;\n        }}\n        .alert-item {{\n            padding: 10px;\n            margin: 5px 0;\n            border-radius: 5px;\n            border-left: 4px solid #ddd;\n        }}\n        .alert-high {{ border-left-color: #dc3545; background-color: #f8d7da; }}\n        .alert-medium {{ border-left-color: #ffc107; background-color: #fff3cd; }}\n        .alert-low {{ border-left-color: #28a745; background-color: #d4edda; }}\n        .controls {{\n            display: flex;\n            gap: 10px;\n            margin-bottom: 20px;\n        }}\n        .btn {{\n            padding: 10px 20px;\n            border: none;\n            border-radius: 5px;\n            cursor: pointer;\n            font-weight: bold;\n        }}\n        .btn-primary {{ background-color: #667eea; color: white; }}\n        .btn-success {{ background-color: #28a745; color: white; }}\n        .btn-warning {{ background-color: #ffc107; color: black; }}\n        .btn-danger {{ background-color: #dc3545; color: white; }}\n        .refresh-info {{\n            text-align: center;\n            color: #666;\n            font-size: 0.9em;\n            margin-top: 20px;\n        }}\n    </style>\n</head>\n<body>\n    <div class="container">\n        <div class="header">\n            <h1>📊 Data Quality Dashboard</h1>\n            <p>Real-time monitoring and management of data quality across the training pipeline</p>\n        </div>\n\n        <div class="controls">\n            <button class="btn btn-primary" onclick="refreshDashboard()">🔄 Refresh</button>\n            <button class="btn btn-success" onclick="startMonitoring()">▶️ Start Monitoring</button>\n            <button class="btn btn-warning" onclick="stopMonitoring()">⏹️ Stop Monitoring</button>\n            <button class="btn btn-primary" onclick="runQualityCheck()">🔍 Run Quality Check</button>\n        </div>\n\n        <div class="dashboard-grid">\n            <div class="card">\n                <h3>📈 System Status</h3>\n                <div id="system-status">\n                    <div class="metric">\n                        <span>Overall Status:</span>\n                        <span class="metric-value status-good" id="overall-status">Loading...</span>\n                    </div>\n                    <div class="metric">\n                        <span>Monitoring Active:</span>\n                        <span class="metric-value" id="monitoring-status">Loading...</span>\n                    </div>\n                    <div class="metric">\n                        <span>Last Update:</span>\n                        <span class="metric-value" id="last-update">Loading...</span>\n                    </div>\n                </div>\n            </div>\n\n            <div class="card">\n                <h3>📊 Quality Metrics</h3>\n                <div id="quality-metrics">\n                    <div class="metric">\n                        <span>Data Gaps:</span>\n                        <span class="metric-value" id="data-gaps">Loading...</span>\n                    </div>\n                    <div class="metric">\n                        <span>Format Issues:</span>\n                        <span class="metric-value" id="format-issues">Loading...</span>\n                    </div>\n                    <div class="metric">\n                        <span>Data Freshness:</span>\n                        <span class="metric-value" id="data-freshness">Loading...</span>\n                    </div>\n                    <div class="metric">\n                        <span>Step3/4 Ready:</span>\n                        <span class="metric-value" id="step-ready">Loading...</span>\n                    </div>\n                </div>\n            </div>\n\n            <div class="card">\n                <h3>🚨 Recent Alerts</h3>\n                <div class="alerts-list" id="recent-alerts">\n                    <div class="metric">\n                        <span>Loading alerts...</span>\n                    </div>\n                </div>\n            </div>\n\n            <div class="card">\n                <h3>⚙️ Quick Actions</h3>\n                <div>\n                    <button class="btn btn-primary" style="width: 100%; margin-bottom: 10px;" onclick="checkETHUSDT()">\n                        Check ETHUSDT\n                    </button>\n                    <button class="btn btn-primary" style="width: 100%; margin-bottom: 10px;" onclick="checkBTCUSDT()">\n                        Check BTCUSDT\n                    </button>\n                    <button class="btn btn-success" style="width: 100%; margin-bottom: 10px;" onclick="runStep1()">\n                        Run Step1\n                    </button>\n                    <button class="btn btn-success" style="width: 100%; margin-bottom: 10px;" onclick="runStep1_5()">\n                        Run Step1.5\n                    </button>\n                </div>\n            </div>\n        </div>\n\n        <div class="refresh-info">\n            Auto-refresh every {self.config.refresh_interval} seconds\n        </div>\n    </div>\n\n    <script>\n        let refreshInterval;\n\n        // Initialize dashboard\n        document.addEventListener('DOMContentLoaded', function() {{\n            refreshDashboard();\n            startAutoRefresh();\n        }});\n\n        function startAutoRefresh() {{\n            refreshInterval = setInterval(refreshDashboard, {self.config.refresh_interval * 1000});\n        }}\n\n        function stopAutoRefresh() {{\n            if (refreshInterval) {{\n                clearInterval(refreshInterval);\n            }}\n        }}\n\n        async function refreshDashboard() {{\n            try {{\n                // Update system status\n                const statusResponse = await fetch('/api/status');\n                const status = await statusResponse.json();\n                updateSystemStatus(status);\n\n                // Update quality metrics\n                const metricsResponse = await fetch('/api/metrics');\n                const metrics = await metricsResponse.json();\n                updateQualityMetrics(metrics);\n\n                // Update alerts\n                const alertsResponse = await fetch('/api/alerts?limit=10');\n                const alerts = await alertsResponse.json();\n                updateAlerts(alerts);\n\n            }} catch (error) {{\n                console.error('Error refreshing dashboard:', error);\n            }}\n        }}\n\n        function updateSystemStatus(status) {{\n            document.getElementById('overall-status').textContent = status.overall_status;\n            document.getElementById('overall-status').className = `metric-value status-${{status.overall_status === 'healthy' ? 'good' : 'error'}}`;\n            document.getElementById('monitoring-status').textContent = status.monitoring_active ? 'Active' : 'Inactive';\n            document.getElementById('last-update').textContent = status.last_update;\n        }}\n\n        function updateQualityMetrics(metrics) {{\n            document.getElementById('data-gaps').textContent = metrics.total_gaps || 0;\n            document.getElementById('format-issues').textContent = metrics.format_issues || 0;\n            document.getElementById('data-freshness').textContent = metrics.data_freshness || 'Unknown';\n            document.getElementById('step-ready').textContent = metrics.step3_step4_ready ? 'Yes' : 'No';\n        }}\n\n        function updateAlerts(alerts) {{\n            const alertsContainer = document.getElementById('recent-alerts');\n            alertsContainer.innerHTML = '';\n\n            if (alerts.length === 0) {{\n                alertsContainer.innerHTML = '<div class="metric"><span>No recent alerts</span></div>';\n                return;\n            }}\n\n            alerts.forEach(alert => {{\n                const alertDiv = document.createElement('div');\n                alertDiv.className = `alert-item alert-${{alert.severity}}`;\n                alertDiv.innerHTML = `\n                    <strong>${{alert.alert_type}}</strong><br>\n                    ${{alert.message}}<br>\n                    <small>${{alert.timestamp}}</small>\n                `;\n                alertsContainer.appendChild(alertDiv);\n            }});\n        }}\n\n        async function startMonitoring() {{\n            try {{\n                const response = await fetch('/api/monitoring/start', {{\n                    method: 'POST',\n                    headers: {{ 'Content-Type': 'application/json' }},\n                    body: JSON.stringify({{\n                        symbols: ['ETHUSDT', 'BTCUSDT'],\n                        exchanges: ['BINANCE'],\n                        timeframes: ['1m']\n                    }})\n                }});\n                const result = await response.json();\n                alert(result.message);\n                refreshDashboard();\n            }} catch (error) {{\n                console.error('Error starting monitoring:', error);\n                alert('Error starting monitoring');\n            }}\n        }}\n\n        async function stopMonitoring() {{\n            try {{\n                const response = await fetch('/api/monitoring/stop', {{ method: 'POST' }});\n                const result = await response.json();\n                alert(result.message);\n                refreshDashboard();\n            }} catch (error) {{\n                console.error('Error stopping monitoring:', error);\n                alert('Error stopping monitoring');\n            }}\n        }}\n\n        async function runQualityCheck() {{\n            const symbol = prompt('Enter symbol (e.g., ETHUSDT):', 'ETHUSDT');\n            if (symbol) {{\n                try {{\n                    const response = await fetch(`/api/quality-check?symbol=${{symbol}}&exchange=BINANCE&timeframe=1m`);\n                    const result = await response.json();\n                    alert(`Quality check result: ${{result.success ? 'Success' : 'Failed'}}`);\n                    refreshDashboard();\n                }} catch (error) {{\n                    console.error('Error running quality check:', error);\n                    alert('Error running quality check');\n                }}\n            }}\n        }}\n\n        async function checkETHUSDT() {{\n            try {{\n                const response = await fetch('/api/quality-check?symbol=ETHUSDT&exchange=BINANCE&timeframe=1m');\n                const result = await response.json();\n                alert(`ETHUSDT quality check: ${{result.success ? 'Success' : 'Failed'}}`);\n                refreshDashboard();\n            }} catch (error) {{\n                console.error('Error checking ETHUSDT:', error);\n                alert('Error checking ETHUSDT');\n            }}\n        }}\n\n        async function checkBTCUSDT() {{\n            try {{\n                const response = await fetch('/api/quality-check?symbol=BTCUSDT&exchange=BINANCE&timeframe=1m');\n                const result = await response.json();\n                alert(`BTCUSDT quality check: ${{result.success ? 'Success' : 'Failed'}}`);\n                refreshDashboard();\n            }} catch (error) {{\n                console.error('Error checking BTCUSDT:', error);\n                alert('Error checking BTCUSDT');\n            }}\n        }}\n\n        async function runStep1() {{\n            try {{\n                const response = await fetch('/api/run-step01', {{ method: 'POST' }});\n                const result = await response.json();\n                alert(`Step1 result: ${{result.success ? 'Success' : 'Failed'}}`);\n                refreshDashboard();\n            }} catch (error) {{\n                console.error('Error running Step1:', error);\n                alert('Error running Step1');\n            }}\n        }}\n\n        async function runStep1_5() {{\n            try {{\n                const response = await fetch('/api/run-step1_5', {{ method: 'POST' }});\n                const result = await response.json();\n                alert(`Step1.5 result: ${{result.success ? 'Success' : 'Failed' }}`);\n                refreshDashboard();\n            }} catch (error) {{\n                console.error('Error running Step1.5:', error);\n                alert('Error running Step1.5');\n            }}\n        }}\n    </script>\n</body>\n</html>\n        """

    @traced(span_name='get_system_status')
    async def _get_system_status(self) -> dict[str, Any]:
        """Get overall system status."""
        try:
            status = {'overall_status': 'healthy', 'monitoring_active': False, 'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'components': {}}
            if self.quality_manager:
                status['components']['quality_manager'] = 'available'
            else:
                status['components']['quality_manager'] = 'unavailable'
                status['overall_status'] = 'degraded'
            if self.monitor:
                status['components']['monitor'] = 'available'
                status['monitoring_active'] = self.monitor.monitoring_active
            else:
                status['components']['monitor'] = 'unavailable'
                status['overall_status'] = 'degraded'
            return status
        except Exception as e:
            logger.exception(f'❌ Error getting system status: {e}')
            return {'overall_status': 'error', 'error': str(e), 'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

    @traced(span_name='get_quality_metrics')
    async def _get_quality_metrics(self) -> dict[str, Any]:
        """Get current quality metrics."""
        try:
            metrics = {'total_gaps': 0, 'format_issues': 0, 'data_freshness': 'unknown', 'step3_step4_ready': False, 'last_check': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            if self.monitor:
                monitor_metrics = self.monitor.get_performance_metrics()
                metrics.update(monitor_metrics)
            return metrics
        except Exception as e:
            logger.exception(f'❌ Error getting quality metrics: {e}')
            return {'error': str(e), 'last_check': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

    @traced(span_name='get_alerts')
    async def _get_alerts(self, symbol: str | None=None, exchange: str | None=None, severity: str | None=None, limit: int=50) -> list[dict[str, Any]]:
        """Get filtered alerts."""
        try:
            if not self.monitor:
                return []
            alerts = self.monitor.get_alerts(symbol=symbol, exchange=exchange, severity=severity, limit=limit)
            return [alert.to_dict() for alert in alerts]
        except Exception as e:
            logger.exception(f'❌ Error getting alerts: {e}')
            return []

    @traced(span_name='acknowledge_alert')
    async def _acknowledge_alert(self, alert_id: int) -> dict[str, Any]:
        """Acknowledge an alert."""
        try:
            if not self.monitor:
                raise HTTPException(status_code=404, detail='Monitor not available')
            success = self.monitor.acknowledge_alert(alert_id)
            return {'success': success, 'message': 'Alert acknowledged' if success else 'Alert not found'}
        except Exception as e:
            logger.exception(f'❌ Error acknowledging alert: {e}')
            raise HTTPException(status_code=500, detail=str(e))

    @traced(span_name='resolve_alert')
    async def _resolve_alert(self, alert_id: int) -> dict[str, Any]:
        """Resolve an alert."""
        try:
            if not self.monitor:
                raise HTTPException(status_code=404, detail='Monitor not available')
            success = self.monitor.resolve_alert(alert_id)
            return {'success': success, 'message': 'Alert resolved' if success else 'Alert not found'}
        except Exception as e:
            logger.exception(f'❌ Error resolving alert: {e}')
            raise HTTPException(status_code=500, detail=str(e))

    @traced(span_name='run_quality_check')
    async def _run_quality_check(self, symbol: str, exchange: str, timeframe: str) -> dict[str, Any]:
        """Run a quality check for specific data."""
        try:
            if not self.quality_manager:
                raise HTTPException(status_code=404, detail='Quality manager not available')
            results = await self.quality_manager.comprehensive_quality_check(symbol=symbol, exchange=exchange, timeframe=timeframe, check_gaps=True, fill_gaps=False, validate_format=True)
            return {'success': results.get('success', False), 'results': results, 'timestamp': datetime.now().isoformat()}
        except Exception as e:
            logger.exception(f'❌ Error running quality check: {e}')
            raise HTTPException(status_code=500, detail=str(e))

    @traced(span_name='get_monitoring_status')
    async def _get_monitoring_status(self) -> dict[str, Any]:
        """Get monitoring status."""
        try:
            if not self.monitor:
                return {'active': False, 'error': 'Monitor not available'}
            return {'active': self.monitor.monitoring_active, 'interval': self.monitor.monitoring_interval, 'metrics': self.monitor.get_performance_metrics()}
        except Exception as e:
            logger.exception(f'❌ Error getting monitoring status: {e}')
            return {'active': False, 'error': str(e)}

    @traced(span_name='start_monitoring')
    async def _start_monitoring(self, symbols: list[str], exchanges: list[str], timeframes: list[str]) -> dict[str, Any]:
        """Start monitoring."""
        try:
            if not self.monitor:
                raise HTTPException(status_code=404, detail='Monitor not available')
            success = await self.monitor.start_monitoring(symbols, exchanges, timeframes)
            return {'success': success, 'message': 'Monitoring started' if success else 'Failed to start monitoring'}
        except Exception as e:
            logger.exception(f'❌ Error starting monitoring: {e}')
            raise HTTPException(status_code=500, detail=str(e))

    @traced(span_name='stop_monitoring')
    async def _stop_monitoring(self) -> dict[str, Any]:
        """Stop monitoring."""
        try:
            if not self.monitor:
                raise HTTPException(status_code=404, detail='Monitor not available')
            await self.monitor.stop_monitoring()
            return {'success': True, 'message': 'Monitoring stopped'}
        except Exception as e:
            logger.exception(f'❌ Error stopping monitoring: {e}')
            raise HTTPException(status_code=500, detail=str(e))

    @traced(span_name='handle_websocket')
    async def _handle_websocket(self, websocket: WebSocket) -> None:
        """Handle WebSocket connections for real-time updates."""
        try:
            await websocket.accept()
            self.websocket_connections.append(websocket)
            logger.info('✅ WebSocket connection established')
            try:
                while True:
                    await asyncio.sleep(5)
                    if not self.monitor:
                        continue
                    metrics = await self._get_quality_metrics()
                    alerts = await self._get_alerts(limit=5)
                    update = {'type': 'update', 'timestamp': datetime.now().isoformat(), 'metrics': metrics, 'recent_alerts': alerts}
                    await websocket.send_text(json.dumps(update))
            except WebSocketDisconnect:
                logger.info('WebSocket connection closed')
            finally:
                if websocket in self.websocket_connections:
                    self.websocket_connections.remove(websocket)
        except Exception as e:
            logger.exception(f'❌ Error handling WebSocket: {e}')

    @traced(span_name='start_dashboard')
    async def start_dashboard(self) -> None:
        """Start the dashboard server."""
        if not FASTAPI_AVAILABLE:
            logger.error('❌ FastAPI not available - cannot start dashboard')
            return
        if not UVICORN_AVAILABLE:
            logger.error('❌ Uvicorn not available - cannot start dashboard server')
            return
        try:
            logger.info(f'🚀 Starting data quality dashboard on {self.config.host}:{self.config.port}')
            if self.monitor:
                await self.monitor.start_monitoring(symbols=['ETHUSDT', 'BTCUSDT'], exchanges=['BINANCE'], timeframes=['1m'])
            uvicorn.run(self.app, host=self.config.host, port=self.config.port, log_level='info')
        except Exception as e:
            logger.exception(f'❌ Error starting dashboard: {e}')

    @traced(span_name='stop_dashboard')
    async def stop_dashboard(self) -> None:
        """Stop the dashboard server."""
        try:
            if self.monitor:
                await self.monitor.stop_monitoring()
            logger.info('🛑 Data quality dashboard stopped')
        except Exception as e:
            logger.exception(f'❌ Error stopping dashboard: {e}')

async def start_data_quality_dashboard(data_cache_path: str='data_cache', host: str='0.0.0.0', port: int=8080) -> DataQualityDashboard:
    """Start the data quality dashboard with default configuration."""
    config = DashboardConfig(host=host, port=port)
    dashboard = DataQualityDashboard(data_cache_path, config)
    logger.info(f'🚀 Starting data quality dashboard on http://{host}:{port}')
    await dashboard.start_dashboard()
    return dashboard
if __name__ == '__main__':
    import asyncio

    async def main() -> None:
        dashboard = await start_data_quality_dashboard()
        try:
            await asyncio.sleep(float('inf'))
        except KeyboardInterrupt:
            await dashboard.stop_dashboard()
    asyncio.run(main())