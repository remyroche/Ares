#!/usr/bin/env python3
"""Data Quality Dashboard Web Interface.

This module provides a web-based dashboard for monitoring and managing data quality.
It includes real-time metrics, alert management, and quality control features.
"""

import asyncio
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.centralized_decorators import (
    handle_errors,
    with_tracing_span,
)
from src.utils.logger import system_logger

logger = system_logger.getChild("DataQualityDashboard")

try:
    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logger.warning("⚠️ FastAPI not available - dashboard will use basic HTTP server")

try:
    import uvicorn
    UVICORN_AVAILABLE = True
except ImportError:
    UVICORN_AVAILABLE = False
    logger.warning("⚠️ Uvicorn not available - dashboard server not available")


class DashboardConfig(BaseModel):
    """Dashboard configuration."""
    host: str = "0.0.0.0"
    port: int = 8080
    refresh_interval: int = 30  # seconds
    max_alerts: int = 100
    enable_websocket: bool = True


class DataQualityDashboard:
    """Web-based data quality dashboard."""

    def __init__(self, data_cache_path: str = "data_cache", config: Optional[DashboardConfig] = None):
        self.data_cache_path = Path(data_cache_path)
        self.data_cache_path.mkdir(exist_ok=True)
        self.config = config or DashboardConfig()
        
        # Initialize components
        self.quality_manager = None
        self.monitor = None
        self.app = None
        self.websocket_connections: List[WebSocket] = []
        
        self._initialize_components()

    def _initialize_components(self) -> None:
        """Initialize dashboard components."""
        try:
            from .enhanced_data_quality_manager import EnhancedDataQualityManager
            self.quality_manager = EnhancedDataQualityManager(str(self.data_cache_path))
            logger.info("✅ Enhanced data quality manager initialized for dashboard")
        except ImportError as e:
            logger.warning(f"⚠️ Could not import EnhancedDataQualityManager: {e}")

        try:
            from .data_quality_monitor import DataQualityMonitor
            self.monitor = DataQualityMonitor(str(self.data_cache_path))
            logger.info("✅ Data quality monitor initialized for dashboard")
        except ImportError as e:
            logger.warning(f"⚠️ Could not import DataQualityMonitor: {e}")

        if FASTAPI_AVAILABLE:
            self._create_fastapi_app()
        else:
            logger.warning("⚠️ FastAPI not available - dashboard will be limited")

    def _create_fastapi_app(self) -> None:
        """Create FastAPI application with routes."""
        self.app = FastAPI(
            title="Data Quality Dashboard",
            description="Real-time data quality monitoring and management",
            version="1.0.0"
        )
        
        # Add routes
        self._add_routes()
        
        # Add static files
        static_dir = self.data_cache_path / "dashboard_static"
        static_dir.mkdir(exist_ok=True)
        self.app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    def _add_routes(self) -> None:
        """Add API routes to the FastAPI app."""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard_home():
            """Main dashboard page."""
            return self._generate_dashboard_html()

        @self.app.get("/api/status")
        async def get_status():
            """Get overall system status."""
            return await self._get_system_status()

        @self.app.get("/api/metrics")
        async def get_metrics():
            """Get current quality metrics."""
            return await self._get_quality_metrics()

        @self.app.get("/api/alerts")
        async def get_alerts(
            symbol: Optional[str] = None,
            exchange: Optional[str] = None,
            severity: Optional[str] = None,
            limit: int = 50
        ):
            """Get filtered alerts."""
            return await self._get_alerts(symbol, exchange, severity, limit)

        @self.app.post("/api/alerts/{alert_id}/acknowledge")
        async def acknowledge_alert(alert_id: int):
            """Acknowledge an alert."""
            return await self._acknowledge_alert(alert_id)

        @self.app.post("/api/alerts/{alert_id}/resolve")
        async def resolve_alert(alert_id: int):
            """Resolve an alert."""
            return await self._resolve_alert(alert_id)

        @self.app.post("/api/quality-check")
        async def run_quality_check(symbol: str, exchange: str, timeframe: str = "1m"):
            """Run a quality check for specific data."""
            return await self._run_quality_check(symbol, exchange, timeframe)

        @self.app.get("/api/monitoring/status")
        async def get_monitoring_status():
            """Get monitoring status."""
            return await self._get_monitoring_status()

        @self.app.post("/api/monitoring/start")
        async def start_monitoring(symbols: List[str], exchanges: List[str], timeframes: List[str]):
            """Start monitoring."""
            return await self._start_monitoring(symbols, exchanges, timeframes)

        @self.app.post("/api/monitoring/stop")
        async def stop_monitoring():
            """Stop monitoring."""
            return await self._stop_monitoring()

        if self.config.enable_websocket:
            @self.app.websocket("/ws")
            async def websocket_endpoint(websocket: WebSocket):
                """WebSocket endpoint for real-time updates."""
                await self._handle_websocket(websocket)

    def _generate_dashboard_html(self) -> str:
        """Generate the main dashboard HTML."""
        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Data Quality Dashboard</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
        }}
        .header p {{
            margin: 10px 0 0 0;
            opacity: 0.9;
        }}
        .dashboard-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        .card {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .card h3 {{
            margin-top: 0;
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        .metric {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 0;
            border-bottom: 1px solid #eee;
        }}
        .metric:last-child {{
            border-bottom: none;
        }}
        .metric-value {{
            font-weight: bold;
            font-size: 1.2em;
        }}
        .status-good {{ color: #28a745; }}
        .status-warning {{ color: #ffc107; }}
        .status-error {{ color: #dc3545; }}
        .alerts-list {{
            max-height: 400px;
            overflow-y: auto;
        }}
        .alert-item {{
            padding: 10px;
            margin: 5px 0;
            border-radius: 5px;
            border-left: 4px solid #ddd;
        }}
        .alert-high {{ border-left-color: #dc3545; background-color: #f8d7da; }}
        .alert-medium {{ border-left-color: #ffc107; background-color: #fff3cd; }}
        .alert-low {{ border-left-color: #28a745; background-color: #d4edda; }}
        .controls {{
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }}
        .btn {{
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
        }}
        .btn-primary {{ background-color: #667eea; color: white; }}
        .btn-success {{ background-color: #28a745; color: white; }}
        .btn-warning {{ background-color: #ffc107; color: black; }}
        .btn-danger {{ background-color: #dc3545; color: white; }}
        .refresh-info {{
            text-align: center;
            color: #666;
            font-size: 0.9em;
            margin-top: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Data Quality Dashboard</h1>
            <p>Real-time monitoring and management of data quality across the training pipeline</p>
        </div>

        <div class="controls">
            <button class="btn btn-primary" onclick="refreshDashboard()">🔄 Refresh</button>
            <button class="btn btn-success" onclick="startMonitoring()">▶️ Start Monitoring</button>
            <button class="btn btn-warning" onclick="stopMonitoring()">⏹️ Stop Monitoring</button>
            <button class="btn btn-primary" onclick="runQualityCheck()">🔍 Run Quality Check</button>
        </div>

        <div class="dashboard-grid">
            <div class="card">
                <h3>📈 System Status</h3>
                <div id="system-status">
                    <div class="metric">
                        <span>Overall Status:</span>
                        <span class="metric-value status-good" id="overall-status">Loading...</span>
                    </div>
                    <div class="metric">
                        <span>Monitoring Active:</span>
                        <span class="metric-value" id="monitoring-status">Loading...</span>
                    </div>
                    <div class="metric">
                        <span>Last Update:</span>
                        <span class="metric-value" id="last-update">Loading...</span>
                    </div>
                </div>
            </div>

            <div class="card">
                <h3>📊 Quality Metrics</h3>
                <div id="quality-metrics">
                    <div class="metric">
                        <span>Data Gaps:</span>
                        <span class="metric-value" id="data-gaps">Loading...</span>
                    </div>
                    <div class="metric">
                        <span>Format Issues:</span>
                        <span class="metric-value" id="format-issues">Loading...</span>
                    </div>
                    <div class="metric">
                        <span>Data Freshness:</span>
                        <span class="metric-value" id="data-freshness">Loading...</span>
                    </div>
                    <div class="metric">
                        <span>Step3/4 Ready:</span>
                        <span class="metric-value" id="step-ready">Loading...</span>
                    </div>
                </div>
            </div>

            <div class="card">
                <h3>🚨 Recent Alerts</h3>
                <div class="alerts-list" id="recent-alerts">
                    <div class="metric">
                        <span>Loading alerts...</span>
                    </div>
                </div>
            </div>

            <div class="card">
                <h3>⚙️ Quick Actions</h3>
                <div>
                    <button class="btn btn-primary" style="width: 100%; margin-bottom: 10px;" onclick="checkETHUSDT()">
                        Check ETHUSDT
                    </button>
                    <button class="btn btn-primary" style="width: 100%; margin-bottom: 10px;" onclick="checkBTCUSDT()">
                        Check BTCUSDT
                    </button>
                    <button class="btn btn-success" style="width: 100%; margin-bottom: 10px;" onclick="runStep1()">
                        Run Step1
                    </button>
                    <button class="btn btn-success" style="width: 100%; margin-bottom: 10px;" onclick="runStep1_5()">
                        Run Step1.5
                    </button>
                </div>
            </div>
        </div>

        <div class="refresh-info">
            Auto-refresh every {self.config.refresh_interval} seconds
        </div>
    </div>

    <script>
        let refreshInterval;
        
        // Initialize dashboard
        document.addEventListener('DOMContentLoaded', function() {{
            refreshDashboard();
            startAutoRefresh();
        }});

        function startAutoRefresh() {{
            refreshInterval = setInterval(refreshDashboard, {self.config.refresh_interval * 1000});
        }}

        function stopAutoRefresh() {{
            if (refreshInterval) {{
                clearInterval(refreshInterval);
            }}
        }}

        async function refreshDashboard() {{
            try {{
                // Update system status
                const statusResponse = await fetch('/api/status');
                const status = await statusResponse.json();
                updateSystemStatus(status);

                // Update quality metrics
                const metricsResponse = await fetch('/api/metrics');
                const metrics = await metricsResponse.json();
                updateQualityMetrics(metrics);

                // Update alerts
                const alertsResponse = await fetch('/api/alerts?limit=10');
                const alerts = await alertsResponse.json();
                updateAlerts(alerts);

            }} catch (error) {{
                console.error('Error refreshing dashboard:', error);
            }}
        }}

        function updateSystemStatus(status) {{
            document.getElementById('overall-status').textContent = status.overall_status;
            document.getElementById('overall-status').className = `metric-value status-${{status.overall_status === 'healthy' ? 'good' : 'error'}}`;
            document.getElementById('monitoring-status').textContent = status.monitoring_active ? 'Active' : 'Inactive';
            document.getElementById('last-update').textContent = status.last_update;
        }}

        function updateQualityMetrics(metrics) {{
            document.getElementById('data-gaps').textContent = metrics.total_gaps || 0;
            document.getElementById('format-issues').textContent = metrics.format_issues || 0;
            document.getElementById('data-freshness').textContent = metrics.data_freshness || 'Unknown';
            document.getElementById('step-ready').textContent = metrics.step3_step4_ready ? 'Yes' : 'No';
        }}

        function updateAlerts(alerts) {{
            const alertsContainer = document.getElementById('recent-alerts');
            alertsContainer.innerHTML = '';

            if (alerts.length === 0) {{
                alertsContainer.innerHTML = '<div class="metric"><span>No recent alerts</span></div>';
                return;
            }}

            alerts.forEach(alert => {{
                const alertDiv = document.createElement('div');
                alertDiv.className = `alert-item alert-${{alert.severity}}`;
                alertDiv.innerHTML = `
                    <strong>${{alert.alert_type}}</strong><br>
                    ${{alert.message}}<br>
                    <small>${{alert.timestamp}}</small>
                `;
                alertsContainer.appendChild(alertDiv);
            }});
        }}

        async function startMonitoring() {{
            try {{
                const response = await fetch('/api/monitoring/start', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{
                        symbols: ['ETHUSDT', 'BTCUSDT'],
                        exchanges: ['BINANCE'],
                        timeframes: ['1m']
                    }})
                }});
                const result = await response.json();
                alert(result.message);
                refreshDashboard();
            }} catch (error) {{
                console.error('Error starting monitoring:', error);
                alert('Error starting monitoring');
            }}
        }}

        async function stopMonitoring() {{
            try {{
                const response = await fetch('/api/monitoring/stop', {{ method: 'POST' }});
                const result = await response.json();
                alert(result.message);
                refreshDashboard();
            }} catch (error) {{
                console.error('Error stopping monitoring:', error);
                alert('Error stopping monitoring');
            }}
        }}

        async function runQualityCheck() {{
            const symbol = prompt('Enter symbol (e.g., ETHUSDT):', 'ETHUSDT');
            if (symbol) {{
                try {{
                    const response = await fetch(`/api/quality-check?symbol=${{symbol}}&exchange=BINANCE&timeframe=1m`);
                    const result = await response.json();
                    alert(`Quality check result: ${{result.success ? 'Success' : 'Failed'}}`);
                    refreshDashboard();
                }} catch (error) {{
                    console.error('Error running quality check:', error);
                    alert('Error running quality check');
                }}
            }}
        }}

        async function checkETHUSDT() {{
            try {{
                const response = await fetch('/api/quality-check?symbol=ETHUSDT&exchange=BINANCE&timeframe=1m');
                const result = await response.json();
                alert(`ETHUSDT quality check: ${{result.success ? 'Success' : 'Failed'}}`);
                refreshDashboard();
            }} catch (error) {{
                console.error('Error checking ETHUSDT:', error);
                alert('Error checking ETHUSDT');
            }}
        }}

        async function checkBTCUSDT() {{
            try {{
                const response = await fetch('/api/quality-check?symbol=BTCUSDT&exchange=BINANCE&timeframe=1m');
                const result = await response.json();
                alert(`BTCUSDT quality check: ${{result.success ? 'Success' : 'Failed'}}`);
                refreshDashboard();
            }} catch (error) {{
                console.error('Error checking BTCUSDT:', error);
                alert('Error checking BTCUSDT');
            }}
        }}

        async function runStep1() {{
            try {{
                const response = await fetch('/api/run-step1', {{ method: 'POST' }});
                const result = await response.json();
                alert(`Step1 result: ${{result.success ? 'Success' : 'Failed'}}`);
                refreshDashboard();
            }} catch (error) {{
                console.error('Error running Step1:', error);
                alert('Error running Step1');
            }}
        }}

        async function runStep1_5() {{
            try {{
                const response = await fetch('/api/run-step01_5', {{ method: 'POST' }});
                const result = await response.json();
                alert(`Step1.5 result: ${{result.success ? 'Success' : 'Failed' }}`);
                refreshDashboard();
            }} catch (error) {{
                console.error('Error running Step1.5:', error);
                alert('Error running Step1.5');
            }}
        }}
    </script>
</body>
</html>
        """

    @with_tracing_span("get_system_status")
    async def _get_system_status(self) -> Dict[str, Any]:
        """Get overall system status."""
        try:
            status = {
                "overall_status": "healthy",
                "monitoring_active": False,
                "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "components": {}
            }

            # Check quality manager
            if self.quality_manager:
                status["components"]["quality_manager"] = "available"
            else:
                status["components"]["quality_manager"] = "unavailable"
                status["overall_status"] = "degraded"

            # Check monitor
            if self.monitor:
                status["components"]["monitor"] = "available"
                status["monitoring_active"] = self.monitor.monitoring_active
            else:
                status["components"]["monitor"] = "unavailable"
                status["overall_status"] = "degraded"

            return status

        except Exception as e:
            logger.exception(f"❌ Error getting system status: {e}")
            return {
                "overall_status": "error",
                "error": str(e),
                "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

    @with_tracing_span("get_quality_metrics")
    async def _get_quality_metrics(self) -> Dict[str, Any]:
        """Get current quality metrics."""
        try:
            metrics = {
                "total_gaps": 0,
                "format_issues": 0,
                "data_freshness": "unknown",
                "step3_step4_ready": False,
                "last_check": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            # Get metrics from monitor if available
            if self.monitor:
                monitor_metrics = self.monitor.get_performance_metrics()
                metrics.update(monitor_metrics)

            return metrics

        except Exception as e:
            logger.exception(f"❌ Error getting quality metrics: {e}")
            return {
                "error": str(e),
                "last_check": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

    @with_tracing_span("get_alerts")
    async def _get_alerts(
        self,
        symbol: Optional[str] = None,
        exchange: Optional[str] = None,
        severity: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get filtered alerts."""
        try:
            if not self.monitor:
                return []

            alerts = self.monitor.get_alerts(
                symbol=symbol,
                exchange=exchange,
                severity=severity,
                limit=limit
            )

            return [alert.to_dict() for alert in alerts]

        except Exception as e:
            logger.exception(f"❌ Error getting alerts: {e}")
            return []

    @with_tracing_span("acknowledge_alert")
    async def _acknowledge_alert(self, alert_id: int) -> Dict[str, Any]:
        """Acknowledge an alert."""
        try:
            if not self.monitor:
                raise HTTPException(status_code=404, detail="Monitor not available")

            success = self.monitor.acknowledge_alert(alert_id)
            return {
                "success": success,
                "message": "Alert acknowledged" if success else "Alert not found"
            }

        except Exception as e:
            logger.exception(f"❌ Error acknowledging alert: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @with_tracing_span("resolve_alert")
    async def _resolve_alert(self, alert_id: int) -> Dict[str, Any]:
        """Resolve an alert."""
        try:
            if not self.monitor:
                raise HTTPException(status_code=404, detail="Monitor not available")

            success = self.monitor.resolve_alert(alert_id)
            return {
                "success": success,
                "message": "Alert resolved" if success else "Alert not found"
            }

        except Exception as e:
            logger.exception(f"❌ Error resolving alert: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @with_tracing_span("run_quality_check")
    async def _run_quality_check(self, symbol: str, exchange: str, timeframe: str) -> Dict[str, Any]:
        """Run a quality check for specific data."""
        try:
            if not self.quality_manager:
                raise HTTPException(status_code=404, detail="Quality manager not available")

            results = await self.quality_manager.comprehensive_quality_check(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                check_gaps=True,
                fill_gaps=False,
                validate_format=True
            )

            return {
                "success": results.get("success", False),
                "results": results,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.exception(f"❌ Error running quality check: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @with_tracing_span("get_monitoring_status")
    async def _get_monitoring_status(self) -> Dict[str, Any]:
        """Get monitoring status."""
        try:
            if not self.monitor:
                return {"active": False, "error": "Monitor not available"}

            return {
                "active": self.monitor.monitoring_active,
                "interval": self.monitor.monitoring_interval,
                "metrics": self.monitor.get_performance_metrics()
            }

        except Exception as e:
            logger.exception(f"❌ Error getting monitoring status: {e}")
            return {"active": False, "error": str(e)}

    @with_tracing_span("start_monitoring")
    async def _start_monitoring(self, symbols: List[str], exchanges: List[str], timeframes: List[str]) -> Dict[str, Any]:
        """Start monitoring."""
        try:
            if not self.monitor:
                raise HTTPException(status_code=404, detail="Monitor not available")

            success = await self.monitor.start_monitoring(symbols, exchanges, timeframes)
            return {
                "success": success,
                "message": "Monitoring started" if success else "Failed to start monitoring"
            }

        except Exception as e:
            logger.exception(f"❌ Error starting monitoring: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @with_tracing_span("stop_monitoring")
    async def _stop_monitoring(self) -> Dict[str, Any]:
        """Stop monitoring."""
        try:
            if not self.monitor:
                raise HTTPException(status_code=404, detail="Monitor not available")

            await self.monitor.stop_monitoring()
            return {
                "success": True,
                "message": "Monitoring stopped"
            }

        except Exception as e:
            logger.exception(f"❌ Error stopping monitoring: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @with_tracing_span("handle_websocket")
    async def _handle_websocket(self, websocket: WebSocket) -> None:
        """Handle WebSocket connections for real-time updates."""
        try:
            await websocket.accept()
            self.websocket_connections.append(websocket)
            logger.info("✅ WebSocket connection established")

            try:
                while True:
                    # Send periodic updates
                    await asyncio.sleep(5)
                    
                    if not self.monitor:
                        continue

                    # Get latest metrics
                    metrics = await self._get_quality_metrics()
                    alerts = await self._get_alerts(limit=5)
                    
                    update = {
                        "type": "update",
                        "timestamp": datetime.now().isoformat(),
                        "metrics": metrics,
                        "recent_alerts": alerts
                    }
                    
                    await websocket.send_text(json.dumps(update))

            except WebSocketDisconnect:
                logger.info("WebSocket connection closed")
            finally:
                if websocket in self.websocket_connections:
                    self.websocket_connections.remove(websocket)

        except Exception as e:
            logger.exception(f"❌ Error handling WebSocket: {e}")

    @with_tracing_span("start_dashboard")
    async def start_dashboard(self) -> None:
        """Start the dashboard server."""
        if not FASTAPI_AVAILABLE:
            logger.error("❌ FastAPI not available - cannot start dashboard")
            return

        if not UVICORN_AVAILABLE:
            logger.error("❌ Uvicorn not available - cannot start dashboard server")
            return

        try:
            logger.info(f"🚀 Starting data quality dashboard on {self.config.host}:{self.config.port}")
            
            # Start monitoring if monitor is available
            if self.monitor:
                await self.monitor.start_monitoring(
                    symbols=["ETHUSDT", "BTCUSDT"],
                    exchanges=["BINANCE"],
                    timeframes=["1m"]
                )

            # Start the server
            uvicorn.run(
                self.app,
                host=self.config.host,
                port=self.config.port,
                log_level="info"
            )

        except Exception as e:
            logger.exception(f"❌ Error starting dashboard: {e}")

    @with_tracing_span("stop_dashboard")
    async def stop_dashboard(self) -> None:
        """Stop the dashboard server."""
        try:
            if self.monitor:
                await self.monitor.stop_monitoring()
            
            logger.info("🛑 Data quality dashboard stopped")

        except Exception as e:
            logger.exception(f"❌ Error stopping dashboard: {e}")


# Convenience functions
async def start_data_quality_dashboard(
    data_cache_path: str = "data_cache",
    host: str = "0.0.0.0",
    port: int = 8080
) -> DataQualityDashboard:
    """Start the data quality dashboard with default configuration."""
    config = DashboardConfig(host=host, port=port)
    dashboard = DataQualityDashboard(data_cache_path, config)
    
    logger.info(f"🚀 Starting data quality dashboard on http://{host}:{port}")
    await dashboard.start_dashboard()
    
    return dashboard


if __name__ == "__main__":
    import asyncio

    async def main():
        dashboard = await start_data_quality_dashboard()
        
        try:
            # Keep the dashboard running
            await asyncio.sleep(float('inf'))
        except KeyboardInterrupt:
            await dashboard.stop_dashboard()

    asyncio.run(main())