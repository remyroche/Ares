"""
Monitoring API Endpoints

This module provides REST API endpoints for monitoring and managing
exchange OHLCV data processing operations.

Features:
- RESTful API endpoints
- Real-time monitoring data
- Configuration management
- Alert management
- Performance metrics
- Health checks
"""

from flask import Flask, jsonify, request, Response
from flask_cors import CORS
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
import threading
import time

# Import our monitoring components
from .monitoring_dashboard import monitoring_dashboard, get_dashboard_data, get_health_check
from .performance_monitor import performance_monitor, get_performance_summary, get_optimization_recommendations
from .config_manager import config_manager, get_config, update_config, get_exchange_config
from .data_validation_suite import advanced_data_validator, validate_ohlcv_data_quality

# Import src/utils/data utilities
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.logger import system_logger

logger = logging.getLogger(__name__)


class MonitoringAPI:
    """
    REST API for monitoring exchange OHLCV data processing.
    
    Provides comprehensive API endpoints for monitoring, configuration,
    and management of all system components.
    """
    
    def __init__(self, host: str = '0.0.0.0', port: int = 5000, debug: bool = False):
        """Initialize the monitoring API"""
        self.host = host
        self.port = port
        self.debug = debug
        
        # Create Flask app
        self.app = Flask(__name__)
        CORS(self.app)  # Enable CORS for web dashboard
        
        # Configure logging
        self.app.logger.setLevel(logging.INFO)
        
        # Register routes
        self._register_routes()
        
        # Start monitoring if not already running
        if not monitoring_dashboard.is_running:
            monitoring_dashboard.start_monitoring()
        
        self.logger = system_logger.getChild("MonitoringAPI")
        self.logger.info(f"✅ MonitoringAPI initialized on {host}:{port}")
    
    def _register_routes(self):
        """Register all API routes"""
        
        # Health check endpoints
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Health check endpoint"""
            try:
                health_data = get_health_check()
                status_code = 200 if health_data['status'] == 'healthy' else 503
                return jsonify(health_data), status_code
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/health/detailed', methods=['GET'])
        def detailed_health_check():
            """Detailed health check endpoint"""
            try:
                dashboard_data = get_dashboard_data()
                return jsonify(dashboard_data), 200
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        # Dashboard data endpoints
        @self.app.route('/dashboard', methods=['GET'])
        def get_dashboard():
            """Get dashboard data"""
            try:
                dashboard_data = get_dashboard_data()
                return jsonify(dashboard_data), 200
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/dashboard/real-time', methods=['GET'])
        def get_real_time_dashboard():
            """Get real-time dashboard data with Server-Sent Events"""
            def generate():
                while True:
                    try:
                        dashboard_data = get_dashboard_data()
                        yield f"data: {json.dumps(dashboard_data)}\n\n"
                        time.sleep(5)  # Update every 5 seconds
                    except Exception as e:
                        yield f"data: {json.dumps({'error': str(e)})}\n\n"
                        break
            
            return Response(generate(), mimetype='text/plain')
        
        # Performance metrics endpoints
        @self.app.route('/metrics/performance', methods=['GET'])
        def get_performance_metrics():
            """Get performance metrics"""
            try:
                metrics = get_performance_summary()
                return jsonify(metrics), 200
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/metrics/performance/export', methods=['GET'])
        def export_performance_metrics():
            """Export performance metrics"""
            try:
                format_type = request.args.get('format', 'json')
                filepath = f"performance_metrics_{int(time.time())}.{format_type}"
                
                performance_monitor.export_metrics(filepath, format_type)
                
                return jsonify({
                    'message': 'Performance metrics exported successfully',
                    'filepath': filepath
                }), 200
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/metrics/optimization', methods=['GET'])
        def get_optimization_recommendations():
            """Get optimization recommendations"""
            try:
                recommendations = get_optimization_recommendations()
                return jsonify({'recommendations': recommendations}), 200
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        # Configuration endpoints
        @self.app.route('/config', methods=['GET'])
        def get_configuration():
            """Get current configuration"""
            try:
                config = get_config()
                return jsonify(config.__dict__), 200
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/config', methods=['PUT'])
        def update_configuration():
            """Update configuration"""
            try:
                updates = request.get_json()
                if not updates:
                    return jsonify({'error': 'No configuration updates provided'}), 400
                
                success = update_config(updates)
                if success:
                    return jsonify({'message': 'Configuration updated successfully'}), 200
                else:
                    return jsonify({'error': 'Failed to update configuration'}), 400
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/config/exchanges/<exchange_name>', methods=['GET'])
        def get_exchange_configuration(exchange_name):
            """Get exchange-specific configuration"""
            try:
                config = get_exchange_config(exchange_name)
                if config:
                    return jsonify(config.__dict__), 200
                else:
                    return jsonify({'error': f'Exchange {exchange_name} not found'}), 404
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/config/exchanges/<exchange_name>', methods=['PUT'])
        def update_exchange_configuration(exchange_name):
            """Update exchange-specific configuration"""
            try:
                updates = request.get_json()
                if not updates:
                    return jsonify({'error': 'No configuration updates provided'}), 400
                
                success = config_manager.update_exchange_config(exchange_name, updates)
                if success:
                    return jsonify({'message': f'Configuration updated for {exchange_name}'}), 200
                else:
                    return jsonify({'error': f'Failed to update configuration for {exchange_name}'}), 400
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        # Exchange status endpoints
        @self.app.route('/exchanges', methods=['GET'])
        def get_exchanges():
            """Get list of available exchanges"""
            try:
                config = get_config()
                exchanges = []
                
                for name, exchange_config in config.exchanges.items():
                    exchanges.append({
                        'name': name,
                        'enabled': exchange_config.enabled,
                        'base_url': exchange_config.base_url,
                        'has_credentials': bool(exchange_config.api_key and exchange_config.api_secret)
                    })
                
                return jsonify({'exchanges': exchanges}), 200
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/exchanges/<exchange_name>/status', methods=['GET'])
        def get_exchange_status(exchange_name):
            """Get status of specific exchange"""
            try:
                dashboard_data = get_dashboard_data()
                exchange_metrics = dashboard_data.get('exchange_metrics', {})
                
                if exchange_name not in exchange_metrics:
                    return jsonify({'error': f'Exchange {exchange_name} not found'}), 404
                
                return jsonify({
                    'exchange': exchange_name,
                    'status': exchange_metrics[exchange_name]
                }), 200
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        # Data validation endpoints
        @self.app.route('/validation/validate', methods=['POST'])
        def validate_data():
            """Validate OHLCV data"""
            try:
                data = request.get_json()
                if not data:
                    return jsonify({'error': 'No data provided'}), 400
                
                # Extract parameters
                ohlcv_data = data.get('data')
                exchange = data.get('exchange', 'binance')
                validation_level = data.get('validation_level', 'standard')
                
                if not ohlcv_data:
                    return jsonify({'error': 'No OHLCV data provided'}), 400
                
                # Convert to DataFrame if needed
                import pandas as pd
                if isinstance(ohlcv_data, list):
                    df = pd.DataFrame(ohlcv_data)
                else:
                    df = pd.DataFrame([ohlcv_data])
                
                # Validate data
                result = validate_ohlcv_data_quality(df, exchange, validation_level)
                
                return jsonify({
                    'is_valid': result.is_valid,
                    'quality_score': result.quality_score,
                    'anomalies': result.anomalies,
                    'warnings': result.warnings,
                    'errors': result.errors,
                    'recommendations': result.recommendations
                }), 200
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        # Alert management endpoints
        @self.app.route('/alerts', methods=['GET'])
        def get_alerts():
            """Get all alerts"""
            try:
                dashboard_data = get_dashboard_data()
                alerts_summary = dashboard_data.get('alerts', {})
                return jsonify(alerts_summary), 200
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/alerts/<alert_id>/resolve', methods=['POST'])
        def resolve_alert(alert_id):
            """Resolve an alert"""
            try:
                success = monitoring_dashboard.resolve_alert(alert_id)
                if success:
                    return jsonify({'message': f'Alert {alert_id} resolved'}), 200
                else:
                    return jsonify({'error': f'Alert {alert_id} not found or already resolved'}), 404
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        # System control endpoints
        @self.app.route('/system/start-monitoring', methods=['POST'])
        def start_monitoring():
            """Start monitoring"""
            try:
                interval = request.json.get('interval', 5.0) if request.json else 5.0
                monitoring_dashboard.start_monitoring(interval)
                return jsonify({'message': 'Monitoring started'}), 200
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/system/stop-monitoring', methods=['POST'])
        def stop_monitoring():
            """Stop monitoring"""
            try:
                monitoring_dashboard.stop_monitoring()
                return jsonify({'message': 'Monitoring stopped'}), 200
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/system/restart', methods=['POST'])
        def restart_system():
            """Restart monitoring system"""
            try:
                monitoring_dashboard.stop_monitoring()
                time.sleep(2)  # Wait for cleanup
                monitoring_dashboard.start_monitoring()
                return jsonify({'message': 'System restarted'}), 200
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        # Error handling
        @self.app.errorhandler(404)
        def not_found(error):
            return jsonify({'error': 'Endpoint not found'}), 404
        
        @self.app.errorhandler(500)
        def internal_error(error):
            return jsonify({'error': 'Internal server error'}), 500
    
    def run(self, threaded: bool = True):
        """Run the API server"""
        try:
            self.logger.info(f"🚀 Starting Monitoring API server on {self.host}:{self.port}")
            self.app.run(
                host=self.host,
                port=self.port,
                debug=self.debug,
                threaded=threaded
            )
        except Exception as e:
            self.logger.error(f"Failed to start API server: {e}")
            raise
    
    def run_async(self):
        """Run the API server in a separate thread"""
        def run_server():
            self.run(threaded=False)
        
        api_thread = threading.Thread(target=run_server, daemon=True)
        api_thread.start()
        
        self.logger.info(f"🚀 Monitoring API server started in background on {self.host}:{self.port}")
        return api_thread


# Global API instance
monitoring_api = MonitoringAPI()


# Convenience functions
def start_monitoring_api(host: str = '0.0.0.0', port: int = 5000, debug: bool = False):
    """Start the monitoring API server"""
    api = MonitoringAPI(host, port, debug)
    api.run()


def start_monitoring_api_async(host: str = '0.0.0.0', port: int = 5000, debug: bool = False):
    """Start the monitoring API server asynchronously"""
    api = MonitoringAPI(host, port, debug)
    return api.run_async()


# Example usage and testing
if __name__ == "__main__":
    # Start the monitoring API
    start_monitoring_api(debug=True)