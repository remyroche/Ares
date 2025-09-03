#!/usr/bin/env python3
"""Real-time Regime Monitoring Dashboard and Infrastructure."""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
import json
from collections import defaultdict, deque
import websocket
import threading
from dataclasses import dataclass, asdict

from src.utils.logger import system_logger
from src.utils.common_operations import ensure_directory, safe_json_dump
from src.monitoring.regime_performance_tracker import RegimePerformanceTracker

logger = system_logger.getChild("RegimeMonitoring")


@dataclass
class RegimeAlert:
    """Represents a regime monitoring alert."""
    timestamp: datetime
    alert_type: str  # 'transition', 'confidence_drop', 'performance_degradation', 'anomaly'
    severity: str    # 'info', 'warning', 'critical'
    regime: str
    message: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'alert_type': self.alert_type,
            'severity': self.severity,
            'regime': self.regime,
            'message': self.message,
            'metadata': self.metadata
        }


class RegimeMonitoringDashboard:
    """Real-time monitoring dashboard for regime analysis."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild("RegimeMonitoringDashboard")
        
        # Initialize components
        self.performance_tracker = RegimePerformanceTracker(config)
        
        # Real-time data storage
        self.current_regime = {}  # symbol -> regime
        self.regime_confidence = {}  # symbol -> confidence
        self.regime_history = defaultdict(lambda: deque(maxlen=1000))
        
        # Alert system
        self.alerts = deque(maxlen=100)
        self.alert_handlers = {
            'console': self._handle_console_alert,
            'file': self._handle_file_alert,
            'webhook': self._handle_webhook_alert
        }
        
        # Metrics storage
        self.real_time_metrics = defaultdict(dict)
        self.update_frequency = config.get('update_frequency', 60)  # seconds
        
        # Dashboard state
        self.is_running = False
        self.last_update = {}
        
        # Initialize storage
        self.dashboard_dir = Path(config.get('dashboard_dir', 'dashboard'))
        ensure_directory(self.dashboard_dir)
        
    async def start(self):
        """Start the monitoring dashboard."""
        
        self.logger.info("Starting regime monitoring dashboard...")
        self.is_running = True
        
        # Start monitoring tasks
        tasks = [
            self._monitor_regimes(),
            self._update_metrics(),
            self._check_alerts(),
            self._save_dashboard_state()
        ]
        
        await asyncio.gather(*tasks)
    
    async def stop(self):
        """Stop the monitoring dashboard."""
        
        self.logger.info("Stopping regime monitoring dashboard...")
        self.is_running = False
    
    async def _monitor_regimes(self):
        """Monitor regime changes and confidence."""
        
        while self.is_running:
            try:
                # Get symbols to monitor
                symbols = self.config.get('symbols', ['BTCUSDT'])
                
                for symbol in symbols:
                    # Get current regime data
                    regime_data = await self._get_current_regime(symbol)
                    
                    if regime_data:
                        # Check for regime change
                        if symbol in self.current_regime:
                            if self.current_regime[symbol] != regime_data['regime']:
                                await self._handle_regime_transition(
                                    symbol,
                                    self.current_regime[symbol],
                                    regime_data['regime'],
                                    regime_data['confidence']
                                )
                        
                        # Update current state
                        self.current_regime[symbol] = regime_data['regime']
                        self.regime_confidence[symbol] = regime_data['confidence']
                        
                        # Store history
                        self.regime_history[symbol].append({
                            'timestamp': datetime.now(),
                            'regime': regime_data['regime'],
                            'confidence': regime_data['confidence']
                        })
                        
                        # Check confidence threshold
                        if regime_data['confidence'] < 0.7:
                            await self._create_alert(
                                alert_type='confidence_drop',
                                severity='warning',
                                regime=regime_data['regime'],
                                message=f"Low regime confidence for {symbol}: {regime_data['confidence']:.2f}",
                                metadata={'symbol': symbol, 'confidence': regime_data['confidence']}
                            )
                
                await asyncio.sleep(self.update_frequency)
                
            except Exception as e:
                self.logger.error(f"Error in regime monitoring: {e}")
                await asyncio.sleep(10)  # Short pause before retry
    
    async def _get_current_regime(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current regime for a symbol."""
        
        try:
            # Load latest regime analysis
            regime_path = Path(f"data/regime_analysis/{symbol}_current_regime.json")
            
            if regime_path.exists():
                with open(regime_path, 'r') as f:
                    return json.load(f)
            
            # Fallback: calculate from latest data
            # This would integrate with the HMM regime discovery
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting regime for {symbol}: {e}")
            return None
    
    async def _handle_regime_transition(self, symbol: str, from_regime: str, 
                                      to_regime: str, confidence: float):
        """Handle regime transition event."""
        
        self.logger.info(f"Regime transition detected for {symbol}: {from_regime} -> {to_regime}")
        
        # Track transition
        await self.performance_tracker.track_regime_transition({
            'timestamp': datetime.now(),
            'symbol': symbol,
            'from_regime': from_regime,
            'to_regime': to_regime,
            'confidence': confidence,
            'detection_lag_minutes': self._estimate_detection_lag(symbol)
        })
        
        # Create alert
        await self._create_alert(
            alert_type='transition',
            severity='info' if confidence > 0.8 else 'warning',
            regime=to_regime,
            message=f"Regime transition for {symbol}: {from_regime} -> {to_regime} (confidence: {confidence:.2f})",
            metadata={
                'symbol': symbol,
                'from_regime': from_regime,
                'to_regime': to_regime,
                'confidence': confidence
            }
        )
    
    def _estimate_detection_lag(self, symbol: str) -> int:
        """Estimate regime detection lag in minutes."""
        
        # This would compare actual transition time with detection time
        # For now, return a placeholder
        return 60  # 1 hour lag
    
    async def _update_metrics(self):
        """Update real-time metrics."""
        
        while self.is_running:
            try:
                symbols = self.config.get('symbols', ['BTCUSDT'])
                
                for symbol in symbols:
                    # Get regime-specific metrics
                    metrics = await self.performance_tracker.calculate_regime_metrics(symbol, 1)
                    
                    # Update real-time storage
                    self.real_time_metrics[symbol] = {
                        'timestamp': datetime.now(),
                        'current_regime': self.current_regime.get(symbol, 'unknown'),
                        'regime_confidence': self.regime_confidence.get(symbol, 0),
                        'metrics': metrics
                    }
                    
                    # Check for performance degradation
                    await self._check_performance_degradation(symbol, metrics)
                
                await asyncio.sleep(self.update_frequency)
                
            except Exception as e:
                self.logger.error(f"Error updating metrics: {e}")
                await asyncio.sleep(10)
    
    async def _check_performance_degradation(self, symbol: str, metrics: Dict[str, Any]):
        """Check for performance degradation by regime."""
        
        for regime in ['bull', 'bear', 'sideways']:
            if regime not in metrics:
                continue
                
            regime_metrics = metrics[regime]
            
            # Check Sharpe ratio
            if regime_metrics['sharpe_ratio'] < 0.3:
                await self._create_alert(
                    alert_type='performance_degradation',
                    severity='warning',
                    regime=regime,
                    message=f"Low Sharpe ratio for {symbol} in {regime} regime: {regime_metrics['sharpe_ratio']:.2f}",
                    metadata={
                        'symbol': symbol,
                        'sharpe_ratio': regime_metrics['sharpe_ratio'],
                        'trade_count': regime_metrics['trade_count']
                    }
                )
            
            # Check win rate
            if regime_metrics['win_rate'] < 0.35:
                await self._create_alert(
                    alert_type='performance_degradation',
                    severity='critical' if regime_metrics['win_rate'] < 0.25 else 'warning',
                    regime=regime,
                    message=f"Low win rate for {symbol} in {regime} regime: {regime_metrics['win_rate']:.2%}",
                    metadata={
                        'symbol': symbol,
                        'win_rate': regime_metrics['win_rate'],
                        'trade_count': regime_metrics['trade_count']
                    }
                )
    
    async def _check_alerts(self):
        """Process and handle alerts."""
        
        while self.is_running:
            try:
                # Process any pending alerts
                while self.alerts:
                    alert = self.alerts.popleft()
                    
                    # Handle alert based on configured handlers
                    for handler_name in self.config.get('alert_handlers', ['console']):
                        if handler_name in self.alert_handlers:
                            await self.alert_handlers[handler_name](alert)
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Error processing alerts: {e}")
                await asyncio.sleep(10)
    
    async def _create_alert(self, alert_type: str, severity: str, 
                          regime: str, message: str, metadata: Dict[str, Any]):
        """Create a new alert."""
        
        alert = RegimeAlert(
            timestamp=datetime.now(),
            alert_type=alert_type,
            severity=severity,
            regime=regime,
            message=message,
            metadata=metadata
        )
        
        self.alerts.append(alert)
        self.logger.info(f"Alert created: {message}")
    
    async def _handle_console_alert(self, alert: RegimeAlert):
        """Handle alert by logging to console."""
        
        log_func = {
            'info': self.logger.info,
            'warning': self.logger.warning,
            'critical': self.logger.error
        }.get(alert.severity, self.logger.info)
        
        log_func(f"[{alert.alert_type.upper()}] {alert.message}")
    
    async def _handle_file_alert(self, alert: RegimeAlert):
        """Handle alert by writing to file."""
        
        alert_file = self.dashboard_dir / f"alerts_{datetime.now().strftime('%Y%m%d')}.jsonl"
        
        with open(alert_file, 'a') as f:
            f.write(json.dumps(alert.to_dict()) + '\n')
    
    async def _handle_webhook_alert(self, alert: RegimeAlert):
        """Handle alert by sending to webhook."""
        
        webhook_url = self.config.get('webhook_url')
        if not webhook_url:
            return
            
        # Send alert to webhook (implementation would go here)
        self.logger.info(f"Would send alert to webhook: {webhook_url}")
    
    async def _save_dashboard_state(self):
        """Periodically save dashboard state."""
        
        while self.is_running:
            try:
                state = {
                    'timestamp': datetime.now().isoformat(),
                    'current_regimes': self.current_regime,
                    'regime_confidence': self.regime_confidence,
                    'real_time_metrics': self._serialize_metrics(self.real_time_metrics),
                    'recent_alerts': [alert.to_dict() for alert in list(self.alerts)[-10:]]
                }
                
                state_file = self.dashboard_dir / 'dashboard_state.json'
                safe_json_dump(state, state_file)
                
                await asyncio.sleep(60)  # Save every minute
                
            except Exception as e:
                self.logger.error(f"Error saving dashboard state: {e}")
                await asyncio.sleep(60)
    
    def _serialize_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize metrics for JSON storage."""
        
        serialized = {}
        
        for symbol, data in metrics.items():
            if isinstance(data, dict) and 'timestamp' in data:
                serialized[symbol] = {
                    'timestamp': data['timestamp'].isoformat(),
                    'current_regime': data.get('current_regime'),
                    'regime_confidence': data.get('regime_confidence'),
                    'metrics': data.get('metrics', {})
                }
        
        return serialized
    
    async def get_dashboard_data(self) -> Dict[str, Any]:
        """Get current dashboard data for display."""
        
        return {
            'timestamp': datetime.now().isoformat(),
            'symbols': list(self.current_regime.keys()),
            'current_regimes': self.current_regime,
            'regime_confidence': self.regime_confidence,
            'real_time_metrics': self.real_time_metrics,
            'recent_alerts': [alert.to_dict() for alert in list(self.alerts)[-20:]],
            'regime_history': self._get_regime_history_summary()
        }
    
    def _get_regime_history_summary(self) -> Dict[str, Any]:
        """Get summary of regime history."""
        
        summary = {}
        
        for symbol, history in self.regime_history.items():
            if not history:
                continue
                
            # Get regime durations
            regime_durations = defaultdict(timedelta)
            last_regime = None
            last_timestamp = None
            
            for entry in history:
                if last_regime and last_regime != entry['regime']:
                    duration = entry['timestamp'] - last_timestamp
                    regime_durations[last_regime] += duration
                
                last_regime = entry['regime']
                last_timestamp = entry['timestamp']
            
            summary[symbol] = {
                'total_transitions': sum(
                    1 for i in range(1, len(history))
                    if history[i]['regime'] != history[i-1]['regime']
                ),
                'regime_durations': {
                    regime: str(duration)
                    for regime, duration in regime_durations.items()
                },
                'average_confidence': np.mean([h['confidence'] for h in history])
            }
        
        return summary


class RegimeMonitoringWebSocket:
    """WebSocket server for real-time regime updates."""
    
    def __init__(self, dashboard: RegimeMonitoringDashboard, port: int = 8765):
        self.dashboard = dashboard
        self.port = port
        self.clients = set()
        
    async def handler(self, websocket, path):
        """Handle WebSocket connections."""
        
        self.clients.add(websocket)
        try:
            # Send initial dashboard data
            data = await self.dashboard.get_dashboard_data()
            await websocket.send(json.dumps(data))
            
            # Keep connection alive and send updates
            while True:
                await asyncio.sleep(5)  # Send updates every 5 seconds
                
                if websocket.closed:
                    break
                    
                data = await self.dashboard.get_dashboard_data()
                await websocket.send(json.dumps(data))
                
        finally:
            self.clients.remove(websocket)
    
    async def start(self):
        """Start WebSocket server."""
        
        import websockets
        await websockets.serve(self.handler, 'localhost', self.port)
        logger.info(f"WebSocket server started on port {self.port}")


# Convenience function for starting monitoring
async def start_regime_monitoring(config: Dict[str, Any]):
    """Start the regime monitoring system."""
    
    dashboard = RegimeMonitoringDashboard(config)
    
    # Optionally start WebSocket server
    if config.get('enable_websocket', False):
        ws_server = RegimeMonitoringWebSocket(dashboard)
        await ws_server.start()
    
    await dashboard.start()


if __name__ == "__main__":
    # Example usage
    async def main():
        config = {
            'symbols': ['BTCUSDT', 'ETHUSDT'],
            'update_frequency': 60,
            'alert_handlers': ['console', 'file'],
            'enable_websocket': True,
            'dashboard_dir': 'dashboard',
            'data_dir': 'data'
        }
        
        await start_regime_monitoring(config)
    
    asyncio.run(main())