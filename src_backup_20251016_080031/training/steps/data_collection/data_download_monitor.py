#!/usr/bin/env python3
"""
Data Download Monitor

This module provides comprehensive monitoring and logging for data download operations.
It tracks download progress, performance metrics, data quality, and provides detailed
reporting capabilities.

Key Features:
- Real-time download progress tracking
- Performance metrics collection
- Data quality monitoring
- Comprehensive logging and reporting
- Alert system for failures and issues
- Historical statistics and trends
"""

import asyncio
import sys
import time
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import system_logger
from src.utils.error_handler import handles_errors
from src.utils.common_operations import safe_json_dump, safe_json_load
from src.utils.comprehensive_function_logger import log_step_functions, log_important_calls, log_all_calls, log_internal_call, log_step_progress, log_data_operation

logger = system_logger.getChild("DataDownloadMonitor")

class DataDownloadMonitor:
    """Comprehensive monitor for data download operations."""
    
    @log_important_calls
    def __init__(self, data_cache_path: str = "data_cache", monitor_file: str = "download_monitor.json"):
        self.data_cache_path = Path(data_cache_path)
        self.monitor_file = Path(monitor_file)
        self.logger = logger.getChild('DataDownloadMonitor')
        
        # Initialize monitoring data
        self.monitoring_data = {
            'sessions': {},
            'statistics': {
                'total_sessions': 0,
                'successful_sessions': 0,
                'failed_sessions': 0,
                'total_downloads': 0,
                'total_rows_downloaded': 0,
                'total_files_created': 0,
                'average_download_time': 0.0,
                'average_rows_per_session': 0.0,
                'last_updated': None
            },
            'performance_metrics': {
                'download_speeds': [],
                'memory_usage': [],
                'error_rates': [],
                'quality_scores': []
            },
            'alerts': [],
            'created_at': datetime.now().isoformat()
        }
        
        # Load existing monitoring data
        self._load_monitoring_data()
        
        # Current session tracking
        self.current_session = None
        self.session_start_time = None
        
        self.logger.info("✅ Data Download Monitor initialized")
    
    @handles_errors(context="start_session")
    @log_all_calls
    def start_session(
        self, 
        session_id: str, 
        symbol: str, 
        exchange: str, 
        data_type: str,
        timeframe: str = "1m",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Start monitoring a new download session.
        
        Args:
            session_id: Unique session identifier
            symbol: Trading symbol
            exchange: Exchange name
            data_type: Type of data being downloaded
            timeframe: Timeframe for the data
            start_date: Start date for download
            end_date: End date for download
            
        Returns:
            Session information dictionary
        """
        try:
            self.session_start_time = time.time()
            self.current_session = session_id
            
            session_info = {
                'session_id': session_id,
                'symbol': symbol,
                'exchange': exchange,
                'data_type': data_type,
                'timeframe': timeframe,
                'start_date': start_date.isoformat() if start_date else None,
                'end_date': end_date.isoformat() if end_date else None,
                'start_time': datetime.now().isoformat(),
                'status': 'running',
                'batches': [],
                'total_batches': 0,
                'successful_batches': 0,
                'failed_batches': 0,
                'total_rows': 0,
                'total_files_created': 0,
                'errors': [],
                'warnings': [],
                'performance_metrics': {
                    'download_speed_rows_per_second': 0.0,
                    'average_batch_time': 0.0,
                    'memory_peak_mb': 0.0,
                    'quality_score': 0.0
                }
            }
            
            self.monitoring_data['sessions'][session_id] = session_info
            self.monitoring_data['statistics']['total_sessions'] += 1
            
            self.logger.info(f"🚀 Started monitoring session {session_id}: {exchange}_{symbol}_{data_type}_{timeframe}")
            return session_info
            
        except Exception as e:
            self.logger.error(f"❌ Error starting session {session_id}: {e}")
            return {
                'session_id': session_id,
                'error': str(e),
                'status': 'failed'
            }
    
    @handles_errors(context="update_batch_progress")
    @log_all_calls
    def update_batch_progress(
        self, 
        session_id: str, 
        batch_number: int, 
        batch_success: bool, 
        rows_downloaded: int = 0,
        batch_duration: float = 0.0,
        file_path: Optional[str] = None,
        error_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Update progress for a batch within a session.
        
        Args:
            session_id: Session identifier
            batch_number: Batch number
            batch_success: Whether the batch was successful
            rows_downloaded: Number of rows downloaded
            batch_duration: Duration of the batch in seconds
            file_path: Path to the created file
            error_message: Error message if batch failed
            
        Returns:
            Updated batch information
        """
        try:
            if session_id not in self.monitoring_data['sessions']:
                self.logger.warning(f"⚠️ Session {session_id} not found in monitoring data")
                return {'error': 'Session not found'}
            
            session = self.monitoring_data['sessions'][session_id]
            
            batch_info = {
                'batch_number': batch_number,
                'success': batch_success,
                'rows_downloaded': rows_downloaded,
                'duration_seconds': batch_duration,
                'file_path': file_path,
                'timestamp': datetime.now().isoformat(),
                'error_message': error_message
            }
            
            session['batches'].append(batch_info)
            session['total_batches'] += 1
            
            if batch_success:
                session['successful_batches'] += 1
                session['total_rows'] += rows_downloaded
                if file_path:
                    session['total_files_created'] += 1
            else:
                session['failed_batches'] += 1
                if error_message:
                    session['errors'].append({
                        'batch_number': batch_number,
                        'error': error_message,
                        'timestamp': datetime.now().isoformat()
                    })
            
            # Update performance metrics
            self._update_session_metrics(session)
            
            self.logger.info(f"📊 Updated batch {batch_number} for session {session_id}: {rows_downloaded} rows, {batch_duration:.2f}s")
            return batch_info
            
        except Exception as e:
            self.logger.error(f"❌ Error updating batch progress: {e}")
            return {'error': str(e)}
    
    @handles_errors(context="end_session")
    @log_all_calls
    def end_session(
        self, 
        session_id: str, 
        final_status: str = 'completed',
        final_error: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        End a monitoring session.
        
        Args:
            session_id: Session identifier
            final_status: Final status ('completed', 'failed', 'cancelled')
            final_error: Final error message if session failed
            
        Returns:
            Final session summary
        """
        try:
            if session_id not in self.monitoring_data['sessions']:
                self.logger.warning(f"⚠️ Session {session_id} not found in monitoring data")
                return {'error': 'Session not found'}
            
            session = self.monitoring_data['sessions'][session_id]
            session['status'] = final_status
            session['end_time'] = datetime.now().isoformat()
            
            if self.session_start_time:
                session['total_duration_seconds'] = time.time() - self.session_start_time
            
            # Update global statistics
            self._update_global_statistics(session, final_status)
            
            # Generate session summary
            summary = self._generate_session_summary(session)
            
            # Save monitoring data
            self._save_monitoring_data()
            
            self.logger.info(f"🏁 Ended session {session_id}: {final_status}, {session['total_rows']} rows, {session['total_files_created']} files")
            
            # Clear current session
            if self.current_session == session_id:
                self.current_session = None
                self.session_start_time = None
            
            return summary
            
        except Exception as e:
            self.logger.error(f"❌ Error ending session {session_id}: {e}")
            return {'error': str(e)}
    
    @handles_errors(context="add_alert")
    @log_all_calls
    def add_alert(
        self, 
        alert_type: str, 
        message: str, 
        severity: str = 'warning',
        session_id: Optional[str] = None,
        batch_number: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Add an alert to the monitoring system.
        
        Args:
            alert_type: Type of alert ('error', 'warning', 'info', 'success')
            message: Alert message
            severity: Alert severity ('low', 'medium', 'high', 'critical')
            session_id: Associated session ID
            batch_number: Associated batch number
            
        Returns:
            Alert information
        """
        try:
            alert = {
                'alert_id': f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.monitoring_data['alerts'])}",
                'type': alert_type,
                'message': message,
                'severity': severity,
                'session_id': session_id,
                'batch_number': batch_number,
                'timestamp': datetime.now().isoformat()
            }
            
            self.monitoring_data['alerts'].append(alert)
            
            # Keep only last 1000 alerts
            if len(self.monitoring_data['alerts']) > 1000:
                self.monitoring_data['alerts'] = self.monitoring_data['alerts'][-1000:]
            
            self.logger.info(f"🚨 Alert added: {alert_type} - {message}")
            return alert
            
        except Exception as e:
            self.logger.error(f"❌ Error adding alert: {e}")
            return {'error': str(e)}
    
    @handles_errors(context="get_session_status")
    @log_all_calls
    def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """Get current status of a session."""
        try:
            if session_id not in self.monitoring_data['sessions']:
                return {'error': 'Session not found'}
            
            session = self.monitoring_data['sessions'][session_id]
            
            # Calculate current progress
            progress = {
                'session_id': session_id,
                'status': session['status'],
                'progress_percentage': 0.0,
                'total_batches': session['total_batches'],
                'successful_batches': session['successful_batches'],
                'failed_batches': session['failed_batches'],
                'total_rows': session['total_rows'],
                'total_files_created': session['total_files_created'],
                'current_duration': 0.0,
                'estimated_completion': None
            }
            
            if session['status'] == 'running' and self.session_start_time:
                progress['current_duration'] = time.time() - self.session_start_time
                
                # Estimate completion if we have some data
                if session['total_batches'] > 0 and session['successful_batches'] > 0:
                    avg_batch_time = progress['current_duration'] / session['total_batches']
                    remaining_batches = max(0, 10 - session['total_batches'])  # Assume max 10 batches
                    estimated_remaining = remaining_batches * avg_batch_time
                    progress['estimated_completion'] = datetime.fromtimestamp(
                        time.time() + estimated_remaining
                    ).isoformat()
            
            return progress
            
        except Exception as e:
            self.logger.error(f"❌ Error getting session status: {e}")
            return {'error': str(e)}
    
    @handles_errors(context="get_monitoring_summary")
    @log_all_calls
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get comprehensive monitoring summary."""
        try:
            stats = self.monitoring_data['statistics']
            
            # Calculate additional metrics
            success_rate = (
                stats['successful_sessions'] / max(stats['total_sessions'], 1) * 100
            )
            
            avg_rows_per_session = (
                stats['total_rows_downloaded'] / max(stats['successful_sessions'], 1)
            )
            
            # Recent activity (last 24 hours)
            recent_sessions = []
            cutoff_time = datetime.now() - timedelta(hours=24)
            
            for session_id, session in self.monitoring_data['sessions'].items():
                session_start = datetime.fromisoformat(session['start_time'])
                if session_start > cutoff_time:
                    recent_sessions.append({
                        'session_id': session_id,
                        'symbol': session['symbol'],
                        'exchange': session['exchange'],
                        'data_type': session['data_type'],
                        'status': session['status'],
                        'total_rows': session['total_rows'],
                        'start_time': session['start_time']
                    })
            
            # Recent alerts
            recent_alerts = []
            for alert in self.monitoring_data['alerts'][-10:]:  # Last 10 alerts
                recent_alerts.append({
                    'type': alert['type'],
                    'message': alert['message'],
                    'severity': alert['severity'],
                    'timestamp': alert['timestamp']
                })
            
            summary = {
                'overview': {
                    'total_sessions': stats['total_sessions'],
                    'successful_sessions': stats['successful_sessions'],
                    'failed_sessions': stats['failed_sessions'],
                    'success_rate': success_rate,
                    'total_downloads': stats['total_downloads'],
                    'total_rows_downloaded': stats['total_rows_downloaded'],
                    'total_files_created': stats['total_files_created'],
                    'average_rows_per_session': avg_rows_per_session
                },
                'performance': {
                    'average_download_time': stats['average_download_time'],
                    'download_speeds': self.monitoring_data['performance_metrics']['download_speeds'][-10:],
                    'quality_scores': self.monitoring_data['performance_metrics']['quality_scores'][-10:]
                },
                'recent_activity': {
                    'sessions': recent_sessions,
                    'alerts': recent_alerts
                },
                'current_session': self.current_session,
                'last_updated': datetime.now().isoformat()
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"❌ Error getting monitoring summary: {e}")
            return {'error': str(e)}
    
    @handles_errors(context="export_monitoring_data")
    @log_all_calls
    def export_monitoring_data(
        self, 
        export_path: Optional[str] = None,
        include_sessions: bool = True,
        include_alerts: bool = True
    ) -> Dict[str, Any]:
        """
        Export monitoring data to a file.
        
        Args:
            export_path: Path to export file (default: auto-generated)
            include_sessions: Whether to include session data
            include_alerts: Whether to include alert data
            
        Returns:
            Export information
        """
        try:
            if export_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                export_path = f"download_monitor_export_{timestamp}.json"
            
            export_data = {
                'export_info': {
                    'exported_at': datetime.now().isoformat(),
                    'monitor_version': '1.0.0',
                    'total_sessions': len(self.monitoring_data['sessions']),
                    'total_alerts': len(self.monitoring_data['alerts'])
                },
                'statistics': self.monitoring_data['statistics'],
                'performance_metrics': self.monitoring_data['performance_metrics']
            }
            
            if include_sessions:
                export_data['sessions'] = self.monitoring_data['sessions']
            
            if include_alerts:
                export_data['alerts'] = self.monitoring_data['alerts']
            
            # Save export file
            success = safe_json_dump(export_data, export_path, indent=2)
            
            if success:
                self.logger.info(f"📁 Exported monitoring data to {export_path}")
                return {
                    'success': True,
                    'export_path': export_path,
                    'file_size_mb': Path(export_path).stat().st_size / (1024 * 1024),
                    'sessions_included': include_sessions,
                    'alerts_included': include_alerts
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to save export file'
                }
                
        except Exception as e:
            self.logger.error(f"❌ Error exporting monitoring data: {e}")
            return {'error': str(e)}
    
    @handles_errors(context="update_session_metrics")
    def _update_session_metrics(self, session: Dict[str, Any]) -> None:
        """Update performance metrics for a session."""
        try:
            if not session['batches']:
                return
            
            # Calculate download speed
            total_rows = session['total_rows']
            if self.session_start_time:
                total_time = time.time() - self.session_start_time
                if total_time > 0:
                    session['performance_metrics']['download_speed_rows_per_second'] = total_rows / total_time
            
            # Calculate average batch time
            successful_batches = [b for b in session['batches'] if b['success']]
            if successful_batches:
                avg_batch_time = sum(b['duration_seconds'] for b in successful_batches) / len(successful_batches)
                session['performance_metrics']['average_batch_time'] = avg_batch_time
            
            # Update quality score (simplified calculation)
            success_rate = session['successful_batches'] / max(session['total_batches'], 1)
            session['performance_metrics']['quality_score'] = success_rate * 100
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error updating session metrics: {e}")
    
    @handles_errors(context="update_global_statistics")
    def _update_global_statistics(self, session: Dict[str, Any], final_status: str) -> None:
        """Update global statistics based on session results."""
        try:
            stats = self.monitoring_data['statistics']
            
            if final_status == 'completed':
                stats['successful_sessions'] += 1
            else:
                stats['failed_sessions'] += 1
            
            stats['total_downloads'] += 1
            stats['total_rows_downloaded'] += session['total_rows']
            stats['total_files_created'] += session['total_files_created']
            
            # Update averages
            if stats['successful_sessions'] > 0:
                stats['average_rows_per_session'] = stats['total_rows_downloaded'] / stats['successful_sessions']
            
            if session.get('total_duration_seconds'):
                # Update average download time
                total_time = stats['average_download_time'] * (stats['total_downloads'] - 1)
                total_time += session['total_duration_seconds']
                stats['average_download_time'] = total_time / stats['total_downloads']
            
            stats['last_updated'] = datetime.now().isoformat()
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error updating global statistics: {e}")
    
    @handles_errors(context="generate_session_summary")
    def _generate_session_summary(self, session: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a comprehensive summary for a session."""
        try:
            summary = {
                'session_id': session['session_id'],
                'symbol': session['symbol'],
                'exchange': session['exchange'],
                'data_type': session['data_type'],
                'timeframe': session['timeframe'],
                'status': session['status'],
                'duration_seconds': session.get('total_duration_seconds', 0),
                'total_batches': session['total_batches'],
                'successful_batches': session['successful_batches'],
                'failed_batches': session['failed_batches'],
                'success_rate': session['successful_batches'] / max(session['total_batches'], 1) * 100,
                'total_rows': session['total_rows'],
                'total_files_created': session['total_files_created'],
                'performance_metrics': session['performance_metrics'],
                'errors': session['errors'],
                'warnings': session['warnings'],
                'start_time': session['start_time'],
                'end_time': session.get('end_time'),
                'summary_generated_at': datetime.now().isoformat()
            }
            
            return summary
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error generating session summary: {e}")
            return {'error': str(e)}
    
    @handles_errors(context="load_monitoring_data")
    def _load_monitoring_data(self) -> None:
        """Load existing monitoring data from file."""
        try:
            if self.monitor_file.exists():
                data = safe_json_load(self.monitor_file, {})
                if data:
                    self.monitoring_data.update(data)
                    self.logger.info(f"📖 Loaded monitoring data from {self.monitor_file}")
            else:
                self.logger.info("📝 No existing monitoring data found, starting fresh")
                
        except Exception as e:
            self.logger.warning(f"⚠️ Error loading monitoring data: {e}")
    
    @handles_errors(context="save_monitoring_data")
    def _save_monitoring_data(self) -> None:
        """Save monitoring data to file."""
        try:
            success = safe_json_dump(self.monitoring_data, self.monitor_file, indent=2)
            if success:
                self.logger.debug(f"💾 Saved monitoring data to {self.monitor_file}")
            else:
                self.logger.warning("⚠️ Failed to save monitoring data")
                
        except Exception as e:
            self.logger.warning(f"⚠️ Error saving monitoring data: {e}")

# Global monitor instance
download_monitor = DataDownloadMonitor()

# Convenience functions
@handles_errors()
def start_download_session(
    session_id: str, 
    symbol: str, 
    exchange: str, 
    data_type: str,
    **kwargs
) -> Dict[str, Any]:
    """Convenience function to start a download session."""
    return download_monitor.start_session(session_id, symbol, exchange, data_type, **kwargs)

@handles_errors()
def update_download_progress(
    session_id: str, 
    batch_number: int, 
    batch_success: bool, 
    **kwargs
) -> Dict[str, Any]:
    """Convenience function to update download progress."""
    return download_monitor.update_batch_progress(session_id, batch_number, batch_success, **kwargs)

@handles_errors()
def end_download_session(
    session_id: str, 
    final_status: str = 'completed',
    **kwargs
) -> Dict[str, Any]:
    """Convenience function to end a download session."""
    return download_monitor.end_session(session_id, final_status, **kwargs)

@handles_errors()
def get_download_status(session_id: str) -> Dict[str, Any]:
    """Convenience function to get download status."""
    return download_monitor.get_session_status(session_id)

@handles_errors()
def get_monitoring_dashboard() -> Dict[str, Any]:
    """Convenience function to get monitoring dashboard data."""
    return download_monitor.get_monitoring_summary()

if __name__ == "__main__":
    # Example usage
    async def test_monitor():
        logger.info("🎯 Testing Data Download Monitor")
        logger.info("=" * 80)
        
        # Start a test session
        session = start_download_session(
            session_id="test_session_001",
            symbol="ETHUSDT",
            exchange="BINANCE",
            data_type="klines",
            timeframe="1m"
        )
        
        logger.info(f"✅ Started session: {session['session_id']}")
        
        # Simulate some batch updates
        for i in range(3):
            update_download_progress(
                session_id=session['session_id'],
                batch_number=i + 1,
                batch_success=True,
                rows_downloaded=1000,
                batch_duration=2.5,
                file_path=f"batch_{i+1}.parquet"
            )
            await asyncio.sleep(0.1)
        
        # End the session
        summary = end_download_session(session['session_id'], 'completed')
        
        logger.info(f"✅ Session completed: {summary['total_rows']} rows in {summary['duration_seconds']:.2f}s")
        
        # Get monitoring summary
        dashboard = get_monitoring_dashboard()
        logger.info(f"📊 Dashboard: {dashboard['overview']['total_sessions']} sessions, {dashboard['overview']['success_rate']:.1f}% success rate")
        
        logger.info("=" * 80)
        logger.info("🎉 Data Download Monitor tests completed!")
        logger.info("=" * 80)
    
    asyncio.run(test_monitor())