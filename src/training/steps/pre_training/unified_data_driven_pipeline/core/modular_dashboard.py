"""
ModularComponent Dashboard

This module provides a simple text-based dashboard for monitoring
ModularComponent instances and pipeline health.
"""

import logging
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from .modular_monitoring import ModularComponentMonitor, ComponentMetrics, PipelineMetrics


class ModularDashboard:
    """
    Simple text-based dashboard for ModularComponent monitoring.
    
    Features:
    - Real-time component status display
    - Performance metrics visualization
    - Health monitoring
    - Alert display
    - Performance recommendations
    """
    
    def __init__(self, monitor: ModularComponentMonitor, refresh_interval: int = 5):
        """Initialize the dashboard."""
        self.monitor = monitor
        self.refresh_interval = refresh_interval
        self.logger = logging.getLogger(__name__)
        self.running = False
        
    def start(self) -> None:
        """Start the dashboard display."""
        self.running = True
        self.logger.info("Starting ModularComponent dashboard")
        
        try:
            while self.running:
                self._display_dashboard()
                time.sleep(self.refresh_interval)
        except KeyboardInterrupt:
            self.logger.info("Dashboard stopped by user")
        except Exception as e:
            self.logger.error(f"Dashboard error: {e}")
        finally:
            self.running = False
    
    def stop(self) -> None:
        """Stop the dashboard."""
        self.running = False
    
    def _display_dashboard(self) -> None:
        """Display the current dashboard."""
        try:
            # Clear screen (works on most terminals)
            print("\033[2J\033[H", end="")
            
            # Get current metrics
            pipeline_metrics = self.monitor.get_pipeline_metrics()
            alerts = self.monitor.get_alerts(limit=10)
            recommendations = self.monitor.get_performance_recommendations()
            
            # Display header
            self._display_header()
            
            # Display pipeline overview
            self._display_pipeline_overview(pipeline_metrics)
            
            # Display component status
            self._display_component_status()
            
            # Display recent alerts
            if alerts:
                self._display_alerts(alerts)
            
            # Display recommendations
            if recommendations:
                self._display_recommendations(recommendations)
            
            # Display footer
            self._display_footer()
            
        except Exception as e:
            self.logger.error(f"Failed to display dashboard: {e}")
    
    def _display_header(self) -> None:
        """Display dashboard header."""
        print("=" * 80)
        print("🔧 ModularComponent Pipeline Dashboard")
        print("=" * 80)
        print(f"📅 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🔄 Refresh: Every {self.refresh_interval} seconds")
        print("=" * 80)
    
    def _display_pipeline_overview(self, metrics: PipelineMetrics) -> None:
        """Display pipeline overview metrics."""
        print("\n📊 PIPELINE OVERVIEW")
        print("-" * 40)
        
        # Health status with color coding
        health_score = metrics.overall_health_score
        if health_score >= 0.9:
            health_status = "🟢 EXCELLENT"
        elif health_score >= 0.7:
            health_status = "🟡 HEALTHY"
        elif health_score >= 0.5:
            health_status = "🟠 DEGRADED"
        else:
            health_status = "🔴 UNHEALTHY"
        
        print(f"Overall Health: {health_status} ({health_score:.1%})")
        print(f"Components: {metrics.healthy_components}/{metrics.total_components} healthy")
        print(f"Executions: {metrics.total_executions:,} total")
        print(f"Success Rate: {(metrics.total_successes/max(1, metrics.total_executions)):.1%}")
        print(f"Avg Execution Time: {metrics.avg_execution_time:.2f}s")
        print(f"Peak Memory: {metrics.peak_memory_usage_mb:.1f} MB")
    
    def _display_component_status(self) -> None:
        """Display individual component status."""
        print("\n🔧 COMPONENT STATUS")
        print("-" * 80)
        print(f"{'Component':<30} {'Status':<12} {'Health':<8} {'Executions':<12} {'Success Rate':<12} {'Avg Time':<10}")
        print("-" * 80)
        
        try:
            # Get component metrics
            component_names = list(self.monitor.component_metrics.keys())
            if not component_names:
                print("No components registered for monitoring")
                return
            
            for name in sorted(component_names):
                metrics = self.monitor.get_component_metrics(name)
                if not metrics:
                    continue
                
                # Status with emoji
                status_emoji = {
                    'excellent': '🟢',
                    'healthy': '🟡',
                    'degraded': '🟠',
                    'unhealthy': '🔴',
                    'unknown': '⚪'
                }.get(metrics.status, '⚪')
                
                status_display = f"{status_emoji} {metrics.status.upper()}"
                
                # Health score
                health_display = f"{metrics.health_score:.1%}"
                
                # Executions
                executions_display = f"{metrics.execution_count:,}"
                
                # Success rate
                success_rate = metrics.success_rate
                success_display = f"{success_rate:.1%}"
                
                # Average execution time
                time_display = f"{metrics.avg_execution_time:.2f}s"
                
                print(f"{name:<30} {status_display:<12} {health_display:<8} {executions_display:<12} {success_display:<12} {time_display:<10}")
        
        except Exception as e:
            print(f"Error displaying component status: {e}")
    
    def _display_alerts(self, alerts: List[Dict[str, Any]]) -> None:
        """Display recent alerts."""
        print("\n🚨 RECENT ALERTS")
        print("-" * 80)
        
        for alert in alerts[-5:]:  # Show last 5 alerts
            timestamp = alert.get('timestamp', 'Unknown')
            alert_type = alert.get('type', 'unknown')
            component = alert.get('component', 'Unknown')
            message = alert.get('message', 'No message')
            
            # Alert type emoji
            type_emoji = {
                'high_error_rate': '🔴',
                'slow_execution': '⏱️',
                'high_memory_usage': '💾',
                'low_health_score': '⚠️'
            }.get(alert_type, '❓')
            
            print(f"{type_emoji} [{timestamp}] {component}: {message}")
    
    def _display_recommendations(self, recommendations: List[str]) -> None:
        """Display performance recommendations."""
        print("\n💡 PERFORMANCE RECOMMENDATIONS")
        print("-" * 80)
        
        for i, recommendation in enumerate(recommendations[:5], 1):  # Show top 5
            print(f"{i}. {recommendation}")
    
    def _display_footer(self) -> None:
        """Display dashboard footer."""
        print("\n" + "=" * 80)
        print("Press Ctrl+C to stop the dashboard")
        print("=" * 80)
    
    def display_snapshot(self) -> None:
        """Display a single snapshot of the dashboard."""
        self._display_dashboard()
    
    def display_component_details(self, component_name: str) -> None:
        """Display detailed information for a specific component."""
        metrics = self.monitor.get_component_metrics(component_name)
        if not metrics:
            print(f"Component '{component_name}' not found")
            return
        
        print(f"\n🔧 COMPONENT DETAILS: {component_name}")
        print("=" * 60)
        print(f"Status: {metrics.status.upper()}")
        print(f"Health Score: {metrics.health_score:.1%}")
        print(f"Executions: {metrics.execution_count:,}")
        print(f"Successes: {metrics.success_count:,}")
        print(f"Errors: {metrics.error_count:,}")
        print(f"Success Rate: {metrics.success_rate:.1%}")
        print(f"Error Rate: {metrics.error_rate:.1%}")
        print(f"Total Execution Time: {metrics.total_execution_time:.2f}s")
        print(f"Average Execution Time: {metrics.avg_execution_time:.2f}s")
        print(f"Memory Usage: {metrics.memory_usage_mb:.1f} MB")
        print(f"Peak Memory Usage: {metrics.peak_memory_usage_mb:.1f} MB")
        print(f"Last Execution: {metrics.last_execution_time or 'Never'}")
        print(f"Last Error: {metrics.last_error_time or 'Never'}")
        if metrics.last_error_message:
            print(f"Last Error Message: {metrics.last_error_message}")
    
    def display_pipeline_report(self) -> None:
        """Display a comprehensive pipeline report."""
        report = self.monitor.generate_report()
        
        print("\n📊 COMPREHENSIVE PIPELINE REPORT")
        print("=" * 80)
        print(f"Generated: {report['timestamp']}")
        
        # Pipeline metrics
        pipeline_metrics = report['pipeline_metrics']
        print(f"\nPipeline Health: {pipeline_metrics['overall_health_score']:.1%}")
        print(f"Total Components: {pipeline_metrics['total_components']}")
        print(f"Healthy Components: {pipeline_metrics['healthy_components']}")
        print(f"Degraded Components: {pipeline_metrics['degraded_components']}")
        print(f"Error Components: {pipeline_metrics['error_components']}")
        print(f"Total Executions: {pipeline_metrics['total_executions']:,}")
        print(f"Total Successes: {pipeline_metrics['total_successes']:,}")
        print(f"Total Errors: {pipeline_metrics['total_errors']:,}")
        
        # Component details
        print(f"\nComponent Details:")
        for name, metrics in report['component_metrics'].items():
            print(f"  {name}: {metrics['status']} ({metrics['health_score']:.1%}) - {metrics['execution_count']} executions")
        
        # Recent alerts
        if report['recent_alerts']:
            print(f"\nRecent Alerts ({len(report['recent_alerts'])}):")
            for alert in report['recent_alerts'][-5:]:
                print(f"  {alert['timestamp']}: {alert['message']}")
        
        # Recommendations
        if report['recommendations']:
            print(f"\nRecommendations ({len(report['recommendations'])}):")
            for i, rec in enumerate(report['recommendations'][:5], 1):
                print(f"  {i}. {rec}")


def create_dashboard(monitor: ModularComponentMonitor, refresh_interval: int = 5) -> ModularDashboard:
    """Create a new ModularDashboard instance."""
    return ModularDashboard(monitor, refresh_interval)