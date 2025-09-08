#!/usr/bin/env python3
"""
Dependency Injection Health Dashboard

A comprehensive monitoring dashboard for the Dependency Injection system
with real-time health monitoring and alerting capabilities.
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import threading
import logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.logger import system_logger

class DIHealthDashboard:
    """Comprehensive dashboard for monitoring DI system health."""

    def __init__(self):
        self.logger = system_logger.getChild('DIHealthDashboard')
        self.health_history = []
        self.alerts = []
        self.monitoring_active = False
        self.monitor_thread = None

        # Health thresholds
        self.thresholds = {
            'max_response_time': 5.0,  # seconds
            'min_utility_availability': 0.95,  # 95%
            'max_error_rate': 0.05  # 5%
        }

    def start_monitoring(self, interval_seconds: int = 300) -> None:
        """Start continuous monitoring of the DI system."""
        if self.monitoring_active:
            self.logger.warning("Monitoring already active")
            return

        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self.monitor_thread.start()
        self.logger.info(f"🔍 Started DI health monitoring (interval: {interval_seconds}s)")

    def stop_monitoring(self) -> None:
        """Stop continuous monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        self.logger.info("🛑 Stopped DI health monitoring")

    def get_current_health(self) -> Dict[str, Any]:
        """Get current DI system health status."""
        try:
            from src.training.steps.model_training.step04_dependency_injection import get_step04_utilities

            start_time = time.time()
            utils = get_step04_utilities()
            response_time = time.time() - start_time

            # Test key utility functions
            test_results = self._test_critical_functions(utils)

            health_score = self._calculate_health_score(test_results, response_time)

            health_data = {
                'timestamp': datetime.now().isoformat(),
                'response_time': response_time,
                'health_score': health_score,
                'status': 'healthy' if health_score >= 0.95 else 'warning' if health_score >= 0.8 else 'critical',
                'test_results': test_results,
                'utility_summary': self._get_utility_summary(utils)
            }

            self.health_history.append(health_data)

            # Keep only last 100 entries
            if len(self.health_history) > 100:
                self.health_history.pop(0)

            return health_data

        except Exception as e:
            self.logger.error(f"❌ Health check failed: {e}")
            error_data = {
                'timestamp': datetime.now().isoformat(),
                'status': 'error',
                'error': str(e),
                'health_score': 0.0
            }
            self.health_history.append(error_data)
            return error_data

    def _test_critical_functions(self, utils) -> Dict[str, Any]:
        """Test critical utility functions."""
        critical_tests = [
            ('common_operations', 'safe_float', lambda f: f(123.45) == 123.45),
            ('common_operations', 'safe_int', lambda f: f(42) == 42),
            ('math_validation', 'validate_positive', lambda f: f(5.0) == 5.0),
            ('math_validation', 'validate_range', lambda f: f(0.5, 0.0, 1.0) == 0.5),
            ('common_utilities', 'create_data_quality_report', lambda f: isinstance(f({}), dict))
        ]

        results = {}
        for utility_type, function_name, test_func in critical_tests:
            try:
                func = utils.get_function(utility_type, function_name)
                if func is None:
                    results[f"{utility_type}.{function_name}"] = {
                        'status': 'failed',
                        'error': 'Function not found'
                    }
                else:
                    # Test the function
                    test_result = test_func(func)
                    results[f"{utility_type}.{function_name}"] = {
                        'status': 'passed' if test_result else 'failed',
                        'function_available': True
                    }
            except Exception as e:
                results[f"{utility_type}.{function_name}"] = {
                    'status': 'error',
                    'error': str(e)
                }

        return results

    def _calculate_health_score(self, test_results: Dict[str, Any], response_time: float) -> float:
        """Calculate overall health score."""
        if not test_results:
            return 0.0

        # Function availability score
        passed_tests = sum(1 for result in test_results.values()
                          if result.get('status') == 'passed')
        total_tests = len(test_results)
        availability_score = passed_tests / total_tests if total_tests > 0 else 0

        # Response time score (inverse - faster is better)
        response_score = max(0, 1 - (response_time / self.thresholds['max_response_time']))

        # Weighted average
        health_score = (availability_score * 0.7) + (response_score * 0.3)

        return round(health_score, 3)

    def _get_utility_summary(self, utils) -> Dict[str, Any]:
        """Get summary of available utilities."""
        try:
            container = utils.container
            summary = container.get_utility_summary()
            return summary
        except Exception as e:
            return {'error': str(e)}

    def get_health_trends(self, hours: int = 24) -> Dict[str, Any]:
        """Get health trends over the specified time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        recent_health = [
            h for h in self.health_history
            if datetime.fromisoformat(h['timestamp']) > cutoff_time
        ]

        if not recent_health:
            return {'error': 'No recent health data available'}

        # Calculate trends
        health_scores = [h.get('health_score', 0) for h in recent_health]
        response_times = [h.get('response_time', 0) for h in recent_health if 'response_time' in h]

        trends = {
            'period_hours': hours,
            'total_checks': len(recent_health),
            'avg_health_score': sum(health_scores) / len(health_scores) if health_scores else 0,
            'min_health_score': min(health_scores) if health_scores else 0,
            'max_health_score': max(health_scores) if health_scores else 0,
            'avg_response_time': sum(response_times) / len(response_times) if response_times else 0,
            'health_trend': self._calculate_trend(health_scores),
            'recent_status': recent_health[-1].get('status', 'unknown')
        }

        return trends

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction."""
        if len(values) < 2:
            return 'insufficient_data'

        recent_avg = sum(values[-5:]) / len(values[-5:]) if len(values) >= 5 else sum(values) / len(values)
        older_avg = sum(values[:-5]) / len(values[:-5]) if len(values) > 5 else recent_avg

        if recent_avg > older_avg + 0.05:
            return 'improving'
        elif recent_avg < older_avg - 0.05:
            return 'degrading'
        else:
            return 'stable'

    def generate_report(self) -> str:
        """Generate a comprehensive health report."""
        current_health = self.get_current_health()
        trends = self.get_health_trends()

        report = f"""
🔍 DEPENDENCY INJECTION SYSTEM HEALTH REPORT
{'='*60}

📊 CURRENT STATUS
   Status: {current_health.get('status', 'unknown').upper()}
   Health Score: {current_health.get('health_score', 0):.1%}
   Response Time: {current_health.get('response_time', 0):.3f}s
   Timestamp: {current_health.get('timestamp', 'unknown')}

📈 HEALTH TRENDS (Last 24h)
   Total Checks: {trends.get('total_checks', 0)}
   Average Health: {trends.get('avg_health_score', 0):.1%}
   Health Trend: {trends.get('health_trend', 'unknown')}
   Avg Response Time: {trends.get('avg_response_time', 0):.3f}s

🧪 FUNCTION TESTS
"""

        test_results = current_health.get('test_results', {})
        for test_name, result in test_results.items():
            status_icon = "✅" if result.get('status') == 'passed' else "❌"
            report += f"   {status_icon} {test_name}: {result.get('status', 'unknown')}\n"

        report += "\n📦 AVAILABLE UTILITIES\n"
        utility_summary = current_health.get('utility_summary', {})
        for utility_type, info in utility_summary.items():
            if isinstance(info, dict):
                function_count = info.get('function_count', 0)
                report += f"   • {utility_type}: {function_count} functions\n"

        if self.alerts:
            report += "\n🚨 ACTIVE ALERTS\n"
            for alert in self.alerts[-5:]:  # Show last 5 alerts
                report += f"   ⚠️ {alert.get('message', 'Unknown alert')}\n"

        return report

    def _monitoring_loop(self, interval_seconds: int) -> None:
        """Continuous monitoring loop."""
        while self.monitoring_active:
            try:
                health = self.get_current_health()

                # Check for alerts
                if health.get('health_score', 1.0) < 0.8:
                    self._create_alert(f"Health score dropped to {health.get('health_score', 0):.1%}")

                if health.get('response_time', 0) > self.thresholds['max_response_time']:
                    self._create_alert(f"Response time exceeded threshold: {health.get('response_time', 0):.3f}s")

                time.sleep(interval_seconds)

            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(interval_seconds)

    def _create_alert(self, message: str) -> None:
        """Create an alert."""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'message': message,
            'severity': 'warning'
        }
        self.alerts.append(alert)

        # Keep only last 50 alerts
        if len(self.alerts) > 50:
            self.alerts.pop(0)

        self.logger.warning(f"🚨 DI Alert: {message}")

def main():
    """Main entry point for the DI health dashboard."""
    print("🏥 Starting DI Health Dashboard...")

    dashboard = DIHealthDashboard()

    # Run initial health check
    print("🔍 Running initial health check...")
    health = dashboard.get_current_health()
    print(f"Current Health: {health.get('status', 'unknown').upper()} ({health.get('health_score', 0):.1%})")

    # Generate and display report
    report = dashboard.generate_report()
    print(report)

    # Start monitoring
    print("🔄 Starting continuous monitoring...")
    dashboard.start_monitoring(interval_seconds=60)  # Check every minute

    try:
        # Keep running for demonstration
        input("Press Enter to stop monitoring...")
    except KeyboardInterrupt:
        pass
    finally:
        dashboard.stop_monitoring()
        print("✅ Monitoring stopped")

if __name__ == '__main__':
    main()
