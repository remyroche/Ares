#!/usr/bin/env python3
"""
Advanced Monitoring System Launcher

This script demonstrates how to set up and use the advanced monitoring and tracking system
for the Ares trading bot.
"""

from src.utils.logger import system_logger
from typing import Any
import asyncio
import signal
import sys

from src.monitoring import MonitoringIntegrationManager
from src.utils.warning_symbols import error, failed, initialization_error, warning

class AdvancedMonitoringLauncher:
    """
    Launcher for the advanced monitoring system.
    """

    def __init__(self):
        """Initialize the launcher."""
        self.logger = system_logger.getChild("AdvancedMonitoringLauncher")
        self.integration_manager: MonitoringIntegrationManager | None = None
        self.is_running = False

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self.logger.info("🚀 Advanced Monitoring Launcher initialized")

    async def start_monitoring(self) -> bool:
        """Start the monitoring system."""
        if not self.integration_manager:
            self.logger.error("Integration manager not initialized")
            return False

        self.logger.info("Starting advanced monitoring system...")

        # Start integration
        success = await self.integration_manager.start_integration()

        if success:
            self.is_running = True
            self.logger.info("✅ Advanced monitoring system started")
            return True
        self.logger.error("Failed to start monitoring integration")
        return False

    async def run_demo(self) -> None:
        """Run a demonstration of the monitoring system."""
        self.logger.info("Running monitoring system demonstration...")

        # Demo loop
        demo_counter = 0
        while self.is_running and demo_counter < 60:  # Run for 5 minutes
            # Get unified dashboard data
            dashboard_data = (
                self.integration_manager.get_unified_dashboard_data()
                if self.integration_manager
                else {}
            )

            # Print status every 10 seconds
            if demo_counter % 10 == 0:
                self._print_status(dashboard_data, demo_counter)

            # Simulate some activity
            await self._simulate_activity()

            demo_counter += 1
            await asyncio.sleep(5)  # Update every 5 seconds

        self.logger.info("✅ Monitoring demonstration completed")

    def _print_status(self, dashboard_data: dict[str, Any], counter: int) -> None:
        """Print current monitoring status."""
        print(f"\n📊 Monitoring Status (Update {counter//10 + 1}/6):")
        print("=" * 50)

        # System metrics
        system_metrics = dashboard_data.get("system_metrics", {})
        print(f"🖥️  System Metrics:")
        print(f"   CPU Usage: {system_metrics.get('cpu_usage', 'N/A')}%")
        print(f"   Memory Usage: {system_metrics.get('memory_usage', 'N/A')}%")
        print(f"   Active Processes: {system_metrics.get('active_processes', 'N/A')}")

        # Performance metrics
        performance_metrics = dashboard_data.get("performance_metrics", {})
        print(f"⚡ Performance Metrics:")
        print(f"   Response Time: {performance_metrics.get('avg_response_time', 'N/A')}ms")
        print(f"   Throughput: {performance_metrics.get('requests_per_second', 'N/A')} req/s")
        print(f"   Error Rate: {performance_metrics.get('error_rate', 'N/A')}%")

        # ML metrics
        ml_metrics = dashboard_data.get("ml_metrics", {})
        print(f"🤖 ML Metrics:")
        print(f"   Model Accuracy: {ml_metrics.get('model_accuracy', 'N/A')}%")
        print(f"   Prediction Latency: {ml_metrics.get('prediction_latency', 'N/A')}ms")
        print(f"   Drift Score: {ml_metrics.get('drift_score', 'N/A')}")

        # Trading metrics
        trading_metrics = dashboard_data.get("trading_metrics", {})
        print(f"💰 Trading Metrics:")
        print(f"   Portfolio Value: ${trading_metrics.get('portfolio_value', 'N/A'):,.2f}")
        print(f"   Daily P&L: ${trading_metrics.get('daily_pnl', 'N/A'):,.2f}")
        print(f"   Win Rate: {trading_metrics.get('win_rate', 'N/A')}%")

        # Alerts
        alerts = dashboard_data.get("alerts", [])
        if alerts:
            print(f"🚨 Active Alerts ({len(alerts)}):")
            for alert in alerts[:3]:  # Show top 3 alerts
                print(f"   - {alert.get('message', 'N/A')}")
        else:
            print("✅ No active alerts")

        print("=" * 50)

    async def _simulate_activity(self) -> None:
        """Simulate some activity for the demo."""
        if not self.integration_manager:
            return

        # Simulate system activity
        await self.integration_manager.record_system_metric("cpu_usage", 45.2)
        await self.integration_manager.record_system_metric("memory_usage", 67.8)

        # Simulate performance activity
        await self.integration_manager.record_performance_metric("response_time", 125.5)
        await self.integration_manager.record_performance_metric("throughput", 150.2)

        # Simulate ML activity
        await self.integration_manager.record_ml_metric("model_accuracy", 87.3)
        await self.integration_manager.record_ml_metric("drift_score", 0.15)

        # Simulate trading activity
        await self.integration_manager.record_trading_metric("portfolio_value", 12500.75)
        await self.integration_manager.record_trading_metric("daily_pnl", 125.50)

    async def generate_report(self, report_type: str = "comprehensive") -> dict[str, Any]:
        """Generate a monitoring report."""
        if not self.integration_manager:
            return {"error": "Integration manager not initialized"}

        return await self.integration_manager.generate_report(report_type)


async def main():
    """Main function for the advanced monitoring launcher."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Advanced Monitoring System Launcher"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run monitoring system demonstration",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to custom configuration file",
    )
    parser.add_argument(
        "--report",
        type=str,
        choices=["comprehensive", "performance", "ml", "trading"],
        help="Generate a specific report type",
    )

    args = parser.parse_args()

    # Initialize launcher
    launcher = AdvancedMonitoringLauncher()

    try:
        # Setup monitoring
        success = await launcher.setup_monitoring()
        if not success:
            print("❌ Failed to setup monitoring system")
            sys.exit(1)

        # Start monitoring
        success = await launcher.start_monitoring()
        if not success:
            print("❌ Failed to start monitoring system")
            sys.exit(1)

        # Handle different modes
        if args.report:
            print(f"📊 Generating {args.report} report...")
            report = await launcher.generate_report(args.report)
            print("Report generated successfully")
            print(report)
        elif args.demo:
            print("🎯 Running monitoring demonstration...")
            await launcher.run_demo()
        else:
            print("🚀 Advanced monitoring system is running...")
            print("Press Ctrl+C to stop")

            # Keep running until interrupted
            while launcher.is_running:
                await asyncio.sleep(1)

    except KeyboardInterrupt:
        print("\n🛑 Received interrupt signal")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        await launcher.stop_monitoring()


if __name__ == "__main__":
    asyncio.run(main())
