#!/usr/bin/env python3
from src.utils.tprint import tprint

"""
Enhanced Monitoring Launcher

Simple launcher script for the enhanced monitoring system that provides
easy integration with existing trading systems.
"""

import asyncio
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import argparse
import sys

from .enhanced_monitoring_orchestrator import EnhancedMonitoringOrchestrator
from .trade_decision_capture import TradeDecisionContextCapture
from .shap_lime_integration import ExplainabilityIntegrator
from src.utils.logger import system_logger
import json
import logging

class EnhancedMonitoringLauncher:
    """
    Launcher for the enhanced monitoring system.

    Provides easy setup and integration with trading systems.
    """

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the enhanced monitoring launcher."""
        self.logger = system_logger.getChild("EnhancedMonitoringLauncher")

        # Load configuration
        if config_path is None:
            config_path = Path(__file__).parent / "enhanced_monitoring_config.yaml"

        self.config_path = Path(config_path)
        self.config = self._load_config()

        # Initialize components
        self.orchestrator = None
        self.context_capture = None
        self.explainability_integrator = None

        self.logger.info(f"Enhanced Monitoring Launcher initialized with config: {self.config_path}")

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)

            self.logger.info("Configuration loaded successfully")
            return config

        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            # Return default configuration
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "enhanced_monitoring": {
                "enable_monitoring": True,
                "enable_explanations": True,
                "enable_real_time_tracking": True,
                "monthly_export_enabled": True,
                "daily_export_enabled": True,
                "export_directory": "enhanced_monitoring_exports",
                "max_decisions_in_memory": 10000,
                "data_retention_days": 365,
                "cleanup_frequency_hours": 24
            },
            "shap_analysis": {
                "enable_shap": True,
                "max_features": 50,
                "explanation_timeout": 30
            },
            "lime_analysis": {
                "enable_lime": True,
                "max_features": 20,
                "num_samples": 1000,
                "explanation_timeout": 30
            },
            "trading_integration": {
                "enable_monitoring": True,
                "capture_explanations": True,
                "capture_performance_metrics": True,
                "real_time_export": False,
                "export_interval_minutes": 60,
                "max_memory_decisions": 10000
            }
        }

    async def initialize(self) -> bool:
        """Initialize the enhanced monitoring system."""
        try:
            self.logger.info("Initializing enhanced monitoring system...")

            # Initialize orchestrator
            self.orchestrator = EnhancedMonitoringOrchestrator(self.config)

            # Initialize context capture
            self.context_capture = TradeDecisionContextCapture(self.config)

            # Initialize explainability integrator
            self.explainability_integrator = ExplainabilityIntegrator(self.config)

            self.logger.info("Enhanced monitoring system initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize enhanced monitoring system: {e}")
            return False

    async def integrate_trading_systems(
        self,
        backtesting_system: Optional[Any] = None,
        paper_trading_system: Optional[Any] = None,
        live_trading_system: Optional[Any] = None
    ) -> bool:
        """Integrate with trading systems."""
        try:
            if not self.orchestrator:
                self.logger.error("Orchestrator not initialized")
                return False

            self.logger.info("Integrating with trading systems...")

            success = await self.orchestrator.integrate_trading_systems(
                backtesting_system=backtesting_system,
                paper_trading_system=paper_trading_system,
                live_trading_system=live_trading_system
            )

            if success:
                self.logger.info("Trading systems integrated successfully")
            else:
                self.logger.warning("Some trading system integrations failed")

            return success

        except Exception as e:
            self.logger.error(f"Failed to integrate trading systems: {e}")
            return False

    async def run_example(self) -> bool:
        """Run the example usage."""
        try:
            self.logger.info("Running enhanced monitoring example...")

            # Import and run the example
            from .example_enhanced_monitoring_usage import example_enhanced_monitoring_usage

            await example_enhanced_monitoring_usage()

            self.logger.info("Example completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to run example: {e}")
            return False

    async def export_data(self) -> bool:
        """Export all monitoring data."""
        try:
            if not self.orchestrator:
                self.logger.error("Orchestrator not initialized")
                return False

            self.logger.info("Exporting monitoring data...")

            success = await self.orchestrator.force_export_all()

            if success:
                self.logger.info("Data exported successfully")
            else:
                self.logger.warning("Some exports failed")

            return success

        except Exception as e:
            self.logger.error(f"Failed to export data: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        try:
            if not self.orchestrator:
                return {"error": "Orchestrator not initialized"}

            return self.orchestrator.get_monitoring_stats()

        except Exception as e:
            self.logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}

    async def cleanup(self):
        """Cleanup resources."""
        try:
            if self.orchestrator:
                await self.orchestrator.force_export_all()

            self.logger.info("Cleanup completed")

        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")

async def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Enhanced Monitoring System Launcher")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--example", action="store_true", help="Run example usage")
    parser.add_argument("--export", action="store_true", help="Export monitoring data")
    parser.add_argument("--stats", action="store_true", help="Show monitoring statistics")
    parser.add_argument("--init-only", action="store_true", help="Initialize only")

    args = parser.parse_args()

    # Initialize launcher
    launcher = EnhancedMonitoringLauncher(args.config)

    try:
        # Initialize system
        if not await launcher.initialize():
            tprint("❌ Failed to initialize enhanced monitoring system")
            sys.exit(1)

        tprint("✅ Enhanced monitoring system initialized")

        if args.init_only:
            tprint("Initialization completed. System ready for integration.")
            return

        # Run example if requested
        if args.example:
            tprint("🚀 Running enhanced monitoring example...")
            if await launcher.run_example():
                tprint("✅ Example completed successfully")
            else:
                tprint("❌ Example failed")
                sys.exit(1)

        # Export data if requested
        if args.export:
            tprint("📊 Exporting monitoring data...")
            if await launcher.export_data():
                tprint("✅ Data exported successfully")
            else:
                tprint("❌ Export failed")
                sys.exit(1)

        # Show stats if requested
        if args.stats:
            tprint("📈 Monitoring Statistics:")
            stats = launcher.get_stats()
            for key, value in stats.items():
                tprint(f"  {key}: {value}")

        # If no specific action requested, show help
        if not any([args.example, args.export, args.stats]):
            tprint("Enhanced Monitoring System ready!")
            tprint("Use --help for available options")
            tprint("Use --example to run the example usage")
            tprint("Use --export to export monitoring data")
            tprint("Use --stats to show monitoring statistics")

    except KeyboardInterrupt:
        tprint("\n🛑 Interrupted by user")

    except Exception as e:
        tprint(f"❌ Error: {e}")
        sys.exit(1)

    finally:
        # Cleanup
        await launcher.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
