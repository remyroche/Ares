#!/usr/bin/env python3
"""
Advanced Monitoring System Launcher

This script demonstrates how to set up and use the advanced monitoring and tracking system
for the Ares trading bot.
"""

import asyncio
import json
import signal
import sys
import time
from datetime import datetime
from typing import Dict, Any

from src.utils.logger import system_logger
from src.monitoring import MonitoringIntegrationManager, setup_monitoring_integration_manager


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

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.is_running = False

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default monitoring configuration."""
        return {
            "monitoring": {
                "enabled": True,
                
                # Metrics Dashboard Configuration
                "metrics_dashboard": {
                    "update_interval": 5,
                    "max_metric_history": 1000,
                    "enable_real_time_updates": True,
                    "enable_websocket_broadcast": True,
                },
                
                # Advanced Tracer Configuration
                "advanced_tracer": {
                    "enable_tracing": True,
                    "correlation_id_header": "X-Correlation-ID",
                    "trace_sampling_rate": 1.0,
                    "max_trace_history": 10000,
                    "enable_performance_tracing": True,
                    "enable_error_tracing": True,
                },
                
                # Correlation Manager Configuration
                "correlation_manager": {
                    "enable_correlation_tracking": True,
                    "correlation_timeout": 300,
                    "max_correlation_history": 10000,
                    "enable_performance_analysis": True,
                    "enable_debug_aggregation": True,
                },
                
                # ML Monitor Configuration
                "ml_monitor": {
                    "enable_online_learning": True,
                    "drift_detection_enabled": True,
                    "feature_importance_tracking": True,
                    "auto_retraining_enabled": True,
                    "drift_threshold": 0.1,
                    "drift_check_interval": 300,
                    "performance_check_interval": 60,
                    "feature_analysis_interval": 600,
                },
                
                # Report Scheduler Configuration
                "report_scheduler": {
                    "enable_automated_reports": True,
                    "default_schedule": "daily",
                    "email_distribution": False,  # Disabled for demo
                    "report_formats": ["json", "html"],
                    "default_recipients": [],
                },
                
                # Tracking System Configuration
                "tracking_system": {
                    "enable_correlation_tracking": True,
                    "enable_ensemble_tracking": True,
                    "enable_regime_tracking": True,
                    "enable_decision_path_tracking": True,
                    "max_tracking_history": 50000,
                },
                
                # Integration Manager Configuration
                "monitoring_integration": {
                    "enable_unified_monitoring": True,
                    "enable_cross_component_tracking": True,
                    "enable_performance_correlation": True,
                },
            }
        }

    async def setup_monitoring(self) -> bool:
        """Setup the monitoring system."""
        try:
            self.logger.info("Setting up advanced monitoring system...")
            
            # Get configuration
            config = self._get_default_config()
            
            # Setup integration manager
            self.integration_manager = await setup_monitoring_integration_manager(config)
            
            if not self.integration_manager:
                self.logger.error("Failed to setup monitoring integration manager")
                return False
            
            self.logger.info("✅ Advanced monitoring system setup completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting up monitoring: {e}")
            return False

    async def start_monitoring(self) -> bool:
        """Start the monitoring system."""
        try:
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
            else:
                self.logger.error("Failed to start monitoring integration")
                return False
                
        except Exception as e:
            self.logger.error(f"Error starting monitoring: {e}")
            return False

    async def run_demo(self) -> None:
        """Run a demonstration of the monitoring system."""
        try:
            self.logger.info("Running monitoring system demonstration...")
            
            # Demo loop
            demo_counter = 0
            while self.is_running and demo_counter < 60:  # Run for 5 minutes
                try:
                    # Get unified dashboard data
                    dashboard_data = self.integration_manager.get_unified_dashboard_data()
                    
                    # Print status every 10 seconds
                    if demo_counter % 10 == 0:
                        self._print_status(dashboard_data, demo_counter)
                    
                    # Simulate some activity
                    await self._simulate_activity()
                    
                    await asyncio.sleep(5)  # Update every 5 seconds
                    demo_counter += 1
                    
                except Exception as e:
                    self.logger.error(f"Error in demo loop: {e}")
                    await asyncio.sleep(5)
                    
        except Exception as e:
            self.logger.error(f"Error running demo: {e}")

    def _print_status(self, dashboard_data: Dict[str, Any], counter: int) -> None:
        """Print monitoring status."""
        try:
            print(f"\n{'='*60}")
            print(f"Advanced Monitoring System Status - {datetime.now().strftime('%H:%M:%S')}")
            print(f"Demo iteration: {counter}")
            print(f"{'='*60}")
            
            # Print integration status
            integration_status = dashboard_data.get("integration_status", {})
            print(f"Integration Status: {'✅ Active' if integration_status.get('is_integrated') else '❌ Inactive'}")
            
            # Print component status
            components = integration_status.get("components", {})
            print("\nComponent Status:")
            for component, active in components.items():
                status = "✅ Active" if active else "❌ Inactive"
                print(f"  {component.replace('_', ' ').title()}: {status}")
            
            # Print cross-component metrics
            cross_metrics = dashboard_data.get("cross_component_metrics", {})
            if cross_metrics:
                print(f"\nActive Components: {len(cross_metrics)}")
                
                # Print tracer statistics
                if "tracer" in cross_metrics:
                    tracer_stats = cross_metrics["tracer"]
                    print(f"  Tracer - Requests: {tracer_stats.get('total_requests', 0)}, "
                          f"Active Spans: {tracer_stats.get('active_spans', 0)}")
                
                # Print correlation statistics
                if "correlation" in cross_metrics:
                    corr_stats = cross_metrics["correlation"]
                    print(f"  Correlation - Requests: {corr_stats.get('total_requests', 0)}, "
                          f"Success Rate: {corr_stats.get('success_rate', 0):.2%}")
                
                # Print ML monitor statistics
                if "ml_monitor" in cross_metrics:
                    ml_stats = cross_metrics["ml_monitor"]
                    print(f"  ML Monitor - Models: {ml_stats.get('total_models', 0)}, "
                          f"Alerts: {ml_stats.get('total_alerts', 0)}")
                
                # Print tracking statistics
                if "tracking" in cross_metrics:
                    track_stats = cross_metrics["tracking"]
                    print(f"  Tracking - Ensembles: {track_stats.get('ensemble_tracking', {}).get('total_ensembles', 0)}, "
                          f"Regimes: {track_stats.get('regime_tracking', {}).get('total_regimes', 0)}")
            
            # Print performance correlations
            correlations = dashboard_data.get("performance_correlations", {})
            if correlations:
                print(f"\nPerformance Correlations: {len(correlations)}")
                for correlation_key, correlation_data in list(correlations.items())[:3]:  # Show first 3
                    corr_value = correlation_data.get("correlation", 0)
                    print(f"  {correlation_key}: {corr_value:.3f}")
            
            print(f"{'='*60}\n")
            
        except Exception as e:
            self.logger.error(f"Error printing status: {e}")

    async def _simulate_activity(self) -> None:
        """Simulate some monitoring activity."""
        try:
            if not self.integration_manager:
                return
            
            # Simulate correlation tracking
            if self.integration_manager.components.correlation_manager:
                correlation_id = f"demo_corr_{int(time.time())}"
                await self.integration_manager.components.correlation_manager.track_correlation_request(
                    correlation_id=correlation_id,
                    component_path=["analyst", "strategist", "tactician"],
                    request_data={"demo": True},
                    metadata={"simulation": True}
                )
                
                # Simulate response after a short delay
                await asyncio.sleep(0.1)
                await self.integration_manager.components.correlation_manager.track_correlation_response(
                    correlation_id=correlation_id,
                    response_data={"result": "success"},
                    error_info=None
                )
            
            # Simulate ensemble tracking
            if self.integration_manager.components.tracking_system:
                await self.integration_manager.components.tracking_system.record_ensemble_tracking(
                    ensemble_id="demo_ensemble",
                    ensemble_type="regime_ensemble",
                    individual_predictions={"model_1": 0.8, "model_2": 0.75},
                    ensemble_weights={"model_1": 0.6, "model_2": 0.4},
                    final_prediction="buy",
                    confidence=0.78
                )
                
                await self.integration_manager.components.tracking_system.record_regime_tracking(
                    regime_type="BULL_TREND",
                    regime_confidence=0.85,
                    regime_probabilities={"BULL_TREND": 0.6, "BEAR_TREND": 0.2, "SIDEWAYS": 0.2},
                    regime_features=["price_momentum", "volatility"],
                    regime_indicators={"momentum_score": 0.7, "volatility_score": 0.3},
                    regime_transition_probability=0.1
                )
            
        except Exception as e:
            self.logger.error(f"Error simulating activity: {e}")

    async def stop_monitoring(self) -> None:
        """Stop the monitoring system."""
        try:
            self.logger.info("Stopping advanced monitoring system...")
            
            self.is_running = False
            
            if self.integration_manager:
                await self.integration_manager.stop_integration()
            
            self.logger.info("🛑 Advanced monitoring system stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping monitoring: {e}")

    async def run(self) -> None:
        """Run the advanced monitoring launcher."""
        try:
            self.logger.info("Starting Advanced Monitoring Launcher...")
            
            # Setup monitoring
            if not await self.setup_monitoring():
                self.logger.error("Failed to setup monitoring system")
                return
            
            # Start monitoring
            if not await self.start_monitoring():
                self.logger.error("Failed to start monitoring system")
                return
            
            # Run demo
            await self.run_demo()
            
        except Exception as e:
            self.logger.error(f"Error running launcher: {e}")
        finally:
            await self.stop_monitoring()


async def main():
    """Main function."""
    launcher = AdvancedMonitoringLauncher()
    await launcher.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Advanced Monitoring Launcher interrupted by user")
    except Exception as e:
        print(f"❌ Error running Advanced Monitoring Launcher: {e}")
        sys.exit(1) 