"""
Integration script to add health check capabilities to existing components.
This shows how to integrate health checks with the Ares trading system components.
"""

import asyncio
import time
from typing import Any

from src.sentinel.health_checker import (
    AnalystHealthMixin,
    ExchangeHealthMixin,
    StrategistHealthMixin,
    TacticianHealthMixin,
    health_checker,
)
from src.sentinel.sentinel import setup_sentinel
from src.utils.logger import system_logger
from src.utils.warning_symbols import (
    error,
    failed,
    problem,
    warning,
)

logger = system_logger.getChild("HealthIntegration")


async def initialize_health_monitoring():
    """Initialize health monitoring for the Ares system."""
    try:
        logger.info("🏥 Initializing health monitoring system...")

        # Setup sentinel
        sentinel = await setup_sentinel()
        if sentinel:
            logger.info("✅ Sentinel initialized successfully")
            await sentinel.start_monitoring()
        else:
            print(failed("❌ Failed to initialize Sentinel"))

        # Register core components for health checking
        # These would be replaced with actual component instances in production
        health_checker.register_component("analyst")
        health_checker.register_component("strategist")
        health_checker.register_component("tactician")
        health_checker.register_component("supervisor")
        health_checker.register_component("sentinel")
        health_checker.register_component("database")
        health_checker.register_component("exchange_binance")
        health_checker.register_component("exchange_mexc")
        health_checker.register_component("model_manager")
        health_checker.register_component("pipeline")

        logger.info(
            f"✅ Registered {len(health_checker.component_checkers)} components for health monitoring",
        )

        # Start periodic health checks
        asyncio.create_task(periodic_health_checks())

        logger.info("🏥 Health monitoring system initialized successfully")
        return True

    except Exception:
        print(failed("❌ Failed to initialize health monitoring: {e}"))
        return False


async def periodic_health_checks():
    """Run periodic health checks on all components."""
    while True:
        try:
            logger.debug("🔍 Running periodic health checks...")

            # Get comprehensive health summary
            health_summary = await health_checker.get_health_summary()

            # Log overall system status
            overall_status = health_summary.get("summary", {}).get(
                "overall_status",
                "unknown",
            )
            overall_score = health_summary.get("summary", {}).get("overall_score", 0)

            if overall_status == "healthy":
                logger.info(f"💚 System healthy - Score: {overall_score}")
            elif overall_status == "warning":
                print(warning("⚠️ System warning - Score: {overall_score}"))
            elif overall_status in ["degraded", "critical"]:
                print(error("🔴 System {overall_status} - Score: {overall_score}"))

            # Log component issues
            components = health_summary.get("health", {}).get("components", {})
            for health_data in components.values():
                status = health_data.get("status", "unknown")
                if status not in ["healthy"]:
                    health_data.get("issues", [])
                    print(problem("⚠️ Component {component_name}: {status} - {issues}"))

        except Exception:
            print(error("Error in periodic health checks: {e}"))

        # Wait 60 seconds before next check
        await asyncio.sleep(60)


def integrate_health_check_with_component(
    component_instance: Any,
    component_type: str,
) -> Any:
    """
    Integrate health check capabilities with an existing component.

    Args:
        component_instance: The component instance to enhance
        component_type: Type of component ('analyst', 'strategist', 'tactician', 'exchange')

    Returns:
        Enhanced component with health check capabilities
    """
    try:
        # Add health check mixin based on component type
        if component_type == "analyst":
            # Add analyst health check methods
            component_instance.__class__ = type(
                component_instance.__class__.__name__ + "WithHealth",
                (component_instance.__class__, AnalystHealthMixin),
                {},
            )
        elif component_type == "strategist":
            component_instance.__class__ = type(
                component_instance.__class__.__name__ + "WithHealth",
                (component_instance.__class__, StrategistHealthMixin),
                {},
            )
        elif component_type == "tactician":
            component_instance.__class__ = type(
                component_instance.__class__.__name__ + "WithHealth",
                (component_instance.__class__, TacticianHealthMixin),
                {},
            )
        elif component_type == "exchange":
            component_instance.__class__ = type(
                component_instance.__class__.__name__ + "WithHealth",
                (component_instance.__class__, ExchangeHealthMixin),
                {},
            )

        # Register with health checker
        health_checker.register_component(component_type, component_instance)

        logger.info(f"✅ Enhanced {component_type} component with health checks")
        return component_instance

    except Exception as e:
        logger.exception(
            f"❌ Failed to integrate health checks with {component_type}: {e}",
        )
        return component_instance


async def get_component_health_report(component_name: str) -> dict[str, Any]:
    """Get a detailed health report for a specific component."""
    try:
        health_data = await health_checker.check_component_health(component_name)

        # Create detailed report
        report = {
            "component": component_name,
            "timestamp": time.time(),
            "health_data": health_data,
            "recommendations": [],
        }

        # Add recommendations based on health status
        status = health_data.get("status", "unknown")
        health_score = health_data.get("health_score", 0)
        issues = health_data.get("issues", [])

        if status == "error":
            report["recommendations"].append(
                "Immediate attention required - component is non-functional",
            )
        elif status == "critical":
            report["recommendations"].append(
                "Critical issues detected - investigate immediately",
            )
        elif status == "degraded":
            report["recommendations"].append("Performance degraded - monitor closely")
        elif status == "warning":
            report["recommendations"].append(
                "Minor issues detected - schedule maintenance",
            )
        elif health_score < 80:
            report["recommendations"].append(
                "Health score below optimal - review configuration",
            )

        # Issue-specific recommendations
        for issue in issues:
            if "not initialized" in issue.lower():
                report["recommendations"].append("Initialize component properly")
            elif "not running" in issue.lower():
                report["recommendations"].append("Start the component service")
            elif "no models" in issue.lower():
                report["recommendations"].append("Load required models")
            elif "stale data" in issue.lower():
                report["recommendations"].append("Update data sources")
            elif "high latency" in issue.lower():
                report["recommendations"].append("Check network connectivity")
            elif "rate limit" in issue.lower():
                report["recommendations"].append("Reduce API call frequency")

        return report

    except Exception as e:
        print(error("Error generating health report for {component_name}: {e}"))
        return {"component": component_name, "error": str(e), "timestamp": time.time()}


async def generate_system_health_dashboard() -> dict[str, Any]:
    """Generate comprehensive system health dashboard data."""
    try:
        # Get overall health summary
        health_summary = await health_checker.get_health_summary()

        # Get individual component reports
        component_reports = {}
        for component_name in health_checker.component_checkers:
            component_reports[component_name] = await get_component_health_report(
                component_name,
            )

        # Create dashboard data
        return {
            "system_overview": {
                "status": health_summary.get("summary", {}).get(
                    "overall_status",
                    "unknown",
                ),
                "score": health_summary.get("summary", {}).get("overall_score", 0),
                "uptime": health_summary.get("summary", {}).get("system_uptime", 0),
                "component_count": health_summary.get("summary", {}).get(
                    "component_count",
                    0,
                ),
                "timestamp": time.time(),
            },
            "resource_usage": {
                "cpu": health_summary.get("summary", {}).get("cpu_usage", 0),
                "memory": health_summary.get("summary", {}).get("memory_usage", 0),
                "disk": health_summary.get("summary", {}).get("disk_usage", 0),
            },
            "component_status": {
                "healthy": health_summary.get("health", {})
                .get("system", {})
                .get("healthy_components", 0),
                "critical": health_summary.get("health", {})
                .get("system", {})
                .get("critical_components", 0),
                "degraded": health_summary.get("health", {})
                .get("system", {})
                .get("degraded_components", 0),
            },
            "component_details": component_reports,
            "alerts": [],  # Will be populated from sentinel
            "trends": {
                "health_score_history": [],  # TODO: Implement trending
                "alert_frequency": 0,
                "component_availability": {},
            },
        }

    except Exception as e:
        print(error("Error generating health dashboard: {e}"))
        return {"error": str(e), "timestamp": time.time()}


# Example usage function
async def demo_health_monitoring():
    """Demonstrate health monitoring capabilities."""
    print("🏥 Ares Health Monitoring Demo")
    print("=" * 50)

    # Initialize health monitoring
    await initialize_health_monitoring()

    # Wait a moment for initialization
    await asyncio.sleep(2)

    # Check overall system health
    print("\n📊 System Health Summary:")
    health_summary = await health_checker.get_health_summary()
    print(
        f"Overall Status: {health_summary.get('summary', {}).get('overall_status', 'unknown')}",
    )
    print(
        f"Health Score: {health_summary.get('summary', {}).get('overall_score', 0)}/100",
    )
    print(f"Components: {health_summary.get('summary', {}).get('component_count', 0)}")

    # Check individual components
    print("\n🔍 Component Health Details:")
    for component_name in health_checker.component_checkers:
        health_data = await health_checker.check_component_health(component_name)
        status = health_data.get("status", "unknown")
        score = health_data.get("health_score", 0)
        print(f"  {component_name}: {status} ({score}/100)")

    # Show system metrics
    print("\n💻 System Metrics:")
    system_metrics = await health_checker.get_system_metrics()
    cpu = system_metrics.get("cpu", {}).get("usage_percent", 0)
    memory = system_metrics.get("memory", {}).get("percent", 0)
    disk = system_metrics.get("disk", {}).get("percent", 0)
    print(f"  CPU: {cpu}%")
    print(f"  Memory: {memory}%")
    print(f"  Disk: {disk}%")

    print("\n✅ Health monitoring demo completed!")


if __name__ == "__main__":
    asyncio.run(demo_health_monitoring())
