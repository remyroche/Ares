#!/usr/bin/env python3
"""
Dependency Injection Usage Example for Ares Trading System

This example demonstrates how to use the new dependency injection system
in practical scenarios.
"""

import asyncio

from src.analyst.di_analyst import DIAnalyst
from src.config import CONFIG
from src.core.dependency_injection import AsyncServiceContainer, ServiceLifetime
from src.core.di_integration import run_di_demonstration
from src.core.di_launcher import launch_paper_trading
from src.core.service_registry import create_configured_container
from src.interfaces.base_interfaces import IAnalyst
from src.utils.logger import system_logger
from src.utils.warning_symbols import (
    failed,
)


async def basic_usage_example():
    """Basic usage example showing simple DI patterns."""
    print("\n=== Basic Usage Example ===")

    # Create configured container
    container = create_configured_container(CONFIG)

    # Register custom analyst implementation
    container.register(
        IAnalyst,
        DIAnalyst,
        lifetime=ServiceLifetime.SINGLETON,
        config={"analysis_interval": 60},
    )

    # Resolve analyst (dependencies automatically injected)
    try:
        analyst = await container.resolve_async(IAnalyst)
        print(f"✅ Created analyst: {type(analyst).__name__}")

        # Initialize analyst
        success = await analyst.initialize()
        print(f"✅ Analyst initialized: {success}")

    except Exception:
        print(failed("Failed to create analyst: {e}"))


async def launcher_usage_example():
    """Example using the DI-aware launcher."""
    print("\n=== Launcher Usage Example ===")

    try:
        # Use convenience function for paper trading
        components = await launch_paper_trading(
            symbol="ETHUSDT",
            exchange="BINANCE",
            config={
                "analyst": {"analysis_interval": 60},
                "use_modular_components": True,
            },
        )

        print(f"✅ Created {len(components)} components:")
        for name, component in components.items():
            print(f"  - {name}: {type(component).__name__}")

        # Access specific components
        analyst = components.get("analyst")
        if analyst:
            status = (
                await analyst.get_training_status()
                if hasattr(analyst, "get_training_status")
                else {}
            )
            print(f"✅ Analyst status: {status}")

    except Exception:
        print(failed("Launcher example failed: {e}"))


async def custom_factory_example():
    """Example showing custom factory usage."""
    print("\n=== Custom Factory Example ===")

    try:
        # Create custom container with specific configuration
        custom_config = CONFIG.copy()
        custom_config.update(
            {
                "analyst": {
                    "analysis_interval": 30,
                    "enable_technical_analysis": True,
                    "enable_dual_model_system": True,
                },
                "strategist": {"risk_tolerance": 0.02},
            },
        )

        container = AsyncServiceContainer(custom_config)

        # Register custom services
        def custom_analyst_factory(container: AsyncServiceContainer) -> DIAnalyst:
            config = container.get_config("analyst", {})
            return DIAnalyst(config=config)

        container.register_factory(IAnalyst, custom_analyst_factory)

        # Resolve custom analyst
        analyst = await container.resolve_async(IAnalyst)
        print(f"✅ Created custom analyst: {type(analyst).__name__}")

    except Exception:
        print(failed("Custom factory example failed: {e}"))


async def scope_management_example():
    """Example demonstrating scope management."""
    print("\n=== Scope Management Example ===")

    try:
        container = AsyncServiceContainer(CONFIG)

        # Register a scoped service
        container.register(
            "scoped_example",
            str,  # Simple example using string
            lifetime=ServiceLifetime.SCOPED,
        )

        # Begin trading session scope
        container.begin_scope("trading_session_1")

        # Resolve services within scope
        service1 = container.resolve("scoped_example")
        service2 = container.resolve("scoped_example")

        print(f"✅ Same instance within scope: {service1 is service2}")

        # End scope
        container.end_scope("trading_session_1")

        # Begin new scope
        container.begin_scope("trading_session_2")
        service3 = container.resolve("scoped_example")

        print(f"✅ Different instance in new scope: {service1 is not service3}")

        container.end_scope("trading_session_2")

    except Exception:
        print(failed("Scope management example failed: {e}"))


async def configuration_injection_example():
    """Example showing configuration injection."""
    print("\n=== Configuration Injection Example ===")

    try:
        # Create container with detailed configuration
        detailed_config = {
            "analyst": {
                "analysis_interval": 60,
                "max_analysis_history": 100,
                "enable_technical_analysis": True,
                "dual_model_system": {"model_type": "ensemble", "ensemble_size": 5},
                "market_health_analyzer": {"health_threshold": 0.7},
            },
            "exchange": {"name": "binance", "api_key": "test_key", "rate_limit": 1200},
        }

        container = create_configured_container(detailed_config)

        # Register analyst
        container.register(
            IAnalyst,
            DIAnalyst,
            config=detailed_config.get("analyst", {}),
        )

        # Resolve and check configuration
        analyst = await container.resolve_async(IAnalyst)

        if hasattr(analyst, "config"):
            print(f"✅ Analyst received configuration with {len(analyst.config)} keys")
            print(f"  - Analysis interval: {analyst.config.get('analysis_interval')}")
            print(
                f"  - Technical analysis: {analyst.config.get('enable_technical_analysis')}",
            )

    except Exception:
        print(failed("Configuration injection example failed: {e}"))


async def integration_demonstration():
    """Run the complete integration demonstration."""
    print("\n=== Complete Integration Demonstration ===")

    try:
        results = await run_di_demonstration(CONFIG)

        print("✅ Integration demonstration completed successfully!")
        print(f"  - Total components: {results['summary']['total_components']}")
        print(f"  - Initialization status: {results['summary']['is_initialized']}")
        print(f"  - Demonstration status: {results['summary']['demonstration_status']}")

    except Exception:
        print(failed("Integration demonstration failed: {e}"))


async def main():
    """Run all dependency injection examples."""
    print("🚀 Ares Trading System - Dependency Injection Examples")
    print("=" * 60)

    try:
        # Run all examples
        await basic_usage_example()
        await launcher_usage_example()
        await custom_factory_example()
        await scope_management_example()
        await configuration_injection_example()
        await integration_demonstration()

        print("\n" + "=" * 60)
        print("✅ All dependency injection examples completed successfully!")

    except Exception as e:
        print(f"\n❌ Examples failed: {e}")
        raise


if __name__ == "__main__":
    # Set up logging
    logger = system_logger.getChild("DIExamples")
    logger.info("Starting dependency injection examples")

    # Run examples
    asyncio.run(main())
