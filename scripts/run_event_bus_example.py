#!/usr/bin/env python3
"""
Script to run the Enhanced Event Bus example.

This script demonstrates the enhanced event bus capabilities including:
- Event publishing and subscribing
- Event persistence and replay
- Event versioning and migration
- Correlation tracking
- Metrics collection
"""

import asyncio
from src.utils.warning_symbols import (
    error,
    warning,
    critical,
    problem,
    failed,
    invalid,
    missing,
    timeout,
    connection_error,
    validation_error,
    initialization_error,
    execution_error,
)
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.examples.enhanced_event_bus_example import main

if __name__ == "__main__":
    print("🚀 Running Enhanced Event Bus Example")
    print("=" * 50)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️ Example interrupted by user")
    except Exception as e:
        print(warning("Error running example: {e}")))
        import traceback

        traceback.print_exc()

    print("=" * 50)
    print("✅ Enhanced Event Bus Example completed")
