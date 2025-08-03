#!/usr/bin/env python3
"""
Demo script for MLFlow bot version tracking.

This script demonstrates how the bot version is automatically included in MLFlow runs
and how to query runs by bot version.
"""

import asyncio
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.utils.mlflow_utils import (
    log_bot_version_to_mlflow,
    log_training_metadata_to_mlflow,
    get_run_with_bot_version,
)
from src.config import ARES_VERSION
import mlflow


async def demo_mlflow_version_tracking():
    """Demonstrate MLFlow bot version tracking functionality."""
    
    print("🚀 Ares Trading Bot - MLFlow Version Tracking Demo")
    print("=" * 60)
    print(f"Current Bot Version: {ARES_VERSION}")
    print()
    
    # Set up MLFlow
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment("Ares_Trading_Models")
    
    print("📊 Starting MLFlow run with bot version tracking...")
    
    with mlflow.start_run(run_name="demo_version_tracking"):
        # Log bot version and training metadata
        log_bot_version_to_mlflow()
        log_training_metadata_to_mlflow(
            symbol="ETHUSDT",
            timeframe="15m",
            model_type="demo_model"
        )
        
        # Log some sample metrics
        mlflow.log_metric("accuracy", 0.85)
        mlflow.log_metric("precision", 0.82)
        mlflow.log_metric("recall", 0.88)
        
        run_id = mlflow.active_run().info.run_id
        print(f"✅ Created MLFlow run: {run_id}")
        print(f"   - Bot Version: {ARES_VERSION}")
        print(f"   - Symbol: ETHUSDT")
        print(f"   - Timeframe: 15m")
        print(f"   - Model Type: demo_model")
        print()
        
        # Retrieve and display run information
        print("📋 Retrieving run information with bot version...")
        run_info = get_run_with_bot_version(run_id)
        
        if run_info:
            print("✅ Run information retrieved:")
            for key, value in run_info.items():
                print(f"   - {key}: {value}")
        else:
            print("❌ Failed to retrieve run information")
        
        print()
        print("📚 Changelog Information:")
        print("   - Manual changelog: docs/BOT_CHANGELOG.md")
        print("   - Version tracking: All MLFlow runs now include bot version")
        print("   - Query runs by version: Use get_run_with_bot_version()")
        print()
        print("🎯 Benefits:")
        print("   - Track which bot version trained each model")
        print("   - Reproduce results with specific bot versions")
        print("   - Compare models across different bot versions")
        print("   - Maintain audit trail of bot changes")


def show_changelog_info():
    """Display information about the changelog system."""
    
    print("\n📖 Changelog System Information")
    print("=" * 40)
    print("The bot now includes a manual changelog system:")
    print()
    print("📄 File: docs/BOT_CHANGELOG.md")
    print("   - Tracks all bot version changes")
    print("   - Documents main changes for each version")
    print("   - Includes dates and descriptions")
    print()
    print("🔄 How to update:")
    print("   1. Update ARES_VERSION in src/config.py")
    print("   2. Add entry to docs/BOT_CHANGELOG.md")
    print("   3. Document main changes")
    print("   4. Commit with descriptive message")
    print()
    print("🔍 MLFlow Integration:")
    print("   - Bot version automatically logged to all MLFlow runs")
    print("   - Training metadata includes version, date, symbol, timeframe")
    print("   - Query runs by version using utility functions")
    print()
    print("📊 Current Version Information:")
    print(f"   - Bot Version: {ARES_VERSION}")
    print("   - Version Date: 2024-11-01")
    print("   - Main Changes: MLFlow integration, changelog system")
    print("   - Status: Production Ready")


if __name__ == "__main__":
    print("Ares Trading Bot - MLFlow Version Tracking Demo")
    print("=" * 60)
    
    # Run the demo
    asyncio.run(demo_mlflow_version_tracking())
    
    # Show changelog information
    show_changelog_info()
    
    print("\n✅ Demo completed successfully!")
    print("Check the MLFlow UI to see the bot version tags in the run.") 