#!/usr/bin/env python3
"""
Simple script to run the training pipeline from step1_7 onwards.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Fix joblib configuration issue
import joblib

joblib.parallel._backend = None

from src.training.step_orchestrator import StepOrchestrator


async def main():
    """Run the pipeline from step1_7 onwards."""

    # Configuration
    symbol = "ETHUSDT"
    exchange = "BINANCE"
    start_step = "step1_7_hmm_regime_discovery"
    data_dir = "data/training"

    print(f"🚀 Starting pipeline from {start_step} for {symbol} on {exchange}")

    # Set environment variables to bypass issues
    os.environ["BLANK_TRAINING_MODE"] = "1"
    os.environ["BYPASS_MEMORY_CHECK"] = "1"
    os.environ["SKIP_RESOURCE_VALIDATION"] = "1"

    # Initialize step orchestrator
    orchestrator = StepOrchestrator(symbol, exchange, data_dir)

    # Configuration for the pipeline
    config = {
        "symbol": symbol,
        "exchange": exchange,
        "data_dir": data_dir,
        "force_rerun": True,
        "blank_training_mode": True,
    }

    try:
        # Execute from step1_7 onwards
        success = await orchestrator.execute_from_step(
            start_step=start_step, config=config, force_rerun=True
        )

        if success:
            print("✅ Pipeline completed successfully from step1_7 onwards!")
        else:
            print("❌ Pipeline failed from step1_7 onwards")
            return False

    except Exception as e:
        print(f"❌ Error running pipeline from step1_7: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    # Run the async main function
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
