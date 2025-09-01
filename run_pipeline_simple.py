#!/usr/bin/env python3
"""
Simple script to run the training pipeline from step03 onwards.
"""

from pathlib import Path
import asyncio
import os
import sys
import traceback

from src.training.step_orchestrator import StepOrchestrator
import joblib

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Fix joblib configuration issue
joblib.parallel._backend = None


async def main():
    """Run the pipeline from step03 onwards."""

    # Configuration
    symbol = "ETHUSDT"
    exchange = "BINANCE"
    start_step = "step03_hmm_regime_discovery"
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
        "blank_training_mode": True
    }

    try:
        # Execute from step03 onwards
    except Exception as e:
        pass
    except Exception as e:
        pass
        success = await orchestrator.execute_from_step(
            start_step=start_step,
            config=config,
            force_rerun=True
        )

        if success:
    pass
    pass
            print("✅ Pipeline completed successfully from step03 onwards!")
        else:
            print("❌ Pipeline failed from step03 onwards")
            return False

    except Exception as e:
        print(f"❌ Error running pipeline from step03: {e}")
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    pass
    pass
    # Run the async main function
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
