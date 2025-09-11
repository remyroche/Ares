import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, '/Users/remyroche/Documents/Ares/src')

async def test_self_healing_hook_production():
    """Test the self-healing hook in production by directly calling HMM regime discovery."""

    print("🔧 Testing Self-Healing Hook in Production")
    print("=" * 50)

    # Import the required modules
    from src.training.steps.market_analysis.sub_pipeline import MarketAnalysisSubPipeline
    from src.training.steps.data_collection.sub_pipeline import DataCollectionSubPipeline
    from src.training.steps.data_collection.data_preparation.step01_5_data_converter import DataConverter
    from src.training.steps.market_analysis.sub_pipeline import SubPipelineConfig, ExecutionMode

    # Create a test configuration
    config = SubPipelineConfig(
        symbol="ETHUSDT",
        exchange="binance",
        timeframe="1m",
        data_dir="data/training",
        mode=ExecutionMode.FULL
    )

    print("📋 Test Configuration:")
    print(f"   Symbol: {config.symbol}")
    print(f"   Exchange: {config.exchange}")
    print(f"   Data Directory: {config.data_dir}")
    print()

    # Create the market analysis pipeline
    market_pipeline = MarketAnalysisSubPipeline(config)

    try:
        print("🚀 Starting HMM Regime Discovery (with self-healing hook)...")

        # This should trigger the self-healing hook if constant features are detected
        result = await market_pipeline._hmm_regime_discovery_pipeline(config)

        if result:
            print("✅ HMM Regime Discovery completed successfully!")
            print(f"   Regimes: {len(result.get('regime_models', []))}")
            print(f"   Statistics: {len(result.get('regime_statistics', {}))}")
            print(f"   Transitions: {len(result.get('regime_transitions', {}))}")
        else:
            print("❌ HMM Regime Discovery returned None")

    except ValueError as e:
        if "constant features" in str(e):
            print(f"🚨 Constant features detected (expected): {e}")
            print("   This means the self-healing hook was triggered!")
        else:
            print(f"❌ Unexpected ValueError: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_self_healing_hook_production())
