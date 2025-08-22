#!/usr/bin/env python3
"""
Direct execution of step2_feature_engineering without pipeline dependencies.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def main():
    """Run step2_feature_engineering directly."""
    try:
        # Import the step function
        from src.training.steps.step2_feature_engineering import run_step
        
        print("🚀 Running step2_feature_engineering directly...")
        
        # Run the step
        result = await run_step(
            symbol="ETHUSDT",
            exchange="BINANCE", 
            output_dir="data/training",
            timeframe="1m",
            force_rerun=True
        )
        
        if result:
            print("✅ Step2 feature engineering completed successfully!")
        else:
            print("❌ Step2 feature engineering failed!")
            
    except Exception as e:
        print(f"❌ Error running step2: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
