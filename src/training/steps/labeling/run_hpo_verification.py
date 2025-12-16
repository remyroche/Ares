
import asyncio
import sys
import os
import logging

# Add src to path
sys.path.append(os.path.abspath("."))

from src.training.steps.labeling.meta_labeling_hpo_sample_weighted import MetaLabelingHPOSampleWeightedStep
from src.utils.tprint import tprint_info

# Setup basic logging
logging.basicConfig(level=logging.INFO)

async def main():
    tprint_info("🚀 Starting HPO Verification Run...")
    
    step = MetaLabelingHPOSampleWeightedStep()
    
    config = {
        "symbol": "ETHUSDT",
        "exchange": "binance",
        "timeframe": "15m",
        "direction": "long",
        "enable_labeling_hpo": True,
        "execution_mode": "full",
        "force_hpo": True,
        "labeling_hpo_start_at": "layer1",  # Start here to test weighting params
        "n_trials": 10, # Keep it manageable for verification
        "confidence_level": 0.05,
    }
    
    try:
        result = await step.execute(config)
        tprint_info(f"✅ HPO Run Completed. Success: {result.get('success')}")
        if not result.get('success'):
             tprint_info(f"❌ Error: {result.get('error')}")
    except Exception as e:
        tprint_info(f"❌ Script failed with exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
