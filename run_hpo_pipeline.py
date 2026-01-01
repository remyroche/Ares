
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Ensure src is in path
sys.path.append(".")

from src.training.steps.labeling.meta_labeling_hpo_sample_weighted import MetaLabelingHPOSampleWeightedStep
from src.utils.tprint import tprint_info, tprint_error


import asyncio

def main():
    parser = argparse.ArgumentParser(description="Run HPO Pipeline")
    parser.add_argument("--symbol", type=str, default="ETHUSDT")
    parser.add_argument("--timeframe", type=str, default="15m")
    parser.add_argument("--start-date", type=str)
    parser.add_argument("--end-date", type=str)
    parser.add_argument("--force-hpo", action="store_true")
    
    args = parser.parse_args()
    
    config = {
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "execution_mode": "full",
        "start_date": args.start_date,
        "end_date": args.end_date,
        "force_hpo": args.force_hpo,
        "layer1_reoptimize": True,
        "labeling_hpo_start_at": "layer0" if args.force_hpo else None,
        # Required for BaseStep to validate config?
        "exchange": "binance", # Adding exchange just in case
    }
    
    tprint_info(f"Starting HPO Pipeline with config: {config}")
    
    # Initialize with step name string
    step = MetaLabelingHPOSampleWeightedStep("meta_labeling_hpo_sample_weighted")
    
    try:
        # Run async
        result = asyncio.run(step.run(config))
        
        if result.get("success"):
            tprint_info("✅ Pipeline completed successfully.")
        else:
            tprint_error(f"❌ Pipeline failed: {result.get('error')}")
            sys.exit(1)
    except Exception as e:
        tprint_error(f"❌ Pipeline crashed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
