
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import logging

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.ml_common.get_specialist_models_outputs import get_enhanced_specialist_models_outputs
from src.utils.tprint import tprint_info, tprint_warning, tprint_success, tprint_error

async def debug_loading():
    symbol = "ETHUSDT"
    exchange = "binance"
    timeframe = "15m"
    direction = "long"
    model = "analyst"
    
    tprint_info(f"🔍 Debugging enhanced specialist loading for {symbol} {exchange} {timeframe} {direction}")
    
    df = get_enhanced_specialist_models_outputs(
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        direction=direction,
        model=model,
        strict=False
    )
    
    if df is not None:
        tprint_success(f"✅ Enhanced specialist DF shape: {df.shape}")
        tprint_info("\nColumns found:")
        for col in sorted(df.columns):
            tprint_info(f"  {col}")
    else:
        tprint_error("❌ get_enhanced_specialist_models_outputs returned None")

if __name__ == "__main__":
    import asyncio
    asyncio.run(debug_loading())
