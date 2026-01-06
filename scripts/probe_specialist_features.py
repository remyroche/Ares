
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.specialist_feature_diagnostics import _prepare_labels, _load_specialist_features

async def probe_features():
    symbol = "ETHUSDT"
    exchange = "binance"
    timeframe = "15m"
    direction = "long"
    model = "analyst"
    target_col = "binary_label_long"
    regime_timeframe = "1h"
    
    try:
        y, training_index, realized_return = _prepare_labels(
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            direction=direction,
            model=model,
            target_col=target_col,
        )
        print(f"Target y size: {len(y)}")
        print(f"Training index size: {len(training_index)}")
        
        specialist_df = _load_specialist_features(
            symbol=symbol,
            exchange=exchange,
            base_timeframe=timeframe,
            regime_timeframe=regime_timeframe,
            direction=direction,
            model=model,
            training_index=training_index,
            enable_risk_hmm_specialist=False,
        )
        
        print(f"Specialist DF shape: {specialist_df.shape}")
        print("\nColumns:")
        for col in sorted(specialist_df.columns):
            print(f"  {col}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(probe_features())
