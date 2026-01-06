
import asyncio
import logging
import pandas as pd
import numpy as np
import os
import sys
from typing import Dict, Any, List

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.steps.base_step import step_registry
import src.training.steps.market_analysis # Ensure registration happens
from src.utils.tprint import tprint_info, tprint_success, tprint_error

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def run_specialist(step_name: str, config: Dict[str, Any]):
    try:
        tprint_info(f"--- Running {step_name} ---")
        step_class = step_registry.get_step(step_name)
        if not step_class:
            tprint_error(f"Step {step_name} not found in registry")
            return
        
        step = step_class(step_name=step_name)
        result = await step.execute(config)
        
        if result and result.get("success") is not False: # Some steps might return dict without 'success' key
            tprint_success(f"Successfully completed {step_name}")
            return result
        else:
            tprint_error(f"Failed to run {step_name}: {result.get('error') if result else 'No result'}")
            return None
    except Exception as e:
        tprint_error(f"Exception running {step_name}: {e}")
        import traceback
        traceback.print_exc()
        return None

async def main():
    config = {
        "symbol": "ETHUSDT",
        "exchange": "binance",
        "timeframe": "15m",
        "direction": "long",
        "is_batch_run": True,
        "afml_target_sampling_rate": 0.10,
    }
    
    specialists = [
        "enhanced_ml_risk_regime_step",
        "enhanced_xgb_macro_regime_step",
        "enhanced_ml_smc_regime_step",
        "enhanced_ml_volume_force_step",
        "enhanced_ml_liquidity_regime_step",
        "enhanced_ml_microstructure_step",
        "enhanced_ml_momentum_persistence_step",
        "enhanced_ml_path_regime_step",
        "enhanced_ml_spectral_step",
        "enhanced_ml_volatility_burst_step",
        "enhanced_xgb_meso_regime_step"
    ]
    
    for spec in specialists:
        await run_specialist(spec, config)

if __name__ == "__main__":
    asyncio.run(main())
