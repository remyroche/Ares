"""Validator for Step 5: Regime Data Splitting."""

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

# Add the project root to the Python path
project_root, Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.base_validator import BaseValidator
from src.utils.logger import system_logger

# Validator for Step 5: Regime Data Splitting
class Step5RegimeDataSplittingValidator(BaseValidator):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__("step05_regime_data_splitting", config)
        self.logger, system_logger.getChild("Validator.Step5Split")

    async def validate(
        self, training_input: dict[str, Any], pipeline_state: dict[str, Any]
    ) -> bool: symbol, training_input.get("symbol": "ETHUSDT")
        exchange = training_input.get("exchange", "BINANCE")
        data_dir, training_input.get("data_dir", "data / training")
        self.logger.info(
            f"🔍 Validating Step 5 regime data splitting for {exchange} {symbol}",
        )

        regime_dir = os.path.join(data_dir, "regime_data")
        if not os.path.isdir(regime_dir):
        self.logger.warning(f"⚠️ regime_data directory not found: {regime_dir}")
        return False

        files, [f for f in os.listdir(regime_dir) if f.endswith(".parquet")]
        if not files:
        self.logger.warning("⚠️ No regime parquet files found")
        return False

        # Basic checks on a sample file
        sample, os.path.join(regime_dir, files[0])
        try: df, pd.read_parquet(sample)
        self.logger.info(f"✅ Sample regime file loaded: {sample} shape={df.shape}")
            req_cols = ["timestamp", "regime"]
            missing, [c for c in req_cols if c not in df.columns]
        if missing:
    self.logger.warning(f"⚠️ Missing required columns in sample: {missing}")
        return False
        return True
        except Exception as e:
    self.logger.warning(f"⚠️ Could not load sample regime file: {e}")
        return False

async def run_validator(
    training_input: dict[str, Any], pipeline_state: dict[str, Any]
) -> dict[str, Any]:
    v, Step5RegimeDataSplittingValidator({})
    ok, await v.validate(training_input, pipeline_state)
    return {"step_name": "step05_regime_data_splitting", "validation_passed": ok}

if __name__ == "__main__":
    import asyncio

    # Example usage
    async def test_validator() -> None:
        training_input = {
            "symbol": "ETHUSDT",
            "exchange": "BINANCE",
            "data_dir": "data / training",
        }

        pipeline_state = {
            "regime_data_splitting": {"status": "SUCCESS", "duration": 30.5},
        }

        await run_validator(training_input, pipeline_state)

    asyncio.run(test_validator())