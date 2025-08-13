# src/training/steps/step2_processing_labeling_feature_engineering.py

import asyncio
from typing import Any

from src.utils.logger import system_logger as _logger

# Reuse the established implementation under a numerically correct step name
from src.training.steps.step4_analyst_labeling_feature_engineering import (
    AnalystLabelingFeatureEngineeringStep,
)


async def run_step(
    symbol: str,
    exchange_name: str = "BINANCE",
    data_dir: str = "data/training",
    timeframe: str = "1m",
    exchange: str = "BINANCE",
    force_rerun: bool = False,
    pipeline_config: dict[str, Any] | None = None,
) -> bool:
    """
    Step 2: Processing, labeling, meta-labeling, and initial feature engineering.

    This step intentionally mirrors the behavior of the prior Step 4 implementation,
    but is exposed under Step 2 to preserve numerical ordering and orchestration clarity.
    """
    _logger.info("🚀 Running Step 2: Processing, labeling, meta-labeling & feature engineering...")

    # Use exchange parameter if provided, otherwise use exchange_name for backward compatibility
    actual_exchange = exchange if exchange != "BINANCE" else exchange_name

    try:
        # Build configuration for the step
        config: dict[str, Any] = {
            "symbol": symbol,
            "exchange": actual_exchange,
            "data_dir": data_dir,
            "timeframe": timeframe,
        }
        if pipeline_config:
            # Pass through relevant pipeline keys (e.g., TB params)
            for k in ("vectorized_labelling_orchestrator",):
                if k in pipeline_config:
                    config[k] = pipeline_config[k]

        step = AnalystLabelingFeatureEngineeringStep(config)
        await step.initialize()

        training_input = {
            "symbol": symbol,
            "exchange": actual_exchange,
            "data_dir": data_dir,
            "timeframe": timeframe,
            "force_rerun": force_rerun,
        }

        pipeline_state: dict[str, Any] = {}
        result = await step.execute(training_input, pipeline_state)

        return result.get("status") == "SUCCESS" if isinstance(result, dict) else True

    except Exception as e:
        _logger.exception(f"Step 2 processing/labeling/FE failed: {e}")
        return False


if __name__ == "__main__":
    async def _test():
        ok = await run_step("ETHUSDT", "BINANCE", "data/training")
        print(f"Step 2 test result: {ok}")

    asyncio.run(_test())