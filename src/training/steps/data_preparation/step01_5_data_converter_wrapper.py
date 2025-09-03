"""BaseStep wrapper for Step 01_5 Data Converter.

This adapter wraps the existing unified data converter implementation so it fits
the BaseStep execution contract and the pipeline orchestration.
"""

from typing import Any, Dict, Tuple

from src.training.base_step import BaseStep
from src.core.decorators import handles_errors
from src.utils.logger import system_logger


class DataConverterStep(BaseStep):
    """Step 01_5: Data Converter implemented using BaseStep contract."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "01_5", "data_converter")

    def _initialize_step(self) -> None:
        self.logger.info("✅ Data converter wrapper initialized")

    def validate_inputs(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Tuple[bool, list]:
        # Requires raw_market_data from step 01 (or allows direct data_dir paths)
        return True, []

    @handles_errors(
        exceptions=(Exception,),
        default_return={"success": False},
        context="data converter execution",
    )
    async def execute_logic(
        self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        # Defer to the existing conversion entrypoint and then expose standard outputs
        from src.training.steps.data_preparation.step01_5_data_converter import (
            run_step as run_step_15,
            UnifiedDataConverter,
        )

        symbol = training_input.get("symbol", "ETHUSDT")
        exchange = training_input.get("exchange", "BINANCE")
        timeframe = training_input.get("timeframe", "1m")
        data_dir = training_input.get("data_dir")
        force_rerun = training_input.get("force_rerun", False)

        self.logger.info("🔄 Running unified data converter (Step 01_5)...")
        ok = await run_step_15(
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            data_dir=data_dir,
            force_rerun=force_rerun,
        )

        # Retrieve produced paths via the converter's helpers for standardized outputs
        converter = UnifiedDataConverter({})
        unified_path = converter.get_unified_data_path(symbol, exchange, timeframe)
        unified_config_path = converter.get_unified_config_path(symbol, exchange, timeframe)

        pipeline_state["unified_data_path"] = unified_path
        pipeline_state["unified_config_path"] = unified_config_path
        pipeline_state["unified_data_ok"] = ok

        return pipeline_state

    def validate_outputs(self, pipeline_state: Dict[str, Any]) -> Tuple[bool, list]:
        errors: list = []
        if "unified_data_path" not in pipeline_state:
            errors.append("Missing unified_data_path")
        return len(errors) == 0, errors

    def get_required_inputs(self) -> list:
        return ["raw_market_data (from step01) or configured data sources"]

    def get_produced_outputs(self) -> list:
        return ["unified_data_path", "unified_config_path", "unified_data_ok"]

    def get_dependencies(self) -> list:
        return ["01_data_collection"]

