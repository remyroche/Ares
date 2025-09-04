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
        # Extract parameters with defaults
        params = self._extract_conversion_parameters(training_input)
        
        # Run the conversion step
        conversion_success = await self._run_conversion_step(params)
        
        # Update pipeline state with results
        self._update_pipeline_state(pipeline_state, params, conversion_success)
        
        return pipeline_state

    def _extract_conversion_parameters(self, training_input: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and validate conversion parameters from training input."""
        return {
            "symbol": training_input.get("symbol", "ETHUSDT"),
            "exchange": training_input.get("exchange", "BINANCE"),
            "timeframe": training_input.get("timeframe", "1m"),
            "data_dir": training_input.get("data_dir"),
            "force_rerun": training_input.get("force_rerun", False)
        }

    async def _run_conversion_step(self, params: Dict[str, Any]) -> bool:
        """Run the actual conversion step."""
        from src.training.steps.data_preparation.step01_5_data_converter import run_step as run_step_15
        
        self.logger.info("🔄 Running unified data converter (Step 01_5)...")
        return await run_step_15(
            symbol=params["symbol"],
            exchange=params["exchange"],
            timeframe=params["timeframe"],
            data_dir=params["data_dir"],
            force_rerun=params["force_rerun"],
        )

    def _update_pipeline_state(self, pipeline_state: Dict[str, Any], params: Dict[str, Any], conversion_success: bool):
        """Update pipeline state with conversion results."""
        from src.training.steps.data_preparation.step01_5_data_converter import UnifiedDataConverter
        
        converter = UnifiedDataConverter({})
        unified_path = converter.get_unified_data_path(params["symbol"], params["exchange"], params["timeframe"])
        unified_config_path = converter.get_unified_config_path(params["symbol"], params["exchange"], params["timeframe"])

        pipeline_state.update({
            "unified_data_path": unified_path,
            "unified_config_path": unified_config_path,
            "unified_data_ok": conversion_success
        })

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

