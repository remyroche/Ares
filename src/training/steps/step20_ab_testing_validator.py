"""Validator for Step 20: Extended A/B Testing."""

import asyncio
from typing import Any, Dict

from src.config import CONFIG
from src.utils.base_validator import BaseValidator
from src.utils.warning_symbols import error, failed


class Step20ABTestingValidator(BaseValidator):
	"""Validator for Step 20: Extended A/B Testing."""

	def __init__(self, config: Dict[str, Any]) -> None:
		super().__init__("step20_ab_testing", config)

	async def validate(
		self,
		training_input: Dict[str, Any],
		pipeline_state: Dict[str, Any],
	) -> bool:
		self.logger.info("🔍 Validating A/B testing step...")
		symbol = training_input.get("symbol", "ETHUSDT")
		exchange = training_input.get("exchange", "BINANCE")
		data_dir = training_input.get("data_dir", "data/training")
		# Verify results file exists
		results_file = f"{data_dir}/{exchange}_{symbol}_ab_test_results.json"
		passed, metrics = self.validate_file_exists(results_file, "ab_results")
		self.validation_results["ab_results"] = metrics
		if not passed:
			self.print(failed("❌ A/B testing results file missing"))
			return False
		return True


async def run_validator(
	training_input: Dict[str, Any],
	pipeline_state: Dict[str, Any],
) -> Dict[str, Any]:
	validator = Step20ABTestingValidator(CONFIG)
	validation_passed = await validator.validate(training_input, pipeline_state)
	return {
		"step_name": "step20_ab_testing",
		"validation_passed": validation_passed,
		"validation_results": validator.validation_results,
		"duration": 0,
		"timestamp": asyncio.get_event_loop().time(),
	}