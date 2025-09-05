
"""Validator for Step 20: Extended A/B Testing."""

import asyncio
import sys
import time
from pathlib import Path
from typing import Any, Dict

# Add project root to path for proper imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import get_logger, system_logger
from src.utils.common_operations import safe_json_load
import json
import logging

logger = get_logger('Step20ABTestingValidator')

class Step20ABTestingValidator:
    """Validator for Step 20: Extended A/B Testing."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = get_logger('Step20ABTestingValidator')
        self.validation_results = {}

    async def validate(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str,
        **kwargs
    ) -> bool:
        self.logger.info("🔍 Validating A/B testing step...")
        
        # Verify results file exists
        results_file = f"{data_dir}/{exchange}_{symbol}_ab_test_results.json"
        results_path = Path(results_file)
        
        if not results_path.exists():
            self.logger.error("❌ A/B testing results file missing")
            self.validation_results["ab_results"] = {"exists": False, "path": results_file}
            return False
        
        # Validate file content
        try:
            results_data = safe_json_load(results_path)
            if not results_data or not isinstance(results_data, dict):
                self.logger.error("❌ A/B testing results file is invalid")
                self.validation_results["ab_results"] = {"exists": True, "valid": False}
                return False
            
            # Check required fields
            required_fields = ["symbol", "exchange", "test_date", "variants", "winner"]
            missing_fields = [field for field in required_fields if field not in results_data]
            
            if missing_fields:
                self.logger.error(f"❌ A/B testing results missing fields: {missing_fields}")
                self.validation_results["ab_results"] = {"exists": True, "valid": False, "missing_fields": missing_fields}
                return False
            
            self.logger.info("✅ A/B testing results validation passed")
            self.validation_results["ab_results"] = {"exists": True, "valid": True, "fields": list(results_data.keys())}
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error validating A/B testing results: {e}")
            self.validation_results["ab_results"] = {"exists": True, "valid": False, "error": str(e)}
            return False


async def run_validator(
    training_input: Dict[str, Any],
    pipeline_state: Dict[str, Any],
) -> Dict[str, Any]:
    """Run the Step 20 AB Testing validator.

    Args:
        training_input: Training input parameters
        pipeline_state: Current pipeline state

    Returns:
        Dictionary containing validation results
    """
    start_time = time.time()
    try:
        symbol = training_input.get("symbol", "ETHUSDT")
        exchange = training_input.get("exchange", "BINANCE")
        timeframe = training_input.get("timeframe", "1m")
        data_dir = training_input.get("data_dir", "data_cache")
        config = training_input.get("config", {})
        kwargs = training_input.get("kwargs", {})
        
        validator = Step20ABTestingValidator(config)
        validation_passed = await validator.validate(symbol, exchange, timeframe, data_dir, **kwargs)
        
        duration = time.time() - start_time
        return {
            "step_name": "step20_ab_testing",
            "validation_passed": validation_passed,
            "validation_results": validator.validation_results,
            "duration": duration,
            "timestamp": time.time(),
        }
    except Exception as e:
        duration = time.time() - start_time
        error_result = {
            "step_name": "step20_ab_testing",
            "validation_passed": False,
            "error": f"Validator execution failed: {str(e)}",
            "error_type": type(e).__name__,
            "validation_results": {},
            "duration": duration,
            "timestamp": time.time(),
        }
        system_logger.error(f"❌ Step20 AB testing validator failed: {str(e)}")
        return error_result
