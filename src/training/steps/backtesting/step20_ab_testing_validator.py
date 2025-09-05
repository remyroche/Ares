
"""Validator for Step 20: Extended A/B Testing."""

import asyncio
import sys
from pathlib import Path
from typing import Any

# Add project root to path for proper imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import get_logger
from src.utils.common_operations import safe_json_load

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
    symbol: str,
    exchange: str,
    timeframe: str,
    data_dir: str,
    **kwargs
) -> dict[str, Any]:
    config = {"symbol": symbol, "exchange": exchange, "data_dir": data_dir}
    validator = Step20ABTestingValidator(config)
    validation_passed = await validator.validate(symbol, exchange, timeframe, data_dir, **kwargs)
    return {
        "step_name": "step20_ab_testing",
        "validation_passed": validation_passed,
        "validation_results": validator.validation_results,
        "duration": 0,
        "timestamp": asyncio.get_event_loop().time(),
    }
