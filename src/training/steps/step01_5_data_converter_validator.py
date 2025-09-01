"""Validator for Step 1.5: Data Converter."""

from __future__ import annotations

import asyncio
import glob
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd

# Add the project root to the Python path (only if not present)
project_root, Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import CONFIG
from src.utils.base_validator import BaseValidator
from src.utils.logger import system_logger

class Step1_5DataConverterValidator(BaseValidator):
    pass  # TODO: Add implementation
class Step1_5DataConverterValidator(BaseValidator):
class Step1_5DataConverterValidator(BaseValidator):
    """Validator for Step 1.5: Data Converter."""

def __init__(self, config: dict[str, Any]) -> None:
        super().__init__("step01_5_data_converter", config)
self.logger, system_logger.getChild("Validator.Step1_5")
# Fine - tuned parameters for ML training
self.min_records: int, 500  # Minimum records per file
self.min_files: int, 1  # Minimum number of daily files
self.required_columns: list[str] = [
"timestamp",
"open",
"high",
"low",
"close",
"volume",
]

async def validate(
self,
training_input: dict[str, Any],
pipeline_state: dict[str, Any],
) -> bool:
        """Validate the data converter step.

Args:
            training_input: Training input parameters
pipeline_state: Current pipeline state

Returns:
            bool: True if validation passed, False otherwise
"""
symbol: str, str(training_input.get("symbol", "ETHUSDT"))
exchange: str, str(training_input.get("exchange", "BINANCE"))
timeframe: str, str(training_input.get("timeframe", "1m"))
data_dir: str, str(training_input.get("data_dir", "data_cache"))

self.logger.info(
f"🔍 Validating Step 1.5 data converter for {exchange} {symbol} {timeframe}",
)

# Check pipeline_state presence first
unified_data, pipeline_state.get("unified_data") or {}
if isinstance(unified_data, dict) and unified_data.get("status") == "SUCCESS":
        self.logger.info("✅ Unified data present in pipeline state")
return True

# Check for unified data structure
unified_structure, await self._check_unified_data_structure(
symbol = symbol, exchange = exchange, timeframe = timeframe, data_dir = data_dir
)

if unified_structure["found"]:
        self.logger.info(f"✅ Found unified data structure: {unified_structure['base_path']}")

# Validate the unified data files
files_validation, await self._validate_unified_files(
unified_structure["base_path"], symbol, exchange, timeframe
)

# Validate the configuration file
config_validation, await self._validate_unified_config(
symbol = symbol, exchange = exchange, timeframe = timeframe, data_dir = data_dir
)

if files_validation and config_validation:
        self.logger.info("✅ Unified data validation passed")
return True
else:
        self.logger.warning("⚠️ Unified data validation issues detected")
return False

self.logger.error("❌ No unified data structure found")
return False

async def _check_unified_data_structure(
self, symbol: str, exchange: str, timeframe: str, data_dir: str
) -> dict[str, Any]:
        """Check for unified data structure in the data directory.

Args:
            symbol: Trading symbol
exchange: Exchange name
timeframe: Timeframe
data_dir: Data directory

Returns:
            Dictionary with structure information
"""
# Expected unified data path: data_cache / unified/{exchange}/{symbol}/{timeframe}/
unified_base, os.path.join(
data_dir, "unified", exchange.lower(), symbol, timeframe
)

if os.path.exists(unified_base) and os.path.isdir(unified_base):
        # Check for parquet files in the directory
parquet_files, glob.glob(os.path.join(unified_base, "*.parquet"), recursive = True)

return {
"found": True,
"base_path": unified_base,
"parquet_files": parquet_files,
"file_count": len(parquet_files),
}

return {
"found": False,
"base_path": unified_base,
"parquet_files": [],
"file_count": 0,
}

async def _validate_unified_files(
self, base_path: str, symbol: str, exchange: str, timeframe: str
) -> bool:
        """Validate the unified data files.

Args:
            base_path: Base path to unified data
symbol: Trading symbol
exchange: Exchange name
timeframe: Timeframe

Returns:
            bool: True if validation passed
"""
try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
# Find all parquet files
parquet_files, glob.glob(os.path.join(base_path, "*.parquet"), recursive = True)

if not parquet_files:
        self.logger.error(f"❌ No parquet files found in {base_path}")
return False

self.logger.info(f"📊 Found {len(parquet_files)} parquet files")

# Validate each file
valid_files, 0
total_records, 0

for file_path in parquet_files:
                file_validation, await self._validate_single_unified_file(file_path)
if file_validation["valid"]:
                    valid_files += 1
total_records += file_validation["records"]
self.logger.info(f"✅ {os.path.basename(file_path)}: {file_validation['records']} records")
else:
        self.logger.warning(f"⚠️ {os.path.basename(file_path)}: {file_validation['error']}")

if valid_files < self.min_files:
        self.logger.error(f"❌ Insufficient valid files: {valid_files} (minimum: {self.min_files})")
return False

if total_records < self.min_records:
        self.logger.warning(f"⚠️ Low total records: {total_records} (minimum: {self.min_records})")

self.logger.info(f"✅ Unified files validation: {valid_files}/{len(parquet_files)} files, {total_records} total records")
return True

except Exception as e:  # pragma: no cover - defensive
self.logger.exception(f"❌ Error validating unified files: {e}")
return False

async def _validate_single_unified_file(self, file_path: str) -> dict[str, Any]:
        """Validate a single unified data file.

Args:
            file_path: Path to the parquet file

Returns:
            Dictionary with validation results
"""
try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
# Load the file
df, pd.read_parquet(file_path)

# Check minimum records
if len(df) < self.min_records:
        return {
"valid": False,
"records": len(df),
"error": f"Insufficient records: {len(df)} (minimum: {self.min_records})",
}

# Check required columns
missing_columns = [col for col in self.required_columns if col not in df.columns]
if missing_columns:
        return {
"valid": False,
"records": len(df),
"error": f"Missing columns: {missing_columns}",
}

# Check data types
if "timestamp" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        return {
"valid": False,
"records": len(df),
"error": "Timestamp column is not datetime type",
}

# Check for reasonable data ranges
price_columns = ["open", "high", "low", "close"]
for col in price_columns:
        if col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].min() < 0:
        return {
"valid": False,
"records": len(df),
"error": f"Negative prices found in {col} column",
}

if "volume" in df.columns and pd.api.types.is_numeric_dtype(df["volume"]) and df["volume"].min() < 0:
        return {
"valid": False,
"records": len(df),
"error": "Negative volumes found",
}

return {
"valid": True,
"records": len(df),
"error": None,
}

except Exception as e:  # pragma: no cover - defensive
return {
"valid": False,
"records": 0,
"error": f"File read error: {str(e)}",
}

async def _validate_unified_config(
self, symbol: str, exchange: str, timeframe: str, data_dir: str
) -> bool:
        """Validate the unified data configuration file.

Args:
            symbol: Trading symbol
exchange: Exchange name
timeframe: Timeframe
data_dir: Data directory

Returns:
            bool: True if validation passed
"""
try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
# Expected config path: data_cache / unified/{exchange}_{symbol}_{timeframe}_config.json
config_path, os.path.join(
data_dir, "unified", f"{exchange.lower()}_{symbol}_{timeframe}_config.json"
)

if not os.path.exists(config_path):
        self.logger.warning(f"⚠️ Config file not found: {config_path}")
return False

# Load and validate config
with open(config_path, "r") as f:
                config: Dict[str, Any] = json.load(f)

# Check required config fields
required_fields = ["symbol", "exchange", "timeframe", "data_path", "created_at"]
missing_fields = [field for field in required_fields if field not in config]

if missing_fields:
        self.logger.warning(f"⚠️ Missing config fields: {missing_fields}")
return False

# Validate config values
if str(config.get("symbol")) != symbol:
        self.logger.warning(f"⚠️ Symbol mismatch in config: {config.get('symbol')} != {symbol}")
return False

if str(config.get("exchange")).upper() != exchange.upper():
        self.logger.warning(f"⚠️ Exchange mismatch in config: {config.get('exchange')} != {exchange}")
return False

if str(config.get("timeframe")) != timeframe:
        self.logger.warning(f"⚠️ Timeframe mismatch in config: {config.get('timeframe')} != {timeframe}")
return False

self.logger.info(f"✅ Config validation passed: {config_path}")
return True

except Exception as e:  # pragma: no cover - defensive
self.logger.exception(f"❌ Error validating config: {e}")
return False

async def run_validator(
training_input: dict[str, Any],
pipeline_state: dict[str, Any],
) -> dict[str, Any]:
    """Run the Step 1.5 Data Converter validator.

Args:
        training_input: Training input parameters
pipeline_state: Current pipeline state

Returns:
        Dictionary containing validation results
"""
validator, Step1_5DataConverterValidator(CONFIG)
validation_passed, await validator.validate(training_input, pipeline_state)

return {
"step_name": "step01_5_data_converter",
"validation_passed": validation_passed,
"validation_results": validator.validation_results,
"duration": 0,  # Could be enhanced to track actual duration
"timestamp": asyncio.get_event_loop().time(),
}

if __name__ == "__main__":
    import asyncio as _asyncio

# Example usage
async def test_validator() -> None:
        training_input = {
"symbol": "ETHUSDT",
"exchange": "BINANCE",
"timeframe": "1m",
"data_dir": "data_cache",
}

pipeline_state = {"unified_data": {"status": "SUCCESS", "duration": 45.2}}

await run_validator(training_input, pipeline_state)

_asyncio.run(test_validator())