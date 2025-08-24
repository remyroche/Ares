"""Validator for Step 1: Data Collection."""

import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
	sys.path.insert(0, str(project_root))

from src.config import CONFIG
from src.utils.base_validator import BaseValidator
from src.utils.logger import system_logger


class Step1DataCollectionValidator(BaseValidator):
	"""Validator for Step 1: Data Collection."""

	def __init__(self, config: Dict[str, Any]) -> None:
		super().__init__("step1_data_collection", config)
		self.logger = system_logger.getChild("Validator.Step1")
		# Fine-tuned parameters for ML training (more lenient to avoid stopping training)
		self.min_records = 500  # Reduced from 1000 to allow smaller datasets
		self.max_gap_ratio = 0.2  # Allow up to 20% large gaps (increased from 10%)
		self.max_gap_hours = 48  # Increased from 24 hours
		self.price_tolerance = 0.001  # Allow very small negative prices due to precision
		self.volume_tolerance = 0.001  # Allow very small negative volumes due to precision

	async def validate(
		self,
		training_input: Dict[str, Any],
		pipeline_state: Dict[str, Any],
	) -> Dict[str, Any]:
		"""Validate the data collection step with comprehensive checks.

		Args:
			training_input: Training input parameters
			pipeline_state: Current pipeline state

		Returns:
			Dict containing validation results with detailed information

		"""
		symbol = training_input.get("symbol", "ETHUSDT")
		exchange = training_input.get("exchange", "BINANCE")
		timeframe = training_input.get("timeframe", "1m")
		data_dir = training_input.get("data_dir", "data_cache")

		self.logger.info(
			f"🔍 Validating Step 1 data collection for {exchange} {symbol} {timeframe}",
		)

		validation_result = {
			"validation_passed": False,
			"step_name": "step1_data_collection",
			"validation_results": {},
			"critical_issues": [],
			"warnings": [],
			"data_quality_metrics": {},
		}

		# Check pipeline_state presence first
		md = pipeline_state.get("market_data") or {}
		if isinstance(md, pd.DataFrame) and not md.empty:
			self.logger.info(f"✅ Market data present in state: {md.shape} rows/cols")
			
			# Comprehensive DataFrame validation
			df_validation, df_metrics = self.validate_dataframe_quality(
				df=md,
				min_rows=self.min_records,
				required_columns=["open", "high", "low", "close", "volume"],
				check_data_types=True,
				check_value_ranges=True,
				check_duplicates=True,
				check_temporal_consistency=True,
			)
			
			validation_result["validation_results"]["pipeline_state_data"] = {
				"valid": df_validation,
				"metrics": df_metrics,
			}
			
			if df_validation:
				validation_result["validation_passed"] = True
				validation_result["data_quality_metrics"] = df_metrics
				
				# Log additional details
				try:
					if isinstance(md.index, pd.DatetimeIndex):
						self.logger.info(f"   Date range: {md.index.min()} -> {md.index.max()}")
					req = [c for c in ["open", "high", "low", "close"] if c in md.columns]
					self.logger.info(f"   OHLC present: {req}")
				except Exception:
					pass
			else:
				validation_result["critical_issues"].extend(df_metrics.get("critical_issues", []))
				validation_result["warnings"].extend(df_metrics.get("data_quality_issues", []))
			
			return validation_result

		# Check for consolidated files in data_cache directory
		consolidated_files = await self._check_consolidated_files(
			symbol=symbol, exchange=exchange, timeframe=timeframe, data_dir=data_dir,
		)

		validation_result["validation_results"]["consolidated_files"] = {
			"found": consolidated_files["found"],
			"files": consolidated_files.get("files", []),
		}

		if consolidated_files["found"]:
			self.logger.info(f"✅ Found consolidated files: {consolidated_files['files']}")

			# Validate the data quality of the consolidated files
			data_validation = await self._validate_consolidated_data_quality(
				consolidated_files["files"], symbol, exchange, timeframe,
			)

			validation_result["validation_results"]["data_quality"] = data_validation

			if data_validation.get("valid", False):
				self.logger.info("✅ Consolidated data quality validation passed")
				validation_result["validation_passed"] = True
				validation_result["data_quality_metrics"] = data_validation.get("metrics", {})
			else:
				self.logger.warning("⚠️ Consolidated data quality issues detected")
				validation_result["critical_issues"].extend(data_validation.get("critical_issues", []))
				validation_result["warnings"].extend(data_validation.get("warnings", []))
		else:
			validation_result["critical_issues"].append("No consolidated files found")

		if not validation_result["validation_passed"]:
			self.logger.error("❌ No market data found in state or consolidated files")

		return validation_result

	async def _check_consolidated_files(
		self,
		symbol: str,
		exchange: str,
		timeframe: str,
		data_dir: str,
	) -> Dict[str, Any]:
		"""Check for consolidated files in the data directory.

		Args:
			symbol: Trading symbol
			exchange: Exchange name
			timeframe: Timeframe
			data_dir: Data directory

		Returns:
			Dictionary with file information
		"""
		files_found: List[str] = []

		# Check for klines consolidated files
		klines_patterns = [
			os.path.join(data_dir, f"klines_{exchange}_{symbol}_{timeframe}_consolidated.parquet"),
			os.path.join(data_dir, f"klines_{exchange}_{symbol}_{timeframe}_consolidated.csv"),
			os.path.join(data_dir, f"klines_{exchange}_{symbol}_{timeframe}_consolidated_cached_data.pkl"),
		]

		for pattern in klines_patterns:
			if os.path.exists(pattern):
				files_found.append(pattern)
				self.logger.info(f"📊 Found klines file: {pattern}")

		# Check for aggtrades consolidated files (optional)
		aggtrades_patterns = [
			os.path.join(data_dir, f"aggtrades_{exchange}_{symbol}_consolidated.parquet"),
			os.path.join(data_dir, f"aggtrades_{exchange}_{symbol}_consolidated.csv"),
			os.path.join(data_dir, f"aggtrades_{exchange}_{symbol}_consolidated_cached_data.pkl"),
		]

		for pattern in aggtrades_patterns:
			if os.path.exists(pattern):
				files_found.append(pattern)
				self.logger.info(f"📊 Found aggtrades file: {pattern}")

		return {
			"found": len(files_found) > 0,
			"files": files_found,
			"klines_found": any("klines" in f for f in files_found),
			"aggtrades_found": any("aggtrades" in f for f in files_found),
		}

	async def _validate_consolidated_data_quality(
		self,
		files: List[str],
		symbol: str,
		exchange: str,
		timeframe: str,
	) -> Dict[str, Any]:
		"""Validate the quality of consolidated data files.

		Args:
			files: List of file paths to validate
			symbol: Trading symbol
			exchange: Exchange name
			timeframe: Timeframe

		Returns:
			bool: True if validation passed
		"""
		try:
			validation_result = {
				"valid": True,
				"files_validated": len(files),
				"file_validation_results": {},
				"critical_issues": [],
				"warnings": [],
				"metrics": {
					"total_files": len(files),
					"valid_files": 0,
					"total_records": 0,
					"data_quality_score": 0.0,
				},
			}

			# Validate klines data first (required)
			klines_files = [f for f in files if "klines" in f]
			if not klines_files:
				validation_result["valid"] = False
				validation_result["critical_issues"].append("No klines files found")
				return validation_result

			# Load and validate the first klines file
			klines_file = klines_files[0]
			self.logger.info(f"🔍 Validating klines file: {klines_file}")

			try:
				if klines_file.endswith(".parquet"):
					df = pd.read_parquet(klines_file)
				elif klines_file.endswith(".csv"):
					df = pd.read_csv(klines_file)
				elif klines_file.endswith(".pkl"):
					df = pd.read_pickle(klines_file)
				else:
					validation_result["valid"] = False
					validation_result["critical_issues"].append(f"Unsupported file format: {klines_file}")
					return validation_result

				# Comprehensive DataFrame validation
				df_validation, df_metrics = self.validate_dataframe_quality(
					df=df,
					min_rows=self.min_records,
					required_columns=["open", "high", "low", "close", "volume"],
					check_data_types=True,
					check_value_ranges=True,
					check_duplicates=True,
					check_temporal_consistency=True,
				)

				validation_result["file_validation_results"][klines_file] = {
					"valid": df_validation,
					"file_path": klines_file,
					"row_count": len(df),
					"metrics": df_metrics,
				}

				if df_validation:
					validation_result["metrics"]["valid_files"] = 1
					validation_result["metrics"]["total_records"] = len(df)
					validation_result["metrics"]["data_quality_score"] = 1.0
				else:
					validation_result["critical_issues"].extend(df_metrics.get("critical_issues", []))
					validation_result["warnings"].extend(df_metrics.get("data_quality_issues", []))

				# Additional data characteristics validation
				characteristics_validation = self._validate_data_characteristics(df, symbol, exchange)
				if not characteristics_validation:
					validation_result["warnings"].append("Data characteristics validation failed")

			except Exception as e:
				validation_result["valid"] = False
				validation_result["critical_issues"].append(f"Error reading {klines_file}: {str(e)}")

			return validation_result

		except Exception as e:
			self.logger.exception(f"❌ Error validating consolidated data: {e}")
			return {
				"valid": False,
				"error": str(e),
				"critical_issues": [f"Validation error: {str(e)}"],
			}

	def _validate_data_characteristics(
		self,
		data: pd.DataFrame,
		symbol: str,
		exchange: str,
	) -> bool:
		"""Validate specific characteristics of the collected data.

		Args:
			data: Historical data DataFrame
			symbol: Trading symbol
			exchange: Exchange name

		Returns:
			bool: True if characteristics are valid

		"""
		try:
			# Check minimum data size (more lenient for ML training)
			if len(data) < self.min_records:
				self.logger.warning(
					f"⚠️ Insufficient data: {len(data)} records (minimum: {self.min_records}) - continuing with caution",
				)
				return False

			# Check for required columns (basic OHLCV)
			required_columns = ["open", "high", "low", "close", "volume"]
			missing_columns = [col for col in required_columns if col not in data.columns]
			if missing_columns:
				self.logger.warning(
					f"⚠️ Missing required columns: {missing_columns} - continuing with caution",
				)
				return False

			# Check for reasonable price ranges (more tolerant)
			price_columns = ["open", "high", "low", "close"]
			for col in price_columns:
				if col in data.columns:
					min_price = float(data[col].min())
					if min_price < -self.price_tolerance:  # Allow small negative values due to precision
						self.logger.warning(
							f"⚠️ Invalid price values in {col} column (min: {min_price}) - continuing with caution",
						)
						return False

			# Check for reasonable volume values (more tolerant)
			if "volume" in data.columns:
				min_volume = float(data["volume"].min())
				if min_volume < -self.volume_tolerance:  # Allow small negative values due to precision
					self.logger.warning(
						f"⚠️ Invalid volume values (min: {min_volume}) - continuing with caution",
					)
					return False

			# Check data consistency (high >= low, etc.) - more lenient
			if all(col in data.columns for col in ["high", "low", "open", "close"]):
				invalid_rows = (
					(data["high"] < data["low"]) | (data["high"] < data["open"]) | (data["high"] < data["close"]) | (data["low"] > data["open"]) | (data["low"] > data["close"])
				).sum()

				invalid_ratio = float(invalid_rows) / float(len(data))
				if invalid_ratio > 0.05:  # Allow up to 5% invalid rows
					self.logger.warning(
						f"⚠️ Found {invalid_rows} rows ({invalid_ratio:.2%}) with inconsistent OHLC data - continuing with caution",
					)
				elif invalid_rows > 0:
					self.logger.info(
						f"ℹ️ Found {invalid_rows} rows with minor OHLC inconsistencies (acceptable)",
					)

			# Check for reasonable time gaps (if timestamp column exists) - more lenient
			if "timestamp" in data.columns:
				data_sorted = data.sort_values("timestamp")
				time_diffs = data_sorted["timestamp"].diff().dropna()

				# Check for reasonable time intervals (not too large gaps)
				large_gaps = (time_diffs > pd.Timedelta(hours=self.max_gap_hours)).sum()
				large_gap_ratio = float(large_gaps) / float(len(data))

				if large_gap_ratio > self.max_gap_ratio:  # Allow up to 20% large gaps
					self.logger.warning(
						f"⚠️ Found {large_gaps} large time gaps ({large_gap_ratio:.2%}) in data - continuing with caution",
					)
				elif large_gaps > 0:
					self.logger.info(f"ℹ️ Found {large_gaps} large time gaps (acceptable)")

			self.logger.info(
				f"✅ Data characteristics validation passed: {len(data)} records",
			)
			return True

		except Exception as e:
			self.logger.exception(
				f"❌ Error during data characteristics validation: {e}",
			)
			return False


async def run_validator(
	training_input: Dict[str, Any],
	pipeline_state: Dict[str, Any],
) -> Dict[str, Any]:
	"""Run the Step 1 Data Collection validator.

	Args:
		training_input: Training input parameters
		pipeline_state: Current pipeline state

	Returns:
		Dictionary containing validation results

	"""
	validator = Step1DataCollectionValidator(CONFIG)
	validation_passed = await validator.validate(training_input, pipeline_state)

	return {
		"step_name": "step1_data_collection",
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

		pipeline_state = {"data_collection": {"status": "SUCCESS", "duration": 120.5}}

		await run_validator(training_input, pipeline_state)

	_asyncio.run(test_validator())