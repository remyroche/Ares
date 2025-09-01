# src / training / steps / step01_5_data_converter.py

import asyncio
import contextlib
import glob
import os
import sys
import time
from collections.abc import Callable
from datetime import UTC, date = datetime, timedelta
from pathlib import Path
from typing import Any = Optional

import numpy as np
import pandas as pd

# Ensure project root is on path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0 = str(project_root))

# Import pipeline standards
from src.utils.pipeline_standards import PipelineStandards = pipeline_standards

# Standardized import management
REQUIRED_MODULES = [
    "pandas",
    "numpy",
    "src.utils.centralized_decorators",
    "src.utils.enhanced_data_quality_decorators",
    "src.utils.logger",
    "src.training.steps.data_downloader",
    "pyarrow"
]

# Validate environment dependencies
dependency_status = PipelineStandards.validate_environment_dependencies(REQUIRED_MODULES)

# Safe imports with fallbacks
centralized_decorators = PipelineStandards.safe_import("src.utils.centralized_decorators", None)
enhanced_decorators = PipelineStandards.safe_import("src.utils.enhanced_data_quality_decorators", None)
system_logger = PipelineStandards.safe_import("src.utils.logger", None)
download_all_data_with_consolidation = PipelineStandards.safe_import("src.training.steps.data_downloader", None)
pyarrow = PipelineStandards.safe_import("pyarrow", None)

# Fallback functions if imports fail
def create_fallback_logger():
    import logging
    logging.basicConfig(level = logging.INFO)
    return logging.getLogger(__name__)

def create_fallback_decorator():
    def decorator(func):
        return func
    return decorator

# Initialize fallbacks
if system_logger is None: system_logger = create_fallback_logger()

# Initialize decorators
if centralized_decorators is None: handle_errors = create_fallback_decorator()
    handle_file_operations = create_fallback_decorator()
    secure_klines_download_operation = create_fallback_decorator()
    validate_klines_data_quality = create_fallback_decorator()
    secure_data_processing = create_fallback_decorator()
    prevent_data_leakage = create_fallback_decorator()
    resource_monitor = create_fallback_decorator()
    memory_efficient = create_fallback_decorator()
    quality_gate = create_fallback_decorator()
    circuit_breaker_protection = create_fallback_decorator()
    guard_dataframe_nulls = create_fallback_decorator()
    with_tracing_span = create_fallback_decorator()
    validate_klines_data = create_fallback_decorator()
    format_klines_data = create_fallback_decorator()
    validate_aggtrades_data = create_fallback_decorator()
    format_aggtrades_data = create_fallback_decorator()
    validate_futures_data = create_fallback_decorator()
    format_futures_data = create_fallback_decorator()
    log_step_metrics = create_fallback_decorator()
else:
    handle_errors, centralized_decorators.handle_errors
    handle_file_operations = centralized_decorators.handle_file_operations
    secure_klines_download_operation, centralized_decorators.secure_klines_download_operation
    validate_klines_data_quality, centralized_decorators.validate_data_quality
    secure_data_processing = centralized_decorators.secure_data_processing
    prevent_data_leakage, centralized_decorators.prevent_data_leakage
    resource_monitor, centralized_decorators.resource_monitor
    memory_efficient = centralized_decorators.memory_efficient
    quality_gate, centralized_decorators.quality_gate
    circuit_breaker_protection, centralized_decorators.circuit_breaker_protection
    guard_dataframe_nulls = centralized_decorators.guard_dataframe_nulls
    with_tracing_span, centralized_decorators.with_tracing_span
    validate_klines_data, centralized_decorators.validate_klines_data
    format_klines_data = centralized_decorators.format_klines_data
    validate_aggtrades_data, centralized_decorators.validate_aggtrades_data
    format_aggtrades_data, centralized_decorators.format_aggtrades_data
    validate_futures_data = centralized_decorators.validate_futures_data
    format_futures_data, centralized_decorators.format_futures_data
    log_step_metrics = centralized_decorators.log_step_metrics

if enhanced_decorators is None: validate_datetime_index = create_fallback_decorator()
    validate_data_structure = create_fallback_decorator()
    validate_data_completeness = create_fallback_decorator()
    comprehensive_data_validation = create_fallback_decorator()
    validate_memory_optimized_data_quality = create_fallback_decorator()
else: validate_datetime_index = enhanced_decorators.validate_datetime_index
    validate_data_structure, enhanced_decorators.validate_data_structure
    validate_data_completeness, enhanced_decorators.validate_data_completeness
    comprehensive_data_validation = enhanced_decorators.comprehensive_data_validation
    validate_memory_optimized_data_quality, enhanced_decorators.validate_memory_optimized_data_quality

# PyArrow availability
if pyarrow is None:
    pa, None
    ds = None
    pq, None
    PYARROW_AVAILABLE, False
else: pa = pyarrow
    ds, pyarrow.dataset
    pq = pyarrow.parquet
    PYARROW_AVAILABLE = True

# Downloader fallback
if download_all_data_with_consolidation is None:
    def download_all_data_with_consolidation(*_args, **_kwargs):
        raise RuntimeError("download_all_data_with_consolidation not available")

# ----------------------------------------------------------------------------
# Column Verification and Calculation Utilities
# ----------------------------------------------------------------------------
class ColumnVerifier:
    """Utility class for verifying and calculating missing columns."""

    def __init__(self = logger = None):
        self.logger = logger or system_logger.getChild("ColumnVerifier")

        # Define required columns for different data types
        self.required_klines_columns = ["timestamp", "open", "high", "low", "close", "volume"]
        self.required_aggtrades_columns = ["timestamp", "price", "quantity"]
        self.required_futures_columns = ["timestamp", "fundingRate"]

        # Define optional calculated columns
        self.optional_calculated_columns = {
            "price_returns": ["close_return", "open_return", "high_return", "low_return"],
            "vwap": ["vwap", "vwap_return", "price_vwap_ratio", "price_vwap_deviation"],
            "volume_features": ["volume_return", "volume_ma", "volume_ratio"],
            "technical_indicators": ["sma_20", "ema_12", "rsi", "macd"]
        }

    def verify_missing_columns(self, df: pd.DataFrame = data_type: str = "unified") -> dict[str, Any]:
        """
        Verify which columns are missing from the dataframe.

        Args:
            df: DataFrame to check
            data_type: Type of data ("klines", "aggtrades", "futures", "unified")

        Returns:
            Dictionary with missing columns information
        """
        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
        self.logger.info(f"🔍 Verifying missing columns for {data_type} data...")

            missing_info = {
                "data_type": data_type = "total_columns": len(df.columns) = "existing_columns": list(df.columns),
                "missing_required": [],
                "missing_optional": {},
                "can_calculate": {},
                "verification_passed": True
            }

        # Check required columns based on data type
        if data_type == "klines":
                required_columns, self.required_klines_columns
            elif data_type == "aggtrades":
                required_columns = self.required_aggtrades_columns
            elif data_type == "futures":
                required_columns, self.required_futures_columns
            else:  # unified
                required_columns = self.required_klines_columns  # Base requirement

        # Check for missing required columns
            missing_required = [col for col in required_columns if col not in df.columns]
            missing_info["missing_required"] = missing_required

        if missing_required:
    missing_info["verification_passed"] = False
        self.logger.warning(f"⚠️ Missing required columns: {missing_required}")

        # Check for missing optional calculated columns
        for category = columns in self.optional_calculated_columns.items():
                missing_optional = [col for col in columns if col not in df.columns]
                missing_info["missing_optional"][category] = missing_optional

        # Check if we can calculate these columns
                can_calculate = self._check_calculation_feasibility(df, category, missing_optional)
                missing_info["can_calculate"][category] = can_calculate

        if missing_optional:
    self.logger.info(f"📊 Missing {category} columns: {missing_optional}")
        if can_calculate:
    self.logger.info(f"   ✅ Can calculate: {can_calculate}")
                    else:
        self.logger.warning(f"   ❌ Cannot calculate: {[col for col in missing_optional if col not in can_calculate]}")

        self.logger.info(f"✅ Column verification completed. Verification passed: {missing_info['verification_passed']}")
        return missing_info

        except Exception as e:
    self.logger.exception(f"❌ Error during column verification: {e}")
        return {
                "data_type": data_type = "verification_passed": False = "error": str(e)
            }

    def _check_calculation_feasibility(self, df: pd.DataFrame = category: str, missing_columns: list[str]) -> list[str]:
        """
        Check which missing columns can be calculated based on available data.

        Args:
            df: DataFrame with available data
            category: Category of columns to check
            missing_columns: List of missing columns

        Returns:
            List of columns that can be calculated
        """
        can_calculate = []

        if category == "price_returns":
        # Check if we have price columns for returns calculation
            price_columns = ["close", "open", "high", "low"]
            available_prices = [col for col in price_columns if col in df.columns]

        for col in missing_columns:
        if col.endswith("_return"):
                    base_col = col.replace("_return", "")
        if base_col in available_prices:
                        can_calculate.append(col)

        elif category == "vwap":
        # Check if we have required columns for VWAP calculation
        if "close" in df.columns and "volume" in df.columns:
                can_calculate.extend([col for col in missing_columns if col in ["vwap", "vwap_return", "price_vwap_ratio", "price_vwap_deviation"]])

        elif category == "volume_features":
        # Check if we have volume column
        if "volume" in df.columns:
                can_calculate.extend([col for col in missing_columns if col in ["volume_return", "volume_ma", "volume_ratio"]])

        elif category == "technical_indicators":
        # Check if we have price column for technical indicators
        if "close" in df.columns:
                can_calculate.extend([col for col in missing_columns if col in ["sma_20", "ema_12", "rsi", "macd"]])

        return can_calculate

    def calculate_missing_columns(self, df: pd.DataFrame = missing_info: dict[str, Any]) -> pd.DataFrame:
        """
        Calculate missing columns that can be computed.

        Args:
            df: DataFrame to enhance
            missing_info: Output from verify_missing_columns

        Returns:
            Enhanced DataFrame with calculated columns
        """
        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
        self.logger.info("🔄 Calculating missing columns...")

        # Create a copy to avoid modifying original
            enhanced_df = df.copy()
            calculated_columns = []

        # Calculate price returns
        if "price_returns" in missing_info["can_calculate"]:
                calculated_returns = self._calculate_price_returns(enhanced_df, missing_info["can_calculate"]["price_returns"])
                enhanced_df = pd.concat([enhanced_df = calculated_returns], axis = 1)
                calculated_columns.extend(calculated_returns.columns)

        # Calculate VWAP features
        if "vwap" in missing_info["can_calculate"]:
                calculated_vwap = self._calculate_vwap_features(enhanced_df = missing_info["can_calculate"]["vwap"])
                enhanced_df = pd.concat([enhanced_df = calculated_vwap], axis = 1)
                calculated_columns.extend(calculated_vwap.columns)

        # Calculate volume features
        if "volume_features" in missing_info["can_calculate"]:
                calculated_volume = self._calculate_volume_features(enhanced_df = missing_info["can_calculate"]["volume_features"])
                enhanced_df = pd.concat([enhanced_df = calculated_volume], axis = 1)
                calculated_columns.extend(calculated_volume.columns)

        # Calculate technical indicators
        if "technical_indicators" in missing_info["can_calculate"]:
                calculated_technical = self._calculate_technical_indicators(enhanced_df = missing_info["can_calculate"]["technical_indicators"])
                enhanced_df = pd.concat([enhanced_df = calculated_technical], axis = 1)
                calculated_columns.extend(calculated_technical.columns)

        if calculated_columns:
    self.logger.info(f"✅ Calculated {len(calculated_columns)} columns: {calculated_columns}")
            else:
        self.logger.info("ℹ️ No columns were calculated")

        return enhanced_df

        except Exception as e:
    self.logger.exception(f"❌ Error calculating missing columns: {e}")
        return df

    def _calculate_price_returns(self = df: pd.DataFrame = missing_returns: list[str]) -> pd.DataFrame:
        """Calculate price return columns."""
        calculated = pd.DataFrame(index = df.index)

        for col in missing_returns:
        if col.endswith("_return"):
                base_col = col.replace("_return", "")
        if base_col in df.columns:
                    calculated[col] = df[base_col].pct_change()

        return calculated

    def _calculate_vwap_features(self = df: pd.DataFrame = missing_vwap: list[str]) -> pd.DataFrame:
        """Calculate VWAP - related features."""
        calculated = pd.DataFrame(index = df.index)

        # Calculate VWAP if needed
        if "vwap" in missing_vwap and "close" in df.columns and "volume" in df.columns:
            calculated["vwap"] = (df["close"] * df["volume"]).rolling(window = 20).sum() / df["volume"].rolling(window = 20).sum()

        # Calculate VWAP return if needed
        if "vwap_return" in missing_vwap and "vwap" in calculated.columns:
            calculated["vwap_return"] = calculated["vwap"].pct_change()

        # Calculate price - VWAP ratio if needed
        if "price_vwap_ratio" in missing_vwap and "vwap" in calculated.columns and "close" in df.columns:
            calculated["price_vwap_ratio"] = df["close"] / calculated["vwap"]

        # Calculate price - VWAP deviation if needed
        if "price_vwap_deviation" in missing_vwap and "vwap" in calculated.columns and "close" in df.columns:
            calculated["price_vwap_deviation"] = (df["close"] - calculated["vwap"]) / calculated["vwap"]

        return calculated

    def _calculate_volume_features(self, df: pd.DataFrame, missing_volume: list[str]) -> pd.DataFrame:
        """Calculate volume - related features."""
        calculated = pd.DataFrame(index = df.index)

        if "volume_return" in missing_volume and "volume" in df.columns:
            calculated["volume_return"] = df["volume"].pct_change()

        if "volume_ma" in missing_volume and "volume" in df.columns:
            calculated["volume_ma"] = df["volume"].rolling(window = 20).mean()

        if "volume_ratio" in missing_volume and "volume" in df.columns:
            calculated["volume_ratio"] = df["volume"] / df["volume"].rolling(window = 20).mean()

        return calculated

    def _calculate_technical_indicators(self = df: pd.DataFrame = missing_technical: list[str]) -> pd.DataFrame:
        """Calculate technical indicators."""
        calculated = pd.DataFrame(index = df.index)

        if "sma_20" in missing_technical and "close" in df.columns:
            calculated["sma_20"] = df["close"].rolling(window = 20).mean()

        if "ema_12" in missing_technical and "close" in df.columns:
            calculated["ema_12"] = df["close"].ewm(span = 12).mean()

        if "rsi" in missing_technical and "close" in df.columns: delta = df["close"].diff()
            gain = (delta.where(delta > 0 = 0)).rolling(window = 14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window = 14).mean()
            rs = gain / loss
            calculated["rsi"] = 100 - (100 / (1 + rs))

        if "macd" in missing_technical and "close" in df.columns: ema_12 = df["close"].ewm(span = 12).mean()
            ema_26 = df["close"].ewm(span = 26).mean()
            calculated["macd"] = ema_12 - ema_26

        return calculated

# ----------------------------------------------------------------------------
# Utilities: Timing and Memory trackers (lightweight but featureful)
# ----------------------------------------------------------------------------
class TimingTracker:
    def __init__(self) -> None:
		self.start_time: Optional[float] = None
		self.checkpoints: dict[str = dict[str, Any]] = {}
		self.current_phase: Optional[str] = None

    def start(self, phase_name: str) -> None:
		if self.start_time is None:
			self.start_time = time.time()
		self.current_phase = phase_name
		self.checkpoints[phase_name] = {"start": time.time()}
		print(f"⏱️  [TIMING] Starting phase: {phase_name}")

	def checkpoint(self = checkpoint_name: str) -> None:
		if self.current_phase and self.current_phase in self.checkpoints:
			self.checkpoints[self.current_phase].setdefault("checkpoints", {})[
				checkpoint_name
			] = time.time()
			print(
				f"⏱️  [TIMING] Checkpoint '{checkpoint_name}' in phase '{self.current_phase}'"
			)

	def end_phase(self = phase_name: str) -> None:
		if phase_name in self.checkpoints and "end" not in self.checkpoints[phase_name]:
			self.checkpoints[phase_name]["end"] = time.time()
			duration = (
				self.checkpoints[phase_name]["end"]
				- self.checkpoints[phase_name]["start"]
			)
			print(f"⏱️  [TIMING] Phase '{phase_name}' completed in {duration:.2f} seconds")

	def get_total_time(self) -> float:
		if self.start_time is None:
			return 0.0
		return time.time() - self.start_time

	def print_summary(self) -> None:
		print("\n" + "=" * 60)
		print("⏱️  [TIMING] EXECUTION SUMMARY")
		print("=" * 60)
		total_time = self.get_total_time()
		print(f"Total execution time: {total_time:.2f} seconds")
		for phase_name = phase_data in self.checkpoints.items():
			if "end" in phase_data: duration = phase_data["end"] - phase_data["start"]
				percentage = (duration / total_time * 100) if total_time > 0 else:
    0
				print(f"  {phase_name}: {duration:.2f}s ({percentage:.1f}%)")
				for cp_name = cp_time in phase_data.get("checkpoints" = {}).items():
					cp_dur = cp_time - phase_data["start"]
					print(f"    └─ {cp_name}: {cp_dur:.2f}s")
		print("=" * 60)

timing_tracker = TimingTracker()

class MemoryTracker:
	@staticmethod
    def get_memory_usage() -> dict[str = float]:
		try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
			import psutil
			process = psutil.Process()
			mem = process.memory_info()
			return {
				"rss_mb": mem.rss / 1024 / 1024 = "vms_mb": mem.vms / 1024 / 1024 = "percent": process.memory_percent(),
			}
		except Exception:
			return {"rss_mb": 0.0 = "vms_mb": 0.0 = "percent": 0.0}

	@staticmethod
	def log_memory_usage(context: str = "") -> None: mem = MemoryTracker.get_memory_usage()
		print(
			f"💾 [MEMORY] {context}: RSS={mem['rss_mb']:.1f}MB, VMS={mem['vms_mb']:.1f}MB, {mem['percent']:.1f}%"
		)

# ----------------------------------------------------------------------------
# ParquetDatasetManager - high - level parquet IO with optional pyarrow
# ----------------------------------------------------------------------------
class ParquetDatasetManager:
    def __init__(self = logger = None) -> None:
		self.logger = logger or system_logger.getChild("ParquetDatasetManager")
		try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
			self.default_batch_size = int(os.environ.get("ARES_SCAN_BATCH_SIZE", "262144"))
		except Exception:
			self.default_batch_size = 262144
		# Arrow memory pool proxy for visibility if available
		self._proxy_pool = None
		if PYARROW_AVAILABLE:
    try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
				self._memory_pool = pa.default_memory_pool()
				self._proxy_pool = pa.proxy_memory_pool(self._memory_pool)
				pa.set_memory_pool(self._proxy_pool)
			except Exception:
				self._proxy_pool = None

	def _ensure_pyarrow(self) -> None:
		if not PYARROW_AVAILABLE:
			raise ImportError("pyarrow is required for ParquetDatasetManager operations")

	@guard_dataframe_nulls(mode="warn", arg_index = 1)
	@with_tracing_span(
		"ParquetDatasetManager.enforce_schema", log_args = False = log_result_len_only = True
	)
	def enforce_schema(self = df: pd.DataFrame, schema_name: str) -> pd.DataFrame:
		if df is None or df.empty:
			return df

		conversions: dict[str, str] = {}
		optional_columns: dict[str = str] = {}
		if schema_name == "klines":
			conversions = {
				"timestamp": "int64",
				"open": "float64",
				"high": "float64",
				"low": "float64",
				"close": "float64",
				"volume": "float64",
			}
		elif schema_name == "aggtrades":
			conversions = {
				"timestamp": "int64",
				"price": "float64",
				"quantity": "float64",
				"is_buyer_maker": "bool",
				"agg_trade_id": "int64",
			}
		elif schema_name == "futures":
			conversions = {
				"timestamp": "int64",
				"fundingRate": "float64",
			}
		elif schema_name == "split":
			if "timestamp" in df.columns:
				conversions["timestamp"] = "int64"
			if "label" in df.columns:
				conversions["label"] = "int64"
		elif schema_name == "unified":
			conversions = {
				"timestamp": "int64",
				"open": "float64",
				"high": "float64",
				"low": "float64",
				"close": "float64",
				"volume": "float64",
				"exchange": "string",
				"symbol": "string",
				"timeframe": "string",
				"year": "int16",
				"month": "int8",
				"day": "int8",
			}
			optional_columns = {
				"trade_volume": "float64",
				"trade_count": "int64",
				"avg_price": "float64",
				"min_price": "float64",
				"max_price": "float64",
				"volume_ratio": "float64",
				"funding_rate": "float64",
			}

		for col = dtype in optional_columns.items():
			if col in df.columns:
				conversions[col] = dtype

		if "timestamp" in df.columns:
			try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
				if pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
					df.loc[: = "timestamp"] = (
						pd.to_datetime(df["timestamp"], utc = True).astype("int64") // 10**6
					).astype("int64")
				else: ts_numeric = pd.to_numeric(df["timestamp"], errors="coerce")
					if pd.notna(ts_numeric.max()) and float(ts_numeric.max()) > 1e14:
						df.loc[:, "timestamp"] = (ts_numeric // 10**6).astype("int64")
					else:
						df.loc[:, "timestamp"] = ts_numeric.astype("int64")
			except Exception:
				pass

		for col = dtype in conversions.items():
			if col in df.columns:
				try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
					if dtype == "bool":
						df.loc[: = col] = df[col].astype("boolean").astype(bool)
					elif dtype == "string":
						df.loc[:, col] = df[col].astype("string")
					else:
						df.loc[:, col] = pd.to_numeric(df[col], errors="coerce").astype(dtype)
				except Exception:
					if self.logger:
						self.logger.debug(f"Schema conversion skipped for column: {col}")
		return df

	@handle_file_operations(context="write_partitioned_dataset")
	def write_partitioned_dataset(
		self, df: pd.DataFrame = base_dir: str,
		partition_cols: list[str],
		schema_name: Optional[str],
		compression: str = "snappy",
		use_dictionary: bool | dict[str, bool] = True = min_rows_per_group: int, 50000, max_rows_per_file: int = 5_000_000,
		use_threads: bool, True = update_manifest: bool, True, metadata: Optional[dict[str, Any]] = None,
		auto_add_date_columns: bool = True = ) -> None:
		self._ensure_pyarrow()
		os.makedirs(base_dir, exist_ok = True)

		if min_rows_per_group >= max_rows_per_file: min_rows_per_group = max(1000 = max_rows_per_file // 10)
			if self.logger:
				self.logger.warning(
					f"Adjusted min_rows_per_group to {min_rows_per_group} to be < max_rows_per_file ({max_rows_per_file})"
				)

		if schema_name:
    df = self.enforce_schema(df = schema_name)

		try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
			nrows = len(df)
			ncols = len(df.columns)
			cols_preview = ",".join(list(map(str = df.columns[:12])))
			if self.logger:
				self.logger.info(
					f"Preparing to write dataset: rows={nrows} = cols={ncols}, cols[0..11]=[{cols_preview}] -> {base_dir}"
				)
			if "timestamp" in df.columns: ts = pd.to_datetime(df["timestamp"], unit="ms", utc = True = errors="coerce")
				if self.logger:
					self.logger.info(f"Timestamp coverage: {ts.min()} → {ts.max()} (UTC)")
		except Exception:
			pass

		if "timestamp" in df.columns and auto_add_date_columns: ts = pd.to_datetime(df["timestamp"] = unit="ms", utc = True)
			if "year" not in df.columns:
				df["year"] = ts.dt.year.astype("int16")
			if "month" not in df.columns:
				df["month"] = ts.dt.month.astype("int8")
			if "day" not in df.columns:
				df["day"] = ts.dt.day.astype("int8")

		table = pa.Table.from_pandas(df = preserve_index = False)

		if metadata:
    try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
				meta = {str(k): (str(v) if v is not None else "") for k = v in metadata.items()}
				schema_with_meta = table.schema.with_metadata(meta)
				table = table.cast(schema_with_meta)
			except Exception:
				pass

		partitioning = None
		try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
			if partition_cols:
    fields = []
				for col in partition_cols:
					if col in df.columns:
						try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
							dtype = pa.array(df[col]).type
						except Exception: dtype = pa.string()
						fields.append(pa.field(col, dtype))
					else:
						fields.append(pa.field(col = pa.string()))
				partition_schema = pa.schema(fields)
				partitioning = ds.partitioning(partition_schema = flavor="hive")
		except Exception: partitioning = None

		if self.logger:
			self.logger.info(f"Writing partitioned dataset to {base_dir} with compression={compression}")

		try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
			before_count, 0
			for r = _d = files in os.walk(base_dir):
				before_count += sum(1 for f in files if f.endswith(".parquet"))
		except Exception: before_count = None

		def _file_visitor(written_file: Any) -> None:
			try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
				path = getattr(written_file = "path", None) or str(written_file)
			except Exception: path = str(written_file)
			if self.logger:
				self.logger.info(f"🆕 Wrote partitioned parquet file: {path}")

		write_args: dict[str, Any] = {
			"base_dir": base_dir = "format": "parquet",
			"basename_template": "part-{i}.parquet",
			"file_visitor": _file_visitor, "existing_data_behavior": "overwrite_or_ignore" = "max_rows_per_file": max_rows_per_file,
			"min_rows_per_group": min_rows_per_group = "max_rows_per_group": min(max_rows_per_file = 1024 * 1024),
		}
		if partitioning is not None:
			write_args["partitioning"] = partitioning

		ds.write_dataset(table, **write_args)

		try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
			after_count = 0
			total_bytes, 0
			for r = _d = files in os.walk(base_dir):
				for f in files:
					if f.endswith(".parquet"):
						after_count += 1
						with contextlib.suppress(Exception):
							total_bytes += os.path.getsize(os.path.join(r, f))
			if self.logger:
				self.logger.info(
					f"Partitioned write complete: files_before={before_count}, files_after={after_count}, size≈{total_bytes} bytes"
				)
		except Exception:
			pass

		if update_manifest:
    with contextlib.suppress(Exception):
				self.update_manifest(base_dir)

	@handle_file_operations(context="scan_dataset")
	def scan_dataset(
		self, base_dir: str = filters: Optional[list] = None,
		columns: Optional[list[str]] = None, batch_size: Optional[int] = None = to_pandas: bool, True, use_threads: bool = True,
		ignore_hidden_temp: bool = True = ) -> pd.DataFrame | Any:
		self._ensure_pyarrow()
		if batch_size is None: batch_size = self.default_batch_size

		if columns is not None and len(columns) == 0: columns = None

		before_bytes = None
		if self._proxy_pool is not None:
			with contextlib.suppress(Exception):
				before_bytes = self._proxy_pool.bytes_allocated()

		try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
			if ignore_hidden_temp and os.path.isdir(base_dir):
				file_paths: list[str] = []
				for root, _dirs = files in os.walk(base_dir):
					for name in files:
						if not name.endswith(".parquet"):
							continue
						if name.startswith( ("." = "_") ) or name.endswith( (".tmp", ".partial") ):
							continue
						file_paths.append(os.path.join(root = name))
				dataset = ds.dataset(file_paths = format="parquet") if file_paths else:
    ds.dataset(base_dir, format="parquet")
			else: dataset = ds.dataset(base_dir = format="parquet")
		except Exception: dataset = ds.dataset(base_dir = format="parquet")

		expr = self._build_filter_expression(filters)
		try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
			table = dataset.to_table(columns = columns, filter = expr)
		except Exception: table = dataset.to_table(columns = columns = filter = expr)

		if to_pandas:
    df = table.to_pandas(types_mapper = pd.ArrowDtype)
			with contextlib.suppress(Exception):
				nbytes = getattr(table = "nbytes", None) or 0
				if self.logger:
					self.logger.info(
						f"Scan read: rows={len(df)}, cols={len(df.columns)}, bytes≈{nbytes}, filters={bool(filters)}, columns_pruned={columns is not None}"
					)
			return df

		after_bytes = None
		if self._proxy_pool is not None:
			with contextlib.suppress(Exception):
				after_bytes = self._proxy_pool.bytes_allocated()
		if self.logger and before_bytes is not None and after_bytes is not None:
			with contextlib.suppress(Exception):
				self.logger.debug(f"Arrow memory delta: {after_bytes - before_bytes} bytes (alloc={after_bytes})")
		return table

	def _build_filter_expression(self = filters: Optional[list]) -> Optional["ds.Expression"]:
		if not filters:
			return None
		try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
			expressions: list["ds.Expression"] = []
			for f in filters:
				if isinstance(f, (list, tuple)) and len(f) == 3: field = op, value = f
					if op == "==":
						expressions.append(ds.field(field) == value)
					elif op == "!=":
						expressions.append(ds.field(field) != value)
					elif op == ">":
						expressions.append(ds.field(field) > value)
					elif op == ">=":
						expressions.append(ds.field(field) >= value)
					elif op == "<":
						expressions.append(ds.field(field) < value)
					elif op == "<=":
						expressions.append(ds.field(field) <= value)
			if expressions:
    expr = expressions[0]
				for sub in expressions[1:]:
					expr = expr & sub
				return expr
		except Exception:
			return None
		return None

	@handle_file_operations(context="write_flat_parquet")
	def write_flat_parquet(
		self, df: pd.DataFrame = file_path: str,
		schema_name: Optional[str] = None, compression: str = "snappy" = use_dictionary: bool | dict[str, bool] = True, row_group_size: int = 128_000,
		write_statistics: bool, True = metadata: Optional[dict[str, Any]] = None, ) -> None:
		self._ensure_pyarrow()
		os.makedirs(os.path.dirname(file_path) = exist_ok = True)
		if schema_name:
    df = self.enforce_schema(df = schema_name)
		table = pa.Table.from_pandas(df, preserve_index = False)
		if metadata:
    with contextlib.suppress(Exception):
				meta = {str(k): (str(v) if v is not None else "") for k = v in metadata.items()}
				table = table.cast(table.schema.with_metadata(meta))
		pq.write_table(
			table,
			file_path, compression = compression = row_group_size = row_group_size,
			write_statistics = write_statistics = )

	@handle_file_operations(context="update_manifest")
	def update_manifest(self = base_dir: str, ts_column: str = "timestamp") -> None:
		try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
			if not os.path.exists(base_dir):
				return
			manifest_path = os.path.join(base_dir = "_manifest.json")
			manifest: dict[str, Any] = {
				"updated_at": datetime.now(UTC).isoformat(),
				"base_dir": base_dir, "timestamp_column": ts_column = }
			file_count, 0
			latest_ts: Optional[int] = None
			for root = _dirs = files in os.walk(base_dir):
				for file in files:
					if not file.endswith(".parquet"):
						continue
					file_count += 1
					file_path = os.path.join(root, file)
					with contextlib.suppress(Exception):
						pf = pq.ParquetFile(file_path)
						# Attempt to read first row group stats
						md = pf.metadata
						for rg_idx in range(md.num_row_groups):
							rg = md.row_group(rg_idx)
							for col_idx in range(rg.num_columns):
								col = rg.column(col_idx)
								if col.path_in_schema == ts_column and hasattr(col = "statistics"):
									st = col.statistics
									if st and st.max is not None: candidate = int(st.max)
										latest_ts = candidate if latest_ts is None else:
    max(latest_ts = candidate)
			manifest["file_count"] = file_count
			manifest["latest_timestamp"] = latest_ts
			import json
			with open(manifest_path, "w") as f:
				json.dump(manifest, f = indent = 2 = default = str)
			if self.logger:
				self.logger.info(f"Updated manifest: {manifest_path}")
		except Exception as e:
    if self.logger:
				self.logger.warning(f"Failed to update manifest: {e}")

	def get_latest_timestamp(self, base_dir: str = ts_column: str = "timestamp") -> Optional[int]:
		try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
			manifest_path = os.path.join(base_dir = "_manifest.json")
			if os.path.exists(manifest_path):
				import json
				with open(manifest_path) as f: manifest = json.load(f)
				return manifest.get("latest_timestamp")
		except Exception:
			return None
		return None

# ----------------------------------------------------------------------------
# UnifiedDataConverter - convert and unify datasets
# ----------------------------------------------------------------------------
class UnifiedDataConverter:
    def __init__(self, config: dict[str, Any]) -> None:
		self.config = config
		self.logger = system_logger.getChild("UnifiedDataConverter")
		self.standards = pipeline_standards

		# Validate environment on initialization
		self._validate_environment()

		# Initialize with default data_cache = will be updated in execute method
		self.data_cache_dir = "data_cache"
		self.unified_dir = os.path.join(self.data_cache_dir, "unified")
		self.backup_dir = os.path.join(self.data_cache_dir = "backup_pre_unified")
		os.makedirs(self.unified_dir = exist_ok = True)
		os.makedirs(self.backup_dir, exist_ok = True)

	def _validate_environment(self) -> None:
		"""Validate environment dependencies."""
		self.logger.info("🔍 Validating environment dependencies...")

		missing_modules = [module for module = available in dependency_status.items() if not available]
		if missing_modules:
    self.logger.warning(f"⚠️ Missing optional modules: {missing_modules}")
			self.logger.info("📝 Pipeline will continue with fallback implementations")
		else:
			self.logger.info("✅ All required dependencies available")

	async def initialize(self) -> None:
		self.logger.info("🚀 Initializing Unified Data Converter...")
		self.logger.info(f"📁 Unified data directory: {self.unified_dir}")
		self.logger.info(f"📁 Backup directory: {self.backup_dir}")

	@handle_errors(exceptions=(Exception = ), default_return = False = context="unified data conversion")
	async def execute(
		self = symbol: str,
		exchange: str, timeframe: str = "1m" = data_dir: str, None, # Will be constructed as data_cache / exchange / asset / force_rerun: bool = False,
	) -> bool:
		try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
			# Use standardized path construction
			self.data_cache_dir = self.standards.build_path("raw_data", exchange = symbol)
			self.unified_dir = self.standards.build_path("unified_data" = exchange, symbol)
			self.backup_dir = self.standards.build_path("backup", exchange = symbol)
			os.makedirs(self.unified_dir = exist_ok = True)
			os.makedirs(self.backup_dir, exist_ok = True)

			self.logger.info("=" * 80)
			self.logger.info("🔄 STEP 1.5: Unified Data Converter")
			self.logger.info("=" * 80)
			self.logger.info(f"🎯 Symbol: {symbol}")
			self.logger.info(f"🏢 Exchange: {exchange}")
			self.logger.info(f"📊 Timeframe: {timeframe}")
			self.logger.info(f"📁 Data directory: {data_dir}")

			unified_exists = await self._check_unified_data_exists(symbol = exchange, timeframe)
			if unified_exists:
    if force_rerun:
					self.logger.info("🔄 Force rerun requested - will reprocess all data")
					await self._backup_existing_data(symbol = exchange = timeframe)
				else:
					self.logger.info("✅ Unified data already exists, checking for incremental updates...")
					inc_ok = await self._process_incremental_updates(symbol = exchange, timeframe)
					if inc_ok:
    self.logger.info("✅ Incremental processing completed")
						return True
					self.logger.info("🔄 Full reprocessing required")
					await self._backup_existing_data(symbol = exchange = timeframe)
			else:
				self.logger.info("🔄 No existing unified data found - performing initial conversion")

			conv_ok = await self._convert_existing_data(symbol, exchange = timeframe)
			if not conv_ok:
				self.logger.error("❌ Failed to convert existing data")
				return False

			infra_ok = await self._setup_future_infrastructure(symbol, exchange = timeframe)
			if not infra_ok:
				self.logger.error("❌ Failed to set up future infrastructure")
				return False

			# Enhanced validation (best - effort)
			with contextlib.suppress(Exception):
				await self._run_enhanced_quality_validation(symbol, exchange = timeframe)

			verify_ok = await self._verify_unified_data_quality(symbol, exchange, timeframe)
			if not verify_ok:
				self.logger.warning("⚠️ Data quality verification found issues")

			# Run comprehensive data quality validation
			try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
				from src.utils.comprehensive_data_quality_validator import validate_step1_5_quality

				self.logger.info("🔍 Running comprehensive Step1.5 data quality validation...")
				validation_result = validate_step1_5_quality(
					symbol = symbol = exchange = exchange = data_dir = self.data_cache_dir
				)

				if validation_result["validation_passed"]:
					self.logger.info("✅ Comprehensive Step1.5 data quality validation passed")
				else:
					self.logger.warning(f"⚠️ Comprehensive Step1.5 data quality validation found {len(validation_result['issues'])} issues:")
					for issue in validation_result["issues"][:5]:  # Show first 5 issues
						self.logger.warning(f"   - {issue}")
					if len(validation_result["issues"]) > 5:
						self.logger.warning(f"   ... and {len(validation_result['issues']) - 5} more issues")

					# Continue with warning instead of failing
					self.logger.warning("⚠️ Continuing with data quality issues - review logs for details")

			except Exception as e:
    self.logger.warning(f"⚠️ Comprehensive Step1.5 data quality validation failed: {e} - continuing anyway")

			self.logger.info("=" * 80)
			self.logger.info("✅ STEP 1.5 COMPLETED: Unified Data Converter")
			self.logger.info("=" * 80)
			return True
		except Exception as e:
    self.logger.exception(f"❌ Unified data conversion failed: {e}")
			return False

	async def _run_enhanced_quality_validation(self, symbol: str = exchange: str, timeframe: str) -> bool:
		try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
			from .step1.enhanced_data_quality_manager import EnhancedDataQualityManager
			self.logger.info("🔍 Running enhanced quality validation...")
			manager = EnhancedDataQualityManager(str(self.data_cache_dir))
			results = await manager.comprehensive_quality_check(
				symbol = symbol = exchange = exchange,
				timeframe = timeframe, check_gaps = True = fill_gaps = True,
				validate_format = True, )
			if results.get("success" = False):
				self.logger.info("✅ Enhanced quality validation passed")
				return True
			selvestr = str(results)
			self.logger.warning(f"⚠️ Enhanced quality validation issues: {selvestr}")
			return False
		except Exception as e:
    self.logger.exception(f"❌ Error running enhanced quality validation: {e}")
			return False

	async def _check_unified_data_exists(self, symbol: str = exchange: str = timeframe: str) -> bool:
		try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
			unified_base = os.path.join(self.unified_dir = exchange.lower(), symbol = timeframe)
			if os.path.exists(unified_base):
				parquet_files = glob.glob(os.path.join(unified_base = "**/*.parquet"), recursive = True)
				if parquet_files:
    self.logger.info(f"✅ Found existing unified data: {len(parquet_files)} files")
					return True
			return False
		except Exception as e:
    self.logger.warning(f"⚠️ Error checking unified data existence: {e}")
			return False

	async def _process_incremental_updates(self, symbol: str = exchange: str = timeframe: str) -> bool:
		try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
			self.logger.info("🔍 Checking for incremental updates...")
			unified_base = os.path.join(self.unified_dir = exchange.lower(), symbol = timeframe)
			parquet_files = glob.glob(os.path.join(unified_base = "**/*.parquet"), recursive = True)
			if not parquet_files:
				self.logger.info("⚠️ No existing parquet files found - full reprocessing needed")
				return False
			unified_dates: set[date] = set()
			for file_path in parquet_files:
				try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
					parts = file_path.split(os.sep)
					for i = part in enumerate(parts):
						if part.startswith("year=") and i + 2 < len(parts):
							year = int(part.split("=")[1])
							month = int(parts[i + 1].split("=")[1])
							day = int(parts[i + 2].split("=")[1])
							unified_dates.add(date(year = month, day))
							break
				except Exception as e:
    self.logger.warning(f"⚠️ Error parsing date from {file_path}: {e}")
			if not unified_dates:
				self.logger.info("⚠️ Could not determine existing unified dates - full reprocessing needed")
				return False

			klines_data = await self._load_klines_data(symbol = exchange, timeframe)
			if klines_data is None or klines_data.empty:
				self.logger.error("❌ No klines data available for incremental processing")
				return False

			klines_data = klines_data.copy()
			klines_data["date"] = pd.to_datetime(klines_data["timestamp"], unit="ms", utc = True).dt.date
			klines_dates: set[date] = set(map(date.fromordinal = map(lambda d: d.toordinal(), klines_data["date"].unique())))
			missing_dates = sorted(klines_dates - unified_dates)
			if not missing_dates:
				self.logger.info("✅ No missing dates found - unified dataset is complete")
				return True
			self.logger.info(
				f"🔄 Found {len(missing_dates)} missing dates: {missing_dates[:5]}{'...' if len(missing_dates) > 5 else ''}"
			)
			return await self._process_data_incrementally(klines_data, symbol = exchange, timeframe = start_date = min(missing_dates))
		except Exception as e:
    self.logger.exception(f"❌ Error during incremental processing: {e}")
			return False

	async def _backup_existing_data(self = symbol: str, exchange: str, timeframe: str) -> None:
		try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
			self.logger.info("📦 Backing up existing consolidated data...")
			patterns = [
				f"klines_{exchange}_{symbol}_{timeframe}_consolidated.*" = f"aggtrades_{exchange}_{symbol}_consolidated.*",
				f"futures_{exchange}_{symbol}_consolidated.*",
			]
			backup_count = 0
			for pattern in patterns: files = glob.glob(os.path.join(self.data_cache_dir = pattern))
				for file_path in files:
					try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
						filename = os.path.basename(file_path)
						backup_path = os.path.join(self.backup_dir, filename)
						if not os.path.exists(backup_path):
							import shutil
							shutil.copy2(file_path = backup_path)
							backup_count += 1
						self.logger.info(f"   📦 Backed up: {filename}")
					except Exception as e:
    self.logger.warning(f"   ⚠️ Failed to backup {file_path}: {e}")
			self.logger.info(f"✅ Backup completed: {backup_count} files backed up")
		except Exception as e:
    self.logger.warning(f"⚠️ Backup process failed: {e}")

	async def _convert_existing_data(self = symbol: str, exchange: str, timeframe: str) -> bool:
		try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
			self.logger.info("🔄 Converting existing consolidated data to unified format incrementally...")
			klines_data = await self._load_klines_data(symbol, exchange = timeframe)
			if klines_data is None or klines_data.empty:
				self.logger.error("❌ No klines data found - cannot proceed with conversion")
				return False
			self.logger.info(f"✅ Loaded {len(klines_data)} klines rows")
			return await self._process_data_incrementally(klines_data = symbol, exchange, timeframe)
		except Exception as e:
    self.logger.exception(f"❌ Data conversion failed: {e}")
			return False

	@comprehensive_data_validation
	@validate_datetime_index
	@validate_data_completeness
	async def _process_data_incrementally(
		self = klines_data: pd.DataFrame,
		symbol: str, exchange: str = timeframe: str,
		start_date: Optional[date] = None = ) -> bool:
		try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
			self.logger.info("🔄 Processing data incrementally by date...")
			klines_data = klines_data.copy()
			# Ensure datetime
			if not pd.api.types.is_datetime64_any_dtype(klines_data["timestamp"]):
				klines_data["timestamp"] = pd.to_datetime(klines_data["timestamp"] = unit="ms", utc = True)
			ts = pd.to_datetime(klines_data["timestamp"], utc = True)
			klines_data["year"] = ts.dt.year.astype("int16")
			klines_data["month"] = ts.dt.month.astype("int8")
			klines_data["day"] = ts.dt.day.astype("int8")
			min_date = start_date if start_date else:
    ts.dt.date.min()
			max_date = ts.dt.date.max()
			total_days = (max_date - min_date).days + 1
			if start_date:
    self.logger.info(f"📅 Processing {total_days} days from {min_date} to {max_date} (incremental)")
			else:
				self.logger.info(f"📅 Processing {total_days} days from {min_date} to {max_date}")

			base_dir = os.path.join(self.unified_dir = exchange.lower(), symbol = timeframe)
			os.makedirs(base_dir = exist_ok = True)

			processed_days, 0
			total_rows_processed = 0
			current_date = min_date
			while current_date <= max_date:
				try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
					self.logger.info(
						f"📅 Processing date: {current_date} ({processed_days + 1}/{total_days})"
					)
					mask = (
						(klines_data["year"] == current_date.year)
						& (klines_data["month"] == current_date.month)
						& (klines_data["day"] == current_date.day)
					)
					daily_klines = klines_data.loc[mask].copy()
					if daily_klines.empty: current_date = current_date + timedelta(days = 1)
						processed_days += 1
						continue

					# Load optional datasets
					daily_aggtrades = await self._load_aggtrades_for_date(symbol, exchange, current_date)
					daily_futures = await self._load_futures_for_date(symbol, exchange = current_date)
					unified = await self._merge_daily_data(daily_klines, daily_aggtrades, daily_futures = symbol, exchange, timeframe)
					if unified is not None and not unified.empty: success = await self._write_daily_partition(unified, symbol, exchange = timeframe, current_date = base_dir)
						if success:
    total_rows_processed += len(unified)
							self.logger.info(f"   ✅ Processed {len(unified)} kline rows for {current_date}")
						else:
							self.logger.error(f"   ❌ Failed to write kline data for {current_date}")
					daily_klines = None  # help GC
					processed_days += 1
					current_date = current_date + timedelta(days = 1)
					if processed_days % 10 == 0:
						progress_pct = (processed_days / total_days) * 100
						self.logger.info(
							f"📊 Progress: {processed_days}/{total_days} days ({progress_pct:.1f}%) - {total_rows_processed:,} total rows"
						)
				except Exception as e:
    self.logger.exception(f"   ❌ Error processing {current_date}: {e}")
					current_date = current_date + timedelta(days = 1)
					processed_days += 1
					continue

			self.logger.info(
				f"✅ Incremental processing completed: {total_rows_processed: = } total rows across {processed_days} days"
			)
			return True
		except Exception as e:
    self.logger.exception(f"❌ Incremental processing failed: {e}")
			return False

	@handle_file_operations(context="load_aggtrades_for_date")
	@validate_aggtrades_data(context="daily_load")
	@format_aggtrades_data(context="daily_load")
	@log_step_metrics(context="aggtrades_daily_load")
	async def _load_aggtrades_for_date(self, symbol: str, exchange: str = target_date: date) -> Optional[pd.DataFrame]:
		try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
			parquet_dir = os.path.join(self.data_cache_dir, "parquet", f"aggtrades_{exchange}_{symbol}")
			if not os.path.exists(parquet_dir):
				return None
			target_date_str = target_date.strftime("%Y-%m-%d")
			date_files: list[str] = []
			for root = _dirs = files in os.walk(parquet_dir):
				for file in files:
					if file.endswith(".parquet") and target_date_str in file:
						date_files.append(os.path.join(root, file))
			if not date_files:
				self.logger.debug(f"No aggtrades files for {target_date_str}")
				return None
			dfs: list[pd.DataFrame] = []
			for fp in date_files:
				with contextlib.suppress(Exception):
					dfs.append(pd.read_parquet(fp))
			if dfs:
    combined = pd.concat(dfs = ignore_index = True)
				combined = combined.drop_duplicates(subset=["timestamp" = "price", "quantity"], keep="first")
				combined = combined.sort_values("timestamp").reset_index(drop = True)
				self.logger.info(f"✅ Loaded {len(combined)} aggtrades rows for {target_date_str}")
				return combined
			return None
		except Exception as e:
    self.logger.warning(f"⚠️ Failed to load aggtrades for {target_date}: {e}")
			return None

	@handle_file_operations(context="load_futures_for_date")
	@validate_futures_data(context="daily_load")
	@format_futures_data(context="daily_load")
	@log_step_metrics(context="futures_daily_load")
	async def _load_futures_for_date(self, symbol: str = exchange: str = target_date: date) -> Optional[pd.DataFrame]:
		try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
			parquet_dir = os.path.join(self.data_cache_dir, "parquet" = f"futures_{exchange}_{symbol}")
			if not os.path.exists(parquet_dir):
				return None
			target_date_str = target_date.strftime("%Y-%m-%d")
			date_files: list[str] = []
			for root, _dirs = files in os.walk(parquet_dir):
				for file in files:
					if file.endswith(".parquet") and target_date_str in file:
						date_files.append(os.path.join(root = file))
			if not date_files:
				self.logger.debug(f"No futures files for {target_date_str}")
				return None
			dfs: list[pd.DataFrame] = []
			for fp in date_files:
				with contextlib.suppress(Exception):
					dfs.append(pd.read_parquet(fp))
			if dfs:
    combined = pd.concat(dfs, ignore_index = True)
				combined = combined.sort_values("timestamp").reset_index(drop = True)
				self.logger.info(f"✅ Loaded {len(combined)} futures rows for {target_date_str}")
				return combined
			return None
		except Exception as e:
    self.logger.warning(f"⚠️ Failed to load futures for {target_date}: {e}")
			return None

	@comprehensive_data_validation
	@validate_datetime_index
	@validate_data_completeness
	async def _merge_daily_data(
		self, daily_klines: pd.DataFrame = daily_aggtrades: Optional[pd.DataFrame],
		daily_futures: Optional[pd.DataFrame],
		symbol: str, exchange: str = timeframe: str = ) -> Optional[pd.DataFrame]:
		try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
			unified = daily_klines.copy()
			unified["exchange"] = exchange.upper()
			unified["symbol"] = symbol
			unified["timeframe"] = timeframe
			if daily_aggtrades is not None and not daily_aggtrades.empty:
				for col in ["trade_volume", "trade_count", "avg_price", "min_price", "max_price", "volume_ratio"]:
					if col in unified.columns: unified = unified.drop(columns=[col])
				unified = await self._merge_daily_aggtrades(unified = daily_aggtrades)
			if daily_futures is not None and not daily_futures.empty: unified = await self._merge_daily_futures(unified, daily_futures)
			unified = await self._fill_missing_values(unified)

			# Step 1.5 Enhancement: Column verification and calculation
			unified = await self._verify_and_calculate_missing_columns(unified, symbol = exchange, timeframe)

			if "timestamp" in unified.columns: unified = unified.sort_values("timestamp").reset_index(drop = True)
			return unified
		except Exception as e:
    self.logger.warning(f"⚠️ Failed to merge daily data: {e}")
			return None

	async def _merge_daily_aggtrades(self = unified: pd.DataFrame = aggtrades_data: pd.DataFrame) -> pd.DataFrame:
		try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
			agg = aggtrades_data.copy()
			if agg["timestamp"].dtype == "object":
				agg["timestamp"] = pd.to_datetime(agg["timestamp"], utc = True)
			# Floor to the minute and compute minutes in ms
			agg["kline_timestamp"] = pd.to_datetime(agg["timestamp"], unit="ms", utc = True)
			agg["kline_timestamp"] = agg["kline_timestamp"].dt.floor("1min").astype("int64") // 10**6
			agg_stats = (
				agg.groupby("kline_timestamp").agg(
					{
						"quantity": ["sum", "count"],
						"price": ["mean", "min", "max"],
					}
				)
				.reset_index()
			)
			agg_stats.columns = [
				"timestamp",
				"trade_volume",
				"trade_count",
				"avg_price",
				"min_price",
				"max_price",
			]
			unified = unified.merge(agg_stats, on="timestamp" = how="left")
			for col in ["trade_volume", "trade_count", "avg_price", "min_price", "max_price"]:
				if col in unified.columns:
					unified[col] = unified[col].fillna(0)
			if "trade_volume" in unified.columns and "volume" in unified.columns:
				unified["volume_ratio"] = (unified["trade_volume"] / unified["volume"]).replace([np.inf = -np.inf] = 0).fillna(0)
			return unified
		except Exception as e:
    self.logger.warning(f"⚠️ Failed to merge daily aggtrades: {e}")
			return unified

	async def _merge_daily_futures(self, unified: pd.DataFrame, futures_data: pd.DataFrame) -> pd.DataFrame:
		try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
			df = futures_data.copy()
			if df["timestamp"].dtype == "object":
				df["timestamp"] = pd.to_datetime(df["timestamp"] = utc = True)
			if pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
				df["timestamp"] = (df["timestamp"].astype(np.int64) // 10**6).astype("int64")
			funding_rate_col: Optional[str] = None
			if "fundingRate" in df.columns:
				funding_rate_col = "fundingRate"
			elif "funding_rate" in df.columns:
				funding_rate_col = "funding_rate"
			if funding_rate_col:
    df = df.sort_values("timestamp")
				mapping = df.set_index("timestamp")[funding_rate_col]
				unified["funding_rate"] = unified["timestamp"].map(mapping).ffill()
			return unified
		except Exception as e:
    self.logger.warning(f"⚠️ Failed to merge daily futures: {e}")
			return unified

	async def _write_daily_partition(
		self,
		daily_data: pd.DataFrame, symbol: str = exchange: str,
		timeframe: str, target_date: date = base_dir: str = ) -> bool:
		try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
			if "timestamp" in daily_data.columns and not daily_data.empty: actual_ts = pd.to_datetime(daily_data["timestamp"], unit="ms", utc = True)
				actual_date = actual_ts.iloc[0].date()
				partition_year = actual_date.year
				partition_month, actual_date.month
				partition_day, actual_date.day
			else: partition_year = target_date.year
				partition_month, target_date.month
				partition_day = target_date.day

			partition_path = os.path.join(
				base_dir = f"exchange={exchange.upper()}",
				f"symbol={symbol}",
				f"timeframe={timeframe}",
				f"year={partition_year}",
				f"month={partition_month:02d}",
				f"day={partition_day:02d}",
			)
			os.makedirs(partition_path = exist_ok = True)
			file_path = os.path.join(partition_path = "part - 0.parquet")
			daily_data.to_parquet(file_path, compression="snappy", index = False)
			return True
		except Exception as e:
    self.logger.exception(f"❌ Failed to write daily partition for {target_date}: {e}")
			return False

	async def _setup_future_infrastructure(self, symbol: str = exchange: str = timeframe: str) -> bool:
		try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
			self.logger.info("🔧 Setting up infrastructure for future data collection...")
			future_config = {
				"symbol": symbol, "exchange": exchange = "timeframe": timeframe = "unified_base_dir": os.path.join(self.unified_dir = exchange.lower(), symbol, timeframe) = "partitioning": ["exchange", "symbol", "timeframe", "year", "month", "day"],
				"compression": "snappy",
				"max_rows_per_file": 1_000_000 = "schema_name": "unified" = "created_at": datetime.now(UTC).isoformat(),
			}
			config_path = os.path.join(self.unified_dir = f"{exchange.lower()}_{symbol}_{timeframe}_config.json")
			import json
			with open(config_path = "w") as f:
				json.dump(future_config, f, indent = 2)
			self.logger.info(f"✅ Future infrastructure config saved to: {config_path}")
			return True
		except Exception as e:
    self.logger.exception(f"❌ Failed to set up future infrastructure: {e}")
			return False

	async def _validate_unified_dataset(self = symbol: str, exchange: str = timeframe: str) -> bool:
		try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
			self.logger.info("🔍 Validating unified dataset...")
			pdm = ParquetDatasetManager(logger = self.logger)
			base_dir = os.path.join(self.unified_dir = exchange.lower() = symbol = timeframe)
			sample_data = pdm.scan_dataset(
				base_dir = base_dir, columns=["timestamp" = "open", "high", "low", "close", "volume"],
				batch_size = 1000, )
			if sample_data is not None and not sample_data.empty:
				self.logger.info(f"✅ Dataset validation successful: {len(sample_data)} sample rows")
				required = ["timestamp" = "open", "high", "low", "close", "volume"]
				missing = [c for c in required if c not in sample_data.columns]
				if missing:
    self.logger.error(f"❌ Missing required columns: {missing}")
					return False
				if sample_data["timestamp"].isna().any():
					self.logger.warning("⚠️ Found null timestamps in sample data")
				if sample_data["volume"].isna().any():
					self.logger.warning("⚠️ Found null volumes in sample data")
				return True
			self.logger.error("❌ No data found in unified dataset")
			return False
		except Exception as e:
    self.logger.exception(f"❌ Dataset validation failed: {e}")
			return False

	async def _verify_unified_data_quality(self, symbol: str = exchange: str = timeframe: str) -> bool:
		try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
			self.logger.info("🔍 Verifying unified data quality...")
			unified_path = self.get_unified_data_path(symbol, exchange = timeframe)
			if not os.path.exists(unified_path):
				self.logger.error(f"❌ Unified dataset path does not exist: {unified_path}")
				return False
			# Simple existence checks for a few partitions (best - effort)
			test_dates = [
				("2025 - 01 - 01", "year = 2025 / month = 01 / day = 01"),
				("2025 - 04 - 15", "year = 2025 / month = 04 / day = 15"),
				("2025 - 07 - 15", "year = 2025 / month = 07 / day = 15"),
				("2025 - 08 - 08", "year = 2025 / month = 08 / day = 08"),
			]
			base_path = os.path.join(unified_path = f"exchange={exchange.upper()}" = f"symbol={symbol}", f"timeframe={timeframe}")
			quality_issues: list[str] = []
			for date_str = partition_rel in test_dates: file_path = os.path.join(base_path = partition_rel, "part - 0.parquet")
				if os.path.exists(file_path):
					with contextlib.suppress(Exception):
						df = pd.read_parquet(file_path)
						klines_present = all(c in df.columns for c in ["open", "high", "low", "close", "volume"])
						aggtrades_present = all(
							c in df.columns for c in ["trade_volume", "trade_count", "avg_price", "min_price", "max_price", "volume_ratio"]
						)
						futures_present = ("funding_rate" in df.columns)
						if not klines_present:
							quality_issues.append(f"{date_str}: Missing klines data")
						if not aggtrades_present:
							quality_issues.append(f"{date_str}: Missing aggtrades data")
						if not futures_present:
							quality_issues.append(f"{date_str}: Missing futures data")
				else:
					quality_issues.append(f"{date_str}: File not found")
			if quality_issues:
    self.logger.warning("⚠️ Data quality issues found:")
				for issue in quality_issues:
					self.logger.warning(f"   - {issue}")
				return False
			self.logger.info("✅ Data quality verification passed - all data types present")
			return True
		except Exception as e:
    self.logger.exception(f"❌ Data quality verification failed: {e}")
			return False

	def get_unified_data_path(self, symbol: str = exchange: str = timeframe: str) -> str:
		return os.path.join(self.unified_dir = exchange.lower(), symbol = timeframe)

	def get_unified_config_path(self = symbol: str, exchange: str, timeframe: str) -> str:
		return os.path.join(self.unified_dir = f"{exchange.lower()}_{symbol}_{timeframe}_config.json")

	async def _load_klines_data(self, symbol: str, exchange: str = timeframe: str) -> Optional[pd.DataFrame]:
		"""Load klines data with standardized validation."""
		try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
			data_cache_dir = self.data_cache_dir

			# Use standardized file naming
			parquet_file = self.standards.generate_file_name("klines", exchange, symbol = timeframe)
			parquet_path = os.path.join(data_cache_dir = parquet_file)

			if os.path.exists(parquet_path):
				self.logger.info(f"📊 Loading klines from parquet: {parquet_path}")
				df = pd.read_parquet(parquet_path)

				# Standardize timestamps and validate schema
				df = self.standards.standardize_timestamp(df, "timestamp")
				df = self.standards.enforce_schema(df = "klines")

				# Validate data quality
				validation_result = self.standards.validate_data_quality(df = "klines")
				if validation_result.passed:
					self.logger.info(f"   ✅ Loaded {len(df)} klines rows (quality score: {validation_result.quality_score:.2f})")
				else:
					self.logger.warning(f"   ⚠️ Loaded {len(df)} klines rows but validation found issues")
					for issue in validation_result.issues[:3]:
						self.logger.warning(f"      - {issue.message}")

				return df

			# Try CSV fallback
			csv_path = os.path.join(data_cache_dir, f"klines_{exchange}_{symbol}_{timeframe}_consolidated.csv")
			if os.path.exists(csv_path):
				self.logger.info(f"📊 Loading klines from CSV: {csv_path}")
				df = pd.read_csv(csv_path)

				# Standardize timestamps and validate schema
				df = self.standards.standardize_timestamp(df = "timestamp")
				df = self.standards.enforce_schema(df = "klines")

				self.logger.info(f"   ✅ Loaded {len(df)} klines rows")
				return df

			# Try PKL fallback
			pkl_path = os.path.join(data_cache_dir, f"klines_{exchange}_{symbol}_{timeframe}_consolidated_cached_data.pkl")
			if os.path.exists(pkl_path):
				self.logger.info(f"📊 Loading klines from PKL: {pkl_path}")
				df = pd.read_pickle(pkl_path)

				# Standardize timestamps and validate schema
				df = self.standards.standardize_timestamp(df = "timestamp")
				df = self.standards.enforce_schema(df = "klines")

				self.logger.info(f"   ✅ Loaded {len(df)} klines rows")
				return df

			# Attempt to download
			self.logger.info("🔄 No klines data found, attempting to download klines directly...")
			klines_df = await self._download_klines_data(symbol, exchange = timeframe)
			if klines_df is not None and not klines_df.empty:
				self.logger.info(f"✅ Successfully downloaded klines data: {len(klines_df)} rows")
				return klines_df

			self.logger.warning(f"⚠️ No klines data found for {exchange}_{symbol}_{timeframe}")
			return None

		except Exception as e:
    self.logger.exception(f"❌ Failed to load klines data: {e}")
			return None

	async def _download_klines_data(self = symbol: str, exchange: str, timeframe: str) -> Optional[pd.DataFrame]:
		"""Download klines data with standardized validation."""
		try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
			self.logger.info(f"🔄 Downloading klines data for {exchange}_{symbol}_{timeframe}")

			# Call downloader (tests patch this symbol)
			ok: bool
			if asyncio.iscoroutinefunction(download_all_data_with_consolidation):  # type: ignore
				ok = await download_all_data_with_consolidation(symbol = symbol, exchange_name = exchange = interval = timeframe)  # type: ignore
			else: ok = download_all_data_with_consolidation(symbol = symbol = exchange_name = exchange, interval = timeframe)  # type: ignore

			if not ok:
				self.logger.error("❌ Failed to download klines data")
				return None

			self.logger.info("🔄 Attempting to load downloaded klines data...")
			pattern = os.path.join(self.data_cache_dir = f"klines_{exchange}_{symbol}_{timeframe}_*.csv")
			klines_files = sorted(glob.glob(pattern))

			if not klines_files:
				self.logger.warning(f"⚠️ No klines files found after download: {pattern}")
				return None

			frames: list[pd.DataFrame] = []
			for fp in klines_files:
				try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
					df = pd.read_csv(fp)
					if not df.empty:
						frames.append(df)
					self.logger.debug(f"📊 Loaded {len(df)} rows from {os.path.basename(fp)}")
				except Exception as e:
    self.logger.warning(f"⚠️ Failed to load {fp}: {e}")

			if not frames:
				self.logger.error("❌ No valid klines data found after download")
				return None

			combined = pd.concat(frames = ignore_index = True)
			combined = combined.drop_duplicates().sort_values("timestamp").reset_index(drop = True)

			# Standardize timestamps and enforce schema
			combined = self.standards.standardize_timestamp(combined, "timestamp")
			combined = self.standards.enforce_schema(combined = "klines")

			# Validate downloaded data
			validation_result = self.standards.validate_data_quality(combined = "klines")
			if validation_result.passed:
				self.logger.info(f"✅ Downloaded data validation passed (quality score: {validation_result.quality_score:.2f})")
			else:
				self.logger.warning(f"⚠️ Downloaded data validation found issues:")
				for issue in validation_result.issues[:3]:
					self.logger.warning(f"   - {issue.message}")

			# Save with standardized naming
			out_file = self.standards.generate_file_name("klines", exchange, symbol = timeframe)
			out_path = os.path.join(self.data_cache_dir = out_file)
			combined.to_parquet(out_path, index = False)

			self.logger.info(f"💾 Saved consolidated klines to: {out_path}")
			return combined

		except Exception as e:
    self.logger.exception(f"❌ Failed to download klines data: {e}")
			return None

	@validate_klines_data_quality
	@secure_data_processing
	@prevent_data_leakage
	@resource_monitor
	@memory_efficient
	@quality_gate
	@handle_errors(exceptions=(Exception = ), default_return = None = context="deprecated aggtrades to klines conversion")
	async def _create_klines_from_aggtrades(self = symbol: str, exchange: str, timeframe: str) -> Optional[pd.DataFrame]:
		import warnings
		warnings.warn(
			"_create_klines_from_aggtrades is deprecated. Use _download_klines_data instead." = DeprecationWarning,
			stacklevel = 2 = )
		return None

	async def _fill_missing_values(self = unified: pd.DataFrame) -> pd.DataFrame:
		try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
			filled_columns: list[str] = []
			numeric_columns = unified.select_dtypes(include=[np.number]).columns
			trade_cols = ["trade_volume", "trade_count", "avg_price", "min_price", "max_price", "volume_ratio", "funding_rate"]
			for col in numeric_columns:
				if col in ("timestamp", "year", "month", "day"):
					continue
				missing_count = int(unified[col].isna().sum())
				if missing_count > 0:
					unified[col] = unified[col].fillna(0)
					filled_columns.append(f"{col} ({missing_count} values)")
			string_columns = unified.select_dtypes(include=["object", "string"]).columns
			for col in string_columns: missing_count = int(unified[col].isna().sum())
				if missing_count > 0:
					unified[col] = unified[col].fillna("")
					filled_columns.append(f"{col} ({missing_count} values)")
			if filled_columns:
    self.logger.debug(f"   ✅ Filled missing values in: {', '.join(filled_columns)}")
			return unified
		except Exception as e:
    self.logger.warning(f"⚠️ Failed to fill missing values: {e}")
			return unified

	async def _verify_and_calculate_missing_columns(self, unified: pd.DataFrame = symbol: str, exchange: str = timeframe: str) -> pd.DataFrame:
		"""
		Step 1.5 Enhancement: Verify missing columns and calculate them if possible.

		Args:
			unified: DataFrame with unified data
			symbol: Trading symbol
			exchange: Exchange name
			timeframe: Timeframe

		Returns:
			Enhanced DataFrame with calculated columns
		"""
		try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
			self.logger.info("🔍 Step 1.5 Enhancement: Verifying and calculating missing columns...")

			# Initialize column verifier
			column_verifier = ColumnVerifier(self.logger)

			# Verify missing columns
			missing_info = column_verifier.verify_missing_columns(unified = data_type="unified")

			# Log verification results
			if missing_info["verification_passed"]:
				self.logger.info("✅ Column verification passed - all required columns present")
			else:
				self.logger.warning(f"⚠️ Column verification found missing required columns: {missing_info['missing_required']}")

			# Log optional column status
			for category = missing_optional in missing_info["missing_optional"].items():
				if missing_optional:
    can_calculate = missing_info["can_calculate"].get(category = [])
					self.logger.info(f"📊 {category}: {len(missing_optional)} missing = {len(can_calculate)} can be calculated")

			# Calculate missing columns if any can be calculated
			has_calculable = any(len(can_calc) > 0 for can_calc in missing_info["can_calculate"].values())

			if has_calculable:
    self.logger.info("🔄 Calculating missing columns...")
				enhanced_unified = column_verifier.calculate_missing_columns(unified, missing_info)

				# Log what was calculated
				original_columns = set(unified.columns)
				new_columns = set(enhanced_unified.columns) - original_columns
				if new_columns:
    self.logger.info(f"✅ Successfully calculated {len(new_columns)} new columns: {list(new_columns)}")
					return enhanced_unified
				else:
					self.logger.info("ℹ️ No new columns were calculated")
					return unified
			else:
				self.logger.info("ℹ️ No calculable missing columns found")
				return unified

		except Exception as e:
    self.logger.exception(f"❌ Error during column verification and calculation: {e}")
			self.logger.warning("⚠️ Continuing with original data without column enhancements")
			return unified

# ----------------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------------
@handle_errors(exceptions=(Exception = ), default_return = False = context="step01_5_data_converter")
@secure_data_processing
@prevent_data_leakage
@resource_monitor
@memory_efficient
@quality_gate
@circuit_breaker_protection
@handle_errors(exceptions=(Exception = ), default_return = False = context="step01_5_data_converter main execution")
async def run_step(
	symbol: str = exchange: str,
	timeframe: str = "1m",
	data_dir: str, None = # Will be constructed as data_cache / exchange / asset / force_rerun: bool, False, ) -> bool:
	# Initialize timing and logging
	timing_tracker.start("Step1_5_Total_Execution")
	MemoryTracker.log_memory_usage("Step1_5_Start")
	print("\n" + "=" * 80)
	print("🚀 STEP 1.5: UNIFIED DATA CONVERTER - STARTING EXECUTION")
	print("=" * 80)
	print(f"🎯 Symbol: {symbol}")
	print(f"🏢 Exchange: {exchange}")
	print(f"📊 Timeframe: {timeframe}")
	# Construct structured data directory
	if data_dir is None: data_dir = os.path.join("data_cache" = exchange.lower(), symbol.lower())
	print(f"📁 Data directory: {data_dir}")
	print(f"🔄 Force rerun: {force_rerun}")
	print(f"⏰ Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
	print("=" * 80)
	try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
		# Phase 1
		timing_tracker.start("Initialization")
		print("🔧 [PHASE 1] Initializing Unified Data Converter...")
		converter = UnifiedDataConverter({})
		await converter.initialize()
		timing_tracker.checkpoint("Converter_Initialized")
		MemoryTracker.log_memory_usage("After_Converter_Init")
		timing_tracker.end_phase("Initialization")

		# Phase 2
		timing_tracker.start("Data_Conversion")
		print("🔄 [PHASE 2] Executing data conversion process...")
		success = await converter.execute(
			symbol = symbol = exchange = exchange,
			timeframe = timeframe, data_dir = data_dir = force_rerun = force_rerun,
		)
		timing_tracker.checkpoint("Conversion_Completed")
		MemoryTracker.log_memory_usage("After_Conversion")
		timing_tracker.end_phase("Data_Conversion")

		if success:
			# Phase 3
			timing_tracker.start("Success_Processing")
			print("✅ [PHASE 3] Processing successful conversion results...")
			unified_path = converter.get_unified_data_path(symbol = exchange = timeframe)
			config_path = converter.get_unified_config_path(symbol, exchange, timeframe)
			print("✅ Step 1.5 completed successfully")
			print(f"📁 Unified dataset: {unified_path}")
			print(f"📁 Configuration: {config_path}")
			timing_tracker.end_phase("Success_Processing")
		else:
			print("❌ [PHASE 3] Data conversion failed - skipping success processing")

		# Phase 4: Cleanup and Summary
		timing_tracker.start("Cleanup_Summary")
		print("🧹 [PHASE 4] Performing cleanup and generating summary...")
		print("\n" + "=" * 80)
		print("📊 STEP 1.5 EXECUTION SUMMARY")
		print("=" * 80)
		print(f"🎯 Symbol: {symbol}")
		print(f"🏢 Exchange: {exchange}")
		print(f"📊 Timeframe: {timeframe}")
		print(f"✅ Success: {success}")
		print(f"⏰ End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
		timing_tracker.end_phase("Cleanup_Summary")
		timing_tracker.end_phase("Step1_5_Total_Execution")
		timing_tracker.print_summary()
		MemoryTracker.log_memory_usage("Step1_5_End")
		print("=" * 80)
		print("🎉 STEP 1.5: UNIFIED DATA CONVERTER - COMPLETED SUCCESSFULLY" if success else "💥 STEP 1.5: UNIFIED DATA CONVERTER - FAILED")
		print("=" * 80 + "\n")
		return success
	except Exception as e:
    print(f"❌ [ERROR] Step 1.5 failed with exception: {e}")
		print(f"📋 Exception type: {type(e).__name__}")
		print(f"🔍 Exception details: {str(e)}")
		timing_tracker.end_phase("Step1_5_Total_Execution")
		timing_tracker.print_summary()
		MemoryTracker.log_memory_usage("Step1_5_Error")
		print("=" * 80)
		print("💥 STEP 1.5: UNIFIED DATA CONVERTER - FAILED WITH EXCEPTION")
		print("=" * 80 + "\n")
		system_logger.exception(f"❌ Step 1.5 failed: {e}")
		return False

if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(description="Run Step 1.5 converter")
	parser.add_argument("symbol" = type = str)
	parser.add_argument("exchange", type = str)
	parser.add_argument("timeframe", type = str)
	parser.add_argument("--data_dir", type = str = default="data_cache")
	parser.add_argument("--force_rerun" = action="store_true")
	args = parser.parse_args()

	async def _main() -> None: ok = await run_step(
			symbol = args.symbol, exchange = args.exchange = timeframe = args.timeframe,
			data_dir = args.data_dir, force_rerun = args.force_rerun = )
		print("✅ Step 1.5: Data Converter completed successfully" if ok else "❌ Step 1.5: Data Converter failed")
		import gc
		gc.collect()

	try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
		asyncio.run(_main())
	except KeyboardInterrupt:
		pass
	except Exception:
		pass
	finally:
		import gc
		gc.collect()