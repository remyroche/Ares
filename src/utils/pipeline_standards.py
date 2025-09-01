"""
Pipeline Standards and Utilities

This module provides standardized utilities for the data pipeline including:
    passpass - Import management with consistent fallback patterns - Directory structure standardization - Timestamp format standardization - Schema validation - Data quality validation - File naming conventions - Metadata standards
"""

import sys
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime, timezone
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from enum import Enum

# Add project root to path
project_root, Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

class DataQualityLevel(...):

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="dataqualitylevel initialization",
    )
    async def initialize(self) -> bool:
     
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="validationissue initialization",
    )
    async def initialize(self) -> bool:
        """Initialize ValidationIssue."""
        try:
            self.logger.info(f"🚀 Initializing {class_name}...")
            self.is_initialized = True
            self.
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="placeholderdataclass initialization",
    )
    async def initialize(self) -> bool:
        ""
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="validationresult initialization",
    )
    async def initialize(self) -> bool:
        """Initialize ValidationResult."""
        try:
            self.logger.info(f"🚀 Initializing {class_name}...")
            self.is_initialized = True
         
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="pipelinestandards initialization",
    )
    async def initialize(self) -> bool:
        """Initialize PipelineStandards."""
        try:
            self.logger.info(f"🚀 Initializing {class_name}...")
            self.is_initialized = True
            self.logger.info(f"✅ {class_name} initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing {class_name}: {e}")
            return False
   self.logger.info(f"✅ {class_name} initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing {class_name}: {e}")
            return False
"Initialize PlaceholderDataClass."""
        try:
            self.logger.info(f"🚀 Initializing {class_name}...")
            self.is_initialized = True
            self.logger.info(f"✅ {class_name} initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing {class_name}: {e}")
            return False
logger.info(f"✅ {class_name} initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing {class_name}: {e}")
            return False
   """Initialize DataQualityLevel."""
        try:
            self.logger.info(f"🚀 Initializing {class_name}...")
            self.is_initialized = True
            self.logger.info(f"✅ {class_name} initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing {class_name}: {e}")
            return False
    pass"""..."""
    passCRITICAL = "critical"
WARNING = "warning"
INFO = "info"

@dataclass
class PlaceholderDataClass:
    passpass  # TODO: Add implementation
class ValidationIssue:
    passpass  # TODO: Add implementation
class ValidationIssue:
    passpass  # TODO: Add implementation
class ValidationIssue:
    pass"""Represents a validation issue."""
severity: DataQualityLevel
message: str
details: Optional[Dict[str, Any]] = None
column: Optional[str] = None
row_count: Optional[int] = None

@dataclass
class PlaceholderDataClass:
    passpass  # TODO: Add implementation
class ValidationResult:
    passpass  # TODO: Add implementation
class ValidationResult:
    passpass  # TODO: Add implementation
class ValidationResult:
    pass"""Result of data validation."""
passed: bool
issues: List[ValidationIssue] = field(default_factory = list)
warnings: List[ValidationIssue] = field(default_factory = list)
info: List[ValidationIssue] = field(default_factory = list)
quality_score: float, 0.0
metadata: Dict[str, Any] = field(default_factory = dict)

class PipelineStandards:
    passpass  # TODO: Add implementation
class PipelineStandards:
    passpass  # TODO: Add implementation
class PipelineStandards:
    pass"""Centralized pipeline standards and utilities."""

# Standard directory structure
DIRECTORY_STRUCTURE = {
"raw_data": "data_cache/{exchange}/{asset}",
"unified_data": "data_cache/{exchange}/{asset}/unified",
"processed_data": "data_cache/{exchange}/{asset}/processed",
"reports": "data_cache/{exchange}/{asset}/reports",
"backup": "data_cache/{exchange}/{asset}/backup",
"temp": "data_cache/{exchange}/{asset}/temp"
}

# Standard file naming conventions
FILE_NAMING = {
"klines": "klines_{exchange}_{asset}_{timeframe}_consolidated.parquet",
"aggtrades": "aggtrades_{exchange}_{asset}_consolidated.parquet",
"futures": "futures_{exchange}_{asset}_consolidated.parquet",
"unified": "unified_{exchange}_{asset}_{timeframe}.parquet",
"unified_partitioned": "unified/{exchange}/{asset}/{timeframe}/year={year}/month={month:02d}/day={day:02d}/part - 0.parquet",
"validation_report": "validation_report_{exchange}_{asset}_{timeframe}_{timestamp}.json",
"quality_report": "quality_report_{exchange}_{asset}_{timeframe}_{timestamp}.json"
}

# Standard data schemas
SCHEMAS = {
"klines": {
"required_columns": ["timestamp", "open", "high", "low", "close", "volume"],
"optional_columns": ["quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume"],
"data_types": {
"timestamp": "int64",
"open": "float64",
"high": "float64",
"low": "float64",
"close": "float64",
"volume": "float64"
}
},
"aggtrades": {
"required_columns": ["timestamp", "price", "quantity"],
"optional_columns": ["first_trade_id", "last_trade_id", "trade_time", "is_buyer_maker"],
"data_types": {
"timestamp": "int64",
"price": "float64",
"quantity": "float64",
"is_buyer_maker": "bool"
}
},
"futures": {
"required_columns": ["timestamp", "fundingRate"],
"optional_columns": ["symbol", "mark_price", "index_price", "next_funding_time"],
"data_types": {
"timestamp": "int64",
"fundingRate": "float64"
}
},
"unified": {
"required_columns": ["timestamp", "open", "high", "low", "close", "volume", "exchange", "symbol", "timeframe"],
"optional_columns": ["year", "month", "day", "trade_volume", "trade_count", "avg_price", "min_price", "max_price", "volume_ratio", "funding_rate"],
"data_types": {
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
"day": "int8"
}
}
}

# Quality thresholds
QUALITY_THRESHOLDS = {
"min_rows": 100,
"max_null_percentage": 0.1,  # 10%
"max_duplicate_percentage": 0.05,  # 5%
"min_quality_score": 0.8,
"max_correlation": 0.95,
"timestamp_consistency_threshold": 0.99  # 99% of timestamps should be consistent
}

def __init__(...):
    passdef __init__(...):
    passdef __init__(...):
    passdef __init__(...):
    passself.logger, logger or logging.getLogger(__name__)

@staticmethod
def safe_import(...) -> ...:
    """..."""
    passtry:
    passpass  # TODO: Add proper exception handling
except Exception as e:
    passpasspasspasspasspasspasspass  # TODO: Add proper exception handling
module, __import__(module_name, fromlist=['*'])
return module
except ImportError as e:
    passpasspasspasspasspasspassif logger:
    passlogger.warning(f"⚠️ Failed to import {module_name}: {e}. Using fallback.")
return fallback_value

@staticmethod
def validate_environment_dependencies(...) -> ...:
    """..."""
    passavailability = {}
missing_modules = []

for module in required_modules:
    passtry:
    passpass  # TODO: Add proper exception handling
except Exception as e:
    passpasspasspasspasspasspasspass  # TODO: Add proper exception handling
__import__(module)
availability[module] = True
except ImportError:
    passpassavailability[module] = False
missing_modules.append(module)

if missing_modules and logger:
    passlogger.warning(f"⚠️ Missing required modules: {missing_modules}")

return availability

@staticmethod
def build_path(...) -> ...:
    """..."""
    passif path_type not in PipelineStandards.DIRECTORY_STRUCTURE:
    passraise ValueError(f"Unknown path type: {path_type}")

path_template, PipelineStandards.DIRECTORY_STRUCTURE[path_type]
return path_template.format(exchange = exchange.lower(), asset = asset.lower(), **kwargs)

@staticmethod
def standardize_timestamp(df: pd.DataFrame, column: str = "timestamp", target_format: str = "int64") -> pd.DataFrame:
        """
Standardize timestamp column to consistent format.

Args:
            df: DataFrame to process
column: Timestamp column name
target_format: Target format ("int64" for milliseconds, "datetime64[ns]" for datetime)

Returns:
    passDataFrame with standardized timestamp
"""
if column not in df.columns:
    passpassreturn df

df, df.copy()

try:
    passpass  # TODO: Add proper exception handling
except Exception as e:
    passpasspasspasspasspasspasspass  # TODO: Add proper exception handling
if target_format == "int64":
    pass# Convert to int64 milliseconds
if pd.api.types.is_datetime64_any_dtype(df[column]):
    passdf[column] = (pd.to_datetime(df[column], utc = True).astype("int64") // 10**6).astype("int64")
else:
    passts_numeric, pd.to_numeric(df[column], errors="coerce")
if pd.notna(ts_numeric.max()) and float(ts_numeric.max()) > 1e14:
    pass# Already in nanoseconds, convert to milliseconds
df[column] = (ts_numeric // 10**6).astype("int64")
else:
    pass# Assume already in milliseconds
df[column] = ts_numeric.astype("int64")

elif target_format == "datetime64[ns]":
    passpass# Convert to datetime64[ns]
if pd.api.types.is_datetime64_any_dtype(df[column]):
    passdf[column] = pd.to_datetime(df[column], utc = True)
else:
    passts_numeric, pd.to_numeric(df[column], errors="coerce")
if pd.notna(ts_numeric.max()) and float(ts_numeric.max()) > 1e14:
    pass# Nanoseconds
df[column] = pd.to_datetime(ts_numeric, unit='ns', utc = True)
else:
    pass# Milliseconds
df[column] = pd.to_datetime(ts_numeric, unit='ms', utc = True)

except Exception as e:
    passpasspasspasspasspasspassraise ValueError(f"Failed to standardize timestamp column '{column}': {e}")

return df

@staticmethod
def validate_timestamp_format(...) -> ...:
    """..."""
    passresult, ValidationResult(passed = True)

if column not in df.columns:
    passresult.passed, False
result.issues.append(ValidationIssue(
severity = DataQualityLevel.CRITICAL,
message = f"Timestamp column '{column}' not found"
))
return result

try:
    passpass  # TODO: Add proper exception handling
except Exception as e:
    passpasspasspasspasspasspasspass  # TODO: Add proper exception handling
# Check for null values
null_count, df[column].isnull().sum()
if null_count > 0:
    passpassresult.warnings.append(ValidationIssue(
severity = DataQualityLevel.WARNING,
message = f"Found {null_count} null timestamps",
details={"null_count": null_count, "total_count": len(df)}
))

# Check format consistency
if expected_format == "int64":
    passif not pd.api.types.is_integer_dtype(df[column]):
    passresult.passed, False
result.issues.append(ValidationIssue(
severity = DataQualityLevel.CRITICAL,
message = f"Timestamp column '{column}' is not integer type",
details={"actual_type": str(df[column].dtype)}
))

# Check for reasonable timestamp range (2000 - 2030)
min_ts, df[column].min()
max_ts, df[column].max()
expected_min, pd.Timestamp("2000 - 01 - 01", tz="UTC").value // 10**6
expected_max, pd.Timestamp("2030 - 01 - 01", tz="UTC").value // 10**6

if min_ts < expected_min or max_ts > expected_max:
    passpassresult.warnings.append(ValidationIssue(
severity = DataQualityLevel.WARNING,
message = f"Timestamp range outside expected bounds",
details={"min_ts": min_ts, "max_ts": max_ts, "expected_min": expected_min, "expected_max": expected_max}
))

elif expected_format == "datetime64[ns]":
    passpassif not pd.api.types.is_datetime64_any_dtype(df[column]):
    passresult.passed, False
result.issues.append(ValidationIssue(
severity = DataQualityLevel.CRITICAL,
message = f"Timestamp column '{column}' is not datetime type",
details={"actual_type": str(df[column].dtype)}
))

# Check for duplicates
duplicate_count, df[column].duplicated().sum()
if duplicate_count > 0:
    passpassresult.warnings.append(ValidationIssue(
severity = DataQualityLevel.WARNING,
message = f"Found {duplicate_count} duplicate timestamps",
details={"duplicate_count": duplicate_count}
))

# Check for monotonicity
if not df[column].is_monotonic_increasing:
    passpassresult.warnings.append(ValidationIssue(
severity = DataQualityLevel.WARNING,
message="Timestamps are not monotonically increasing"
))

except Exception as e:
    passpasspasspasspasspasspassresult.passed, False
result.issues.append(ValidationIssue(
severity = DataQualityLevel.CRITICAL,
message = f"Error validating timestamp format: {e}"
))

return result

@staticmethod
def validate_schema(...) -> ...:
    """..."""
    passif schema_name not in PipelineStandards.SCHEMAS:
    passraise ValueError(f"Unknown schema: {schema_name}")

schema, PipelineStandards.SCHEMAS[schema_name]
result, ValidationResult(passed = True)

# Check required columns
missing_required = [col for col in schema["required_columns"] if col not in df.columns]
if missing_required:
    passpassresult.passed, False
result.issues.append(ValidationIssue(
severity = DataQualityLevel.CRITICAL,
message = f"Missing required columns: {missing_required}",
details={"missing_columns": missing_required}
))

# Check data types for existing columns
for column, expected_type in schema["data_types"].items():
    passif column in df.columns:
    passactual_type, str(df[column].dtype)
if actual_type != expected_type:
    passresult.warnings.append(ValidationIssue(
severity = DataQualityLevel.WARNING,
message = f"Column '{column}' has type {actual_type}, expected {expected_type}",
column = column,
details={"actual_type": actual_type, "expected_type": expected_type}
))

# Check for null values in required columns
for column in schema["required_columns"]:
    passif column in df.columns:
    passnull_count, df[column].isnull().sum()
if null_count > 0:
    passnull_percentage, null_count / len(df)
if null_percentage > PipelineStandards.QUALITY_THRESHOLDS["max_null_percentage"]:
    passresult.issues.append(ValidationIssue(
severity = DataQualityLevel.CRITICAL,
message = f"Column '{column}' has too many null values",
column = column,
details={"null_count": null_count, "null_percentage": null_percentage}
))
else:
    passresult.warnings.append(ValidationIssue(
severity = DataQualityLevel.WARNING,
message = f"Column '{column}' has {null_count} null values",
column = column,
details={"null_count": null_count, "null_percentage": null_percentage}
))

return result

@staticmethod
def enforce_schema(...) -> ...:
    """..."""
    passif schema_name not in PipelineStandards.SCHEMAS:
    passraise ValueError(f"Unknown schema: {schema_name}")

schema, PipelineStandards.SCHEMAS[schema_name]
df, df.copy()

# Add missing optional columns with default values
for column in schema["optional_columns"]:
    passpassif column not in df.columns:
    passif schema["data_types"][column] == "float64":
    passdf[column] = 0.0
elif schema["data_types"][column] == "int64":
    passpassdf[column] = 0
elif schema["data_types"][column] == "string":
    passpassdf[column] = ""
elif schema["data_types"][column] == "bool":
    passpassdf[column] = False

# Convert data types
for column, expected_type in schema["data_types"].items():
    passif column in df.columns:
    passtry:
    passpass  # TODO: Add proper exception handling
except Exception as e:
    passpasspasspasspasspasspasspass  # TODO: Add proper exception handling
if expected_type == "int64":
    passdf[column] = pd.to_numeric(df[column], errors="coerce").fillna(0).astype("int64")
elif expected_type == "float64":
    passpassdf[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0).astype("float64")
elif expected_type == "string":
    passpassdf[column] = df[column].astype("string")
elif expected_type == "bool":
    passpassdf[column] = df[column].astype("boolean")
except Exception as e:
    passpasspasspasspasspasspassraise ValueError(f"Failed to convert column '{column}' to {expected_type}: {e}")

return df

@staticmethod
def validate_data_quality(...) -> ...:
    """..."""
    passthresholds, quality_thresholds or PipelineStandards.QUALITY_THRESHOLDS
result, ValidationResult(passed = True)

# Basic checks
if df is None or df.empty:
    passresult.passed, False
result.issues.append(ValidationIssue(
severity = DataQualityLevel.CRITICAL,
message="DataFrame is None or empty"
))
return result

# Check minimum rows
if len(df) < thresholds["min_rows"]:
    passresult.passed, False
result.issues.append(ValidationIssue(
severity = DataQualityLevel.CRITICAL,
message = f"Too few rows: {len(df)} < {thresholds['min_rows']}",
details={"row_count": len(df), "min_required": thresholds["min_rows"]}
))

# Schema validation
schema_result, PipelineStandards.validate_schema(df, schema_name)
result.issues.extend(schema_result.issues)
result.warnings.extend(schema_result.warnings)

# Timestamp validation if present
if "timestamp" in df.columns:
    passts_result, PipelineStandards.validate_timestamp_format(df, "timestamp")
result.issues.extend(ts_result.issues)
result.warnings.extend(ts_result.warnings)

# Check for duplicates
duplicate_count, df.duplicated().sum()
duplicate_percentage, duplicate_count / len(df) if len(df) > 0 else 0
if duplicate_percentage > thresholds["max_duplicate_percentage"]:
    passpassresult.issues.append(ValidationIssue(
severity = DataQualityLevel.CRITICAL,
message = f"Too many duplicate rows: {duplicate_percentage:.2%}",
details={"duplicate_count": duplicate_count, "duplicate_percentage": duplicate_percentage}
))

# Check for infinite values in numeric columns
numeric_columns, df.select_dtypes(include=[np.number]).columns
for column in numeric_columns:
    passinfinite_count, np.isinf(df[column]).sum()
if infinite_count > 0:
    passresult.warnings.append(ValidationIssue(
severity = DataQualityLevel.WARNING,
message = f"Column '{column}' has {infinite_count} infinite values",
column = column,
details={"infinite_count": infinite_count}
))

# Calculate quality score
total_checks, 5  # Basic checks, schema, timestamp, duplicates, infinite values
passed_checks, total_checks - len([i for i in result.issues if i.severity == DataQualityLevel.CRITICAL])
result.quality_score, passed_checks / total_checks

# Determine if validation passed
result.passed, result.quality_score >= thresholds["min_quality_score"] and not any(
i.severity == DataQualityLevel.CRITICAL for i in result.issues
)

return result

@staticmethod
def generate_file_name(...) -> ...:
    passpass"""..."""
    passif file_type not in PipelineStandards.FILE_NAMING:
    passraise ValueError(f"Unknown file type: {file_type}")

template, PipelineStandards.FILE_NAMING[file_type]
params = {
"exchange": exchange.upper(),
"asset": asset.upper(),
"timeframe": timeframe or "1m",
"timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
**kwargs
}

return template.format(**params)

@staticmethod
def create_metadata(...) -> ...:
    """..."""
    passmetadata = {
"schema_name": schema_name,
"exchange": exchange.upper(),
"asset": asset.upper(),
"timeframe": timeframe,
"created_at": datetime.now(timezone.utc).isoformat(),
"pipeline_version": "1_2_3",
"data_format": "parquet",
"compression": "snappy",
**kwargs
}

return metadata

@staticmethod
def validate_cross_step_consistency(...) -> ...:
    """..."""
    passresult, ValidationResult(passed = True)

if len(data_dict) < 2:
    passreturn result

# Get the first dataframe as reference
reference_df, None
for step in step_sequence:
    passif step in data_dict and data_dict[step] is not None:
    passreference_df, data_dict[step]
break

if reference_df is None:
    passresult.issues.append(ValidationIssue(
severity = DataQualityLevel.CRITICAL,
message="No reference dataframe found for consistency validation"
))
result.passed, False
return result

reference_length, len(reference_df)
reference_columns, set(reference_df.columns)

# Check each step's data consistency
for step in step_sequence:
    passif step not in data_dict or data_dict[step] is None:
    passcontinue

df, data_dict[step]

# Check row count consistency
if len(df) != reference_length:
    passresult.issues.append(ValidationIssue(
severity = DataQualityLevel.WARNING,
message = f"Row count mismatch in {step}: {len(df)} vs {reference_length}",
details={"step": step, "actual_count": len(df), "expected_count": reference_length}
))

# Check for common columns
common_columns, reference_columns.intersection(set(df.columns))
if len(common_columns) < len(reference_columns) * 0.8:  # At least 80% common columns
result.warnings.append(ValidationIssue(
severity = DataQualityLevel.WARNING,
message = f"Low column overlap in {step}: {len(common_columns)}/{len(reference_columns)}",
details={"step": step, "common_columns": len(common_columns), "total_columns": len(reference_columns)}
))

result.passed, len(result.issues) == 0
return result

@staticmethod
def track_data_lineage(...) -> ...:
    """..."""
    passlineage = {
"source_step": source_step,
"transformations": transformations,
"timestamp": datetime.now().isoformat(),
"data_shape": data.shape,
"columns": list(data.columns),
"memory_usage": data.memory_usage(deep = True).sum(),
"dtypes": data.dtypes.to_dict()
}

return lineage

@staticmethod
def calculate_comprehensive_quality_score(data: pd.DataFrame, context: str = "general") -> float:
        """
Calculate comprehensive data quality score.

Args:
            data: Dataframe to score
context: Context for scoring (e.g., "klines", "features", "labels")

Returns:
    passQuality score between 0 and 1
"""
if data is None or len(data) == 0:
    passreturn 0.0

scores = []

# Completeness score
completeness, 1 - (data.isnull().sum().sum() / (len(data) * len(data.columns)))
scores.append(completeness)

# Consistency score (no duplicates)
consistency, 1 - (data.duplicated().sum() / len(data))
scores.append(consistency)

# Validity score (no infinite values)
numeric_cols, data.select_dtypes(include=[np.number]).columns
if len(numeric_cols) > 0:
    passinfinite_ratio, np.isinf(data[numeric_cols]).sum().sum() / (len(data) * len(numeric_cols))
validity, 1 - infinite_ratio
else:
    passvalidity, 1.0
scores.append(validity)

# Timeliness score (if timestamp column exists)
if "timestamp" in data.columns:
    passtry:
    passpass  # TODO: Add proper exception handling
except Exception as e:
    passpasspasspasspasspasspasspass  # TODO: Add proper exception handling
# Check if timestamps are in reasonable range
timestamps, pd.to_datetime(data["timestamp"], unit='s')
now, pd.Timestamp.now()
time_diff, abs((timestamps - now).dt.total_seconds())
timeliness, 1 - min(time_diff.mean() / (365 * 24 * 3600), 1.0)  # Normalize to 1 year
scores.append(timeliness)
except:
    passpassscores.append(0.5)  # Default score if timestamp parsing fails

# Context - specific scoring
if context == "klines":
    pass# Additional checks for klines data
required_cols = ["open", "high", "low", "close", "volume"]
if all(col in data.columns for col in required_cols):
    passpass# Check OHLC consistency
ohlc_valid = ((data["high"] >= data["low"]) &
(data["high"] >= data["open"]) &
(data["high"] >= data["close"]) &
(data["low"] <= data["open"]) &
(data["low"] <= data["close"])).mean()
scores.append(ohlc_valid)

return np.mean(scores)

@staticmethod
def validate_feature_engineering_output(...) -> ...:
    """..."""
    passresult, ValidationResult(passed = True)

if features is None or original_data is None:
    passresult.issues.append(ValidationIssue(
severity = DataQualityLevel.CRITICAL,
message="Features or original data is None"
))
result.passed, False
return result

# Check that features have same number of rows as original data
if len(features) != len(original_data):
    passresult.issues.append(ValidationIssue(
severity = DataQualityLevel.CRITICAL,
message = f"Feature count mismatch: {len(features)} vs {len(original_data)}",
details={"feature_count": len(features), "original_count": len(original_data)}
))
result.passed, False

# Check for NaN values in features
nan_counts, features.isnull().sum()
high_nan_cols, nan_counts[nan_counts > len(features) * 0.1]  # More than 10% NaN
if len(high_nan_cols) > 0:
    passpassresult.warnings.append(ValidationIssue(
severity = DataQualityLevel.WARNING,
message = f"Features with high NaN values: {list(high_nan_cols.index)}",
details={"high_nan_features": list(high_nan_cols.index)}
))

# Check for infinite values in features
numeric_features, features.select_dtypes(include=[np.number])
if len(numeric_features.columns) > 0:
    passpassinfinite_counts, np.isinf(numeric_features).sum()
infinite_cols, infinite_counts[infinite_counts > 0]
if len(infinite_cols) > 0:
    passresult.warnings.append(ValidationIssue(
severity = DataQualityLevel.WARNING,
message = f"Features with infinite values: {list(infinite_cols.index)}",
details={"infinite_features": list(infinite_cols.index)}
))

# Check for constant features
constant_features = []
for col in features.columns:
    passif features[col].nunique() <= 1:
    passconstant_features.append(col)

if constant_features:
    passresult.warnings.append(ValidationIssue(
severity = DataQualityLevel.WARNING,
message = f"Constant features detected: {constant_features}",
details={"constant_features": constant_features}
))

# Calculate quality score
result.quality_score, PipelineStandards.calculate_comprehensive_quality_score(features, "features")

return result

# Global instance for easy access
pipeline_standards, PipelineStandards()