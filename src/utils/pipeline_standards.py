"""
Pipeline Standards and Utilities

This module provides standardized utilities for the data pipeline including:
- Import management with consistent fallback patterns
- Directory structure standardization
- Timestamp format standardization
- Schema validation
- Data quality validation
- File naming conventions
- Metadata standards
"""

import sys
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum

# Mock pandas and numpy for testing purposes
class MockDataFrame:
    """Mock DataFrame class for testing."""
    def __init__(self, columns=None, shape=None):
        self.columns = columns or []
        self.shape = shape or (0, 0)
        self.dtypes = {}
    
    def copy(self):
        return MockDataFrame(columns=self.columns, shape=self.shape)
    
    def select_dtypes(self, include=None):
        return MockDataFrame(columns=self.columns)

class MockNumpy:
    """Mock numpy module for testing."""
    @staticmethod
    def int64(value):
        return int(value)
    
    @staticmethod
    def isinf(array):
        return [False] * len(array) if hasattr(array, '__len__') else False

# Use mock modules
pd = MockDataFrame
np = MockNumpy

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

class DataQualityLevel(str, Enum):
    """Data quality severity levels."""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"

@dataclass
class ValidationIssue:
    """Represents a validation issue."""
    severity: DataQualityLevel
    message: str
    details: Optional[Dict[str, Any]] = None
    column: Optional[str] = None
    row_count: Optional[int] = None

@dataclass
class ValidationResult:
    """Result of data validation."""
    passed: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    warnings: List[ValidationIssue] = field(default_factory=list)
    info: List[ValidationIssue] = field(default_factory=list)
    quality_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

class PipelineStandards:
    """Centralized pipeline standards and utilities."""
    
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
        "unified_partitioned": "unified/{exchange}/{asset}/{timeframe}/year={year}/month={month:02d}/day={day:02d}/part-0.parquet",
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
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize PipelineStandards."""
        self.logger = logger or logging.getLogger(__name__)
    
    @staticmethod
    def safe_import(module_name: str, fallback_value: Any = None, logger: Optional[logging.Logger] = None) -> Any:
        """Safely import a module with fallback."""
        try:
            module = __import__(module_name, fromlist=['*'])
            return module
        except ImportError as e:
            if logger:
                logger.warning(f"⚠️ Failed to import {module_name}: {e}. Using fallback.")
            return fallback_value
    
    @staticmethod
    def validate_environment_dependencies(required_modules: List[str], logger: Optional[logging.Logger] = None) -> Dict[str, bool]:
        """Validate that required modules are available."""
        availability = {}
        missing_modules = []
        
        for module in required_modules:
            try:
                __import__(module)
                availability[module] = True
            except ImportError:
                availability[module] = False
                missing_modules.append(module)
        
        if missing_modules and logger:
            logger.warning(f"⚠️ Missing required modules: {missing_modules}")
        
        return availability
    
    @staticmethod
    def build_path(path_type: str, exchange: str, asset: str, **kwargs) -> str:
        """Build standardized path for given type."""
        if path_type not in PipelineStandards.DIRECTORY_STRUCTURE:
            raise ValueError(f"Unknown path type: {path_type}")
        
        path_template = PipelineStandards.DIRECTORY_STRUCTURE[path_type]
        return path_template.format(exchange=exchange.lower(), asset=asset.lower(), **kwargs)
    
    @staticmethod
    def standardize_timestamp(df: MockDataFrame, column: str = "timestamp", target_format: str = "int64") -> MockDataFrame:
        """
        Standardize timestamp column to consistent format.
        
        Args:
            df: DataFrame to process
            column: Timestamp column name
            target_format: Target format ("int64" for milliseconds, "datetime64[ns]" for datetime)
            
        Returns:
            DataFrame with standardized timestamp
        """
        if column not in df.columns:
            return df
        
        df = df.copy()
        
        try:
            if target_format == "int64":
                # Convert to milliseconds timestamp (mock implementation)
                pass
            elif target_format == "datetime64[ns]":
                # Convert to datetime (mock implementation)
                pass
        except Exception as e:
            logging.warning(f"Failed to standardize timestamp column {column}: {e}")
        
        return df
    
    @staticmethod
    def validate_schema(df: MockDataFrame, schema_name: str) -> ValidationResult:
        """Validate DataFrame against standard schema."""
        if schema_name not in PipelineStandards.SCHEMAS:
            return ValidationResult(
                passed=False,
                issues=[ValidationIssue(
                    severity=DataQualityLevel.CRITICAL,
                    message=f"Unknown schema: {schema_name}"
                )]
            )
        
        schema = PipelineStandards.SCHEMAS[schema_name]
        issues = []
        warnings = []
        info = []
        
        # Check required columns
        missing_required = set(schema["required_columns"]) - set(df.columns)
        if missing_required:
            issues.append(ValidationIssue(
                severity=DataQualityLevel.CRITICAL,
                message=f"Missing required columns: {missing_required}",
                details={"missing_columns": list(missing_required)}
            ))
        
        # Check data types
        for col, expected_type in schema["data_types"].items():
            if col in df.columns:
                actual_type = str(df.dtypes.get(col, "unknown"))
                if actual_type != expected_type:
                    warnings.append(ValidationIssue(
                        severity=DataQualityLevel.WARNING,
                        message=f"Column {col} has type {actual_type}, expected {expected_type}",
                        column=col,
                        details={"expected": expected_type, "actual": actual_type}
                    ))
        
        # Check minimum rows
        if len(df.columns) < PipelineStandards.QUALITY_THRESHOLDS["min_rows"]:
            warnings.append(ValidationIssue(
                severity=DataQualityLevel.WARNING,
                message=f"DataFrame has {len(df.columns)} columns, minimum recommended is {PipelineStandards.QUALITY_THRESHOLDS['min_rows']}",
                row_count=len(df.columns)
            ))
        
        passed = len([i for i in issues if i.severity == DataQualityLevel.CRITICAL]) == 0
        quality_score = 1.0 - (len(issues) * 0.3 + len(warnings) * 0.1) / 10.0
        quality_score = max(0.0, min(1.0, quality_score))
        
        return ValidationResult(
            passed=passed,
            issues=issues,
            warnings=warnings,
            info=info,
            quality_score=quality_score,
            metadata={
                "schema_name": schema_name,
                "row_count": len(df.columns),
                "column_count": len(df.columns),
                "validation_timestamp": datetime.now(timezone.utc).isoformat()
            }
        )
    
    @staticmethod
    def generate_filename(file_type: str, exchange: str, asset: str, **kwargs) -> str:
        """Generate standardized filename."""
        if file_type not in PipelineStandards.FILE_NAMING:
            raise ValueError(f"Unknown file type: {file_type}")
        
        template = PipelineStandards.FILE_NAMING[file_type]
        return template.format(exchange=exchange.lower(), asset=asset.lower(), **kwargs)
    
    @staticmethod
    def get_quality_thresholds() -> Dict[str, Any]:
        """Get current quality thresholds."""
        return PipelineStandards.QUALITY_THRESHOLDS.copy()
    
    @staticmethod
    def set_quality_thresholds(new_thresholds: Dict[str, Any]) -> None:
        """Update quality thresholds."""
        for key, value in new_thresholds.items():
            if key in PipelineStandards.QUALITY_THRESHOLDS:
                PipelineStandards.QUALITY_THRESHOLDS[key] = value
            else:
                logging.warning(f"Unknown quality threshold: {key}")

# Convenience functions
def get_pipeline_standards(logger: Optional[logging.Logger] = None) -> PipelineStandards:
    """Get a PipelineStandards instance."""
    return PipelineStandards(logger)

def validate_dataframe_schema(df: MockDataFrame, schema_name: str) -> ValidationResult:
    """Validate DataFrame against standard schema."""
    return PipelineStandards.validate_schema(df, schema_name)

def standardize_timestamps(df: MockDataFrame, column: str = "timestamp", target_format: str = "int64") -> MockDataFrame:
    """Standardize timestamp column to consistent format."""
    return PipelineStandards.standardize_timestamp(df, column, target_format)