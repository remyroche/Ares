"""
Data Validator

Comprehensive data validation and quality checks for trading data.
Ensures data integrity, consistency, and quality before processing.
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd
import numpy as np

from src.utils.logger import system_logger
from src.core.decorators import handles_errors, traced, log_execution_time
from src.utils.tprint import (
    tprint_info, tprint_warning, tprint_error, tprint_success,
    tprint_structured, LogLevel
)
from ..utils.error_handling import (
    TradingError, TradingErrorSeverity, trading_error_handler,
    critical_operation, require_no_fallback
)

logger = system_logger.getChild('DataValidator')

class DataQualityLevel(Enum):
    """Data quality levels."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    CRITICAL = "critical"

class ValidationRule(Enum):
    """Validation rule types."""
    OHLC_CONSISTENCY = "ohlc_consistency"
    MISSING_DATA = "missing_data"
    EXTREME_VALUES = "extreme_values"
    VOLUME_SPIKES = "volume_spikes"
    PRICE_GAPS = "price_gaps"
    TIMESTAMP_ORDER = "timestamp_order"
    DUPLICATE_DATA = "duplicate_data"
    DATA_FRESHNESS = "data_freshness"

@dataclass
class ValidationResult:
    """Result of data validation."""
    is_valid: bool
    quality_score: float
    failed_rules: List[ValidationRule]
    warnings: List[str]
    errors: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DataQualityReport:
    """Comprehensive data quality report."""
    symbol: str
    timestamp: datetime
    overall_quality: DataQualityLevel
    validation_results: Dict[ValidationRule, ValidationResult]
    recommendations: List[str]
    data_stats: Dict[str, Any]

class DataValidator:
    """
    Data Validator for Trading Data

    Performs comprehensive validation and quality checks on market data
    including OHLC consistency, missing data detection, outlier detection,
    and data freshness validation.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize data validator.

        Args:
            config: Configuration dictionary
        """
        tprint_info("🔍 Initializing Data Validator...")
        self.config: Dict[str, Any] = config
        self.logger = logger.getChild('DataValidator')

        # Validation thresholds
        self.price_tolerance: float = config.get('price_tolerance', 0.1)  # 10%
        self.volume_tolerance: float = config.get('volume_tolerance', 5.0)  # 500%
        self.missing_data_threshold: float = config.get('missing_data_threshold', 0.05)  # 5%
        self.outlier_threshold: float = config.get('outlier_threshold', 2.5)  # 2.5 std devs (more appropriate for financial data)
        self.freshness_threshold_minutes: int = config.get('freshness_threshold_minutes', 5)

        # Quality scoring weights
        self.quality_weights: Dict[ValidationRule, float] = {
            ValidationRule.OHLC_CONSISTENCY: 0.25,
            ValidationRule.MISSING_DATA: 0.20,
            ValidationRule.EXTREME_VALUES: 0.20,
            ValidationRule.VOLUME_SPIKES: 0.15,
            ValidationRule.PRICE_GAPS: 0.10,
            ValidationRule.TIMESTAMP_ORDER: 0.05,
            ValidationRule.DUPLICATE_DATA: 0.03,
            ValidationRule.DATA_FRESHNESS: 0.02
        }

        # Historical data for validation
        self.price_history: Dict[str, List[float]] = {}
        self.volume_history: Dict[str, List[float]] = {}

    async def initialize(self) -> None:
        """Initialize data validator."""
        tprint_success("✅ Data Validator initialized successfully")

    @handles_errors
    async def validate_market_data(
        self,
        data: Union[pd.DataFrame, Dict[str, Any]],
        symbol: str
    ) -> ValidationResult:
        """
        Validate market data.

        Args:
            data: Market data to validate
            symbol: Trading symbol

        Returns:
            Validation result
        """
        try:
            # Convert to DataFrame if needed
            if isinstance(data, dict):
                df = pd.DataFrame([data])
            else:
                df = data.copy()

            failed_rules = []
            warnings = []
            errors = []

            # Run all validation checks
            validation_results = {}

            # 1. OHLC Consistency Check
            ohlc_result = await self._validate_ohlc_consistency(df, symbol)
            validation_results[ValidationRule.OHLC_CONSISTENCY] = ohlc_result
            if not ohlc_result.is_valid:
                failed_rules.append(ValidationRule.OHLC_CONSISTENCY)
                errors.extend(ohlc_result.errors)

            # 2. Missing Data Check
            missing_result = await self._validate_missing_data(df)
            validation_results[ValidationRule.MISSING_DATA] = missing_result
            if not missing_result.is_valid:
                failed_rules.append(ValidationRule.MISSING_DATA)
                errors.extend(missing_result.errors)

            # 3. Extreme Values Check
            extreme_result = await self._validate_extreme_values(df, symbol)
            validation_results[ValidationRule.EXTREME_VALUES] = extreme_result
            if not extreme_result.is_valid:
                failed_rules.append(ValidationRule.EXTREME_VALUES)
                warnings.extend(extreme_result.warnings)

            # 4. Volume Spikes Check
            volume_result = await self._validate_volume_spikes(df, symbol)
            validation_results[ValidationRule.VOLUME_SPIKES] = volume_result
            if not volume_result.is_valid:
                failed_rules.append(ValidationRule.VOLUME_SPIKES)
                warnings.extend(volume_result.warnings)

            # 5. Price Gaps Check
            gap_result = await self._validate_price_gaps(df, symbol)
            validation_results[ValidationRule.PRICE_GAPS] = gap_result
            if not gap_result.is_valid:
                failed_rules.append(ValidationRule.PRICE_GAPS)
                warnings.extend(gap_result.warnings)

            # 6. Timestamp Order Check
            timestamp_result = await self._validate_timestamp_order(df)
            validation_results[ValidationRule.TIMESTAMP_ORDER] = timestamp_result
            if not timestamp_result.is_valid:
                failed_rules.append(ValidationRule.TIMESTAMP_ORDER)
                errors.extend(timestamp_result.errors)

            # 7. Duplicate Data Check
            duplicate_result = await self._validate_duplicate_data(df)
            validation_results[ValidationRule.DUPLICATE_DATA] = duplicate_result
            if not duplicate_result.is_valid:
                failed_rules.append(ValidationRule.DUPLICATE_DATA)
                warnings.extend(duplicate_result.warnings)

            # 8. Data Freshness Check
            freshness_result = await self._validate_data_freshness(df)
            validation_results[ValidationRule.DATA_FRESHNESS] = freshness_result
            if not freshness_result.is_valid:
                failed_rules.append(ValidationRule.DATA_FRESHNESS)
                warnings.extend(freshness_result.warnings)

            # Calculate overall quality score
            quality_score = await self._calculate_quality_score(validation_results)

            # Determine if data is valid (critical errors)
            # Check if any critical validation rules failed
            critical_rules = {ValidationRule.OHLC_CONSISTENCY, ValidationRule.MISSING_DATA}
            failed_critical = any(
                critical_rule in r.failed_rules 
                for r in validation_results.values() 
                if not r.is_valid
                for critical_rule in critical_rules
            )
            is_valid = not failed_critical

            return ValidationResult(
                is_valid=is_valid,
                quality_score=quality_score,
                failed_rules=failed_rules,
                warnings=warnings,
                errors=errors,
                metadata={'validation_results': validation_results}
            )

        except Exception as e:
            tprint_error(f"❌ Error validating market data: {str(e)}")
            return ValidationResult(
                is_valid=False,
                quality_score=0.0,
                failed_rules=[ValidationRule.MISSING_DATA],
                warnings=[],
                errors=[str(e)]
            )

    async def _validate_ohlc_consistency(self, df: pd.DataFrame, symbol: str) -> ValidationResult:
        """Validate OHLC data consistency."""
        errors: List[str] = []
        warnings: List[str] = []

        required_columns: List[str] = ['open', 'high', 'low', 'close']
        missing_columns: List[str] = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            error_msg: str = f"Missing required columns: {missing_columns}"
            errors.append(error_msg)
            tprint_error(f"❌ OHLC validation failed for {symbol}: {error_msg}")

        if not errors and len(df) > 0:
            # Check for logical OHLC relationships
            ohlc_invalid = df[
                (df['high'] < df['low']) |
                (df['high'] < df['open']) |
                (df['high'] < df['close']) |
                (df['low'] > df['open']) |
                (df['low'] > df['close'])
            ]

            if not ohlc_invalid.empty:
                errors.append(f"Invalid OHLC relationships in {len(ohlc_invalid)} rows")

            # Check for extreme price movements
            if 'close' in df.columns and len(df) > 1:
                price_changes = df['close'].pct_change().abs()
                extreme_changes = price_changes[price_changes > self.price_tolerance]

                if not extreme_changes.empty:
                    warnings.append(f"Extreme price changes detected: {len(extreme_changes)} occurrences")

        return ValidationResult(
            is_valid=len(errors) == 0,
            quality_score=1.0 if len(errors) == 0 else 0.5,
            failed_rules=[],
            warnings=warnings,
            errors=errors
        )

    async def _validate_missing_data(self, df: pd.DataFrame) -> ValidationResult:
        """Validate for missing data."""
        errors: List[str] = []
        warnings: List[str] = []

        if df.empty:
            error_msg: str = "DataFrame is empty"
            errors.append(error_msg)
            tprint_error(f"❌ Missing data validation failed: {error_msg}")
            return ValidationResult(
                is_valid=False,
                quality_score=0.0,
                failed_rules=[ValidationRule.MISSING_DATA],
                warnings=warnings,
                errors=errors
            )

        # Check for missing values
        missing_counts = df.isnull().sum()
        total_cells: int = df.size
        missing_percentage: float = missing_counts.sum() / total_cells

        if missing_percentage > self.missing_data_threshold:
            error_msg = f"Missing data percentage ({missing_percentage:.2%}) exceeds threshold ({self.missing_data_threshold:.2%})"
            errors.append(error_msg)
            tprint_error(f"❌ Missing data validation failed: {error_msg}")

        if missing_percentage > 0:
            warning_msg: str = f"Missing data in {missing_counts[missing_counts > 0].to_dict()}"
            warnings.append(warning_msg)
            tprint_warning(f"⚠️ Missing data detected: {warning_msg}")

        return ValidationResult(
            is_valid=len(errors) == 0,
            quality_score=max(0.0, 1.0 - missing_percentage * 2),
            failed_rules=[ValidationRule.MISSING_DATA] if errors else [],
            warnings=warnings,
            errors=errors
        )

    async def _validate_extreme_values(self, df: pd.DataFrame, symbol: str) -> ValidationResult:
        """Validate for extreme values."""
        warnings: List[str] = []
        errors: List[str] = []

        if df.empty:
            return ValidationResult(
                is_valid=True,
                quality_score=1.0,
                failed_rules=[],
                warnings=warnings,
                errors=errors
            )

        # Check for extreme price values
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                if not df[col].empty:
                    # Use IQR method for outlier detection
                    Q1: float = df[col].quantile(0.25)
                    Q3: float = df[col].quantile(0.75)
                    IQR: float = Q3 - Q1
                    lower_bound: float = Q1 - (self.outlier_threshold * IQR)
                    upper_bound: float = Q3 + (self.outlier_threshold * IQR)

                    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

                    if not outliers.empty:
                        warning_msg: str = f"Outliers detected in {col}: {len(outliers)} values"
                        warnings.append(warning_msg)
                        tprint_warning(f"⚠️ Extreme values validation warning for {symbol}: {warning_msg}")

        # Check for zero or negative prices
        price_columns: List[str] = [col for col in ['open', 'high', 'low', 'close'] if col in df.columns]
        if price_columns:
            negative_prices = df[(df[price_columns] <= 0).any(axis=1)]
            if not negative_prices.empty:
                error_msg: str = f"Zero or negative prices detected: {len(negative_prices)} rows"
                errors.append(error_msg)
                tprint_error(f"❌ Extreme values validation failed for {symbol}: {error_msg}")

        return ValidationResult(
            is_valid=len(errors) == 0,
            quality_score=1.0 if len(warnings) == 0 else 0.8,
            failed_rules=[],
            warnings=warnings,
            errors=errors
        )

    async def _validate_volume_spikes(self, df: pd.DataFrame, symbol: str) -> ValidationResult:
        """Validate for volume spikes."""
        warnings: List[str] = []
        errors: List[str] = []

        if 'volume' not in df.columns or df.empty:
            return ValidationResult(
                is_valid=True,
                quality_score=1.0,
                failed_rules=[],
                warnings=warnings,
                errors=errors
            )

        # Calculate volume statistics
        mean_volume: float = df['volume'].mean()
        std_volume: float = df['volume'].std()

        if std_volume > 0:
            # Detect volume spikes using configured tolerance
            # Convert tolerance to multiplier (5.0 = 500% = 5x multiplier)
            volume_multiplier: float = self.volume_tolerance if self.volume_tolerance > 1.0 else 1.0 + self.volume_tolerance
            volume_spikes = df[df['volume'] > mean_volume * volume_multiplier]

            if not volume_spikes.empty:
                warning_msg: str = f"Volume spikes detected: {len(volume_spikes)} occurrences"
                warnings.append(warning_msg)
                tprint_warning(f"⚠️ Volume spikes validation warning for {symbol}: {warning_msg}")

        # Check for zero volume periods
        zero_volume = df[df['volume'] == 0]
        if not zero_volume.empty:
            warning_msg = f"Zero volume periods: {len(zero_volume)} occurrences"
            warnings.append(warning_msg)
            tprint_warning(f"⚠️ Volume spikes validation warning for {symbol}: {warning_msg}")

        return ValidationResult(
            is_valid=len(errors) == 0,
            quality_score=1.0 if len(warnings) == 0 else 0.9,
            failed_rules=[],
            warnings=warnings,
            errors=errors
        )

    async def _validate_price_gaps(self, df: pd.DataFrame, symbol: str) -> ValidationResult:
        """Validate for price gaps."""
        warnings = []
        errors = []

        if 'close' not in df.columns or len(df) < 2:
            return ValidationResult(
                is_valid=True,
                quality_score=1.0,
                failed_rules=[],
                warnings=warnings,
                errors=errors
            )

        # Calculate price gaps between consecutive periods
        # Only if we have timestamp information to verify continuity
        timestamp_cols = [col for col in df.columns if 'timestamp' in col.lower()]
        
        if timestamp_cols and len(df) > 1:
            timestamp_col = timestamp_cols[0]
            timestamps = pd.to_datetime(df[timestamp_col], errors='coerce')
            time_diffs = timestamps.diff().dropna()
            
            # Only check gaps for consecutive periods (no missing time periods)
            consecutive_mask = pd.Series([True] * len(df))
            if len(time_diffs) > 0:
                expected_interval = time_diffs.median()
                # Mark rows where time gap is reasonable (within 2x expected)
                consecutive_mask[1:] = time_diffs <= (2 * expected_interval)
            
            # Calculate price changes only for consecutive periods
            price_changes = df['close'].pct_change().abs()
            price_changes = price_changes[consecutive_mask]
        else:
            # Fallback to simple pct_change if no timestamp info
            price_changes = df['close'].pct_change().abs()
        
        gap_threshold = 0.05  # 5% gap
        significant_gaps = price_changes[price_changes > gap_threshold]

        if not significant_gaps.empty:
            warnings.append(f"Significant price gaps detected: {len(significant_gaps)} occurrences")

        return ValidationResult(
            is_valid=len(errors) == 0,
            quality_score=1.0 if len(warnings) == 0 else 0.95,
            failed_rules=[],
            warnings=warnings,
            errors=errors
        )

    async def _validate_timestamp_order(self, df: pd.DataFrame) -> ValidationResult:
        """Validate timestamp ordering."""
        errors = []
        warnings = []

        if df.empty:
            return ValidationResult(
                is_valid=True,
                quality_score=1.0,
                failed_rules=[],
                warnings=warnings,
                errors=errors
            )

        # Check for timestamp column
        timestamp_cols = [col for col in df.columns if 'timestamp' in col.lower() or 'time' in col.lower()]

        if not timestamp_cols:
            warnings.append("No timestamp column found")
            return ValidationResult(
                is_valid=False,
                quality_score=0.5,
                failed_rules=[ValidationRule.TIMESTAMP_ORDER],
                warnings=warnings,
                errors=errors
            )

        timestamp_col = timestamp_cols[0]
        
        # Normalize timestamps to UTC for comparison
        try:
            timestamps = df[timestamp_col].copy()
            # Convert to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(timestamps):
                timestamps = pd.to_datetime(timestamps, errors='coerce')
            
            # Ensure timezone-aware (UTC)
            if timestamps.dt.tz is None:
                timestamps = timestamps.dt.tz_localize('UTC')
            else:
                timestamps = timestamps.dt.tz_convert('UTC')
            
            # Check if monotonic increasing
            if timestamps.is_monotonic_increasing:
                # Check for reasonable time intervals
                if len(timestamps) > 1:
                    time_diffs = timestamps.diff().dropna()
                    if len(time_diffs) > 0:
                        expected_interval = time_diffs.median()

                        # Check for large gaps (more than 2x expected interval)
                        large_gaps = time_diffs[time_diffs > 2 * expected_interval]

                        if not large_gaps.empty:
                            warnings.append(f"Large time gaps detected: {len(large_gaps)} occurrences")
            else:
                errors.append("Timestamps are not in chronological order")
        except Exception as e:
            errors.append(f"Failed to validate timestamp order: {str(e)}")

        return ValidationResult(
            is_valid=len(errors) == 0,
            quality_score=1.0 if len(errors) == 0 else 0.0,
            failed_rules=[ValidationRule.TIMESTAMP_ORDER] if errors else [],
            warnings=warnings,
            errors=errors
        )

    async def _validate_duplicate_data(self, df: pd.DataFrame) -> ValidationResult:
        """Validate for duplicate data."""
        warnings = []
        errors = []

        if df.empty:
            return ValidationResult(
                is_valid=True,
                quality_score=1.0,
                failed_rules=[],
                warnings=warnings,
                errors=errors
            )

        # Check for duplicate rows
        duplicate_rows = df[df.duplicated()]
        if not duplicate_rows.empty:
            warnings.append(f"Duplicate rows detected: {len(duplicate_rows)} rows")

        # Check for duplicate timestamps if timestamp column exists
        timestamp_cols = [col for col in df.columns if 'timestamp' in col.lower()]
        if timestamp_cols:
            timestamp_col = timestamp_cols[0]
            timestamp_duplicates = df[df.duplicated(subset=[timestamp_col])]
            if not timestamp_duplicates.empty:
                warnings.append(f"Duplicate timestamps detected: {len(timestamp_duplicates)} rows")

        return ValidationResult(
            is_valid=len(errors) == 0,
            quality_score=1.0 if len(warnings) == 0 else 0.95,
            failed_rules=[],
            warnings=warnings,
            errors=errors
        )

    async def _validate_data_freshness(self, df: pd.DataFrame) -> ValidationResult:
        """Validate data freshness."""
        warnings = []
        errors = []

        if df.empty:
            return ValidationResult(
                is_valid=False,
                quality_score=0.0,
                failed_rules=[ValidationRule.DATA_FRESHNESS],
                warnings=warnings,
                errors=["No data available"]
            )

        # Check for timestamp column
        timestamp_cols = [col for col in df.columns if 'timestamp' in col.lower()]

        if not timestamp_cols:
            warnings.append("No timestamp column found for freshness check")
            return ValidationResult(
                is_valid=False,
                quality_score=0.5,
                failed_rules=[ValidationRule.DATA_FRESHNESS],
                warnings=warnings,
                errors=errors
            )

        timestamp_col = timestamp_cols[0]
        latest_timestamp = df[timestamp_col].max()
        
        # Ensure latest_timestamp is a datetime object
        if not isinstance(latest_timestamp, datetime):
            try:
                # Try to convert if it's a pandas Timestamp
                if hasattr(latest_timestamp, 'to_pydatetime'):
                    latest_timestamp = latest_timestamp.to_pydatetime()
                elif isinstance(latest_timestamp, pd.Timestamp):
                    latest_timestamp = latest_timestamp.to_pydatetime()
                else:
                    errors.append(f"Timestamp column contains non-datetime values: {type(latest_timestamp)}")
                    return ValidationResult(
                        is_valid=False,
                        quality_score=0.0,
                        failed_rules=[ValidationRule.DATA_FRESHNESS],
                        warnings=warnings,
                        errors=errors
                    )
            except Exception as e:
                errors.append(f"Failed to convert timestamp: {str(e)}")
                return ValidationResult(
                    is_valid=False,
                    quality_score=0.0,
                    failed_rules=[ValidationRule.DATA_FRESHNESS],
                    warnings=warnings,
                    errors=errors
                )
        
        # Ensure timezone-aware (use UTC)
        if latest_timestamp.tzinfo is None:
            latest_timestamp = latest_timestamp.replace(tzinfo=timezone.utc)
        elif latest_timestamp.tzinfo != timezone.utc:
            latest_timestamp = latest_timestamp.astimezone(timezone.utc)
        
        now = datetime.now(timezone.utc)

        # Calculate age of latest data
        age_minutes = (now - latest_timestamp).total_seconds() / 60

        if age_minutes > self.freshness_threshold_minutes:
            warnings.append(f"Data is stale: {age_minutes:.1f} minutes old")

        return ValidationResult(
            is_valid=age_minutes <= self.freshness_threshold_minutes,
            quality_score=max(0.0, 1.0 - (age_minutes / (self.freshness_threshold_minutes * 2))),
            failed_rules=[ValidationRule.DATA_FRESHNESS] if age_minutes > self.freshness_threshold_minutes else [],
            warnings=warnings,
            errors=errors
        )

    async def _calculate_quality_score(self, validation_results: Dict[ValidationRule, ValidationResult]) -> float:
        """Calculate overall data quality score."""
        total_weight: float = sum(self.quality_weights.values())
        weighted_score: float = 0.0

        for rule, result in validation_results.items():
            if rule in self.quality_weights:
                weight: float = self.quality_weights[rule]
                weighted_score += weight * result.quality_score

        score: float = weighted_score / total_weight if total_weight > 0 else 0.0
        tprint_info(f"📊 Calculated quality score: {score:.3f}")
        return score

    async def generate_quality_report(
        self,
        data: Union[pd.DataFrame, Dict[str, Any]],
        symbol: str
    ) -> DataQualityReport:
        """
        Generate comprehensive data quality report.

        Args:
            data: Market data to analyze
            symbol: Trading symbol

        Returns:
            Data quality report
        """
        validation_result = await self.validate_market_data(data, symbol)

        # Determine overall quality level
        if validation_result.quality_score >= 0.9:
            overall_quality = DataQualityLevel.HIGH
        elif validation_result.quality_score >= 0.7:
            overall_quality = DataQualityLevel.MEDIUM
        elif validation_result.quality_score >= 0.5:
            overall_quality = DataQualityLevel.LOW
        else:
            overall_quality = DataQualityLevel.CRITICAL

        # Generate recommendations
        recommendations = await self._generate_recommendations(validation_result)

        # Calculate data statistics
        data_stats = await self._calculate_data_stats(data)

        return DataQualityReport(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc),
            overall_quality=overall_quality,
            validation_results=validation_result.metadata.get('validation_results', {}),
            recommendations=recommendations,
            data_stats=data_stats
        )

    async def _generate_recommendations(self, validation_result: ValidationResult) -> List[str]:
        """Generate data quality recommendations."""
        recommendations: List[str] = []

        if ValidationRule.MISSING_DATA in validation_result.failed_rules:
            recommendations.append("Consider implementing data imputation strategies")
            recommendations.append("Review data collection process for missing data sources")

        if ValidationRule.EXTREME_VALUES in validation_result.failed_rules:
            recommendations.append("Implement outlier detection and filtering")
            recommendations.append("Review data source for data quality issues")

        if ValidationRule.OHLC_CONSISTENCY in validation_result.failed_rules:
            recommendations.append("Fix OHLC data consistency issues")
            recommendations.append("Validate data processing pipeline")

        if validation_result.quality_score < 0.7:
            recommendations.append("Consider using alternative data sources")
            recommendations.append("Implement real-time data validation")

        return recommendations

    async def _calculate_data_stats(self, data: Union[pd.DataFrame, Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate basic data statistics."""
        df: pd.DataFrame
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = data.copy()

        stats: Dict[str, Any] = {
            'row_count': len(df),
            'column_count': len(df.columns),
            'missing_data_percentage': (df.isnull().sum().sum() / df.size * 100) if df.size > 0 else 0,
            'duplicate_rows': df.duplicated().sum(),
            'data_types': df.dtypes.to_dict()
        }

        # Numeric column statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if not df.empty and len(numeric_cols) > 0:
            stats['numeric_stats'] = {
                col: {
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max()
                } for col in numeric_cols
            }

        return stats

    async def cleanup(self) -> None:
        """Clean up resources."""
        self.price_history.clear()
        self.volume_history.clear()
        tprint_info("🧹 Data Validator cleaned up successfully")

# Factory functions
async def create_data_validator(config: Dict[str, Any]) -> DataValidator:
    """Create and initialize a data validator."""
    validator = DataValidator(config)
    await validator.initialize()
    return validator

def get_data_validator() -> Optional[DataValidator]:
    """Get the global data validator instance."""
    return None
