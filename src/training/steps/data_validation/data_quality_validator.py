"""
Data Quality Validator

This module provides comprehensive data quality validation for trading data,
ensuring data integrity, completeness, and consistency before model training.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import time
import gc

# Core utilities
from src.utils.common_operations import (
    safe_json_dump, safe_json_load, safe_file_exists, ensure_directory,
    safe_mean, safe_std, safe_float, safe_int, get_current_datetime,
    safe_append, safe_extend, safe_dict_get, safe_lower, safe_upper,
    format_datetime, validate_file_path, get_file_size, check_disk_space
)
from src.utils.math_validation import (
    safe_divide, safe_log, safe_sqrt, safe_power, validate_finite,
    validate_positive, validate_range, safe_kelly_calculation,
    safe_weighted_average, safe_percentage_change, MathValidationError
)
from src.utils.parquet_utils import get_parquet_utils, ParquetUtils
from src.core.decorators import (
    handles_errors, validates, traced, log_execution_time, 
    timeout, error_boundary, compose, validate_data_quality, 
    monitor_step_execution, ensure_data_integrity, validate_pipeline_step
)
from src.utils.enhanced_error_handler import handle_errors_with_tracking
from src.utils.logger import system_logger

class DataQualityLevel(Enum):
    """Data quality levels for validation."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    MINIMAL = "minimal"

class ValidationResult(Enum):
    """Validation result types."""
    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"
    SKIP = "skip"

@dataclass
class QualityMetric:
    """Data quality metric result."""
    name: str
    value: float
    threshold: float
    result: ValidationResult
    message: str
    severity: DataQualityLevel
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class DataQualityReport:
    """Comprehensive data quality report."""
    symbol: str
    exchange: str
    timeframe: str
    total_records: int
    quality_score: float
    metrics: List[QualityMetric]
    issues: List[str]
    warnings: List[str]
    recommendations: List[str]
    validation_time: float
    timestamp: datetime = field(default_factory=datetime.now)

class DataQualityValidator:
    """
    Comprehensive data quality validator for trading data.
    
    This validator performs multiple quality checks including:
    - Data completeness and consistency
    - Statistical validation
    - Temporal integrity
    - Schema validation
    - Outlier detection
    - Missing data analysis
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize data quality validator."""
        self.config = config
        self.logger = system_logger.getChild('DataQualityValidator')
        self.parquet_utils = get_parquet_utils()
        
        # Quality thresholds
        self.quality_config = config.get('data_quality', {})
        self.completeness_threshold = self.quality_config.get('completeness_threshold', 0.95)
        self.consistency_threshold = self.quality_config.get('consistency_threshold', 0.90)
        self.outlier_threshold = self.quality_config.get('outlier_threshold', 0.05)
        self.missing_data_threshold = self.quality_config.get('missing_data_threshold', 0.02)
        
        # Validation settings
        self.enable_statistical_validation = self.quality_config.get('enable_statistical_validation', True)
        self.enable_temporal_validation = self.quality_config.get('enable_temporal_validation', True)
        self.enable_schema_validation = self.quality_config.get('enable_schema_validation', True)
        self.enable_outlier_detection = self.quality_config.get('enable_outlier_detection', True)
        
        self.validation_results: List[QualityMetric] = []
        self.quality_report: Optional[DataQualityReport] = None

    @handles_errors(Exception, fallback=False, log_level='ERROR')
    @validates(strict=True)
    @traced("data_quality_validation")
    @log_execution_time
    async def validate_data_quality(
        self, 
        symbol: str, 
        exchange: str, 
        timeframe: str, 
        data_dir: str
    ) -> DataQualityReport:
        """
        Perform comprehensive data quality validation.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Data timeframe
            data_dir: Data directory path
            
        Returns:
            DataQualityReport: Comprehensive quality report
        """
        start_time = time.time()
        self.logger.info(f"🔍 Starting data quality validation for {symbol} on {exchange}")
        
        try:
            # Load data
            data_file = f"{data_dir}/aggtrades_{exchange}_{symbol}_consolidated.parquet"
            if not safe_file_exists(data_file):
                raise FileNotFoundError(f"Data file not found: {data_file}")
            
            df = self.parquet_utils.read_parquet(data_file)
            if df.empty:
                raise ValueError("Data file is empty")
            
            self.logger.info(f"📊 Loaded {len(df)} records for validation")
            
            # Initialize report
            self.quality_report = DataQualityReport(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                total_records=len(df),
                quality_score=0.0,
                metrics=[],
                issues=[],
                warnings=[],
                recommendations=[],
                validation_time=0.0
            )
            
            # Perform validation checks
            await self._validate_completeness(df)
            await self._validate_consistency(df)
            await self._validate_schema(df)
            await self._validate_temporal_integrity(df)
            await self._validate_statistical_properties(df)
            await self._detect_outliers(df)
            await self._analyze_missing_data(df)
            
            # Calculate overall quality score
            self._calculate_quality_score()
            
            # Generate recommendations
            self._generate_recommendations()
            
            validation_time = time.time() - start_time
            self.quality_report.validation_time = validation_time
            
            self.logger.info(f"✅ Data quality validation completed in {validation_time:.2f}s")
            self.logger.info(f"📊 Overall quality score: {self.quality_report.quality_score:.2f}")
            
            return self.quality_report
            
        except Exception as e:
            self.logger.error(f"❌ Data quality validation failed: {e}")
            raise

    @handles_errors(Exception, fallback=False, log_level='WARNING')
    @traced("completeness_validation")
    async def _validate_completeness(self, df: pd.DataFrame) -> None:
        """Validate data completeness."""
        self.logger.info("🔍 Validating data completeness...")
        
        # Check for required columns
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            metric = QualityMetric(
                name="required_columns",
                value=0.0,
                threshold=1.0,
                result=ValidationResult.FAIL,
                message=f"Missing required columns: {missing_columns}",
                severity=DataQualityLevel.CRITICAL
            )
            self.validation_results.append(metric)
            self.quality_report.issues.append(f"Missing required columns: {missing_columns}")
        else:
            metric = QualityMetric(
                name="required_columns",
                value=1.0,
                threshold=1.0,
                result=ValidationResult.PASS,
                message="All required columns present",
                severity=DataQualityLevel.HIGH
            )
            self.validation_results.append(metric)
        
        # Check data volume
        min_records = self.quality_config.get('min_records', 1000)
        if len(df) < min_records:
            metric = QualityMetric(
                name="data_volume",
                value=len(df) / min_records,
                threshold=1.0,
                result=ValidationResult.WARNING,
                message=f"Low data volume: {len(df)} records (minimum: {min_records})",
                severity=DataQualityLevel.MEDIUM
            )
            self.validation_results.append(metric)
            self.quality_report.warnings.append(f"Low data volume: {len(df)} records")
        else:
            metric = QualityMetric(
                name="data_volume",
                value=1.0,
                threshold=1.0,
                result=ValidationResult.PASS,
                message=f"Sufficient data volume: {len(df)} records",
                severity=DataQualityLevel.HIGH
            )
            self.validation_results.append(metric)

    @handles_errors(Exception, fallback=False, log_level='WARNING')
    @traced("consistency_validation")
    async def _validate_consistency(self, df: pd.DataFrame) -> None:
        """Validate data consistency."""
        self.logger.info("🔍 Validating data consistency...")
        
        # Check OHLC consistency
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            inconsistent_ohlc = (
                (df['high'] < df['low']) |
                (df['high'] < df['open']) |
                (df['high'] < df['close']) |
                (df['low'] > df['open']) |
                (df['low'] > df['close'])
            ).sum()
            
            consistency_ratio = 1.0 - (inconsistent_ohlc / len(df))
            
            if consistency_ratio < self.consistency_threshold:
                metric = QualityMetric(
                    name="ohlc_consistency",
                    value=consistency_ratio,
                    threshold=self.consistency_threshold,
                    result=ValidationResult.FAIL,
                    message=f"OHLC inconsistency: {inconsistent_ohlc} records",
                    severity=DataQualityLevel.HIGH
                )
                self.validation_results.append(metric)
                self.quality_report.issues.append(f"OHLC inconsistency: {inconsistent_ohlc} records")
            else:
                metric = QualityMetric(
                    name="ohlc_consistency",
                    value=consistency_ratio,
                    threshold=self.consistency_threshold,
                    result=ValidationResult.PASS,
                    message="OHLC data is consistent",
                    severity=DataQualityLevel.HIGH
                )
                self.validation_results.append(metric)
        
        # Check for duplicate timestamps
        if 'timestamp' in df.columns:
            duplicate_timestamps = df['timestamp'].duplicated().sum()
            if duplicate_timestamps > 0:
                metric = QualityMetric(
                    name="duplicate_timestamps",
                    value=1.0 - (duplicate_timestamps / len(df)),
                    threshold=0.99,
                    result=ValidationResult.WARNING,
                    message=f"Duplicate timestamps: {duplicate_timestamps} records",
                    severity=DataQualityLevel.MEDIUM
                )
                self.validation_results.append(metric)
                self.quality_report.warnings.append(f"Duplicate timestamps: {duplicate_timestamps} records")

    @handles_errors(Exception, fallback=False, log_level='WARNING')
    @traced("schema_validation")
    async def _validate_schema(self, df: pd.DataFrame) -> None:
        """Validate data schema."""
        self.logger.info("🔍 Validating data schema...")
        
        # Check data types
        expected_types = {
            'timestamp': 'datetime64[ns]',
            'open': 'float64',
            'high': 'float64',
            'low': 'float64',
            'close': 'float64',
            'volume': 'float64'
        }
        
        type_issues = []
        for col, expected_type in expected_types.items():
            if col in df.columns:
                actual_type = str(df[col].dtype)
                if actual_type != expected_type:
                    type_issues.append(f"{col}: expected {expected_type}, got {actual_type}")
        
        if type_issues:
            metric = QualityMetric(
                name="data_types",
                value=0.0,
                threshold=1.0,
                result=ValidationResult.WARNING,
                message=f"Data type issues: {type_issues}",
                severity=DataQualityLevel.MEDIUM
            )
            self.validation_results.append(metric)
            self.quality_report.warnings.extend(type_issues)
        else:
            metric = QualityMetric(
                name="data_types",
                value=1.0,
                threshold=1.0,
                result=ValidationResult.PASS,
                message="All data types are correct",
                severity=DataQualityLevel.HIGH
            )
            self.validation_results.append(metric)

    @handles_errors(Exception, fallback=False, log_level='WARNING')
    @traced("temporal_validation")
    async def _validate_temporal_integrity(self, df: pd.DataFrame) -> None:
        """Validate temporal integrity."""
        if not self.enable_temporal_validation or 'timestamp' not in df.columns:
            return
            
        self.logger.info("🔍 Validating temporal integrity...")
        
        # Check for future timestamps
        current_time = datetime.now()
        future_timestamps = (df['timestamp'] > current_time).sum()
        
        if future_timestamps > 0:
            metric = QualityMetric(
                name="future_timestamps",
                value=1.0 - (future_timestamps / len(df)),
                threshold=1.0,
                result=ValidationResult.WARNING,
                message=f"Future timestamps: {future_timestamps} records",
                severity=DataQualityLevel.MEDIUM
            )
            self.validation_results.append(metric)
            self.quality_report.warnings.append(f"Future timestamps: {future_timestamps} records")
        
        # Check timestamp ordering
        if not df['timestamp'].is_monotonic_increasing:
            metric = QualityMetric(
                name="timestamp_ordering",
                value=0.0,
                threshold=1.0,
                result=ValidationResult.WARNING,
                message="Timestamps are not in chronological order",
                severity=DataQualityLevel.MEDIUM
            )
            self.validation_results.append(metric)
            self.quality_report.warnings.append("Timestamps are not in chronological order")

    @handles_errors(Exception, fallback=False, log_level='WARNING')
    @traced("statistical_validation")
    async def _validate_statistical_properties(self, df: pd.DataFrame) -> None:
        """Validate statistical properties."""
        if not self.enable_statistical_validation:
            return
            
        self.logger.info("🔍 Validating statistical properties...")
        
        # Check for negative prices
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in df.columns:
                negative_prices = (df[col] <= 0).sum()
                if negative_prices > 0:
                    metric = QualityMetric(
                        name=f"{col}_negative_prices",
                        value=1.0 - (negative_prices / len(df)),
                        threshold=1.0,
                        result=ValidationResult.FAIL,
                        message=f"Negative {col} prices: {negative_prices} records",
                        severity=DataQualityLevel.CRITICAL
                    )
                    self.validation_results.append(metric)
                    self.quality_report.issues.append(f"Negative {col} prices: {negative_prices} records")
        
        # Check for zero volume
        if 'volume' in df.columns:
            zero_volume = (df['volume'] == 0).sum()
            if zero_volume > 0:
                metric = QualityMetric(
                    name="zero_volume",
                    value=1.0 - (zero_volume / len(df)),
                    threshold=0.95,
                    result=ValidationResult.WARNING,
                    message=f"Zero volume records: {zero_volume}",
                    severity=DataQualityLevel.MEDIUM
                )
                self.validation_results.append(metric)
                self.quality_report.warnings.append(f"Zero volume records: {zero_volume}")

    @handles_errors(Exception, fallback=False, log_level='WARNING')
    @traced("outlier_detection")
    async def _detect_outliers(self, df: pd.DataFrame) -> None:
        """Detect outliers in the data."""
        if not self.enable_outlier_detection:
            return
            
        self.logger.info("🔍 Detecting outliers...")
        
        # Use IQR method for outlier detection
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                outlier_ratio = outliers / len(df)
                
                if outlier_ratio > self.outlier_threshold:
                    metric = QualityMetric(
                        name=f"{col}_outliers",
                        value=1.0 - outlier_ratio,
                        threshold=1.0 - self.outlier_threshold,
                        result=ValidationResult.WARNING,
                        message=f"High outlier ratio in {col}: {outlier_ratio:.2%}",
                        severity=DataQualityLevel.MEDIUM
                    )
                    self.validation_results.append(metric)
                    self.quality_report.warnings.append(f"High outlier ratio in {col}: {outlier_ratio:.2%}")

    @handles_errors(Exception, fallback=False, log_level='WARNING')
    @traced("missing_data_analysis")
    async def _analyze_missing_data(self, df: pd.DataFrame) -> None:
        """Analyze missing data patterns."""
        self.logger.info("🔍 Analyzing missing data...")
        
        missing_data = df.isnull().sum()
        total_missing = missing_data.sum()
        missing_ratio = total_missing / (len(df) * len(df.columns))
        
        if missing_ratio > self.missing_data_threshold:
            metric = QualityMetric(
                name="missing_data",
                value=1.0 - missing_ratio,
                threshold=1.0 - self.missing_data_threshold,
                result=ValidationResult.WARNING,
                message=f"High missing data ratio: {missing_ratio:.2%}",
                severity=DataQualityLevel.MEDIUM
            )
            self.validation_results.append(metric)
            self.quality_report.warnings.append(f"High missing data ratio: {missing_ratio:.2%}")
        else:
            metric = QualityMetric(
                name="missing_data",
                value=1.0 - missing_ratio,
                threshold=1.0 - self.missing_data_threshold,
                result=ValidationResult.PASS,
                message=f"Acceptable missing data ratio: {missing_ratio:.2%}",
                severity=DataQualityLevel.HIGH
            )
            self.validation_results.append(metric)

    def _calculate_quality_score(self) -> None:
        """Calculate overall quality score."""
        if not self.validation_results:
            self.quality_report.quality_score = 0.0
            return
        
        # Weight metrics by severity
        severity_weights = {
            DataQualityLevel.CRITICAL: 1.0,
            DataQualityLevel.HIGH: 0.8,
            DataQualityLevel.MEDIUM: 0.6,
            DataQualityLevel.LOW: 0.4,
            DataQualityLevel.MINIMAL: 0.2
        }
        
        weighted_scores = []
        for metric in self.validation_results:
            weight = severity_weights.get(metric.severity, 0.5)
            score = metric.value * weight
            weighted_scores.append(score)
        
        self.quality_report.quality_score = safe_mean(weighted_scores) if weighted_scores else 0.0
        self.quality_report.metrics = self.validation_results

    def _generate_recommendations(self) -> None:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        # Analyze failed validations
        failed_metrics = [m for m in self.validation_results if m.result == ValidationResult.FAIL]
        warning_metrics = [m for m in self.validation_results if m.result == ValidationResult.WARNING]
        
        if failed_metrics:
            recommendations.append("Address critical data quality issues before training")
            recommendations.append("Consider data cleaning and preprocessing")
        
        if warning_metrics:
            recommendations.append("Monitor data quality metrics during training")
            recommendations.append("Consider additional validation steps")
        
        if self.quality_report.quality_score < 0.8:
            recommendations.append("Data quality is below recommended threshold")
            recommendations.append("Consider data augmentation or alternative data sources")
        
        if self.quality_report.quality_score >= 0.9:
            recommendations.append("Data quality is excellent - proceed with training")
        
        self.quality_report.recommendations = recommendations

    @handles_errors(Exception, fallback=False, log_level='ERROR')
    @traced("save_quality_report")
    async def save_quality_report(self, output_dir: str) -> str:
        """Save quality report to file."""
        if not self.quality_report:
            raise ValueError("No quality report available to save")
        
        ensure_directory(output_dir)
        report_file = f"{output_dir}/data_quality_report_{self.quality_report.symbol}_{self.quality_report.timeframe}.json"
        
        # Convert to JSON-serializable format
        report_data = {
            'symbol': self.quality_report.symbol,
            'exchange': self.quality_report.exchange,
            'timeframe': self.quality_report.timeframe,
            'total_records': self.quality_report.total_records,
            'quality_score': self.quality_report.quality_score,
            'metrics': [
                {
                    'name': m.name,
                    'value': m.value,
                    'threshold': m.threshold,
                    'result': m.result.value,
                    'message': m.message,
                    'severity': m.severity.value,
                    'timestamp': m.timestamp.isoformat()
                }
                for m in self.quality_report.metrics
            ],
            'issues': self.quality_report.issues,
            'warnings': self.quality_report.warnings,
            'recommendations': self.quality_report.recommendations,
            'validation_time': self.quality_report.validation_time,
            'timestamp': self.quality_report.timestamp.isoformat()
        }
        
        safe_json_dump(report_data, report_file, indent=2)
        self.logger.info(f"💾 Quality report saved to: {report_file}")
        
        return report_file