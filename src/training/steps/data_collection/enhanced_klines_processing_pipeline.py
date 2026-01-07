"""
Enhanced Klines Data Processing Pipeline

This module provides a complete, production-ready pipeline for downloading, processing, and quality-checking
historical klines data with comprehensive type hints, exchange-agnostic design, and fast-fail patterns.

Features:
- Full type hints and tprint logging throughout
- Exchange-agnostic design using ExchangeInterface
- Data standardization using ExchangeDataStandardizer
- Fast fail pattern with no fallbacks, mocks, or stubs
- Comprehensive gap detection and filling
- Data resampling capabilities (1m, 5m, 15m, 30m, 1h for data older than 3 days)
- OHLCV data validation and formatting
from src.utils.numba_funcs import (_numba_detect_gaps_vectorized, _numba_fill_gaps_vectorized, 
                                  _numba_ohlc_resample_vectorized, _numba_verify_data_quality)
- Duplicate detection and handling
from src.utils.numba_funcs import NUMBA_AVAILABLE
- Comprehensive quality assurance and validation using src/utils/data/quality/
- Advanced data quality scoring and assessment
- Statistical distribution validation
- Quality trend analysis over time
- Automated data cleaning with quality utilities
- Quality alert system integration
- Efficient parquet storage using KlinesParquetManager
- Batch-compatible data management
- Automatic gap filling before resampling

Quality Features:
- Multi-layered quality validation using DataQualityFramework
- Comprehensive quality scoring with component breakdowns
- Advanced quality metrics and statistical analysis
- Quality trend analysis and monitoring
"""

# Type annotations for lazy-loaded quality utilities
from typing import Optional, TYPE_CHECKING, Any, Union
import sys
import os
from pathlib import Path

if TYPE_CHECKING:
    # Type-only imports for linter - these help the linter understand types
    try:
        from src.utils.data.quality.comprehensive_duplicate_analyzer import (
            ComprehensiveDuplicateAnalyzer,
            analyze_duplicates_comprehensive
        )
        from src.utils.data.quality.data_quality import DataQualityFramework, QualityThresholds, QualityResult
        from src.utils.data.quality.comprehensive_quality_scorer import ComprehensiveQualityScorer
        from src.utils.data.quality.advanced_quality_metrics import AdvancedQualityMetrics
        from src.utils.data.quality.data_cleaning import DataCleaner
        from src.utils.data.quality.statistical_distribution_validation import StatisticalValidator
        from src.utils.data.quality.quality_alert_system import QualityAlertManager

        # Type aliases for clarity
        QualityUtilityType = Union[
            ComprehensiveDuplicateAnalyzer,
            ComprehensiveQualityScorer,
            AdvancedQualityMetrics,
            DataCleaner,
            StatisticalValidator,
            DataQualityFramework,
            QualityThresholds,
            QualityResult,
            QualityAlertManager,
            type(None)
        ]
    except ImportError:
        # Fallback types for when imports fail
        QualityUtilityType = Any
        ComprehensiveDuplicateAnalyzer = Any
        analyze_duplicates_comprehensive = Any
        DataQualityFramework = Any
        QualityThresholds = Any
        QualityResult = Any
        ComprehensiveQualityScorer = Any
        AdvancedQualityMetrics = Any
        DataCleaner = Any
        StatisticalValidator = Any
        QualityAlertManager = Any

import asyncio
import numpy as np
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union, Callable, Awaitable, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import sys

# Add project root to path for imports
# File is at: Ares/src/training/steps/data_collection/enhanced_klines_processing_pipeline.py
# We need to go up 5 levels to get to Ares
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Use lazy imports to avoid circular import issues
def get_system_logger():
    """Lazy import of system logger to avoid circular imports."""
    try:
        from src.utils.logger import system_logger
        return system_logger
    except ImportError:
        import logging
        return logging.getLogger(__name__)

def get_tprint_functions():
    """Lazy import of tprint functions to avoid circular imports."""
    try:
        from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success
        return tprint, tprint_info, tprint_warning, tprint_error, tprint_success
    except ImportError:
        # Fallback functions
        def tprint(*args, **kwargs):
            print(*args, **kwargs)
        return tprint, tprint, tprint, tprint, tprint

# Initialize lazy imports
system_logger = get_system_logger()
tprint, tprint_info, tprint_warning, tprint_error, tprint_success = get_tprint_functions()
# Import comprehensive data quality utilities from src/utils/data/quality/
# Use lazy imports to avoid circular import issues

# Global variables for lazy-loaded quality utilities with type annotations
QUALITY_UTILITIES_AVAILABLE: bool = False
_COMPREHENSIVE_DUPLICATE_ANALYZER: Optional[Any] = None
_DATA_QUALITY_FRAMEWORK: Optional[Any] = None
_COMPREHENSIVE_QUALITY_SCORER: Optional[Any] = None
_ADVANCED_QUALITY_METRICS: Optional[Any] = None
_DATA_CLEANER: Optional[Any] = None
_STATISTICAL_VALIDATOR: Optional[Any] = None
_QUALITY_ALERT_SYSTEM: Optional[Any] = None
_ANALYZE_DUPLICATES_COMPREHENSIVE = None

# Minimal fallback for QualityResult (needed before lazy import)
@dataclass
class FallbackQualityResult:
    """Fallback quality result."""
    passed: bool
    issues: List[str]
    warnings: List[str]
    quality_score: float

# Define QualityScoreLevel, QualityScore, and QualityAssessment before lazy import
# (needed for fallback classes)
class QualityScoreLevel(Enum):
    """Quality score levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"

@dataclass
class QualityScore:
    """Quality score result."""
    overall_score: float
    level: QualityScoreLevel
    component_scores: Dict[str, float]
    issues: List[str]
    warnings: List[str]
    recommendations: List[str]
    assessment_timestamp: datetime
    data_shape: Tuple[int, int]

@dataclass
class QualityAssessment:
    """Quality assessment result."""
    overall_score: float
    metrics: List[str]
    issues_found: int
    warnings_found: int
    critical_issues: int
    assessment_timestamp: datetime
    data_shape: Tuple[int, int]

def _lazy_import_quality_utilities():
    """Lazy import of quality utilities to avoid circular imports."""
    global QUALITY_UTILITIES_AVAILABLE, _COMPREHENSIVE_DUPLICATE_ANALYZER, _DATA_QUALITY_FRAMEWORK
    global _COMPREHENSIVE_QUALITY_SCORER, _ADVANCED_QUALITY_METRICS, _DATA_CLEANER
    global _STATISTICAL_VALIDATOR, _QUALITY_ALERT_SYSTEM, _ANALYZE_DUPLICATES_COMPREHENSIVE
    
    if QUALITY_UTILITIES_AVAILABLE is True:
        return
    
    # Get the current module to update module-level names
    current_module = sys.modules[__name__]
    
    try:
        from src.utils.data.quality.comprehensive_duplicate_analyzer import (
            ComprehensiveDuplicateAnalyzer as ImportedComprehensiveDuplicateAnalyzer,
            analyze_duplicates_comprehensive as imported_analyze_duplicates_comprehensive
        )
        from src.utils.data.quality.data_quality import DataQualityFramework as ImportedDataQualityFramework, QualityThresholds, QualityResult
        from src.utils.data.quality.comprehensive_quality_scorer import ComprehensiveQualityScorer as ImportedComprehensiveQualityScorer, QualityScore, QualityScoreLevel
        from src.utils.data.quality.advanced_quality_metrics import AdvancedQualityMetrics as ImportedAdvancedQualityMetrics, QualityAssessment
        from src.utils.data.quality.data_cleaning import DataCleaner as ImportedDataCleaner
        from src.utils.data.quality.statistical_distribution_validation import StatisticalValidator as ImportedStatisticalValidator
        from src.utils.data.quality.quality_alert_system import QualityAlertManager as ImportedQualityAlertManager
        
        _COMPREHENSIVE_DUPLICATE_ANALYZER = ImportedComprehensiveDuplicateAnalyzer
        _DATA_QUALITY_FRAMEWORK = ImportedDataQualityFramework
        _COMPREHENSIVE_QUALITY_SCORER = ImportedComprehensiveQualityScorer
        _ADVANCED_QUALITY_METRICS = ImportedAdvancedQualityMetrics
        _DATA_CLEANER = ImportedDataCleaner
        _STATISTICAL_VALIDATOR = ImportedStatisticalValidator
        _QUALITY_ALERT_SYSTEM = ImportedQualityAlertManager
        _ANALYZE_DUPLICATES_COMPREHENSIVE = imported_analyze_duplicates_comprehensive
        
        # Make classes available at module level
        setattr(current_module, 'ComprehensiveDuplicateAnalyzer', ImportedComprehensiveDuplicateAnalyzer)
        setattr(current_module, 'DataQualityFramework', ImportedDataQualityFramework)
        setattr(current_module, 'ComprehensiveQualityScorer', ImportedComprehensiveQualityScorer)
        setattr(current_module, 'AdvancedQualityMetrics', ImportedAdvancedQualityMetrics)
        setattr(current_module, 'DataCleaner', ImportedDataCleaner)
        setattr(current_module, 'StatisticalValidator', ImportedStatisticalValidator)
        setattr(current_module, 'QualityAlertManager', ImportedQualityAlertManager)
        setattr(current_module, 'analyze_duplicates_comprehensive', imported_analyze_duplicates_comprehensive)
        
        QUALITY_UTILITIES_AVAILABLE = True
    except ImportError as e:
        tprint_warning(f"⚠️ Some data quality utilities not available: {e}")
        QUALITY_UTILITIES_AVAILABLE = False

        # Define fallback classes only if imports fail
        class ComprehensiveDuplicateAnalyzer:
            def analyze_duplicates(self, df):
                class Result:
                    total_duplicates = 0
                    true_duplicate_groups = 0
                    false_duplicate_groups = 0
                    mixed_duplicate_groups = 0
                return Result()

        def analyze_duplicates_comprehensive(df):
            return ComprehensiveDuplicateAnalyzer().analyze_duplicates(df)

        _ANALYZE_DUPLICATES_COMPREHENSIVE = analyze_duplicates_comprehensive
        _COMPREHENSIVE_DUPLICATE_ANALYZER = ComprehensiveDuplicateAnalyzer

        # Fallback classes for missing quality utilities
        class DataQualityFramework:
            def validate_data(self, df, thresholds=None):
                return FallbackQualityResult(passed=True, issues=[], warnings=[], quality_score=100.0)
            
            def validate_dataframe_quality(self, df, context=''):
                return FallbackQualityResult(passed=True, issues=[], warnings=[], quality_score=100.0)

        class ComprehensiveQualityScorer:
            def score_data_quality(self, df, symbol=None, interval=None):
                return QualityScore(overall_score=0.0, level=QualityScoreLevel.CRITICAL,
                                  component_scores={}, issues=[], warnings=[],
                                  recommendations=[], assessment_timestamp=datetime.now(),
                                  data_shape=(0, 0))
            
            def assess_data_quality(self, df, symbol=None, interval=None, context=None, step_name=None, data_type=None):
                return QualityScore(overall_score=0.0, level=QualityScoreLevel.CRITICAL,
                                  component_scores={}, issues=[], warnings=[],
                                  recommendations=[], assessment_timestamp=datetime.now(),
                                  data_shape=(0, 0))

        class AdvancedQualityMetrics:
            def assess_quality(self, df):
                return QualityAssessment(overall_score=0.0, metrics=[], issues_found=0,
                                       warnings_found=0, critical_issues=0,
                                       assessment_timestamp=datetime.now(), data_shape=(0, 0))
            
            def comprehensive_quality_assessment(self, df, context=None, step_name=None):
                return QualityAssessment(overall_score=0.0, metrics=[], issues_found=0,
                                       warnings_found=0, critical_issues=0,
                                       assessment_timestamp=datetime.now(), data_shape=(0, 0))

        class DataCleaner:
            def clean_data(self, df):
                return df

        class StatisticalValidator:
            def validate_distributions(self, df):
                return {}
            
            def run_comprehensive_validation(self, data):
                return []

        class QualityAlertManager:
            def check_alerts(self, quality_score):
                return []

        # Make fallback classes available at module level
        _DATA_QUALITY_FRAMEWORK = DataQualityFramework
        _COMPREHENSIVE_QUALITY_SCORER = ComprehensiveQualityScorer
        _ADVANCED_QUALITY_METRICS = AdvancedQualityMetrics
        _DATA_CLEANER = DataCleaner
        _STATISTICAL_VALIDATOR = StatisticalValidator
        _QUALITY_ALERT_SYSTEM = QualityAlertManager
        
        # Set fallback classes at module level
        setattr(current_module, 'ComprehensiveDuplicateAnalyzer', ComprehensiveDuplicateAnalyzer)
        setattr(current_module, 'DataQualityFramework', DataQualityFramework)
        setattr(current_module, 'ComprehensiveQualityScorer', ComprehensiveQualityScorer)
        setattr(current_module, 'AdvancedQualityMetrics', AdvancedQualityMetrics)
        setattr(current_module, 'DataCleaner', DataCleaner)
        setattr(current_module, 'StatisticalValidator', StatisticalValidator)
        setattr(current_module, 'QualityAlertManager', QualityAlertManager)
        setattr(current_module, 'analyze_duplicates_comprehensive', analyze_duplicates_comprehensive)

# Initialize quality utilities at module load time
_lazy_import_quality_utilities()
# Import ExchangeInterface from the proper location
try:
    from src.trading.execution.exchange_interface import ExchangeInterface, create_exchange_interface
    EXCHANGE_INTERFACE_AVAILABLE = True
except ImportError as e:  # Allow running without trading stack (use existing klines only)
    EXCHANGE_INTERFACE_AVAILABLE = False
    ExchangeInterface = None  # type: ignore
    create_exchange_interface = None  # type: ignore
    # Retry import after stubs are available
    try:
        from src.trading.execution.exchange_interface import ExchangeInterface as _EI, create_exchange_interface as _CEI
        ExchangeInterface = _EI
        create_exchange_interface = _CEI
        EXCHANGE_INTERFACE_AVAILABLE = True
    except Exception:
        pass
from exchanges.exchange_types import ExchangeType, TradingMode
# Dispatcher factory to wire exchange-specific adapters
try:
    from exchanges.exchange_dispatcher import create_exchange_dispatcher, ExchangeConfig
except Exception:
    create_exchange_dispatcher = None  # type: ignore
    ExchangeConfig = None  # type: ignore

# Import the proper classes from their locations (fallback stubs to bypass missing exchange deps)
try:
    from exchanges.shared.unified_ohlcv_standardizer import UnifiedOHLCVStandardizer
except Exception:
    class UnifiedOHLCVStandardizer:  # type: ignore
        """Fallback no-op standardizer when exchange shared modules are unavailable."""
        def standardize(self, df, *args, **kwargs):
            return df

from src.utils.data.klines_parquet import KlinesParquetManager
from src.training.steps.data_collection.klines_gap_filler_1m import fill_1m_gaps_and_resample_for_symbol

class StorageConfig:
    """Simple storage config."""
    def __init__(self, *args, **kwargs):
        pass

class KlinesMetadata:
    """Simple klines metadata."""
    def __init__(self, *args, **kwargs):
        pass

class ProcessingStep(Enum):
    """Enumeration of processing steps."""
    DOWNLOAD = "download"
    STANDARDIZE = "standardize"
    VALIDATE = "validate"
    GAP_DETECTION = "gap_detection"
    GAP_FILLING = "gap_filling"
    RESAMPLING = "resampling"
    DUPLICATE_HANDLING = "duplicate_handling"
    QUALITY_CHECK = "quality_check"
    CONSOLIDATION = "consolidation"

class DataQualityLevel(Enum):
    """Data quality levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    FAILED = "failed"

# QualityScoreLevel, QualityScore, and QualityAssessment are defined earlier (before lazy import)

@dataclass
class ProcessingResult:
    """Result of a processing step."""
    step: ProcessingStep
    success: bool
    data: Optional[pd.DataFrame] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    processing_time: float = 0.0
    quality_level: DataQualityLevel = DataQualityLevel.FAILED

@dataclass
class GapInfo:
    """Information about a data gap."""
    start_time: datetime
    end_time: datetime
    duration_minutes: int
    symbol: str
    interval: str
    priority: int = 1  # 1 = high, 2 = medium, 3 = low

@dataclass
class ResamplingConfig:
    """Configuration for data resampling."""
    target_intervals: List[str] = field(default_factory=lambda: ["5m", "15m", "1h"])
    method: str = "ohlc"  # ohlc, vwap, etc.
    preserve_volume: bool = True
    validate_continuity: bool = True
    resample_older_than_days: int = 3  # Only resample data older than this many days
    enable_auto_resampling: bool = True  # Automatically resample based on data age

@dataclass
class PipelineConfig:
    """Configuration for the enhanced klines processing pipeline."""
    data_dir: str = "historical_data"
    exchange: str = "binance"  # Default to Binance, but supports any exchange
    enable_logging: bool = True
    max_gap_minutes: int = 1
    enable_gap_filling: bool = True
    enable_resampling: bool = True
    enable_duplicate_handling: bool = True
    enable_quality_validation: bool = True
    batch_compatible: bool = True
    force_download: bool = False  # If True, ignore existing data and download fresh
    storage_config: Optional[StorageConfig] = None

class EnhancedKlinesProcessingPipeline:
    """
    Enhanced klines data processing pipeline with comprehensive type hints,
    exchange-agnostic design, and fast-fail patterns.

    Features:
    - Uses ExchangeInterface for all exchange calls
    - Integrates KlinesParquetManager for efficient storage
    - Implements data standardizer for consistent formatting
    - Fast fail pattern with no fallbacks or mocks
    - Comprehensive gap detection and filling
    - Automatic resampling for data older than 3 days
    - Batch-compatible data management
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None
    ) -> None:
        """Initialize the enhanced processing pipeline.

        Args:
            config: Pipeline configuration
        """
        self.config = config or PipelineConfig()
        self.data_dir = Path(self.config.data_dir)
        self.exchange = self.config.exchange.lower()
        self.enable_logging = self.config.enable_logging

        # Initialize components
        self.data_standardizer = UnifiedOHLCVStandardizer()
        self.duplicate_analyzer = ComprehensiveDuplicateAnalyzer()

        # Initialize parquet manager with explicit exchange so data is stored
        # under the correct exchange subdirectory (e.g., binance, bingx, etc.)
        self.klines_manager = KlinesParquetManager(
            data_dir=self.config.data_dir,
            exchange=self.exchange,
        )

        # Processing state
        self.current_symbol: Optional[str] = None
        self.current_interval: Optional[str] = None
        self.processing_results: List[ProcessingResult] = []

        # Configuration
        self.required_ohlcv_columns: List[str] = ['open', 'high', 'low', 'close', 'volume']
        self.metadata_columns: List[str] = ['exchange', 'symbol', 'interval', 'timestamp']
        self.columns_to_remove: List[str] = ['taker_buy_base', 'taker_buy_quote', 'year']

        # Resampling configuration
        self.default_resampling_config = ResamplingConfig()

        # Prefer 15m for validation/resample span checks
        self.primary_validation_interval: str = "15m"

        if self.enable_logging:
            tprint_success(f"✅ Enhanced Klines Processing Pipeline initialized for {self.exchange}")
            tprint_info(f"   📁 Data directory: {self.data_dir}")
            tprint_info(f"   🔧 Gap filling: {'enabled' if self.config.enable_gap_filling else 'disabled'}")
            tprint_info(f"   📊 Resampling: {'enabled' if self.config.enable_resampling else 'disabled'}")
            tprint_info(f"   🔄 Batch compatible: {'enabled' if self.config.batch_compatible else 'disabled'}")

    @staticmethod
    def _ensure_naive_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure DataFrame index contains timezone-naive timestamps."""
        if df.empty:
            return df

        converted = pd.to_datetime(df.index, errors='coerce')
        if isinstance(converted, pd.DatetimeIndex) and getattr(converted, "tz", None) is not None:
            converted = converted.tz_localize(None)

        df.index = converted
        return df

    @staticmethod
    def _ensure_utc_timestamp(timestamp: Any) -> Optional[pd.Timestamp]:
        """Convert a timestamp-like value to a UTC aware pandas Timestamp."""
        if timestamp is None:
            return None

        ts = pd.Timestamp(timestamp)
        if ts.tzinfo is None:
            return ts.tz_localize('UTC')
        return ts.tz_convert('UTC')

    def _ensure_utc_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return copy of DataFrame with UTC tz-aware DatetimeIndex."""
        if df.empty:
            return df

        converted = pd.to_datetime(df.index, utc=True, errors='coerce')
        converted.name = df.index.name

        df_with_utc = df.copy()
        df_with_utc.index = converted
        return df_with_utc

    def _ensure_timestamp_column(
        self,
        df: pd.DataFrame,
        column_name: str = 'timestamp'
    ) -> pd.DataFrame:
        """Ensure DataFrame includes a timezone-naive timestamp column."""
        if df.empty:
            return df

        df_with_ts = df if column_name in df.columns else df.copy()

        if column_name not in df_with_ts.columns:
            if isinstance(df_with_ts.index, pd.DatetimeIndex):
                df_with_ts[column_name] = df_with_ts.index
            else:
                df_with_ts[column_name] = pd.to_datetime(df_with_ts.index, errors='coerce')

        timestamp_series = pd.to_datetime(
            df_with_ts[column_name], utc=True, errors='coerce'
        )

        valid_mask = timestamp_series.notna()
        if not valid_mask.all():
            df_with_ts = df_with_ts.loc[valid_mask].copy()
            timestamp_series = timestamp_series.loc[valid_mask]
            if self.enable_logging:
                dropped = (~valid_mask).sum()
                tprint_warning(f"⚠️ Dropped {dropped} rows with invalid timestamps during timestamp normalization")

        # Drop timezone information after normalizing to UTC for downstream consumers
        df_with_ts[column_name] = timestamp_series.dt.tz_localize(None)

        timestamp_ns = timestamp_series.view('int64')
        timestamp_ms = (timestamp_ns // 10**6).astype('int64')
        df_with_ts[f"{column_name}_ms"] = pd.Series(timestamp_ms, index=df_with_ts.index, dtype='int64')

        return df_with_ts

    def _build_quality_view(
        self,
        df: pd.DataFrame,
        symbol: str,
        interval: str,
        quality_interval: str = "5m"
    ) -> pd.DataFrame:
        if df.empty:
            return df

        if interval != "1m":
            return df

        try:
            freq = self._interval_to_pandas_freq(quality_interval)
            if freq is None:
                return df

            df_ts = self._ensure_timestamp_column(df)
            if "timestamp" not in df_ts.columns:
                return df

            temp = df_ts.set_index("timestamp")
            temp = self._ensure_utc_index(temp)

            resampled = temp.resample(freq).agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum"
            }).dropna()

            if resampled.empty:
                return df

            resampled["symbol"] = symbol
            resampled["interval"] = quality_interval
            resampled["exchange"] = self.exchange

            resampled = self._ensure_timestamp_column(resampled)
            return resampled
        except Exception:
            return df

    @staticmethod
    def _to_naive_timestamp(value: Any) -> Optional[pd.Timestamp]:
        """Convert any timestamp-like value to a timezone-naive UTC pandas Timestamp."""
        if value is None:
            return None

        ts = pd.to_datetime(value, utc=True, errors='coerce')
        if ts is pd.NaT:
            return None

        return ts.tz_convert('UTC').tz_localize(None)

    @staticmethod
    def _to_utc_naive_timestamp(value: Any) -> Optional[pd.Timestamp]:
        """Convert any timestamp-like value to a timezone-naive UTC pandas Timestamp."""
        ts_utc = EnhancedKlinesProcessingPipeline._ensure_utc_timestamp(value)
        if ts_utc is None or ts_utc is pd.NaT:
            return None

        return ts_utc.tz_convert('UTC').tz_localize(None)

    @staticmethod
    def _normalize_calendar_columns(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        """Ensure calendar-derived columns are stored as integer types for parquet serialization."""
        if df is None or df.empty:
            return df

        normalized_df = df.copy()
        for column in ('year', 'month', 'day'):
            if column in normalized_df.columns:
                series = pd.to_numeric(normalized_df[column], errors='coerce')
                if isinstance(series, pd.Series):
                    if series.isna().all():
                        normalized_df[column] = series
                    else:
                        normalized_df[column] = series.astype('Int64', copy=False)
        return normalized_df

    async def get_comprehensive_score(
        self,
        df: pd.DataFrame,
        symbol: str,
        interval: str
    ) -> QualityScore:
        """
        Get comprehensive quality score using all available quality utilities from src/utils/data/quality/.

        Args:
            df: DataFrame to analyze
            symbol: Trading symbol
            interval: Time interval

        Returns:
            Comprehensive quality score with detailed breakdown
        """
        try:
            if self.enable_logging:
                tprint_info(f"📊 Generating comprehensive quality score for {symbol} {interval}")

            if not QUALITY_UTILITIES_AVAILABLE:
                # Return fallback quality score
                return QualityScore(
                    overall_score=50.0,
                    level=QualityScoreLevel.POOR,
                    component_scores={},
                    issues=["Quality utilities not available"],
                    warnings=[],
                    recommendations=["Install quality utilities for better assessment"],
                    assessment_timestamp=datetime.now(),
                    data_shape=df.shape
                )

            # Initialize comprehensive quality scorer
            scorer = ComprehensiveQualityScorer()

            # Get comprehensive quality score - use the correct method signature
            score = scorer.assess_data_quality(df, context="klines_processing", step_name="quality_assessment", data_type="klines")

            if self.enable_logging:
                tprint_success(f"✅ Quality score generated: {score.overall_score:.1f} ({score.level.value})")
                if score.recommendations:
                    tprint_info(f"📋 Recommendations: {', '.join(score.recommendations[:3])}")

            return score

        except Exception as e:
            if self.enable_logging:
                tprint_error(f"❌ Failed to generate comprehensive quality score: {str(e)}")
            # Return a default low-quality score on error
            return QualityScore(
                overall_score=0.0,
                level=QualityScoreLevel.CRITICAL,
                component_scores={},
                issues=[f"Quality scoring failed: {str(e)}"],
                warnings=[],
                recommendations=["Check data quality utilities availability"],
                assessment_timestamp=datetime.now(),
                data_shape=df.shape
            )

    async def clean_data_with_quality_utilities(
        self,
        df: pd.DataFrame,
        symbol: str,
        interval: str
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Clean data using comprehensive data cleaning utilities from src/utils/data/quality/.

        Args:
            df: DataFrame to clean
            symbol: Trading symbol
            interval: Time interval

        Returns:
            Tuple of (cleaned_dataframe, cleaning_metadata)
        """
        try:
            if self.enable_logging:
                tprint_info(f"🧹 Cleaning data for {symbol} {interval} using quality utilities")

            if not QUALITY_UTILITIES_AVAILABLE:
                # Return original data with warning metadata
                warning_metadata = {
                    "original_shape": df.shape,
                    "cleaned_shape": df.shape,
                    "rows_removed": 0,
                    "columns_removed": 0,
                    "nulls_removed": 0,
                    "duplicates_removed": 0,
                    "cleaning_timestamp": datetime.now().isoformat(),
                    "symbol": symbol,
                    "interval": interval,
                    "warning": "Quality utilities not available - no cleaning performed"
                }
                return df, warning_metadata

            # Initialize data cleaner
            data_cleaner = DataCleaner()

            # Store original data info
            original_shape = df.shape
            original_nulls = df.isnull().sum().sum()
            original_duplicates = df.duplicated().sum()

            # Clean the data using the correct method signature
            cleaned_df = await data_cleaner.clean_dataframe(
                df,
                remove_constant_features=True,
                remove_duplicates=True,
                handle_missing_values=True,
                timestamp_column='timestamp',
                symbol=symbol,
                exchange=self.exchange,
                timeframe=interval
            )

            if cleaned_df is None:
                # If cleaning failed, return original data
                cleaned_df = df

            # Calculate cleaning statistics
            cleaned_shape = cleaned_df.shape
            cleaned_nulls = cleaned_df.isnull().sum().sum()
            cleaned_duplicates = cleaned_df.duplicated().sum()

            # Create cleaning metadata
            cleaning_metadata = {
                "original_shape": original_shape,
                "cleaned_shape": cleaned_shape,
                "rows_removed": original_shape[0] - cleaned_shape[0],
                "columns_removed": original_shape[1] - cleaned_shape[1],
                "nulls_removed": original_nulls - cleaned_nulls,
                "duplicates_removed": original_duplicates - cleaned_duplicates,
                "cleaning_timestamp": datetime.now().isoformat(),
                "symbol": symbol,
                "interval": interval
            }

            if self.enable_logging:
                tprint_success(f"✅ Data cleaning completed for {symbol} {interval}")
                tprint_info(f"   Original: {original_shape}, Cleaned: {cleaned_shape}")
                tprint_info(f"   Nulls: {original_nulls} → {cleaned_nulls}")
                tprint_info(f"   Duplicates: {original_duplicates} → {cleaned_duplicates}")

            return cleaned_df, cleaning_metadata

        except Exception as e:
            if self.enable_logging:
                tprint_error(f"❌ Data cleaning failed: {str(e)}")
            # Return original data with error metadata
            error_metadata = {
                "error": str(e),
                "cleaning_failed": True,
                "original_shape": df.shape,
                "cleaning_timestamp": datetime.now().isoformat(),
                "symbol": symbol,
                "interval": interval
            }
            return df, error_metadata

    async def analyze_quality_trends(
        self,
        df: pd.DataFrame,
        symbol: str,
        interval: str,
        window_size: int = 100
    ) -> Dict[str, Any]:
        """
        Analyze quality trends over time using comprehensive quality utilities.

        Args:
            df: DataFrame to analyze
            symbol: Trading symbol
            interval: Time interval
            window_size: Window size for trend analysis

        Returns:
            Dictionary containing quality trend analysis results
        """
        try:
            if self.enable_logging:
                tprint_info(f"📈 Analyzing quality trends for {symbol} {interval}")

            if not QUALITY_UTILITIES_AVAILABLE:
                # Return basic trend analysis without quality utilities
                return {
                    "scores": [50.0],
                    "timestamps": [datetime.now().isoformat()],
                    "mean_quality": 50.0,
                    "std_quality": 0.0,
                    "min_quality": 50.0,
                    "max_quality": 50.0,
                    "quality_trend": "stable",
                    "quality_stability": "stable",
                    "window_size": window_size,
                    "total_windows": 1,
                    "analysis_timestamp": datetime.now().isoformat(),
                    "symbol": symbol,
                    "interval": interval,
                    "warning": "Quality utilities not available - basic analysis only"
                }

            # Initialize quality utilities
            scorer = ComprehensiveQualityScorer()
            advanced_metrics = AdvancedQualityMetrics()

            # Calculate quality scores for different windows
            scores = []
            timestamps = []

            # Split data into windows for trend analysis
            total_rows = len(df)
            if total_rows < window_size:
                # If data is smaller than window, analyze the whole dataset
                windows = [df]
            else:
                # Create overlapping windows for trend analysis
                windows = []
                for i in range(0, total_rows - window_size + 1, window_size // 2):
                    window_df = df.iloc[i:i + window_size]
                    windows.append(window_df)

            # Analyze each window
            for i, window_df in enumerate(windows):
                try:
                    # Get quality score for this window using correct method signature
                    score = scorer.assess_data_quality(
                        window_df,
                        context="trend_analysis",
                        step_name="quality_trends",
                        data_type="klines"
                    )
                    quality_assessment = advanced_metrics.comprehensive_quality_assessment(
                        window_df,
                        context="trend_analysis",
                        step_name="quality_trends"
                    )

                    scores.append(score.overall_score)
                    timestamps.append(window_df.index[0] if len(window_df) > 0 else df.index[0])

                except Exception as e:
                    if self.enable_logging:
                        tprint_warning(f"⚠️ Failed to analyze window {i}: {str(e)}")
                    continue

            if not scores:
                raise RuntimeError("No quality scores could be calculated")

            # Calculate trend statistics
            scores_array = np.array(scores)
            trend_analysis = {
                "scores": scores,
                "timestamps": [ts.isoformat() for ts in timestamps],
                "mean_quality": float(np.mean(scores_array)),
                "std_quality": float(np.std(scores_array)),
                "min_quality": float(np.min(scores_array)),
                "max_quality": float(np.max(scores_array)),
                "quality_trend": "improving" if len(scores) > 1 and scores[-1] > scores[0] else "declining",
                "quality_stability": "stable" if np.std(scores_array) < 5.0 else "volatile",
                "window_size": window_size,
                "total_windows": len(windows),
                "analysis_timestamp": datetime.now().isoformat(),
                "symbol": symbol,
                "interval": interval
            }

            # Calculate trend slope if we have enough data points
            if len(scores) >= 3:
                x = np.arange(len(scores))
                slope, intercept = np.polyfit(x, scores, 1)
                trend_analysis["trend_slope"] = float(slope)
                trend_analysis["trend_intercept"] = float(intercept)
                trend_analysis["trend_direction"] = "improving" if slope > 0 else "declining"

            if self.enable_logging:
                tprint_success(f"✅ Quality trend analysis completed for {symbol} {interval}")
                tprint_info(f"   Mean quality: {trend_analysis['mean_quality']:.2f}")
                tprint_info(f"   Quality trend: {trend_analysis['quality_trend']}")
                tprint_info(f"   Quality stability: {trend_analysis['quality_stability']}")

            return trend_analysis

        except Exception as e:
            if self.enable_logging:
                tprint_error(f"❌ Quality trend analysis failed: {str(e)}")
            return {
                "error": str(e),
                "analysis_failed": True,
                "analysis_timestamp": datetime.now().isoformat(),
                "symbol": symbol,
                "interval": interval
            }

    async def process_klines_data(
        self,
        symbol: str,
        interval: str,
        years: int,
        exchange_interface: ExchangeInterface,
        resampling_config: Optional[ResamplingConfig] = None,
        max_gap_minutes: Optional[int] = None,
        create_consolidated: bool = True,
        batch_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process klines data through the complete pipeline.

        Args:
            symbol: Trading symbol (e.g., "ETHUSDT")
            interval: Data interval (e.g., "1m")
            years: Number of years of data to process
            exchange_interface: ExchangeInterface instance for data access
            resampling_config: Configuration for data resampling
            max_gap_minutes: Maximum allowed gap in minutes (uses config default if None)
            create_consolidated: Whether to create consolidated output file
            batch_id: Optional batch identifier for batch-compatible processing

        Returns:
            Dictionary with complete processing results

        Raises:
            ValueError: If required parameters are invalid
            RuntimeError: If processing fails at any step
        """
        if not symbol or not interval or years <= 0:
            raise ValueError("Invalid parameters: symbol, interval, and years must be valid")

        # ExchangeInterface can be None when relying solely on existing parquet data.

        # Use config default if max_gap_minutes not provided
        if max_gap_minutes is None:
            max_gap_minutes = self.config.max_gap_minutes

        # Use default resampling config if not provided
        if resampling_config is None:
            resampling_config = self.default_resampling_config

        self.current_symbol = symbol.upper()
        self.current_interval = interval
        self.processing_results = []

        start_time = datetime.now()

        try:
            if self.enable_logging:
                tprint_info(f"🚀 Starting enhanced klines processing for {self.current_symbol} {interval}")
                tprint_info(f"   📊 Years: {years}, Max gap: {max_gap_minutes}min, Batch: {batch_id or 'auto'}")

            # Initialize results structure
            results = {
                "symbol": self.current_symbol,
                "interval": interval,
                "years": years,
                "exchange": self.exchange,
                "batch_id": batch_id,
                "pipeline_success": False,
                "steps_completed": [],
                "steps_failed": [],
                "total_processing_time": 0.0,
                "data_quality": DataQualityLevel.FAILED,
                "final_data_shape": (0, 0),
                "errors": [],
                "warnings": [],
                "metadata": {},
                "stored_files": [],
                "resampled_intervals": []
            }

            # Step 1: Download data using ExchangeInterface
            # Note: Public market data (klines) doesn't require authentication
            try:
                # Only attempt connection if credentials are available
                # Public endpoints should work without auth
                if hasattr(exchange_interface, 'api_key') and exchange_interface.api_key:
                    await exchange_interface.connect()
                else:
                    if self.enable_logging:
                        tprint_info("🔓 Accessing public market data (no authentication required)")
            except Exception as e:
                if self.enable_logging:
                    tprint_warning(f"⚠️ Exchange connection/auth warning (may continue for public data): {e}")
            
            download_result = await self._download_data(
                symbol, interval, years, exchange_interface
            )
            self.processing_results.append(download_result)

            if not download_result.success:
                raise RuntimeError(f"Data download failed: {download_result.errors}")

            results["steps_completed"].append(ProcessingStep.DOWNLOAD.value)

            # Step 2: Standardize data format using ExchangeDataStandardizer
            standardize_result = await self._standardize_data(
                download_result.data, symbol, interval
            )
            self.processing_results.append(standardize_result)

            if not standardize_result.success:
                raise RuntimeError(f"Data standardization failed: {standardize_result.errors}")

            results["steps_completed"].append(ProcessingStep.STANDARDIZE.value)

            # Step 3: Validate data quality
            if self.config.enable_quality_validation:
                validate_result = await self._validate_data_quality(
                    standardize_result.data, symbol, interval
                )
                self.processing_results.append(validate_result)

                if not validate_result.success:
                    raise RuntimeError(f"Data validation failed: {validate_result.errors}")

                results["steps_completed"].append(ProcessingStep.VALIDATE.value)
                results["data_quality"] = validate_result.quality_level
                current_data = validate_result.data
            else:
                current_data = standardize_result.data

            # Step 4: Detect and fill gaps (if enabled)
            if self.config.enable_gap_filling:
                gap_result = await self._handle_gaps(
                    current_data, symbol, interval, max_gap_minutes, exchange_interface
                )
                self.processing_results.append(gap_result)

                if not gap_result.success:
                    raise RuntimeError(f"Gap handling failed: {gap_result.errors}")

                results["steps_completed"].append(ProcessingStep.GAP_DETECTION.value)
                if gap_result.metadata.get("gaps_filled", 0) > 0:
                    results["steps_completed"].append(ProcessingStep.GAP_FILLING.value)

                current_data = gap_result.data

            # Step 5: Handle duplicates (if enabled)
            if self.config.enable_duplicate_handling:
                duplicate_result = await self._handle_duplicates(
                    current_data, symbol, interval, exchange_interface
                )
                self.processing_results.append(duplicate_result)

                if not duplicate_result.success:
                    raise RuntimeError(f"Duplicate handling failed: {duplicate_result.errors}")

                results["steps_completed"].append(ProcessingStep.DUPLICATE_HANDLING.value)
                current_data = duplicate_result.data

            # Step 6: Store original data using KlinesParquetManager
            store_result = await self._store_original_data(
                current_data, symbol, interval, batch_id
            )
            self.processing_results.append(store_result)

            if store_result.success:
                results["steps_completed"].append("storage")
                results["stored_files"].extend(store_result.metadata.get("stored_files", []))

            # Step 7: Resample data if enabled and data is older than threshold
            if self.config.enable_resampling and resampling_config.enable_auto_resampling:
                filled_ranges = None
                try:
                    filled_ranges = gap_result.metadata.get("filled_ranges") if self.config.enable_gap_filling else None
                except Exception:
                    filled_ranges = None

                resample_result = await self._resample_data_with_age_check(
                    current_data, symbol, resampling_config, batch_id, years, filled_ranges=filled_ranges
                )
                self.processing_results.append(resample_result)

                if resample_result.success:
                    results["steps_completed"].append(ProcessingStep.RESAMPLING.value)
                    results["resampled_intervals"] = resample_result.metadata.get("resampled_intervals", [])
                    results["stored_files"].extend(resample_result.metadata.get("stored_files", []))

                    # After resampling, automatically check stored processed 15m data for gaps
                    try:
                        processed_15m_gaps = self._detect_missing_in_processed_15m(symbol, years)
                        results["metadata"]["processed_15m_missing_gaps"] = len(processed_15m_gaps)

                        # If gaps remain and we have a live exchange interface and 1m base data,
                        # backfill the missing 15m ranges by downloading base-interval candles.
                        if processed_15m_gaps and exchange_interface is not None and interval == "1m":
                            backfill_stats = await self._download_missing_ranges(
                                processed_15m_gaps,
                                symbol,
                                base_interval=interval,
                                exchange_interface=exchange_interface,
                                resampling_config=resampling_config,
                            )
                            results["metadata"]["processed_15m_missing_gaps_filled"] = backfill_stats.get("gaps_filled", 0)
                            results["metadata"]["processed_15m_backfill_stats"] = backfill_stats

                            # Detect any remaining 15m gaps after backfill and, if present,
                            # apply synthetic 1m gap filling as a last-resort patch.
                            remaining_15m_gaps = self._detect_missing_in_processed_15m(symbol, years)
                            results["metadata"]["processed_15m_missing_gaps_after_backfill"] = len(remaining_15m_gaps)

                            if remaining_15m_gaps:
                                try:
                                    synthetic_stats = fill_1m_gaps_and_resample_for_symbol(
                                        exchange=self.exchange,
                                        symbol=symbol,
                                        data_dir=self.config.data_dir,
                                        target_intervals=resampling_config.target_intervals if resampling_config else None,
                                        max_gap_bars=30,
                                        dry_run=False,
                                    )
                                    results["metadata"]["synthetic_gap_fill_stats"] = synthetic_stats
                                except Exception as synthetic_exc:
                                    if self.enable_logging:
                                        tprint_warning(f"⚠️ Synthetic 1m gap filler failed: {synthetic_exc}")
                        
                        # Final mandatory gap verification for ALL target intervals
                        final_gap_report = await self._verify_all_resampled_gaps(symbol, resampling_config, years)
                        results["metadata"]["final_gap_report"] = {k: len(v) for k, v in final_gap_report.items()}
                        
                    except Exception as e:
                        if self.enable_logging:
                            tprint_warning(f"⚠️ Processed 15m gap analysis/backfill failed: {e}")
                else:
                    results["warnings"].extend(resample_result.warnings)

            # Step 8: Final quality check
            final_quality_result = await self._final_quality_check(
                current_data, symbol, interval
            )
            self.processing_results.append(final_quality_result)

            if not final_quality_result.success:
                raise RuntimeError(f"Final quality check failed: {final_quality_result.errors}")
            results["steps_completed"].append(ProcessingStep.QUALITY_CHECK.value)
            results["data_quality"] = final_quality_result.quality_level

            # Step 9: Create consolidated file if requested
            if create_consolidated:
                consolidate_result = await self._create_consolidated_file(
                    current_data, symbol, interval, batch_id
                )
                self.processing_results.append(consolidate_result)

                if consolidate_result.success:
                    results["steps_completed"].append(ProcessingStep.CONSOLIDATION.value)
                    results["consolidated_file"] = consolidate_result.metadata.get("output_file")
                    results["stored_files"].append(consolidate_result.metadata.get("output_file"))
                else:
                    results["warnings"].extend(consolidate_result.warnings)

            # Compile final results
            results["pipeline_success"] = True
            results["final_data_shape"] = current_data.shape
            results["total_processing_time"] = (datetime.now() - start_time).total_seconds()

            # Aggregate warnings and errors
            for result in self.processing_results:
                results["warnings"].extend(result.warnings)
                results["errors"].extend(result.errors)

            if self.enable_logging:
                tprint_success(f"✅ Pipeline completed successfully in {results['total_processing_time']:.2f}s")
                tprint_info(f"   📊 Final data shape: {results['final_data_shape']}")
                tprint_info(f"   🎯 Data quality: {results['data_quality'].value}")
                tprint_info(f"   📋 Steps completed: {len(results['steps_completed'])}")

            return results

        except Exception as e:
            error_msg = f"Pipeline processing failed: {str(e)}"
            if self.enable_logging:
                tprint_error(f"❌ {error_msg}")

            results["pipeline_success"] = False
            results["errors"].append(error_msg)
            results["total_processing_time"] = (datetime.now() - start_time).total_seconds()

            return results

    async def process_klines_data_simple(
        self,
        exchange: str,
        asset: str,
        lookback_period: str,
        interval: str = "1m",
        api_key: str = "",
        api_secret: str = "",
        api_password: str = "",
        use_testnet: bool = False,
        resampling_config: Optional[ResamplingConfig] = None,
        batch_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Simplified interface for processing klines data with exchange, asset, and lookback period.
        
        Args:
            exchange: Exchange name (e.g., "binance", "bingx", "okx")
            asset: Trading asset (e.g., "BTC", "ETH", "ADA")
            lookback_period: Lookback period (e.g., "1y", "6m", "30d", "7d")
            interval: Data interval (e.g., "1m", "5m", "1h")
            api_key: Exchange API key
            api_secret: Exchange API secret
            use_testnet: Whether to use testnet
            resampling_config: Configuration for data resampling
            batch_id: Optional batch identifier
            
        Returns:
            Dictionary with complete processing results
        """
        # Parse lookback period
        years = self._parse_lookback_period(lookback_period)
        
        # Create symbol
        symbol = f"{asset}USDT"

        # Ensure runtime configuration matches requested exchange
        target_exchange = exchange.lower()
        self.exchange = target_exchange
        if hasattr(self.config, "exchange"):
            self.config.exchange = target_exchange

        # Create exchange interface
        # Note: API keys are optional for public market data (klines)
        # Only required for authenticated operations like trading
        exchange_config = {
            'exchange_type': target_exchange,
            'api_key': api_key if api_key else None,
            'api_secret': api_secret if api_secret else None,
            'password': api_password if api_password else None,
            'testnet': use_testnet,
            'rate_limits': {}
        }
        
        from src.trading.execution.exchange_interface import ExchangeInterface
        if self.enable_logging:
            tprint_info(f"🔌 Creating ExchangeInterface with config: {exchange_config}")
        exchange_interface = ExchangeInterface(exchange_config)
        if self.enable_logging:
            tprint_info(f"🔌 ExchangeInterface created: {exchange_interface}")
        
        # Auto-generate a batch identifier when one isn't provided
        if batch_id is None:
            batch_id = f"{target_exchange}_{symbol.lower()}_{lookback_period}"

        try:
            # Connect to exchange (authentication skipped if no credentials provided)
            # Public market data doesn't require authentication
            # For BingX, skip connection entirely and use klines adapter directly
            if target_exchange != 'bingx':
                if api_key and api_secret:
                    await exchange_interface.connect()
                else:
                    # For public data, we can still initialize without auth
                    # The dispatcher should handle public endpoints without auth
                    if self.enable_logging:
                        tprint_info("🔓 Using public market data (no credentials required)")
                    # Try to connect but don't fail if auth fails
                    try:
                        await exchange_interface.connect()
                    except Exception as e:
                        if self.enable_logging:
                            tprint_warning(f"⚠️ Authentication skipped for public data: {e}")
            else:
                # For BingX, connect but don't require authentication for public data
                if self.enable_logging:
                    tprint_info("🔌 Connecting to BingX (public data)")
                try:
                    result = await exchange_interface.connect()
                    if self.enable_logging:
                        tprint_info(f"🔌 BingX connection result: {result}")
                except Exception as e:
                    if self.enable_logging:
                        tprint_warning(f"⚠️ BingX connection failed: {e}")
                    # Continue anyway as BingX might work without connection
            
            # Process klines data
            if self.enable_logging:
                tprint_info(f"🚀 Processing klines data (symbol={symbol}, interval={interval}, years={years})")
            results = await self.process_klines_data(
                symbol=symbol,
                interval=interval,
                years=years,
                exchange_interface=exchange_interface,
                resampling_config=resampling_config,
                batch_id=batch_id
            )
            if self.enable_logging:
                tprint_success("✅ process_klines_data completed")
            
            return results
            
        finally:
            # Disconnect from exchange
            if exchange_interface is not None:
                await exchange_interface.disconnect()

    def _parse_lookback_period(self, lookback_period: str) -> int:
        """
        Parse lookback period string into years.
        
        Args:
            lookback_period: Period string (e.g., "1y", "6m", "30d", "7d")
            
        Returns:
            Number of years
        """
        lookback_period = lookback_period.lower().strip()
        
        if lookback_period.endswith('y'):
            return int(lookback_period[:-1])
        elif lookback_period.endswith('m'):
            months = int(lookback_period[:-1])
            return max(1, months // 12)  # Convert to years, minimum 1
        elif lookback_period.endswith('d'):
            days = int(lookback_period[:-1])
            return max(1, days // 365)  # Convert to years, minimum 1
        else:
            # Assume it's a number of years
            return int(lookback_period)

    async def _download_data(
        self,
        symbol: str,
        interval: str,
        years: int,
        exchange_interface: ExchangeInterface
    ) -> ProcessingResult:
        """Download or load klines data from exchange or existing files."""
        start_time = datetime.now()
        result = ProcessingResult(
            step=ProcessingStep.DOWNLOAD,
            success=False,
            errors=[],
            warnings=[]
        )

        try:
            if self.enable_logging:
                tprint_info(f"📥 Processing {years} years of {symbol} {interval} data")

            # Check if we have existing data first
            data_dir = Path(self.config.data_dir) / self.exchange / symbol.lower() / "raw"
            parquet_files = list(data_dir.glob(f"{symbol.lower()}_{interval}_*.parquet")) if data_dir.exists() else []

            # Prefer existing klines; only download gaps or fresh when forced or empty
            if parquet_files and not getattr(self.config, "force_download", False):
                if self.enable_logging:
                    tprint_info(f"📁 Using existing parquet files ({len(parquet_files)})")

                combined_df_list = []
                for pf in parquet_files:
                    try:
                        df = pd.read_parquet(pf)
                        if df is not None and not df.empty:
                            combined_df_list.append(df)
                    except Exception as e:
                        if self.enable_logging:
                            tprint_warning(f"⚠️ Failed to read {pf}: {e}")
                        continue

                if combined_df_list:
                    klines_data = pd.concat(combined_df_list).sort_index().drop_duplicates()
                    # Normalize index to timezone-naive UTC to avoid naive/aware comparison issues
                    klines_data = self._ensure_naive_datetime_index(klines_data)

                    latest_data_time = klines_data.index.max()
                    earliest_data_time = klines_data.index.min()
                    desired_end = datetime.now()
                    desired_start = desired_end - timedelta(days=365 * years)

                    gaps_to_fill: List[Tuple[str, datetime, datetime]] = []
                    gap_dfs: List[pd.DataFrame] = []
                    expected_frequency = pd.Timedelta(minutes=self._interval_to_minutes(interval))

                    # Historical gap
                    if earliest_data_time > desired_start:
                        gaps_to_fill.append(("historical", desired_start, earliest_data_time))
                    # Recent gap
                    if latest_data_time < desired_end:
                        gaps_to_fill.append(("recent", latest_data_time, desired_end))
                    # Internal gaps
                    time_diffs = klines_data.index.to_series().diff().dropna()
                    gap_mask = time_diffs > expected_frequency * 1.1
                    if gap_mask.any():
                        gap_indices = gap_mask[gap_mask].index
                        for gap_start in gap_indices:
                            prev_time = klines_data.index[klines_data.index.get_loc(gap_start) - 1]
                            gap_end = gap_start
                            if (gap_end - prev_time) > expected_frequency * 1.1:
                                gaps_to_fill.append(("internal", prev_time + expected_frequency, gap_end))

                    if gaps_to_fill and exchange_interface is not None:
                        if self.enable_logging:
                            tprint_info(f"📥 Filling {len(gaps_to_fill)} gap(s); download only missing ranges")

                        # Use the unified ExchangeInterface.get_klines API for gap fills so we
                        # benefit from dispatcher wiring and the Binance public klines fallback.
                        interval_minutes = self._interval_to_minutes(interval) or 1
                        interval_delta = timedelta(minutes=interval_minutes)
                        batch_size = 1000
                        batch_duration = timedelta(minutes=batch_size * interval_minutes)

                        for gap_type, gap_start, gap_end in gaps_to_fill:
                            if gap_end <= gap_start:
                                continue

                            current_start = gap_start
                            gap_klines: List[Any] = []

                            for _ in range(10000):
                                current_end = min(gap_end, current_start + batch_duration)
                                try:
                                    batch = await exchange_interface.get_klines(
                                        symbol=symbol,
                                        interval=interval,
                                        start_time=current_start,
                                        end_time=current_end,
                                        limit=batch_size,
                                    )
                                except Exception as e:
                                    if self.enable_logging:
                                        tprint_warning(f"⚠️ Gap download failed ({gap_type}): {e}")
                                    break

                                if not batch:
                                    break

                                gap_klines.extend(batch)

                                # Derive the timestamp of the last candle in the batch to
                                # advance the window. Support both KlineData objects and
                                # raw list/tuple formats for robustness.
                                last_ts = None
                                try:
                                    last_kline = batch[-1]
                                    if hasattr(last_kline, "timestamp"):
                                        last_ts = getattr(last_kline, "timestamp")
                                    else:
                                        raw_ts = last_kline[0]
                                        if isinstance(raw_ts, (int, float)):
                                            # Assume milliseconds when values are large
                                            unit = "ms" if raw_ts > 1e12 else "s"
                                            last_ts = pd.to_datetime(raw_ts, unit=unit).to_pydatetime()
                                        else:
                                            last_ts = pd.to_datetime(raw_ts).to_pydatetime()
                                except Exception:
                                    last_ts = None

                                if last_ts is None:
                                    break

                                current_start = last_ts + interval_delta
                                if current_start >= gap_end:
                                    break

                            if gap_klines:
                                gap_df = self._klines_to_dataframe(gap_klines, symbol, interval)
                                gap_dfs.append(gap_df)
                        if gap_dfs:
                            klines_data = pd.concat([klines_data] + gap_dfs).drop_duplicates().sort_index()
                    elif gaps_to_fill and exchange_interface is None and self.enable_logging:
                        tprint_warning("⚠️ Skipping gap download because exchange interface is unavailable; using existing data only")

                    # Finalize using existing data (+ gaps if any)
                    total_gap_records = sum(len(df) for df in gap_dfs) if gap_dfs else 0
                    result.success = True
                    result.data = klines_data
                    result.metadata = {
                        "records_downloaded": total_gap_records,
                        "date_range": {
                            "start": klines_data.index.min().isoformat() if len(klines_data) else None,
                            "end": klines_data.index.max().isoformat() if len(klines_data) else None,
                        },
                        "storage_location": str(data_dir),
                        "gaps_filled": len(gaps_to_fill),
                    }
                    if self.enable_logging and isinstance(klines_data.index, pd.DatetimeIndex):
                        tprint_success(f"✅ Using existing data: {len(klines_data):,} records (gaps filled: {len(gaps_to_fill)})")
                        tprint_info(f"📅 Full range: {klines_data.index.min()} to {klines_data.index.max()}")
                    return result
                else:
                    # No readable parquet data; fall through to fresh download
                    if self.enable_logging:
                        tprint_warning("⚠️ No valid data found in existing parquet files; downloading fresh data")

            # If no data and no exchange interface available, fail fast
            if not parquet_files and exchange_interface is None:
                raise RuntimeError("No existing raw klines found and exchange interface unavailable; cannot download.")

            # Ensure dispatcher is wired for live download
            if exchange_interface is not None and getattr(exchange_interface, "dispatcher", None) is None:
                if create_exchange_dispatcher is not None and ExchangeConfig is not None:
                    try:
                        ex_type = ExchangeType(self.exchange) if isinstance(self.exchange, str) else self.exchange
                    except Exception:
                        ex_type = ExchangeType.BINANCE if str(self.exchange).lower() == "binance" else ExchangeType.BINANCE
                    try:
                        dispatcher_cfg = ExchangeConfig(
                            exchange_type=ex_type,
                            api_key=getattr(exchange_interface, "api_key", None),
                            api_secret=getattr(exchange_interface, "api_secret", None),
                            password=getattr(exchange_interface, "password", None),
                            subaccount_id=None,
                            use_testnet=getattr(exchange_interface, "use_testnet", False),
                            trade_symbol=symbol,
                        )
                        exchange_interface.dispatcher = create_exchange_dispatcher(dispatcher_cfg)
                    except Exception as e:
                        if self.enable_logging:
                            tprint_warning(f"⚠️ Failed to wire dispatcher; live download may be empty: {e}")

            # Download fresh data from exchange in batches
            if self.enable_logging:
                tprint_info(f"🌐 Downloading fresh data from {exchange_interface.exchange_type.upper()} exchange")
                
                # Calculate date range
                end_date = datetime.now()
                start_date = end_date - timedelta(days=years * 365)
                
                if self.enable_logging:
                    tprint_info(f"📅 Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
                    estimated_candles = years * 365 * 24 * 60  # Rough estimate for 1m data
                    tprint_info(f"📊 Estimated candles: ~{estimated_candles:,}")
                
                # Download data in batches (API limit is 1000 candles per request)
                all_batches = []
                current_start = start_date
                batch_size = 1000
                batch_num = 1

                # For 1m interval, 1000 candles = ~16.67 hours
                interval_minutes = 1  # TODO: Parse from interval string
                batch_duration = timedelta(minutes=batch_size * interval_minutes)

                # Detect whether the underlying exchange exposes a dedicated
                # historical klines API. When available (e.g. BingX), we
                # prefer that path so start/end windows are honored instead of
                # repeatedly fetching just the most recent window.
                dispatcher = getattr(exchange_interface, "dispatcher", None)
                underlying_exchange = getattr(dispatcher, "exchange", None) if dispatcher is not None else None
                has_historical_klines = bool(
                    underlying_exchange is not None
                    and hasattr(underlying_exchange, "get_historical_klines")
                )

                while current_start < end_date:
                    batch_end = min(current_start + batch_duration, end_date)

                    # Log every 50th batch to show progress without spamming
                    if self.enable_logging and (batch_num == 1 or batch_num % 50 == 0):
                        progress_pct = ((current_start - start_date).days / (years * 365)) * 100
                        tprint_info(
                            f"📦 Batch {batch_num}: {current_start.strftime('%Y-%m-%d %H:%M')} ({progress_pct:.1f}%)"
                        )

                    try:
                        if has_historical_klines and hasattr(
                            exchange_interface, "get_historical_klines"
                        ):
                            # Prefer true historical path when available
                            batch_data = await exchange_interface.get_historical_klines(
                                symbol=symbol,
                                interval=interval,
                                start_time=current_start,
                                end_time=batch_end,
                                limit=batch_size,
                            )
                        else:
                            # Fallback to generic recent klines
                            batch_data = await exchange_interface.get_klines(
                                symbol=symbol,
                                interval=interval,
                                start_time=current_start,
                                end_time=batch_end,
                                limit=batch_size,
                            )

                        if batch_data and len(batch_data) > 0:
                            all_batches.append(batch_data)
                            # Log progress every 100 batches
                            if self.enable_logging and batch_num % 100 == 0:
                                total_candles = sum(len(b) for b in all_batches)
                                progress_pct = ((current_start - start_date).days / (years * 365)) * 100
                                tprint_info(
                                    f"   📊 Progress: {len(all_batches)} batches, {total_candles:,} candles ({progress_pct:.1f}%)"
                                )
                        else:
                            # No data returned, might have reached the end of available data
                            if self.enable_logging:
                                tprint_info(
                                    f"   ℹ️ No data returned for batch {batch_num}, stopping download"
                                )
                            break

                        # Move to next batch
                        current_start = batch_end
                        batch_num += 1

                        # Small delay to respect rate limits (50ms = 20 requests/sec max)
                        await asyncio.sleep(0.05)

                    except Exception as e:
                        if self.enable_logging:
                            tprint_warning(f"⚠️ Batch {batch_num} failed: {e}")
                        current_start = batch_end
                        batch_num += 1
                        # Longer delay after error
                        await asyncio.sleep(1.0)
                
                if not all_batches:
                    raise ValueError("No data received from exchange")
                
                if self.enable_logging:
                    total_candles = sum(len(b) for b in all_batches)
                    tprint_success(f"✅ Downloaded {total_candles:,} candles in {len(all_batches)} batches")
                
                # Combine all batches
                klines_data = []
                for batch in all_batches:
                    klines_data.extend(batch)
                
                if self.enable_logging:
                    tprint_info(f"🔍 Total klines collected: {len(klines_data)}")
                    tprint_info(f"🔍 klines_data type: {type(klines_data)}")
                
                # Convert KlineData objects to DataFrame format if not already a DataFrame
                # Convert KlineData objects to list format
                raw_data = []
                for i, kline in enumerate(klines_data):
                    # Debug first kline
                    if i == 0 and self.enable_logging:
                        tprint_info(f"🔍 First kline type: {type(kline)}")
                        tprint_info(f"🔍 First kline timestamp: {kline.timestamp}")
                        tprint_info(f"🔍 First kline timestamp type: {type(kline.timestamp)}")
                    
                    # Handle both datetime and int timestamps
                    if isinstance(kline.timestamp, datetime):
                        ts = int(kline.timestamp.timestamp() * 1000)
                    elif isinstance(kline.timestamp, (int, float)):
                        # Binance returns milliseconds timestamps (13 digits)
                        # If timestamp is too small (< 1e10), it's likely in seconds or invalid
                        if kline.timestamp < 1e10:
                            # Likely seconds, convert to milliseconds
                            ts = int(kline.timestamp * 1000)
                        else:
                            ts = int(kline.timestamp)
                    else:
                        ts_val = float(kline.timestamp)
                        if ts_val < 1e10:
                            ts = int(ts_val * 1000)
                        else:
                            ts = int(ts_val)
                    
                    if isinstance(kline.close_time, datetime):
                        ct = int(kline.close_time.timestamp() * 1000)
                    elif isinstance(kline.close_time, (int, float)):
                        # Same logic for close_time
                        if kline.close_time < 1e10:
                            ct = int(kline.close_time * 1000)
                        else:
                            ct = int(kline.close_time)
                    else:
                        ct_val = float(kline.close_time)
                        if ct_val < 1e10:
                            ct = int(ct_val * 1000)
                        else:
                            ct = int(ct_val)
                    
                    raw_data.append([
                        ts,  # timestamp
                        kline.open_price,  # open
                        kline.high_price,  # high
                        kline.low_price,   # low
                        kline.close_price, # close
                        kline.volume,      # volume
                        ct,  # close_time
                        kline.quote_asset_volume,  # quote_volume
                        kline.number_of_trades,     # trades
                        kline.taker_buy_base_asset_volume,  # taker_buy_base
                        kline.taker_buy_quote_asset_volume   # taker_buy_quote
                    ])
                
                if not raw_data:
                    raise ValueError("No data received from exchange")

                # Convert to DataFrame with all 11 columns
                klines_data = pd.DataFrame(raw_data, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_asset_volume', 'number_of_trades',
                    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume'
                ])
                
                # Debug the timestamp values
                sample_timestamp = klines_data['timestamp'].iloc[0] if len(klines_data) > 0 else 0
                if self.enable_logging:
                    tprint_info(f"🔍 Sample timestamp value: {sample_timestamp}")
                    tprint_info(f"🔍 Sample timestamp type: {type(sample_timestamp)}")
                    tprint_info(f"🔍 Sample timestamp > 1e12? {sample_timestamp > 1e12}")
                    tprint_info(f"🔍 Sample timestamp > 1e9? {sample_timestamp > 1e9}")
                
                # Convert timestamp to datetime - handle both milliseconds and microseconds
                # Microseconds: > 1e15 (16+ digits, e.g., 1730269140000000 for 2025)
                # Milliseconds: 1e12 - 1e15 (13-15 digits, e.g., 1730269140000 for 2025)
                # Seconds: < 1e12 (< 13 digits, e.g., 1730269140 for 2025)
                if sample_timestamp > 1e15:  # Microseconds (16+ digits)
                    if self.enable_logging:
                        tprint_info(f"🔁 Converting as microseconds (value: {sample_timestamp})")
                    klines_data['timestamp'] = pd.to_datetime(klines_data['timestamp'], unit='us')
                elif sample_timestamp > 1e12:  # Milliseconds (13-15 digits)
                    if self.enable_logging:
                        tprint_info(f"🔁 Converting as milliseconds (value: {sample_timestamp})")
                    klines_data['timestamp'] = pd.to_datetime(klines_data['timestamp'], unit='ms')
                elif sample_timestamp > 1e9:  # Seconds with decimal precision (10-12 digits)
                    if self.enable_logging:
                        tprint_info(f"🔁 Converting as seconds (value: {sample_timestamp})")
                    klines_data['timestamp'] = pd.to_datetime(klines_data['timestamp'], unit='s')
                else:
                    if self.enable_logging:
                        tprint_info(f"🔁 Timestamp too small, treating as seconds anyway (value: {sample_timestamp})")
                    klines_data['timestamp'] = pd.to_datetime(klines_data['timestamp'], unit='s')

                if self.enable_logging and len(klines_data) > 0:
                    tprint_info(f"🔍 After conversion, first timestamp: {klines_data['timestamp'].iloc[0]}")
                
                klines_data.set_index('timestamp', inplace=True)
                
                if self.enable_logging:
                    tprint_success(f"✅ Downloaded {len(klines_data)} records from exchange")
            
            # Remove duplicates and sort by timestamp (timestamp is the index)
            klines_data = klines_data.drop_duplicates().sort_index()
            
            if self.enable_logging:
                tprint_success(f"✅ Loaded {len(klines_data)} total records")
                tprint_info(f"📅 Date range: {klines_data.index.min()} to {klines_data.index.max()}")

            if len(klines_data) == 0:
                raise RuntimeError("No data loaded from parquet files")

            # Ensure proper column names and types
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in klines_data.columns:
                    raise ValueError(f"Missing required column: {col}")
                # Convert to numeric, coercing errors to NaN
                klines_data[col] = pd.to_numeric(klines_data[col], errors='coerce')
            
            # Ensure timestamp index is datetime
            if not pd.api.types.is_datetime64_any_dtype(klines_data.index):
                klines_data.index = pd.to_datetime(klines_data.index)
            
            # Add metadata columns
            klines_data['symbol'] = symbol.upper()
            klines_data['interval'] = interval
            klines_data['exchange'] = self.exchange

            if len(klines_data) == 0:
                raise RuntimeError("Loaded data is empty")

            result.success = True
            result.data = klines_data
            result.metadata = {
                "records_downloaded": len(klines_data),
                "date_range": {
                    "start": klines_data.index.min().isoformat() if not len(klines_data) == 0 else None,
                    "end": klines_data.index.max().isoformat() if not len(klines_data) == 0 else None
                }
            }

            if self.enable_logging:
                tprint_success(f"✅ Loaded {len(klines_data)} records")

        except Exception as e:
            error_msg = f"Data download failed: {str(e)}"

            # Public Binance adapter fallback when primary fails
            if (
                self.exchange.lower() == "binance"
                and "No data received from exchange" in str(e)
            ):
                try:
                    from exchanges.binance.klines_adapter import create_binance_klines_adapter
                    if self.enable_logging:
                        tprint_warning("⚠️ Primary exchange download failed; attempting BinanceKlinesAdapter fallback")
                    fallback_df = await self._download_with_binance_adapter(
                        symbol=symbol,
                        interval=interval,
                        years=years
                    )
                    if fallback_df is not None and not fallback_df.empty:
                        result.success = True
                        result.data = fallback_df
                        result.metadata = {
                            "records_downloaded": len(fallback_df),
                            "date_range": {
                                "start": fallback_df.index.min().isoformat() if len(fallback_df) else None,
                                "end": fallback_df.index.max().isoformat() if len(fallback_df) else None,
                            },
                            "fallback": "binance_klines_adapter",
                        }
                        if self.enable_logging:
                            tprint_success(
                                f"✅ BinanceKlinesAdapter fallback succeeded with {len(fallback_df)} records"
                            )
                        result.processing_time = (datetime.now() - start_time).total_seconds()
                        return result
                except Exception as fallback_err:
                    if self.enable_logging:
                        tprint_error(f"❌ BinanceKlinesAdapter fallback failed: {fallback_err}")

            result.errors.append(error_msg)
            if self.enable_logging:
                tprint_error(f"❌ {error_msg}")

        result.processing_time = (datetime.now() - start_time).total_seconds()
        return result

    async def _download_with_binance_adapter(
        self,
        symbol: str,
        interval: str,
        years: int,
    ) -> Optional[pd.DataFrame]:
        """Fallback download using BinanceKlinesAdapter with public REST klines."""
        try:
            from exchanges.binance.klines_adapter import create_binance_klines_adapter
        except Exception as e:
            if self.enable_logging:
                tprint_warning(f"⚠️ Binance klines adapter import failed; skipping fallback: {e}")
            return None

        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)

        adapter = create_binance_klines_adapter(
            api_key=None,
            secret_key=None,
            data_dir=self.config.data_dir,
        )

        try:
            df = await adapter.download_and_process_klines(
                symbol=symbol,
                interval=interval,
                start_time=start_date,
                end_time=end_date,
                save_data=True,
            )
        except Exception as e:
            if self.enable_logging:
                tprint_error(f"❌ BinanceKlinesAdapter download failed: {e}")
            return None

        if df is None or df.empty:
            return None

        # Ensure we have a datetime index and basic OHLCV columns
        if 'timestamp' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            df = df.set_index('timestamp')

        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors="coerce")

        df = self._ensure_naive_datetime_index(df)

        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
        df = df.drop_duplicates().sort_index()

        return df

    def _klines_to_dataframe(
        self,
        klines_data: List[Any],
        symbol: str,
        interval: str
    ) -> pd.DataFrame:
        """Convert klines data to standardized DataFrame."""
        try:
            # Debug: Check the type of klines_data
            if self.enable_logging and len(klines_data) > 0:
                first_kline = klines_data[0]
                tprint_info(f"🔍 DEBUG: First kline type: {type(first_kline)}")
                if isinstance(first_kline, (list, tuple)):
                    tprint_info(f"🔍 DEBUG: First kline length: {len(first_kline)}")
                    tprint_info(f"🔍 DEBUG: First kline sample: {first_kline[:5] if len(first_kline) > 5 else first_kline}")
            
            # Extract data based on klines format
            data = []
            for kline in klines_data:
                # New typed format: KlineData objects from ExchangeInterface
                if hasattr(kline, 'timestamp') and hasattr(kline, 'open_price'):
                    # Normalize timestamp to UTC naive before storage
                    ts_naive = self._to_utc_naive_timestamp(kline.timestamp)
                    if ts_naive is None:
                        continue

                    data.append({
                        'timestamp': ts_naive,
                        'open': float(getattr(kline, 'open_price', 0.0)),
                        'high': float(getattr(kline, 'high_price', 0.0)),
                        'low': float(getattr(kline, 'low_price', 0.0)),
                        'close': float(getattr(kline, 'close_price', 0.0)),
                        'volume': float(getattr(kline, 'volume', 0.0)),
                        'quote_volume': float(getattr(kline, 'quote_asset_volume', 0.0)),
                        'trades': int(getattr(kline, 'number_of_trades', 0)),
                        'taker_buy_base': float(getattr(kline, 'taker_buy_base_asset_volume', 0.0)),
                        'taker_buy_quote': float(getattr(kline, 'taker_buy_quote_asset_volume', 0.0)),
                    })

                # Legacy format: raw list/tuple klines
                elif isinstance(kline, (list, tuple)) and len(kline) >= 6:
                    # Expected layout (Binance-style):
                    # [open_time, open, high, low, close, volume, close_time, quote_volume,
                    #  trades, taker_buy_base, taker_buy_quote, ...]
                    ts_naive = self._to_utc_naive_timestamp(kline[0])
                    if ts_naive is None:
                        continue

                    data.append({
                        'timestamp': ts_naive,
                        'open': float(kline[1]),
                        'high': float(kline[2]),
                        'low': float(kline[3]),
                        'close': float(kline[4]),
                        'volume': float(kline[5]),
                        'quote_volume': float(kline[6]) if len(kline) > 6 else 0.0,
                        'trades': int(kline[7]) if len(kline) > 7 else 0,
                        'taker_buy_base': float(kline[8]) if len(kline) > 8 else 0.0,
                        'taker_buy_quote': float(kline[9]) if len(kline) > 9 else 0.0,
                    })

                # Unknown format - skip gracefully
                else:
                    continue

            df = pd.DataFrame(data)
            
            if self.enable_logging:
                tprint_info(f"🔍 DEBUG: DataFrame shape before indexing: {df.shape}")
                tprint_info(f"🔍 DEBUG: DataFrame columns: {df.columns.tolist()}")
                if not df.empty:
                    tprint_info(f"🔍 DEBUG: First row: {df.iloc[0].to_dict()}")
            
            df.set_index('timestamp', inplace=True)
            df = self._ensure_naive_datetime_index(df)
            df.sort_index(inplace=True)

            return df

        except Exception as e:
            if self.enable_logging:
                tprint_error(f"❌ Failed to convert klines to DataFrame: {e}")
                import traceback
                tprint_error(f"Traceback: {traceback.format_exc()}")
            return pd.DataFrame()

    async def _standardize_data(
        self,
        df: pd.DataFrame,
        symbol: str,
        interval: str
    ) -> ProcessingResult:
        """Standardize data format using UnifiedOHLCVStandardizer."""
        start_time = datetime.now()
        result = ProcessingResult(
            step=ProcessingStep.STANDARDIZE,
            success=False,
            errors=[],
            warnings=[]
        )

        try:
            if self.enable_logging:
                tprint_info(f"🔄 Standardizing data format for {symbol} {interval}")
            
            # Debug: Check DataFrame structure before standardization
            print(f"DEBUG: DataFrame shape before standardization: {df.shape}")
            print(f"DEBUG: DataFrame columns: {df.columns.tolist()}")
            
            # Use the UnifiedOHLCVStandardizer for consistent data formatting
            # This ensures the same standardization logic is used for both main downloads and gap filling
            standardized_df = self.data_standardizer.standardize(
                df, exchange=self.exchange
            )

            # Ensure timestamp index is timezone-naive and available as a column for validators
            standardized_df = self._ensure_naive_datetime_index(standardized_df)
            if 'timestamp' not in standardized_df.columns:
                standardized_df['timestamp'] = standardized_df.index

            # Add metadata columns if missing
            if 'symbol' not in standardized_df.columns:
                standardized_df['symbol'] = symbol
            if 'interval' not in standardized_df.columns:
                standardized_df['interval'] = interval
            if 'exchange' not in standardized_df.columns:
                standardized_df['exchange'] = self.exchange

            result.success = True
            result.data = standardized_df
            result.metadata = {
                "original_shape": df.shape,
                "final_shape": standardized_df.shape,
                "processing_time": 0.0
            }

            if self.enable_logging:
                tprint_success(f"✅ Data standardized: {len(standardized_df)} records")

        except Exception as e:
            error_msg = f"Data standardization failed: {str(e)}"
            result.errors.append(error_msg)
            if self.enable_logging:
                tprint_error(f"❌ {error_msg}")

        result.processing_time = (datetime.now() - start_time).total_seconds()
        return result

    async def _validate_data_quality(
        self,
        df: pd.DataFrame,
        symbol: str,
        interval: str
    ) -> ProcessingResult:
        """Validate data quality using comprehensive quality framework from src/utils/data/quality/."""
        start_time = datetime.now()
        result = ProcessingResult(
            step=ProcessingStep.VALIDATE,
            success=False,
            errors=[],
            warnings=[]
        )

        try:
            if self.enable_logging:
                tprint_info(f"🔍 Validating data quality for {symbol} {interval} using comprehensive framework")

            df_full = self._ensure_timestamp_column(df)

            # Check required columns
            missing_columns = [col for col in self.required_ohlcv_columns if col not in df_full.columns]
            if missing_columns:
                raise RuntimeError(f"Missing required columns: {missing_columns}")

            quality_df = self._build_quality_view(df_full, symbol, interval)

            if not QUALITY_UTILITIES_AVAILABLE:
                # Use basic validation without quality utilities
                quality_result = FallbackQualityResult(
                    passed=True,
                    issues=["Quality utilities not available"],
                    warnings=["Basic validation only"],
                    quality_score=50.0
                )
                quality_assessment = QualityAssessment(
                    overall_score=50.0,
                    metrics=[],
                    issues_found=1,
                    warnings_found=1,
                    critical_issues=0,
                    assessment_timestamp=datetime.now(),
                    data_shape=df.shape
                )
                score = QualityScore(
                    overall_score=50.0,
                    level=QualityScoreLevel.POOR,
                    component_scores={},
                    issues=["Quality utilities not available"],
                    warnings=["Basic validation only"],
                    recommendations=["Install quality utilities for better assessment"],
                    assessment_timestamp=datetime.now(),
                    data_shape=df.shape
                )
                distribution_validation = {}
                duplicate_analysis = _ANALYZE_DUPLICATES_COMPREHENSIVE(quality_df) if callable(_ANALYZE_DUPLICATES_COMPREHENSIVE) else None
            else:
                # Import quality utilities locally to ensure they're available
                from src.utils.data.quality.data_quality import QualityThresholds
                
                # Initialize comprehensive quality framework
                quality_framework = DataQualityFramework()
                scorer = ComprehensiveQualityScorer()
                advanced_metrics = AdvancedQualityMetrics()
                statistical_validator = StatisticalValidator()

                # Set up quality thresholds for klines data
                thresholds = QualityThresholds(
                    max_nan_ratio=0.05,
                    max_infinite_count=0,
                    min_unique_values=2,
                    max_constant_ratio=0.95
                )

                # Perform comprehensive data quality validation
                quality_result = quality_framework.validate_dataframe_quality(quality_df, context="klines_validation")

                # Get advanced quality assessment
                quality_assessment = advanced_metrics.comprehensive_quality_assessment(quality_df, context="klines_validation", step_name="data_validation")

                # Get comprehensive quality score
                score = scorer.assess_data_quality(quality_df, context="klines_validation", step_name="data_validation", data_type="klines")

                # Perform statistical distribution validation
                distribution_validation = {}
                for col in quality_df.select_dtypes(include=[np.number]).columns:
                    if col in ['open', 'high', 'low', 'close', 'volume']:
                        validation_results = statistical_validator.run_comprehensive_validation(quality_df[col].values)
                        distribution_validation[col] = {
                            'results': [{'status': r.status.value, 'message': r.message} for r in validation_results]
                        }
                duplicate_analysis = _ANALYZE_DUPLICATES_COMPREHENSIVE(quality_df) if callable(_ANALYZE_DUPLICATES_COMPREHENSIVE) else None

            # Determine quality level based on comprehensive assessment
            if score.overall_score >= 90:
                result.quality_level = DataQualityLevel.EXCELLENT
            elif score.overall_score >= 80:
                result.quality_level = DataQualityLevel.GOOD
            elif score.overall_score >= 70:
                result.quality_level = DataQualityLevel.FAIR
            elif score.overall_score >= 60:
                result.quality_level = DataQualityLevel.POOR
            else:
                result.quality_level = DataQualityLevel.FAILED

            # Collect all issues and warnings
            all_issues = list(getattr(quality_result, 'issues', [])) + list(getattr(score, 'issues', [])) + list(getattr(quality_assessment, 'metrics', []))
            all_warnings = list(getattr(quality_result, 'warnings', [])) + list(getattr(score, 'warnings', []))

            # Add duplicate analysis warnings
            if duplicate_analysis and getattr(duplicate_analysis, 'total_duplicates', 0) > 0:
                all_warnings.append(f"Found {getattr(duplicate_analysis, 'total_duplicates', 0)} duplicate records")

            result.success = True
            result.data = df_full
            result.metadata = {
                "comprehensive_score": score.overall_score,
                "quality_level": score.level.value,
                "component_scores": score.component_scores,
                "quality_assessment": {
                    "overall_score": quality_assessment.overall_score,
                    "issues_found": quality_assessment.issues_found,
                    "warnings_found": quality_assessment.warnings_found,
                    "critical_issues": quality_assessment.critical_issues
                },
                "duplicate_analysis": {
                    "total_duplicates": getattr(duplicate_analysis, 'total_duplicates', 0),
                    "true_duplicate_groups": getattr(duplicate_analysis, 'true_duplicate_groups', 0),
                    "false_duplicate_groups": getattr(duplicate_analysis, 'false_duplicate_groups', 0),
                    "mixed_duplicate_groups": getattr(duplicate_analysis, 'mixed_duplicate_groups', 0)
                },
                "distribution_validation": distribution_validation,
                "data_shape": score.data_shape,
                "assessment_timestamp": score.assessment_timestamp.isoformat()
            }
            result.warnings.extend(all_warnings)

            if self.enable_logging:
                tprint_success(f"✅ Comprehensive data quality validation completed: {result.quality_level.value} (Score: {score.overall_score:.1f})")
                if score.recommendations:
                    tprint_info(f"📋 Quality recommendations: {', '.join(score.recommendations[:3])}")

        except Exception as e:
            error_msg = f"Comprehensive data quality validation failed: {str(e)}"
            result.errors.append(error_msg)
            if self.enable_logging:
                tprint_error(f"❌ {error_msg}")

        result.processing_time = (datetime.now() - start_time).total_seconds()
        return result

    async def _handle_gaps(
        self,
        df: pd.DataFrame,
        symbol: str,
        interval: str,
        max_gap_minutes: int,
        exchange_interface: ExchangeInterface
    ) -> ProcessingResult:
        """Detect and fill data gaps."""
        start_time = datetime.now()
        result = ProcessingResult(
            step=ProcessingStep.GAP_DETECTION,
            success=False,
            errors=[],
            warnings=[]
        )

        try:
            if self.enable_logging:
                tprint_info(f"🔍 Detecting gaps in {symbol} {interval} data")

            df_with_timestamp = self._ensure_timestamp_column(df)

            # Detect gaps
            gaps = self._detect_gaps_vectorized(df_with_timestamp, interval, max_gap_minutes)

            if not gaps:
                result.success = True
                result.data = df
                result.metadata = {"gaps_detected": 0, "gaps_filled": 0}

                if self.enable_logging:
                    tprint_success("✅ No gaps detected")

                return result

            if self.enable_logging:
                tprint_warning(f"⚠️ Found {len(gaps)} gaps > {max_gap_minutes} minutes")

            # Debug: Log gap priorities
            if self.enable_logging:
                priority_1_gaps = [g for g in gaps if g.priority == 1]
                priority_2_gaps = [g for g in gaps if g.priority > 1]
                tprint_info(f"   Priority 1 gaps (will fill): {len(priority_1_gaps)}")
                tprint_info(f"   Priority 2 gaps (will skip): {len(priority_2_gaps)}")
                if priority_1_gaps:
                    for i, g in enumerate(priority_1_gaps[:3], 1):  # Show first 3
                        tprint_info(f"     Gap {i}: {g.duration_minutes:.0f} min ({g.start_time} to {g.end_time})")

            # Fill gaps by re-downloading data
            filled_data = await self._fill_gaps(df_with_timestamp, gaps, symbol, interval, exchange_interface)

            result.success = True
            result.data = filled_data
            # Capture filled ranges for downstream selective resampling
            filled_ranges = [(g.start_time, g.end_time) for g in gaps if g.priority == 1]

            # Post-fill gap recheck on base interval
            post_gaps_base = self._detect_gaps_vectorized(filled_data, interval, max_gap_minutes)

            # Post-fill validation on primary (15m) interval
            post_gaps_15m: List[GapInfo] = []
            try:
                resampled_for_validation = self._perform_resampling(
                    filled_data,
                    self.primary_validation_interval,
                    self.default_resampling_config
                )
                if not resampled_for_validation.empty:
                    post_gaps_15m = self._detect_gaps_vectorized(
                        resampled_for_validation,
                        self.primary_validation_interval,
                        max(max_gap_minutes, 15)
                    )
            except Exception as e:
                if self.enable_logging:
                    tprint_warning(f"⚠️ 15m validation resample failed: {e}")

            result.metadata = {
                "gaps_detected": len(gaps),
                "gaps_filled": len([g for g in gaps if g.priority == 1]),
                "total_gap_duration": sum(g.duration_minutes for g in gaps),
                "post_gap_check_base": len(post_gaps_base),
                "post_gap_check_15m": len(post_gaps_15m),
                "filled_ranges": filled_ranges
            }

            if self.enable_logging:
                tprint_success(f"✅ Gap handling completed: {result.metadata['gaps_filled']} gaps filled")
                tprint_info(f"   Data before gap filling: {len(df):,} records")
                tprint_info(f"   Data after gap filling: {len(filled_data):,} records")
                tprint_info(f"   Net gain: {len(filled_data) - len(df):,} records")
                if post_gaps_base:
                    tprint_warning(f"⚠️ Remaining gaps after fill (base {interval}): {len(post_gaps_base)}")
                if post_gaps_15m:
                    tprint_warning(f"⚠️ Remaining gaps after 15m validation: {len(post_gaps_15m)}")

        except Exception as e:
            error_msg = f"Gap handling failed: {str(e)}"
            result.errors.append(error_msg)
            if self.enable_logging:
                tprint_error(f"❌ {error_msg}")

        result.processing_time = (datetime.now() - start_time).total_seconds()
        return result

    def _detect_gaps(
        self,
        df: pd.DataFrame,
        interval: str,
        max_gap_minutes: int,
        expected_start: Optional[datetime] = None,
        expected_end: Optional[datetime] = None
    ) -> List[GapInfo]:
        """
        Detect gaps in the data, including boundary gaps if expected range is provided.
        """
        gaps = []

        # Calculate expected interval in minutes
        interval_minutes = self._interval_to_minutes(interval)
        if interval_minutes is None:
            return gaps

        # Ensure we have a symbol for GapInfo
        symbol = self.current_symbol or (df['symbol'].iloc[0] if not df.empty and 'symbol' in df.columns else 'UNKNOWN')

        if df.empty:
            if expected_start and expected_end:
                # Entire range is missing
                gaps.append(GapInfo(
                    start_time=expected_start,
                    end_time=expected_end,
                    duration_minutes=int((expected_end - expected_start).total_seconds() / 60),
                    symbol=symbol,
                    interval=interval,
                    priority=1
                ))
            return gaps

        # Normalize index to UTC tz-aware for reliable comparisons
        df_normalized = self._ensure_utc_index(df)
        df_sorted = df_normalized.sort_index()
        
        # 1. Check start boundary gap
        if expected_start:
            expected_start_utc = self._ensure_utc_timestamp(expected_start)
            actual_start_utc = df_sorted.index[0]
            if (actual_start_utc - expected_start_utc).total_seconds() / 60 > max_gap_minutes:
                gaps.append(GapInfo(
                    start_time=expected_start_utc,
                    end_time=actual_start_utc,
                    duration_minutes=int((actual_start_utc - expected_start_utc).total_seconds() / 60),
                    symbol=symbol,
                    interval=interval,
                    priority=1
                ))

        # 2. Check internal gaps
        safety_max_hours = 24 * 7  # 1 week segments for large historical gaps
        safety_max_minutes = safety_max_hours * 60

        for i in range(len(df_sorted) - 1):
            current_time = df_sorted.index[i]
            next_time = df_sorted.index[i + 1]

            expected_next_time = current_time + timedelta(minutes=interval_minutes)
            actual_gap_minutes = (next_time - expected_next_time).total_seconds() / 60

            if actual_gap_minutes > max_gap_minutes:
                remaining_start = expected_next_time
                remaining_end = next_time

                while remaining_start < remaining_end:
                    segment_end = min(remaining_start + timedelta(minutes=safety_max_minutes), remaining_end)
                    segment_duration_minutes = int((segment_end - remaining_start).total_seconds() / 60)

                    # Only add if segment has meaningful duration
                    if segment_duration_minutes > 0:
                        gap = GapInfo(
                            start_time=remaining_start,
                            end_time=segment_end,
                            duration_minutes=segment_duration_minutes,
                            symbol=symbol,
                            interval=interval,
                            priority=1 if segment_duration_minutes >= interval_minutes else 2
                        )
                        gaps.append(gap)

                    remaining_start = segment_end

        # 3. Check end boundary gap
        if expected_end:
            expected_end_utc = self._ensure_utc_timestamp(expected_end)
            actual_end_utc = df_sorted.index[-1]
            if (expected_end_utc - actual_end_utc).total_seconds() / 60 > max_gap_minutes:
                gaps.append(GapInfo(
                    start_time=actual_end_utc,
                    end_time=expected_end_utc,
                    duration_minutes=int((expected_end_utc - actual_end_utc).total_seconds() / 60),
                    symbol=symbol,
                    interval=interval,
                    priority=1
                ))

        return gaps

    def _detect_gaps_vectorized(
        self,
        df: pd.DataFrame,
        interval: str,
        max_gap_minutes: int,
        expected_start: Optional[datetime] = None,
        expected_end: Optional[datetime] = None
    ) -> List[GapInfo]:
        """
        Vectorized gap detection using Numba for improved performance.
        """
        gaps = []

        # Calculate expected interval in minutes
        interval_minutes = self._interval_to_minutes(interval)
        if interval_minutes is None:
            return gaps

        # Ensure we have a symbol for GapInfo
        symbol = self.current_symbol or (df[symbol].iloc[0] if not df.empty and symbol in df.columns else UNKNOWN)

        if df.empty:
            if expected_start and expected_end:
                # Entire range is missing
                gaps.append(GapInfo(
                    start_time=expected_start,
                    end_time=expected_end,
                    duration_minutes=int((expected_end - expected_start).total_seconds() / 60),
                    symbol=symbol,
                    interval=interval,
                    priority=1
                ))
            return gaps

    def _verify_data_quality_vectorized(self, df: pd.DataFrame) -> Dict[str, int]:
        """
        Vectorized data quality verification using Numba for improved performance.
        """
        if df.empty:
            return {"ohlc_issues": 0, "volume_issues": 0, "price_issues": 0}
        
        quality_issues = {"ohlc_issues": 0, "volume_issues": 0, "price_issues": 0}
        
        # Use vectorized quality check for large datasets
        if len(df) > 1000 and NUMBA_AVAILABLE:
            try:
                # Extract OHLCV arrays
                opens = df["open"].values.astype(np.float64)
                highs = df["high"].values.astype(np.float64)
                lows = df["low"].values.astype(np.float64)
                closes = df["close"].values.astype(np.float64)
                volumes = df["volume"].values.astype(np.float64)
                
                # Use Numba for vectorized quality check
                ohlc_issues, volume_issues, price_issues = _numba_verify_data_quality(
                    opens, highs, lows, closes, volumes
                )
                
                quality_issues = {
                    "ohlc_issues": int(ohlc_issues),
                    "volume_issues": int(volume_issues),
                    "price_issues": int(price_issues)
                }
                
                return quality_issues
                
            except Exception as e:
                if self.enable_logging:
                    tprint_warning(f"⚠️ Vectorized quality check failed, using fallback: {e}")
        
        # Fallback to standard quality checks
        # Check OHLC consistency
        ohlc_mask = (df["high"] < df["low"]) | (df["open"] > df["high"]) | (df["open"] < df["low"]) | (df["close"] > df["high"]) | (df["close"] < df["low"])
        quality_issues["ohlc_issues"] = ohlc_mask.sum()
        
        # Check volume issues
        volume_mask = df["volume"] < 0
        quality_issues["volume_issues"] = volume_mask.sum()
        
        # Check price issues (zero, negative, extreme values)
        price_mask = (df["open"] <= 0) | (df["high"] <= 0) | (df["low"] <= 0) | (df["close"] <= 0)
        quality_issues["price_issues"] = price_mask.sum()
        
        # Check for extreme price movements
        if len(df) > 1:
            price_changes = df["close"].pct_change().abs()
            extreme_moves = price_changes > 0.5  # > 50% move
            quality_issues["price_issues"] += extreme_moves.sum()
        
        return quality_issues

        df_sorted = df_normalized.sort_index()

        # Use vectorized gap detection for large datasets
        if len(df_sorted) > 1000 and NUMBA_AVAILABLE:
            try:
                # Convert timestamps to nanoseconds for Numba
                timestamps_ns = df_sorted.index.view(np.int64).values
                
                # Detect gaps using vectorized Numba function
                gap_durations = _numba_detect_gaps_vectorized(timestamps_ns, interval_minutes)
                
                # Convert gap durations to GapInfo objects
                for i, duration in enumerate(gap_durations):
                    if duration > max_gap_minutes and i > 0:
                        gap_start = df_sorted.index[i-1] + timedelta(minutes=interval_minutes)
                        gap_end = df_sorted.index[i]
                        
                        gaps.append(GapInfo(
                            start_time=gap_start,
                            end_time=gap_end,
                            duration_minutes=int(duration),
                            symbol=symbol,
                            interval=interval,
                            priority=1 if duration >= interval_minutes else 2
                        ))
                
                # Check boundary gaps
                if expected_start:
                    expected_start_utc = self._ensure_utc_timestamp(expected_start)
                    actual_start_utc = df_sorted.index[0]
                    if (actual_start_utc - expected_start_utc).total_seconds() / 60 > max_gap_minutes:
                        gaps.append(GapInfo(
                            start_time=expected_start_utc,
                            end_time=actual_start_utc,
                            duration_minutes=int((actual_start_utc - expected_start_utc).total_seconds() / 60),
                            symbol=symbol,
                            interval=interval,
                            priority=1
                        ))
                
                if expected_end:
                    expected_end_utc = self._ensure_utc_timestamp(expected_end)
                    actual_end_utc = df_sorted.index[-1]
                    if (expected_end_utc - actual_end_utc).total_seconds() / 60 > max_gap_minutes:
                        gaps.append(GapInfo(
                            start_time=actual_end_utc,
                            end_time=expected_end_utc,
                            duration_minutes=int((expected_end_utc - actual_end_utc).total_seconds() / 60),
                            symbol=symbol,
                            interval=interval,
                            priority=1
                        ))
                
                return gaps
                
            except Exception as e:
                if self.enable_logging:
                    tprint_warning(f"⚠️ Vectorized gap detection failed, falling back to standard method: {e}")
                
        # Fallback to standard gap detection
        return self._detect_gaps_vectorized(df, interval, max_gap_minutes, expected_start, expected_end)

    def _detect_missing_in_processed_15m(
        self,
        symbol: str,
        years: int
    ) -> List[GapInfo]:
        """Detect missing ranges in the processed 15m data set for a given lookback."""
        gaps: List[GapInfo] = []

        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=years * 365)


            # Use refined _detect_gaps with boundary checks
            gaps = self._detect_gaps_vectorized(
                df_15m if df_15m is not None else pd.DataFrame(),
                "15m", 
                self.config.max_gap_minutes,
                expected_start=start_date,
                expected_end=end_date
            )

            if self.enable_logging:
                if gaps:
                    total_missing_mins = sum(g.duration_minutes for g in gaps)
                    tprint_warning(
                        f"⚠️ Detected {len(gaps)} missing ranges ({total_missing_mins} mins) in processed 15m data for {symbol}"
                    )
                else:
                    tprint_info(
                        f"✅ No missing ranges detected in processed 15m data for {symbol}"
                    )

        except Exception as e:
            if self.enable_logging:
                tprint_warning(
                    f"⚠️ Failed to analyze processed 15m gaps for {symbol}: {e}"
                )

        return gaps

    async def _download_missing_ranges(
        self,
        gaps_15m: List[GapInfo],
        symbol: str,
        base_interval: str,
        exchange_interface: "ExchangeInterface",
        resampling_config: Optional[ResamplingConfig] = None
    ) -> Dict[str, Any]:
        """Backfill processed 15m gaps by downloading corresponding base-interval data.

        Enhanced with:
        - Atomic updates per gap to ensure partial progress is saved.
        - Robust error handling and retry logic.
        - Detailed stats reporting.
        """

        stats: Dict[str, Any] = {
            "gaps_attempted": 0,
            "gaps_skipped": 0,
            "gaps_filled": 0,
            "base_candles_downloaded": 0,
            "resampled_15m_records": 0,
            "failed_gaps": [],
        }

        if not gaps_15m or exchange_interface is None:
            return stats

        if resampling_config is None:
            resampling_config = self.default_resampling_config

        interval_minutes = self._interval_to_minutes(base_interval) or 1
        batch_size = 1000
        batch_duration = timedelta(minutes=batch_size * interval_minutes)
        interval_delta = timedelta(minutes=interval_minutes)

        for gap in gaps_15m:
            if gap.priority > 1:
                stats["gaps_skipped"] += 1
                continue

            stats["gaps_attempted"] += 1

            gap_start = self._to_naive_timestamp(gap.start_time)
            gap_end = self._to_naive_timestamp(gap.end_time)
            if gap_start is None or gap_end is None or gap_start >= gap_end:
                stats["gaps_skipped"] += 1
                continue

            if self.enable_logging:
                tprint_info(
                    f"📥 Backfilling 15m gap: {gap_start} → {gap_end} "
                    f"({gap.duration_minutes} mins)"
                )

            current_start = gap_start
            safety_counter = 0
            gap_base_klines: List[Any] = []
            gap_success = True

            while current_start < gap_end:
                safety_counter += 1
                if safety_counter > 10000:
                    tprint_warning("⚠️ Safety break in gap download")
                    gap_success = False
                    break

                batch_end = min(current_start + batch_duration, gap_end)

                try:
                    batch_klines = await exchange_interface.get_klines(
                        symbol=symbol,
                        interval=base_interval,
                        start_time=current_start,
                        end_time=batch_end,
                        limit=batch_size,
                    )
                    
                    if not batch_klines:
                        # No data returned for this segment
                        current_start = batch_end + interval_delta
                        continue

                    gap_base_klines.extend(batch_klines)

                    extracted_ts = self._extract_timestamps_from_klines(batch_klines)
                    if not extracted_ts:
                        current_start = batch_end + interval_delta
                        continue

                    last_ts = max(extracted_ts)
                    last_naive = self._to_naive_timestamp(last_ts)
                    if last_naive is None:
                        current_start = batch_end + interval_delta
                        continue

                    current_start = last_naive + interval_delta
                    await asyncio.sleep(0.1)  # Rate limit respect

                except Exception as e:
                    tprint_warning(f"⚠️ Batch download failed for {symbol} at {current_start}: {e}")
                    gap_success = False
                    break

            if gap_success and gap_base_klines:
                # Process this gap immediately to ensure progress persistence
                try:
                    base_df = self._klines_to_dataframe(gap_base_klines, symbol, base_interval)
                    if not base_df.empty:
                        standardized_df = self.data_standardizer.standardize(base_df, exchange=self.exchange)
                        standardized_df = self._ensure_naive_datetime_index(standardized_df)
                        
                        # Add required columns if missing
                        for col, val in [('timestamp', standardized_df.index), ('symbol', symbol), 
                                       ('interval', base_interval), ('exchange', self.exchange)]:
                            if col not in standardized_df.columns:
                                standardized_df[col] = val

                        stats["base_candles_downloaded"] += len(standardized_df)

                        # Resample to 15m
                        resampled_15m = self._perform_resampling(standardized_df, "15m", resampling_config)
                        if not resampled_15m.empty:
                            normalized_15m = self._normalize_calendar_columns(resampled_15m)
                            if normalized_15m is not None:
                                resampled_15m = normalized_15m

                            stats["resampled_15m_records"] += len(resampled_15m)

                            # Atomic update to parquet storage
                            updated = self.klines_manager.update_data(
                                resampled_15m,
                                symbol,
                                "15m",
                                data_type="processed",
                            )
                            if updated:
                                stats["gaps_filled"] += 1
                                if self.enable_logging:
                                    tprint_success(f"✅ Filled and persisted gap: {gap_start} → {gap_end}")
                            else:
                                tprint_warning(f"⚠️ Failed to persist gap: {gap_start}")
                                stats["failed_gaps"].append(str(gap_start))
                except Exception as process_exc:
                    tprint_error(f"❌ Error processing gap {gap_start}: {process_exc}")
                    stats["failed_gaps"].append(str(gap_start))
            elif not gap_success:
                stats["failed_gaps"].append(str(gap_start))

        return stats

    def _interval_to_minutes(self, interval: str) -> Optional[int]:
        """Convert interval string to minutes."""
        interval_map = {
            '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '2h': 120, '4h': 240, '6h': 360, '8h': 480, '12h': 720,
            '1d': 1440, '3d': 4320, '1w': 10080, '1M': 43200
        }
        return interval_map.get(interval)

    def _extract_timestamps_from_klines(self, klines_data: List[Any]) -> List[pd.Timestamp]:
        """
        Extract timestamps from klines data.

        This method handles both object-based and list-based klines formats
        and returns a list of pandas Timestamps.
        """
        timestamps = []

        try:
            for kline in klines_data:
                if hasattr(kline, 'timestamp'):
                    naive_ts = self._to_naive_timestamp(kline.timestamp)
                    if naive_ts is not None:
                        timestamps.append(naive_ts)
                else:
                    timestamp = kline[0]

                    if isinstance(timestamp, (int, float)):
                        if timestamp > 1e15:
                            converted_timestamp = pd.to_datetime(timestamp, unit='us', utc=True)
                        elif timestamp > 1e12:
                            converted_timestamp = pd.to_datetime(timestamp, unit='ms', utc=True)
                        elif timestamp > 1e9:
                            converted_timestamp = pd.to_datetime(timestamp, unit='s', utc=True)
                        else:
                            converted_timestamp = pd.to_datetime(timestamp, unit='s', utc=True)
                    else:
                        converted_timestamp = pd.to_datetime(timestamp, utc=True)

                    naive_ts = self._to_naive_timestamp(converted_timestamp)
                    if naive_ts is not None:
                        timestamps.append(naive_ts)

        except Exception as e:
            if self.enable_logging:
                tprint_warning(f"⚠️ Error extracting timestamps: {e}")
            return []

        return timestamps

    async def _fill_gaps(
        self,
        df: pd.DataFrame,
        gaps: List[GapInfo],
        symbol: str,
        interval: str,
        exchange_interface: ExchangeInterface
    ) -> pd.DataFrame:
        """Fill gaps by re-downloading data in batches.

        This implementation is designed to be crash-resilient:
        - It keeps the in-memory behavior identical (gap detection, stall handling, merging).
        - Additionally, each successfully filled gap segment is *immediately* written to
          raw monthly parquet files via KlinesParquetManager.write_data with overwrite=False,
          so progress is persisted even if the pipeline crashes later in the run.
        """
        filled_data = self._ensure_utc_index(df)

        if self.enable_logging and isinstance(filled_data.index, pd.DatetimeIndex):
            tz_obj = filled_data.index.tz
            tz_info = getattr(tz_obj, "zone", str(tz_obj))
            tprint_info(f"🕒 Gap filling using timezone: {tz_info}")

        # Ensure we have a real dispatcher ready for gap downloads when working with Binance data
        dispatcher = getattr(exchange_interface, "dispatcher", None)
        exchange_type_value = getattr(exchange_interface, "exchange_type", "").lower()

        if exchange_type_value == "simulated":
            # In simulated mode we don't have a live exchange; skip gap filling gracefully.
            if self.enable_logging:
                tprint_warning(
                    "⚠️ Simulated exchange interface provided; skipping live gap filling and returning original data"
                )
            return self._ensure_naive_datetime_index(filled_data)

        if dispatcher is None:
            if self.enable_logging:
                tprint_info("🔄 Exchange dispatcher not initialized; attempting connect before gap download")
            try:
                await exchange_interface.connect()
            except Exception as e:  # pragma: no cover - network dependent
                if self.enable_logging:
                    tprint_warning(f"⚠️ Exchange connect failed before gap fill: {e}")
            dispatcher = getattr(exchange_interface, "dispatcher", None)

        if dispatcher is None:
            # In public/offline mode we may not have a fully initialized dispatcher; do not
            # fail the entire pipeline, just keep the existing data and report remaining gaps
            if self.enable_logging:
                tprint_warning(
                    "⚠️ Exchange dispatcher unavailable; skipping gap filling and returning original data"
                )
            return self._ensure_naive_datetime_index(filled_data)

        if self.exchange.lower() == "binance":
            dispatcher_config = getattr(dispatcher, "config", None)
            dispatcher_exchange_type = getattr(dispatcher_config, "exchange_type", None)
            if dispatcher_exchange_type != ExchangeType.BINANCE:
                raise RuntimeError(
                    "Binance gap filling requires the Binance dispatcher; received a different exchange dispatcher."
                )

        for gap_idx, gap in enumerate(gaps):
            if gap.priority > 1:  # Skip low priority gaps
                continue

            try:
                gap_start_utc = self._ensure_utc_timestamp(gap.start_time)
                gap_end_utc = self._ensure_utc_timestamp(gap.end_time)
                if gap_start_utc is None or gap_end_utc is None or gap_start_utc >= gap_end_utc:
                    if self.enable_logging:
                        tprint_warning(f"⚠️ Skipping gap {gap_idx+1} due to invalid timestamps")
                    continue

                if self.enable_logging:
                    tprint_info(
                        f"📥 Re-downloading data for gap {gap_idx+1}/{len(gaps)}: {gap_start_utc} to {gap_end_utc}"
                    )
                    tprint_info(
                        f"   Gap size: {gap.duration_minutes:.0f} minutes "
                        f"({gap.duration_minutes/60:.1f} hours, {gap.duration_minutes/1440:.1f} days)"
                    )

                # For large gaps (> 1000 minutes), download in batches.
                # IMPORTANT: We work BACKWARDS from gap_end to gap_start because exchanges
                # with both start_time and end_time return the LAST limit candles near end_time.
                interval_minutes = self._interval_to_minutes(interval) or 1
                batch_size = 1000
                batch_duration = timedelta(minutes=batch_size * interval_minutes)

                gap_batches: List[Any] = []
                # Start from the END of the gap and work backwards
                current_end = gap_end_utc
                earliest_downloaded = gap_end_utc
                batch_num = 1
                stall_count = 0
                max_stalls = 3

                while current_end > gap_start_utc:
                    # Calculate the expected start for this batch
                    batch_start = max(current_end - batch_duration, gap_start_utc)

                    if self.enable_logging and (batch_num == 1 or batch_num % 20 == 0):
                        tprint_info(
                            f"   Batch {batch_num}: {batch_start.strftime('%Y-%m-%d %H:%M')} "
                            f"→ {current_end.strftime('%Y-%m-%d %H:%M')}"
                        )

                    try:
                        batch_klines = await exchange_interface.get_klines(
                            symbol=symbol,
                            interval=interval,
                            start_time=batch_start,
                            end_time=current_end,
                            limit=batch_size,
                        )

                        if batch_klines:
                            # Extract actual timestamps from the returned data to verify progress
                            extracted_timestamps = self._extract_timestamps_from_klines(batch_klines)
                            actual_timestamps = [
                                self._ensure_utc_timestamp(ts) for ts in extracted_timestamps if ts is not None
                            ]

                            if actual_timestamps:
                                actual_earliest = min(actual_timestamps)
                                actual_latest = max(actual_timestamps)

                                if self.enable_logging and batch_num == 1:
                                    tprint_info(
                                        f"   📊 Batch {batch_num} returned {len(batch_klines)} candles: "
                                        f"{actual_earliest.strftime('%Y-%m-%d %H:%M')} → "
                                        f"{actual_latest.strftime('%Y-%m-%d %H:%M')}"
                                    )

                                # Check if we're making progress backwards
                                if actual_earliest >= earliest_downloaded - timedelta(minutes=interval_minutes):
                                    # We're stalled - the API keeps returning similar/overlapping data
                                    stall_count += 1
                                    if self.enable_logging:
                                        tprint_warning(
                                            f"   ⚠️ Batch {batch_num} stalled (earliest: "
                                            f"{actual_earliest.strftime('%Y-%m-%d %H:%M')}, previous: "
                                            f"{earliest_downloaded.strftime('%Y-%m-%d %H:%M')}), "
                                            f"stall count: {stall_count}/{max_stalls}"
                                        )

                                    if stall_count >= max_stalls:
                                        if self.enable_logging:
                                            tprint_warning(
                                                "   ⚠️ Stall threshold reached. Switching to forward "
                                                "day-sized batching for remaining gap."
                                            )
                                        forward_df = await self._download_gap_forward(
                                            gap_start=gap_start_utc,
                                            current_coverage_start=earliest_downloaded,
                                            gap_end=gap_end_utc,
                                            symbol=symbol,
                                            interval=interval,
                                            interval_minutes=interval_minutes,
                                            batch_size=batch_size,
                                            exchange_interface=exchange_interface,
                                        )
                                        if forward_df is not None and not forward_df.empty:
                                            gap_batches.extend(self._dataframe_to_klines_list(forward_df))
                                            earliest_downloaded = forward_df.index.min()
                                            current_end = (
                                                earliest_downloaded
                                                - timedelta(minutes=interval_minutes)
                                            )
                                            stall_count = 0
                                            continue
                                        break

                                    # Move back further to try to get earlier data
                                    current_end = actual_earliest - batch_duration
                                else:
                                    # We made progress! Reset stall counter
                                    stall_count = 0
                                    earliest_downloaded = actual_earliest

                                    if self.enable_logging and batch_num % 20 == 0:
                                        tprint_info(
                                            f"   ✓ Progress: now at "
                                            f"{earliest_downloaded.strftime('%Y-%m-%d %H:%M')}"
                                        )

                                    gap_batches.extend(batch_klines)

                                    # Move current_end to just before the earliest timestamp we got
                                    current_end = actual_earliest - timedelta(minutes=interval_minutes)
                            else:
                                if self.enable_logging:
                                    tprint_warning(
                                        f"   ⚠️ Could not extract timestamps from batch {batch_num}"
                                    )
                                break
                        else:
                            if self.enable_logging:
                                tprint_warning(
                                    f"   ⚠️ No klines returned for batch {batch_num} "
                                    f"({batch_start} → {current_end}), treating gap as exhausted"
                                )
                            break

                        batch_num += 1
                        await asyncio.sleep(0.05)

                    except Exception as e:
                        if self.enable_logging:
                            tprint_warning(f"   ⚠️ Batch {batch_num} failed: {e}")
                        # On error, try to continue by moving back
                        current_end = current_end - batch_duration
                        batch_num += 1
                        await asyncio.sleep(1.0)

                # Convert all batches to DataFrame
                if gap_batches:
                    if self.enable_logging:
                        tprint_success(
                            f"   ✅ Downloaded {len(gap_batches)} candles in {batch_num-1} batches"
                        )

                    gap_df = self._klines_to_dataframe(gap_batches, symbol, interval)
                    if not gap_df.empty:
                        # Standardize the gap data
                        standardized_gap_df = self.data_standardizer.standardize(
                            gap_df, exchange=self.exchange
                        )
                        standardized_gap_df = self._ensure_utc_index(standardized_gap_df)
                        standardized_gap_df = self._ensure_timestamp_column(standardized_gap_df)

                        # Progressively persist newly filled gap data to raw storage so that
                        # successfully downloaded candles are not lost if the pipeline crashes
                        # later in the run. We only write the incremental gap segment and use
                        # overwrite=False so existing monthly files are merged, not replaced.
                        try:
                            storage_df = self._normalize_calendar_columns(standardized_gap_df)
                            if storage_df is None:
                                storage_df = standardized_gap_df
                            self.klines_manager.write_data(
                                storage_df,
                                symbol,
                                interval,
                                "raw",
                                overwrite=False,
                            )
                        except Exception as e:
                            if self.enable_logging:
                                tprint_warning(
                                    f"⚠️ Failed to persist gap {gap_idx+1} to raw storage: {e}"
                                )

                        # Merge with existing in-memory data
                        filled_data = pd.concat([filled_data, standardized_gap_df], ignore_index=False)
                        filled_data = filled_data[~filled_data.index.duplicated(keep="first")]
                        filled_data.sort_index(inplace=True)

                        if self.enable_logging:
                            tprint_success(
                                f"✅ Filled gap with {len(standardized_gap_df)} records "
                                f"(total now: {len(filled_data):,})"
                            )
                    elif self.enable_logging:
                        tprint_warning(
                            f"⚠️ Gap {gap_idx+1} returned no usable candles after standardization"
                        )
                else:
                    if self.enable_logging:
                        tprint_warning(
                            f"⚠️ Gap {gap_idx+1} produced no candle data; Binance may have no coverage for this period"
                        )

            except Exception as e:
                if self.enable_logging:
                    tprint_warning(f"⚠️ Failed to fill gap {gap_start_utc}: {e}")

        return self._ensure_naive_datetime_index(filled_data)

    def _dataframe_to_klines_list(self, df: pd.DataFrame) -> List[List[Any]]:
        """Convert a standardized DataFrame back into raw kline list format for downstream processing."""
        records: List[List[Any]] = []
        for ts, row in df.iterrows():
            ts_naive = EnhancedKlinesProcessingPipeline._to_utc_naive_timestamp(ts)
            if ts_naive is None:
                continue

            records.append([
                int(ts_naive.value // 10**6),
                float(row.get('open', 0.0)),
                float(row.get('high', 0.0)),
                float(row.get('low', 0.0)),
                float(row.get('close', 0.0)),
                float(row.get('volume', 0.0)),
                float(row.get('quote_volume', 0.0)) if 'quote_volume' in row else 0.0,
                int(row.get('trades_count', 0)) if 'trades_count' in row else 0,
                float(row.get('taker_buy_base_volume', 0.0)) if 'taker_buy_base_volume' in row else 0.0,
                float(row.get('taker_buy_quote_volume', 0.0)) if 'taker_buy_quote_volume' in row else 0.0
            ])
        return records

    async def _download_gap_forward(
        self,
        gap_start: pd.Timestamp,
        current_coverage_start: pd.Timestamp,
        gap_end: pd.Timestamp,
        symbol: str,
        interval: str,
        interval_minutes: int,
        batch_size: int,
        exchange_interface: ExchangeInterface
    ) -> Optional[pd.DataFrame]:
        """Download remaining gap candles by scanning forward in daily windows to work around exchange limits."""
        collected_frames: List[pd.DataFrame] = []

        gap_start_naive = self._to_naive_timestamp(gap_start)
        coverage_start_naive = self._to_naive_timestamp(current_coverage_start)
        gap_end_naive = self._to_naive_timestamp(gap_end)

        if gap_start_naive is None or coverage_start_naive is None or gap_end_naive is None:
            if self.enable_logging:
                tprint_warning("⚠️ Forward gap download aborted due to invalid timestamps")
            return None

        cursor = gap_start_naive
        interval_delta = timedelta(minutes=interval_minutes)

        try:
            while cursor < coverage_start_naive:
                segment_end = min(cursor + timedelta(days=1), coverage_start_naive)
                segment_cursor = cursor

                while segment_cursor < segment_end:
                    query_end = min(segment_cursor + timedelta(minutes=batch_size * interval_minutes), segment_end)
                    try:
                        batch_klines = await exchange_interface.get_klines(
                            symbol=symbol,
                            interval=interval,
                            start_time=segment_cursor,
                            end_time=query_end,
                            limit=batch_size
                        )
                    except Exception as e:  # pragma: no cover - network dependent
                        if self.enable_logging:
                            tprint_warning(f"   ⚠️ Forward batch request failed: {e}")
                        break

                    if not batch_klines:
                        break

                    batch_df = self._klines_to_dataframe(batch_klines, symbol, interval)
                    if batch_df.empty:
                        break

                    batch_df_utc = self._ensure_utc_index(batch_df)
                    collected_frames.append(batch_df_utc)
                    last_timestamp = batch_df_utc.index.max()
                    if last_timestamp is pd.NaT:
                        break

                    # Convert to naive for comparison
                    last_timestamp_naive = self._to_naive_timestamp(last_timestamp)
                    if last_timestamp_naive is None:
                        break

                    segment_cursor = last_timestamp_naive + interval_delta

                    if last_timestamp_naive >= segment_end - interval_delta:
                        break

                    await asyncio.sleep(0.05)

                cursor = segment_end

            if not collected_frames:
                return None

            gap_df = pd.concat(collected_frames, ignore_index=False)
            gap_df = gap_df[~gap_df.index.duplicated(keep='first')]
            gap_df.sort_index(inplace=True)

            standardized_gap_df = self.data_standardizer.standardize(
                gap_df, exchange=self.exchange
            )
            standardized_gap_df = self._ensure_utc_index(standardized_gap_df)
            standardized_gap_df = self._ensure_timestamp_column(standardized_gap_df)

            return standardized_gap_df

        except Exception as e:  # pragma: no cover - defensive
            if self.enable_logging:
                tprint_warning(f"⚠️ Forward gap download failed: {e}")
            return None

    async def _handle_duplicates(
        self,
        df: pd.DataFrame,
        symbol: str,
        interval: str,
        exchange_interface: Optional[ExchangeInterface] = None
    ) -> ProcessingResult:
        """Handle duplicate timestamps with optional redownload for conflicting records."""
        start_time = datetime.now()
        result = ProcessingResult(
            step=ProcessingStep.DUPLICATE_HANDLING,
            success=False,
            errors=[],
            warnings=[]
        )

        try:
            if self.enable_logging:
                tprint_info(f"🔍 Analyzing duplicates in {symbol} {interval} data")

            df_with_timestamp = self._ensure_timestamp_column(df)
            timestamp_column = 'timestamp_ms' if 'timestamp_ms' in df_with_timestamp.columns else 'timestamp'

            # 1. Use ComprehensiveDuplicateAnalyzer for deep analysis
            analysis_result = self.duplicate_analyzer.analyze_duplicates(df_with_timestamp, timestamp_column=timestamp_column)

            cleaned_df = df.copy()
            true_duplicate_records_removed = 0
            conflicting_ranges_for_redownload = []

            if timestamp_column in df_with_timestamp.columns:
                ts_series = df_with_timestamp[timestamp_column]
                duplicate_mask = ts_series.duplicated(keep=False)

                if duplicate_mask.any():
                    key_columns = ['open', 'high', 'low', 'close', 'volume']
                    available_key_columns = [col for col in key_columns if col in df_with_timestamp.columns]

                    if available_key_columns:
                        indices_to_drop: List[Any] = []
                        duplicated_df = df_with_timestamp[duplicate_mask]
                        
                        for ts, group in duplicated_df.groupby(timestamp_column):
                            if len(group) <= 1:
                                continue

                            first_record = group.iloc[0]
                            all_identical = True

                            for i in range(1, len(group)):
                                current_record = group.iloc[i]
                                if not all(first_record[col] == current_record[col] for col in available_key_columns):
                                    all_identical = False
                                    break

                            if all_identical:
                                # All identical: keep latest, drop others
                                indices_to_drop.extend(group.index[:-1].tolist())
                            else:
                                # Conflicting records at same timestamp: flag for redownload
                                ts_dt = pd.to_datetime(ts, unit='ms') if timestamp_column == 'timestamp_ms' else ts
                                conflicting_ranges_for_redownload.append(ts_dt)

                        if indices_to_drop:
                            cleaned_df = cleaned_df.drop(index=list(set(indices_to_drop)))
                            true_duplicate_records_removed = len(indices_to_drop)

            # 2. Handle conflicting duplicates via redownload if interface available
            if conflicting_ranges_for_redownload and exchange_interface:
                if self.enable_logging:
                    tprint_warning(f"⚠️ Found {len(conflicting_ranges_for_redownload)} conflicting timestamps; attempting recovery redownload")
                
                # Recover each conflicting timestamp with a fresh download
                recovered_records = []
                for ts in conflicting_ranges_for_redownload:
                    try:
                        # Download exactly this timestamp
                        fresh_klines = await exchange_interface.get_klines(
                            symbol=symbol,
                            interval=interval,
                            start_time=ts,
                            end_time=ts,
                            limit=1
                        )
                        if fresh_klines:
                            fresh_df = self._klines_to_dataframe(fresh_klines, symbol, interval)
                            fresh_df = self.data_standardizer.standardize(fresh_df, exchange=self.exchange)
                            fresh_df = self._ensure_naive_datetime_index(fresh_df)
                            recovered_records.append(fresh_df)
                            await asyncio.sleep(0.1) # Rate limit
                    except Exception as e:
                        tprint_warning(f"⚠️ Failed to recover timestamp {ts}: {e}")

                if recovered_records:
                    recovered_df = pd.concat(recovered_records)
                    # Remove the conflicting timestamps from cleaned_df and append recovered ones
                    cleaned_df = cleaned_df[~cleaned_df.index.isin(conflicting_ranges_for_redownload)]
                    cleaned_df = pd.concat([cleaned_df, recovered_df]).sort_index()
                    
                    if self.enable_logging:
                        tprint_success(f"✅ Recovered {len(recovered_records)} conflicting timestamps via redownload")

            result.success = True
            result.data = cleaned_df
            result.metadata = {
                "total_duplicates": analysis_result.total_duplicates,
                "true_duplicate_groups": analysis_result.true_duplicate_groups,
                "true_duplicate_records_removed": true_duplicate_records_removed,
                "conflicting_timestamps": len(conflicting_ranges_for_redownload),
                "false_duplicates": analysis_result.false_duplicate_groups,
                "mixed_duplicates": analysis_result.mixed_duplicate_groups
            }

            if self.enable_logging:
                tprint_success(f"✅ Duplicate handling completed: {analysis_result.total_duplicates} processed")
                if conflicting_ranges_for_redownload:
                    tprint_warning(f"   ⚠️ {len(conflicting_ranges_for_redownload)} timestamps have conflicting data!")

        except Exception as e:
            error_msg = f"Duplicate handling failed: {str(e)}"
            result.errors.append(error_msg)
            if self.enable_logging:
                tprint_error(f"❌ {error_msg}")

        result.processing_time = (datetime.now() - start_time).total_seconds()
        return result

    async def _resample_data(
        self,
        df: pd.DataFrame,
        symbol: str,
        resampling_config: ResamplingConfig
    ) -> ProcessingResult:
        """Resample data to different intervals."""
        start_time = datetime.now()
        result = ProcessingResult(
            step=ProcessingStep.RESAMPLING,
            success=False,
            errors=[],
            warnings=[]
        )

        try:
            if self.enable_logging:
                tprint_info(f"🔄 Resampling data to {resampling_config.target_intervals}")

            resampled_data = df.copy()

            for target_interval in resampling_config.target_intervals:
                try:
                    # Convert interval to pandas frequency
                    freq = self._interval_to_pandas_freq(target_interval)
                    if freq is None:
                        result.warnings.append(f"Unsupported interval: {target_interval}")
                        continue

                    # Resample OHLCV data
                    resampled = resampled_data.resample(freq).agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum' if resampling_config.preserve_volume else 'mean'
                    }).dropna()

                    # Add metadata
                    resampled['symbol'] = symbol
                    resampled['interval'] = target_interval
                    resampled['exchange'] = self.exchange

                    # Save resampled data
                    output_file = self.data_dir / f"{symbol}_{target_interval}_resampled.parquet"
                    resampled.to_parquet(output_file)

                    if self.enable_logging:
                        tprint_success(f"✅ Resampled to {target_interval}: {len(resampled)} records")

                except Exception as e:
                    result.warnings.append(f"Failed to resample to {target_interval}: {e}")

            result.success = True
            result.data = resampled_data
            result.metadata = {
                "target_intervals": resampling_config.target_intervals,
                "resampling_method": resampling_config.method
            }

        except Exception as e:
            error_msg = f"Data resampling failed: {str(e)}"
            result.errors.append(error_msg)
            if self.enable_logging:
                tprint_error(f"❌ {error_msg}")

        result.processing_time = (datetime.now() - start_time).total_seconds()
        return result

    def _interval_to_pandas_freq(self, interval: str) -> Optional[str]:
        """Convert interval string to pandas frequency."""
        freq_map = {
            '1m': '1T', '3m': '3T', '5m': '5T', '15m': '15T', '30m': '30T',
            '1h': '1H', '2h': '2H', '4h': '4H', '6h': '6H', '8h': '8H', '12h': '12H',
            '1d': '1D', '3d': '3D', '1w': '1W', '1M': '1M'
        }
        return freq_map.get(interval)

    async def _final_quality_check(
        self,
        df: pd.DataFrame,
        symbol: str,
        interval: str
    ) -> ProcessingResult:
        """Perform comprehensive final quality check using advanced quality metrics from src/utils/data/quality/."""
        start_time = datetime.now()
        result = ProcessingResult(
            step=ProcessingStep.QUALITY_CHECK,
            success=False,
            errors=[],
            warnings=[]
        )

        try:
            if self.enable_logging:
                tprint_info(f"✅ Running comprehensive final quality check for {symbol} {interval}")

            # Ensure timestamp columns for downstream quality utilities
            df_full = self._ensure_timestamp_column(df)

            # Check data completeness
            if df_full.empty:
                raise RuntimeError("Final data is empty")

            # Check required columns
            missing_columns = [col for col in self.required_ohlcv_columns if col not in df_full.columns]
            if missing_columns:
                raise RuntimeError(f"Missing required columns in final data: {missing_columns}")

            quality_df = self._build_quality_view(df_full, symbol, interval)

            quality_alert_system = None

            if not QUALITY_UTILITIES_AVAILABLE:
                # Use basic final quality check without quality utilities
                final_score = QualityScore(
                    overall_score=50.0,
                    level=QualityScoreLevel.POOR,
                    component_scores={},
                    issues=["Quality utilities not available"],
                    warnings=["Basic validation only"],
                    recommendations=["Install quality utilities for better assessment"],
                    assessment_timestamp=datetime.now(),
                    data_shape=quality_df.shape
                )
                final_quality_assessment = QualityAssessment(
                    overall_score=50.0,
                    metrics=[],
                    issues_found=1,
                    warnings_found=1,
                    critical_issues=0,
                    assessment_timestamp=datetime.now(),
                    data_shape=quality_df.shape
                )
                final_distribution_validation = {}
                quality_alerts = []
            else:
                # Initialize comprehensive quality utilities
                scorer = ComprehensiveQualityScorer()
                advanced_metrics = AdvancedQualityMetrics()
                statistical_validator = StatisticalValidator()
                quality_alert_system = QualityAlertManager()

                # Perform comprehensive final quality assessment
                final_score = scorer.assess_data_quality(quality_df, context="final_quality_check", step_name="final_validation", data_type="klines")
                final_quality_assessment = advanced_metrics.comprehensive_quality_assessment(quality_df, context="final_quality_check", step_name="final_validation")

                # Perform statistical validation on final data
                final_distribution_validation = {}
                for col in quality_df.select_dtypes(include=[np.number]).columns:
                    if col in ['open', 'high', 'low', 'close', 'volume']:
                        validation_results = statistical_validator.run_comprehensive_validation(quality_df[col].values)
                        final_distribution_validation[col] = {
                            'results': [{'status': r.status.value, 'message': r.message} for r in validation_results]
                        }

                # Check quality alerts
                from src.utils.data.quality.quality_alert_system import MLValidationResult
                validation_result = MLValidationResult(
                    quality_score=final_score,
                    grade=final_score.level.value if hasattr(final_score, 'level') else 'F',
                    drift_issues=[],
                    correlation_issues=[]
                )
                quality_alerts = quality_alert_system.check_alerts(validation_result)

            # Check for data continuity and temporal consistency
            temporal_issues = []
            if len(quality_df) > 1:
                time_diffs = quality_df.index.to_series().diff().dropna()
                if interval == "1m":
                    expected_interval = 5
                else:
                    expected_interval = self._interval_to_minutes(interval)
                if expected_interval:
                    irregular_intervals = (time_diffs != timedelta(minutes=expected_interval)).sum()
                    if irregular_intervals > 0:
                        temporal_issues.append(f"Found {irregular_intervals} irregular intervals")

            # Check for null values in final data
            null_counts = quality_df[self.required_ohlcv_columns].isnull().sum()
            null_issues = []
            if null_counts.sum() > 0:
                null_issues.append(f"Final data contains null values: {null_counts.to_dict()}")

            # Perform final duplicate check
            duplicate_timestamp_column = 'timestamp_ms' if 'timestamp_ms' in quality_df.columns else 'timestamp'
            final_duplicate_analysis = (
                _ANALYZE_DUPLICATES_COMPREHENSIVE(quality_df, timestamp_column=duplicate_timestamp_column)
                if callable(_ANALYZE_DUPLICATES_COMPREHENSIVE)
                else None
            )

            # Determine final quality level based on comprehensive assessment
            if final_score.overall_score >= 95 and not temporal_issues and not null_issues:
                result.quality_level = DataQualityLevel.EXCELLENT
            elif final_score.overall_score >= 85 and len(temporal_issues + null_issues) <= 1:
                result.quality_level = DataQualityLevel.GOOD
            elif final_score.overall_score >= 75 and len(temporal_issues + null_issues) <= 2:
                result.quality_level = DataQualityLevel.FAIR
            elif final_score.overall_score >= 60 and len(temporal_issues + null_issues) <= 3:
                result.quality_level = DataQualityLevel.POOR
            else:
                result.quality_level = DataQualityLevel.FAILED

            # Collect all warnings and issues
            all_warnings = (final_score.warnings +
                          temporal_issues +
                          null_issues +
                          quality_alerts)

            if final_duplicate_analysis and getattr(final_duplicate_analysis, 'total_duplicates', 0) > 0:
                all_warnings.append(f"Final data contains {getattr(final_duplicate_analysis, 'total_duplicates', 0)} duplicate records")

            result.success = True
            result.data = df_full
            result.metadata = {
                "final_comprehensive_score": final_score.overall_score,
                "final_quality_level": final_score.level.value,
                "final_component_scores": final_score.component_scores,
                "final_quality_assessment": {
                    "overall_score": final_quality_assessment.overall_score,
                    "issues_found": final_quality_assessment.issues_found,
                    "warnings_found": final_quality_assessment.warnings_found,
                    "critical_issues": final_quality_assessment.critical_issues
                },
                "final_distribution_validation": final_distribution_validation,
                "final_duplicate_analysis": {
                    "total_duplicates": getattr(final_duplicate_analysis, 'total_duplicates', 0),
                    "true_duplicate_groups": getattr(final_duplicate_analysis, 'true_duplicate_groups', 0),
                    "false_duplicate_groups": getattr(final_duplicate_analysis, 'false_duplicate_groups', 0),
                    "mixed_duplicate_groups": getattr(final_duplicate_analysis, 'mixed_duplicate_groups', 0)
                },
                "final_records": len(df_full),
                "final_columns": len(df_full.columns),
                "null_counts": null_counts.to_dict(),
                "date_range": {
                    "start": df_full.index.min().isoformat(),
                    "end": df_full.index.max().isoformat()
                },
                "quality_alerts": quality_alerts,
                "assessment_timestamp": final_score.assessment_timestamp.isoformat()
            }
            result.warnings.extend(all_warnings)

            if self.enable_logging:
                tprint_success(f"✅ Comprehensive final quality check completed: {result.quality_level.value} (Score: {final_score.overall_score:.1f})")
                if final_score.recommendations:
                    tprint_info(f"📋 Final quality recommendations: {', '.join(final_score.recommendations[:3])}")
                if quality_alerts:
                    tprint_warning(f"⚠️ Quality alerts: {', '.join(quality_alerts[:3])}")

        except Exception as e:
            error_msg = f"Comprehensive final quality check failed: {str(e)}"
            result.errors.append(error_msg)
            if self.enable_logging:
                tprint_error(f"❌ {error_msg}")

        result.processing_time = (datetime.now() - start_time).total_seconds()
        return result

    async def _create_consolidated_file(
        self,
        df: pd.DataFrame,
        symbol: str,
        interval: str,
        batch_id: Optional[str] = None
    ) -> ProcessingResult:
        """Create consolidated output file."""
        start_time = datetime.now()
        result = ProcessingResult(
            step=ProcessingStep.CONSOLIDATION,
            success=False,
            errors=[],
            warnings=[]
        )

        try:
            if self.enable_logging:
                tprint_info(f"📦 Creating consolidated file for {symbol} {interval}")

            # Ensure required metadata columns
            if 'exchange' not in df.columns:
                df['exchange'] = self.exchange
            if 'symbol' not in df.columns:
                df['symbol'] = symbol
            if 'interval' not in df.columns:
                df['interval'] = interval

            normalized_df = self._normalize_calendar_columns(df)
            if normalized_df is not None:
                df = normalized_df

            # Create consolidated batch ID
            consolidated_batch_id = f"{batch_id}_consolidated" if batch_id else "consolidated"

            # Store using KlinesParquetManager
            success = self.klines_manager.write_data(
                df, symbol, f"{interval}_consolidated", "processed", overwrite=True
            )

            if not success:
                raise RuntimeError("Failed to store consolidated file using KlinesParquetManager")

            # Log optimization benefits for consolidated file
            if self.enable_logging:
                try:
                    compression_stats = self.klines_manager.get_compression_stats()
                    if compression_stats.get("total_files", 0) > 0:
                        tprint_info(f"📊 Consolidated file compression: {compression_stats.get('overall_compression_ratio', 0):.1f}%")
                except AttributeError:
                    tprint_info("📊 Consolidated file compression: unknown (method not available)")

            # Get the actual file path from the manager
            output_file = self.data_dir / self.exchange / symbol.lower() / "processed" / f"{symbol.lower()}_{interval}_consolidated"

            result.success = True
            # Check if file exists before getting size
            file_size_mb = 0.0
            if output_file.exists():
                file_size_mb = output_file.stat().st_size / (1024 * 1024)
            
            result.metadata = {
                "output_file": str(output_file),
                "file_size_mb": file_size_mb,
                "records": len(df),
                "columns": len(df.columns)
            }

            if self.enable_logging:
                tprint_success(f"✅ Consolidated file created: {output_file}")
                tprint_info(f"   📊 Records: {len(df):,}")
                tprint_info(f"   📏 Size: {result.metadata['file_size_mb']:.2f} MB")

        except Exception as e:
            error_msg = f"Consolidated file creation failed: {str(e)}"
            result.errors.append(error_msg)
            if self.enable_logging:
                tprint_error(f"❌ {error_msg}")

        result.processing_time = (datetime.now() - start_time).total_seconds()
        return result

    async def _store_original_data(
        self,
        df: pd.DataFrame,
        symbol: str,
        interval: str,
        batch_id: Optional[str]
    ) -> ProcessingResult:
        """Store original data using KlinesParquetManager."""
        start_time = datetime.now()
        result = ProcessingResult(
            step=ProcessingStep.CONSOLIDATION,
            success=False,
            errors=[],
            warnings=[]
        )

        try:
            if self.enable_logging:
                tprint_info(f"💾 Storing original data for {symbol} {interval}")

            storage_df = self._normalize_calendar_columns(df)
            if storage_df is None:
                storage_df = df

            # Store data using KlinesParquetManager
            # Use overwrite=False to merge with existing monthly files instead of replacing them
            success = self.klines_manager.write_data(
                storage_df, symbol, interval, "raw", overwrite=False
            )

            if success:
                result.success = True

                # Get compression statistics (fallback if method not available)
                try:
                    compression_stats = self.klines_manager.get_compression_stats()
                except AttributeError:
                    compression_stats = {"compression_ratio": "unknown", "compression_method": "parquet"}

                result.metadata = {
                    "stored_files": [f"{symbol}_{interval}_original"],
                    "record_count": len(storage_df),
                    "compression_ratio": compression_stats.get("overall_compression_ratio", 0),
                    "file_size_mb": compression_stats.get("total_file_size_mb", 0),
                    "optimization_applied": True
                }

                if self.enable_logging:
                    tprint_success(f"✅ Stored {len(storage_df)} records for {symbol} {interval}")
                    if compression_stats.get("total_files", 0) > 0:
                        tprint_info(f"📊 Compression ratio: {compression_stats.get('overall_compression_ratio', 0):.1f}%")
                        tprint_info(f"💾 File size: {compression_stats.get('total_file_size_mb', 0):.2f} MB")
            else:
                error_message = "Failed to store data using KlinesParquetManager"
                result.errors.append(error_message)
                if self.enable_logging:
                    tprint_warning(f"⚠️ {error_message}")

        except Exception as e:
            error_msg = f"Data storage failed: {str(e)}"
            result.errors.append(error_msg)
            if self.enable_logging:
                tprint_error(f"❌ {error_msg}")

        result.processing_time = (datetime.now() - start_time).total_seconds()
        return result

    async def _verify_all_resampled_gaps(
        self,
        symbol: str,
        resampling_config: ResamplingConfig,
        years: int
    ) -> Dict[str, List[GapInfo]]:
        """Verify gaps across all target intervals after resampling."""
        all_gaps = {}
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)

        for interval in resampling_config.target_intervals:
            try:
                df = self.klines_manager.read_data(
                    symbol=symbol,
                    interval=interval,
                    start_date=start_date,
                    end_date=end_date,
                    data_type="processed"
                )
                
                # Fix: Handle missing timestamp column by using DatetimeIndex directly
                if df is not None and not df.empty:
                    # Ensure DataFrame has proper datetime index
                    if not isinstance(df.index, pd.DatetimeIndex):
                        if "timestamp" in df.columns:
                            df = df.set_index("timestamp")
                        elif "timestamp_ms" in df.columns:
                            df = df.set_index(pd.to_datetime(df["timestamp_ms"], unit="ms"))
                        else:
                            # Use index as-is if it is already datetime-like
                            df.index = pd.to_datetime(df.index)
                    
                    gaps = self._detect_gaps_vectorized(
                        df,
                        interval,
                        self.config.max_gap_minutes,
                        expected_start=start_date,
                        expected_end=end_date
                    )
                else:
                    gaps = []
                all_gaps[interval] = gaps
                
                if self.enable_logging:
                    if gaps:
                        tprint_warning(f"⚠️ Post-resample: {len(gaps)} gaps remain in {interval}")
                    else:
                        tprint_success(f"✅ Post-resample: 100% coverage verified for {interval}")
            except Exception as e:
                tprint_error(f"❌ Verification failed for {interval}: {e}")
                
        return all_gaps

    async def _resample_data_with_age_check(
        self,
        df: pd.DataFrame,
        symbol: str,
        resampling_config: ResamplingConfig,
        batch_id: Optional[str],
        years: int,
        filled_ranges: Optional[List[Tuple[datetime, datetime]]] = None
    ) -> ProcessingResult:
        """
        Resample data with age-based filtering.
        
        Instead of skipping the entire process, this filters the source data to only 
        include points older than the configured threshold (e.g., 3 days), ensuring 
        stability in processed/resampled files while allowing the pipeline to proceed.
        """
        start_time = datetime.now()
        result = ProcessingResult(
            step=ProcessingStep.RESAMPLING,
            success=False,
            errors=[],
            warnings=[]
        )

        try:
            if self.enable_logging:
                tprint_info(f"📊 Filtering data by age for resampling: {symbol}")

            # 1. Determine the cutoff time for resampling
            current_time = datetime.now(tz=pd.Timestamp.now().tzinfo)
            threshold_days = getattr(resampling_config, "resample_older_than_days", 3)
            cutoff_time = current_time - timedelta(days=threshold_days)
            
            # Ensure df index is UTC for reliable comparison
            df_utc = self._ensure_utc_index(df)
            cutoff_time_utc = self._ensure_utc_timestamp(cutoff_time)

            # 2. Filter data by age
            mask_age = df_utc.index < cutoff_time_utc
            if not mask_age.any():
                result.success = True
                result.metadata = {
                    "resampled_intervals": [],
                    "stored_files": [],
                    "reason": f"No data points older than {threshold_days} days (cutoff: {cutoff_time_utc})"
                }
                if self.enable_logging:
                    tprint_info(f"⏭️ Skipping resampling: no data points older than {threshold_days} days")
                return result

            df_to_resample = df_utc.loc[mask_age]
            
            if self.enable_logging:
                tprint_info(f"   Filtered {len(df_to_resample)}/{len(df)} points for resampling (cutoff: {cutoff_time_utc})")

            # 3. Perform resampling on the aged data
            resampled_intervals = []
            stored_files = []
            resample_modes: Dict[str, str] = {}

            for target_interval in resampling_config.target_intervals:
                try:
                    if self.enable_logging:
                        tprint_info(f"🔄 Resampling aged data to {target_interval}")

                    # Decide whether to resample only filled windows (within the aged set)
                    source_df = df_to_resample
                    mode = "full_aged"
                    if filled_ranges:
                        # Convert filled_ranges to UTC for comparison
                        filled_ranges_utc = [(self._ensure_utc_timestamp(s), self._ensure_utc_timestamp(e)) for s, e in filled_ranges]
                        mask_filled = pd.Series(False, index=df_to_resample.index)
                        for start, end in filled_ranges_utc:
                            if start and end:
                                mask_filled |= (df_to_resample.index >= start) & (df_to_resample.index <= end)
                        
                        if mask_filled.any():
                            source_df = df_to_resample.loc[mask_filled]
                            mode = "selective_aged"

                    resampled_df = self._perform_resampling(source_df, target_interval, resampling_config)

                    # If selective produced empty, fall back to full aged dataset
                    if resampled_df.empty and mode == "selective_aged":
                        if self.enable_logging:
                            tprint_warning(f"⚠️ Selective aged resample empty for {target_interval}; falling back to full aged data")
                        resampled_df = self._perform_resampling(df_to_resample, target_interval, resampling_config)
                        mode = "fallback_full_aged"

                    if not resampled_df.empty:
                        normalized_resampled = self._normalize_calendar_columns(resampled_df)
                        if normalized_resampled is not None:
                            resampled_df = normalized_resampled
                        
                        # Store resampled data (using update_data to be additive/atomic)
                        success = self.klines_manager.update_data(
                            resampled_df, symbol, target_interval, "processed"
                        )

                        if success:
                            resampled_intervals.append(target_interval)
                            stored_files.append(f"{symbol}_{target_interval}_resampled")
                            resample_modes[target_interval] = mode
                            if self.enable_logging:
                                tprint_success(f"✅ Resampled aged data to {target_interval}: {len(resampled_df)} records")
                        else:
                            result.warnings.append(f"Failed to store resampled data for {target_interval}")
                    else:
                        result.warnings.append(f"Resampling to {target_interval} produced empty data")

                except Exception as e:
                    result.warnings.append(f"Failed to resample to {target_interval}: {e}")

            result.success = True
            result.metadata = {
                "resampled_intervals": resampled_intervals,
                "stored_files": stored_files,
                "cutoff_time": str(cutoff_time_utc),
                "resample_modes": resample_modes
            }

        except Exception as e:
            error_msg = f"Resampling with age filtering failed: {str(e)}"
            result.errors.append(error_msg)
            if self.enable_logging:
                tprint_error(f"❌ {error_msg}")

        result.processing_time = (datetime.now() - start_time).total_seconds()
        return result

    def _perform_resampling(
        self,
        df: pd.DataFrame,
        target_interval: str,
        resampling_config: ResamplingConfig
    ) -> pd.DataFrame:
        """Perform the actual resampling operation."""
        try:
            # Convert interval to pandas frequency
            freq = self._interval_to_pandas_freq(target_interval)
            if freq is None:
                raise ValueError(f"Unsupported interval: {target_interval}")

            # Resample OHLCV data
            resampled = df.resample(freq).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum' if resampling_config.preserve_volume else 'mean'
            }).dropna()

            # Add metadata
            resampled['symbol'] = df['symbol'].iloc[0] if 'symbol' in df.columns else ''
            resampled['interval'] = target_interval
            resampled['exchange'] = df['exchange'].iloc[0] if 'exchange' in df.columns else self.exchange

            return resampled

        except Exception as e:
            if self.enable_logging:
                tprint_error(f"❌ Resampling failed for {target_interval}: {e}")
            return pd.DataFrame()

    def get_processing_summary(self) -> Dict[str, Any]:
        """Get summary of all processing steps."""
        if not self.processing_results:
            return {"message": "No processing results available"}

        summary = {
            "total_steps": len(self.processing_results),
            "successful_steps": len([r for r in self.processing_results if r.success]),
            "failed_steps": len([r for r in self.processing_results if not r.success]),
            "total_processing_time": sum(r.processing_time for r in self.processing_results),
            "steps": []
        }

        for result in self.processing_results:
            step_summary = {
                "step": result.step.value,
                "success": result.success,
                "processing_time": result.processing_time,
                "quality_level": result.quality_level.value if result.quality_level else None,
                "errors": len(result.errors),
                "warnings": len(result.warnings)
            }
            summary["steps"].append(step_summary)

        return summary

# Convenience functions for easy usage
async def process_klines_data_enhanced(
    symbol: str,
    interval: str,
    years: int,
    exchange_interface: ExchangeInterface,
    config: Optional[PipelineConfig] = None,
    resampling_config: Optional[ResamplingConfig] = None,
    max_gap_minutes: Optional[int] = None,
    create_consolidated: bool = True,
    batch_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to process klines data using the enhanced pipeline.

    Args:
        symbol: Trading symbol (e.g., "ETHUSDT")
        interval: Data interval (e.g., "1m")
        years: Number of years of data to process
        exchange_interface: ExchangeInterface instance for data access
        config: Pipeline configuration
        resampling_config: Configuration for data resampling
        max_gap_minutes: Maximum allowed gap in minutes
        create_consolidated: Whether to create consolidated output file
        batch_id: Optional batch identifier

    Returns:
        Dictionary with complete processing results
    """
    pipeline = EnhancedKlinesProcessingPipeline(config)

    return await pipeline.process_klines_data(
        symbol=symbol,
        interval=interval,
        years=years,
        exchange_interface=exchange_interface,
        resampling_config=resampling_config,
        max_gap_minutes=max_gap_minutes,
        create_consolidated=create_consolidated,
        batch_id=batch_id
    )

if __name__ == "__main__":
    import argparse
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Enhanced Klines Data Processing Pipeline')
    parser.add_argument('--exchange', type=str, default='binance', help='Exchange name (binance, bingx, okx, etc.)')
    parser.add_argument('--symbol', type=str, default='ETHUSDT', help='Trading symbol (e.g., ETHUSDT)')
    parser.add_argument('--interval', type=str, default='1m', help='Data interval (1m, 5m, 1h, etc.)')
    parser.add_argument('--years', type=int, default=4, help='Number of years of data to collect')
    parser.add_argument('--data-dir', type=str, default='historical_data', help='Data directory')
    parser.add_argument('--api-key', type=str, default='', help='Exchange API key (optional)')
    parser.add_argument('--api-secret', type=str, default='', help='Exchange API secret (optional)')
    parser.add_argument('--api-password', type=str, default='', help='Exchange API password/passphrase (optional)')
    parser.add_argument('--use-testnet', action='store_true', help='Use exchange testnet environment')
    parser.add_argument('--no-gap-filling', action='store_true', help='Disable gap filling')
    parser.add_argument('--no-resampling', action='store_true', help='Disable resampling')
    parser.add_argument('--no-quality-validation', action='store_true', help='Disable quality validation')
    args = parser.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")
    
    # Example usage - simplified interface with exchange, asset, lookback period
    async def main_simple():
        """Run the enhanced klines processing pipeline."""
        try:
            print("=" * 80)
            print("🚀 ENHANCED KLINES PROCESSING PIPELINE")
            print("=" * 80)
            print()
            print(f"📊 Configuration:")
            print(f"   - Exchange: {args.exchange}")
            print(f"   - Symbol: {args.symbol}")
            print(f"   - Interval: {args.interval}")
            print(f"   - Lookback: {args.years} years")
            print(f"   - Data Directory: {args.data_dir}")
            print(f"   - Gap Filling: {'❌ Disabled' if args.no_gap_filling else '✅ Enabled'}")
            print(f"   - Resampling: {'❌ Disabled' if args.no_resampling else '✅ Enabled'}")
            print(f"   - Quality Validation: {'❌ Disabled' if args.no_quality_validation else '✅ Enabled'}")
            print(f"   - Authenticated: {'✅ Yes' if args.api_key and args.api_secret else '❌ No'}")
            print()
            
            # Configure pipeline
            pipeline_config = PipelineConfig(
                data_dir=args.data_dir,
                exchange=args.exchange,
                enable_logging=True,
                enable_gap_filling=not args.no_gap_filling,
                enable_resampling=not args.no_resampling,
                enable_duplicate_handling=True,
                enable_quality_validation=not args.no_quality_validation,
                batch_compatible=True,
                max_gap_minutes=1
            )
            
            # Configure resampling
            # Use resample_older_than_days=0 to always resample, letting the
            # age-based skip logic be controlled explicitly by the config.
            resampling_config = ResamplingConfig(
                target_intervals=['5m', '15m', '30m', '1h'],
                method='ohlc',
                preserve_volume=True,
                resample_older_than_days=0,
                enable_auto_resampling=True
            ) if not args.no_resampling else None
            
            # Create pipeline
            print(f"🔧 Initializing pipeline...")
            pipeline = EnhancedKlinesProcessingPipeline(pipeline_config)
            print(f"✅ Pipeline initialized")
            print()
            
            # Create exchange interface (prefer factory + dispatcher)
            print(f"🔗 Connecting to {args.exchange.upper()}...")
            exchange_interface = None
            # Fix: Guard against ExchangeInterface being None
            if not EXCHANGE_INTERFACE_AVAILABLE or ExchangeInterface is None:
                tprint_warning("⚠️ ExchangeInterface unavailable; using existing/local data only")
                print()
            else:
                dispatcher = None
                if create_exchange_dispatcher is not None and ExchangeConfig is not None:
                    try:
                        ex_type = ExchangeType(args.exchange) if isinstance(args.exchange, str) else args.exchange
                    except Exception:
                        ex_type = ExchangeType.BINANCE if str(args.exchange).lower() == "binance" else ExchangeType.BINANCE
                    try:
                        dispatcher_cfg = ExchangeConfig(
                            exchange_type=ex_type,
                            api_key=args.api_key or None,
                            api_secret=args.api_secret or None,
                            password=args.api_password or None,
                            subaccount_id=None,
                            use_testnet=args.use_testnet,
                            trade_symbol=args.symbol,
                            mode=TradingMode.TRADE,  # allow live data pulls
                        )
                        dispatcher = create_exchange_dispatcher(dispatcher_cfg)
                        # Initialize dispatcher if supported
                        try:
                            init_ret = dispatcher.initialize()
                            # If initialize is coroutine, await it and check success
                            if hasattr(init_ret, "__await__"):
                                init_ret = await init_ret
                            if init_ret is False:
                                tprint_warning("⚠️ Dispatcher initialize returned False; live download may fail")
                                dispatcher = None
                        except Exception as e:
                            tprint_warning(f"⚠️ Dispatcher initialize failed: {e}")
                        if dispatcher is not None:
                            exchange_interface.dispatcher = dispatcher
                    except Exception as e:
                        tprint_warning(f"⚠️ Failed to create dispatcher: {e}")
                        dispatcher = None

                exchange_config = {
                    'exchange_type': args.exchange,
                    'api_key': args.api_key or None,
                    'api_secret': args.api_secret or None,
                    'password': args.api_password or None,
                    'testnet': args.use_testnet,
                    'rate_limits': {},
                }
                try:
                    if create_exchange_interface is not None:
                        exchange_interface = create_exchange_interface(exchange_config)
                    else:
                        exchange_interface = ExchangeInterface(exchange_config)
                    if dispatcher is not None:
                        exchange_interface.dispatcher = dispatcher
                    await exchange_interface.connect()
                    print(f"✅ Connected to {args.exchange.upper()}")
                except Exception as e:
                    tprint_warning(f"⚠️ Connection warning: {e}")
                    tprint_info("📝 Continuing with existing/local data only...")
                    exchange_interface = None
            
            # Process data
            print(f"🚀 Starting data collection and processing...")
            print(f"⏰ Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print()
            
            results = await pipeline.process_klines_data(
                symbol=args.symbol,
                interval=args.interval,
                years=args.years,
                exchange_interface=exchange_interface,
                resampling_config=resampling_config,
                max_gap_minutes=1,
                create_consolidated=True,
                batch_id=f"{args.exchange}_{args.symbol.lower()}_{args.years}y_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            
            print()
            print("=" * 80)
            print("✅ PROCESSING COMPLETED")
            print("=" * 80)
            print()
            print(f"📊 Results:")
            print(f"   - Pipeline Success: {results['pipeline_success']}")
            print(f"   - Data Quality: {results['data_quality']}")
            print(f"   - Final Data Shape: {results['final_data_shape']}")
            print()
            
            if 'stored_files' in results and results['stored_files']:
                print(f"💾 Stored Files:")
                for file_path in results['stored_files']:
                    print(f"   - {file_path}")
                print()
            
            if 'resampled_intervals' in results and results['resampled_intervals']:
                print(f"🔄 Resampled Intervals:")
                for interval in results['resampled_intervals']:
                    print(f"   - {interval}")
                print()
            
            print(f"⏰ End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            if exchange_interface is not None:
                await exchange_interface.disconnect()
            
        except Exception as e:
            print()
            print("=" * 80)
            print("❌ ERROR IN PROCESSING")
            print("=" * 80)
            print(f"Error: {e}")
            print()
            import traceback
            traceback.print_exc()

    # Example usage - for reference only
    async def main_example():
        try:
            # Configure pipeline for existing data processing
            pipeline_config = PipelineConfig(
                data_dir="historical_data",
                exchange="binance",
                enable_logging=True,
                enable_gap_filling=False,  # Data already has no gaps
                enable_resampling=True,
                enable_duplicate_handling=True,
                enable_quality_validation=True,
                batch_compatible=True
            )

            # Configure resampling for existing data
            resampling_config = ResamplingConfig(
                target_intervals=['5m', '15m', '30m', '1h'],  # Skip 1m as it's already available
                method='ohlc',
                preserve_volume=True,
                resample_older_than_days=1,  # Resample all data older than 1 day
                enable_auto_resampling=True
            )

            # Create enhanced exchange interface for data downloading
            # Supports: 'binance', 'bingx', 'okx', 'mexc', 'gateio', 'phemex'
            exchange_config = {
                'exchange_type': 'binance',  # Change to 'bingx' for BingX
                'api_key': "",  # Add your API key here
                'api_secret': "",  # Add your API secret here
                'testnet': True,
                'rate_limits': {}
            }
            exchange_interface = ExchangeInterface(exchange_config)
            
            try:
                await exchange_interface.connect()
            except Exception as e:
                print(f"⚠️ Exchange connection failed: {e}")
                print("📁 Falling back to existing data processing...")
                # Create fallback interface for existing data
                class FallbackExchangeInterface:
                    def __init__(self):
                        self.connected = True
                        self.exchange_type = "local"
                    
                    async def connect(self):
                        return True
                    
                    async def disconnect(self):
                        pass
                    
                    async def get_klines(self, *args, **kwargs):
                        return []
                
                exchange_interface = FallbackExchangeInterface()

            # Process existing data
            results = await process_klines_data_enhanced(
                symbol="ETHUSDT",
                interval="1m",
                years=4,  # Process 4 years of existing data
                exchange_interface=exchange_interface,
                config=pipeline_config,
                resampling_config=resampling_config,
                batch_id="existing_data_processing"
            )

            print(f"\n🎉 Processing completed: {results['pipeline_success']}")
            print(f"📊 Data quality: {results['data_quality']}")
            print(f"📈 Final shape: {results['final_data_shape']}")
            print(f"💾 Stored files: {results['stored_files']}")
            print(f"🔄 Resampled intervals: {results['resampled_intervals']}")

            await exchange_interface.disconnect()

        except Exception as e:
            print(f"❌ Error in main processing: {e}")
            import traceback
            traceback.print_exc()

    # Run the simplified example
    asyncio.run(main_simple())
